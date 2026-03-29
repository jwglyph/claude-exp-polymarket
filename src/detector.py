"""Divergence detector — the core strategy engine.

Compares real BTC price (from external feeds) against Polymarket's
implied BTC probabilities to find exploitable price lag.

Strategy:
- Polymarket has BTC bracket markets: "Will BTC be above $X?"
- Each bracket has a YES price = implied probability BTC > $X
- We compute what the "fair" YES price should be given the real BTC price
- When Polymarket's price diverges by > threshold, we trade

Example:
  Real BTC = $96,500
  Market: "BTC above $95,000?" YES = $0.65 (Polymarket thinks 65%)
  But real price is already above $95k, so fair value is ~$0.85+
  Divergence = 0.20 → BUY YES
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from math import exp, log

from .polymarket import BTCBracket


class Signal(Enum):
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    HOLD = "HOLD"


@dataclass
class Opportunity:
    bracket: BTCBracket
    signal: Signal
    real_btc_price: float
    fair_value: float  # What the YES token should be worth
    market_price: float  # What Polymarket is showing
    edge: float  # fair_value - market_price (signed)
    divergence_pct: float  # |edge| as percentage
    timestamp: float
    confidence: float  # 0-1, how confident we are in the signal

    @property
    def is_actionable(self) -> bool:
        return self.signal != Signal.HOLD

    def __repr__(self) -> str:
        return (
            f"Opportunity({self.signal.value} | "
            f"BTC=${self.real_btc_price:,.0f} vs threshold=${self.bracket.threshold_price:,.0f} | "
            f"fair={self.fair_value:.3f} market={self.market_price:.3f} | "
            f"edge={self.edge:+.3f} ({self.divergence_pct:.1%}))"
        )


class DivergenceDetector:
    """Detects price divergences between real BTC and Polymarket brackets."""

    def __init__(self, divergence_threshold: float = 0.003) -> None:
        self.divergence_threshold = divergence_threshold
        self._history: list[Opportunity] = []

    def estimate_fair_value(
        self,
        real_price: float,
        threshold_price: float,
        volatility: float = 0.02,
        time_to_expiry_hours: float = 24.0,
    ) -> float:
        """Estimate fair YES probability using a simplified log-normal model.

        Given real BTC price and a threshold, compute P(BTC > threshold)
        using a simple model that accounts for current distance from threshold
        and time-based uncertainty.

        Args:
            real_price: Current BTC price from aggregated feeds
            threshold_price: The market's price threshold (e.g., $95,000)
            volatility: Assumed hourly volatility (default 2%)
            time_to_expiry_hours: Hours until market resolves

        Returns:
            Fair probability (0-1) that BTC will be above threshold
        """
        if threshold_price <= 0:
            return 1.0

        # Distance from threshold as a ratio
        ratio = real_price / threshold_price

        # Standard deviation over the time period
        sigma = volatility * (time_to_expiry_hours ** 0.5)

        if sigma < 0.001:
            # Very short time / low vol: binary outcome
            return 1.0 if ratio > 1.0 else 0.0

        # Log-normal CDF approximation: P(BTC > threshold)
        # Using a logistic approximation of the normal CDF
        d = log(ratio) / sigma
        # Logistic approximation: Phi(x) ≈ 1 / (1 + exp(-1.7 * x))
        prob = 1.0 / (1.0 + exp(-1.7 * d))

        return max(0.01, min(0.99, prob))

    def scan_brackets(
        self,
        brackets: list[BTCBracket],
        real_btc_price: float,
        time_to_expiry_hours: float = 24.0,
    ) -> list[Opportunity]:
        """Scan all brackets for divergence opportunities.

        Returns list of actionable opportunities sorted by edge size.
        """
        opportunities: list[Opportunity] = []

        for bracket in brackets:
            fair = self.estimate_fair_value(
                real_btc_price,
                bracket.threshold_price,
                time_to_expiry_hours=time_to_expiry_hours,
            )

            market_yes = bracket.yes_price
            edge = fair - market_yes
            abs_edge = abs(edge)
            divergence_pct = abs_edge

            # Determine signal
            if edge > self.divergence_threshold:
                signal = Signal.BUY_YES
                confidence = min(1.0, abs_edge / 0.10)  # Full confidence at 10% edge
            elif edge < -self.divergence_threshold:
                signal = Signal.BUY_NO
                confidence = min(1.0, abs_edge / 0.10)
            else:
                signal = Signal.HOLD
                confidence = 0.0

            opp = Opportunity(
                bracket=bracket,
                signal=signal,
                real_btc_price=real_btc_price,
                fair_value=fair,
                market_price=market_yes,
                edge=edge,
                divergence_pct=divergence_pct,
                timestamp=time.time(),
                confidence=confidence,
            )
            opportunities.append(opp)

        # Sort by edge magnitude, biggest first
        opportunities.sort(key=lambda o: abs(o.edge), reverse=True)
        self._history.extend(o for o in opportunities if o.is_actionable)

        return opportunities

    def get_actionable(self, opportunities: list[Opportunity]) -> list[Opportunity]:
        """Filter to only actionable opportunities."""
        return [o for o in opportunities if o.is_actionable]

    @property
    def recent_signals(self) -> list[Opportunity]:
        """Return signals from the last 60 seconds."""
        cutoff = time.time() - 60
        return [o for o in self._history if o.timestamp > cutoff]
