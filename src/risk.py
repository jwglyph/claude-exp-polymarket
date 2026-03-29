"""Risk manager — enforces per-trade and daily risk limits.

Rules from the strategy:
- Max 0.5% risk per trade
- Max 2% daily drawdown cap
- Position sizing based on edge and confidence
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import TradingConfig
from .detector import Opportunity, Signal


@dataclass
class Trade:
    opportunity: Opportunity
    side: str  # "YES" or "NO"
    size_usdc: float
    price: float
    timestamp: float = field(default_factory=time.time)
    pnl: float = 0.0
    status: str = "pending"  # pending, filled, cancelled, failed


@dataclass
class RiskState:
    """Tracks daily P&L and exposure."""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_volume: float = 0.0
    open_exposure: float = 0.0
    day_start: float = field(default_factory=time.time)
    trades: list[Trade] = field(default_factory=list)

    def reset_if_new_day(self) -> None:
        now = time.time()
        # Reset at midnight UTC (86400 seconds per day)
        if (now // 86400) > (self.day_start // 86400):
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.daily_volume = 0.0
            self.open_exposure = 0.0
            self.day_start = now
            self.trades = []


class RiskManager:
    """Enforces trading risk limits."""

    def __init__(self, config: TradingConfig, portfolio_value: float = 1000.0) -> None:
        self.config = config
        self.portfolio_value = portfolio_value
        self.state = RiskState()

    def can_trade(self, opportunity: Opportunity) -> tuple[bool, str]:
        """Check if a trade is allowed under risk limits.

        Returns (allowed, reason).
        """
        self.state.reset_if_new_day()

        # Check daily cap
        max_daily_loss = self.portfolio_value * self.config.daily_risk_cap
        if abs(self.state.daily_pnl) >= max_daily_loss and self.state.daily_pnl < 0:
            return False, f"Daily loss cap hit: ${self.state.daily_pnl:.2f}"

        # Check per-trade risk
        trade_size = self.compute_position_size(opportunity)
        max_trade_risk = self.portfolio_value * self.config.max_risk_per_trade
        if trade_size > max_trade_risk:
            return False, f"Trade size ${trade_size:.2f} exceeds max risk ${max_trade_risk:.2f}"

        # Check total exposure
        max_exposure = self.portfolio_value * 0.20  # Max 20% of portfolio exposed
        if self.state.open_exposure + trade_size > max_exposure:
            return False, f"Would exceed exposure limit: ${self.state.open_exposure + trade_size:.2f}"

        # Minimum edge check
        if opportunity.divergence_pct < self.config.divergence_threshold:
            return False, f"Edge too small: {opportunity.divergence_pct:.3%}"

        return True, "OK"

    def compute_position_size(self, opportunity: Opportunity) -> float:
        """Kelly-inspired position sizing based on edge and confidence.

        Uses a fraction of Kelly criterion for safety.
        """
        edge = abs(opportunity.edge)
        confidence = opportunity.confidence

        # Kelly fraction: edge / odds, but we use half-Kelly for safety
        if opportunity.signal == Signal.BUY_YES:
            odds = 1.0 / opportunity.market_price - 1.0 if opportunity.market_price > 0 else 0
        else:
            odds = 1.0 / (1.0 - opportunity.market_price) - 1.0 if opportunity.market_price < 1 else 0

        if odds <= 0:
            return 0.0

        kelly = edge / (odds if odds > 0 else 1.0)
        half_kelly = kelly * 0.5

        # Cap at configured order size
        size = min(
            half_kelly * self.portfolio_value * confidence,
            self.config.order_size_usdc,
            self.portfolio_value * self.config.max_risk_per_trade,
        )

        return max(0.0, size)

    def record_trade(self, trade: Trade) -> None:
        """Record a trade execution."""
        self.state.trades.append(trade)
        self.state.daily_trades += 1
        self.state.daily_volume += trade.size_usdc
        self.state.open_exposure += trade.size_usdc

    def record_pnl(self, pnl: float) -> None:
        """Record realized P&L."""
        self.state.daily_pnl += pnl

    def close_exposure(self, amount: float) -> None:
        """Reduce open exposure when a position is closed."""
        self.state.open_exposure = max(0, self.state.open_exposure - amount)

    @property
    def stats(self) -> dict:
        self.state.reset_if_new_day()
        return {
            "daily_pnl": self.state.daily_pnl,
            "daily_trades": self.state.daily_trades,
            "daily_volume": self.state.daily_volume,
            "open_exposure": self.state.open_exposure,
            "portfolio_value": self.portfolio_value,
            "risk_utilization": (
                abs(self.state.daily_pnl) / (self.portfolio_value * self.config.daily_risk_cap)
                if self.state.daily_pnl < 0 and self.portfolio_value > 0 and self.config.daily_risk_cap > 0
                else 0.0
            ),
        }
