"""Trade executor — places orders on Polymarket's CLOB.

Handles order creation, submission, and tracking via the
Polymarket CLOB API (py-clob-client).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import aiohttp

from .config import Settings
from .detector import Opportunity, Signal
from .risk import RiskManager, Trade


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class OrderResult:
    order_id: str
    status: OrderStatus
    filled_size: float
    filled_price: float
    timestamp: float = field(default_factory=time.time)
    error: str | None = None


class TradeExecutor:
    """Executes trades on Polymarket CLOB API."""

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        session: aiohttp.ClientSession | None = None,
        dry_run: bool = True,
    ) -> None:
        self._settings = settings
        self._risk = risk_manager
        self._session = session
        self._owns_session = session is None
        self._dry_run = dry_run
        self._order_history: list[OrderResult] = []

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.trading.order_timeout_seconds)
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    def _build_order_payload(self, opportunity: Opportunity, size: float) -> dict:
        """Build the CLOB API order payload."""
        if opportunity.signal == Signal.BUY_YES:
            token_id = opportunity.bracket.token_id_yes
            side = "BUY"
            price = min(opportunity.market_price + self._settings.trading.max_slippage, 0.99)
        elif opportunity.signal == Signal.BUY_NO:
            token_id = opportunity.bracket.token_id_no
            side = "BUY"
            price = min(opportunity.bracket.no_price + self._settings.trading.max_slippage, 0.99)
        else:
            return {}

        return {
            "tokenID": token_id,
            "price": round(price, 4),
            "size": round(size, 2),
            "side": side,
            "type": "GTC",  # Good-till-cancelled
        }

    async def execute(self, opportunity: Opportunity) -> OrderResult | None:
        """Execute a trade for the given opportunity.

        Checks risk limits, computes position size, and places the order.
        Returns None if trade is rejected by risk manager.
        """
        # Risk check
        allowed, reason = self._risk.can_trade(opportunity)
        if not allowed:
            print(f"[executor] Trade rejected: {reason}")
            return None

        # Position sizing
        size = self._risk.compute_position_size(opportunity)
        if size < 1.0:  # Minimum $1 trade
            print(f"[executor] Position too small: ${size:.2f}")
            return None

        payload = self._build_order_payload(opportunity, size)
        if not payload:
            return None

        # Dry run mode — simulate execution
        if self._dry_run:
            return self._simulate_execution(opportunity, payload, size)

        # Live execution via CLOB API
        return await self._submit_order(opportunity, payload, size)

    def _simulate_execution(
        self, opportunity: Opportunity, payload: dict, size: float
    ) -> OrderResult:
        """Simulate order execution for paper trading."""
        result = OrderResult(
            order_id=f"DRY-{int(time.time() * 1000)}",
            status=OrderStatus.FILLED,
            filled_size=size,
            filled_price=payload["price"],
        )
        self._order_history.append(result)

        # Record with risk manager
        trade = Trade(
            opportunity=opportunity,
            side="YES" if opportunity.signal == Signal.BUY_YES else "NO",
            size_usdc=size,
            price=payload["price"],
            status="filled",
        )
        self._risk.record_trade(trade)

        print(
            f"[executor] DRY RUN: {opportunity.signal.value} "
            f"${size:.2f} @ {payload['price']:.4f} | "
            f"BTC=${opportunity.real_btc_price:,.0f} threshold=${opportunity.bracket.threshold_price:,.0f} | "
            f"edge={opportunity.edge:+.3f}"
        )

        return result

    async def _submit_order(
        self, opportunity: Opportunity, payload: dict, size: float
    ) -> OrderResult:
        """Submit order to Polymarket CLOB API.

        Uses the REST API with API key authentication.
        For production use, integrate py-clob-client for proper
        EIP-712 signing.
        """
        session = await self._ensure_session()
        creds = self._settings.credentials

        headers = {
            "POLY_API_KEY": creds.api_key,
            "POLY_API_SECRET": creds.api_secret,
            "POLY_PASSPHRASE": creds.api_passphrase,
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                f"{self._settings.clob_api_url}/order",
                json=payload,
                headers=headers,
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = OrderResult(
                        order_id=data.get("orderID", "unknown"),
                        status=OrderStatus.SUBMITTED,
                        filled_size=0,
                        filled_price=payload["price"],
                    )
                else:
                    error_text = await resp.text()
                    result = OrderResult(
                        order_id="",
                        status=OrderStatus.FAILED,
                        filled_size=0,
                        filled_price=0,
                        error=f"HTTP {resp.status}: {error_text}",
                    )
        except Exception as e:
            result = OrderResult(
                order_id="",
                status=OrderStatus.FAILED,
                filled_size=0,
                filled_price=0,
                error=str(e),
            )

        self._order_history.append(result)

        if result.status != OrderStatus.FAILED:
            trade = Trade(
                opportunity=opportunity,
                side="YES" if opportunity.signal == Signal.BUY_YES else "NO",
                size_usdc=size,
                price=payload["price"],
                status=result.status.value,
            )
            self._risk.record_trade(trade)

        return result

    @property
    def order_history(self) -> list[OrderResult]:
        return list(self._order_history)

    @property
    def is_dry_run(self) -> bool:
        return self._dry_run
