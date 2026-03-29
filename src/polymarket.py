"""Polymarket BTC market monitor.

Fetches BTC-related prediction market prices from Polymarket's
Gamma API (market discovery) and CLOB API (order book / pricing).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import aiohttp

from .config import Settings


@dataclass
class MarketInfo:
    """A Polymarket BTC prediction market."""
    condition_id: str
    question: str
    token_id_yes: str
    token_id_no: str
    outcome_yes_price: float
    outcome_no_price: float
    volume: float
    end_date: str
    active: bool
    fetched_at: float = field(default_factory=time.time)

    @property
    def implied_btc_price(self) -> float | None:
        """Extract implied BTC price from market question if possible.

        Polymarket BTC markets typically ask "Will BTC be above $X on date Y?"
        The YES price implies the market's probability that BTC > $X.
        """
        return None  # Parsed in the scanner

    @property
    def age_ms(self) -> float:
        return (time.time() - self.fetched_at) * 1000


@dataclass
class BTCBracket:
    """A price bracket extracted from a BTC prediction market.

    Example: "BTC above $95,000?" with YES at $0.72
    means the market thinks there's a 72% chance BTC is above $95k.
    """
    threshold_price: float  # The dollar threshold (e.g., 95000)
    yes_price: float  # Price of YES token (0-1)
    no_price: float  # Price of NO token (0-1)
    market: MarketInfo
    condition_id: str
    token_id_yes: str
    token_id_no: str


class PolymarketMonitor:
    """Discovers and monitors BTC prediction markets on Polymarket."""

    def __init__(self, settings: Settings, session: aiohttp.ClientSession | None = None) -> None:
        self._settings = settings
        self._session = session
        self._owns_session = session is None
        self._markets: dict[str, MarketInfo] = {}
        self._brackets: list[BTCBracket] = []

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    async def discover_btc_markets(self) -> list[MarketInfo]:
        """Find active BTC price prediction markets via Gamma API."""
        session = await self._ensure_session()
        markets: list[MarketInfo] = []

        try:
            async with session.get(
                f"{self._settings.gamma_api_url}/events",
                params={
                    "tag": "crypto",
                    "status": "active",
                    "limit": 50,
                },
            ) as resp:
                if resp.status != 200:
                    print(f"[polymarket] Gamma API returned HTTP {resp.status}")
                    return markets
                events = await resp.json()

            if not isinstance(events, list):
                print(f"[polymarket] Unexpected API response type: {type(events).__name__}")
                return markets

            for event in events:
                title = (event.get("title") or "").lower()
                if "btc" not in title and "bitcoin" not in title:
                    continue

                for market_data in event.get("markets", []):
                    if not market_data.get("active"):
                        continue
                    tokens = market_data.get("clobTokenIds", [])
                    if len(tokens) < 2:
                        continue
                    prices = market_data.get("outcomePrices", [])
                    if len(prices) < 2:
                        continue

                    market = MarketInfo(
                        condition_id=market_data.get("conditionId", ""),
                        question=market_data.get("question", ""),
                        token_id_yes=tokens[0],
                        token_id_no=tokens[1],
                        outcome_yes_price=float(prices[0]),
                        outcome_no_price=float(prices[1]),
                        volume=float(market_data.get("volume", 0)),
                        end_date=market_data.get("endDate", ""),
                        active=True,
                    )
                    markets.append(market)
                    self._markets[market.condition_id] = market

        except Exception as e:
            print(f"[polymarket] Error discovering markets: {e}")

        return markets

    async def fetch_orderbook(self, token_id: str) -> dict:
        """Fetch order book for a specific token from the CLOB."""
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self._settings.clob_api_url}/book",
                params={"token_id": token_id},
            ) as resp:
                return await resp.json()
        except Exception as e:
            print(f"[polymarket] Error fetching orderbook: {e}")
            return {}

    async def fetch_midpoint(self, token_id: str) -> float | None:
        """Get midpoint price for a token."""
        session = await self._ensure_session()
        try:
            async with session.get(
                f"{self._settings.clob_api_url}/midpoint",
                params={"token_id": token_id},
            ) as resp:
                data = await resp.json()
                mid = data.get("mid")
                return float(mid) if mid else None
        except Exception:
            return None

    async def refresh_prices(self) -> list[MarketInfo]:
        """Re-fetch prices for all known markets."""
        session = await self._ensure_session()
        updated: list[MarketInfo] = []

        for cid, market in list(self._markets.items()):
            try:
                async with session.get(
                    f"{self._settings.gamma_api_url}/markets/{cid}",
                ) as resp:
                    data = await resp.json()

                prices = data.get("outcomePrices", [])
                if len(prices) >= 2:
                    market = MarketInfo(
                        condition_id=cid,
                        question=market.question,
                        token_id_yes=market.token_id_yes,
                        token_id_no=market.token_id_no,
                        outcome_yes_price=float(prices[0]),
                        outcome_no_price=float(prices[1]),
                        volume=float(data.get("volume", market.volume)),
                        end_date=market.end_date,
                        active=data.get("active", True),
                    )
                    self._markets[cid] = market
                    updated.append(market)
            except Exception:
                continue

        return updated

    def parse_btc_brackets(self) -> list[BTCBracket]:
        """Parse market questions to extract BTC price brackets.

        Looks for patterns like:
        - "Will Bitcoin be above $95,000 on March 31?"
        - "BTC above $100k?"
        """
        import re

        brackets: list[BTCBracket] = []
        # Capture optional k/K suffix as a separate group
        price_pattern = re.compile(
            r"\$\s*([\d,]+(?:\.\d+)?)\s*([kK])?", re.IGNORECASE
        )

        for market in self._markets.values():
            if not market.active:
                continue
            question = market.question
            match = price_pattern.search(question)
            if not match:
                continue

            raw = match.group(1).replace(",", "")
            threshold = float(raw)
            # Handle "100k" -> 100000
            if match.group(2):
                threshold *= 1000

            brackets.append(
                BTCBracket(
                    threshold_price=threshold,
                    yes_price=market.outcome_yes_price,
                    no_price=market.outcome_no_price,
                    market=market,
                    condition_id=market.condition_id,
                    token_id_yes=market.token_id_yes,
                    token_id_no=market.token_id_no,
                )
            )

        brackets.sort(key=lambda b: b.threshold_price)
        self._brackets = brackets
        return brackets

    @property
    def brackets(self) -> list[BTCBracket]:
        return list(self._brackets)

    @property
    def markets(self) -> dict[str, MarketInfo]:
        return dict(self._markets)
