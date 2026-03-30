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
        """Find active BTC price prediction markets.

        Searches both the CLOB sampling-markets endpoint (which has
        current BTC price-bracket markets) and the Gamma events API.
        """
        session = await self._ensure_session()
        markets: list[MarketInfo] = []

        # ── Source 1: CLOB sampling-markets (more reliable for price brackets) ──
        try:
            async with session.get(
                f"{self._settings.clob_api_url}/sampling-markets",
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for m in data.get("data", []):
                        question = (m.get("question") or "").lower()
                        if "bitcoin" not in question and "btc" not in question:
                            continue
                        if not m.get("active") or m.get("closed"):
                            continue
                        tokens = m.get("tokens", [])
                        if len(tokens) < 2:
                            continue

                        yes_token = next((t for t in tokens if t.get("outcome") == "Yes"), tokens[0])
                        no_token = next((t for t in tokens if t.get("outcome") == "No"), tokens[1])

                        market = MarketInfo(
                            condition_id=m.get("condition_id", ""),
                            question=m.get("question", ""),
                            token_id_yes=yes_token.get("token_id", ""),
                            token_id_no=no_token.get("token_id", ""),
                            outcome_yes_price=float(yes_token.get("price", 0)),
                            outcome_no_price=float(no_token.get("price", 0)),
                            volume=0,
                            end_date=m.get("end_date_iso", ""),
                            active=True,
                        )
                        markets.append(market)
                        self._markets[market.condition_id] = market
                else:
                    print(f"[polymarket] CLOB sampling-markets returned HTTP {resp.status}")
        except Exception as e:
            print(f"[polymarket] Error fetching CLOB sampling-markets: {e}")

        # ── Source 2: Gamma events API (fallback / additional markets) ──
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
                return markets

            for event in events:
                title = (event.get("title") or "").lower()
                if "btc" not in title and "bitcoin" not in title:
                    continue

                for market_data in event.get("markets", []):
                    if not market_data.get("active"):
                        continue
                    cid = market_data.get("conditionId", "")
                    if cid in self._markets:
                        continue  # Already found via CLOB
                    tokens = market_data.get("clobTokenIds", [])
                    if len(tokens) < 2:
                        continue
                    prices = market_data.get("outcomePrices", [])
                    if len(prices) < 2:
                        continue

                    market = MarketInfo(
                        condition_id=cid,
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
            print(f"[polymarket] Error discovering Gamma markets: {e}")

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
        """Re-fetch prices for all known markets using CLOB midpoint API."""
        session = await self._ensure_session()
        updated: list[MarketInfo] = []

        for cid, market in list(self._markets.items()):
            try:
                # Use CLOB midpoint for the YES token
                mid = await self.fetch_midpoint(market.token_id_yes)
                if mid is not None:
                    market = MarketInfo(
                        condition_id=cid,
                        question=market.question,
                        token_id_yes=market.token_id_yes,
                        token_id_no=market.token_id_no,
                        outcome_yes_price=mid,
                        outcome_no_price=max(0, 1.0 - mid),
                        volume=market.volume,
                        end_date=market.end_date,
                        active=True,
                    )
                    self._markets[cid] = market
                    updated.append(market)
            except Exception:
                continue

        return updated

    def parse_btc_brackets(self) -> list[BTCBracket]:
        """Parse market questions to extract BTC price brackets.

        Handles two market types:
        - "Will Bitcoin reach $120,000 by ...?" → YES = BTC goes above threshold
        - "Will Bitcoin dip to $55,000 by ...?" → YES = BTC goes below threshold
          (inverted: we swap YES/NO so YES always means "above threshold")

        Filters out non-price-bracket markets (e.g. "MicroStrategy sells Bitcoin").
        """
        import re

        brackets: list[BTCBracket] = []
        price_pattern = re.compile(
            r"\$\s*([\d,]+(?:\.\d+)?)\s*([kK])?", re.IGNORECASE
        )
        # Only match markets that are clearly price-level markets
        price_keywords = re.compile(
            r"(reach|above|below|dip|hit|over|under)", re.IGNORECASE
        )

        for market in self._markets.values():
            if not market.active:
                continue
            question = market.question
            if not price_keywords.search(question):
                continue
            match = price_pattern.search(question)
            if not match:
                continue

            raw = match.group(1).replace(",", "")
            threshold = float(raw)
            if match.group(2):
                threshold *= 1000

            # Skip if threshold is unreasonably far from BTC prices
            # (e.g. "$1m" markets, "$1700" old markets)
            if threshold < 10_000 or threshold > 500_000:
                continue

            # "dip to" markets: YES means BTC goes BELOW threshold
            # Invert so YES always means "above threshold" for the detector
            is_dip = bool(re.search(r"\b(dip|below|under)\b", question, re.IGNORECASE))
            if is_dip:
                yes_price = market.outcome_no_price
                no_price = market.outcome_yes_price
                token_yes = market.token_id_no
                token_no = market.token_id_yes
            else:
                yes_price = market.outcome_yes_price
                no_price = market.outcome_no_price
                token_yes = market.token_id_yes
                token_no = market.token_id_no

            brackets.append(
                BTCBracket(
                    threshold_price=threshold,
                    yes_price=yes_price,
                    no_price=no_price,
                    market=market,
                    condition_id=market.condition_id,
                    token_id_yes=token_yes,
                    token_id_no=token_no,
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
