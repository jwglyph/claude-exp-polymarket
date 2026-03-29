"""Aggregate real-time BTC prices from multiple public sources.

The core insight: Polymarket updates BTC contract prices slower than
real price feeds. We pull from multiple sources and take the median
to get a robust "true" BTC price, then compare against Polymarket.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from statistics import median

import aiohttp


@dataclass
class PricePoint:
    source: str
    price: float
    timestamp: float = field(default_factory=time.time)

    @property
    def age_ms(self) -> float:
        return (time.time() - self.timestamp) * 1000


class PriceFeedAggregator:
    """Fetches BTC/USD from multiple exchanges and returns the median."""

    # Max age before a price is considered stale
    STALE_THRESHOLD_S = 10.0

    def __init__(self, session: aiohttp.ClientSession | None = None) -> None:
        self._session = session
        self._owns_session = session is None
        self._latest: dict[str, PricePoint] = {}

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5)
            )
            self._owns_session = True
        return self._session

    async def close(self) -> None:
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()

    # ── Individual feed fetchers ──

    async def _fetch_binance(self, session: aiohttp.ClientSession) -> PricePoint | None:
        """Binance spot BTC/USDT ticker."""
        try:
            async with session.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbol": "BTCUSDT"},
            ) as resp:
                data = await resp.json()
                return PricePoint("binance", float(data["price"]))
        except Exception:
            return None

    async def _fetch_coinbase(self, session: aiohttp.ClientSession) -> PricePoint | None:
        """Coinbase spot price."""
        try:
            async with session.get(
                "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            ) as resp:
                data = await resp.json()
                return PricePoint("coinbase", float(data["data"]["amount"]))
        except Exception:
            return None

    async def _fetch_kraken(self, session: aiohttp.ClientSession) -> PricePoint | None:
        """Kraken BTC/USD ticker."""
        try:
            async with session.get(
                "https://api.kraken.com/0/public/Ticker",
                params={"pair": "XBTUSD"},
            ) as resp:
                data = await resp.json()
                result = data["result"]["XXBTZUSD"]
                # 'c' is last trade closed [price, lot-volume]
                return PricePoint("kraken", float(result["c"][0]))
        except Exception:
            return None

    async def _fetch_bybit(self, session: aiohttp.ClientSession) -> PricePoint | None:
        """Bybit BTC/USDT last traded price."""
        try:
            async with session.get(
                "https://api.bybit.com/v5/market/tickers",
                params={"category": "spot", "symbol": "BTCUSDT"},
            ) as resp:
                data = await resp.json()
                item = data["result"]["list"][0]
                return PricePoint("bybit", float(item["lastPrice"]))
        except Exception:
            return None

    async def _fetch_okx(self, session: aiohttp.ClientSession) -> PricePoint | None:
        """OKX BTC/USDT ticker."""
        try:
            async with session.get(
                "https://www.okx.com/api/v5/market/ticker",
                params={"instId": "BTC-USDT"},
            ) as resp:
                data = await resp.json()
                return PricePoint("okx", float(data["data"][0]["last"]))
        except Exception:
            return None

    # ── Aggregation ──

    async def fetch_all(self) -> dict[str, PricePoint]:
        """Fetch from all sources concurrently, update internal state."""
        session = await self._ensure_session()
        fetchers = [
            self._fetch_binance(session),
            self._fetch_coinbase(session),
            self._fetch_kraken(session),
            self._fetch_bybit(session),
            self._fetch_okx(session),
        ]
        results = await asyncio.gather(*fetchers, return_exceptions=True)
        for result in results:
            if isinstance(result, PricePoint):
                self._latest[result.source] = result
        return dict(self._latest)

    def get_fresh_prices(self) -> list[PricePoint]:
        """Return only non-stale prices."""
        now = time.time()
        return [
            p
            for p in self._latest.values()
            if (now - p.timestamp) < self.STALE_THRESHOLD_S
        ]

    def get_median_price(self) -> float | None:
        """Compute median BTC price from fresh sources."""
        fresh = self.get_fresh_prices()
        if len(fresh) < 2:
            return None
        return median(p.price for p in fresh)

    def get_best_price(self) -> PricePoint | None:
        """Most recent price point across all sources."""
        fresh = self.get_fresh_prices()
        if not fresh:
            return None
        return max(fresh, key=lambda p: p.timestamp)
