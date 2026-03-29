"""Tests for price feed aggregator."""

import time

from src.price_feeds import PriceFeedAggregator, PricePoint


class TestPricePoint:
    def test_age_ms(self):
        p = PricePoint("test", 95000.0, time.time() - 1.0)
        assert 900 < p.age_ms < 1200  # ~1000ms with some tolerance


class TestAggregator:
    def test_median_requires_minimum_sources(self):
        agg = PriceFeedAggregator()
        # No data yet
        assert agg.get_median_price() is None

    def test_median_calculation(self):
        agg = PriceFeedAggregator()
        now = time.time()
        agg._latest = {
            "a": PricePoint("a", 95000, now),
            "b": PricePoint("b", 95100, now),
            "c": PricePoint("c", 95050, now),
        }
        median = agg.get_median_price()
        assert median == 95050.0

    def test_stale_prices_excluded(self):
        agg = PriceFeedAggregator()
        old = time.time() - 30  # 30 seconds old
        agg._latest = {
            "old": PricePoint("old", 90000, old),
            "new1": PricePoint("new1", 95000, time.time()),
            "new2": PricePoint("new2", 95100, time.time()),
        }
        fresh = agg.get_fresh_prices()
        assert len(fresh) == 2
        assert all(p.source != "old" for p in fresh)
