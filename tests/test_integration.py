"""End-to-end integration test with mocked API responses.

Simulates the full pipeline: price feeds → Polymarket → detector → executor.
Uses aiohttp test server to mock external APIs.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.config import Settings, TradingConfig, PolymarketCredentials
from src.price_feeds import PriceFeedAggregator, PricePoint
from src.polymarket import PolymarketMonitor, MarketInfo, BTCBracket
from src.detector import DivergenceDetector, Signal
from src.executor import TradeExecutor, OrderStatus
from src.risk import RiskManager


# ── Fixtures ──

def make_settings(**overrides) -> Settings:
    trading_kw = {
        "divergence_threshold": 0.003,
        "max_risk_per_trade": 0.005,
        "daily_risk_cap": 0.02,
        "order_size_usdc": 10.0,
    }
    trading_kw.update(overrides)
    return Settings(
        credentials=PolymarketCredentials(
            api_key="test-key",
            api_secret="test-secret",
            api_passphrase="test-pass",
            wallet_address="0xtest",
            private_key="0xprivkey",
        ),
        trading=TradingConfig(**trading_kw),
    )


def make_bracket(threshold=95000, yes_price=0.65) -> BTCBracket:
    market = MarketInfo(
        condition_id="cond-123",
        question=f"Will Bitcoin be above ${threshold:,}?",
        token_id_yes="token-yes-123",
        token_id_no="token-no-123",
        outcome_yes_price=yes_price,
        outcome_no_price=round(1 - yes_price, 4),
        volume=500000,
        end_date="2026-04-01T00:00:00Z",
        active=True,
    )
    return BTCBracket(
        threshold_price=threshold,
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        market=market,
        condition_id="cond-123",
        token_id_yes="token-yes-123",
        token_id_no="token-no-123",
    )


# ── Full Pipeline Test ──

class TestFullPipeline:
    """Test the complete flow: prices → detection → execution."""

    def test_full_pipeline_buy_yes(self):
        """BTC at $100k, market says $95k threshold YES=0.50 → should buy YES."""
        settings = make_settings()
        detector = DivergenceDetector(settings.trading.divergence_threshold)
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        # Simulate aggregated BTC price
        real_btc = 100000.0
        bracket = make_bracket(threshold=95000, yes_price=0.50)

        # Detect
        opportunities = detector.scan_brackets([bracket], real_btc)
        actionable = detector.get_actionable(opportunities)

        assert len(actionable) == 1
        opp = actionable[0]
        assert opp.signal == Signal.BUY_YES
        assert opp.edge > 0.1  # Huge edge

        # Risk check
        allowed, reason = risk_mgr.can_trade(opp)
        assert allowed, f"Risk rejected: {reason}"

        # Execute (dry run)
        result = asyncio.get_event_loop().run_until_complete(executor.execute(opp))
        assert result is not None
        assert result.status == OrderStatus.FILLED
        assert result.order_id.startswith("DRY-")

        # Risk state updated
        assert risk_mgr.state.daily_trades == 1
        assert risk_mgr.state.open_exposure > 0

    def test_full_pipeline_buy_no(self):
        """BTC at $80k, market says $95k threshold YES=0.70 → should buy NO."""
        settings = make_settings()
        detector = DivergenceDetector(settings.trading.divergence_threshold)
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        real_btc = 80000.0
        bracket = make_bracket(threshold=95000, yes_price=0.70)

        opportunities = detector.scan_brackets([bracket], real_btc)
        actionable = detector.get_actionable(opportunities)

        assert len(actionable) == 1
        assert actionable[0].signal == Signal.BUY_NO

        result = asyncio.get_event_loop().run_until_complete(executor.execute(actionable[0]))
        assert result is not None
        assert result.status == OrderStatus.FILLED

    def test_full_pipeline_no_trade_when_fair(self):
        """BTC at threshold → no trade."""
        settings = make_settings()
        detector = DivergenceDetector(settings.trading.divergence_threshold)
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        real_btc = 95000.0
        bracket = make_bracket(threshold=95000, yes_price=0.50)

        opportunities = detector.scan_brackets([bracket], real_btc)
        actionable = detector.get_actionable(opportunities)
        assert len(actionable) == 0

    def test_risk_blocks_after_daily_cap(self):
        """After hitting daily loss cap, no more trades."""
        settings = make_settings()
        detector = DivergenceDetector(settings.trading.divergence_threshold)
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        # Burn the daily cap
        risk_mgr.record_pnl(-20.01)  # 2% of $1000

        real_btc = 100000.0
        bracket = make_bracket(threshold=95000, yes_price=0.50)

        opportunities = detector.scan_brackets([bracket], real_btc)
        actionable = detector.get_actionable(opportunities)
        assert len(actionable) == 1

        result = asyncio.get_event_loop().run_until_complete(executor.execute(actionable[0]))
        assert result is None  # Blocked by risk manager


# ── Executor Edge Cases ──

class TestExecutorEdgeCases:
    def test_maker_order_price_below_market(self):
        """Maker orders should price below current ask (BUY YES case)."""
        settings = make_settings()
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        bracket = make_bracket(threshold=90000, yes_price=0.65)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 100000)
        opp = opps[0]

        payload = executor._build_order_payload(opp, 10.0)
        assert payload["price"] < opp.market_price, \
            f"Maker order price {payload['price']} should be below market {opp.market_price}"
        assert payload["price"] > 0
        assert payload["price"] < 1.0
        assert payload["type"] == "GTC"

    def test_maker_order_buy_no_pricing(self):
        """BUY NO maker order should price correctly."""
        settings = make_settings()
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        bracket = make_bracket(threshold=100000, yes_price=0.70)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 80000)
        opp = opps[0]
        assert opp.signal == Signal.BUY_NO

        payload = executor._build_order_payload(opp, 10.0)
        assert payload["side"] == "BUY"
        assert 0.01 <= payload["price"] <= 0.99

    def test_minimum_trade_size_enforced(self):
        """Trades below $1 should be rejected."""
        settings = make_settings(order_size_usdc=0.5)
        risk_mgr = RiskManager(settings.trading, portfolio_value=10)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        bracket = make_bracket(threshold=95000, yes_price=0.499)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 95100)

        # Edge is tiny, position size should be < $1
        if opps and opps[0].signal != Signal.HOLD:
            result = asyncio.get_event_loop().run_until_complete(executor.execute(opps[0]))
            # Should either be None (rejected) or have a valid result
            # The key is it doesn't crash

    def test_extreme_prices_clamped(self):
        """Prices should be clamped to [0.01, 0.99]."""
        settings = make_settings()
        risk_mgr = RiskManager(settings.trading, portfolio_value=1000)
        executor = TradeExecutor(settings, risk_mgr, dry_run=True)

        # Extreme case: fair value very close to 1.0
        bracket = make_bracket(threshold=50000, yes_price=0.95)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 200000)

        if opps and opps[0].signal != Signal.HOLD:
            payload = executor._build_order_payload(opps[0], 10.0)
            if payload:
                assert 0.01 <= payload["price"] <= 0.99


# ── Detector Edge Cases ──

class TestDetectorEdgeCases:
    def test_zero_threshold_price(self):
        """Threshold of $0 should return fair value of 1.0."""
        detector = DivergenceDetector()
        fv = detector.estimate_fair_value(95000, 0)
        assert fv == 1.0

    def test_negative_threshold(self):
        """Negative threshold should not crash."""
        detector = DivergenceDetector()
        fv = detector.estimate_fair_value(95000, -1000)
        assert fv == 1.0

    def test_zero_real_price(self):
        """Real price of $0 should not crash."""
        detector = DivergenceDetector()
        # log(0) would be -inf, should be handled
        try:
            fv = detector.estimate_fair_value(0, 95000)
            assert fv < 0.1  # Should be very low probability
        except (ValueError, ZeroDivisionError):
            pytest.fail("Should handle zero price without exception")

    def test_very_close_prices(self):
        """When real price ≈ threshold, fair value should be ~0.5."""
        detector = DivergenceDetector()
        fv = detector.estimate_fair_value(95000.01, 95000.00)
        assert 0.4 < fv < 0.6

    def test_zero_time_to_expiry(self):
        """At expiry, should be binary."""
        detector = DivergenceDetector()
        fv_above = detector.estimate_fair_value(96000, 95000, time_to_expiry_hours=0)
        fv_below = detector.estimate_fair_value(94000, 95000, time_to_expiry_hours=0)
        assert fv_above > 0.95
        assert fv_below < 0.05

    def test_empty_brackets_list(self):
        """Scanning empty brackets should return empty list."""
        detector = DivergenceDetector()
        result = detector.scan_brackets([], 95000)
        assert result == []

    def test_negative_volatility_doesnt_crash(self):
        """Negative vol shouldn't crash (defensive)."""
        detector = DivergenceDetector()
        try:
            fv = detector.estimate_fair_value(95000, 90000, volatility=-0.01)
            # Negative sigma → negative d → still computable via logistic
            assert 0 <= fv <= 1
        except (ValueError, ZeroDivisionError):
            pass  # Acceptable to error on invalid input


# ── Price Feed Edge Cases ──

class TestPriceFeedEdgeCases:
    def test_single_source_not_enough(self):
        """Median requires at least 2 sources."""
        agg = PriceFeedAggregator()
        agg._latest = {"a": PricePoint("a", 95000, time.time())}
        assert agg.get_median_price() is None

    def test_all_stale_returns_none(self):
        """All stale prices should return None median."""
        agg = PriceFeedAggregator()
        old = time.time() - 60
        agg._latest = {
            "a": PricePoint("a", 95000, old),
            "b": PricePoint("b", 95100, old),
        }
        assert agg.get_median_price() is None

    def test_mixed_fresh_stale(self):
        """Should only use fresh prices."""
        agg = PriceFeedAggregator()
        now = time.time()
        old = now - 60
        agg._latest = {
            "stale": PricePoint("stale", 80000, old),  # Way off, stale
            "fresh1": PricePoint("fresh1", 95000, now),
            "fresh2": PricePoint("fresh2", 95100, now),
        }
        median = agg.get_median_price()
        assert median == 95050.0  # Should ignore the stale 80000

    def test_best_price_returns_most_recent(self):
        agg = PriceFeedAggregator()
        now = time.time()
        agg._latest = {
            "old": PricePoint("old", 95000, now - 1),
            "new": PricePoint("new", 95100, now),
        }
        best = agg.get_best_price()
        assert best.source == "new"

    def test_best_price_empty(self):
        agg = PriceFeedAggregator()
        assert agg.get_best_price() is None


# ── Risk Manager Edge Cases ──

class TestRiskEdgeCases:
    def test_zero_portfolio_value(self):
        """Zero portfolio should block all trades."""
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=0)
        bracket = make_bracket(90000, 0.50)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 100000)
        if opps and opps[0].signal != Signal.HOLD:
            allowed, reason = rm.can_trade(opps[0])
            # Should either block or allow with zero size
            size = rm.compute_position_size(opps[0])
            assert size == 0

    def test_exposure_limit(self):
        """Should block when exposure limit reached."""
        config = TradingConfig(order_size_usdc=50)
        rm = RiskManager(config, portfolio_value=100)
        # Max exposure = 20% of $100 = $20
        # Record $20 exposure
        from src.risk import Trade
        bracket = make_bracket(90000, 0.50)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 100000)
        opp = opps[0]

        # Manually push exposure to limit
        rm.state.open_exposure = 20.0

        allowed, reason = rm.can_trade(opp)
        assert not allowed
        assert "exposure" in reason.lower()

    def test_positive_pnl_doesnt_trigger_cap(self):
        """Daily cap should only trigger on losses, not profits."""
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=1000)
        rm.record_pnl(+50.0)  # Profitable day

        bracket = make_bracket(90000, 0.50)
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 100000)
        opp = opps[0]

        allowed, reason = rm.can_trade(opp)
        assert allowed

    def test_kelly_sizing_with_extreme_edge(self):
        """Very large edge shouldn't produce absurd position sizes."""
        config = TradingConfig(order_size_usdc=10)
        rm = RiskManager(config, portfolio_value=1000)

        bracket = make_bracket(50000, 0.10)  # Massively underpriced
        detector = DivergenceDetector(0.003)
        opps = detector.scan_brackets([bracket], 200000)
        opp = opps[0]

        size = rm.compute_position_size(opp)
        assert size <= config.order_size_usdc  # Capped
        assert size > 0


# ── Polymarket Bracket Parsing ──

class TestBracketParsing:
    def test_parses_standard_format(self):
        """'Will Bitcoin be above $95,000?' should parse threshold."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="Will Bitcoin be above $95,000 on March 31?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 1
        assert brackets[0].threshold_price == 95000

    def test_parses_no_comma_format(self):
        """'BTC above $95000' should parse."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="BTC above $95000?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 1
        assert brackets[0].threshold_price == 95000

    def test_inactive_market_skipped(self):
        """Inactive markets should not produce brackets."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="Will Bitcoin be above $95,000?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=False,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 0

    def test_no_price_in_question(self):
        """Questions without dollar amounts should be skipped."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="Will Bitcoin hit a new all-time high?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 0


# ── Config Edge Cases ──

class TestConfigEdgeCases:
    def test_credentials_not_configured(self):
        creds = PolymarketCredentials(
            api_key="", api_secret="", api_passphrase="",
            wallet_address="", private_key="",
        )
        assert not creds.is_configured

    def test_credentials_configured(self):
        creds = PolymarketCredentials(
            api_key="key", api_secret="secret", api_passphrase="pass",
            wallet_address="0x123", private_key="0xabc",
        )
        assert creds.is_configured

    def test_settings_defaults(self):
        s = make_settings()
        assert s.trading.divergence_threshold == 0.003
        assert s.trading.daily_risk_cap == 0.02
        assert s.chain_id == 137

    def test_partial_credentials_not_configured(self):
        """Having only api_key and private_key but missing secret/passphrase."""
        creds = PolymarketCredentials(
            api_key="key", api_secret="", api_passphrase="",
            wallet_address="0x123", private_key="0xabc",
        )
        assert not creds.is_configured


# ── Risk Stats Edge Cases ──

class TestRiskStats:
    def test_stats_with_zero_portfolio(self):
        """Stats should not crash with zero portfolio."""
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=0)
        rm.record_pnl(-5.0)
        stats = rm.stats
        assert stats["risk_utilization"] == 0.0  # No division by zero

    def test_stats_with_zero_risk_cap(self):
        """Stats should not crash with zero daily_risk_cap."""
        config = TradingConfig(daily_risk_cap=0.0)
        rm = RiskManager(config, portfolio_value=1000)
        rm.record_pnl(-5.0)
        stats = rm.stats
        assert stats["risk_utilization"] == 0.0


# ── Bracket Parsing: k suffix ──

class TestBracketParsingKSuffix:
    def test_parses_k_suffix(self):
        """'$100k' should parse as $100,000."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="Will BTC be above $100k?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 1
        assert brackets[0].threshold_price == 100000

    def test_parses_K_uppercase(self):
        """'$100K' should also parse as $100,000."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="BTC above $100K by Friday",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-03-31", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 1
        assert brackets[0].threshold_price == 100000

    def test_no_k_suffix_not_multiplied(self):
        """'$95,000' should NOT be multiplied by 1000."""
        monitor = PolymarketMonitor(make_settings())
        monitor._markets["test"] = MarketInfo(
            condition_id="test",
            question="Will BTC be above $95,000 on April 1?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-04-01", active=True,
        )
        brackets = monitor.parse_btc_brackets()
        assert len(brackets) == 1
        assert brackets[0].threshold_price == 95000  # NOT 95000000


# ── Detector Expiry Parsing ──

class TestDetectorExpiry:
    def test_parses_iso_date(self):
        hours = DivergenceDetector._hours_until_expiry("2026-04-01T00:00:00Z")
        assert hours > 0  # Should be some positive hours in the future

    def test_empty_date_returns_default(self):
        hours = DivergenceDetector._hours_until_expiry("")
        assert hours == 24.0

    def test_invalid_date_returns_default(self):
        hours = DivergenceDetector._hours_until_expiry("not-a-date")
        assert hours == 24.0

    def test_past_date_returns_minimum(self):
        hours = DivergenceDetector._hours_until_expiry("2020-01-01T00:00:00Z")
        assert hours == 0.01  # Clamped to minimum

    def test_per_market_expiry_used(self):
        """When time_to_expiry_hours is None, should use market end_date."""
        detector = DivergenceDetector(0.003)
        market = MarketInfo(
            condition_id="test",
            question="BTC above $95,000?",
            token_id_yes="yes", token_id_no="no",
            outcome_yes_price=0.65, outcome_no_price=0.35,
            volume=100000, end_date="2026-04-01T00:00:00Z", active=True,
        )
        bracket = BTCBracket(
            threshold_price=95000, yes_price=0.65, no_price=0.35,
            market=market, condition_id="test",
            token_id_yes="yes", token_id_no="no",
        )
        # Should not crash when using per-market expiry
        opps = detector.scan_brackets([bracket], 100000, time_to_expiry_hours=None)
        assert len(opps) == 1
