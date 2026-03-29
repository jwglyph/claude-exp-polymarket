"""Tests for the risk manager."""

from src.config import TradingConfig
from src.detector import DivergenceDetector, Signal
from src.risk import RiskManager
from tests.test_detector import _make_bracket


def _make_opportunity(edge: float = 0.05, signal: Signal = Signal.BUY_YES):
    detector = DivergenceDetector(divergence_threshold=0.003)
    bracket = _make_bracket(90000, 0.65)
    opps = detector.scan_brackets([bracket], 100000, time_to_expiry_hours=24)
    return opps[0]


class TestRiskManager:
    def test_allows_trade_within_limits(self):
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=1000)
        opp = _make_opportunity()
        allowed, reason = rm.can_trade(opp)
        assert allowed
        assert reason == "OK"

    def test_blocks_after_daily_cap(self):
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=1000)
        # Simulate hitting daily loss cap (2% of $1000 = $20)
        rm.record_pnl(-20.0)
        opp = _make_opportunity()
        allowed, reason = rm.can_trade(opp)
        assert not allowed
        assert "Daily loss cap" in reason

    def test_position_sizing_positive(self):
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=1000)
        opp = _make_opportunity()
        size = rm.compute_position_size(opp)
        assert size > 0
        assert size <= config.order_size_usdc

    def test_stats_tracking(self):
        config = TradingConfig()
        rm = RiskManager(config, portfolio_value=1000)
        stats = rm.stats
        assert stats["daily_pnl"] == 0
        assert stats["daily_trades"] == 0
        assert stats["portfolio_value"] == 1000
