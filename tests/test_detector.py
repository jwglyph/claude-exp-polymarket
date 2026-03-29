"""Tests for the divergence detector."""

import pytest

from src.detector import DivergenceDetector, Signal
from src.polymarket import BTCBracket, MarketInfo


def _make_bracket(threshold: float, yes_price: float) -> BTCBracket:
    market = MarketInfo(
        condition_id="test",
        question=f"Will BTC be above ${threshold:,.0f}?",
        token_id_yes="yes-token",
        token_id_no="no-token",
        outcome_yes_price=yes_price,
        outcome_no_price=round(1 - yes_price, 4),
        volume=100000,
        end_date="2026-04-01",
        active=True,
    )
    return BTCBracket(
        threshold_price=threshold,
        yes_price=yes_price,
        no_price=round(1 - yes_price, 4),
        market=market,
        condition_id="test",
        token_id_yes="yes-token",
        token_id_no="no-token",
    )


class TestFairValue:
    def test_price_well_above_threshold(self):
        detector = DivergenceDetector()
        # BTC at $100k, threshold at $90k — should be very high probability
        fair = detector.estimate_fair_value(100000, 90000, time_to_expiry_hours=24)
        assert fair > 0.8

    def test_price_well_below_threshold(self):
        detector = DivergenceDetector()
        # BTC at $80k, threshold at $100k
        fair = detector.estimate_fair_value(80000, 100000, time_to_expiry_hours=24)
        assert fair < 0.1

    def test_price_at_threshold(self):
        detector = DivergenceDetector()
        # BTC exactly at threshold — should be ~0.5
        fair = detector.estimate_fair_value(95000, 95000, time_to_expiry_hours=24)
        assert 0.45 < fair < 0.55

    def test_short_time_to_expiry_above(self):
        detector = DivergenceDetector()
        # Very close to expiry, price above threshold
        fair = detector.estimate_fair_value(96000, 95000, time_to_expiry_hours=0.01)
        assert fair > 0.95

    def test_short_time_to_expiry_below(self):
        detector = DivergenceDetector()
        fair = detector.estimate_fair_value(94000, 95000, time_to_expiry_hours=0.01)
        assert fair < 0.05


class TestScanBrackets:
    def test_detects_buy_yes_signal(self):
        """When market underprices YES, we should get BUY_YES."""
        detector = DivergenceDetector(divergence_threshold=0.003)
        # BTC at $100k, threshold $90k → fair value ~0.95+
        # But market only shows 0.65 → huge edge
        bracket = _make_bracket(90000, 0.65)
        opps = detector.scan_brackets([bracket], 100000, time_to_expiry_hours=24)
        assert len(opps) == 1
        assert opps[0].signal == Signal.BUY_YES
        assert opps[0].edge > 0.1

    def test_detects_buy_no_signal(self):
        """When market overprices YES, we should get BUY_NO."""
        detector = DivergenceDetector(divergence_threshold=0.003)
        # BTC at $80k, threshold $100k → fair value very low
        # But market shows YES at 0.70 → should buy NO
        bracket = _make_bracket(100000, 0.70)
        opps = detector.scan_brackets([bracket], 80000, time_to_expiry_hours=24)
        assert len(opps) == 1
        assert opps[0].signal == Signal.BUY_NO
        assert opps[0].edge < -0.1

    def test_hold_when_no_divergence(self):
        """When prices are fairly valued, signal should be HOLD."""
        detector = DivergenceDetector(divergence_threshold=0.003)
        # Fair value for price at threshold ≈ 0.5
        bracket = _make_bracket(95000, 0.50)
        opps = detector.scan_brackets([bracket], 95000, time_to_expiry_hours=24)
        assert len(opps) == 1
        assert opps[0].signal == Signal.HOLD

    def test_multiple_brackets_sorted_by_edge(self):
        detector = DivergenceDetector(divergence_threshold=0.003)
        brackets = [
            _make_bracket(90000, 0.50),  # Small edge (BTC at 95k)
            _make_bracket(80000, 0.50),  # Bigger edge (BTC way above)
        ]
        opps = detector.scan_brackets(brackets, 95000, time_to_expiry_hours=24)
        assert len(opps) == 2
        # Should be sorted by absolute edge, biggest first
        assert abs(opps[0].edge) >= abs(opps[1].edge)
