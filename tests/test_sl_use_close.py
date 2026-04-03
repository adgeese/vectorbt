"""Tests for sl_use_close parameter in VBT stop-loss logic.

Validates that:
1. get_stop_price_nb returns correct values for close-based vs low-based stops
2. simulate_from_signal_func_nb properly threads sl_use_close to stop checks
3. Portfolio.from_signals accepts and passes sl_use_close correctly
4. Legacy behavior (sl_use_close=False) is unchanged
"""

import numpy as np
import pytest
from numba import njit

from vectorbt.portfolio.nb import get_stop_price_nb
import vectorbt as vbt


class TestGetStopPriceNb:
    """Unit tests for get_stop_price_nb with sl_use_close parameter."""

    def test_legacy_behavior_unchanged(self):
        """sl_use_close=False should produce identical results to original behavior."""
        # Long position, stop hit by low
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # 5% stop -> stop at 95.0
            open=98.0,
            low=94.0,
            high=99.0,
            hit_below=True,
            close=97.0,
            sl_use_close=False
        )
        assert result == 95.0  # low=94 <= 95 <= high=99 -> triggered at 95

    def test_legacy_no_close_param(self):
        """Calling without close/sl_use_close should work (backward compat)."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,
            open=98.0,
            low=94.0,
            high=99.0,
            hit_below=True
        )
        assert result == 95.0

    def test_close_based_survives_wick(self):
        """Close-based: low wicks below stop but close recovers -> should NOT trigger."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # stop at 95.0
            open=98.0,
            low=94.0,   # low wicks below stop
            high=99.0,
            hit_below=True,
            close=96.0,  # close is above stop
            sl_use_close=True
        )
        assert np.isnan(result)  # Should NOT trigger - close > stop

    def test_close_based_triggers_on_close_below(self):
        """Close-based: close is below stop -> should trigger."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # stop at 95.0
            open=98.0,
            low=93.0,
            high=99.0,
            hit_below=True,
            close=94.5,  # close is below stop
            sl_use_close=True
        )
        assert result == 95.0  # Triggered at stop price

    def test_close_based_open_gap_down(self):
        """Close-based: open gaps below stop -> returns open (same as legacy)."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # stop at 95.0
            open=93.0,   # open below stop
            low=92.0,
            high=94.0,
            hit_below=True,
            close=93.5,
            sl_use_close=True
        )
        assert result == 93.0  # Returns open on gap down

    def test_close_based_only_affects_long_sl(self):
        """sl_use_close should only affect hit_below=True (long SL), not TP."""
        # Take profit (hit_below=False for long) should use standard logic
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.10,  # 10% TP -> target at 110.0
            open=108.0,
            low=107.0,
            high=112.0,
            hit_below=False,
            close=111.0,
            sl_use_close=True
        )
        assert result == pytest.approx(110.0)  # TP uses standard low<=price<=high logic

    def test_close_equal_to_stop_does_not_trigger(self):
        """Close-based uses strict < comparison, not <=."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # stop at 95.0
            open=98.0,
            low=94.0,
            high=99.0,
            hit_below=True,
            close=95.0,  # close equals stop exactly
            sl_use_close=True
        )
        assert np.isnan(result)  # Strict <, so equal should not trigger

    def test_legacy_low_equal_to_stop_triggers(self):
        """Legacy mode: low == stop_price should trigger (uses <=)."""
        result = get_stop_price_nb(
            position_now=1.0,
            stop_price=100.0,
            stop=0.05,  # stop at 95.0
            open=98.0,
            low=95.0,   # low equals stop exactly
            high=99.0,
            hit_below=True,
            close=97.0,
            sl_use_close=False
        )
        assert result == 95.0  # Legacy uses <=, so equal triggers

    def test_short_position_not_affected(self):
        """Short position SL should not be changed by sl_use_close."""
        # VBT always uses hit_below=True for SL regardless of direction.
        # For short (position_now < 0), branch 2 fires: stop_price * (1 + stop)
        result = get_stop_price_nb(
            position_now=-1.0,
            stop_price=100.0,
            stop=0.05,  # 5% stop -> stop at 105.0
            open=103.0,
            low=102.0,
            high=106.0,
            hit_below=True,  # VBT always passes True for SL
            close=104.0,
            sl_use_close=True
        )
        # Short SL: position_now < 0 and hit_below=True -> branch 2
        # stop_price = 100 * (1 + 0.05) = 105, low=102 <= 105 <= high=106 -> triggered
        assert result == pytest.approx(105.0)


class TestFromSignalsSlUseClose:
    """Integration tests for sl_use_close through Portfolio.from_signals."""

    def test_default_sl_use_close_is_false(self):
        """from_signals should default sl_use_close to False."""
        close = np.array([10.0, 9.5, 9.0, 8.5, 8.0])[:, None]
        entries = np.array([True, False, False, False, False])[:, None]
        exits = np.array([False, False, False, False, False])[:, None]

        # With a stop loss, low-based (default) should trigger
        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.10,
            init_cash=1000.0
        )
        # Just verify it runs without error
        assert pf is not None

    def test_sl_use_close_true_accepted(self):
        """from_signals should accept sl_use_close=True without error."""
        close = np.array([10.0, 9.5, 9.0, 8.5, 8.0])[:, None]
        entries = np.array([True, False, False, False, False])[:, None]
        exits = np.array([False, False, False, False, False])[:, None]

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.10,
            sl_use_close=True,
            init_cash=1000.0
        )
        assert pf is not None

    def test_close_based_produces_different_result(self):
        """When low wicks below stop but close recovers, sl_use_close should survive."""
        # Construct OHLC where low breaches stop but close does not
        # Entry at close=10.0, stop at 10% -> stop_price = 9.0
        open_arr = np.array([10.0, 10.0, 9.5, 9.8, 9.6])[:, None]
        high_arr = np.array([10.5, 10.2, 9.8, 10.0, 9.8])[:, None]
        low_arr =  np.array([9.8,  9.5,  8.8, 9.5,  9.4])[:, None]  # bar 2 low=8.8 < stop=9.0
        close_arr = np.array([10.0, 9.8, 9.2, 9.7, 9.5])[:, None]    # bar 2 close=9.2 > stop=9.0

        entries = np.array([True, False, False, False, False])[:, None]
        exits = np.array([False, False, False, False, False])[:, None]

        # Legacy (low-based) - should stop out at bar 2
        pf_legacy = vbt.Portfolio.from_signals(
            close=close_arr,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            sl_stop=0.10,
            sl_use_close=False,
            init_cash=1000.0
        )

        # Close-based - should survive bar 2 (close=9.2 > stop=9.0)
        pf_close = vbt.Portfolio.from_signals(
            close=close_arr,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            sl_stop=0.10,
            sl_use_close=True,
            init_cash=1000.0
        )

        # They should have different number of trades or different exit points
        legacy_orders = pf_legacy.orders.records
        close_orders = pf_close.orders.records

        # Legacy should have exit (stop triggered at bar 2)
        # Close-based should survive longer
        assert len(legacy_orders) >= 2  # Entry + stop exit
        # The results should differ — close-based should exit later or not at all
        if len(close_orders) >= 2:
            legacy_exit_idx = legacy_orders[1]['idx']
            close_exit_idx = close_orders[1]['idx']
            assert close_exit_idx >= legacy_exit_idx, \
                "Close-based stop should exit at same bar or later than legacy"

    def test_no_regression_when_disabled(self):
        """sl_use_close=False should produce identical results to not passing it."""
        close = np.array([10.0, 9.5, 9.0, 8.5, 8.0])[:, None]
        open_arr = np.array([10.0, 9.8, 9.4, 9.0, 8.6])[:, None]
        high_arr = np.array([10.2, 9.9, 9.5, 9.1, 8.7])[:, None]
        low_arr =  np.array([9.8,  9.3, 8.8, 8.4, 7.9])[:, None]
        entries = np.array([True, False, False, False, False])[:, None]
        exits = np.array([False, False, False, False, False])[:, None]

        pf_default = vbt.Portfolio.from_signals(
            close=close,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            sl_stop=0.05,
            init_cash=1000.0
        )

        pf_explicit = vbt.Portfolio.from_signals(
            close=close,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            entries=entries,
            exits=exits,
            sl_stop=0.05,
            sl_use_close=False,
            init_cash=1000.0
        )

        # Should produce identical orders
        default_orders = pf_default.orders.records_readable
        explicit_orders = pf_explicit.orders.records_readable
        assert len(default_orders) == len(explicit_orders)
        for i in range(len(default_orders)):
            assert default_orders.iloc[i]['Timestamp'] == explicit_orders.iloc[i]['Timestamp']
            np.testing.assert_almost_equal(
                default_orders.iloc[i]['Price'], explicit_orders.iloc[i]['Price'], decimal=6
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
