"""Tests for circuit_breaker_max_stops parameter in VBT portfolio simulation.

Validates that:
1. circuit_breaker_max_stops=0 (disabled) does not suppress any entries
2. circuit_breaker_max_stops=N suppresses entries after N stop-exits
3. Stop counting is correct — only stop-loss exits increment the counter
4. Portfolio.from_signals accepts and passes circuit_breaker_max_stops correctly
5. Legacy behavior (no circuit_breaker) is unchanged
"""

import numpy as np
import pandas as pd
import pytest
import vectorbt as vbt


class TestCircuitBreakerDisabled:
    """When circuit_breaker_max_stops=0, no entries should be suppressed."""

    def test_default_no_suppression(self):
        """Default (0) should not suppress any entries."""
        # Price that goes up, triggers stop, then another entry
        close = pd.Series([100, 99, 98, 95, 90, 100, 110, 105, 95, 100])
        entries = pd.Series([True, False, False, False, False, True, False, False, False, False])
        exits = pd.Series([False, False, False, False, False, False, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,  # 6% stop loss
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=0  # disabled
        )
        trades = pf.trades.records_readable
        # Both entries should execute (circuit breaker disabled)
        assert len(trades) >= 2, f"Expected at least 2 trades with CB disabled, got {len(trades)}"

    def test_explicit_zero_same_as_default(self):
        """circuit_breaker_max_stops=0 should behave identically to not passing it."""
        close = pd.Series([100, 99, 95, 100, 110])
        entries = pd.Series([True, False, False, True, False])
        exits = pd.Series([False, False, False, False, False])

        pf_default = vbt.Portfolio.from_signals(
            close=close, entries=entries, exits=exits,
            sl_stop=0.06, init_cash=10000, size=1
        )
        pf_zero = vbt.Portfolio.from_signals(
            close=close, entries=entries, exits=exits,
            sl_stop=0.06, init_cash=10000, size=1,
            circuit_breaker_max_stops=0
        )
        # Same number of trades
        assert len(pf_default.trades.records_readable) == len(pf_zero.trades.records_readable)


class TestCircuitBreakerEnabled:
    """When circuit_breaker_max_stops > 0, entries should be suppressed after N stop-exits."""

    def test_blocks_entry_after_one_stop(self):
        """circuit_breaker_max_stops=1 should block all entries after the first stop-exit."""
        # Scenario: Enter at 100, stop at ~94 (6%), re-enter attempt at bar 5
        close = pd.Series([100.0, 99.0, 98.0, 93.0, 100.0, 110.0, 105.0])
        entries = pd.Series([True, False, False, False, True, False, False])
        exits = pd.Series([False, False, False, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=1
        )
        trades = pf.trades.records_readable
        # First trade enters and gets stopped out. Second entry should be BLOCKED.
        assert len(trades) == 1, f"Expected 1 trade (2nd blocked by CB), got {len(trades)}"

    def test_allows_entries_before_threshold(self):
        """circuit_breaker_max_stops=2 should allow re-entry after first stop but block after second."""
        # Scenario: Enter, stop, enter again, stop again, try to enter (blocked)
        close = pd.Series([
            100.0, 93.0,   # trade 1: enter 100, stop at ~94 -> stopped at 93
            100.0, 93.0,   # trade 2: re-enter 100, stop again -> stopped at 93
            100.0, 110.0   # trade 3 attempt: should be BLOCKED
        ])
        entries = pd.Series([True, False, True, False, True, False])
        exits = pd.Series([False, False, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=2
        )
        trades = pf.trades.records_readable
        # 2 trades allowed, 3rd blocked
        assert len(trades) == 2, f"Expected 2 trades (3rd blocked by CB=2), got {len(trades)}"

    def test_max_stops_three(self):
        """circuit_breaker_max_stops=3 allows 3 stop-exits then blocks."""
        close = pd.Series([
            100.0, 93.0,   # trade 1: stopped
            100.0, 93.0,   # trade 2: stopped
            100.0, 93.0,   # trade 3: stopped
            100.0, 110.0   # trade 4 attempt: BLOCKED
        ])
        entries = pd.Series([True, False, True, False, True, False, True, False])
        exits = pd.Series([False, False, False, False, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=3
        )
        trades = pf.trades.records_readable
        assert len(trades) == 3, f"Expected 3 trades (4th blocked by CB=3), got {len(trades)}"


class TestCircuitBreakerWithSignalExits:
    """Signal-based exits (not stop-loss) should NOT increment the circuit breaker counter."""

    def test_signal_exits_dont_count(self):
        """Only stop-loss exits should count toward circuit breaker, not signal exits."""
        close = pd.Series([
            100.0, 105.0, 110.0,  # trade 1: enter 100, exit via signal at 110 (profit)
            100.0, 93.0,          # trade 2: enter 100, stopped at ~94
            100.0, 110.0          # trade 3: should be ALLOWED (only 1 stop-exit)
        ])
        entries = pd.Series([True, False, False, True, False, True, False])
        exits = pd.Series([False, False, True, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=1
        )
        trades = pf.trades.records_readable
        # Trade 1: signal exit (doesn't count). Trade 2: stop-exit (count=1).
        # Trade 3: blocked because count=1 >= max_stops=1.
        # So we expect exactly 2 trades completed.
        assert len(trades) == 2, f"Expected 2 trades (signal exit + stop, 3rd blocked), got {len(trades)}"


class TestCircuitBreakerIntegrationWithPortfolio:
    """Integration tests using realistic portfolio scenarios."""

    def test_profitable_trades_unaffected(self):
        """Circuit breaker should not affect entries when there are no stop-outs."""
        close = pd.Series([100, 105, 110, 100, 105, 110, 100, 105, 110])
        entries = pd.Series([True, False, False, True, False, False, True, False, False])
        exits = pd.Series([False, False, True, False, False, True, False, False, True])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=1
        )
        trades = pf.trades.records_readable
        # All 3 trades should execute — no stops hit
        assert len(trades) == 3, f"Expected 3 trades (no stops hit), got {len(trades)}"

    def test_with_trailing_stop(self):
        """Circuit breaker should work with trailing stops."""
        close = pd.Series([
            100.0, 105.0, 110.0, 103.0,  # trade 1: enters 100, trails to 110, stopped at 103 (~6.4% from peak)
            110.0, 103.0,                 # trade 2 attempt: should be BLOCKED (1 stop already)
        ])
        entries = pd.Series([True, False, False, False, True, False])
        exits = pd.Series([False, False, False, False, False, False])

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            sl_stop=0.06,
            sl_trail=True,
            init_cash=10000,
            size=1,
            circuit_breaker_max_stops=1
        )
        trades = pf.trades.records_readable
        # 1 trade allowed, 2nd blocked by circuit breaker
        assert len(trades) == 1, f"Expected 1 trade (trailing stop + CB=1), got {len(trades)}"
