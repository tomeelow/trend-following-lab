import numpy as np
import pandas as pd
import pytest

from conftest import perturb_after
from quant.strategy import (
    execution_prices,
    month_end_prices,
    monthly_returns,
    run_backtest,
    signals,
    strategy_weights,
    trailing_returns,
)


def test_monthly_return_is_simple_percent_change():
    """A month that goes 100 -> 110 is +10%, then 110 -> 99 is -10%."""
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"]))
    rets = monthly_returns(prices)["X"]
    assert rets.iloc[1] == pytest.approx(0.10)
    assert rets.iloc[2] == pytest.approx(-0.10)


def test_unfinished_month_is_dropped(daily):
    """If data stops mid-month, that month has no real month-end price and must not be used."""
    cut = daily.loc[:"2012-06-15"]
    monthly = month_end_prices(cut)
    assert monthly.index[-1] == pd.Timestamp("2012-05-31")


def test_trailing_return_looks_back_exactly_n_months(daily):
    """The 12-month trend at month t is P(t) / P(t-12) - 1, nothing else."""
    monthly = month_end_prices(daily)
    trend = trailing_returns(monthly, 12)
    t = monthly.index[30]
    expected = monthly.iloc[30] / monthly.iloc[18] - 1
    pd.testing.assert_series_equal(trend.loc[t], expected, check_names=False)


def test_position_held_in_month_t_comes_from_signal_at_t_minus_1(daily):
    """The core anti-lookahead rule: the signal from the END of a month is only acted on NEXT month."""
    monthly = month_end_prices(daily)
    sig = signals(trailing_returns(monthly, 12))
    weights = strategy_weights(monthly, 12)
    n = monthly.shape[1]
    for i in range(1, len(weights)):
        t, prev = weights.index[i], monthly.index[monthly.index.get_loc(weights.index[i]) - 1]
        pd.testing.assert_series_equal(weights.loc[t], sig.loc[prev] / n, check_names=False)


def test_positions_never_depend_on_future_prices(daily):
    """Point-in-time test: wreck all prices after a date; every position decided by then must be identical."""
    decided_by = "2011-06-30"
    before = strategy_weights(month_end_prices(daily), 12)
    after = strategy_weights(month_end_prices(perturb_after(daily, decided_by)), 12)
    # Positions held up to and including July 2011 were decided on or before June 30.
    upto = before.index[before.index <= pd.Timestamp("2011-07-31")]
    pd.testing.assert_frame_equal(before.loc[upto], after.loc[upto])


def test_truncated_history_gives_same_past_positions(daily):
    """Running the backtest 'back then' (with less data) must reproduce today's view of the past."""
    full = strategy_weights(month_end_prices(daily), 12)
    short = strategy_weights(month_end_prices(daily.loc[:"2012-03-31"]), 12)
    pd.testing.assert_frame_equal(full.loc[short.index], short)


def test_trades_execute_on_the_next_trading_day(daily):
    """With a 1-day lag we trade at the close of the day AFTER the signal, never at the signal's own close."""
    monthly = month_end_prices(daily)
    exec_px = execution_prices(daily, monthly.index, lag_days=1)
    for d in monthly.index[:-1]:
        next_day = daily.index[daily.index.get_loc(d) + 1]
        assert next_day > d
        pd.testing.assert_series_equal(exec_px.loc[d], daily.loc[next_day], check_names=False)


def test_zero_lag_reproduces_the_phase1_backtest(daily):
    """lag_days=0 must give exactly the original month-end backtest (so the two phases agree)."""
    from quant.strategy import execution_returns

    monthly = month_end_prices(daily)
    phase1 = run_backtest(monthly, 12)
    lag0 = run_backtest(monthly, 12, asset_returns=execution_returns(daily, monthly.index, 0))
    assert np.isclose(phase1.portfolio_value.iloc[-1], lag0.portfolio_value.iloc[-1])
