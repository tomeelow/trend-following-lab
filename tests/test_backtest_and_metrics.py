import numpy as np
import pandas as pd
import pytest

from quant import metrics as M
from quant.backtest import cash_returns_from_yield, equal_weight, simulate

MONTHS = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])


def returns_frame(a, b):
    return pd.DataFrame({"A": a, "B": b}, index=MONTHS)


def test_portfolio_return_is_weighted_sum_by_hand():
    """50% in A (+10%) and 50% in B (-4%) must return +3% in that month."""
    rets = returns_frame([np.nan, 0.10, 0.02], [np.nan, -0.04, 0.0])
    weights = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=MONTHS[1:])
    result = simulate(weights, rets, initial_capital=1000)
    assert result.portfolio_return.iloc[0] == pytest.approx(0.03)
    assert result.portfolio_value.iloc[1] == pytest.approx(1030)


def test_first_month_pays_to_buy_in():
    """Starting from all cash, buying 60% of the portfolio is 60% turnover."""
    rets = returns_frame([np.nan, 0.0, 0.0], [np.nan, 0.0, 0.0])
    weights = pd.DataFrame({"A": [0.4, 0.4], "B": [0.2, 0.2]}, index=MONTHS[1:])
    result = simulate(weights, rets)
    assert result.turnover.iloc[0] == pytest.approx(0.6)
    assert result.turnover.iloc[1] == pytest.approx(0.0)  # nothing moved, nothing to trade


def test_costs_are_charged_on_turnover():
    """10 bps on 100% turnover costs exactly 0.1% of the portfolio."""
    rets = returns_frame([np.nan, 0.05, 0.0], [np.nan, 0.05, 0.0])
    weights = equal_weight(MONTHS[1:], ["A", "B"])
    result = simulate(weights, rets, cost_bps=10)
    assert result.cost.iloc[0] == pytest.approx(0.001)
    assert result.portfolio_return.iloc[0] == pytest.approx((1 - 0.001) * 1.05 - 1)


def test_cash_earns_interest():
    """A portfolio that is 100% cash grows at exactly the cash rate."""
    rets = returns_frame([np.nan, 0.5, -0.5], [np.nan, 0.5, -0.5])
    weights = pd.DataFrame(0.0, index=MONTHS[1:], columns=["A", "B"])
    cash = pd.Series(0.01, index=MONTHS)
    result = simulate(weights, rets, cash_returns=cash, initial_capital=100)
    assert result.portfolio_value.iloc[-1] == pytest.approx(100 * 1.01**2)


def test_cash_rate_is_the_one_known_at_the_start_of_the_month():
    """The yield quoted at the end of January is what cash earns during February."""
    days = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    yields = pd.Series([1.2, 2.4, 3.6], index=days)
    cash = cash_returns_from_yield(yields, pd.DatetimeIndex(days))
    assert cash.iloc[1] == pytest.approx(0.012 / 12)
    assert cash.iloc[2] == pytest.approx(0.024 / 12)


def test_cagr_of_doubling_in_one_year_is_100_percent():
    """$100 growing to $200 over exactly 12 months is a 100% annual growth rate."""
    value = pd.Series(np.linspace(100, 200, 13))
    assert M.cagr(value) == pytest.approx(1.0)


def test_max_drawdown_of_a_known_path():
    """100 -> 120 -> 60 is a 50% fall from the peak, even though it later recovers."""
    value = pd.Series([100, 120, 60, 90, 130.0], index=pd.date_range("2020-01-31", periods=5, freq="ME"))
    assert M.max_drawdown(value) == pytest.approx(-0.5)
    episodes = M.drawdown_episodes(value)
    assert episodes.loc[0, "Bottom"] == pd.Timestamp("2020-03-31")
    assert episodes.loc[0, "Months to recover"] == 2  # March bottom -> new high in May


def test_sharpe_matches_the_formula():
    """Sharpe = average return above cash ÷ its standard deviation × √12."""
    rets = pd.Series([0.02, 0.0, 0.03, -0.01])
    rf = pd.Series(0.001, index=rets.index)
    excess = rets - 0.001
    assert M.sharpe_ratio(rets, rf) == pytest.approx(excess.mean() / excess.std() * np.sqrt(12))
