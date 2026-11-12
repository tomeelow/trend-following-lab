"""Phase 1: a very simple trend-following (time-series momentum) strategy.

The rule, applied to each ETF independently, once a month:

    If the ETF's return over the last 12 months is positive -> hold it.
    Otherwise -> hold cash instead.

Capital is split equally into one "slot" per ETF. With 5 ETFs, each slot is 20%
of the portfolio; a slot is either invested in its ETF or sitting in cash.
At every month end we look at the signals and rebalance back to those weights.

The pipeline (each function is one step, and the app shows each one):

    daily prices
      -> month_end_prices        last trading day's price of each month
      -> monthly_returns         % change from one month end to the next
      -> trailing_returns        % change over the last 12 months   (the "trend")
      -> signals                 1 if trend > 0 else 0
      -> target_weights          signal / number of ETFs, applied NEXT month
      -> run_backtest            weights x returns -> portfolio value over time
"""

import numpy as np
import pandas as pd

from quant.backtest import BacktestResult, simulate


def month_end_prices(daily: pd.DataFrame, drop_incomplete: bool = True) -> pd.DataFrame:
    """Price on the last trading day of each month (indexed by that actual date).

    If the data ends mid-month (e.g. today is the 10th), that last month isn't
    finished, so its "month-end" price isn't real yet. We drop it by default.
    """
    month = daily.index.to_period("M")
    monthly = daily.groupby(month).tail(1)
    if drop_incomplete:
        last_day = daily.index[-1]
        month_is_over = (last_day + pd.offsets.BMonthEnd(0)) == last_day
        if not month_is_over:
            monthly = monthly.iloc[:-1]
    return monthly


def monthly_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """Simple return for each month: P_t / P_{t-1} - 1.

    The return on row t is what you earned *during* month t, i.e. from the end
    of month t-1 to the end of month t.
    """
    return monthly_prices.pct_change()


def trailing_returns(monthly_prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """Return over the past `lookback_months`: P_t / P_{t-12} - 1.

    Row t only uses prices up to and including month end t, so this number is
    known at the close of month t.
    """
    return monthly_prices / monthly_prices.shift(lookback_months) - 1


def signals(trailing: pd.DataFrame) -> pd.DataFrame:
    """1 = trend is up (be invested), 0 = trend is down (be in cash).

    Months without enough history (the first 12) stay NaN = "no signal yet".
    """
    return (trailing > 0).astype(float).where(trailing.notna())


def target_weights(sig: pd.DataFrame) -> pd.DataFrame:
    """Portfolio weight of each ETF *during* each month.

    Two things happen here:
      1. Equal slots: each ETF gets 1/N of the capital when its signal is 1.
      2. shift(1): the signal computed at the end of month t decides what we
         hold during month t+1. Without the shift we'd be using month t's
         return to decide whether to hold the ETF during month t - i.e. peeking
         into the future (lookahead bias).
    """
    n_assets = sig.shape[1]
    return (sig / n_assets).shift(1)


def strategy_weights(monthly_prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """Weights held each month, starting from the first month that has a signal."""
    weights = target_weights(signals(trailing_returns(monthly_prices, lookback_months)))
    return weights.loc[weights.dropna().index[0]:]


def execution_prices(
    daily: pd.DataFrame, decision_dates: pd.DatetimeIndex, lag_days: int = 1
) -> pd.DataFrame:
    """Price at which each month's trades actually happen (Phase 3).

    The signal uses the closing price of the month's last trading day, and you
    only know that price once the market has closed. So the earliest realistic
    trade is at the close `lag_days` trading days later. lag_days=0 reproduces
    the Phase 1/2 simplification (trading at the very price that made the signal).

    Row t = price on the execution day that follows decision date t (NaN if that
    day isn't in the data yet). Indexed by the decision date.
    """
    positions = daily.index.get_indexer(decision_dates)
    if (positions < 0).any():
        raise ValueError("Every decision date must be a trading day in `daily`")
    positions = positions + lag_days
    out = pd.DataFrame(np.nan, index=decision_dates, columns=daily.columns)
    valid = positions < len(daily)
    out.loc[valid] = daily.iloc[positions[valid]].to_numpy()
    return out


def execution_returns(daily: pd.DataFrame, decision_dates: pd.DatetimeIndex, lag_days: int = 1) -> pd.DataFrame:
    """Return of each holding period, labelled like monthly_returns: row t is the
    period whose weights were decided at t-1, i.e. execution(t-1) -> execution(t)."""
    return execution_prices(daily, decision_dates, lag_days).pct_change()


def run_backtest(
    monthly_prices: pd.DataFrame,
    lookback_months: int = 12,
    initial_capital: float = 10_000.0,
    cash_returns: pd.Series | None = None,
    cost_bps: float = 0.0,
    asset_returns: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run the strategy month by month.

    Portfolio return in month t = sum over ETFs of (weight_t x return_t)
                                  + cash_weight_t x cash_return_t

    With the defaults (no costs, cash earns 0%, trade at the signal's own close)
    this is the frictionless Phase 1 backtest. Phase 2 passes T-bill rates and a
    trading cost. Phase 3 passes `asset_returns` from `execution_returns` so trades
    happen the day after the signal.
    """
    weights = strategy_weights(monthly_prices, lookback_months)
    if asset_returns is None:
        asset_returns = monthly_returns(monthly_prices)
    # With a trading delay the newest holding period may not have finished yet.
    weights = weights.loc[weights.index.intersection(asset_returns.dropna().index)]
    return simulate(
        weights,
        asset_returns,
        cash_returns=cash_returns,
        cost_bps=cost_bps,
        initial_capital=initial_capital,
    )


def current_positions(monthly_prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """What the strategy says to hold for the coming month, from the latest month end."""
    trend = trailing_returns(monthly_prices, lookback_months).iloc[-1]
    sig = signals(trend.to_frame().T).iloc[0]
    return pd.DataFrame(
        {"trailing_return": trend, "signal": sig, "weight": sig / len(sig)}
    )
