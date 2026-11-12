"""The backtest engine: turn a table of weights into a portfolio value over time.

Phase 1 did this inline for one strategy. Phase 2 needs to run the same
accounting for several portfolios (the strategy and the buy-and-hold
benchmarks) and add two pieces of realism:

  * Trading costs. Every time you buy or sell you lose a little to commissions
    and the bid/ask spread. We charge `cost_bps` basis points (1 bp = 0.01%) on
    every dollar traded.
  * Interest on cash. Cash parked at a broker or in T-bills earns interest;
    Phase 1 assumed 0%.

Accounting for one month t:
    1. At the start of the month we trade from the weights we *drifted to* last
       month into this month's target weights. Turnover = total traded.
    2. Pay costs:  cost_t = turnover_t x cost_bps / 10,000
    3. Earn returns:  gross_t = sum_i w_i r_i + w_cash r_cash
    4. Net return:  (1 - cost_t)(1 + gross_t) - 1
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    weights: pd.DataFrame        # ETF weights held during each month
    cash_weight: pd.Series       # 1 - sum of ETF weights
    asset_returns: pd.DataFrame  # each ETF's return in each month
    cash_returns: pd.Series      # what cash earned in each month
    contributions: pd.DataFrame  # weight x return, per ETF
    gross_return: pd.Series      # portfolio return before costs
    turnover: pd.Series          # fraction of the portfolio traded at the start of each month
    cost: pd.Series              # cost as a fraction of the portfolio
    cost_dollars: pd.Series      # cost in $
    portfolio_return: pd.Series  # after costs
    portfolio_value: pd.Series   # starts one month before the first return, at initial_capital


def simulate(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    cash_returns: pd.Series | None = None,
    cost_bps: float = 0.0,
    initial_capital: float = 10_000.0,
) -> BacktestResult:
    """Run any set of monthly target weights through the accounting above.

    `weights` rows are the months the weights are *held*; `asset_returns` must
    contain those months (it may contain more).
    """
    weights = weights.fillna(0.0)
    rets = asset_returns.reindex(index=weights.index, columns=weights.columns)
    if rets.isna().any().any():
        raise ValueError("Missing asset returns for some held months")
    cash_r = (
        pd.Series(0.0, index=weights.index)
        if cash_returns is None
        else cash_returns.reindex(weights.index).fillna(0.0)
    )

    cash_w = 1 - weights.sum(axis=1)
    contributions = weights * rets
    gross = contributions.sum(axis=1) + cash_w * cash_r

    # Weights don't stay put: an ETF that rose 10% while the rest of the
    # portfolio rose 2% is now a bigger share. These are the weights at the end
    # of each month, i.e. what we start from when we rebalance next month.
    drifted = (weights * (1 + rets)).div(1 + gross, axis=0)
    # Before the first month we hold nothing (all cash), so the first trade
    # is buying the initial positions.
    previous = drifted.shift(1).fillna(0.0)
    # Only ETF trades cost money; cash just absorbs the difference.
    turnover = (weights - previous).abs().sum(axis=1)

    cost = turnover * cost_bps / 10_000
    net = (1 - cost) * (1 + gross) - 1

    # Prepend the starting point: the month end before the first held month.
    all_months = asset_returns.index
    first = all_months.get_loc(weights.index[0])
    if first == 0:
        raise ValueError("Need one month of history before the first held month")
    start_date = all_months[first - 1]
    value = pd.concat(
        [pd.Series({start_date: initial_capital}), initial_capital * (1 + net).cumprod()]
    )
    value_before = pd.Series(value.iloc[:-1].to_numpy(), index=weights.index)

    return BacktestResult(
        weights=weights,
        cash_weight=cash_w,
        asset_returns=rets,
        cash_returns=cash_r,
        contributions=contributions,
        gross_return=gross,
        turnover=turnover,
        cost=cost,
        cost_dollars=cost * value_before,
        portfolio_return=net,
        portfolio_value=value,
    )


# ------------------------------------------------------------------ benchmarks
def equal_weight(index: pd.Index, tickers: list[str]) -> pd.DataFrame:
    """Always 1/N in every ETF, rebalanced monthly: the strategy with the signal switched off."""
    return pd.DataFrame(1 / len(tickers), index=index, columns=tickers)


def all_in(index: pd.Index, ticker: str) -> pd.DataFrame:
    """100% in a single ETF, never sold."""
    return pd.DataFrame({ticker: 1.0}, index=index)


# ------------------------------------------------------------------ cash
def cash_returns_from_yield(daily_yield_pct: pd.Series, months: pd.DatetimeIndex) -> pd.Series:
    """Monthly return on cash, from the 13-week T-bill yield (annual %, e.g. 4.3).

    The yield quoted at the end of month t-1 is the rate you lock in for month
    t, so we shift by one month (same idea as the signal shift in Phase 1).
    """
    by_month = daily_yield_pct.groupby(daily_yield_pct.index.to_period("M")).last()
    monthly_rate = (by_month / 100 / 12).shift(1)
    aligned = monthly_rate.reindex(months.to_period("M")).to_numpy()
    return pd.Series(aligned, index=months).fillna(0.0)
