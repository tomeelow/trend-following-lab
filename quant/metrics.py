"""Phase 2: is the strategy actually good?

Every function takes monthly returns or a portfolio value series and returns
one number (or a small table). With monthly data there are 12 periods a year,
which is where the 12 and sqrt(12) below come from.
"""

import numpy as np
import pandas as pd

PERIODS_PER_YEAR = 12


def cagr(value: pd.Series) -> float:
    """Compound annual growth rate: the constant yearly return that turns the
    starting value into the ending value over the same number of years."""
    years = (len(value) - 1) / PERIODS_PER_YEAR
    return (value.iloc[-1] / value.iloc[0]) ** (1 / years) - 1


def annual_volatility(returns: pd.Series) -> float:
    """How much returns swing around their average, scaled to a year.

    Monthly std x sqrt(12): if months are independent, variance grows with time,
    so standard deviation grows with the square root of time.
    """
    return returns.std() * np.sqrt(PERIODS_PER_YEAR)


def sharpe_ratio(returns: pd.Series, risk_free: pd.Series) -> float:
    """Return earned *above cash* per unit of volatility, annualized.

    Excess return = what you made minus what a T-bill would have paid. A Sharpe
    of 0.5 means each 1% of annual volatility bought ~0.5% of extra return.
    """
    excess = returns - risk_free.reindex(returns.index).fillna(0.0)
    return excess.mean() / excess.std() * np.sqrt(PERIODS_PER_YEAR)


def drawdown(value: pd.Series) -> pd.Series:
    """How far below its previous best the portfolio is at each date (0 = at a new high)."""
    return value / value.cummax() - 1


def max_drawdown(value: pd.Series) -> float:
    """The worst peak-to-bottom loss over the whole period."""
    return drawdown(value).min()


def drawdown_episodes(value: pd.Series, top: int = 5) -> pd.DataFrame:
    """The `top` deepest drawdowns: when they started, bottomed and recovered."""
    dd = drawdown(value)
    episodes = []  # (peak, bottom, recovery or None)
    peak = trough = dd.index[0]
    in_drawdown = False
    for date, depth in dd.items():
        if depth == 0:  # at a new high
            if in_drawdown:
                episodes.append((peak, trough, date))
                in_drawdown = False
            peak = date
        elif not in_drawdown:  # first month below the peak
            in_drawdown, trough = True, date
        elif depth < dd[trough]:  # new bottom
            trough = date
    if in_drawdown:  # still under water at the end
        episodes.append((peak, trough, None))

    def months(a: pd.Timestamp, b: pd.Timestamp) -> int:
        return (b.year - a.year) * 12 + (b.month - a.month)

    rows = []
    for peak, trough, recovery in episodes:
        rows.append({
            "Peak": peak,
            "Bottom": trough,
            "Recovered": recovery,
            "Depth": dd[trough],
            "Months falling": months(peak, trough),
            "Months to recover": months(trough, recovery) if recovery is not None else None,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("Depth").head(top).reset_index(drop=True)


def calendar_year_returns(returns: pd.Series) -> pd.Series:
    """Compound the monthly returns within each calendar year."""
    return (1 + returns).groupby(returns.index.year).prod() - 1


def summary(result, risk_free: pd.Series) -> dict[str, float]:
    """All the headline numbers for one backtest (a quant.backtest.BacktestResult)."""
    value, rets = result.portfolio_value, result.portfolio_return
    years = len(rets) / PERIODS_PER_YEAR
    return {
        "Final value": value.iloc[-1],
        "Total return": value.iloc[-1] / value.iloc[0] - 1,
        "CAGR": cagr(value),
        "Volatility": annual_volatility(rets),
        "Sharpe ratio": sharpe_ratio(rets, risk_free),
        "Max drawdown": max_drawdown(value),
        "Worst month": rets.min(),
        "Best month": rets.max(),
        "% months positive": (rets > 0).mean(),
        "Avg. invested": result.weights.sum(axis=1).mean(),
        "Turnover / year": result.turnover.sum() / years,
        "Costs paid": result.cost_dollars.sum(),
    }


def stats_from_returns(returns: pd.Series, risk_free: pd.Series) -> dict[str, float]:
    """CAGR / volatility / Sharpe / max drawdown for any stretch of monthly returns.

    Handy for sub-periods, where there is no ready-made value series.
    """
    value = pd.concat([pd.Series([1.0]), (1 + returns).cumprod().reset_index(drop=True)])
    return {
        "CAGR": cagr(value),
        "Volatility": annual_volatility(returns),
        "Sharpe ratio": sharpe_ratio(returns, risk_free),
        "Max drawdown": max_drawdown(value),
    }
