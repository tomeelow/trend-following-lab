"""Phase 3: make the backtest trustworthy.

Three ways a backtest lies to you, and the tools to catch them:

1. Lookahead bias: using information you wouldn't have had at the time.
   `cheating_weights` deliberately does this so you can see how good a broken
   backtest looks.
2. Overfitting: trying many settings and keeping the best one. The best
   in-sample setting is partly just luck, and luck doesn't repeat.
   `split_half_sharpes` shows how poorly "best in the first half" predicts
   the second half.
3. Judging a choice on the same data used to make it. `walk_forward` only
   ever picks a setting using data from *before* the period it's tested on.
"""

from dataclasses import dataclass

import pandas as pd

from quant.backtest import simulate
from quant.metrics import sharpe_ratio
from quant.strategy import run_backtest, signals, trailing_returns


# ------------------------------------------------------------------ 1. lookahead
def cheating_weights(monthly_prices: pd.DataFrame, lookback_months: int = 12) -> pd.DataFrame:
    """DON'T DO THIS. Same rule as Phase 1 but without the one-month shift: the
    signal from the end of month t decides what we hold *during* month t, i.e.
    we use month t's closing price to decide whether to own the ETF in month t."""
    sig = signals(trailing_returns(monthly_prices, lookback_months))
    weights = sig / sig.shape[1]
    return weights.loc[weights.dropna().index[0]:]


def cheating_backtest(monthly_prices, asset_returns, lookback_months=12, **kwargs):
    weights = cheating_weights(monthly_prices, lookback_months)
    weights = weights.loc[weights.index.intersection(asset_returns.dropna().index)]
    return simulate(weights, asset_returns, **kwargs)


# ------------------------------------------------------------------ 2. many settings
def candidate_returns(
    monthly_prices: pd.DataFrame,
    asset_returns: pd.DataFrame,
    lookbacks: list[int],
    cash_returns: pd.Series | None = None,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Monthly (after-cost) returns of the strategy for every candidate lookback.

    Columns = lookbacks. Rows are trimmed to the months where *every* candidate
    is running, so they're all compared over identical periods.
    """
    out = {
        lb: run_backtest(monthly_prices, lb, cash_returns=cash_returns, cost_bps=cost_bps,
                         asset_returns=asset_returns).portfolio_return
        for lb in lookbacks
    }
    return pd.DataFrame(out).dropna()


def sharpe_by_column(returns: pd.DataFrame, risk_free: pd.Series) -> pd.Series:
    return returns.apply(lambda col: sharpe_ratio(col, risk_free))


def split_half_sharpes(cand: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    """Sharpe of every lookback in the first half vs. the second half of history.

    If the best first-half settings were *really* better, they'd also tend to
    be better in the second half. Usually the relationship is weak.
    """
    middle = len(cand) // 2
    first, second = cand.iloc[:middle], cand.iloc[middle:]
    out = pd.DataFrame({
        "first_half": sharpe_by_column(first, risk_free),
        "second_half": sharpe_by_column(second, risk_free),
    })
    out.attrs["split_date"] = second.index[0]
    return out


# ------------------------------------------------------------------ 3. walk-forward
@dataclass
class WalkForwardResult:
    returns: pd.Series        # out-of-sample monthly returns, stitched together
    choices: pd.DataFrame     # one row per test year: train window, chosen lookback, its train Sharpe


def walk_forward(
    cand: pd.DataFrame,
    risk_free: pd.Series,
    first_test_year: int,
    min_train_months: int = 36,
) -> WalkForwardResult:
    """Each January, pick the lookback with the best Sharpe on *all data before
    that January* (an expanding training window), then use it for the next 12
    months. The test year's data is never used to make that year's choice.

    Simplification: switching lookback in January is treated as if you had been
    running the new lookback all along (the one-off trade to switch isn't charged).
    """
    pieces, rows = [], []
    for year in range(first_test_year, cand.index[-1].year + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = cand[cand.index < cutoff]
        test = cand[(cand.index >= cutoff) & (cand.index.year == year)]
        if len(train) < min_train_months or test.empty:
            continue
        train_sharpe = sharpe_by_column(train, risk_free)
        chosen = int(train_sharpe.idxmax())
        pieces.append(test[chosen])
        rows.append({
            "Test year": year,
            "Trained on": f"{train.index[0]:%b %Y} – {train.index[-1]:%b %Y}",
            "Months of training": len(train),
            "Chosen lookback": chosen,
            "Its training Sharpe": train_sharpe[chosen],
        })
    return WalkForwardResult(pd.concat(pieces), pd.DataFrame(rows))
