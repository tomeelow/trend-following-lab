"""Shared test data. Tests never touch the network: they run on synthetic prices
whose properties we control, so a failure always means the *code* changed."""

import numpy as np
import pandas as pd
import pytest


def make_prices(start="2008-01-01", end="2014-12-31", tickers=("AAA", "BBB", "CCC"), seed=0):
    """Random-walk daily prices with volatility that drifts between calm and wild
    regimes (so volatility models have something to learn)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    cols = {}
    for i, t in enumerate(tickers):
        regime = np.exp(np.cumsum(rng.normal(0, 0.05, len(days))))  # slowly wandering vol level
        vol = 0.01 * np.clip(regime / regime.mean(), 0.3, 4)
        rets = rng.normal(0.0003 * (i - 1), vol)
        cols[t] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(cols, index=days)


@pytest.fixture
def daily():
    return make_prices()


def perturb_after(daily: pd.DataFrame, date: str, factor: float = 3.0) -> pd.DataFrame:
    """Wreck every price after `date`. Anything computed *as of* `date` must not change."""
    out = daily.copy()
    after = out.index > pd.Timestamp(date)
    # Jump some ETFs up and others down overnight, then keep shaking them, so
    # anything that peeks at these prices will certainly change.
    jumps = np.array([factor if i % 2 == 0 else 1 / factor for i in range(out.shape[1])])
    wobble = 1 + 0.2 * np.sin(np.arange(after.sum()))[:, None]
    out.loc[after] = out.loc[after].to_numpy() * jumps * wobble
    return out
