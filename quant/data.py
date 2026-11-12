"""Step 0: get historical prices.

We download *adjusted* daily closing prices from Yahoo Finance. "Adjusted" means
dividends and splits are folded into the price, so the % change of the adjusted
price is the ETF's *total return* (what you'd actually have earned).

Each ticker is cached as data/<TICKER>.csv so we don't hit the network on every run.
"""

import time
import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download_one(ticker: str) -> pd.Series:
    history = yf.Ticker(ticker).history(period="max", auto_adjust=True)
    if history.empty:
        raise ValueError(f"No data returned for {ticker!r}")
    close = history["Close"]
    close.index = close.index.tz_localize(None).normalize()  # drop timezone, keep the date
    close.name = ticker
    return close


def _is_stale(path: Path, max_age_hours: float | None) -> bool:
    return max_age_hours is not None and time.time() - path.stat().st_mtime > max_age_hours * 3600


def load_prices(tickers: list[str], refresh: bool = False, max_age_hours: float | None = None) -> pd.DataFrame:
    """Daily adjusted close prices, one column per ticker.

    A cached file is re-downloaded when `refresh` is set or it is older than
    `max_age_hours`. If that download fails (Yahoo rate-limits cloud servers now
    and then) the older cached copy is used instead of crashing.

    Only dates where *every* ticker has a price are kept, so the backtest
    starts once the youngest ETF exists.
    """
    DATA_DIR.mkdir(exist_ok=True)
    columns = []
    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"
        if not path.exists():
            _download_one(ticker).to_csv(path)
        elif refresh or _is_stale(path, max_age_hours):
            try:
                _download_one(ticker).to_csv(path)
            except Exception as exc:  # network / rate limit: keep the cached copy
                warnings.warn(f"Could not refresh {ticker} ({exc}); using cached prices.", stacklevel=2)
        columns.append(pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(ticker))
    return pd.concat(columns, axis=1).dropna()


def load_tbill_yield(refresh: bool = False, max_age_hours: float | None = None) -> pd.Series:
    """Daily 13-week US Treasury bill yield in annual % (e.g. 4.3 = 4.3%/year).

    This is the standard "risk-free rate": what you earn on cash with ~no risk.
    """
    return load_prices(["^IRX"], refresh=refresh, max_age_hours=max_age_hours)["^IRX"]
