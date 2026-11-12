"""Step 0: get historical prices.

We download *adjusted* daily closing prices from Yahoo Finance. "Adjusted" means
dividends and splits are folded into the price, so the % change of the adjusted
price is the ETF's *total return* (what you'd actually have earned).

Each ticker is cached as data/<TICKER>.csv so we only hit the network once.
"""

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


def load_prices(tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted close prices, one column per ticker.

    Only dates where *every* ticker has a price are kept, so the backtest
    starts once the youngest ETF exists.
    """
    DATA_DIR.mkdir(exist_ok=True)
    columns = []
    for ticker in tickers:
        path = DATA_DIR / f"{ticker}.csv"
        if refresh or not path.exists():
            _download_one(ticker).to_csv(path)
        columns.append(pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].rename(ticker))
    return pd.concat(columns, axis=1).dropna()


def load_tbill_yield(refresh: bool = False) -> pd.Series:
    """Daily 13-week US Treasury bill yield in annual % (e.g. 4.3 = 4.3%/year).

    This is the standard "risk-free rate": what you earn on cash with ~no risk.
    """
    return load_prices(["^IRX"], refresh=refresh)["^IRX"]
