"""Shared pieces for the Streamlit pages: settings, cached data loading, chart styling."""

import time
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant.data import load_prices, load_tbill_yield
from quant.strategy import month_end_prices

UNIVERSE = {
    "SPY": "US large-cap stocks (S&P 500)",
    "EFA": "Developed markets ex-US stocks",
    "EEM": "Emerging markets stocks",
    "TLT": "US Treasuries, 20+ years",
    "GLD": "Gold",
    "QQQ": "US tech-heavy stocks (Nasdaq 100)",
    "IEF": "US Treasuries, 7-10 years",
    "VNQ": "US real estate (REITs)",
}
# Fixed color per ticker, so a color always means the same ETF.
COLORS = dict(zip(UNIVERSE, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                             "#e87ba4", "#008300", "#6250d6", "#e34948"]))
CASH_COLOR = "#a3a29c"
IN_COLOR = "#2a78d6"


@dataclass
class Settings:
    tickers: list[str]
    lookback: int
    capital: float
    refresh_token: tuple[int, int]


# Prices are re-downloaded at most every DATA_MAX_AGE_HOURS, so a long-running (hosted)
# app picks up new months on its own.
DATA_MAX_AGE_HOURS = 12


def settings() -> Settings:
    """The sidebar values (the widgets live in app.py and write to session state).

    `refresh_token` is the cache key for everything built from prices: (which 12-hour
    window we're in, how many times "Re-download prices" was pressed). When either
    changes, every cached result is recomputed from fresh data.
    """
    s = st.session_state
    window = int(time.time() // (DATA_MAX_AGE_HOURS * 3600))
    return Settings(list(s.tickers), int(s.lookback), float(s.capital), (window, s.get("refresh_token", 0)))


@st.cache_data(show_spinner="Downloading prices…", max_entries=20)
def daily_prices(tickers: tuple[str, ...], refresh_token: tuple[int, int]) -> pd.DataFrame:
    return load_prices(list(tickers), refresh=refresh_token[1] > 0, max_age_hours=DATA_MAX_AGE_HOURS)


@st.cache_data(show_spinner="Downloading T-bill rates…", max_entries=4)
def tbill_yield(refresh_token: tuple[int, int]) -> pd.Series:
    return load_tbill_yield(refresh=refresh_token[1] > 0, max_age_hours=DATA_MAX_AGE_HOURS)


def monthly_prices(cfg: Settings) -> pd.DataFrame:
    return month_end_prices(daily_prices(tuple(cfg.tickers), cfg.refresh_token))


# ------------------------------------------------------------------ formatting
def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    """Newest month on top, dates shown without the 00:00:00 time."""
    out = df.sort_index(ascending=False)
    out.index = out.index.strftime("%Y-%m-%d")
    return out


def pct(x: float) -> str:
    return f"{x:+.1%}"


def style(fig: go.Figure, height: int = 380, yfmt: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_xaxes(showgrid=False)
    if yfmt:
        fig.update_yaxes(tickformat=yfmt)
    return fig


# ------------------------------------------------------------------ Phases 3 & 4: realistic setup
COST_BPS = 10          # trading cost used from Phase 3 on
LAG_DAYS = 1           # trade at the close of the day after the signal
FIRST_FORECAST_YEAR = 2008  # first year with out-of-sample volatility forecasts


@dataclass
class Market:
    daily: pd.DataFrame     # daily prices of the selected ETFs
    monthly: pd.DataFrame   # month-end prices (decision dates)
    rf: pd.Series           # monthly T-bill return (cash rate)
    returns: pd.DataFrame   # holding-period returns with the trading delay, plus a SPY column


@st.cache_data(show_spinner="Preparing data…", max_entries=20)
def market(tickers: tuple[str, ...], refresh_token: tuple[int, int], lag_days: int = LAG_DAYS) -> Market:
    from quant.backtest import cash_returns_from_yield
    from quant.strategy import execution_returns

    daily = daily_prices(tickers, refresh_token)
    monthly = month_end_prices(daily)
    returns = execution_returns(daily, monthly.index, lag_days)
    spy_daily = daily_prices(("SPY",), refresh_token)
    returns["SPY"] = execution_returns(spy_daily, monthly.index, lag_days)["SPY"]
    rf = cash_returns_from_yield(tbill_yield(refresh_token), monthly.index)
    return Market(daily, monthly, rf, returns)


@st.cache_data(show_spinner="Fitting volatility models (GARCH + LightGBM, walk-forward)… ~15 s the first time",
               persist="disk", max_entries=8)
def vol_forecasts(tickers: tuple[str, ...], refresh_token: tuple[int, int]) -> dict:
    from quant import volatility as V

    daily = daily_prices(tickers, refresh_token)
    dates = month_end_prices(daily).index
    realized = V.realized_next_month(daily, dates)
    garch, garch_params = V.garch_forecast(daily, dates, first_year=FIRST_FORECAST_YEAR - 1)
    lgbm, importance = V.lgbm_forecast(daily, dates, realized, first_year=FIRST_FORECAST_YEAR)
    return {
        "realized": realized,
        "forecasts": {
            "Moving avg. (1 month)": V.moving_average_forecast(daily, dates, 21),
            "Moving avg. (3 months)": V.moving_average_forecast(daily, dates, 63),
            "GARCH(1,1)": garch,
            "LightGBM": lgbm,
        },
        "garch_params": garch_params,
        "importance": importance,
    }
