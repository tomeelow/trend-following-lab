"""Phase 4: forecast next month's volatility for each ETF.

Every forecast is made at a month-end *decision date* t using only daily data
up to and including t, and it targets the realized volatility of month t+1
(the month we'll hold the positions for). All numbers are annualized.

Models, from simplest to fanciest:
  * Moving average: the volatility of the last N trading days. "Next month will
    look like the recent past."
  * GARCH(1,1): a classic statistical model in which today's variance is a mix
    of a long-run average, yesterday's variance, and yesterday's surprise.
    Volatility jumps after shocks and then decays back to normal.
  * LightGBM: a machine-learning model (gradient-boosted trees) that learns the
    mapping from a set of features (recent vols, recent returns) to next
    month's volatility, retrained once a year on past data only.
"""

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from arch import arch_model

TRADING_DAYS = 252
HORIZON_DAYS = 21  # ~ one month of trading days


def daily_returns(daily_prices: pd.DataFrame) -> pd.DataFrame:
    return daily_prices.pct_change().iloc[1:]


# ------------------------------------------------------------------ the target
def realized_next_month(daily_prices: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """What actually happened: the volatility of daily returns during month t+1,
    labelled with decision date t. NaN when month t+1 isn't complete yet."""
    rets = daily_returns(daily_prices)
    by_month = rets.groupby(rets.index.to_period("M")).std() * np.sqrt(TRADING_DAYS)
    # Only months that have a real month-end (i.e. are finished) count.
    by_month = by_month[by_month.index.isin(decision_dates.to_period("M"))]
    next_month = decision_dates.to_period("M") + 1
    out = by_month.reindex(next_month)
    out.index = decision_dates
    return out


# ------------------------------------------------------------------ 1. moving average
def moving_average_forecast(
    daily_prices: pd.DataFrame, decision_dates: pd.DatetimeIndex, window: int = 63
) -> pd.DataFrame:
    """Standard deviation of the last `window` daily returns, annualized."""
    vol = daily_returns(daily_prices).rolling(window).std() * np.sqrt(TRADING_DAYS)
    return vol.reindex(decision_dates)


# ------------------------------------------------------------------ 2. GARCH
def garch_forecast(
    daily_prices: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    first_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """GARCH(1,1) forecasts of the next 21 trading days' volatility.

    Parameters are re-estimated every January using only data before that
    January. During the year the fitted model is updated with each new day of
    data (that's just the GARCH recursion, no re-fitting). To make lookahead
    impossible, the model for year Y is never even given data after year Y.

    Returns (forecasts, latest fitted parameters per ETF).
    """
    rets = daily_returns(daily_prices) * 100  # arch works best with % returns
    forecasts = pd.DataFrame(np.nan, index=decision_dates, columns=daily_prices.columns)
    params = {}
    for ticker in daily_prices.columns:
        r = rets[ticker].dropna()
        for year in range(first_year, decision_dates[-1].year + 1):
            dates = decision_dates[decision_dates.year == year]
            if dates.empty:
                continue
            visible = r[r.index <= dates[-1]]  # nothing after this year's last decision
            model = arch_model(visible, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
            fit_end = visible.index[visible.index >= pd.Timestamp(year=year, month=1, day=1)][0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit(last_obs=fit_end, disp="off", show_warning=False)
                var = fitted.forecast(horizon=HORIZON_DAYS, start=dates[0], reindex=False).variance
            month_var = var.loc[dates].sum(axis=1)  # variance over the next 21 days, in %^2
            forecasts.loc[dates, ticker] = np.sqrt(month_var * TRADING_DAYS / HORIZON_DAYS) / 100
            params[ticker] = fitted.params
    return forecasts, pd.DataFrame(params).T


# ------------------------------------------------------------------ 3. LightGBM
FEATURES = {
    "vol_5d": "volatility of the last 5 days",
    "vol_21d": "volatility of the last month",
    "vol_63d": "volatility of the last 3 months",
    "vol_252d": "volatility of the last year",
    "vol_ewma": "exponentially weighted volatility (recent days count more)",
    "vol_ratio": "last month's vol ÷ last year's vol (is it unusually calm or wild?)",
    "ret_21d": "return over the last month",
    "ret_63d": "return over the last 3 months",
    "drawdown_252d": "distance below the 1-year high",
    "max_move_21d": "largest single-day move in the last month",
    "etf": "which ETF this row is about",
}


def vol_features(daily_prices: pd.DataFrame, decision_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per (decision date, ETF). Each feature only looks backwards from t."""
    rets = daily_returns(daily_prices)
    ann = np.sqrt(TRADING_DAYS)
    parts = {
        "vol_5d": rets.rolling(5).std() * ann,
        "vol_21d": rets.rolling(21).std() * ann,
        "vol_63d": rets.rolling(63).std() * ann,
        "vol_252d": rets.rolling(252).std() * ann,
        "vol_ewma": np.sqrt((rets**2).ewm(alpha=0.06).mean()) * ann,
        "ret_21d": daily_prices.pct_change(21),
        "ret_63d": daily_prices.pct_change(63),
        "drawdown_252d": daily_prices / daily_prices.rolling(252).max() - 1,
        "max_move_21d": rets.abs().rolling(21).max(),
    }
    parts["vol_ratio"] = parts["vol_21d"] / parts["vol_252d"]
    long = pd.concat(
        {name: frame.reindex(decision_dates).stack(future_stack=True) for name, frame in parts.items()},
        axis=1,
    )
    long.index.names = ["date", "etf"]
    long["etf"] = pd.Categorical(long.index.get_level_values("etf"), categories=list(daily_prices.columns))
    return long[list(FEATURES)]


LGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "seed": 0,
    "deterministic": True,
    "verbose": -1,
}
LGBM_ROUNDS = 300


def training_rows(dates: pd.Index, year: int) -> np.ndarray:
    """Which rows may be used to train the model that forecasts during `year`.

    Row t's answer (the volatility of month t+1) is only known once month t+1
    has ended. Training in January of `year` we know every month up to
    December of year-1, so row t is usable only if t+1 <= December(year-1).
    Using any later row would leak the future into training.
    """
    last_known = pd.Period(year=year - 1, month=12, freq="M")
    return (pd.DatetimeIndex(dates).to_period("M") + 1 <= last_known)


def lgbm_forecast(
    daily_prices: pd.DataFrame,
    decision_dates: pd.DatetimeIndex,
    target: pd.DataFrame,
    first_year: int,
) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward LightGBM: every January, train on all rows whose answer is
    already known, then forecast each month of that year.

    The model predicts log(volatility): volatility is always positive and its
    big spikes are tamer on a log scale. Returns (forecasts, feature importance
    of the most recent model).
    """
    X = vol_features(daily_prices, decision_dates)
    y = np.log(target.stack(future_stack=True).reindex(X.index))
    usable = X.drop(columns="etf").notna().all(axis=1)
    dates = X.index.get_level_values("date")

    preds = pd.Series(np.nan, index=X.index)
    importance = pd.Series(dtype=float)
    for year in range(first_year, decision_dates[-1].year + 1):
        train = usable & training_rows(dates, year) & y.notna()
        test = usable & (dates.year == year)
        if train.sum() < 50 or not test.any():
            continue
        model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X[train], y[train], categorical_feature=["etf"]),
            num_boost_round=LGBM_ROUNDS,
        )
        preds[test] = np.exp(model.predict(X[test]))
        importance = pd.Series(model.feature_importance("gain"), index=X.columns)
    forecasts = preds.unstack("etf").reindex(index=decision_dates, columns=daily_prices.columns)
    return forecasts, (importance / importance.sum()).sort_values()


# ------------------------------------------------------------------ evaluation
def forecast_errors(forecast: pd.DataFrame, realized: pd.DataFrame) -> dict[str, float]:
    """How good is a set of forecasts? (lower is better, except R²)

    * RMSE: typical size of the miss, in volatility points (punishes big misses).
    * MAE: average size of the miss.
    * QLIKE: the standard loss for volatility forecasts. It punishes
      *under*-forecasting risk more than over-forecasting it, which is what
      you want for risk management.
    * R² (log): share of the ups and downs in (log) volatility the forecast explains.
    """
    f = forecast.stack(future_stack=True)
    r = realized.stack(future_stack=True)
    both = pd.concat([f, r], axis=1, keys=["f", "r"]).dropna()
    err = both["f"] - both["r"]
    ratio = both["r"] ** 2 / both["f"] ** 2
    log_f, log_r = np.log(both["f"]), np.log(both["r"])
    return {
        "RMSE": float(np.sqrt((err**2).mean())),
        "MAE": float(err.abs().mean()),
        "QLIKE": float((ratio - np.log(ratio) - 1).mean()),
        "R² (log)": float(1 - ((log_r - log_f) ** 2).sum() / ((log_r - log_r.mean()) ** 2).sum()),
        "Forecasts": float(len(both)),
    }
