"""Phase 4 page: forecast volatility and use it to size positions."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_shared import CASH_COLOR, COLORS, COST_BPS, FIRST_FORECAST_YEAR, market, settings, style, vol_forecasts
from quant import metrics as M
from quant.backtest import simulate
from quant.sizing import vol_target_weights
from quant.strategy import signals, target_weights, trailing_returns
from quant.volatility import FEATURES, TRADING_DAYS, daily_returns, forecast_errors

MODEL_COLORS = {
    "Moving avg. (1 month)": "#2a78d6",
    "Moving avg. (3 months)": "#eb6834",
    "GARCH(1,1)": "#1baf7a",
    "LightGBM": "#eda100",
}
REALIZED_COLOR = "#8a8984"
EQUAL, PERFECT = "Equal slots (Phase 1 rule)", "Perfect foresight (impossible)"
ZOOMS = {"All": None, "2008 crisis": ("2008-06-01", "2009-12-31"),
         "2020 crash": ("2019-10-01", "2020-12-31"), "2022": ("2021-10-01", "2023-06-30")}

cfg = settings()
tickers, lookback, capital = cfg.tickers, cfg.lookback, cfg.capital
mkt = market(tuple(tickers), cfg.refresh_token)
vf = vol_forecasts(tuple(tickers), cfg.refresh_token)
realized, forecasts = vf["realized"], vf["forecasts"]

# Months where every model has an out-of-sample forecast and we know the answer.
dates = mkt.monthly.index
in_eval = (dates.year >= FIRST_FORECAST_YEAR)
for f in forecasts.values():
    in_eval &= f.notna().all(axis=1).to_numpy()
in_eval &= realized.notna().all(axis=1).to_numpy()
eval_dates = dates[in_eval]

# ---------------------------------------------------------------- header
st.title("Phase 4 — Forecasting volatility and sizing positions")
st.markdown(
    "**Volatility** measures how violently a price moves: an ETF with 20% annual volatility "
    "typically ends a year within ±20% of its average. Phases 1–3 gave every ETF the same 20% "
    "slot, but emerging-market stocks swing far more than Treasuries, and *everything* swings more "
    "in a crisis. If we could **forecast** next month's volatility, we could hold less of whatever "
    "is about to be wild and more of whatever is calm.\n\n"
    "This page builds three forecasting approaches, scores them fairly, then uses them to size "
    "positions. Every forecast is made at a month end using **only data up to that day** "
    f"(walk-forward, like Phase 3), and the realistic settings from Phase 3 still apply."
)

with st.container(border=True):
    c1, c2 = st.columns([3, 2])
    etf = c1.segmented_control("ETF to chart (tabs 1–4)", tickers, default=tickers[0], key="p4_etf") or tickers[0]
    zoom = c2.segmented_control("Zoom", list(ZOOMS), default="All", key="p4_zoom") or "All"


def zoomed(fig: go.Figure, default_start: str | None = None) -> go.Figure:
    window = ZOOMS[zoom]
    if window:
        fig.update_xaxes(range=list(window))
    elif default_start:
        fig.update_xaxes(range=[default_start, str(dates[-1].date())])
    return fig


def forecast_chart(models: list[str], height: int = 380) -> None:
    """Realized next-month volatility of `etf` vs. the chosen models' forecasts."""
    fig = go.Figure()
    r = realized[etf].dropna()
    fig.add_scatter(x=r.index, y=r, name="What actually happened", mode="lines+markers",
                    line=dict(color=REALIZED_COLOR, width=1), marker=dict(size=4),
                    hovertemplate="%{y:.1%}<extra>actual</extra>")
    for name in models:
        f = forecasts[name][etf].dropna()
        f = f[f.index.year >= FIRST_FORECAST_YEAR]
        fig.add_scatter(x=f.index, y=f, name=name, line=dict(color=MODEL_COLORS[name], width=2),
                        hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
    fig.update_yaxes(title="Next month's volatility (annualized)", rangemode="tozero")
    st.plotly_chart(zoomed(style(fig, height, ".0%"), f"{FIRST_FORECAST_YEAR}-01-01"), width="stretch")
    st.caption("Each point is dated at the month end when the forecast was made; it predicts the "
               "volatility of the *following* month.")


tabs = st.tabs(["1 · Volatility clusters", "2 · Moving average", "3 · GARCH", "4 · LightGBM",
                "5 · Which forecast is best?", "6 · Position sizing"])

# ---------------------------------------------------------------- 1. clusters
with tabs[0]:
    st.subheader("Why volatility can be forecast at all")
    st.markdown(
        "You can't predict whether tomorrow will be up or down. But you *can* predict roughly how "
        "**big** the move will be, because volatility **clusters**: wild days come in bunches, and "
        "calm stretches last. Look at the daily returns: the big spikes aren't spread evenly, they "
        "crowd together (2008, 2020, 2022)."
    )
    rets = daily_returns(mkt.daily)[etf]
    fig = go.Figure(go.Scatter(x=rets.index, y=rets, mode="lines", line=dict(color=COLORS.get(etf, "#2a78d6"), width=1),
                               hovertemplate="%{x|%d %b %Y}: %{y:+.2%}<extra></extra>"))
    fig.update_yaxes(title=f"{etf} daily return")
    st.plotly_chart(zoomed(style(fig, 280, ".0%")), width="stretch")

    by_month = daily_returns(mkt.daily).groupby(daily_returns(mkt.daily).index.to_period("M")).std() * np.sqrt(TRADING_DAYS)
    persistence = {t: by_month[t].corr(by_month[t].shift(1)) for t in tickers}
    v = by_month[etf]
    fig = go.Figure(go.Scatter(x=v.index.to_timestamp("M"), y=v, mode="lines",
                               line=dict(color=COLORS.get(etf, "#2a78d6"), width=2),
                               hovertemplate="%{x|%b %Y}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(title="Volatility in that month", rangemode="tozero")
    st.plotly_chart(zoomed(style(fig, 260, ".0%")), width="stretch")
    st.markdown(
        "**How much does one month's volatility tell you about the next?** Correlation between each "
        "month's volatility and the following month's (1 = perfectly predictable, 0 = no link):"
    )
    st.dataframe(pd.DataFrame([persistence], index=["Month-to-month correlation"]).style.format("{:.2f}"))
    st.markdown("Those are high numbers. That predictability is what every model below exploits.")

# ---------------------------------------------------------------- 2. moving average
with tabs[1]:
    st.subheader("Model 1: moving average, \"next month will look like the recent past\"")
    st.markdown(
        "The simplest forecast: measure the volatility of the last *N* trading days and assume it "
        "continues.\n\n"
        r"$$\hat\sigma_{\text{next month}} = \text{std}(\text{last } N \text{ daily returns}) \times \sqrt{252}$$"
        "\n\n(×√252 turns a daily number into a yearly one: there are ~252 trading days a year.) "
        "The only choice is the window **N**, and it's a trade-off:\n"
        "- **1 month (21 days)**: reacts fast when a crisis starts, but jumps around and overreacts.\n"
        "- **3 months (63 days)**: smoother, but slow. It keeps saying 'calm' after a storm has "
        "started, and 'stormy' long after it's over."
    )
    forecast_chart(["Moving avg. (1 month)", "Moving avg. (3 months)"])
    st.markdown("Try the **2008 crisis** zoom: watch both lines lag behind the spike, then stay high "
                "while the actual volatility falls.")

# ---------------------------------------------------------------- 3. GARCH
with tabs[2]:
    st.subheader("Model 2: GARCH(1,1), shocks fade back to normal")
    st.markdown(
        "GARCH is the classic statistical volatility model (its predecessor, ARCH, earned Robert Engle "
        "the 2003 Nobel prize in economics). Every day it "
        "updates its estimate of variance (volatility squared) from three ingredients:\n\n"
        r"$$\sigma^2_{t} = \underbrace{\omega}_{\text{long-run floor}} + \underbrace{\alpha\,\varepsilon^2_{t-1}}_{\text{yesterday's surprise}} + \underbrace{\beta\,\sigma^2_{t-1}}_{\text{yesterday's estimate}}$$"
        "\n\n- A big move yesterday (large surprise $\\varepsilon$) pushes volatility up **immediately** "
        "(weight α).\n"
        "- Most of yesterday's level carries over (weight β), so shocks **fade gradually**.\n"
        "- Over time it drifts back to a **long-run average**. That's the big advantage over a "
        "moving average: GARCH 'knows' that crises calm down.\n\n"
        "To forecast a month ahead, the formula is simply run forward 21 days and the daily "
        "variances added up. The parameters ω, α, β are re-estimated every January using only "
        "the data before that January."
    )
    p = vf["garch_params"].loc[tickers]
    persistence = p["alpha[1]"] + p["beta[1]"]
    long_run = np.sqrt(p["omega"] / (1 - persistence) * TRADING_DAYS) / 100
    half_life = np.log(0.5) / np.log(persistence)
    table = pd.DataFrame({
        "α (reaction to surprises)": p["alpha[1]"],
        "β (memory)": p["beta[1]"],
        "α + β (persistence)": persistence,
        "Long-run volatility": long_run,
        "Shock half-life (trading days)": half_life,
    })
    st.markdown(f"**Fitted parameters** (latest annual fit, estimated on data up to the end of {dates[-1].year - 1})")
    st.dataframe(table.style.format({"α (reaction to surprises)": "{:.3f}", "β (memory)": "{:.3f}",
                                     "α + β (persistence)": "{:.3f}", "Long-run volatility": "{:.1%}",
                                     "Shock half-life (trading days)": "{:.0f}"}))
    st.caption("Half-life: after a shock, how many trading days until half of the extra volatility is "
               "gone. Stocks (high α) react sharply; bonds and gold (low α, high β) move slowly.")
    forecast_chart(["GARCH(1,1)", "Moving avg. (3 months)"])

# ---------------------------------------------------------------- 4. LightGBM
with tabs[3]:
    st.subheader("Model 3: LightGBM, let a machine-learning model find the pattern")
    st.markdown(
        "LightGBM builds hundreds of small **decision trees** (\"if last month's vol is above 25% "
        "and the price is 10% below its high, then…\"), each one correcting the mistakes of the "
        "previous ones. It isn't told how volatility works; it learns from examples.\n\n"
        "**One example** = one ETF at one month end. The **inputs** (features) describe the recent "
        "past; the **answer** (target) is the volatility of the following month. All ETFs' "
        "examples are pooled, so the model learns from 5× more data."
    )
    st.dataframe(pd.DataFrame({"Feature": list(FEATURES), "Meaning": list(FEATURES.values())}),
                 hide_index=True)
    st.markdown(
        "**Training without cheating.** Each January the model is retrained on all examples whose "
        "answer is *already known*. Subtle point: the example from the end of November has, as its "
        "answer, December's volatility, which is only known on Dec 31. So in January 2015 the model "
        "may use examples up to **November 2014**, not December 2014 (whose answer, January 2015, "
        "hasn't happened yet). Getting this wrong is called **label leakage**, and there's a test "
        "for it in Phase 3."
    )
    imp = vf["importance"]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=MODEL_COLORS["LightGBM"],
                           hovertemplate="%{y}: %{x:.0%} of the model's improvement<extra></extra>"))
    fig.update_xaxes(tickformat=".0%", title="Share of total importance (latest model)")
    st.markdown("**What the model relies on** (feature importance)")
    st.plotly_chart(style(fig, 380), width="stretch")
    st.caption("Mostly recent volatility, especially the exponentially weighted version. The model "
               "largely rediscovered 'recent volatility predicts future volatility'.")
    forecast_chart(["LightGBM", "GARCH(1,1)"])

# ---------------------------------------------------------------- 5. evaluation
with tabs[4]:
    st.subheader("Scoring the forecasts fairly")
    st.markdown(
        f"Every model is scored on **exactly the same** {len(eval_dates)} month-ends × "
        f"{len(tickers)} ETFs ({eval_dates[0]:%b %Y} – {eval_dates[-1]:%b %Y}), all out-of-sample: "
        "no model had seen the answer when it made the forecast.\n\n"
        "- **RMSE**: typical size of a miss in volatility points (squares the errors, so big misses "
        "count a lot).\n"
        "- **MAE**: average size of a miss.\n"
        "- **QLIKE**: the standard score for volatility forecasts. It punishes **under**-estimating "
        "risk harder than over-estimating it, which is what you want when risk is what you're "
        "managing.\n"
        "- **R² (log)**: share of the month-to-month ups and downs the forecast explains.\n\n"
        "Lower is better for the first three, higher for R². The simplest model is the **baseline**: "
        "a fancy model is only worth it if it beats the baseline."
    )
    scores = pd.DataFrame({
        name: forecast_errors(f.loc[eval_dates, tickers], realized.loc[eval_dates, tickers])
        for name, f in forecasts.items()
    }).T.drop(columns="Forecasts")
    st.dataframe(
        scores.style.format({"RMSE": "{:.2%}", "MAE": "{:.2%}", "QLIKE": "{:.3f}", "R² (log)": "{:.2f}"})
        .highlight_min(subset=["RMSE", "MAE", "QLIKE"], props="font-weight: bold; color: #1baf7a")
        .highlight_max(subset=["R² (log)"], props="font-weight: bold; color: #1baf7a"),
    )
    st.caption("Best in each column is highlighted.")

    fig = go.Figure(go.Bar(x=scores.index, y=scores["QLIKE"], marker_color=[MODEL_COLORS[m] for m in scores.index],
                           hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_yaxes(title="QLIKE (lower = better)")
    st.plotly_chart(style(fig, 300), width="stretch")

    per_etf = pd.DataFrame({
        name: {t: forecast_errors(f.loc[eval_dates, [t]], realized.loc[eval_dates, [t]])["QLIKE"] for t in tickers}
        for name, f in forecasts.items()
    })
    st.markdown("**QLIKE per ETF** (does the winner win everywhere?)")
    st.dataframe(per_etf.style.format("{:.3f}").highlight_min(axis=1, props="font-weight: bold; color: #1baf7a"))

    best = scores["QLIKE"].idxmin()
    st.markdown(
        f"**Lesson:** the best forecaster here is **{best}**. Machine learning is not automatically "
        "better: with only a few thousand examples, a model with the right structure built in "
        "(GARCH knows volatility reverts to normal) can beat a flexible one that has to learn "
        "everything from scratch. Always score fancy models against a simple baseline, on "
        "out-of-sample data."
    )

# ---------------------------------------------------------------- 6. sizing
with tabs[5]:
    st.subheader("Using forecasts to size positions (volatility targeting)")
    st.markdown(
        "Instead of giving every 'on' ETF the same amount of **money**, give each the same amount of "
        "**risk**:\n\n"
        r"$$\text{weight} = \text{signal} \times \frac{1}{N} \times \frac{\text{target volatility}}{\text{forecast volatility}}$$"
        "\n\nIf an ETF is forecast to be twice as wild as the target, it gets half a slot. If it's "
        "calm, it gets more (up to 2 slots). If the total adds up to more than the maximum "
        "exposure, everything is scaled down together. The trend signal still decides in/out, "
        "volatility only decides **how much**."
    )
    with st.container(border=True):
        c1, c2 = st.columns(2)
        target = c1.slider("Target volatility per slot (%)", 5, 25, 15, key="p4_target",
                           help="Each slot is sized so its forecast volatility equals this.") / 100
        max_gross = c2.select_slider("Maximum total exposure", [1.0, 1.25, 1.5, 2.0], value=1.0, key="p4_gross",
                                     format_func=lambda x: f"{x:.0%}",
                                     help="100% = never borrow. Above 100% means borrowing at the T-bill rate "
                                          "to hold more than you own (leverage).")

    sig = signals(trailing_returns(mkt.monthly, lookback))
    first_held = dates[dates.year >= FIRST_FORECAST_YEAR][1]  # held after the first forecast month end
    held = mkt.returns.dropna().index
    held = held[held >= first_held]

    def run(weights: pd.DataFrame):
        w = weights.loc[held]
        assert w.notna().all().all(), "every held month needs a forecast"
        return simulate(w, mkt.returns, mkt.rf, COST_BPS, capital)

    sized = {EQUAL: run(target_weights(sig))}
    for name in ["Moving avg. (3 months)", "GARCH(1,1)", "LightGBM"]:
        sized[f"Sized by {name}"] = run(vol_target_weights(sig, forecasts[name], target, max_gross=max_gross))
    sized[PERFECT] = run(vol_target_weights(sig, realized, target, max_gross=max_gross))
    colors = {EQUAL: "#6250d6", PERFECT: "#e34948",
              **{f"Sized by {n}": MODEL_COLORS[n] for n in ["Moving avg. (3 months)", "GARCH(1,1)", "LightGBM"]}}

    fig = go.Figure()
    for name, r in sized.items():
        v = r.portfolio_value
        fig.add_scatter(x=v.index, y=v, name=name,
                        line=dict(color=colors[name], width=2, dash="dash" if name == PERFECT else "solid"),
                        hovertemplate="$%{y:,.0f}<extra>" + name + "</extra>")
    fig.update_yaxes(type="log", tickprefix="$", title="Portfolio value (log)")
    st.plotly_chart(style(fig, 420), width="stretch")

    table = pd.DataFrame({n: M.summary(r, mkt.rf) for n, r in sized.items()}).T
    table = table[["CAGR", "Volatility", "Sharpe ratio", "Max drawdown", "Avg. invested", "Turnover / year"]]
    st.dataframe(table.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe ratio": "{:.2f}",
                                     "Max drawdown": "{:.1%}", "Avg. invested": "{:.0%}",
                                     "Turnover / year": "{:.0%}"}))

    real_models = [n for n in sized if n not in (EQUAL, PERFECT)]
    best_real = max(real_models, key=lambda n: table.loc[n, "Sharpe ratio"])
    st.markdown(
        f"**Reading it.** Compare Sharpe ratios (return per unit of risk). Equal slots: "
        f"**{table.loc[EQUAL, 'Sharpe ratio']:.2f}**; best forecast-based sizing ({best_real.removeprefix('Sized by ')}): "
        f"**{table.loc[best_real, 'Sharpe ratio']:.2f}**; perfect foresight: "
        f"**{table.loc[PERFECT, 'Sharpe ratio']:.2f}**. The perfect-foresight line uses next month's "
        "*actual* volatility, which nobody can know. It's there as a ceiling: it shows how much "
        "better forecasts *could* help, and how far real forecasts are from that.\n\n"
        "Sizing shifts money away from whatever is forecast to be wild, so the sized portfolios run "
        "with **lower volatility** for similar return. Lower the target and you hold less overall "
        "(less risk, less return); raise it or allow more than 100% exposure and you borrow to hold "
        "more. Real borrowing costs more than the T-bill rate assumed here."
    )

    st.markdown("**Rolling 12-month volatility of each portfolio**: sizing should make risk steadier")
    fig = go.Figure()
    for name, r in sized.items():
        rv = r.portfolio_return.rolling(12).std() * np.sqrt(12)
        fig.add_scatter(x=rv.index, y=rv, name=name,
                        line=dict(color=colors[name], width=2, dash="dash" if name == PERFECT else "solid"),
                        hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
    st.plotly_chart(style(fig, 320, ".0%"), width="stretch")

    pick = st.segmented_control("Positions held by", list(sized), default="Sized by GARCH(1,1)",
                                key="p4_weights_pick") or EQUAL
    w = sized[pick].weights.copy()
    w["Cash"] = sized[pick].cash_weight.clip(lower=0)
    fig = go.Figure()
    for col in w.columns:
        color = CASH_COLOR if col == "Cash" else COLORS.get(col, "#2a78d6")
        fig.add_scatter(x=w.index, y=w[col], name=col, stackgroup="w", mode="lines",
                        line=dict(width=0.5, color=color, shape="hv"), fillcolor=color,
                        hovertemplate="%{y:.0%}<extra>" + col + "</extra>")
    fig.update_yaxes(title="Weight held during month")
    st.plotly_chart(style(fig, 360, ".0%"), width="stretch")
    st.caption("Compare with Phase 1's equal slots: in 2008, 2020 and 2022 the sized positions shrink "
               "as forecast volatility jumps, before the trend signal has even switched them off.")
