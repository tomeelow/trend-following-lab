"""Phase 1 page: build the strategy, one step per tab."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_shared import (
    CASH_COLOR, COLORS, IN_COLOR, daily_prices, monthly_prices, newest_first, pct, settings, style,
)
from quant.strategy import (
    current_positions,
    monthly_returns,
    run_backtest,
    signals,
    trailing_returns,
)

cfg = settings()
tickers, lookback, capital = cfg.tickers, cfg.lookback, cfg.capital
daily = daily_prices(tuple(tickers), cfg.refresh_token)
monthly = monthly_prices(cfg)
rets = monthly_returns(monthly)
trend = trailing_returns(monthly, lookback)
sig = signals(trend)
# Frictionless version: no trading costs, cash earns 0%. Phase 2 adds realism.
result = run_backtest(monthly, lookback_months=lookback, initial_capital=capital)

# ---------------------------------------------------------------- header
st.title("Phase 1 — Simple trend following")
st.markdown(
    f"**Rule:** at each month end, for each ETF: if its last **{lookback}-month return "
    "is positive, hold it**, otherwise hold cash. Each ETF owns an equal slot "
    f"(1/{len(tickers)} = {1/len(tickers):.0%} of the portfolio). Rebalance monthly."
)

final = result.portfolio_value.iloc[-1]
years = (result.portfolio_value.index[-1] - result.portfolio_value.index[0]).days / 365.25
c1, c2, c3, c4 = st.columns(4)
c1.metric("Backtest period", f"{years:.1f} years",
          help=f"{result.portfolio_value.index[0]:%b %Y} → {result.portfolio_value.index[-1]:%b %Y}")
c2.metric("Start value", f"${capital:,.0f}")
c3.metric("End value", f"${final:,.0f}")
c4.metric("Total return", f"{final / capital - 1:+.0%}", help=f"Over {years:.1f} years, before any costs.")

tabs = st.tabs([
    "1 · Prices", "2 · Monthly returns", "3 · Trend signal",
    "4 · Positions", "5 · Backtest", "Today's signal",
])

# ---------------------------------------------------------------- 1. prices
with tabs[0]:
    st.subheader("Step 1 — Daily adjusted prices")
    st.markdown(
        "Downloaded from Yahoo Finance. *Adjusted* prices include dividends, so their % change "
        "is the total return. Below, every ETF is rescaled to start at 100 so you can compare "
        f"them. The data starts **{daily.index[0]:%d %b %Y}**: the first date on which all "
        "selected ETFs exist."
    )
    log_scale = st.toggle("Log scale", value=True, key="log_prices",
                          help="On a log scale, equal vertical distances = equal % moves.")
    norm = daily / daily.iloc[0] * 100
    fig = go.Figure()
    for t in tickers:
        fig.add_scatter(x=norm.index, y=norm[t], name=t, line=dict(color=COLORS[t], width=2))
    fig.update_yaxes(type="log" if log_scale else "linear", title="Value (start = 100)")
    st.plotly_chart(style(fig, 420), width="stretch")
    with st.expander("Raw data (last 10 days)"):
        st.dataframe(newest_first(daily.tail(10)).style.format("{:.2f}"))

# ---------------------------------------------------------------- 2. monthly returns
with tabs[1]:
    st.subheader("Step 2 — Month-end prices → monthly returns")
    st.markdown(
        "We only make decisions once a month, so we keep just the **last trading day's price** of "
        "each month, then compute\n\n"
        r"$$r_t = \frac{P_t}{P_{t-1}} - 1$$"
        "\n\nRow *t* = what you'd have earned **during** month *t* (end of last month → end of this month). "
        "An unfinished current month is dropped because its 'month-end' price doesn't exist yet."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Month-end prices**")
        st.dataframe(newest_first(monthly).style.format("{:.2f}"), height=420)
    with right:
        st.markdown("**Monthly returns**")
        st.dataframe(newest_first(rets).style.format("{:+.2%}", na_rep="—"), height=420)

    pick = st.selectbox("Monthly returns of", tickers, key="ret_pick")
    r = rets[pick].dropna()
    fig = go.Figure(go.Bar(
        x=r.index, y=r, marker_color=[COLORS[pick] if v >= 0 else CASH_COLOR for v in r],
        hovertemplate="%{x|%b %Y}: %{y:+.2%}<extra></extra>",
    ))
    st.plotly_chart(style(fig, 300, ".0%"), width="stretch")

# ---------------------------------------------------------------- 3. trend signal
with tabs[2]:
    st.subheader(f"Step 3 — {lookback}-month return → signal")
    st.markdown(
        f"At each month end we look back {lookback} months:\n\n"
        rf"$$\text{{trend}}_t = \frac{{P_t}}{{P_{{t-{lookback}}}}} - 1 \qquad "
        r"\text{signal}_t = \begin{cases}1 & \text{trend}_t > 0\\ 0 & \text{otherwise}\end{cases}$$"
        f"\n\nThe first {lookback} months have no signal (not enough history)."
    )
    pick = st.selectbox("ETF", tickers, key="sig_pick")
    t_ser = trend[pick].dropna()
    held = sig[pick].shift(1)  # what was actually held during each month

    fig = go.Figure()
    fig.add_scatter(x=t_ser.index, y=t_ser, name=f"{lookback}-month return",
                    line=dict(color=COLORS[pick], width=2),
                    hovertemplate="%{y:+.1%}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=CASH_COLOR, width=1, dash="dot"))
    fig.update_layout(title=f"{pick}: trailing {lookback}-month return (above 0 → invested next month)")
    st.plotly_chart(style(fig, 320, ".0%"), width="stretch")

    # Price with shaded "invested" months
    p = monthly[pick]
    fig = go.Figure()
    in_runs = (held == 1)
    start = None
    for i, (date, on) in enumerate(in_runs.items()):
        if on and start is None:
            start = monthly.index[i - 1]  # holding month t means from end of t-1 to end of t
        if (not on or i == len(in_runs) - 1) and start is not None:
            end = date if on else monthly.index[i - 1]
            fig.add_vrect(x0=start, x1=end, fillcolor=IN_COLOR, opacity=0.12, line_width=0)
            start = None
    fig.add_scatter(x=p.index, y=p, name=f"{pick} month-end price",
                    line=dict(color=COLORS[pick], width=2),
                    hovertemplate="$%{y:.2f}<extra></extra>")
    fig.update_yaxes(type="log", title="Price ($, log)")
    fig.update_layout(title=f"{pick} price — shaded = months the strategy held {pick}")
    st.plotly_chart(style(fig, 320), width="stretch")

    st.markdown("**All signals at once** (blue = invested, gray = cash)")
    s = sig.dropna(how="all")
    fig = go.Figure(go.Heatmap(
        x=s.index, y=s.columns, z=s.T.values,
        colorscale=[[0, "#d9d8d3"], [1, IN_COLOR]], zmin=0, zmax=1, showscale=False,
        xgap=1, ygap=3,
        hovertemplate="%{y} · %{x|%b %Y}: signal %{z}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(style(fig, 60 + 45 * len(tickers)), width="stretch")

# ---------------------------------------------------------------- 4. positions
with tabs[3]:
    st.subheader("Step 4 — Signals → portfolio weights")
    st.markdown(
        f"Each ETF owns a {1/len(tickers):.0%} slot: weight = signal ÷ {len(tickers)}. "
        "Whatever isn't invested sits in **cash** (earning 0% here, for simplicity).\n\n"
        "**The crucial detail — the one-month shift.** The signal computed at the *end* of "
        "month *t* can only be acted on from then on, so it sets the weights for month *t+1*:\n\n"
        r"$$w_{t+1} = \frac{\text{signal}_t}{N}$$"
        "\n\nIf we skipped the shift, we'd be deciding whether to own an ETF in March using "
        "March's own price move — information we couldn't have had. That's *lookahead bias*, and "
        "it makes backtests look far better than reality."
    )
    w = result.weights.copy()
    w["Cash"] = result.cash_weight
    fig = go.Figure()
    for col in w.columns:
        color = CASH_COLOR if col == "Cash" else COLORS[col]
        fig.add_scatter(x=w.index, y=w[col], name=col, stackgroup="w", mode="lines",
                        line=dict(width=0.5, color=color, shape="hv"), fillcolor=color,
                        hovertemplate="%{y:.0%}<extra>" + col + "</extra>")
    fig.update_yaxes(range=[0, 1], title="Weight held during month")
    st.plotly_chart(style(fig, 400, ".0%"), width="stretch")

    st.markdown("**Side by side: signal at month end vs. weight held the following month**")
    pick = st.selectbox("ETF", tickers, key="w_pick")
    side = pd.DataFrame({
        f"{lookback}m return at month end": trend[pick],
        "signal at month end": sig[pick],
        "weight held DURING this month": result.weights[pick],
    }).loc[result.weights.index[0]:]
    side = newest_first(side)
    st.dataframe(side.style.format({
        side.columns[0]: "{:+.1%}", side.columns[1]: "{:.0f}", side.columns[2]: "{:.0%}",
    }, na_rep="—"), height=360)

# ---------------------------------------------------------------- 5. backtest
with tabs[4]:
    st.subheader("Step 5 — The backtest: weights × returns, compounded")
    st.markdown(
        "Each month, the portfolio's return is the weighted sum of what it held:\n\n"
        r"$$R^{\text{port}}_t = \sum_i w_{i,t}\, r_{i,t} + w_{\text{cash},t} \cdot 0$$"
        "\n\nThen we compound: "
        r"$V_t = V_{t-1}\,(1 + R^{\text{port}}_t)$."
        " No trading costs or interest on cash here — the **Phase 2** page adds both "
        "and compares this against simply buying and holding."
    )
    log_scale = st.toggle("Log scale", value=False, key="log_bt")
    v = result.portfolio_value
    fig = go.Figure(go.Scatter(
        x=v.index, y=v, name="Strategy", line=dict(color=IN_COLOR, width=2),
        hovertemplate="%{x|%b %Y}: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_yaxes(type="log" if log_scale else "linear", title="Portfolio value ($)", tickprefix="$")
    st.plotly_chart(style(fig, 420), width="stretch")

    st.markdown("**The ledger — every month of the backtest**")
    st.caption("Contribution = weight × return. Contributions add up to the portfolio return.")
    ledger = pd.concat(
        {"Weight": result.weights, "ETF return": result.asset_returns,
         "Contribution": result.contributions},
        axis=1,
    )
    ledger[("Portfolio", "Cash weight")] = result.cash_weight
    ledger[("Portfolio", "Return")] = result.portfolio_return
    ledger[("Portfolio", "Value ($)")] = result.portfolio_value
    fmt = {c: "{:.0%}" for c in ledger.columns if c[0] == "Weight" or c[1] == "Cash weight"}
    fmt |= {c: "{:+.2%}" for c in ledger.columns if c[0] in ("ETF return", "Contribution") or c[1] == "Return"}
    fmt[("Portfolio", "Value ($)")] = "${:,.0f}"
    st.dataframe(newest_first(ledger).style.format(fmt), height=460)

# ---------------------------------------------------------------- today
with tabs[5]:
    last = monthly.index[-1]
    st.subheader(f"What to hold now (signal from {last:%d %b %Y} close)")
    st.markdown(
        "Based on the last completed month end. These weights apply until the next month end."
    )
    pos = current_positions(monthly, lookback)
    cols = st.columns(len(tickers) + 1)
    for col, t in zip(cols, tickers):
        on = pos.loc[t, "signal"] == 1
        col.metric(t, "Invested" if on else "Cash", pct(pos.loc[t, "trailing_return"]),
                   help=f"{lookback}-month return: {pct(pos.loc[t, 'trailing_return'])}")
    cols[-1].metric("Cash", f"{1 - pos['weight'].sum():.0%}")
