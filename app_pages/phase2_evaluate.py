"""Phase 2 page: is the strategy actually any good?"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_shared import daily_prices, monthly_prices, settings, style, tbill_yield
from quant import metrics as M
from quant.backtest import all_in, cash_returns_from_yield, equal_weight, simulate
from quant.strategy import month_end_prices, monthly_returns, run_backtest

STRATEGY, BH_SAME, BH_SPY = "Trend strategy", "Buy & hold: same ETFs", "Buy & hold: SPY"
PORTFOLIO_COLORS = {STRATEGY: "#2a78d6", BH_SAME: "#eb6834", BH_SPY: "#1baf7a"}

cfg = settings()
tickers, lookback, capital = cfg.tickers, cfg.lookback, cfg.capital
monthly = monthly_prices(cfg)

# Returns for every asset any portfolio might hold. SPY is always needed for the
# SPY benchmark, even if it isn't one of the selected ETFs; align it by month.
all_rets = monthly_returns(monthly)
spy = month_end_prices(daily_prices(("SPY",), cfg.refresh_token))["SPY"].pct_change()
spy.index = spy.index.to_period("M")
all_rets["SPY"] = spy.reindex(monthly.index.to_period("M")).to_numpy()

# Risk-free rate: the 13-week T-bill, as a monthly return.
rf = cash_returns_from_yield(tbill_yield(cfg.refresh_token), monthly.index)

# ---------------------------------------------------------------- header + controls
st.title("Phase 2 — Is the strategy actually any good?")
st.markdown(
    "A backtest number on its own means nothing. \\$10k → \\$36k sounds great until you learn that "
    "doing nothing clever would have made more. So we compare **three portfolios over exactly "
    "the same months**, with the same trading costs:"
)
st.markdown(
    f"- **{STRATEGY}**: the Phase 1 rule.\n"
    f"- **{BH_SAME}**: always {1/len(tickers):.0%} in each of the same ETFs, rebalanced monthly. "
    "This is the strategy *with the trend rule switched off*, so the gap between these two is "
    "exactly what the rule adds (or costs).\n"
    f"- **{BH_SPY}**: everything in US stocks, never touched. What most people would do."
)

with st.container(border=True):
    c1, c2 = st.columns([3, 2], vertical_alignment="center")
    cost_bps = c1.slider(
        "Trading cost (basis points per $ traded)", 0, 100, 10, step=5, key="cost_bps",
        help="1 basis point (bp) = 0.01%. 10 bps = you lose $1 on every $1,000 you buy or sell "
             "(commission + bid/ask spread).",
    )
    cash_interest = c2.toggle(
        "Cash earns T-bill interest", value=True, key="cash_interest",
        help="When the strategy is out of an ETF, that money sits in cash. In reality cash earns "
             "roughly the 3-month Treasury bill rate. Phase 1 assumed 0%.",
    )
cash = rf if cash_interest else None


def run_all(cost: float) -> dict:
    strat = run_backtest(monthly, lookback, capital, cash_returns=cash, cost_bps=cost)
    months = strat.weights.index
    return {
        STRATEGY: strat,
        BH_SAME: simulate(equal_weight(months, tickers), all_rets, cash, cost, capital),
        BH_SPY: simulate(all_in(months, "SPY"), all_rets, cash, cost, capital),
    }


results = run_all(cost_bps)
gross = run_all(0)
stats = pd.DataFrame({name: M.summary(r, rf) for name, r in results.items()})
stats.loc["CAGR before costs"] = [M.cagr(r.portfolio_value) for r in gross.values()]
s, b = stats[STRATEGY], stats[BH_SAME]
start, end = results[STRATEGY].portfolio_value.index[[0, -1]]

st.markdown(f"**Trend strategy vs. buy & hold of the same ETFs** · {start:%b %Y} → {end:%b %Y}")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Return per year (CAGR)", f"{s['CAGR']:.1%}", f"{(s['CAGR'] - b['CAGR']) * 100:+.1f} pp")
m2.metric("Volatility", f"{s['Volatility']:.1%}",
          f"{(s['Volatility'] - b['Volatility']) * 100:+.1f} pp", delta_color="inverse")
m3.metric("Sharpe ratio", f"{s['Sharpe ratio']:.2f}", f"{s['Sharpe ratio'] - b['Sharpe ratio']:+.2f}")
m4.metric("Max drawdown", f"{s['Max drawdown']:.1%}",
          f"{(s['Max drawdown'] - b['Max drawdown']) * 100:+.1f} pp")
st.caption("pp = percentage points. Arrows are green when the strategy is better than buy & hold.")

tabs = st.tabs(["1 · Growth", "2 · Return & volatility", "3 · Sharpe ratio",
                "4 · Drawdowns", "5 · Costs", "Scorecard"])

# ---------------------------------------------------------------- 1. growth
with tabs[0]:
    st.subheader("Portfolio value, side by side")
    st.markdown(f"Each line starts with \\${capital:,.0f} on the same date and pays the same costs.")
    log_scale = st.toggle("Log scale", value=True, key="p2_log",
                          help="Equal vertical distances = equal % moves.")
    fig = go.Figure()
    for name, r in results.items():
        v = r.portfolio_value
        fig.add_scatter(x=v.index, y=v, name=name, line=dict(color=PORTFOLIO_COLORS[name], width=2),
                        hovertemplate="$%{y:,.0f}<extra>" + name + "</extra>")
    fig.update_yaxes(type="log" if log_scale else "linear", tickprefix="$", title="Portfolio value")
    st.plotly_chart(style(fig, 440), width="stretch")
    st.caption(
        "SPY is a tough, slightly unfair bar: we only know *now* that US stocks would be the "
        "best of these markets over the next 20 years. The fair comparison for the trend rule "
        f"is **{BH_SAME}**."
    )

# ---------------------------------------------------------------- 2. return & volatility
with tabs[1]:
    st.subheader("How much did it make, and how bumpy was the ride?")
    left, right = st.columns(2)
    years = (len(results[STRATEGY].portfolio_value) - 1) / 12
    with left:
        st.markdown(
            "**CAGR (compound annual growth rate)**: the steady yearly return that would turn the "
            "start value into the end value.\n\n"
            r"$$\text{CAGR} = \left(\frac{V_\text{end}}{V_\text{start}}\right)^{1/\text{years}} - 1$$"
            f"\n\nStrategy: \\${capital:,.0f} → \\${s['Final value']:,.0f} over {years:.1f} years "
            f"= **{s['CAGR']:.1%} per year**."
        )
    with right:
        st.markdown(
            "**Volatility**: how much the monthly returns swing around their average, scaled to a "
            "year. Bigger = bumpier.\n\n"
            r"$$\sigma_\text{annual} = \text{std}(\text{monthly returns}) \times \sqrt{12}$$"
            f"\n\nStrategy: **{s['Volatility']:.1%}** vs. {b['Volatility']:.1%} for buy & hold. "
            "Roughly: in a typical year, returns land within ± one volatility of the average."
        )

    st.markdown("**Return in each calendar year**")
    yearly = pd.DataFrame({n: M.calendar_year_returns(r.portfolio_return) for n, r in results.items()})
    counts = results[STRATEGY].portfolio_return.groupby(results[STRATEGY].portfolio_return.index.year).size()
    labels = [f"{y}*" if counts[y] < 12 else str(y) for y in yearly.index]
    fig = go.Figure()
    for name in yearly.columns:
        fig.add_bar(x=labels, y=yearly[name], name=name, marker_color=PORTFOLIO_COLORS[name],
                    hovertemplate="%{y:+.1%}<extra>" + name + "</extra>")
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(type="category")
    st.plotly_chart(style(fig, 400, ".0%"), width="stretch")
    st.caption("\\* = partial year. Look at 2008 and 2022 (strategy loses far less) vs. strong years "
               "like 2009 or 2019 (strategy lags because it re-enters late).")

    st.markdown("**How often each monthly return happened**: a narrower hill = lower volatility")
    fig = go.Figure()
    for name in (STRATEGY, BH_SAME):
        fig.add_histogram(x=results[name].portfolio_return, name=name, opacity=0.6,
                          xbins=dict(size=0.01), marker_color=PORTFOLIO_COLORS[name],
                          hovertemplate="%{y} months<extra>" + name + "</extra>")
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(title="Monthly return")
    fig.update_yaxes(title="Number of months")
    st.plotly_chart(style(fig, 320, None).update_xaxes(tickformat=".0%"), width="stretch")

# ---------------------------------------------------------------- 3. sharpe
with tabs[2]:
    st.subheader("Return per unit of risk")
    st.markdown(
        "Comparing raw returns is unfair: you can almost always get more return by taking more "
        "risk (for example, borrowing to invest more). The **Sharpe ratio** asks: *for every unit "
        "of volatility I sat through, how much did I earn above simply holding cash?*\n\n"
        r"$$\text{Sharpe} = \frac{\text{avg}(R_t - R^\text{cash}_t)}{\text{std}(R_t - R^\text{cash}_t)}\times\sqrt{12}$$"
        "\n\n$R^\\text{cash}$ is the T-bill rate (the *risk-free rate*). It's subtracted even when the "
        "toggle above is off, because cash interest is what you give up by investing at all. "
        "Rough guide: below 0.3 poor, around 0.5 typical for stocks, above 1 excellent (and rare)."
    )
    fig = go.Figure()
    for name in results:
        fig.add_scatter(
            x=[stats.loc["Volatility", name]], y=[stats.loc["CAGR", name]], name=name,
            mode="markers+text", text=[name], textposition="top center",
            marker=dict(size=16, color=PORTFOLIO_COLORS[name], line=dict(width=2, color="white")),
            hovertemplate="Volatility %{x:.1%}<br>CAGR %{y:.1%}<extra>" + name + "</extra>",
        )
    fig.update_xaxes(title="Volatility (risk) →", tickformat=".0%", showgrid=True,
                     range=[0, stats.loc["Volatility"].max() * 1.25])
    fig.update_yaxes(title="Return per year (CAGR) →", tickformat=".0%",
                     range=[min(0, stats.loc["CAGR"].min() * 1.25), max(stats.loc["CAGR"].max(), 0.01) * 1.25])
    fig.update_layout(showlegend=False, hovermode="closest",
                      title="Risk vs. return: up-and-left is better")
    st.plotly_chart(style(fig, 380), width="stretch")
    rows = {}
    for name, r in results.items():
        excess = r.portfolio_return - rf.reindex(r.portfolio_return.index).fillna(0)
        rows[name] = {
            "① Avg. return above cash, per year": f"{excess.mean() * 12:.1%}",
            "② Volatility of that, per year": f"{excess.std() * np.sqrt(12):.1%}",
            "Sharpe = ① ÷ ②": f"{stats.loc['Sharpe ratio', name]:.2f}",
        }
    st.markdown("**The calculation**")
    st.dataframe(pd.DataFrame(rows).T)
    st.caption("The return above uses the simple average of monthly returns, so it differs "
               "slightly from CAGR. Averages are what the Sharpe formula uses.")

# ---------------------------------------------------------------- 4. drawdowns
with tabs[3]:
    st.subheader("How much could you have lost, and for how long?")
    st.markdown(
        "A **drawdown** is how far the portfolio sits below its previous best value:\n\n"
        r"$$\text{DD}_t = \frac{V_t}{\max_{s \le t} V_s} - 1$$"
        "\n\nThe **max drawdown** is the worst one: the most you'd have lost if you'd bought at the "
        "worst possible peak. It matters because losses are harder to undo than gains are to make:"
    )
    st.dataframe(pd.DataFrame({
        "Loss": ["−10%", "−20%", "−33%", "−50%"],
        "Gain needed to get back": ["+11%", "+25%", "+50%", "+100%"],
    }), hide_index=True, width="content")

    st.markdown("**“Underwater” chart**: how far below its previous peak each portfolio was (0% = at a new high)")
    fig = go.Figure()
    for name, r in results.items():
        dd = M.drawdown(r.portfolio_value)
        fig.add_scatter(x=dd.index, y=dd, name=name, line=dict(color=PORTFOLIO_COLORS[name], width=2),
                        fill="tozeroy" if name == STRATEGY else None,
                        hovertemplate="%{y:.1%}<extra>" + name + "</extra>")
    fig.update_yaxes(title="Below previous peak")
    st.plotly_chart(style(fig, 380, ".0%"), width="stretch")

    pick = st.segmented_control("Worst drawdowns of", list(results), default=STRATEGY, key="dd_pick")
    episodes = M.drawdown_episodes(results[pick or STRATEGY].portfolio_value)
    if episodes.empty:
        st.info("This portfolio never fell below a previous peak.")
        st.stop()
    for col in ("Peak", "Bottom", "Recovered"):
        episodes[col] = episodes[col].map(lambda d: "not yet" if pd.isna(d) else f"{d:%b %Y}")
    st.dataframe(
        episodes.style.format({"Depth": "{:.1%}", "Months to recover": "{:.0f}"}, na_rep="—"),
        hide_index=True,
    )

# ---------------------------------------------------------------- 5. costs
with tabs[4]:
    st.subheader("What trading actually costs")
    st.markdown(
        "Every trade loses a little: a broker commission and the **bid/ask spread** (you buy "
        "slightly above and sell slightly below the 'real' price). We model this as a fixed "
        f"**{cost_bps} bps per dollar traded**.\n\n"
        "**Turnover** = how much of the portfolio was traded. Moving a 20% slot into cash is 20% "
        "turnover. Buy & hold still trades a little: once a month it tops up the ETFs that fell "
        "and trims the ones that rose, to get back to equal weights.\n\n"
        r"$$\text{cost}_t = \text{turnover}_t \times \text{cost per \$}$$"
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategy turnover / year", f"{s['Turnover / year']:.0%}")
    c2.metric("Buy & hold turnover / year", f"{b['Turnover / year']:.0%}")
    c3.metric("Strategy costs paid", f"${s['Costs paid']:,.0f}")
    c4.metric("CAGR lost to costs", f"{(s['CAGR before costs'] - s['CAGR']) * 100:.2f} pp")

    t = results[STRATEGY].turnover
    fig = go.Figure(go.Bar(x=t.index, y=t, marker_color=PORTFOLIO_COLORS[STRATEGY],
                           hovertemplate="%{x|%b %Y}: %{y:.0%} traded<extra></extra>"))
    fig.update_layout(title="Strategy turnover each month: spikes = signals flipping (first bar = initial purchase)")
    st.plotly_chart(style(fig, 280, ".0%"), width="stretch")

    st.markdown("**What if trading were more expensive?** Return per year at different cost levels")
    levels = list(range(0, 101, 10))
    sweep = pd.DataFrame(
        {lvl: {n: M.cagr(r.portfolio_value) for n, r in run_all(lvl).items()} for lvl in levels}
    ).T
    fig = go.Figure()
    for name in sweep.columns:
        fig.add_scatter(x=sweep.index, y=sweep[name], name=name, mode="lines+markers",
                        line=dict(color=PORTFOLIO_COLORS[name], width=2), marker=dict(size=8),
                        hovertemplate="%{y:.2%}<extra>" + name + "</extra>")
    fig.add_vline(x=cost_bps, line=dict(color="#a3a29c", dash="dot"),
                  annotation_text="current setting", annotation_position="bottom right")
    fig.update_xaxes(title="Trading cost (bps per $ traded)")
    st.plotly_chart(style(fig, 340, ".1%"), width="stretch")
    st.caption(
        "For big ETFs like SPY the real spread is ~1 bp and many brokers charge no commission, so "
        "10 bps is on the cautious side. Not modeled: **taxes**. In a taxable account every sale "
        "can trigger capital-gains tax, which hurts a strategy that trades more."
    )

# ---------------------------------------------------------------- scorecard
with tabs[5]:
    st.subheader("Everything in one table")
    money = {"Final value", "Costs paid"}
    ratio = {"Sharpe ratio"}
    order = ["Final value", "Total return", "CAGR", "CAGR before costs", "Volatility",
             "Sharpe ratio", "Max drawdown", "Worst month", "Best month", "% months positive",
             "Avg. invested", "Turnover / year", "Costs paid"]
    shown = stats.loc[order].copy().astype(object)
    for row in order:
        fmt = "${:,.0f}" if row in money else "{:.2f}" if row in ratio else "{:.1%}"
        shown.loc[row] = [fmt.format(v) for v in stats.loc[row]]
    st.dataframe(shown, height=490)
    st.markdown(
        "**How to read it.** There is no single 'best' number. Higher CAGR and Sharpe are better; "
        "volatility and drawdown closer to 0 are better. The question is which trade-off you'd "
        "actually live with: more return, or a smoother ride you're more likely to stick with."
    )
