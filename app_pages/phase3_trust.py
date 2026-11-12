"""Phase 3 page: can we trust the backtest?"""

import ast
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app_shared import COST_BPS, LAG_DAYS, market, settings, style
from quant import metrics as M
from quant.backtest import equal_weight, simulate
from quant.strategy import run_backtest
from quant.validation import (
    candidate_returns,
    cheating_backtest,
    sharpe_by_column,
    split_half_sharpes,
    walk_forward,
)

ROOT = Path(__file__).resolve().parent.parent
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#a3a29c"
LOOKBACKS = list(range(1, 25))
FIRST_WF_YEAR = 2010

cfg = settings()
tickers, lookback, capital = cfg.tickers, cfg.lookback, cfg.capital
mkt = market(tuple(tickers), cfg.refresh_token)
monthly, rf, rets = mkt.monthly, mkt.rf, mkt.returns
realistic = dict(cash_returns=rf, cost_bps=COST_BPS)


def stats_row(returns: pd.Series) -> dict:
    s = M.stats_from_returns(returns, rf)
    return {"CAGR": s["CAGR"], "Volatility": s["Volatility"], "Sharpe ratio": s["Sharpe ratio"],
            "Max drawdown": s["Max drawdown"]}


def show_stats(rows: dict[str, dict]) -> None:
    df = pd.DataFrame(rows).T
    st.dataframe(df.style.format({"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe ratio": "{:.2f}",
                                  "Max drawdown": "{:.1%}"}))


def growth_chart(series: dict[str, tuple[pd.Series, str, str]], height: int = 400) -> None:
    """series: name -> (monthly returns, color, dash)"""
    fig = go.Figure()
    for name, (r, color, dash) in series.items():
        v = capital * (1 + r).cumprod()
        fig.add_scatter(x=v.index, y=v, name=name, line=dict(color=color, width=2, dash=dash),
                        hovertemplate="$%{y:,.0f}<extra>" + name + "</extra>")
    fig.update_yaxes(type="log", tickprefix="$", title="Portfolio value (log)")
    st.plotly_chart(style(fig, height), width="stretch")


# ---------------------------------------------------------------- header
st.title("Phase 3 — Can we trust the backtest?")
st.markdown(
    "A backtest is a simulation, and simulations are easy to fool. The three classic ways:\n"
    "1. **Lookahead bias**: the code uses information that wasn't available yet.\n"
    "2. **Overfitting**: you try many settings and keep the best-looking one, which is partly luck.\n"
    "3. **Testing on the data you tuned on**: then the result says nothing about the future.\n\n"
    "From this page on, every backtest uses **realistic settings**: "
    f"{COST_BPS} bps trading costs, cash earns T-bill interest, and trades happen at the close of "
    f"the **trading day after** the signal (tab 2)."
)

tabs = st.tabs(["1 · Lookahead bias", "2 · Trade timing", "3 · Automated tests",
                "4 · Overfitting", "5 · Walk-forward", "6 · Different periods"])

# ---------------------------------------------------------------- 1. lookahead
with tabs[0]:
    st.subheader("What a lookahead bug looks like")
    st.markdown(
        "Here's the whole difference between the honest strategy and a broken one: a single "
        "missing `.shift(1)`. Without it, the signal computed from the **end** of a month decides "
        "what we hold **during** that same month. In other words, we look at how the month ended "
        "and then 'decide' whether we'd have liked to own it."
    )
    st.code(
        "# honest: decide at the end of month t, hold during month t+1\n"
        "weights = (signal / n_assets).shift(1)\n\n"
        "# broken: decide at the end of month t, hold during month t (needs a time machine)\n"
        "weights = signal / n_assets",
        language="python",
    )
    honest = run_backtest(monthly, lookback, capital, asset_returns=rets, **realistic)
    cheat = cheating_backtest(monthly, rets, lookback, initial_capital=capital, **realistic)
    common = honest.portfolio_return.index.intersection(cheat.portfolio_return.index)
    growth_chart({
        "Honest backtest": (honest.portfolio_return.loc[common], BLUE, "solid"),
        "With lookahead bug": (cheat.portfolio_return.loc[common], RED, "dash"),
    })
    show_stats({"Honest backtest": stats_row(honest.portfolio_return.loc[common]),
                "With lookahead bug": stats_row(cheat.portfolio_return.loc[common])})
    st.markdown(
        "The broken version looks **dramatically** better, and nothing crashes or warns you. "
        "That's what makes lookahead bugs so dangerous: they don't look like bugs, they look like "
        "a great strategy. Rule of thumb: if a simple strategy looks amazing, suspect the backtest "
        "first. The automated tests in tab 3 exist to catch exactly this."
    )

# ---------------------------------------------------------------- 2. timing
with tabs[1]:
    st.subheader("When can you actually trade?")
    d = monthly.index[-2]
    nxt = mkt.daily.index[mkt.daily.index.get_loc(d) + 1]
    st.markdown(
        f"The signal uses the **closing price** of the month's last trading day, say "
        f"**{d:%a %d %b %Y}**. That price only exists once the market has closed, so you can't also "
        f"trade at it. The earliest realistic trade is at the close of the next trading day, "
        f"**{nxt:%a %d %b %Y}**.\n\n"
        "Phases 1 and 2 quietly assumed you could trade at the very price that produced the "
        "signal (a delay of 0 days). From now on we use a **1-day delay**. Below, the same strategy "
        "with different delays:"
    )
    timing = {}
    for lag in (0, 1, 2, 5):
        r = run_backtest(monthly, lookback, capital, asset_returns=market(tuple(tickers), cfg.refresh_token, lag).returns,
                         **realistic)
        timing[f"{lag}-day delay" + (" (Phases 1–2)" if lag == 0 else " (from now on)" if lag == 1 else "")] = \
            r.portfolio_return
    common = pd.concat(timing, axis=1).dropna().index
    show_stats({k: stats_row(v.loc[common]) for k, v in timing.items()})
    st.markdown(
        "The 0-day version is a little better, and it's the one you **can't** have. The gap is "
        "small here because this strategy trades slowly. For fast strategies (daily or intraday), "
        "getting timestamps wrong can create the *entire* profit."
    )
    with st.expander("How this is implemented"):
        st.markdown(
            "`quant/strategy.py → execution_prices()` finds, for every month-end decision date, the "
            "price `lag_days` trading days later. Each month's return is then measured from one "
            "execution price to the next, so the portfolio only starts earning from the moment the "
            "trade could really have happened."
        )

# ---------------------------------------------------------------- 3. tests
with tabs[2]:
    st.subheader("Automated tests: proof the rules are followed")
    st.markdown(
        "A test is a small program that checks one fact about the code and fails loudly if it ever "
        "stops being true. The most important ones here are **point-in-time tests**: take the "
        "price history, *wreck every price after some date*, rerun, and check that everything "
        "decided before that date is **exactly** the same. If anything changed, the code was "
        "peeking at the future.\n\n"
        "The tests run on made-up prices, not downloaded data, so a failure always means the code "
        "changed. You can also run them yourself in a terminal: `uv run pytest -v`."
    )

    def docstrings() -> dict[str, str]:
        out = {}
        for path in (ROOT / "tests").glob("test_*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    out[node.name] = " ".join((ast.get_docstring(node) or "").split())
        return out

    if st.button("Run all tests", icon=":material/play_arrow:", type="primary"):
        with st.spinner("Running pytest…"), tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.xml"
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", f"--junitxml={report}"],
                cwd=ROOT, capture_output=True, text=True, timeout=600,
            )
            rows, docs = [], docstrings()
            for case in ET.parse(report).getroot().iter("testcase"):
                failed = case.find("failure") is not None or case.find("error") is not None
                rows.append({
                    "Result": "❌ failed" if failed else "✅ passed",
                    "Test": case.get("name", "").removeprefix("test_").replace("_", " "),
                    "What it checks": docs.get(case.get("name", ""), ""),
                    "File": case.get("classname", "").split(".")[-1],
                    "Seconds": float(case.get("time", 0)),
                })
            st.session_state.test_results = (pd.DataFrame(rows), proc.stdout.strip().splitlines()[-1])

    if "test_results" in st.session_state:
        table, summary = st.session_state.test_results
        n_failed = table["Result"].str.contains("failed").sum()
        if n_failed:
            st.error(summary, icon=":material/error:")
        else:
            st.success(summary, icon=":material/check_circle:")
        st.dataframe(table, hide_index=True, height=35 * (len(table) + 1) + 3,
                     column_config={"What it checks": st.column_config.TextColumn(width="large"),
                                    "Seconds": st.column_config.NumberColumn(format="%.2f")})

    with st.expander("Do the tests actually catch bugs? (checked by breaking the code on purpose)"):
        st.markdown(
            "I inserted each of these bugs into the real code, one at a time, and confirmed that "
            "at least one test failed. Then I restored the code.\n\n"
            "| Planted bug | Caught by |\n|---|---|\n"
            "| Remove the one-month signal shift | *position held in month t comes from signal at t minus 1*, *positions never depend on future prices* |\n"
            "| Trade at the signal's own closing price | *trades execute on the next trading day* |\n"
            "| Walk-forward also trains on the test year | *walk forward trains only on the past* |\n"
            "| GARCH parameters fitted using later data | *garch forecasts ignore the future* |\n"
            "| LightGBM features peek one day ahead | *lgbm forecasts ignore the future* |\n"
            "| LightGBM trains on answers not known yet | *lgbm never trains on answers it could not know yet* |"
        )

# ---------------------------------------------------------------- shared: all lookbacks
@st.cache_data(show_spinner="Backtesting all 24 lookbacks…", max_entries=20)
def all_lookbacks(tickers: tuple[str, ...], token: tuple[int, int]) -> pd.DataFrame:
    m = market(tickers, token)
    return candidate_returns(m.monthly, m.returns, LOOKBACKS, cash_returns=m.rf, cost_bps=COST_BPS)


cand = all_lookbacks(tuple(tickers), cfg.refresh_token)
full_sharpe = sharpe_by_column(cand, rf)
best_lb = int(full_sharpe.idxmax())

# ---------------------------------------------------------------- 4. overfitting
with tabs[3]:
    st.subheader("Why 'pick the best setting' is a trap")
    st.markdown(
        f"Here is the Sharpe ratio of the strategy for **every lookback from 1 to 24 months**, "
        f"over the whole history ({cand.index[0]:%b %Y} – {cand.index[-1]:%b %Y}). It's tempting to "
        f"pick the tallest bar (**{best_lb} months**) and report its Sharpe of "
        f"**{full_sharpe[best_lb]:.2f}**. But with 24 tries, *some* bar will be tallest just by luck."
    )
    colors = [RED if lb == best_lb else (BLUE if lb == lookback else GRAY) for lb in LOOKBACKS]
    fig = go.Figure(go.Bar(x=LOOKBACKS, y=full_sharpe.values, marker_color=colors,
                           hovertemplate="%{x}-month lookback: Sharpe %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="Lookback (months)", dtick=1)
    fig.update_yaxes(title="Sharpe ratio")
    st.plotly_chart(style(fig, 320), width="stretch")
    st.caption(f"Red = best in hindsight ({best_lb}). Blue = your sidebar setting ({lookback}).")

    st.markdown(
        "**The test:** if the best settings were *genuinely* better, the lookbacks that did well in "
        "the first half of history should also do well in the second half. Each dot is one "
        "lookback:"
    )
    halves = split_half_sharpes(cand, rf)
    first_best = int(halves["first_half"].idxmax())
    rank_second = int(halves["second_half"].rank(ascending=False)[first_best])
    corr = halves.corr(method="spearman").iloc[0, 1]
    fig = go.Figure(go.Scatter(
        x=halves["first_half"], y=halves["second_half"], mode="markers+text",
        text=[str(i) for i in halves.index], textposition="top center",
        marker=dict(size=11, color=[RED if i == first_best else BLUE for i in halves.index],
                    line=dict(width=1, color="white")),
        hovertemplate="Lookback %{text}: first half %{x:.2f}, second half %{y:.2f}<extra></extra>",
    ))
    split = halves.attrs["split_date"]
    fig.update_xaxes(title=f"Sharpe, first half (until {split:%b %Y})", showgrid=True)
    fig.update_yaxes(title=f"Sharpe, second half (from {split:%b %Y})")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(style(fig, 420), width="stretch")
    luck = ("This time it held up reasonably well, but don't read too much into one example: "
            if rank_second <= 5 else "So it fell back into the pack. ")
    st.markdown(
        f"The first half's winner (**{first_best} months**, red) ranked **#{rank_second} of 24** in the "
        f"second half. {luck}look at the whole cloud. The rank correlation between the two halves is "
        f"**{corr:.2f}** (1 = perfectly consistent, 0 = no relationship), so knowing how a lookback did "
        "in the past tells you little about how it will do next. Picking the in-sample winner and "
        "reporting its number **overstates** what you'd really get. That's overfitting."
    )

# ---------------------------------------------------------------- 5. walk-forward
with tabs[4]:
    st.subheader("Walk-forward: only ever decide with data you'd have had")
    st.markdown(
        "Walk-forward testing simulates honestly *choosing* the lookback over time:\n"
        f"1. In January {FIRST_WF_YEAR}, look at all history **before** that date and pick the lookback "
        "with the best Sharpe.\n"
        "2. Trade it for the next 12 months. Those months are **out-of-sample**: they played no part "
        "in the choice.\n"
        "3. Next January, add the year that just finished to the history, choose again, and repeat.\n\n"
        "Stitching the out-of-sample years together gives a realistic estimate of what the "
        "'pick the best lookback' process would have earned."
    )
    wf = walk_forward(cand, rf, FIRST_WF_YEAR)
    oos = wf.returns.index

    fig = go.Figure()
    start = cand.index[0]
    year_ms = 365.25 * 24 * 3600 * 1000
    for i, row in wf.choices.iterrows():
        y = int(row["Test year"])
        cutoff = pd.Timestamp(year=row["Test year"], month=1, day=1)
        fig.add_bar(y=[y], x=[(cutoff - start).total_seconds() * 1000], base=[start], orientation="h",
                    marker_color=GRAY, name="Training data", showlegend=bool(i == 0), hovertemplate="trained on %{base|%b %Y} → " + f"{cutoff:%b %Y}<extra></extra>")
        fig.add_bar(y=[y], x=[year_ms], base=[cutoff], orientation="h", marker_color=BLUE,
                    name="Out-of-sample year", showlegend=bool(i == 0),
                    text=[f"{row['Chosen lookback']} months"], textposition="outside", textangle=0,
                    hovertemplate=f"traded {row['Test year']} with a {row['Chosen lookback']}-month lookback<extra></extra>")
    fig.update_layout(barmode="overlay", hovermode="closest")
    fig.update_xaxes(type="date")
    fig.update_yaxes(autorange="reversed", title="Test year", dtick=1)
    fig.update_xaxes(range=[start, pd.Timestamp(year=int(wf.choices["Test year"].max()) + 3, month=1, day=1)])
    st.plotly_chart(style(fig, 560), width="stretch")
    st.caption("Each row is one January decision: gray = data used to choose, blue = the year it was "
               "traded, label = chosen lookback.")

    bh = simulate(equal_weight(oos, tickers), rets, rf, COST_BPS).portfolio_return
    lines = {
        "Walk-forward (honest choice)": (wf.returns, BLUE, "solid"),
        f"Fixed {lookback} months (chosen in advance)": (cand[lookback].loc[oos], AQUA, "solid"),
        f"Best in hindsight: {best_lb} months (cheating)": (cand[best_lb].loc[oos], RED, "dash"),
        "Buy & hold: same ETFs": (bh.loc[oos], ORANGE, "dot"),
    }
    st.markdown(f"**Out-of-sample results, {oos[0]:%b %Y} – {oos[-1]:%b %Y}**")
    growth_chart(lines)
    show_stats({k: stats_row(v[0]) for k, v in lines.items()})
    wf_sharpe = M.sharpe_ratio(wf.returns, rf)
    hs_sharpe = M.sharpe_ratio(cand[best_lb].loc[oos], rf)
    st.markdown(
        f"The hindsight pick shows a Sharpe of **{hs_sharpe:.2f}**, but nobody could have known to pick "
        f"it. The honest walk-forward process got **{wf_sharpe:.2f}**. The gap between the two is "
        "the cost of overfitting. Notice also that re-optimizing every year did **not** beat simply "
        "fixing a sensible lookback in advance and leaving it alone.\n\n"
        f"And the honest conclusion: over these years, buy & hold of the same ETFs had a Sharpe of "
        f"**{M.sharpe_ratio(bh.loc[oos], rf):.2f}**. Measured out-of-sample, the trend rule's main "
        "benefit here is the smaller drawdowns, not better risk-adjusted returns."
    )
    with st.expander("Which lookback was chosen each year"):
        st.dataframe(wf.choices.style.format({"Its training Sharpe": "{:.2f}"}), hide_index=True)

# ---------------------------------------------------------------- 6. periods
with tabs[5]:
    st.subheader("Does it work in every era, or just one?")
    st.markdown(
        f"Same rules ({lookback}-month lookback, fixed in advance), split into separate periods. A "
        "strategy whose whole result comes from one lucky stretch is fragile. You want it to behave "
        "*consistently*: here, giving up some return in strong markets and losing less in bad ones."
    )
    strat = run_backtest(monthly, lookback, capital, asset_returns=rets, **realistic).portfolio_return
    months = strat.index
    portfolios = {
        "Trend strategy": (strat, BLUE),
        "Buy & hold: same ETFs": (simulate(equal_weight(months, tickers), rets, rf, COST_BPS).portfolio_return, ORANGE),
        "Buy & hold: SPY": (simulate(pd.DataFrame({"SPY": 1.0}, index=months), rets, rf, COST_BPS).portfolio_return, AQUA),
    }
    periods = [("2006–2010", "2006", "2010"), ("2011–2015", "2011", "2015"),
               ("2016–2020", "2016", "2020"), (f"2021–{months[-1].year}", "2021", str(months[-1].year))]
    rows = []
    for label, a, b in periods:
        for name, (r, _) in portfolios.items():
            rows.append({"Period": label, "Portfolio": name, **stats_row(r.loc[a:b])})
    table = pd.DataFrame(rows)

    metric = st.segmented_control("Compare by", ["Sharpe ratio", "CAGR", "Max drawdown"],
                                  default="Sharpe ratio", key="p3_period_metric") or "Sharpe ratio"
    fig = go.Figure()
    for name, (_, color) in portfolios.items():
        sub = table[table["Portfolio"] == name]
        fig.add_bar(x=sub["Period"], y=sub[metric], name=name, marker_color=color,
                    hovertemplate="%{y:" + (".2f" if metric == "Sharpe ratio" else ".1%") + "}<extra>" + name + "</extra>")
    fig.update_layout(barmode="group", bargap=0.3)
    st.plotly_chart(style(fig, 360, None if metric == "Sharpe ratio" else ".0%"), width="stretch")
    st.dataframe(
        table.set_index(["Period", "Portfolio"]).style.format(
            {"CAGR": "{:.1%}", "Volatility": "{:.1%}", "Sharpe ratio": "{:.2f}", "Max drawdown": "{:.1%}"}),
        height=35 * (len(table) + 1) + 3,
    )
