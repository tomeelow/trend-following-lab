"""Trend-following learning app. Run with:  uv run streamlit run app.py"""

import streamlit as st

from app_shared import UNIVERSE

st.set_page_config(page_title="Trend following", page_icon=":material/trending_up:", layout="wide")

# ---------------------------------------------------------------- shared sidebar
with st.sidebar:
    st.header("Settings")
    st.multiselect(
        "ETFs (pick 3–5)",
        options=list(UNIVERSE),
        default=["SPY", "EFA", "EEM", "TLT", "GLD"],
        max_selections=5,
        format_func=lambda t: f"{t} — {UNIVERSE[t]}",
        key="tickers",
    )
    st.slider("Trend lookback (months)", 1, 24, 12, key="lookback")
    st.number_input("Starting capital ($)", 1_000, 1_000_000, 10_000, step=1_000, key="capital")
    if st.button("Re-download prices", icon=":material/refresh:"):
        st.session_state.refresh_token = st.session_state.get("refresh_token", 0) + 1
    st.caption(
        "Lookback is adjustable so you can play with it, but notice the temptation "
        "to pick whichever value looks best. That's overfitting — Phase 3."
    )

page = st.navigation(
    [
        st.Page(
            "app_pages/phase1_build.py",
            title="Phase 1 · Build",
            icon=":material/construction:",
            help="The strategy step by step: prices, returns, trend signal, positions and the backtest.",
        ),
        st.Page(
            "app_pages/phase2_evaluate.py",
            title="Phase 2 · Evaluate",
            icon=":material/analytics:",
            help="Is it any good? Compared with buy-and-hold: return, volatility, Sharpe, drawdowns, costs.",
        ),
        st.Page(
            "app_pages/phase3_trust.py",
            title="Phase 3 · Trust",
            icon=":material/verified:",
            help="Lookahead bias, trade timing, automated tests, overfitting and walk-forward validation.",
        ),
        st.Page(
            "app_pages/phase4_volatility.py",
            title="Phase 4 · Volatility",
            icon=":material/ssid_chart:",
            help="Forecast volatility (moving average, GARCH, LightGBM), score the forecasts, size positions.",
        ),
    ],
    position="top",
)

if not st.session_state.tickers:
    st.warning("Pick at least one ETF in the sidebar.")
    st.stop()

page.run()
