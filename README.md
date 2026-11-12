# quant: learning trend following step by step

| Phase | Topic | Status |
|---|---|---|
| 1 | Simple trend-following strategy (12-month return > 0 → hold, else cash) | ✅ |
| 2 | Evaluate it: buy-and-hold benchmarks, CAGR, volatility, Sharpe, drawdowns, costs | ✅ |
| 3 | Make the backtest trustworthy: trade timing, automated tests, overfitting, walk-forward | ✅ |
| 4 | Volatility forecasting (moving average, GARCH, LightGBM) and position sizing | ✅ |

## Run

```sh
uv run streamlit run app.py      # opens http://localhost:8501
uv run pytest -v                 # the test suite (also runnable from the Phase 3 page)
```

## Layout

- `quant/data.py`: downloads adjusted daily prices (and the T-bill yield, `^IRX`) from Yahoo Finance, cached in `data/`
- `quant/strategy.py`: the strategy, one function per step (prices → returns → trend → signal → weights), plus execution timing
- `quant/backtest.py`: the engine that turns any monthly weights into a portfolio value (turnover, trading costs, interest on cash), plus the buy-and-hold benchmarks
- `quant/metrics.py`: CAGR, volatility, Sharpe ratio, drawdowns, yearly returns
- `quant/validation.py`: lookahead demo, all-lookbacks backtests, split-half overfitting check, walk-forward selection
- `quant/volatility.py`: volatility target, moving-average / GARCH / LightGBM forecasts (all walk-forward), forecast scoring
- `quant/sizing.py`: volatility-targeted position sizes
- `tests/`: pytest suite on synthetic prices, including point-in-time (no-lookahead) tests
- `app.py`: Streamlit entry point (shared sidebar + navigation); `app_pages/`: one page per phase
