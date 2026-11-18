# Trend Following Lab

[![tests](https://github.com/tomeelow/trend-following-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/tomeelow/trend-following-lab/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.13-blue)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://trend-following-lab.streamlit.app)

A multi-asset **trend-following strategy** built from scratch, and then put through the
checks a backtest needs before anyone should believe it: realistic trade timing, costs,
interest on cash, an automated no-lookahead test suite, an overfitting check,
walk-forward validation, and volatility-targeted position sizing driven by GARCH and
LightGBM forecasts.

The point is the process rather than the strategy. Each of the four phases is a page in an
interactive Streamlit app that shows every intermediate step, from raw prices to the
final equity curve, so each number can be traced back to where it came from.

**Live demo: [trend-following-lab.streamlit.app](https://trend-following-lab.streamlit.app)**. Nothing to install.

## Headline results

Five ETFs (SPY, EFA, EEM, TLT, GLD), 12-month lookback, monthly rebalancing, trading at the
close of the day *after* the signal, 10 bps per unit of turnover, idle cash earning the
3-month T-bill rate. Held months Dec 2005 to Oct 2026, data from Yahoo Finance to 18 Nov 2026.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Avg. invested |
|---|---:|---:|---:|---:|---:|
| **Trend following (12-month)** | 6.2% | **7.6%** | **0.62** | **−9.4%** | 70% |
| Equal weight, always invested | 7.9% | 12.4% | 0.54 | −33.7% | 100% |
| SPY buy & hold | 10.9% | 16.7% | 0.60 | −52.9% | 100% |
| *Same rule with a lookahead bug* | *11.9%* | *7.3%* | *1.36* | *−6.5%* | *70%* |

- **The strategy is a risk reducer, not a return engine.** It earns less than buying and
  holding, at a similar Sharpe ratio, with less than a fifth of SPY's worst drawdown. By
  the autumn 2008 crash it had cut its exposure to 20–40%.
- **A one-line bug doubles the Sharpe ratio.** Dropping the one-month shift between
  signal and position (the last row) makes the backtest look outstanding. The test suite
  exists so that this cannot happen silently.
- **Tuning the lookback is mostly fitting noise.** The best lookback on one half of the
  history barely predicts the best on the other half (correlation of Sharpe ratios across
  24 lookbacks: 0.23). Picking the lookback walk-forward each January from past data alone
  gave a Sharpe of 0.44 from 2010, *below* the 0.50 of simply keeping 12 months.
- **GARCH forecasts volatility best. LightGBM does not beat it.** Over 1,125
  out-of-sample ETF-months, GARCH(1,1) has the lowest QLIKE (0.29) against 0.39–0.41 for
  LightGBM and moving averages. With a few thousand rows, built-in structure beats
  flexibility.
- **Volatility targeting smooths the ride and adds little return.** Sizing positions by
  forecast volatility lowers portfolio volatility (7.5% → 6.3–7.0%) and lifts the Sharpe
  from 0.52 to about 0.56. Perfect foresight of next month's volatility would reach 0.99:
  that gap is the value of a better forecast.

Numbers move a little each month as new prices arrive. The app recomputes all of them
from fresh data.

## The four phases

| Phase | Question | What the page shows |
|---|---|---|
| **1 · Build** | What does the strategy actually do? | Prices → month-end prices → 12-month returns → signals → weights → portfolio value, each step as a table and chart, plus current positions |
| **2 · Evaluate** | Is it any good? | Buy-and-hold benchmarks (equal weight in the same ETFs, and SPY), CAGR, volatility, Sharpe, drawdown episodes, calendar-year returns, turnover, and returns at trading costs from 0 to 100 bps |
| **3 · Trust** | Should I believe the backtest? | Lookahead bias demo, next-day execution, the live pytest suite, Sharpe for all 24 lookbacks, the split-half overfitting check, walk-forward lookback selection |
| **4 · Volatility** | Can forecasting risk improve it? | Moving-average, GARCH(1,1) and LightGBM forecasts (all walk-forward), scored by RMSE, MAE, QLIKE and R², feature importance, and volatility-targeted sizing against a perfect-foresight bound |

## Method

### The strategy

Time-series momentum, applied to each ETF independently once a month:

```
if return over the last 12 months > 0:  hold the ETF in its 1/N slot
else:                                   hold that slot in cash
```

The signal computed at the close of month *t* decides the position held during month
*t + 1*. Trades execute at the close of the next trading day, because the month-end close
is only known once the market has shut.

### The backtest engine

[`quant/backtest.py`](quant/backtest.py) turns any table of monthly target weights into a
portfolio value. The strategy, benchmarks and volatility-sized variants all go through the
same accounting:

- weights **drift** with returns between rebalances, so turnover is measured against the
  drifted weights rather than last month's targets;
- costs are charged on turnover, including the first purchase;
- uninvested weight earns the T-bill yield **known at the start of the month**.

### Making it trustworthy

- **No-lookahead tests.** Every price after a cut-off date is scrambled, and the tests
  assert that no position, forecast or training set before that date changes. They cover
  the signal, the GARCH and LightGBM forecasts, and the walk-forward split.
- **Overfitting check.** Sharpe ratios for lookbacks of 1 to 24 months, first half of
  history against second half.
- **Walk-forward selection.** Each January the lookback with the best Sharpe on all
  *earlier* data is chosen and used for the next 12 months.

### Volatility forecasting and sizing

All forecasts are made at a month end from data up to that day, and target the realised
volatility of the following month.

| Model | How it works |
|---|---|
| Moving average | Standard deviation of the last 21 or 63 daily returns |
| GARCH(1,1) | Re-estimated every January on past returns only, then updated day by day; forecasts variance over the next 21 trading days |
| LightGBM | Pooled across ETFs and retrained every January on rows whose answer was already known. It predicts log volatility from 11 features: realised vol at 5 days to 1 year, EWMA vol, the vol ratio, recent returns, drawdown, the largest daily move and the ETF |

Position size: `weight = signal × (1/N) × min(2, target vol / forecast vol)`, scaled down
together if total exposure would exceed 100%. The trend signal still decides in or out.
Volatility only decides how much.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). Python 3.13 is installed automatically.

```bash
git clone https://github.com/tomeelow/trend-following-lab.git
cd trend-following-lab
uv run streamlit run app.py      # opens http://localhost:8501
uv run pytest -v                 # 25 tests, all on synthetic prices, no network
```

The first run downloads daily prices from Yahoo Finance into `data/` (not committed). The
sidebar picks 3 to 5 ETFs from a universe of 8 and sets the lookback and starting
capital, and every page updates.

## Tests

The suite runs on synthetic random-walk prices with regime-switching volatility, so a
failure always means the code changed, never the market. It checks:

- **Accounting:** portfolio returns by hand, costs on turnover, cash interest, the cash
  rate known in advance, CAGR, max drawdown and Sharpe against their formulas.
- **Timing:** unfinished months are dropped, positions come from the previous month's
  signal, and trades happen on the next trading day.
- **Point-in-time:** positions and the moving-average, GARCH and LightGBM forecasts are
  unchanged when future prices are scrambled, the forecast target is the following
  month's volatility, and LightGBM never trains on an answer it could not have known yet.
- **Walk-forward and sizing:** each year is chosen from earlier data only, and volatility
  sizing respects the signal, the slot cap and the exposure cap.

They run on every push via GitHub Actions and can also be run from the Phase 3 page of the
app.

## Limitations

- **Small universe, one rule.** Five ETFs and a single 12-month rule. A real trend
  portfolio would hold dozens of futures across asset classes and blend several lookbacks.
- **About 20 years of monthly data**, roughly 250 decisions. Sharpe ratio differences of
  0.05 to 0.1 between variants are within noise.
- **Survivorship and data quality.** Today's ETFs, Yahoo Finance adjusted closes, no
  second vendor to cross-check against.
- **Simple costs.** A flat 10 bps per unit of turnover, with no spreads, market impact or
  taxes.
- Nothing here is investment advice.

## Hosting

The app runs on [Streamlit Community Cloud](https://share.streamlit.io) from `main` (main
file `app.py`, Python 3.13). The host installs from `uv.lock`, so don't add a
`requirements.txt`: it would take precedence over the lockfile.

- **Fresh data:** prices are downloaded again from Yahoo Finance when the cached copy is
  more than 12 hours old. If Yahoo refuses (it sometimes rate-limits cloud servers), the
  app keeps using the cached copy.
- **Staying awake:** free apps sleep after a while without visitors.
  [`keep-awake.yml`](.github/workflows/keep-awake.yml) opens the app in a headless browser
  every 6 hours ([`scripts/keep_awake.py`](scripts/keep_awake.py)) and presses the wake-up
  button if it's asleep. GitHub pauses scheduled workflows after 60 days without commits.
  Re-enable it from the Actions tab if that happens.

## Project structure

```
quant/
  data.py          adjusted daily prices and the T-bill yield (^IRX) from Yahoo Finance, cached
  strategy.py      the rule, one function per step, plus next-day execution
  backtest.py      weights → portfolio value: drift, turnover, costs, interest on cash; benchmarks
  metrics.py       CAGR, volatility, Sharpe, drawdowns and drawdown episodes, yearly returns
  validation.py    lookahead demo, all-lookback backtests, split-half check, walk-forward
  volatility.py    moving-average, GARCH and LightGBM forecasts (walk-forward), forecast scoring
  sizing.py        volatility-targeted position sizes
app.py             Streamlit entry point: shared sidebar and navigation
app_pages/         one page per phase
app_shared.py      settings, cached data loading, chart styling
tests/             pytest suite on synthetic prices
scripts/           keep-awake visit for the hosted app
```

## Built with

Python 3.13 · pandas · arch (GARCH) · LightGBM · Plotly · Streamlit · yfinance · pytest ·
uv · GitHub Actions

## References

- Moskowitz, T., Ooi, Y. H. & Pedersen, L. H. (2012). Time series momentum. *Journal of Financial Economics*.
- Hurst, B., Ooi, Y. H. & Pedersen, L. H. (2017). A century of evidence on trend-following investing. *Journal of Portfolio Management*.
- Faber, M. (2007). A quantitative approach to tactical asset allocation. *Journal of Wealth Management*.
- Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics*.
- Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
- Bailey, D., Borwein, J., López de Prado, M. & Zhu, Q. (2014). Pseudo-mathematics and financial charlatanism: the effects of backtest overfitting. *Notices of the AMS*.

## License

[MIT](LICENSE)
