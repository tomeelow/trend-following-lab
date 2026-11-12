import numpy as np
import pandas as pd
import pytest

from conftest import perturb_after
from quant.sizing import vol_target_weights
from quant.strategy import month_end_prices
from quant.validation import walk_forward
from quant.volatility import (
    garch_forecast,
    lgbm_forecast,
    moving_average_forecast,
    realized_next_month,
    training_rows,
)


# ------------------------------------------------------------------ walk-forward
def make_candidates(seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", "2016-12-31", freq="ME")
    return pd.DataFrame(rng.normal(0.005, 0.03, (len(idx), 4)), index=idx, columns=[3, 6, 9, 12])


def test_walk_forward_trains_only_on_the_past():
    """Making one lookback look amazing DURING 2014 must not change which lookback was chosen FOR 2014."""
    cand = make_candidates()
    rf = pd.Series(0.0, index=cand.index)
    before = walk_forward(cand, rf, first_test_year=2013).choices.set_index("Test year")
    tampered = cand.copy()
    tampered.loc["2014", 3] = 0.10  # +10% every month in 2014
    after = walk_forward(tampered, rf, first_test_year=2013).choices.set_index("Test year")
    assert before.loc[2014, "Chosen lookback"] == after.loc[2014, "Chosen lookback"]
    assert after.loc[2015, "Chosen lookback"] == 3  # it only becomes attractive afterwards


def test_walk_forward_returns_are_only_test_years():
    """The stitched walk-forward result contains only out-of-sample months, starting with the first test year."""
    cand = make_candidates()
    wf = walk_forward(cand, pd.Series(0.0, index=cand.index), first_test_year=2013)
    assert wf.returns.index.min() == pd.Timestamp("2013-01-31")
    assert len(wf.returns) == len(cand.loc["2013":])


# ------------------------------------------------------------------ volatility
def test_target_is_the_following_months_volatility(daily):
    """The label for decision date t is the volatility of month t+1 (not month t)."""
    dates = month_end_prices(daily).index
    target = realized_next_month(daily, dates)
    t = dates[20]
    nxt = daily.pct_change().loc[f"{dates[21]:%Y-%m}"]
    assert target.loc[t, "AAA"] == pytest.approx(nxt["AAA"].std() * np.sqrt(252))
    assert np.isnan(target.iloc[-1]).all()  # the month after the last one isn't over yet


def test_moving_average_forecast_is_point_in_time(daily):
    """Wreck every price after a month end: the moving-average forecast made on that day must not change."""
    dates = month_end_prices(daily).index
    t = dates[30]
    a = moving_average_forecast(daily, dates, 63).loc[t]
    b = moving_average_forecast(perturb_after(daily, str(t.date())), dates, 63).loc[t]
    pd.testing.assert_series_equal(a, b)


def test_garch_forecasts_ignore_the_future(daily):
    """Wreck every price after June 30, 2012: all forecasts made up to that date must be identical.
    (Catches parameters being re-fitted with data from later in the year.)"""
    dates = month_end_prices(daily).index
    a, _ = garch_forecast(daily, dates, first_year=2011)
    b, _ = garch_forecast(perturb_after(daily, "2012-06-30"), dates, first_year=2011)
    pd.testing.assert_frame_equal(a.loc[:"2012-06-30"], b.loc[:"2012-06-30"])
    assert a.loc["2012"].notna().all().all()


def test_lgbm_never_trains_on_answers_it_could_not_know_yet():
    """Training in January 2015: the November 2014 row's answer (December's vol) is known, December 2014's isn't."""
    dates = pd.to_datetime(["2014-10-31", "2014-11-28", "2014-12-31", "2015-01-30"])
    assert training_rows(dates, 2015).tolist() == [True, True, False, False]


def test_lgbm_forecasts_ignore_the_future(daily):
    """Wreck every price after June 30, 2012: all LightGBM forecasts made up to that date must be identical."""
    dates = month_end_prices(daily).index
    a, _ = lgbm_forecast(daily, dates, realized_next_month(daily, dates), first_year=2012)
    wrecked = perturb_after(daily, "2012-06-30")
    b, _ = lgbm_forecast(wrecked, dates, realized_next_month(wrecked, dates), first_year=2012)
    pd.testing.assert_frame_equal(a.loc[:"2012-06-30"], b.loc[:"2012-06-30"])
    assert a.loc["2012"].notna().all().all()


# ------------------------------------------------------------------ sizing
def test_vol_targeting_sizes_and_caps():
    """Riskier asset -> smaller slot; total never above the cap; decided one month ahead."""
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    sig = pd.DataFrame({"calm": 1.0, "wild": 1.0}, index=idx)
    vol = pd.DataFrame({"calm": 0.06, "wild": 0.24}, index=idx)
    w = vol_target_weights(sig, vol, target_vol=0.12, max_slot_scale=2.0, max_gross=1.0)
    assert w.iloc[0].isna().all()  # nothing decided before the first month end
    assert w.iloc[1]["calm"] == pytest.approx(0.5 * 2.0 * (1.0 / 1.25))  # 2x slot, then shrunk to 100% total
    assert w.iloc[1]["wild"] == pytest.approx(0.5 * 0.5 * (1.0 / 1.25))
    assert w.iloc[1].sum() == pytest.approx(1.0)


def test_vol_targeting_respects_signal():
    """Volatility only decides HOW MUCH: an ETF whose trend signal is off still gets zero weight."""
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    sig = pd.DataFrame({"A": [0.0, 0.0], "B": [1.0, 1.0]}, index=idx)
    vol = pd.DataFrame({"A": 0.1, "B": 0.1}, index=idx)
    w = vol_target_weights(sig, vol, target_vol=0.1)
    assert w.iloc[1]["A"] == 0.0
