"""Phase 4: use volatility forecasts to size positions (volatility targeting).

Phase 1 gives every "on" ETF the same 1/N of the portfolio. But a 20% slot in
emerging-market stocks (vol ~22%) carries far more risk than a 20% slot in
Treasuries (vol ~14%), and any ETF is much riskier in a crisis than in a calm
year. Volatility targeting sizes each slot so that it carries a fixed amount
of *risk* instead of a fixed amount of *money*:

    slot scale = target volatility / forecast volatility      (capped)
    weight     = signal x (1/N) x slot scale

High forecast vol -> smaller position; low forecast vol -> bigger position.
If the total would exceed `max_gross` (e.g. 100% = no borrowing), every
position is shrunk proportionally.
"""

import pandas as pd


def vol_target_weights(
    sig: pd.DataFrame,
    vol_forecast: pd.DataFrame,
    target_vol: float = 0.12,
    max_slot_scale: float = 2.0,
    max_gross: float = 1.0,
) -> pd.DataFrame:
    """Weights held during each month, sized by the forecast made at the
    previous month end (shifted by one month, exactly like the signal)."""
    n_assets = sig.shape[1]
    scale = (target_vol / vol_forecast).clip(upper=max_slot_scale)
    raw = sig * scale / n_assets
    gross = raw.sum(axis=1)
    shrink = (max_gross / gross).clip(upper=1.0).where(gross > 0, 1.0)
    sized = raw.mul(shrink, axis=0)
    return sized.shift(1)
