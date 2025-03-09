"""Seasonal-naive baseline: "demand on this day = demand on the same weekday
last week". It's the standard sanity-check baseline in demand forecasting
(cheap, no training, captures weekly seasonality for free) -- if the global
XGBoost model in `model.py` can't beat this by a believable margin, that's
a real finding worth reporting honestly, not a test to delete.

Forecasted recursively, exactly like the real model (`model.recursive_forecast`):
day 8 of the forecast horizon looks up day 1's value, which for day 8 is
itself a forecast, not an actual -- because in a genuine 56-day-ahead
deployment, day 1 of the horizon is also unobserved at the moment day 8 is
forecast. Evaluating it any other way would flatter the baseline.
"""

from __future__ import annotations

import pandas as pd


def seasonal_naive_forecast(history_df: pd.DataFrame, forecast_dates: list[pd.Timestamp]) -> pd.DataFrame:
    ext = history_df[["date", "sku_id", "units"]].copy()
    predictions = []
    for date in sorted(forecast_dates):
        lookup_date = date - pd.Timedelta(days=7)
        lookup = ext.loc[ext["date"] == lookup_date, ["sku_id", "units"]].rename(columns={"units": "forecast"})
        lookup["date"] = date
        predictions.append(lookup)
        fill = lookup.rename(columns={"forecast": "units"})[["date", "sku_id", "units"]]
        ext = pd.concat([ext, fill], ignore_index=True)
    return pd.concat(predictions, ignore_index=True)[["date", "sku_id", "forecast"]]
