"""Tests for the seasonal-naive baseline."""

from __future__ import annotations

import pandas as pd

from demand_forecast.baseline import seasonal_naive_forecast


def _history(n_days=30, sku_ids=("A",)):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for sku in sku_ids:
        for i, d in enumerate(dates):
            rows.append({"date": d, "sku_id": sku, "units": float(i)})
    return pd.DataFrame(rows)


def test_first_forecast_day_equals_actual_seven_days_earlier():
    history = _history(n_days=30)
    forecast_dates = [pd.Timestamp("2024-01-31")]  # day 31 -> lag from day 24 (index 23, units=23.0)
    pred = seasonal_naive_forecast(history, forecast_dates)
    assert len(pred) == 1
    assert pred.iloc[0]["forecast"] == 23.0


def test_recursive_beyond_one_week_uses_its_own_prior_forecast():
    """Day 15 of a 14-day horizon needs day 8's value, which is itself a
    forecast (not an actual) -- the function should not crash or silently
    look up an actual that doesn't exist yet."""
    history = _history(n_days=20)
    forecast_dates = list(pd.date_range("2024-01-21", periods=14, freq="D"))
    pred = seasonal_naive_forecast(history, forecast_dates)
    assert len(pred) == 14
    assert pred["forecast"].isna().sum() == 0


def test_multi_sku_forecasts_dont_mix():
    history = _history(n_days=20, sku_ids=("A", "B"))
    history.loc[history["sku_id"] == "B", "units"] += 1000
    forecast_dates = [pd.Timestamp("2024-01-25")]
    pred = seasonal_naive_forecast(history, forecast_dates)
    a_val = pred[pred["sku_id"] == "A"]["forecast"].iloc[0]
    b_val = pred[pred["sku_id"] == "B"]["forecast"].iloc[0]
    assert b_val - a_val >= 999
