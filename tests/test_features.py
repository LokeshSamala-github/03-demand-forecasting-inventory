"""Tests for feature engineering, in particular that lag/rolling features
never leak same-day or future information."""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_forecast.features import (
    add_calendar_features,
    add_lag_and_rolling_features,
    build_feature_frame,
    feature_columns,
)


def _toy_sales(n_days=40, sku_ids=("A", "B")):
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for sku in sku_ids:
        for i, d in enumerate(dates):
            rows.append({"date": d, "sku_id": sku, "category": "Toys", "units": i + (0 if sku == "A" else 100), "promo_flag": 0})
    return pd.DataFrame(rows)


def test_calendar_features_correct():
    df = add_calendar_features(_toy_sales(n_days=8, sku_ids=("A",)))
    # 2024-01-01 is a Monday -> dow 0
    assert df.iloc[0]["dow"] == 0
    assert df.iloc[0]["is_weekend"] == 0
    saturday_row = df[df["date"] == pd.Timestamp("2024-01-06")].iloc[0]
    assert saturday_row["dow"] == 5
    assert saturday_row["is_weekend"] == 1


def test_lag_feature_equals_shifted_value():
    df = add_lag_and_rolling_features(_toy_sales(n_days=40, sku_ids=("A",)), lags=(7,), rolling_windows=(7,))
    df = df.sort_values("date").reset_index(drop=True)
    # units for sku A is just the day index i, so lag_7 on day i should be i-7
    for i in range(10, 20):
        assert df.loc[i, "lag_7"] == df.loc[i, "units"] - 7


def test_lag_features_are_nan_for_early_rows_not_leaked_from_future():
    df = add_lag_and_rolling_features(_toy_sales(n_days=20, sku_ids=("A",)), lags=(7,), rolling_windows=(7,))
    df = df.sort_values("date").reset_index(drop=True)
    assert df.loc[:5, "lag_7"].isna().all()
    # no lag value should ever come from a later date than the row itself
    for i in range(7, 20):
        assert df.loc[i, "lag_7"] < df.loc[i, "units"]


def test_rolling_mean_excludes_current_day():
    df = add_lag_and_rolling_features(_toy_sales(n_days=20, sku_ids=("A",)), lags=(1,), rolling_windows=(3,))
    df = df.sort_values("date").reset_index(drop=True)
    # units = day index; roll_mean_3 at day i should be mean(i-3, i-2, i-1), never including i
    row = df.loc[10]
    assert abs(row["roll_mean_3"] - (7 + 8 + 9) / 3) < 1e-9


def test_features_do_not_mix_across_skus():
    df = add_lag_and_rolling_features(_toy_sales(n_days=15, sku_ids=("A", "B")), lags=(1,), rolling_windows=(3,))
    df = df.sort_values(["sku_id", "date"]).reset_index(drop=True)
    b_rows = df[df["sku_id"] == "B"].reset_index(drop=True)
    # SKU B's lag_1 should stay in B's own ~100+ range, never pick up A's ~0-15 range
    assert (b_rows.loc[1:, "lag_1"] >= 95).all()


def test_build_feature_frame_has_all_expected_columns():
    sales_df = _toy_sales(n_days=40, sku_ids=("A", "B"))
    sku_meta_df = pd.DataFrame({"sku_id": ["A", "B"], "category": ["Toys", "Toys"]})
    featured = build_feature_frame(sales_df, sku_meta_df)
    for col in feature_columns():
        assert col in featured.columns
    assert not featured["category"].isna().any()


def test_build_feature_frame_row_count_matches_input():
    sales_df = _toy_sales(n_days=25, sku_ids=("A", "B", "C"))
    sku_meta_df = pd.DataFrame({"sku_id": ["A", "B", "C"], "category": ["X", "Y", "Z"]})
    featured = build_feature_frame(sales_df, sku_meta_df)
    assert len(featured) == len(sales_df)
    assert np.array_equal(sorted(featured["sku_id"].unique()), ["A", "B", "C"])
