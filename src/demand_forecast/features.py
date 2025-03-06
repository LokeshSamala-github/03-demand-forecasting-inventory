"""Feature engineering shared by training, backtesting, and live serving.

The model is a single *global* gradient-boosted model trained across every
SKU at once (rather than one model per SKU) -- with ~25 SKUs and only a
couple of years of daily history each, a per-SKU model has too little data
to learn a weekly/annual pattern reliably, whereas a global model borrows
statistical strength across SKUs (learns "what a promo day generally does"
once) while still telling SKUs apart via the `sku_id` / `category`
categorical features. This is also standard modern practice for retail
demand forecasting at scale (one model to maintain, not thousands).

Lag and rolling-window features are computed with `shift(1)` before the
rolling window, and lags are always >= 1 day, so a row's features never see
that row's own (or a later row's) target -- this is what keeps recursive
forecasting (see `model.py`) honest instead of accidentally peeking at the
answer.
"""

from __future__ import annotations

import pandas as pd

from demand_forecast.config import FORECAST_CONFIG

CALENDAR_FEATURES = ["dow", "is_weekend", "month", "day_of_year", "week_of_year"]
CATEGORICAL_FEATURES = ["sku_id", "category"]
TARGET = "units"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dow"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    return df


def lag_feature_columns(lags: tuple[int, ...] = FORECAST_CONFIG.lags) -> list[str]:
    return [f"lag_{n}" for n in lags]


def rolling_feature_columns(windows: tuple[int, ...] = FORECAST_CONFIG.rolling_windows) -> list[str]:
    cols = []
    for w in windows:
        cols += [f"roll_mean_{w}", f"roll_std_{w}"]
    return cols


def feature_columns() -> list[str]:
    return (
        CALENDAR_FEATURES
        + CATEGORICAL_FEATURES
        + ["promo_flag"]
        + lag_feature_columns()
        + rolling_feature_columns()
    )


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    lags: tuple[int, ...] = FORECAST_CONFIG.lags,
    rolling_windows: tuple[int, ...] = FORECAST_CONFIG.rolling_windows,
) -> pd.DataFrame:
    """Adds lag_N and roll_mean_W / roll_std_W columns, computed per SKU from
    `units`. Must be called on a frame that is already sorted by
    (sku_id, date) with one row per SKU per day (no gaps) -- see
    `merge_units_into_history` for how recursive forecasting keeps this
    contract when `units` beyond the training cutoff is itself predicted."""
    df = df.sort_values(["sku_id", "date"]).copy()
    grouped = df.groupby("sku_id", observed=True)["units"]
    for n in lags:
        df[f"lag_{n}"] = grouped.shift(n)
    shifted = grouped.shift(1)
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = shifted.groupby(df["sku_id"], observed=True).rolling(w).mean().reset_index(
            level=0, drop=True
        )
        df[f"roll_std_{w}"] = shifted.groupby(df["sku_id"], observed=True).rolling(w).std().reset_index(
            level=0, drop=True
        )
    return df


def set_categorical_dtypes(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Applies a fixed, saved category set to sku_id/category so the encoding
    XGBoost sees at inference exactly matches training (even if a batch
    happens to contain only a subset of SKUs)."""
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


def category_manifest(df: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(df[col].astype(str).unique().tolist()) for col in CATEGORICAL_FEATURES}


def build_feature_frame(sales_df: pd.DataFrame, sku_meta_df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end feature build for a (sku_id, date, units, promo_flag,
    category) frame that already has one row per SKU per calendar day."""
    df = sales_df.merge(sku_meta_df[["sku_id", "category"]], on="sku_id", how="left", suffixes=("", "_meta"))
    if "category_meta" in df.columns:
        df["category"] = df["category_meta"]
        df = df.drop(columns=["category_meta"])
    df = add_calendar_features(df)
    df = add_lag_and_rolling_features(df)
    return df
