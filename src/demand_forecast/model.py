"""The global forecasting model: training, and honest recursive forecasting.

Design note on why forecasting here is *recursive*: the model's features
include lags (`lag_7`, `lag_14`, ...) and rolling statistics of `units`. If
we naively built those features once over the full history (train + test)
and evaluated row by row, a lag-7 feature for a test-period row would
sometimes be filled from an *actual* value a few days earlier in the same
test period -- information a real deployment would not have yet, because
that "earlier" value hasn't been observed at forecast time either. That's
train/serve (and here, backtest/live) skew hiding as a leak.

`recursive_forecast` instead walks the forecast horizon one day at a time:
for each day, it rebuilds lag/rolling features from whatever is in
`history` so far (actual values before the cutoff, its own predictions
after), predicts, and only then appends that prediction to `history` before
moving to the next day. This is slower than a single vectorized `.predict()`
call but it's the only way to get an evaluation number that means what it
claims to mean -- and it's exactly what `api.py` does to forecast forward
from the end of the real data too, so backtest and live serving share one
code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from demand_forecast.config import FORECAST_CONFIG
from demand_forecast.features import (
    add_calendar_features,
    add_lag_and_rolling_features,
    feature_columns,
    set_categorical_dtypes,
)

TARGET_LOG = "log_units"


def train_model(train_df: pd.DataFrame, categories: dict[str, list[str]]) -> xgb.XGBRegressor:
    """Fits the global model on a feature frame that already has lag/rolling
    columns (rows with any NaN lag, i.e. the first `min_history_days` of
    each SKU's history, must already be dropped by the caller)."""
    df = set_categorical_dtypes(train_df, categories)
    X = df[feature_columns()]
    y = np.log1p(df["units"].astype(float))
    model = xgb.XGBRegressor(**FORECAST_CONFIG.xgb_params)
    model.fit(X, y)
    return model


def predict_units(model: xgb.XGBRegressor, feature_df: pd.DataFrame, categories: dict[str, list[str]]) -> np.ndarray:
    df = set_categorical_dtypes(feature_df, categories)
    X = df[feature_columns()]
    log_pred = model.predict(X)
    return np.clip(np.expm1(log_pred), 0.0, None)


def recursive_forecast(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    categories: dict[str, list[str]],
    forecast_dates: list[pd.Timestamp],
    future_promo_flag: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Forecasts `forecast_dates` (must be consecutive, sorted, immediately
    after the last date in `history_df`) one day at a time, feeding each
    day's prediction back in as history for the next day's lag features.

    `history_df` needs columns date, sku_id, category, units, promo_flag and
    should contain the *actual* observed series up to (not including) the
    first forecast date. `future_promo_flag`, if given, is a
    (date, sku_id, promo_flag) frame telling the model about planned
    promotions during the forecast horizon (a real retailer plans promos
    weeks ahead, so treating them as known exogenous inputs is realistic);
    it defaults to "no promo" for every forecast day.
    """
    sku_ids = history_df["sku_id"].unique()
    sku_category = history_df.drop_duplicates("sku_id").set_index("sku_id")["category"].to_dict()

    skeleton_rows = []
    for date in forecast_dates:
        for sku_id in sku_ids:
            skeleton_rows.append({"date": date, "sku_id": sku_id, "category": sku_category[sku_id], "units": np.nan})
    skeleton = pd.DataFrame(skeleton_rows)
    if future_promo_flag is not None and len(future_promo_flag) > 0:
        skeleton = skeleton.merge(future_promo_flag, on=["date", "sku_id"], how="left")
        skeleton["promo_flag"] = skeleton["promo_flag"].fillna(0).astype(int)
    else:
        skeleton["promo_flag"] = 0

    working = pd.concat(
        [history_df[["date", "sku_id", "category", "units", "promo_flag"]], skeleton],
        ignore_index=True,
    ).sort_values(["sku_id", "date"])

    predictions = []
    for date in forecast_dates:
        featured = add_calendar_features(working)
        featured = add_lag_and_rolling_features(featured, FORECAST_CONFIG.lags, FORECAST_CONFIG.rolling_windows)
        day_rows = featured[featured["date"] == date].copy()
        preds = predict_units(model, day_rows, categories)
        day_rows["forecast"] = preds
        predictions.append(day_rows[["date", "sku_id", "forecast"]])

        # feed the prediction back in as this day's "units" so the next
        # iteration's lag/rolling features see it.
        working.loc[working["date"] == date, "units"] = day_rows.set_index("sku_id")["forecast"].reindex(
            working.loc[working["date"] == date, "sku_id"]
        ).to_numpy()

    return pd.concat(predictions, ignore_index=True)
