"""End-to-end tests against the real trained pipeline (see
tests/conftest.py::trained_pipeline) -- real behavioral assertions, not just
"doesn't crash"."""

from __future__ import annotations

import json

import pandas as pd


def test_pipeline_produces_all_artifacts(trained_pipeline):
    paths = trained_pipeline
    assert paths.model_file.exists()
    assert paths.feature_manifest.exists()
    assert paths.metrics_file.exists()
    assert paths.backtest_csv.exists()
    assert paths.forecast_plot_png.exists()
    assert paths.forecast_plot_png.stat().st_size > 0


def test_metrics_json_has_both_models_and_is_finite(trained_pipeline):
    with open(trained_pipeline.metrics_file) as f:
        metrics = json.load(f)
    overall = metrics["overall"]
    for model_name in ("xgboost_global", "seasonal_naive"):
        assert model_name in overall
        for metric_name in ("mape", "wape", "rmse"):
            value = overall[model_name][metric_name]
            assert value == value  # not NaN
            assert value >= 0


def test_model_beats_seasonal_naive_baseline_on_wape(trained_pipeline):
    """The core forecasting claim: the global XGBoost model should beat the
    trivial seasonal-naive baseline by a believable margin on this
    structured synthetic data. If this regresses, that's a real finding to
    investigate (see README), not a test to delete."""
    with open(trained_pipeline.metrics_file) as f:
        metrics = json.load(f)
    model_wape = metrics["overall"]["xgboost_global"]["wape"]
    baseline_wape = metrics["overall"]["seasonal_naive"]["wape"]
    assert model_wape < baseline_wape


def test_backtest_predictions_cover_every_sku(trained_pipeline):
    backtest = pd.read_csv(trained_pipeline.backtest_csv)
    assert backtest["forecast_xgboost"].isna().sum() == 0
    assert backtest["forecast_seasonal_naive"].isna().sum() == 0
    assert (backtest["forecast_xgboost"] >= 0).all()


def test_trending_up_sku_forecast_trends_upward(trained_pipeline):
    """A SKU with a strong synthetic upward trend should have its recursive
    forecast trend upward too, not just its accuracy metric look okay."""
    import demand_forecast.config as cfg

    sku_meta = pd.read_csv(cfg.PATHS.sku_metadata_csv)
    backtest = pd.read_csv(trained_pipeline.backtest_csv, parse_dates=["date"])

    up_skus = sku_meta[sku_meta["trend_type"] == "up"]["sku_id"]
    assert len(up_skus) > 0

    # aggregate across all "up" SKUs' forecasts to get a stable trend signal
    up_backtest = backtest[backtest["sku_id"].isin(up_skus)].sort_values("date")
    daily_total = up_backtest.groupby("date")["forecast_xgboost"].sum()
    first_half = daily_total.iloc[: len(daily_total) // 2].mean()
    second_half = daily_total.iloc[len(daily_total) // 2 :].mean()
    assert second_half >= first_half * 0.95  # forecast should not be trending down
