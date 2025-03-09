"""Train the global forecaster, backtest it honestly, and report results.

Usage:
    python -m demand_forecast.forecast [--test-days 56]

Pipeline:
    1. Load raw daily sales + SKU metadata.
    2. Time-based split: the last `test_days` days of *each SKU's* series are
       held out. This is not a random row split -- a random split would let
       the model train on rows from the middle of the test period and "peek"
       at the future via lag features, which is exactly the kind of leakage
       that makes a forecasting eval meaningless.
    3. Build lag/rolling/calendar features on the training period only, fit
       the global XGBoost model.
    4. Recursively forecast the held-out period per SKU (see `model.py` for
       why this has to be recursive), and do the same with the seasonal-naive
       baseline.
    5. Score both with MAPE / WAPE / RMSE, per SKU and aggregated, and save
       metrics.json, backtest_predictions.csv, a forecast-vs-actual plot, and
       the fitted model + feature manifest for `optimize.py` and `api.py` to
       reuse without retraining.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from demand_forecast.baseline import seasonal_naive_forecast  # noqa: E402
from demand_forecast.config import FORECAST_CONFIG, PATHS  # noqa: E402
from demand_forecast.features import build_feature_frame, category_manifest, lag_feature_columns  # noqa: E402
from demand_forecast.metrics import summarize  # noqa: E402
from demand_forecast.model import recursive_forecast, train_model  # noqa: E402


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    sales_df = pd.read_csv(PATHS.raw_sales_csv, parse_dates=["date"])
    sku_meta_df = pd.read_csv(PATHS.sku_metadata_csv)
    return sales_df, sku_meta_df


def time_based_split(sales_df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Returns (history_up_to_cutoff, forecast_dates). `forecast_dates` are
    the last `test_days` calendar days present in the data, identical across
    every SKU (the synthetic data has no gaps, so this is safe)."""
    all_dates = sorted(sales_df["date"].unique())
    forecast_dates = list(all_dates[-test_days:])
    cutoff = forecast_dates[0]
    history = sales_df[sales_df["date"] < cutoff].copy()
    return history, forecast_dates


def evaluate_by_sku(actual_df: pd.DataFrame, pred_df: pd.DataFrame, name: str) -> pd.DataFrame:
    merged = actual_df.merge(pred_df, on=["date", "sku_id"], how="inner")
    rows = []
    for sku_id, g in merged.groupby("sku_id", observed=True):
        m = summarize(g["units"].to_numpy(), g["forecast"].to_numpy())
        m["sku_id"] = sku_id
        m["model"] = name
        rows.append(m)
    return pd.DataFrame(rows)


def plot_forecast_vs_actual(actual_df: pd.DataFrame, model_pred: pd.DataFrame, sku_ids: list[str], out_path) -> None:
    fig, axes = plt.subplots(len(sku_ids), 1, figsize=(9, 3 * len(sku_ids)), sharex=False)
    if len(sku_ids) == 1:
        axes = [axes]
    for ax, sku_id in zip(axes, sku_ids, strict=True):
        a = actual_df[actual_df["sku_id"] == sku_id].sort_values("date")
        p = model_pred[model_pred["sku_id"] == sku_id].sort_values("date")
        ax.plot(a["date"], a["units"], label="actual", color="#1f2937", linewidth=1.5)
        ax.plot(p["date"], p["forecast"], label="forecast", color="#2563eb", linewidth=1.5, linestyle="--")
        ax.set_title(sku_id, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=FORECAST_CONFIG.test_days)
    args = parser.parse_args()

    sales_df, sku_meta_df = load_raw()
    history, forecast_dates = time_based_split(sales_df, args.test_days)

    train_featured = build_feature_frame(history, sku_meta_df)
    train_featured = train_featured.dropna(subset=lag_feature_columns())
    categories = category_manifest(sales_df)

    model = train_model(train_featured, categories)

    actual_test = sales_df.merge(sku_meta_df[["sku_id"]], on="sku_id", how="inner")
    actual_test = actual_test[actual_test["date"].isin(forecast_dates)]

    future_promo = sales_df.loc[sales_df["date"].isin(forecast_dates), ["date", "sku_id", "promo_flag"]]
    model_pred = recursive_forecast(model, history, categories, forecast_dates, future_promo_flag=future_promo)
    baseline_pred = seasonal_naive_forecast(history, forecast_dates)

    model_scores = evaluate_by_sku(actual_test, model_pred, "xgboost_global")
    baseline_scores = evaluate_by_sku(actual_test, baseline_pred, "seasonal_naive")

    merged_actual = actual_test[["date", "sku_id", "units"]]
    model_overall = summarize(
        merged_actual.merge(model_pred, on=["date", "sku_id"])["units"].to_numpy(),
        merged_actual.merge(model_pred, on=["date", "sku_id"])["forecast"].to_numpy(),
    )
    baseline_overall = summarize(
        merged_actual.merge(baseline_pred, on=["date", "sku_id"])["units"].to_numpy(),
        merged_actual.merge(baseline_pred, on=["date", "sku_id"])["forecast"].to_numpy(),
    )

    metrics = {
        "test_days": args.test_days,
        "n_skus": int(sales_df["sku_id"].nunique()),
        "n_train_rows": int(len(train_featured)),
        "overall": {"xgboost_global": model_overall, "seasonal_naive": baseline_overall},
        "wape_improvement_pct": round(
            100 * (baseline_overall["wape"] - model_overall["wape"]) / baseline_overall["wape"], 2
        )
        if baseline_overall["wape"] > 0
        else None,
        "by_sku": {
            "xgboost_global": model_scores.set_index("sku_id")[["mape", "wape", "rmse"]].to_dict(orient="index"),
            "seasonal_naive": baseline_scores.set_index("sku_id")[["mape", "wape", "rmse"]].to_dict(orient="index"),
        },
    }

    PATHS.metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with open(PATHS.metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    backtest_out = actual_test[["date", "sku_id", "units"]].merge(
        model_pred.rename(columns={"forecast": "forecast_xgboost"}), on=["date", "sku_id"]
    )
    backtest_out = backtest_out.merge(
        baseline_pred.rename(columns={"forecast": "forecast_seasonal_naive"}), on=["date", "sku_id"]
    )
    PATHS.backtest_csv.parent.mkdir(parents=True, exist_ok=True)
    backtest_out.to_csv(PATHS.backtest_csv, index=False)

    sample_skus = list(sku_meta_df.sort_values("sku_id")["sku_id"].unique()[:2]) + list(
        sku_meta_df[sku_meta_df["trend_type"] == "up"]["sku_id"].iloc[:1]
    ) + list(sku_meta_df[sku_meta_df["category"] == "Electronics-Accessories"]["sku_id"].iloc[:1])
    sample_skus = list(dict.fromkeys(sample_skus))[:4]
    plot_forecast_vs_actual(actual_test, model_pred, sample_skus, PATHS.forecast_plot_png)

    PATHS.model_file.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(PATHS.model_file))
    manifest = {"categories": categories, "feature_columns": list(train_featured.columns)}
    with open(PATHS.feature_manifest, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Train rows: {len(train_featured):,} | Test window: {forecast_dates[0].date()} .. {forecast_dates[-1].date()}")
    print(f"XGBoost   -- WAPE {model_overall['wape']:.2f}%  MAPE {model_overall['mape']:.2f}%  RMSE {model_overall['rmse']:.2f}")
    print(f"Seasonal-naive -- WAPE {baseline_overall['wape']:.2f}%  MAPE {baseline_overall['mape']:.2f}%  RMSE {baseline_overall['rmse']:.2f}")
    print(f"Saved model -> {PATHS.model_file}")
    print(f"Saved metrics -> {PATHS.metrics_file}")


if __name__ == "__main__":
    main()
