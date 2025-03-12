"""Inventory optimization: safety stock, reorder point, and EOQ, computed
from the forecaster's own point estimate and its own backtested uncertainty.

Usage:
    python -m demand_forecast.optimize

This is the part that actually connects forecasting to a business decision.
A hardcoded "keep 20% extra stock as a buffer" is not optimization, it just
hides the forecast's accuracy (or lack of it) behind a fixed number. Instead:

- **Safety stock** uses the classic formula `z * sigma_LT`, where `sigma_LT`
  is the demand standard deviation *during the lead time*, built from
  `sigma`, the model's own out-of-sample daily forecast-error standard
  deviation on the backtest window (`reports/backtest_predictions.csv`) --
  not the raw historical demand std dev, which would include seasonal and
  promotional variation the model already explains. A SKU the model forecasts
  well (small backtest residual std) gets a small safety stock; a SKU the
  model forecasts poorly gets a bigger one, automatically -- see
  `tests/test_optimize.py::test_safety_stock_increases_with_forecast_uncertainty`.
- **Reorder point** = expected demand during the lead time (from the
  forecaster's forward-looking point forecast) + safety stock.
- **EOQ** (economic order quantity) trades ordering cost against holding
  cost using the standard `sqrt(2 * D * S / H)` formula, with per-SKU cost
  assumptions from `data/generate_data.py` (documented there and in the
  README) standing in for a real ERP's cost master data.
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm

from demand_forecast.config import OPTIMIZE_CONFIG, PATHS
from demand_forecast.model import recursive_forecast


def z_score(service_level: float) -> float:
    """Standard-normal z for a target in-stock (service level) probability,
    e.g. z(0.95) ~= 1.645."""
    if not 0.5 <= service_level < 1.0:
        raise ValueError("service_level should be a probability in [0.5, 1.0)")
    return float(norm.ppf(service_level))


def safety_stock(daily_demand_std: float, lead_time_days: float, service_level: float) -> float:
    """z * sigma_LT, with sigma_LT = daily_demand_std * sqrt(lead_time_days)
    (demand on different days assumed independent, so variance during the
    lead time scales linearly with lead time -- the standard textbook
    assumption absent day-to-day demand correlation data)."""
    if daily_demand_std < 0 or lead_time_days < 0:
        raise ValueError("daily_demand_std and lead_time_days must be non-negative")
    z = z_score(service_level)
    return z * daily_demand_std * math.sqrt(lead_time_days)


def reorder_point(avg_daily_demand: float, lead_time_days: float, safety_stock_units: float) -> float:
    return avg_daily_demand * lead_time_days + safety_stock_units


def economic_order_quantity(annual_demand: float, ordering_cost: float, holding_cost_per_unit_per_year: float) -> float:
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit_per_year <= 0:
        return 0.0
    return math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit_per_year)


def load_model_and_manifest() -> tuple[xgb.XGBRegressor, dict]:
    import json

    model = xgb.XGBRegressor()
    model.load_model(str(PATHS.model_file))
    with open(PATHS.feature_manifest) as f:
        manifest = json.load(f)
    return model, manifest["categories"]


def backtest_residual_std(backtest_df: pd.DataFrame) -> pd.Series:
    """Per-SKU standard deviation of (actual - forecast) over the held-out
    backtest window -- the forecaster's own honest uncertainty estimate."""
    residual = backtest_df["units"] - backtest_df["forecast_xgboost"]
    return residual.groupby(backtest_df["sku_id"]).std().fillna(0.0)


def forward_avg_daily_demand(
    model: xgb.XGBRegressor,
    sales_df: pd.DataFrame,
    sku_meta_df: pd.DataFrame,
    categories: dict,
) -> pd.Series:
    """Forecasts forward from the end of the observed data out to the
    longest lead time in the SKU catalog, then averages each SKU's forecast
    over *its own* lead-time window -- the "expected demand during the time
    it takes to receive a new order" that both the reorder point and EOQ
    need."""
    horizon = int(sku_meta_df["lead_time_days"].max())
    last_date = sales_df["date"].max()
    forecast_dates = list(pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D"))
    pred = recursive_forecast(model, sales_df, categories, forecast_dates)

    avg_by_sku = {}
    lead_time_by_sku = sku_meta_df.set_index("sku_id")["lead_time_days"]
    for sku_id, lt in lead_time_by_sku.items():
        window = pred[pred["sku_id"] == sku_id].sort_values("date").head(int(lt))
        avg_by_sku[sku_id] = float(window["forecast"].mean()) if len(window) else 0.0
    return pd.Series(avg_by_sku, name="avg_daily_demand_forecast")


def build_recommendations(
    sku_meta_df: pd.DataFrame,
    avg_daily_demand: pd.Series,
    residual_std: pd.Series,
    service_level: float = OPTIMIZE_CONFIG.service_level,
) -> pd.DataFrame:
    df = sku_meta_df.set_index("sku_id").copy()
    df["avg_daily_demand_forecast"] = avg_daily_demand.reindex(df.index)
    fallback_std = float(residual_std.replace(0, np.nan).median()) if residual_std.notna().any() else 0.0
    df["forecast_daily_std"] = residual_std.reindex(df.index).fillna(fallback_std)
    df["service_level"] = service_level
    df["z_score"] = z_score(service_level)

    df["safety_stock"] = df.apply(
        lambda r: safety_stock(r["forecast_daily_std"], r["lead_time_days"], service_level), axis=1
    )
    df["reorder_point"] = df.apply(
        lambda r: reorder_point(r["avg_daily_demand_forecast"], r["lead_time_days"], r["safety_stock"]), axis=1
    )
    df["annual_demand_forecast"] = df["avg_daily_demand_forecast"] * 365
    df["holding_cost_per_unit_per_year"] = df["unit_cost"] * df["holding_cost_rate"]
    df["economic_order_qty"] = df.apply(
        lambda r: economic_order_quantity(
            r["annual_demand_forecast"], r["ordering_cost"], r["holding_cost_per_unit_per_year"]
        ),
        axis=1,
    )
    for col in ["avg_daily_demand_forecast", "forecast_daily_std", "safety_stock", "reorder_point", "economic_order_qty"]:
        df[col] = df[col].round(2)
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-level", type=float, default=OPTIMIZE_CONFIG.service_level)
    args = parser.parse_args()

    sales_df = pd.read_csv(PATHS.raw_sales_csv, parse_dates=["date"])
    sku_meta_df = pd.read_csv(PATHS.sku_metadata_csv)
    backtest_df = pd.read_csv(PATHS.backtest_csv, parse_dates=["date"])

    model, categories = load_model_and_manifest()
    residual_std = backtest_residual_std(backtest_df)
    avg_daily_demand = forward_avg_daily_demand(model, sales_df, sku_meta_df, categories)

    recs = build_recommendations(sku_meta_df, avg_daily_demand, residual_std, args.service_level)
    PATHS.reorder_csv.parent.mkdir(parents=True, exist_ok=True)
    recs.to_csv(PATHS.reorder_csv, index=False)

    print(f"Wrote reorder recommendations for {len(recs)} SKUs -> {PATHS.reorder_csv}")
    print(recs[["sku_id", "avg_daily_demand_forecast", "safety_stock", "reorder_point", "economic_order_qty"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
