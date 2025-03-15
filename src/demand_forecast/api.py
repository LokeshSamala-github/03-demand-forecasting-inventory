"""FastAPI serving layer.

Run with:
    uvicorn demand_forecast.api:app --reload

Endpoints:
    GET  /health                liveness probe
    GET  /skus                  catalog of SKUs the model knows about
    GET  /forecast/{sku_id}      recursive N-day-ahead demand forecast
    GET  /reorder/{sku_id}       safety stock / reorder point / EOQ recommendation

Both /forecast and /reorder share one loaded model and one loaded sales
history (see the `lifespan` context manager), and /reorder reuses the exact
same `optimize.py` functions the CLI pipeline uses -- no duplicated formulas
between "the report" and "the API".
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from demand_forecast import __version__
from demand_forecast.config import OPTIMIZE_CONFIG, PATHS
from demand_forecast.model import recursive_forecast
from demand_forecast.optimize import backtest_residual_std, build_recommendations, load_model_and_manifest
from demand_forecast.schemas import ForecastPoint, ForecastResponse, HealthResponse, ReorderResponse, SkuInfo

logger = logging.getLogger(__name__)

MAX_HORIZON_DAYS = 90


class ModelStore:
    """Everything the endpoints need, loaded once at startup instead of
    per-request (re-reading CSVs and re-loading an XGBoost booster on every
    call would make even a demo API annoyingly slow)."""

    model = None
    categories: dict | None = None
    sales_df: pd.DataFrame | None = None
    sku_meta_df: pd.DataFrame | None = None
    residual_std: pd.Series | None = None

    @property
    def loaded(self) -> bool:
        return self.model is not None


store = ModelStore()


def _load() -> None:
    store.model, store.categories = load_model_and_manifest()
    store.sales_df = pd.read_csv(PATHS.raw_sales_csv, parse_dates=["date"])
    store.sku_meta_df = pd.read_csv(PATHS.sku_metadata_csv)
    backtest_df = pd.read_csv(PATHS.backtest_csv, parse_dates=["date"])
    store.residual_std = backtest_residual_std(backtest_df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load()
        logger.info("model + data loaded")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("startup load skipped: %s", exc)
    yield


app = FastAPI(
    title="Demand Forecasting + Inventory Optimizer API",
    description="Per-SKU demand forecasts and forecast-uncertainty-aware reorder recommendations.",
    version=__version__,
    lifespan=lifespan,
)


def _require_sku(sku_id: str) -> None:
    if not store.loaded:
        raise HTTPException(status_code=503, detail="model/data not loaded yet -- run the training pipeline first")
    if sku_id not in set(store.sku_meta_df["sku_id"]):
        raise HTTPException(status_code=404, detail=f"unknown sku_id '{sku_id}'")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse | JSONResponse:
    if store.loaded:
        return HealthResponse(status="ok", model_loaded=True)
    return JSONResponse(status_code=503, content={"status": "degraded", "model_loaded": False})


@app.get("/skus", response_model=list[SkuInfo])
def list_skus() -> list[SkuInfo]:
    if not store.loaded:
        raise HTTPException(status_code=503, detail="model/data not loaded yet")
    return [SkuInfo(**row) for row in store.sku_meta_df[["sku_id", "category", "lead_time_days"]].to_dict("records")]


@app.get("/forecast/{sku_id}", response_model=ForecastResponse)
def forecast(sku_id: str, horizon_days: int = Query(14, ge=1, le=MAX_HORIZON_DAYS)) -> ForecastResponse:
    _require_sku(sku_id)
    last_date = store.sales_df["date"].max()
    forecast_dates = list(pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D"))
    pred = recursive_forecast(store.model, store.sales_df, store.categories, forecast_dates)
    sku_pred = pred[pred["sku_id"] == sku_id].sort_values("date")
    category = store.sku_meta_df.set_index("sku_id").loc[sku_id, "category"]
    return ForecastResponse(
        sku_id=sku_id,
        category=category,
        horizon_days=horizon_days,
        forecast_from=(last_date + pd.Timedelta(days=1)).date(),
        points=[ForecastPoint(date=row.date.date(), forecast_units=round(row.forecast, 2)) for row in sku_pred.itertuples()],
    )


@app.get("/reorder/{sku_id}", response_model=ReorderResponse)
def reorder(sku_id: str, service_level: float = Query(OPTIMIZE_CONFIG.service_level, ge=0.5, lt=1.0)) -> ReorderResponse:
    _require_sku(sku_id)
    from demand_forecast.optimize import forward_avg_daily_demand

    sku_meta_row = store.sku_meta_df[store.sku_meta_df["sku_id"] == sku_id]
    avg_daily_demand = forward_avg_daily_demand(store.model, store.sales_df, sku_meta_row, store.categories)
    recs = build_recommendations(sku_meta_row, avg_daily_demand, store.residual_std, service_level)
    row = recs.iloc[0]
    return ReorderResponse(
        sku_id=row["sku_id"],
        category=row["category"],
        lead_time_days=int(row["lead_time_days"]),
        service_level=service_level,
        avg_daily_demand_forecast=float(row["avg_daily_demand_forecast"]),
        forecast_daily_std=float(row["forecast_daily_std"]),
        safety_stock=float(row["safety_stock"]),
        reorder_point=float(row["reorder_point"]),
        economic_order_qty=float(row["economic_order_qty"]),
        unit_cost=float(row["unit_cost"]),
    )
