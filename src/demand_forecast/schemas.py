"""Pydantic request/response models for the API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


class ForecastPoint(BaseModel):
    date: date
    forecast_units: float


class ForecastResponse(BaseModel):
    sku_id: str
    category: str
    horizon_days: int
    forecast_from: date
    points: list[ForecastPoint]


class ReorderResponse(BaseModel):
    sku_id: str
    category: str
    lead_time_days: int
    service_level: float = Field(description="Target probability of not stocking out during the lead time")
    avg_daily_demand_forecast: float
    forecast_daily_std: float = Field(description="Backtested daily forecast-error std dev for this SKU")
    safety_stock: float
    reorder_point: float
    economic_order_qty: float
    unit_cost: float


class SkuInfo(BaseModel):
    sku_id: str
    category: str
    lead_time_days: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
