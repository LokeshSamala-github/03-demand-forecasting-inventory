"""Tests for the inventory-optimization layer.

The monotonic-uncertainty test is the one the project spec calls out
explicitly: a more uncertain forecast (bigger backtest residual std) must
produce a bigger safety stock, holding demand level constant, because that
is the entire point of computing safety stock from the model's own
uncertainty instead of a fixed buffer percentage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from demand_forecast.optimize import (
    backtest_residual_std,
    build_recommendations,
    economic_order_quantity,
    reorder_point,
    safety_stock,
    z_score,
)


def test_z_score_matches_known_values():
    assert math.isclose(z_score(0.5), 0.0, abs_tol=1e-9)
    assert math.isclose(z_score(0.95), 1.645, abs_tol=0.001)
    assert math.isclose(z_score(0.99), 2.326, abs_tol=0.001)


def test_safety_stock_increases_with_forecast_uncertainty():
    """Core requirement: holding lead time and service level constant,
    increasing forecast-error std dev must strictly increase safety stock."""
    lead_time_days = 7
    service_level = 0.95
    stds = [1.0, 2.0, 5.0, 10.0, 20.0]
    values = [safety_stock(s, lead_time_days, service_level) for s in stds]
    assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))


def test_safety_stock_zero_uncertainty_is_zero():
    assert safety_stock(0.0, lead_time_days=10, service_level=0.95) == 0.0


def test_safety_stock_increases_with_lead_time():
    short = safety_stock(daily_demand_std=5.0, lead_time_days=3, service_level=0.95)
    long = safety_stock(daily_demand_std=5.0, lead_time_days=21, service_level=0.95)
    assert long > short


def test_safety_stock_increases_with_service_level():
    lo = safety_stock(daily_demand_std=5.0, lead_time_days=7, service_level=0.90)
    hi = safety_stock(daily_demand_std=5.0, lead_time_days=7, service_level=0.99)
    assert hi > lo


def test_reorder_point_equals_lead_time_demand_plus_safety_stock():
    ss = safety_stock(4.0, 7, 0.95)
    rp = reorder_point(avg_daily_demand=20.0, lead_time_days=7, safety_stock_units=ss)
    assert math.isclose(rp, 20.0 * 7 + ss)


def test_eoq_increases_with_ordering_cost():
    small = economic_order_quantity(annual_demand=3650, ordering_cost=20, holding_cost_per_unit_per_year=2)
    large = economic_order_quantity(annual_demand=3650, ordering_cost=200, holding_cost_per_unit_per_year=2)
    assert large > small


def test_eoq_decreases_with_holding_cost():
    low_holding = economic_order_quantity(annual_demand=3650, ordering_cost=50, holding_cost_per_unit_per_year=1)
    high_holding = economic_order_quantity(annual_demand=3650, ordering_cost=50, holding_cost_per_unit_per_year=10)
    assert high_holding < low_holding


def test_eoq_zero_demand_is_zero():
    assert economic_order_quantity(0, 50, 2) == 0.0


def test_z_score_rejects_out_of_range_service_level():
    with pytest.raises(ValueError):
        z_score(1.0)
    with pytest.raises(ValueError):
        z_score(0.3)


def test_build_recommendations_pure_function():
    sku_meta = pd.DataFrame(
        {
            "sku_id": ["LOW-UNCERTAINTY", "HIGH-UNCERTAINTY"],
            "category": ["Cat", "Cat"],
            "unit_cost": [10.0, 10.0],
            "holding_cost_rate": [0.25, 0.25],
            "ordering_cost": [40.0, 40.0],
            "lead_time_days": [7, 7],
        }
    )
    avg_daily_demand = pd.Series({"LOW-UNCERTAINTY": 20.0, "HIGH-UNCERTAINTY": 20.0})
    residual_std = pd.Series({"LOW-UNCERTAINTY": 1.0, "HIGH-UNCERTAINTY": 15.0})

    recs = build_recommendations(sku_meta, avg_daily_demand, residual_std, service_level=0.95).set_index("sku_id")

    # same demand level, same lead time -> reorder point differs only via safety stock
    assert recs.loc["HIGH-UNCERTAINTY", "safety_stock"] > recs.loc["LOW-UNCERTAINTY", "safety_stock"]
    assert recs.loc["HIGH-UNCERTAINTY", "reorder_point"] > recs.loc["LOW-UNCERTAINTY", "reorder_point"]
    # EOQ depends only on demand/cost assumptions, which are identical here
    assert math.isclose(recs.loc["HIGH-UNCERTAINTY", "economic_order_qty"], recs.loc["LOW-UNCERTAINTY", "economic_order_qty"])


def test_backtest_residual_std_computed_per_sku():
    backtest = pd.DataFrame(
        {
            "sku_id": ["A", "A", "A", "B", "B", "B"],
            "units": [10, 12, 8, 100, 50, 150],
            "forecast_xgboost": [10, 10, 10, 100, 100, 100],
        }
    )
    stds = backtest_residual_std(backtest)
    assert stds["B"] > stds["A"]


# ---- End-to-end (uses the real trained pipeline from conftest.py) ----


def test_reorder_csv_has_positive_sane_values(trained_pipeline):
    recs = pd.read_csv(trained_pipeline.reorder_csv)
    assert len(recs) > 0
    assert (recs["safety_stock"] >= 0).all()
    assert (recs["reorder_point"] >= recs["safety_stock"]).all()
    assert (recs["economic_order_qty"] > 0).all()
    assert (recs["avg_daily_demand_forecast"] >= 0).all()


def test_reorder_csv_service_level_matches_config(trained_pipeline):
    import demand_forecast.config as cfg

    recs = pd.read_csv(trained_pipeline.reorder_csv)
    assert np.allclose(recs["service_level"], cfg.OPTIMIZE_CONFIG.service_level)
