"""API tests. Depends on `trained_pipeline` so the app's lifespan loads a
real model + real data, not mocks."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(trained_pipeline):
    from demand_forecast.api import app

    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_list_skus(client):
    resp = client.get("/skus")
    assert resp.status_code == 200
    skus = resp.json()
    assert len(skus) > 0
    assert {"sku_id", "category", "lead_time_days"} <= set(skus[0].keys())


def test_forecast_known_sku(client):
    sku_id = client.get("/skus").json()[0]["sku_id"]
    resp = client.get(f"/forecast/{sku_id}", params={"horizon_days": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sku_id"] == sku_id
    assert body["horizon_days"] == 10
    assert len(body["points"]) == 10
    assert all(p["forecast_units"] >= 0 for p in body["points"])


def test_forecast_unknown_sku_404(client):
    resp = client.get("/forecast/NOT-A-REAL-SKU")
    assert resp.status_code == 404


def test_forecast_horizon_bounds_enforced(client):
    sku_id = client.get("/skus").json()[0]["sku_id"]
    resp = client.get(f"/forecast/{sku_id}", params={"horizon_days": 0})
    assert resp.status_code == 422
    resp = client.get(f"/forecast/{sku_id}", params={"horizon_days": 10000})
    assert resp.status_code == 422


def test_reorder_known_sku(client):
    sku_id = client.get("/skus").json()[0]["sku_id"]
    resp = client.get(f"/reorder/{sku_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sku_id"] == sku_id
    assert body["safety_stock"] >= 0
    assert body["reorder_point"] >= body["safety_stock"]
    assert body["economic_order_qty"] > 0


def test_reorder_respects_service_level_query_param(client):
    sku_id = client.get("/skus").json()[0]["sku_id"]
    low = client.get(f"/reorder/{sku_id}", params={"service_level": 0.80}).json()
    high = client.get(f"/reorder/{sku_id}", params={"service_level": 0.99}).json()
    assert high["safety_stock"] >= low["safety_stock"]


def test_reorder_unknown_sku_404(client):
    resp = client.get("/reorder/NOT-A-REAL-SKU")
    assert resp.status_code == 404
