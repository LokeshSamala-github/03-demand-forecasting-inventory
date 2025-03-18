"""Shared pytest fixtures.

The whole test session runs against a throwaway project directory instead of
the real data/, models/, reports/ folders. This works by setting the DF_*
environment variables that `demand_forecast.config.Paths` already supports
(see config.py's `_env_path` helper) *before* anything in demand_forecast
gets imported -- conftest.py is guaranteed to run before any test module's
imports, so every `Paths()` instance constructed anywhere in the app picks
up the temp location automatically. No per-module monkeypatch needed, and
it's the same override mechanism a real deployment would use to point at a
different data directory.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (str(SRC), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

_TMP = Path(tempfile.mkdtemp(prefix="demand_forecast_tests_"))
os.environ.setdefault("DF_RAW_SALES_CSV", str(_TMP / "data" / "raw" / "sales.csv"))
os.environ.setdefault("DF_SKU_METADATA_CSV", str(_TMP / "data" / "raw" / "sku_metadata.csv"))
os.environ.setdefault("DF_MODEL_FILE", str(_TMP / "models" / "forecast_model.json"))
os.environ.setdefault("DF_FEATURE_MANIFEST", str(_TMP / "models" / "feature_manifest.json"))
os.environ.setdefault("DF_METRICS_FILE", str(_TMP / "reports" / "metrics.json"))
os.environ.setdefault("DF_BACKTEST_CSV", str(_TMP / "reports" / "backtest_predictions.csv"))
os.environ.setdefault("DF_FORECAST_PLOT_PNG", str(_TMP / "reports" / "forecast_vs_actual.png"))
os.environ.setdefault("DF_REORDER_CSV", str(_TMP / "reports" / "reorder_recommendations.csv"))

import pytest  # noqa: E402  (import order is deliberate -- env vars must be set first)

from demand_forecast.config import PATHS  # noqa: E402

TEST_N_DAYS = 420  # ~14 months: enough for weekly + a bit of annual signal, fast to train
TEST_HOLDOUT_DAYS = 21


@pytest.fixture(scope="session", autouse=True)
def _raw_data() -> Path:
    """Generate a small synthetic dataset once per test session."""
    from data.generate_data import generate

    sales_df, sku_meta_df = generate(n_days=TEST_N_DAYS, seed=7)
    PATHS.raw_sales_csv.parent.mkdir(parents=True, exist_ok=True)
    sales_df.to_csv(PATHS.raw_sales_csv, index=False)
    sku_meta_df.to_csv(PATHS.sku_metadata_csv, index=False)
    return PATHS.raw_sales_csv


@pytest.fixture(scope="session")
def trained_pipeline(_raw_data):
    """Actually run the real forecast + optimize pipeline once (small data,
    short holdout) so downstream tests exercise genuine trained-model and
    computed-recommendation behaviour rather than mocks."""
    argv_backup = sys.argv
    try:
        from demand_forecast import forecast as forecast_mod
        from demand_forecast import optimize as optimize_mod

        sys.argv = ["forecast.py", "--test-days", str(TEST_HOLDOUT_DAYS)]
        forecast_mod.main()

        sys.argv = ["optimize.py"]
        optimize_mod.main()
    finally:
        sys.argv = argv_backup
    return PATHS
