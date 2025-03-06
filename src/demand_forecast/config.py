"""Central configuration.

A dataclass, not a settings framework, because the whole point is that every
path and hyperparameter used by every stage is declared in exactly one place.
Every path is overridable via an environment variable of the same name
(see `_env_path`) -- this is how the test suite points the whole pipeline at
a throwaway temp directory (see tests/conftest.py) without touching the real
data/, models/, reports/ folders, and it's the same mechanism a real
deployment would use to point at a different data volume.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_sales_csv: Path = field(
        default_factory=lambda: _env_path("DF_RAW_SALES_CSV", PROJECT_ROOT / "data" / "raw" / "sales.csv")
    )
    sku_metadata_csv: Path = field(
        default_factory=lambda: _env_path(
            "DF_SKU_METADATA_CSV", PROJECT_ROOT / "data" / "raw" / "sku_metadata.csv"
        )
    )
    model_file: Path = field(
        default_factory=lambda: _env_path("DF_MODEL_FILE", PROJECT_ROOT / "models" / "forecast_model.json")
    )
    feature_manifest: Path = field(
        default_factory=lambda: _env_path(
            "DF_FEATURE_MANIFEST", PROJECT_ROOT / "models" / "feature_manifest.json"
        )
    )
    metrics_file: Path = field(
        default_factory=lambda: _env_path("DF_METRICS_FILE", PROJECT_ROOT / "reports" / "metrics.json")
    )
    backtest_csv: Path = field(
        default_factory=lambda: _env_path(
            "DF_BACKTEST_CSV", PROJECT_ROOT / "reports" / "backtest_predictions.csv"
        )
    )
    forecast_plot_png: Path = field(
        default_factory=lambda: _env_path(
            "DF_FORECAST_PLOT_PNG", PROJECT_ROOT / "reports" / "forecast_vs_actual.png"
        )
    )
    reorder_csv: Path = field(
        default_factory=lambda: _env_path(
            "DF_REORDER_CSV", PROJECT_ROOT / "reports" / "reorder_recommendations.csv"
        )
    )


@dataclass(frozen=True)
class ForecastConfig:
    """Split, feature and model hyperparameters."""

    test_days: int = 56  # ~8 weeks held out at the end of the series, per SKU
    lags: tuple[int, ...] = (7, 14, 21, 28)
    rolling_windows: tuple[int, ...] = (7, 28)
    min_history_days: int = 35  # rows dropped until this many lag days exist
    xgb_params: dict = field(
        default_factory=lambda: {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 400,
            "subsample": 0.85,
            "colsample_bytree": 0.8,
            "min_child_weight": 4,
            "reg_lambda": 1.2,
            "random_state": 42,
            "n_jobs": -1,
            "enable_categorical": True,
        }
    )


@dataclass(frozen=True)
class OptimizeConfig:
    """Inventory-optimization assumptions not already carried per-SKU in
    sku_metadata.csv (see data/generate_data.py for cost/lead-time
    assumptions, which ARE per-SKU)."""

    service_level: float = 0.95  # target probability of not stocking out during lead time


PATHS = Paths()
FORECAST_CONFIG = ForecastConfig()
OPTIMIZE_CONFIG = OptimizeConfig()
