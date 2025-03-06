"""Demand forecasting + inventory optimization.

A self-contained pipeline from raw daily SKU sales to a served reorder
recommendation:

    generate_data.py -> features.py (calendar + lag features)
                      -> model.py (global XGBoost forecaster, recursive backtest)
                      -> forecast.py (train, evaluate vs. seasonal-naive baseline)
                      -> optimize.py (safety stock / reorder point / EOQ from
                                       the forecast's own uncertainty)
                      -> api.py (FastAPI serving)

Every stage is a plain, importable module so it can be run as a script or
wired into an orchestrator without modification.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("demand-forecast-optimizer")
except PackageNotFoundError:  # running from source without an editable install
    __version__ = "0.0.0+dev"

__all__ = ["__version__"]
