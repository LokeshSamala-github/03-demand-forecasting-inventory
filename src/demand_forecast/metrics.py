"""Forecast accuracy metrics.

WAPE (weighted absolute percentage error) is the headline metric: it's
`sum(|actual - forecast|) / sum(actual)`, which stays well-behaved on
low-volume, near-zero-demand SKU-days where plain MAPE explodes (a
divide-by-near-zero on any single day can dominate a MAPE average). MAPE is
still reported for comparability with other write-ups, but WAPE is what
`forecast.py` uses to decide "did the model actually beat the baseline".
"""

from __future__ import annotations

import numpy as np


def mape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1.0) -> float:
    """Mean absolute percentage error. `eps` floors the denominator so
    zero-demand days don't produce an infinite term; this makes the number
    usably finite but is exactly why WAPE, not MAPE, is the headline metric
    for sparse count data."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    denom = np.maximum(actual, eps)
    return float(np.mean(np.abs(actual - forecast) / denom) * 100)


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    total_actual = np.sum(actual)
    if total_actual <= 0:
        return 0.0 if np.sum(np.abs(forecast)) == 0 else float("inf")
    return float(np.sum(np.abs(actual - forecast)) / total_actual * 100)


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def summarize(actual: np.ndarray, forecast: np.ndarray) -> dict:
    return {"mape": mape(actual, forecast), "wape": wape(actual, forecast), "rmse": rmse(actual, forecast)}
