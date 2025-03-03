"""Synthetic multi-SKU daily demand generator.

Builds a daily sales history for ~25 SKUs across 5 categories, each with a
genuinely different demand *shape* (not a scaled copy of one master curve):

- a per-SKU trend (up / down / flat), applied multiplicatively and clipped so
  it can't run away to zero or infinity over a multi-year history
- a weekly seasonality profile (weekend-heavy, weekday-heavy, or flat)
  particular to the SKU's category
- an annual seasonality bump (a Gaussian hump around a category-specific peak
  day-of-year -- e.g. beverages peak in summer, electronics accessories peak
  around the winter holidays)
- random promotional days (a `promo_flag` feature the forecaster can use)
  with a multiplicative spike
- realistic count-data noise: a negative binomial draw around the day's
  expected demand (mean = the deterministic signal above), not Gaussian noise
  bolted onto a formula, so low-volume SKUs show the right kind of
  over-dispersion and near-zero days actually occur.

Also writes `sku_metadata.csv`: the per-SKU business assumptions (unit cost,
holding cost rate, ordering cost, lead time) that the inventory-optimization
layer needs and that a real company would pull from its ERP.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class CategoryProfile:
    """Category-level defaults; individual SKUs jitter around these so no
    two SKUs in a category are identical, but the category still has a
    recognizable shape."""

    name: str
    base_demand_range: tuple[float, float]
    trend_types: tuple[str, ...]  # which trend types this category's SKUs may draw
    weekly_pattern: str  # "weekend_heavy" | "weekday_heavy" | "flat"
    annual_peak_doy: float | None  # day-of-year of the seasonal peak, or None
    annual_amplitude_range: tuple[float, float]
    annual_width: float
    promo_prob_range: tuple[float, float]
    dispersion_range: tuple[float, float]  # negative-binomial "r" -- smaller = noisier
    unit_cost_range: tuple[float, float]
    holding_cost_rate_range: tuple[float, float]  # annual, fraction of unit cost
    ordering_cost_range: tuple[float, float]
    lead_time_days_range: tuple[int, int]

    def __repr__(self) -> str:  # noqa: D401 -- terse debug repr, not prose
        return f"CategoryProfile({self.name})"


CATEGORIES: list[CategoryProfile] = [
    CategoryProfile(
        name="Beverages",
        base_demand_range=(40, 140),
        trend_types=("flat", "up"),
        weekly_pattern="weekend_heavy",
        annual_peak_doy=200,  # mid-July
        annual_amplitude_range=(0.35, 0.65),
        annual_width=45,
        promo_prob_range=(0.04, 0.08),
        dispersion_range=(4, 9),
        unit_cost_range=(1.5, 4.0),
        holding_cost_rate_range=(0.22, 0.28),
        ordering_cost_range=(25, 50),
        lead_time_days_range=(3, 7),
    ),
    CategoryProfile(
        name="Snacks",
        base_demand_range=(30, 110),
        trend_types=("flat", "up", "down"),
        weekly_pattern="weekend_heavy",
        annual_peak_doy=350,  # late December
        annual_amplitude_range=(0.2, 0.45),
        annual_width=25,
        promo_prob_range=(0.08, 0.14),
        dispersion_range=(3, 7),
        unit_cost_range=(2.0, 6.0),
        holding_cost_rate_range=(0.20, 0.26),
        ordering_cost_range=(20, 45),
        lead_time_days_range=(2, 6),
    ),
    CategoryProfile(
        name="Household",
        base_demand_range=(15, 60),
        trend_types=("flat",),
        weekly_pattern="weekday_heavy",
        annual_peak_doy=None,
        annual_amplitude_range=(0.0, 0.08),
        annual_width=40,
        promo_prob_range=(0.02, 0.05),
        dispersion_range=(6, 14),
        unit_cost_range=(4.0, 12.0),
        holding_cost_rate_range=(0.18, 0.24),
        ordering_cost_range=(15, 35),
        lead_time_days_range=(4, 10),
    ),
    CategoryProfile(
        name="Electronics-Accessories",
        base_demand_range=(8, 35),
        trend_types=("up",),
        weekly_pattern="weekend_heavy",
        annual_peak_doy=335,  # early December (holiday shopping)
        annual_amplitude_range=(0.8, 1.6),
        annual_width=30,
        promo_prob_range=(0.03, 0.06),
        dispersion_range=(2, 5),
        unit_cost_range=(12.0, 60.0),
        holding_cost_rate_range=(0.25, 0.32),
        ordering_cost_range=(60, 140),
        lead_time_days_range=(10, 21),
    ),
    CategoryProfile(
        name="Personal-Care",
        base_demand_range=(12, 45),
        trend_types=("flat", "down"),
        weekly_pattern="flat",
        annual_peak_doy=10,  # New Year resolution bump
        annual_amplitude_range=(0.15, 0.3),
        annual_width=15,
        promo_prob_range=(0.03, 0.06),
        dispersion_range=(5, 11),
        unit_cost_range=(3.0, 15.0),
        holding_cost_rate_range=(0.20, 0.26),
        ordering_cost_range=(20, 45),
        lead_time_days_range=(5, 12),
    ),
]

SKUS_PER_CATEGORY = 5
TREND_SLOPE_RANGE = {"up": (0.00035, 0.00075), "down": (-0.00065, -0.00025), "flat": (-0.00008, 0.00008)}

WEEKLY_PATTERNS = {
    # index 0 = Monday ... 6 = Sunday
    "weekend_heavy": np.array([0.85, 0.85, 0.88, 0.95, 1.10, 1.45, 1.35]),
    "weekday_heavy": np.array([1.15, 1.20, 1.18, 1.15, 1.05, 0.70, 0.65]),
    "flat": np.array([1.0, 1.02, 1.0, 0.99, 1.02, 0.98, 0.99]),
}


def _annual_multiplier(doy: np.ndarray, peak_doy: float | None, amplitude: float, width: float) -> np.ndarray:
    if peak_doy is None or amplitude <= 0:
        return np.ones_like(doy, dtype=float)
    # wrap-around distance so a peak near day 1 or day 365 is handled correctly
    diff = np.minimum(np.abs(doy - peak_doy), 365 - np.abs(doy - peak_doy))
    return 1.0 + amplitude * np.exp(-0.5 * (diff / width) ** 2)


def generate(
    start_date: str = "2023-01-01",
    n_days: int = 913,  # ~2.5 years
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sales_df, sku_metadata_df)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, periods=n_days, freq="D")
    dow = dates.dayofweek.to_numpy()
    doy = dates.dayofyear.to_numpy().astype(float)
    days_since_start = np.arange(n_days, dtype=float)

    sku_rows = []
    sales_frames = []
    sku_counter = 0
    for cat in CATEGORIES:
        prefix = "".join(w[0] for w in cat.name.split("-")).upper()
        for i in range(SKUS_PER_CATEGORY):
            sku_counter += 1
            sku_id = f"SKU-{prefix}-{i + 1:02d}"
            base_demand = rng.uniform(*cat.base_demand_range)
            trend_type = cat.trend_types[rng.integers(len(cat.trend_types))]
            trend_slope = rng.uniform(*TREND_SLOPE_RANGE[trend_type])
            amplitude = rng.uniform(*cat.annual_amplitude_range)
            promo_prob = rng.uniform(*cat.promo_prob_range)
            dispersion_r = rng.uniform(*cat.dispersion_range)
            unit_cost = round(rng.uniform(*cat.unit_cost_range), 2)
            holding_cost_rate = round(rng.uniform(*cat.holding_cost_rate_range), 3)
            ordering_cost = round(rng.uniform(*cat.ordering_cost_range), 2)
            lead_time_days = int(rng.integers(cat.lead_time_days_range[0], cat.lead_time_days_range[1] + 1))

            trend_factor = np.clip(1.0 + trend_slope * days_since_start, 0.3, 3.0)
            weekly_mult = WEEKLY_PATTERNS[cat.weekly_pattern][dow]
            annual_mult = _annual_multiplier(doy, cat.annual_peak_doy, amplitude, cat.annual_width)

            promo_flag = (rng.random(n_days) < promo_prob).astype(int)
            promo_mult = np.where(promo_flag == 1, rng.uniform(1.5, 3.0, size=n_days), 1.0)

            lam = np.clip(base_demand * trend_factor * weekly_mult * annual_mult * promo_mult, 0.3, None)

            # Negative-binomial count noise: mean = lam, variance = lam + lam^2/r
            # (over-dispersed relative to Poisson, which is the realistic shape
            # for retail demand -- more so for slower-moving, higher-dispersion SKUs).
            p = dispersion_r / (dispersion_r + lam)
            units = rng.negative_binomial(dispersion_r, p)

            df = pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": sku_id,
                    "category": cat.name,
                    "units": units,
                    "promo_flag": promo_flag,
                }
            )
            sales_frames.append(df)
            sku_rows.append(
                {
                    "sku_id": sku_id,
                    "category": cat.name,
                    "trend_type": trend_type,
                    "unit_cost": unit_cost,
                    "holding_cost_rate": holding_cost_rate,
                    "ordering_cost": ordering_cost,
                    "lead_time_days": lead_time_days,
                }
            )

    sales_df = pd.concat(sales_frames, ignore_index=True).sort_values(["sku_id", "date"]).reset_index(drop=True)
    sku_meta_df = pd.DataFrame(sku_rows)
    return sales_df, sku_meta_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--n-days", type=int, default=913)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None, help="Defaults to data/raw next to this script")
    args = parser.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        try:
            from demand_forecast.config import PATHS

            out_dir = PATHS.raw_sales_csv.parent
        except Exception:  # pragma: no cover - fallback if package not importable yet
            out_dir = PROJECT_ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    sales_df, sku_meta_df = generate(start_date=args.start_date, n_days=args.n_days, seed=args.seed)
    sales_path = out_dir / "sales.csv"
    meta_path = out_dir / "sku_metadata.csv"
    sales_df.to_csv(sales_path, index=False)
    sku_meta_df.to_csv(meta_path, index=False)
    print(f"Wrote {len(sales_df):,} rows for {sku_meta_df['sku_id'].nunique()} SKUs -> {sales_path}")
    print(f"Wrote SKU metadata ({len(sku_meta_df)} rows) -> {meta_path}")


if __name__ == "__main__":
    main()
