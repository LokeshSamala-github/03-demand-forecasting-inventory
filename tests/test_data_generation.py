"""Tests for the synthetic data generator: real behavioral checks, not just
"doesn't crash" -- specifically that different SKUs actually have different
demand shapes rather than being scaled copies of one curve."""

from __future__ import annotations

import numpy as np
import pandas as pd

from data.generate_data import CATEGORIES, SKUS_PER_CATEGORY, generate


def test_row_and_sku_counts():
    sales_df, sku_meta_df = generate(n_days=200, seed=1)
    expected_skus = len(CATEGORIES) * SKUS_PER_CATEGORY
    assert sku_meta_df["sku_id"].nunique() == expected_skus
    assert len(sales_df) == expected_skus * 200
    assert set(sales_df["sku_id"]) == set(sku_meta_df["sku_id"])


def test_no_negative_or_missing_units():
    sales_df, _ = generate(n_days=300, seed=2)
    assert sales_df["units"].isna().sum() == 0
    assert (sales_df["units"] >= 0).all()
    assert sales_df["units"].dtype.kind in "iu"


def test_no_date_gaps_per_sku():
    sales_df, _ = generate(n_days=250, seed=3)
    for _sku_id, g in sales_df.groupby("sku_id"):
        dates = pd.to_datetime(g["date"]).sort_values()
        gaps = dates.diff().dropna()
        assert (gaps == pd.Timedelta(days=1)).all()


def test_promo_flag_present_and_sparse():
    sales_df, _ = generate(n_days=600, seed=4)
    assert set(sales_df["promo_flag"].unique()) <= {0, 1}
    promo_rate = sales_df["promo_flag"].mean()
    # promo days should be a minority of days, but they do happen
    assert 0.0 < promo_rate < 0.25


def test_sku_metadata_has_business_assumptions():
    _, sku_meta_df = generate(n_days=200, seed=5)
    for col in ["unit_cost", "holding_cost_rate", "ordering_cost", "lead_time_days", "trend_type"]:
        assert col in sku_meta_df.columns
    assert (sku_meta_df["unit_cost"] > 0).all()
    assert (sku_meta_df["lead_time_days"] > 0).all()
    assert set(sku_meta_df["trend_type"]) <= {"up", "down", "flat"}


def test_skus_are_not_scaled_copies_of_one_curve():
    """The core 'real structure' requirement: different SKUs should have
    genuinely different demand shapes, not the same curve times a constant.
    We check this by normalizing every SKU's series to zero mean / unit
    variance and confirming the normalized series are NOT all near-identical
    (a scaled copy would normalize to something close to identical)."""
    sales_df, sku_meta_df = generate(n_days=730, seed=6)
    pivot = sales_df.pivot(index="date", columns="sku_id", values="units")
    normalized = (pivot - pivot.mean()) / pivot.std()

    # correlation between every pair of SKUs' *normalized* series
    corr = normalized.corr()
    off_diag = corr.to_numpy()[~np.eye(len(corr), dtype=bool)]
    # if every SKU were a scaled copy of one master curve, correlations
    # would cluster near 1.0; real variety means most pairs correlate weakly
    assert np.median(np.abs(off_diag)) < 0.5


def test_trending_up_sku_has_higher_late_than_early_mean():
    sales_df, sku_meta_df = generate(n_days=913, seed=8)
    up_skus = sku_meta_df[sku_meta_df["trend_type"] == "up"]["sku_id"]
    assert len(up_skus) > 0
    for sku_id in up_skus:
        s = sales_df[sales_df["sku_id"] == sku_id].sort_values("date")["units"].to_numpy()
        early_mean = s[:180].mean()
        late_mean = s[-180:].mean()
        assert late_mean > early_mean


def test_flat_sku_has_stable_mean_across_halves():
    sales_df, sku_meta_df = generate(n_days=913, seed=9)
    flat_skus = sku_meta_df[sku_meta_df["trend_type"] == "flat"]["sku_id"]
    assert len(flat_skus) > 0
    for sku_id in flat_skus:
        s = sales_df[sales_df["sku_id"] == sku_id].sort_values("date")["units"].to_numpy()
        first_half_mean = s[: len(s) // 2].mean()
        second_half_mean = s[len(s) // 2 :].mean()
        ratio = second_half_mean / max(first_half_mean, 1e-6)
        assert 0.6 < ratio < 1.6


def test_demand_is_overdispersed_not_poisson():
    """Real count data (and our negative-binomial generator) has variance
    noticeably above the mean for at least some SKUs; a clean Poisson-only
    process would have variance ~= mean."""
    sales_df, _ = generate(n_days=600, seed=10)
    stats = sales_df.groupby("sku_id")["units"].agg(["mean", "var"])
    overdispersed = stats["var"] > stats["mean"] * 1.2
    assert overdispersed.mean() > 0.5


def test_generate_is_deterministic_given_seed():
    sales_a, _ = generate(n_days=100, seed=123)
    sales_b, _ = generate(n_days=100, seed=123)
    pd.testing.assert_frame_equal(sales_a, sales_b)
