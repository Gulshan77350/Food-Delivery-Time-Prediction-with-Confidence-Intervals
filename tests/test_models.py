"""Tests for the conformal layer and metrics, on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from src.models.conformal import (
    CQR, MondrianCQR, conformal_quantile, cqr_correction, cqr_correction_asymmetric,
)
from src.models.evaluate import coverage, interval_metrics, interval_score, pinball
from src.models.train_quantile import predict_quantiles, train_quantile_models

FAST = {"n_estimators": 200, "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 20,
        "verbose": -1, "random_state": 0}


def _heteroscedastic(n, seed):
    """y = 2x + noise whose scale grows with x: intervals SHOULD widen with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 10, n)
    g = rng.integers(0, 2, n)
    y = 2 * x + rng.normal(0, 0.5 + 0.4 * x + 2.0 * g, n)  # group 1 is much noisier
    return pd.DataFrame({"x": x, "g": g.astype(float)}), y, g


@pytest.fixture(scope="module")
def fitted():
    X_tr, y_tr, _ = _heteroscedastic(4000, 0)
    X_val, y_val, _ = _heteroscedastic(1000, 1)
    models = train_quantile_models(X_tr, y_tr, X_val, y_val, FAST)
    return models


def test_conformal_quantile_finite_sample_level():
    scores = np.arange(1, 11, dtype=float)  # n=10
    # ceil(11*0.8)/10 = 0.9 -> 9th order statistic
    assert conformal_quantile(scores, 0.2) == 9.0
    # alpha so small that ceil((n+1)(1-a)) > n -> infinite interval
    assert conformal_quantile(scores, 0.01) == float("inf")


def test_manual_cqr_matches_mapie(fitted):
    X_c, y_c, _ = _heteroscedastic(2000, 2)
    X_t, _, _ = _heteroscedastic(500, 3)
    lo, hi = fitted[0.1].predict(X_c), fitted[0.9].predict(X_c)

    q = cqr_correction(lo, hi, y_c, 0.2)
    pred = CQR(fitted, 0.8, symmetric=True).calibrate(X_c, y_c).predict(X_t)
    np.testing.assert_allclose(pred["lo"], fitted[0.1].predict(X_t) - q, atol=0.05)
    np.testing.assert_allclose(pred["hi"], fitted[0.9].predict(X_t) + q, atol=0.05)

    q_lo, q_hi = cqr_correction_asymmetric(lo, hi, y_c, 0.2)
    pa = CQR(fitted, 0.8, symmetric=False).calibrate(X_c, y_c).predict(X_t)
    np.testing.assert_allclose(pa["lo"], fitted[0.1].predict(X_t) - q_lo, atol=0.05)
    np.testing.assert_allclose(pa["hi"], fitted[0.9].predict(X_t) + q_hi, atol=0.05)


def test_cqr_hits_target_coverage_on_exchangeable_data(fitted):
    """The guarantee: averaged over many calib/test draws, coverage >= target."""
    covs = []
    for seed in range(10, 20):
        X_c, y_c, _ = _heteroscedastic(1000, seed)
        X_t, y_t, _ = _heteroscedastic(2000, seed + 100)
        p = CQR(fitted, 0.8).calibrate(X_c, y_c).predict(X_t)
        covs.append(coverage(y_t, p["lo"], p["hi"]))
    assert np.mean(covs) >= 0.79
    assert np.mean(covs) <= 0.83  # and not wastefully wide


def test_cqr_intervals_adapt_to_noise(fitted):
    X_c, y_c, _ = _heteroscedastic(2000, 4)
    p = CQR(fitted, 0.8).calibrate(X_c, y_c).predict(pd.DataFrame({"x": [1.0, 9.0], "g": [0.0, 0.0]}))
    widths = (p["hi"] - p["lo"]).to_numpy()
    assert widths[1] > 1.5 * widths[0]


def test_mondrian_restores_group_coverage(fitted):
    X_c, y_c, g_c = _heteroscedastic(4000, 5)
    X_t, y_t, g_t = _heteroscedastic(4000, 6)
    mond = MondrianCQR(fitted, 0.8, min_group_size=100).calibrate(X_c, y_c, g_c)
    p = mond.predict(X_t, g_t)
    for grp in (0, 1):
        m = g_t == grp
        assert coverage(y_t[m], p["lo"][m], p["hi"][m]) == pytest.approx(0.8, abs=0.035)


def test_mondrian_small_group_falls_back_to_global(fitted):
    X_c, y_c, _ = _heteroscedastic(500, 7)
    groups = np.array(["big"] * 495 + ["tiny"] * 5)
    mond = MondrianCQR(fitted, 0.8, min_group_size=100).calibrate(X_c, y_c, groups)
    assert "tiny" not in mond.group_q and mond.summary().set_index("group").loc["tiny", "uses_global_fallback"]


def test_quantile_rearrangement_removes_crossing(fitted):
    X_t, _, _ = _heteroscedastic(1000, 8)
    preds, _ = predict_quantiles(fitted, X_t)
    assert (np.diff(preds, axis=1) >= 0).all()


def test_metrics_known_values():
    y = np.array([10.0, 20.0, 30.0])
    assert pinball(y, y, 0.9) == 0
    assert pinball([10.0], [8.0], 0.9) == pytest.approx(0.9 * 2)   # under-predict: alpha * err
    assert pinball([10.0], [12.0], 0.9) == pytest.approx(0.1 * 2)  # over-predict: (1-alpha) * err
    assert coverage(y, y - 1, y + 1) == 1.0
    # one miss of 5 below the lower bound: width 2 + (2/0.2)*5
    assert interval_score([0.0], [5.0], [7.0], 0.2) == pytest.approx(2 + 10 * 5)
    m = interval_metrics(y, y - 1, y + 1, y, 0.8)
    assert m["mean_width"] == 2 and m["coverage_gap"] == pytest.approx(0.2)
