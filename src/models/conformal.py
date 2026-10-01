"""Conformalized Quantile Regression (CQR) on top of the LightGBM quantile models.

What CQR does (Romano, Patterson & Candes, 2019)
------------------------------------------------
1. Fit lower/upper quantile models q_lo(x), q_hi(x) on TRAIN.
2. On a separate CALIBRATION set compute conformity scores
       E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i))
   (positive = the true value fell outside the raw interval, by that much).
3. Take Q = the ceil((n+1)(1-a))/n empirical quantile of E.
4. Predict [q_lo(x) - Q, q_hi(x) + Q].

Guarantee: if calibration and test points are EXCHANGEABLE, then
P(y in interval) >= 1 - a, for any model and any data distribution. This is
MARGINAL coverage (on average over all orders), not conditional coverage for
each segment, which is why src/analysis/error_slices.py checks segments
separately, and why `MondrianCQR` below calibrates per segment.

Why the calibration split must be separate
------------------------------------------
If the residuals came from rows the models were trained on (or early-stopped
on), they'd be too small, Q would be too small, and coverage would silently
fall below target. Here: train -> fit, val -> early stopping/tuning,
calib -> conformal scores only, test -> report. Never mixed.

Honest caveat (time series)
---------------------------
Our split is chronological, so exchangeability is an approximation. If
delivery times drift (new city, monsoon, a festival week), coverage on future
data can drop below target. Mitigations used in production: recalibrate on a
rolling recent window, or adaptive conformal inference (Gibbs & Candes, 2021).
We calibrate on the window *immediately before* test to keep the gap small.

Two implementations are provided:
* `CQR`: wraps MAPIE's ConformalizedQuantileRegressor (prefit models): the
  library implementation reviewers expect.
* `cqr_correction`: a 5-line numpy version. tests/test_models.py asserts it
  matches MAPIE, which proves we understand what the library is doing.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from mapie.regression import ConformalizedQuantileRegressor

# MAPIE hands LightGBM numpy arrays (column order is preserved), which triggers a
# harmless sklearn "X does not have valid feature names" warning on every call.
warnings.filterwarnings("ignore", message="X does not have valid feature names")


# --------------------------------------------------------------------------- #
# Reference implementation
# --------------------------------------------------------------------------- #
def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of conformity scores:
    the k-th smallest score with k = ceil((n+1)(1-alpha)).

    Note: np.quantile(scores, k/n, method="higher") is NOT the same thing (it
    indexes position (n-1)*k/n, which can land one order statistic higher, e.g.
    n=10, k=9 -> 10th score). That is slightly conservative and is what MAPIE
    effectively returns; the difference is <= one order statistic, negligible at
    our n~6,000, but tests/test_models.py pins the exact definition.
    """
    scores = np.sort(np.asarray(scores, dtype=float))
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    if k > n:  # too few calibration points for this alpha -> infinite interval
        return float("inf")
    return float(scores[k - 1])


def cqr_scores(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(lo) - y, y - np.asarray(hi))


def cqr_correction(lo, hi, y, alpha: float) -> float:
    """Symmetric CQR correction Q (same widening on both sides)."""
    return conformal_quantile(cqr_scores(lo, hi, y), alpha)


def cqr_correction_asymmetric(lo, hi, y, alpha: float) -> tuple[float, float]:
    """Separate corrections for each side, each at alpha/2 (more conservative,
    but corrects one-sided bias, e.g. a P90 that is systematically too low)."""
    y = np.asarray(y)
    return conformal_quantile(np.asarray(lo) - y, alpha / 2), conformal_quantile(y - np.asarray(hi), alpha / 2)


# --------------------------------------------------------------------------- #
# MAPIE wrapper
# --------------------------------------------------------------------------- #
class CQR:
    """Global CQR via MAPIE. `models` = {alpha_lo: m, 0.5: m, alpha_hi: m}, already fitted."""

    def __init__(self, models: dict, confidence_level: float = 0.8, symmetric: bool = True):
        self.confidence_level = confidence_level
        self.symmetric = symmetric
        a = 1 - confidence_level
        self.q_lo, self.q_hi = round(a / 2, 6), round(1 - a / 2, 6)
        missing = {self.q_lo, 0.5, self.q_hi} - set(models)
        if missing:
            raise ValueError(f"need quantile models for {sorted(missing)}")
        self.models = models
        self.mapie = ConformalizedQuantileRegressor(
            estimator=[models[self.q_lo], models[self.q_hi], models[0.5]],  # MAPIE order: lower, upper, median
            confidence_level=confidence_level,
            prefit=True,
        )

    def calibrate(self, X_calib, y_calib) -> "CQR":
        self.mapie.conformalize(X_calib, y_calib)
        self.n_calib = len(y_calib)
        return self

    def predict(self, X) -> pd.DataFrame:
        p50, pis = self.mapie.predict_interval(X, symmetric_correction=self.symmetric)
        lo, hi = pis[:, 0, 0], pis[:, 1, 0]
        # MAPIE's median comes from the P50 model; keep it inside the interval.
        return pd.DataFrame({"lo": lo, "p50": np.clip(p50, lo, hi), "hi": hi})

    def correction(self, X_calib, y_calib) -> float:
        """Recover Q for reporting (how many minutes conformal added per side)."""
        lo, hi = self.models[self.q_lo].predict(X_calib), self.models[self.q_hi].predict(X_calib)
        return cqr_correction(lo, hi, np.asarray(y_calib), 1 - self.confidence_level)


# --------------------------------------------------------------------------- #
# Group-conditional ("Mondrian") CQR
# --------------------------------------------------------------------------- #
class MondrianCQR:
    """CQR with a separate correction per segment (Vovk's Mondrian conformal).

    Marginal CQR can over-cover easy segments and under-cover hard ones while
    still hitting 80% overall. Calibrating within each group restores the
    coverage guarantee *per group* (given exchangeability within the group),
    at the cost of needing enough calibration points per group. Groups with
    fewer than `min_group_size` calibration rows fall back to the global Q.
    """

    def __init__(self, models: dict, confidence_level: float = 0.8, min_group_size: int = 100):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.q_lo, self.q_hi = round(self.alpha / 2, 6), round(1 - self.alpha / 2, 6)
        self.models = models
        self.min_group_size = min_group_size

    def _raw(self, X):
        return self.models[self.q_lo].predict(X), self.models[0.5].predict(X), self.models[self.q_hi].predict(X)

    def calibrate(self, X_calib, y_calib, groups) -> "MondrianCQR":
        lo, _, hi = self._raw(X_calib)
        scores = cqr_scores(lo, hi, np.asarray(y_calib))
        groups = pd.Series(np.asarray(groups, dtype=object)).fillna("unknown").astype(str)
        self.global_q = conformal_quantile(scores, self.alpha)
        self.group_q, self.group_n = {}, {}
        for g, idx in groups.groupby(groups).groups.items():
            self.group_n[g] = len(idx)
            if len(idx) >= self.min_group_size:
                self.group_q[g] = conformal_quantile(scores[np.asarray(idx)], self.alpha)
        return self

    def predict(self, X, groups) -> pd.DataFrame:
        lo, p50, hi = self._raw(X)
        g = pd.Series(np.asarray(groups, dtype=object)).fillna("unknown").astype(str)
        q = g.map(self.group_q).fillna(self.global_q).to_numpy(dtype=float)
        lo, hi = lo - q, hi + q
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        return pd.DataFrame({"lo": lo, "p50": np.clip(p50, lo, hi), "hi": hi})

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "group": list(self.group_n),
            "n_calib": list(self.group_n.values()),
            "correction_min": [self.group_q.get(g, self.global_q) for g in self.group_n],
            "uses_global_fallback": [g not in self.group_q for g in self.group_n],
        }).sort_values("correction_min")
