"""Metrics and diagnostic plots for point and interval forecasts.

Metrics
-------
* Pinball loss per quantile: the proper scoring rule each quantile model is
  trained on. Lower is better; comparable only at the same quantile.
* Empirical coverage: share of true values inside [lo, hi]. Should be close to
  the target (0.80). Above target = intervals wider than needed; below =
  overconfident (the dangerous direction for an ETA promise).
* Mean / median interval width (minutes): the price of coverage. Compare
  methods ONLY at equal coverage; a 100%-coverage 0-120 min interval is useless.
* Interval (Winkler) score: width + (2/alpha) * miss distance. A single number
  that rewards narrow intervals and penalises misses; lower is better.
* MAE / RMSE of the point forecast (P50 or the L2 baseline): the conventional
  comparison everybody understands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, root_mean_squared_error

from src.analysis.plotting import BLUE, INK, MUTED, ORANGE, plt


def pinball(y, q_pred, alpha: float) -> float:
    return float(mean_pinball_loss(y, q_pred, alpha=alpha))


def coverage(y, lo, hi) -> float:
    y = np.asarray(y)
    return float(((y >= lo) & (y <= hi)).mean())


def interval_score(y, lo, hi, alpha: float) -> float:
    y, lo, hi = map(np.asarray, (y, lo, hi))
    return float(np.mean((hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)))


def point_metrics(y, pred) -> dict:
    return {"mae": float(mean_absolute_error(y, pred)), "rmse": float(root_mean_squared_error(y, pred))}


def interval_metrics(y, lo, hi, p50=None, confidence_level: float = 0.8) -> dict:
    alpha = 1 - confidence_level
    y, lo, hi = map(np.asarray, (y, lo, hi))
    width = hi - lo
    out = {
        "coverage": coverage(y, lo, hi),
        "coverage_gap": coverage(y, lo, hi) - confidence_level,
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "interval_score": interval_score(y, lo, hi, alpha),
        "miss_below_rate": float((y < lo).mean()),   # delivered faster than promised range
        "miss_above_rate": float((y > hi).mean()),   # LATE vs the promised range: the costly side
        f"pinball_q{alpha / 2:.2f}": pinball(y, lo, alpha / 2),
        f"pinball_q{1 - alpha / 2:.2f}": pinball(y, hi, 1 - alpha / 2),
    }
    if p50 is not None:
        out["pinball_q0.50"] = pinball(y, p50, 0.5)
        out.update({f"p50_{k}": v for k, v in point_metrics(y, p50).items()})
    return out


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """results = {method_name: metrics dict} -> tidy table for README / MLflow."""
    cols = ["coverage", "mean_width", "median_width", "interval_score", "miss_above_rate",
            "worst_segment_coverage", "n_segments_under", "mean_z2", "p50_mae", "p50_rmse", "mae", "rmse"]
    df = pd.DataFrame(results).T
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_calibration_curve(curve: pd.DataFrame):
    """curve columns: confidence_level, method, coverage, mean_width."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot([0.45, 1], [0.45, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
    colors = {"Raw quantile regression": ORANGE, "CQR (conformal)": BLUE}
    for method, g in curve.groupby("method"):
        c = colors.get(method, INK)
        ax1.plot(g["confidence_level"], g["coverage"], marker="o", ms=6, lw=2, color=c, label=method)
        ax2.plot(g["confidence_level"], g["mean_width"], marker="o", ms=6, lw=2, color=c, label=method)
    ax1.set(title="Calibration: target vs empirical coverage (test)", xlabel="target coverage",
            ylabel="empirical coverage", xlim=(0.45, 1), ylim=(0.45, 1))
    ax1.legend(loc="upper left")
    ax2.set(title="Price of coverage: interval width", xlabel="target coverage", ylabel="mean interval width (min)")
    fig.tight_layout()
    return fig


def plot_feature_importance(model, top: int = 15, title: str = "P50 model: feature importance (gain)"):
    booster = model.booster_
    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    imp = (imp / imp.sum()).sort_values().tail(top)
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1))
    bars = ax.barh(imp.index, imp.values, color=BLUE, height=0.6)
    for b, v in zip(bars, imp.values):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", color=MUTED, fontsize=8)
    ax.set(title=title, xlabel="share of total gain")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    return fig


def plot_intervals_sample(y, lo, hi, p50, n: int = 80, seed: int = 0,
                          title: str = "CQR 80% intervals on random test orders"):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=min(n, len(y)), replace=False)
    order = idx[np.argsort(np.asarray(p50)[idx])]
    y, lo, hi, p50 = (np.asarray(a)[order] for a in (y, lo, hi, p50))
    inside = (y >= lo) & (y <= hi)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.vlines(x, lo, hi, color=BLUE, alpha=0.35, lw=4, label="P10-P90 (conformal)")
    ax.scatter(x, p50, color=BLUE, s=10, zorder=3, label="P50")
    ax.scatter(x[inside], y[inside], color=INK, s=14, marker="x", zorder=4, label="actual (inside)")
    ax.scatter(x[~inside], y[~inside], color=ORANGE, s=26, marker="x", zorder=4, label="actual (outside)")
    ax.set(title=f"{title}: {inside.mean():.0%} inside", xlabel="orders sorted by P50", ylabel="minutes")
    ax.set_xticks([])
    ax.legend(ncol=4, loc="upper left")
    fig.tight_layout()
    return fig


def plot_width_vs_error(lo, hi, y, p50, bins: int = 8):
    """Adaptivity check: do wider intervals really go to harder orders?"""
    width = np.asarray(hi) - np.asarray(lo)
    err = np.abs(np.asarray(y) - np.asarray(p50))
    df = pd.DataFrame({"width": width, "err": err})
    df["bin"] = pd.qcut(df["width"], bins, duplicates="drop")
    g = df.groupby("bin", observed=True).agg(width=("width", "mean"), mae=("err", "mean"), n=("err", "size"))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(g["width"], g["mae"], marker="o", ms=7, lw=2, color=BLUE)
    for w, m, n in g.itertuples(index=False):
        ax.annotate(f"n={n}", (w, m), textcoords="offset points", xytext=(0, 7), ha="center", color=MUTED, fontsize=7)
    ax.set(title="Adaptivity: wider intervals <-> larger errors", xlabel="predicted interval width (min, binned)",
           ylabel="actual |error| of P50 (min)")
    fig.tight_layout()
    return fig
