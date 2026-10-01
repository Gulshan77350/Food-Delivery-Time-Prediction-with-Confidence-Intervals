"""Segmented error analysis: where do the intervals break?

Conformal prediction guarantees MARGINAL coverage: 80% averaged over all
orders. It says nothing about any individual segment. A model can hit 80%
overall while covering 90% of easy off-peak orders and only 60% of rush-hour
jams, and the jams are exactly where a broken ETA promise hurts most. So we
slice the test set and report, per segment:

    n, MAE of P50, mean pinball loss (P10/P50/P90), coverage, coverage gap, width

and flag the weakest segments honestly. Small segments get a binomial
standard error so we don't over-read noise: with n=50 and true coverage 80%,
the SE is 5.7 points, so a 72% reading is not evidence of a problem.

Usage:
    (called from src/pipeline.py; standalone)
    uv run python -m src.analysis.error_slices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.plotting import BLUE, GRID, INK, MUTED, ORANGE, plt
from src.models.evaluate import pinball

ROOT = Path(__file__).resolve().parents[2]
PRED_PARQUET = ROOT / "reports" / "test_predictions.parquet"
OUT_DIR = ROOT / "reports"

SEGMENTS = ["city", "city_tier", "weather", "traffic", "rush_window", "distance_bucket",
            "festival", "multiple_deliveries", "vehicle_type"]
MIN_N_FLAG = 100  # don't "flag" segments smaller than this: too noisy


def slice_table(pred: pd.DataFrame, segments=SEGMENTS, lo="lo", hi="hi", p50="p50",
                target_coverage: float = 0.8) -> pd.DataFrame:
    """pred needs: y, lo, hi, p50, raw_p10, raw_p90 + segment columns."""
    rows = []
    for seg in segments:
        key = pred[seg].astype("object").where(pred[seg].notna(), "unknown").astype(str)
        for val, g in pred.groupby(key, observed=True):
            y = g["y"].to_numpy()
            inside = (y >= g[lo]) & (y <= g[hi])
            cov = float(inside.mean())
            rows.append({
                "segment": seg, "value": val, "n": len(g),
                "mae_p50": float(np.abs(y - g[p50]).mean()),
                "pinball_mean": float(np.mean([pinball(y, g["raw_p10"], 0.1), pinball(y, g["raw_p50"], 0.5),
                                               pinball(y, g["raw_p90"], 0.9)])),
                "coverage": cov,
                "coverage_se": float(np.sqrt(target_coverage * (1 - target_coverage) / len(g))),
                "coverage_gap": cov - target_coverage,
                "late_rate": float((y > g[hi]).mean()),
                "mean_width": float((g[hi] - g[lo]).mean()),
                "raw_coverage": float(((y >= g["raw_p10"]) & (y <= g["raw_p90"])).mean()),
            })
    t = pd.DataFrame(rows)
    # z-score of the gap: |z| > 2 means "unlikely to be sampling noise"
    t["gap_z"] = t["coverage_gap"] / t["coverage_se"]
    return t


def conditional_summary(pred: pd.DataFrame, lo: str, hi: str, target_coverage: float = 0.8,
                        min_n: int = MIN_N_FLAG) -> dict:
    """How well does an interval method hold coverage *inside* segments?

    * worst_segment_coverage: lowest coverage among segments with n >= min_n
    * n_segments_under: segments significantly below target (z < -2)
    * mean_z2: mean squared z-score of the coverage gaps. ~1 means gaps look like
      pure sampling noise (well calibrated in every segment); >>1 means systematic
      over/under-coverage across segments.
    """
    t = slice_table(pred, lo=lo, hi=hi, target_coverage=target_coverage)
    big = t[t["n"] >= min_n]
    return {"worst_segment_coverage": float(big["coverage"].min()),
            "n_segments_under": int((big["gap_z"] < -2).sum()),
            "n_segments_tested": int(len(big)),
            "mean_z2": float((big["gap_z"] ** 2).mean())}


def weakest_segments(table: pd.DataFrame, k: int = 5, min_n: int = MIN_N_FLAG) -> pd.DataFrame:
    big = table[table["n"] >= min_n]
    return big.sort_values("coverage").head(k)


def headline(table: pd.DataFrame, min_n: int = MIN_N_FLAG, target: float = 0.8) -> str:
    w = weakest_segments(table, 1, min_n).iloc[0]
    best = table[table["n"] >= min_n].sort_values("coverage").iloc[-1]
    return (f"Weakest segment: {w['segment']}={w['value']} (n={int(w['n'])}): coverage drops to "
            f"{w['coverage']:.1%} vs {target:.0%} target (z={w['gap_z']:.1f}), late-rate {w['late_rate']:.1%}, "
            f"MAE {w['mae_p50']:.1f} min. Strongest: {best['segment']}={best['value']} at {best['coverage']:.1%}.")


def plot_segment(table: pd.DataFrame, segment: str, target: float = 0.8, compare_col: str | None = None):
    """Dot plot (not bars: coverage axis doesn't start at 0, and bars on a truncated
    axis exaggerate differences) with 95% CI whiskers; flagged groups in orange."""
    from matplotlib.lines import Line2D

    t = table[table["segment"] == segment].sort_values("coverage")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.36 * len(t) + 1.9), sharey=True,
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    y = np.arange(len(t))
    flagged = (t["n"] >= MIN_N_FLAG) & (t["gap_z"] < -2)
    ax1.axvline(target, color=INK, ls="--", lw=1)
    ax1.errorbar(t["coverage"], y, xerr=1.96 * t["coverage_se"], fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    if compare_col:
        ax1.scatter(t[compare_col], y, color=MUTED, marker="|", s=140, lw=2, zorder=3)
    ax1.scatter(t["coverage"], y, color=np.where(flagged, ORANGE, BLUE), s=60, zorder=4,
                edgecolor="white", linewidth=1.5)
    ax1.set_yticks(y, [f"{v}  (n={n:,})" for v, n in zip(t["value"], t["n"])])
    lo = min(t["coverage"].min(), (t[compare_col].min() if compare_col else 1)) - 0.06
    ax1.set(xlim=(max(0.0, lo), 1.0), xlabel=f"coverage of {target:.0%} interval (dashed = target)",
            title=f"Coverage by {segment}")
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", ms=7, label="conformal (CQR)"),
               Line2D([], [], color=ORANGE, marker="o", ls="", ms=7, label="significantly below target (z<-2)"),
               Line2D([], [], color=MUTED, marker="|", ls="", ms=10, mew=2, label="raw quantile, before conformal"),
               Line2D([], [], color=MUTED, lw=1, label="95% CI")]
    ax1.legend(handles=handles if compare_col else handles[:2] + handles[3:], loc="upper center",
               bbox_to_anchor=(0.5, -0.32 if len(t) < 6 else -0.12), ncol=2, fontsize=7)
    ax2.barh(y, t["mae_p50"], color=BLUE, height=0.6)
    for yi, v, w in zip(y, t["mae_p50"], t["mean_width"]):
        ax2.text(v + 0.1, yi, f"{v:.1f}  (width {w:.1f})", va="center", color=MUTED, fontsize=8)
    ax2.set(xlabel="MAE of P50 (min)", title=f"Error by {segment}", xlim=(0, t["mae_p50"].max() * 1.6))
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    return fig


def run(pred: pd.DataFrame, out_dir: Path = OUT_DIR, target: float = 0.8) -> dict:
    """Builds the table + figures, writes them to out_dir; returns paths for MLflow logging."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table = slice_table(pred, target_coverage=target)
    table.to_csv(out_dir / "error_slices.csv", index=False)

    paths = {"table": out_dir / "error_slices.csv", "figures": []}
    for seg in ["city", "weather", "rush_window", "traffic", "distance_bucket", "city_tier", "festival", "multiple_deliveries"]:
        fig = plot_segment(table, seg, target, compare_col="raw_coverage")
        p = fig_dir / f"slice_{seg}.png"
        fig.savefig(p)
        plt.close(fig)
        paths["figures"].append(p)

    weak = weakest_segments(table, 8)
    text = headline(table, target=target)
    md = ["# Error slices (test set, CQR 80% intervals)", "", f"**{text}**", "",
          "## Weakest segments (n >= 100)", "", weak.round(3).to_markdown(index=False), "",
          "## All segments", "", table.round(3).to_markdown(index=False)]
    (out_dir / "error_slices.md").write_text("\n".join(md), encoding="utf-8")
    paths["markdown"] = out_dir / "error_slices.md"
    paths["headline"] = text
    paths["table_df"] = table
    return paths


if __name__ == "__main__":
    res = run(pd.read_parquet(PRED_PARQUET))
    print(res["headline"])
