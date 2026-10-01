"""End-to-end training + evaluation + tracking run.

    uv run python -m src.pipeline                      # main run (at-order features)
    uv run python -m src.pipeline --feature-set pickup # comparison run incl. prep time
    uv run python -m src.pipeline --confidence 0.9     # different target coverage

What one run does (all logged to one MLflow run):

  1. Point baseline     LightGBM L2 regression          -> MAE / RMSE ("before")
  2. Baseline + conf.   split conformal on |residual|   -> constant-width interval
  3. Raw quantiles      LightGBM P10 / P50 / P90        -> adaptive but uncalibrated
  4. CQR                MAPIE conformalized quantiles   -> adaptive AND calibrated ("after")
  5. Mondrian CQR       CQR calibrated per segment      -> fixes the weakest segment
  6. Calibration curve  steps 3-4 at 6 coverage levels
  7. Error slices       coverage / MAE / pinball by city, weather, rush hour, ...

Data discipline: train=fit, val=early stopping + choosing the Mondrian
segment, calib=conformal scores only, test=reported once.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mapie.regression import SplitConformalRegressor

from src.analysis import error_slices
from src.analysis.plotting import BLUE, GRID, INK, MUTED, ORANGE, plt
from src.data.features import ARTIFACTS_PATH, FEATURES_PARQUET, TARGET
from src.models.conformal import CQR, MondrianCQR
from src.models.evaluate import (
    comparison_table, interval_metrics, pinball, plot_calibration_curve, plot_feature_importance,
    plot_intervals_sample, plot_width_vs_error, point_metrics,
)
from src.models.train_quantile import (
    BEST_PARAMS_JSON, QUANTILES, SPLITS, feature_list, load_params, load_splits, predict_quantiles,
    train_baseline, train_quantile_model, train_quantile_models,
)
from src.tracking import mlflow_utils as mu

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MODELS_DIR = ROOT / "models"
BUNDLE = MODELS_DIR / "eta_bundle.joblib"

CURVE_LEVELS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MONDRIAN_CANDIDATES = ["traffic", "city_tier", "rush_window", "weather", "festival", "multiple_deliveries", "city"]
KEEP_COLS = ["order_id", "order_ts", "city", "city_tier", "weather", "weather_bucket", "traffic", "rush_window",
             "is_rush_hour", "distance_bucket", "distance_km", "festival", "multiple_deliveries", "vehicle_type",
             "order_type", "rest_lat", "rest_lon", "cust_lat", "cust_lon", "hour"]


def _str_groups(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), "unknown").astype(str)


def choose_mondrian_segment(cqr: CQR, frames, data, min_n: int = 100) -> tuple[str, pd.DataFrame]:
    """Pick the segment column containing the most significantly under-covered
    group under *global* CQR, measured on VAL (never test), so the choice can't
    leak test information."""
    X_val, y_val = data["val"]
    p = cqr.predict(X_val)
    inside = (y_val >= p["lo"].to_numpy()) & (y_val <= p["hi"].to_numpy())
    rows = []
    for col in MONDRIAN_CANDIDATES:
        g = pd.DataFrame({"g": _str_groups(frames["val"][col]), "inside": inside})
        stats = g.groupby("g")["inside"].agg(["mean", "size"])
        stats = stats[stats["size"] >= min_n]
        z = (stats["mean"] - cqr.confidence_level) / np.sqrt(
            cqr.confidence_level * (1 - cqr.confidence_level) / stats["size"])
        rows.append({"segment": col, "worst_group": stats["mean"].idxmin(),
                     "worst_group_val_coverage": stats["mean"].min(), "min_z": z.min(), "n_groups": len(stats)})
    # most *significantly* under-covered group (z-score, so many small groups don't win by noise)
    table = pd.DataFrame(rows).sort_values("min_z")
    return table.iloc[0]["segment"], table


def calibration_curve(models_mid, params, data) -> pd.DataFrame:
    (X_tr, y_tr), (X_val, y_val), (X_c, y_c), (X_te, y_te) = (data[s] for s in SPLITS)
    cache = dict(models_mid)
    rows = []
    for level in CURVE_LEVELS:
        a = 1 - level
        q_lo, q_hi = round(a / 2, 6), round(1 - a / 2, 6)
        for q in (q_lo, q_hi):
            if q not in cache:
                cache[q] = train_quantile_model(q, X_tr, y_tr, X_val, y_val, params)
        lo_raw, hi_raw = cache[q_lo].predict(X_te), cache[q_hi].predict(X_te)
        lo_raw, hi_raw = np.minimum(lo_raw, hi_raw), np.maximum(lo_raw, hi_raw)
        m_raw = interval_metrics(y_te, lo_raw, hi_raw, confidence_level=level)
        cqr = CQR({q_lo: cache[q_lo], 0.5: cache[0.5], q_hi: cache[q_hi]}, level).calibrate(X_c, y_c)
        p = cqr.predict(X_te)
        m_cqr = interval_metrics(y_te, p["lo"], p["hi"], confidence_level=level)
        rows += [{"confidence_level": level, "method": "Raw quantile regression", **m_raw},
                 {"confidence_level": level, "method": "CQR (conformal)", **m_cqr}]
    return pd.DataFrame(rows)[["confidence_level", "method", "coverage", "mean_width", "interval_score"]]


def plot_method_comparison(table: pd.DataFrame, target: float):
    t = table.dropna(subset=["coverage"]).iloc[::-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 0.55 * len(t) + 1.4), sharey=True)
    y = np.arange(len(t))
    # dots, not bars: the coverage axis is zoomed, and truncated bars exaggerate gaps
    se = np.sqrt(target * (1 - target) / table.attrs.get("n_test", 6000))
    ax1.axvspan(target - 1.96 * se, target + 1.96 * se, color=GRID, alpha=0.7, lw=0)
    ax1.axvline(target, color=INK, ls="--", lw=1)
    ax1.scatter(t["coverage"], y, s=70, zorder=3, edgecolor="white", linewidth=1.5,
                color=[ORANGE if c < target - 1.96 * se else BLUE for c in t["coverage"]])
    for yi, c in zip(y, t["coverage"]):
        ax1.text(c + 0.006, yi + 0.18, f"{c:.1%}", va="bottom", color=MUTED, fontsize=8)
    ax1.set_yticks(y, t.index)
    ax1.set(xlim=(0.70, 0.86), xlabel="test coverage (dashed = target, band = sampling noise +/-1.96 SE)",
            title="Coverage")
    ax2.barh(y, t["mean_width"], color=BLUE, height=0.6)
    for yi, w in zip(y, t["mean_width"]):
        ax2.text(w + 0.1, yi, f"{w:.1f} min", va="center", color=MUTED, fontsize=8)
    ax2.set(xlabel="mean interval width (min)", title="Width (narrower is better at equal coverage)",
            xlim=(0, t["mean_width"].max() * 1.3))
    for ax in (ax1, ax2):
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def run(feature_set: str = "order", confidence: float = 0.8, save_bundle: bool = True) -> dict:
    alpha = 1 - confidence
    q_lo, q_hi = round(alpha / 2, 6), round(1 - alpha / 2, 6)
    quantiles = tuple(sorted({q_lo, 0.5, q_hi}))
    REPORTS.mkdir(exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    frames, data = load_splits(feature_set)
    feats = feature_list(feature_set)
    (X_tr, y_tr), (X_val, y_val), (X_c, y_c), (X_te, y_te) = (data[s] for s in SPLITS)
    params = load_params()
    tuned = BEST_PARAMS_JSON.exists()

    mu.setup()
    run_name = f"cqr-{feature_set}-{int(confidence * 100)}"
    with mu.start_run(run_name, tags={"stage": "train", "feature_set": feature_set, "tuned": str(tuned)}) as active:
        # ---------------- params & data lineage ----------------
        mu.log_params_flat({k: v for k, v in params.items() if k != "verbose"}, prefix="lgbm_")
        mlflow.log_params({"feature_set": feature_set, "confidence_level": confidence, "n_features": len(feats),
                           "quantiles": str(quantiles), "data_md5": mu.file_md5(FEATURES_PARQUET)})
        for s in SPLITS:
            f = frames[s]
            mlflow.log_params({f"{s}_rows": len(f), f"{s}_start": str(f["order_ts"].min().date()),
                               f"{s}_end": str(f["order_ts"].max().date())})
        mlflow.log_text("\n".join(feats), "features.txt")

        results: dict[str, dict] = {}

        # ---------------- 1. point baseline ----------------
        base = train_baseline(X_tr, y_tr, X_val, y_val, params)
        base_te = base.predict(X_te)
        results["1. Point baseline (L2 LightGBM)"] = point_metrics(y_te, base_te)

        # ---------------- 2. baseline + split conformal (constant width) ----------------
        scr = SplitConformalRegressor(estimator=base, confidence_level=confidence, prefit=True).conformalize(X_c, y_c)
        _, pis = scr.predict_interval(X_te)
        results["2. Baseline + split conformal"] = {
            **interval_metrics(y_te, pis[:, 0, 0], pis[:, 1, 0], base_te, confidence), **point_metrics(y_te, base_te)}

        # ---------------- 3. raw quantile regression ----------------
        models = train_quantile_models(X_tr, y_tr, X_val, y_val, params, quantiles)
        raw_te, crossing = predict_quantiles(models, X_te)
        results["3. Raw quantile regression"] = interval_metrics(y_te, raw_te[:, 0], raw_te[:, 2], raw_te[:, 1], confidence)
        mlflow.log_metrics({f"best_iter_q{q}": models[q].best_iteration_ for q in quantiles}
                           | {"best_iter_baseline": base.best_iteration_, "quantile_crossing_rate": crossing})

        # ---------------- 4. CQR (MAPIE) ----------------
        cqr = CQR(models, confidence, symmetric=True).calibrate(X_c, y_c)
        p_te = cqr.predict(X_te)
        results["4. CQR (MAPIE)"] = interval_metrics(y_te, p_te["lo"], p_te["hi"], p_te["p50"], confidence)
        cqr_asym = CQR(models, confidence, symmetric=False).calibrate(X_c, y_c)
        pa = cqr_asym.predict(X_te)
        results["4b. CQR asymmetric (MAPIE)"] = interval_metrics(y_te, pa["lo"], pa["hi"], pa["p50"], confidence)
        q_correction = cqr.correction(X_c, y_c)
        mlflow.log_metric("cqr_correction_minutes", q_correction)

        # ---------------- 5. Mondrian CQR ----------------
        seg_col, seg_choice = choose_mondrian_segment(cqr, frames, data)
        mond = MondrianCQR(models, confidence).calibrate(X_c, y_c, _str_groups(frames["calib"][seg_col]))
        pm = mond.predict(X_te, _str_groups(frames["test"][seg_col]))
        results[f"5. Mondrian CQR (by {seg_col})"] = interval_metrics(y_te, pm["lo"], pm["hi"], pm["p50"], confidence)
        mlflow.log_param("mondrian_segment", seg_col)
        # Post-hoc, clearly labelled: traffic is the strongest driver and the test slices showed
        # traffic=high as weakest. Reported for transparency, NOT as the headline method.
        mond_tr = MondrianCQR(models, confidence).calibrate(X_c, y_c, _str_groups(frames["calib"]["traffic"]))
        pmt = mond_tr.predict(X_te, _str_groups(frames["test"]["traffic"]))
        if seg_col != "traffic":
            results["5b. Mondrian CQR (by traffic, post-hoc)"] = interval_metrics(
                y_te, pmt["lo"], pmt["hi"], pmt["p50"], confidence)

        # val-set sanity check of the main method (calibration happened on calib, so val is also out-of-sample)
        pv = cqr.predict(X_val)
        val_cqr = interval_metrics(y_val, pv["lo"], pv["hi"], pv["p50"], confidence)

        # ---------------- metrics -> MLflow ----------------
        slug = {"1. Point baseline (L2 LightGBM)": "baseline", "2. Baseline + split conformal": "baseline_conformal",
                "3. Raw quantile regression": "raw_qr", "4. CQR (MAPIE)": "cqr", "4b. CQR asymmetric (MAPIE)": "cqr_asym"}
        for name, m in results.items():
            mu.log_metrics_flat(m, prefix=f"test_{slug.get(name, 'mondrian')}_")
        mu.log_metrics_flat(val_cqr, prefix="val_cqr_")
        for i, q in enumerate(quantiles):
            mlflow.log_metric(f"test_pinball_q{q}", pinball(y_te, raw_te[:, i], q))

        table = comparison_table(results)
        table.attrs["n_test"] = len(y_te)
        table.round(4).to_csv(REPORTS / f"comparison_{feature_set}.csv")
        mu.log_df(table.round(4).reset_index(names="method"), "comparison.csv")
        mu.log_df(seg_choice, "mondrian_segment_choice.csv")
        mu.log_df(mond.summary(), "mondrian_corrections.csv")

        # ---------------- 6. calibration curve ----------------
        curve = calibration_curve({q: models[q] for q in quantiles}, params, data)
        curve.to_csv(REPORTS / f"calibration_curve_{feature_set}.csv", index=False)
        mu.log_df(curve, "calibration_curve.csv")

        # ---------------- 7. error slices ----------------
        pred = frames["test"][KEEP_COLS].copy()
        pred["y"] = y_te
        pred[["lo", "p50", "hi"]] = p_te[["lo", "p50", "hi"]].to_numpy()
        pred[["raw_p10", "raw_p50", "raw_p90"]] = raw_te
        pred["baseline"] = base_te
        pred[["mondrian_lo", "mondrian_hi"]] = pm[["lo", "hi"]].to_numpy()
        pred[["mondrian_traffic_lo", "mondrian_traffic_hi"]] = pmt[["lo", "hi"]].to_numpy()
        pred["base_conf_lo"], pred["base_conf_hi"] = pis[:, 0, 0], pis[:, 1, 0]
        interval_cols = {"2. Baseline + split conformal": ("base_conf_lo", "base_conf_hi"),
                         "3. Raw quantile regression": ("raw_p10", "raw_p90"),
                         "4. CQR (MAPIE)": ("lo", "hi"),
                         f"5. Mondrian CQR (by {seg_col})": ("mondrian_lo", "mondrian_hi"),
                         "5b. Mondrian CQR (by traffic, post-hoc)": ("mondrian_traffic_lo", "mondrian_traffic_hi")}
        for name, (lo_c, hi_c) in interval_cols.items():
            if name in results:
                cond = error_slices.conditional_summary(pred, lo_c, hi_c, confidence)
                results[name].update(cond)
                table.loc[name, list(cond)] = list(cond.values())
                mu.log_metrics_flat(cond, prefix=f"test_{slug.get(name, 'mondrian')}_")
        table.round(4).to_csv(REPORTS / f"comparison_{feature_set}.csv")
        mu.log_df(table.round(4).reset_index(names="method"), "comparison_with_conditional.csv")
        slice_out = error_slices.run(pred, REPORTS, confidence)
        slices = slice_out["table_df"]
        mond_slices = error_slices.slice_table(pred, lo="mondrian_lo", hi="mondrian_hi", target_coverage=confidence)
        weakest = error_slices.weakest_segments(slices, 1).iloc[0]
        wm = mond_slices[(mond_slices.segment == weakest.segment) & (mond_slices.value == weakest.value)].iloc[0]
        mt_slices = error_slices.slice_table(pred, lo="mondrian_traffic_lo", hi="mondrian_traffic_hi",
                                             target_coverage=confidence)
        wt = mt_slices[(mt_slices.segment == weakest.segment) & (mt_slices.value == weakest.value)].iloc[0]
        mlflow.log_metrics({"weakest_segment_coverage_cqr": weakest.coverage,
                            "weakest_segment_coverage_mondrian": wm.coverage})
        mlflow.set_tag("weakest_segment", f"{weakest.segment}={weakest.value}")
        mlflow.set_tag("weakest_segment_headline", slice_out["headline"][:500])
        mlflow.log_artifact(str(slice_out["table"]), "slices")
        mlflow.log_artifact(str(slice_out["markdown"]), "slices")
        for f in slice_out["figures"]:
            mlflow.log_artifact(str(f), "slices/figures")
        mond_slices.to_csv(REPORTS / "error_slices_mondrian.csv", index=False)

        # ---------------- figures ----------------
        figs = {
            "calibration_curve.png": plot_calibration_curve(curve),
            "feature_importance_p50.png": plot_feature_importance(models[0.5]),
            f"feature_importance_p{int(q_hi * 100)}.png": plot_feature_importance(
                models[q_hi], title=f"P{int(q_hi * 100)} model: feature importance (gain)"),
            "intervals_sample.png": plot_intervals_sample(y_te, p_te["lo"], p_te["hi"], p_te["p50"]),
            "width_vs_error.png": plot_width_vs_error(p_te["lo"], p_te["hi"], y_te, p_te["p50"]),
            "method_comparison.png": plot_method_comparison(table, confidence),
        }
        for name, fig in figs.items():
            fig.savefig(FIGS / (name if feature_set == "order" else f"{feature_set}_{name}"))
            mu.log_figure(fig, name)
            plt.close(fig)

        # ---------------- model artifacts ----------------
        stage_dir = MODELS_DIR / f"run_{feature_set}"
        shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True)
        for q, m in models.items():
            m.booster_.save_model(str(stage_dir / f"lgbm_q{int(round(q * 100)):02d}.txt"))
        base.booster_.save_model(str(stage_dir / "lgbm_baseline.txt"))
        bundle = {
            "feature_set": feature_set, "features": feats, "confidence_level": confidence,
            "quantile_models": models, "baseline": base, "cqr": cqr, "mondrian": mond, "mondrian_segment": seg_col,
            "mondrian_traffic": mond_tr,
            "feature_artifacts": joblib.load(ARTIFACTS_PATH), "cqr_correction": q_correction,
            "metrics": {k: v for k, v in results.items()}, "mlflow_run_id": active.info.run_id,
        }
        joblib.dump(bundle, stage_dir / "eta_bundle.joblib")
        mlflow.log_artifacts(str(stage_dir), "model")

        summary = {
            "run_id": active.info.run_id, "feature_set": feature_set, "confidence_level": confidence,
            "tuned_params": tuned, "results": results, "val_cqr": val_cqr, "cqr_correction_min": q_correction,
            "quantile_crossing_rate": crossing, "mondrian_segment": seg_col,
            "weakest_segment": {"segment": weakest.segment, "value": weakest.value, "n": int(weakest.n),
                                "coverage_cqr": float(weakest.coverage), "coverage_mondrian": float(wm.coverage),
                                "width_cqr": float(weakest.mean_width), "width_mondrian": float(wm.mean_width),
                                "coverage_mondrian_traffic": float(wt.coverage),
                                "width_mondrian_traffic": float(wt.mean_width)},
            "headline": slice_out["headline"],
            "split_rows": {s: len(frames[s]) for s in SPLITS},
        }
        (REPORTS / f"metrics_{feature_set}.json").write_text(json.dumps(summary, indent=2, default=float))
        mlflow.log_dict(summary, "summary.json")

    if save_bundle:
        shutil.copy2(stage_dir / "eta_bundle.joblib", BUNDLE)
        pred.to_parquet(REPORTS / "test_predictions.parquet", index=False)

    _print_summary(table, summary, curve)
    return summary


def _print_summary(table: pd.DataFrame, summary: dict, curve: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\n=== Test-set comparison ===")
    print(table.round(3).to_string())
    print("\n=== Calibration curve ===")
    print(curve.pivot(index="confidence_level", columns="method", values="coverage").round(3).to_string())
    print(f"\nCQR correction: {summary['cqr_correction_min']:+.2f} min per side | "
          f"quantile crossing: {summary['quantile_crossing_rate']:.2%} | Mondrian segment: {summary['mondrian_segment']}")
    w = summary["weakest_segment"]
    print(summary["headline"])
    print(f"Mondrian CQR ({summary['mondrian_segment']}) on that segment: {w['coverage_cqr']:.1%} -> "
          f"{w['coverage_mondrian']:.1%} (width {w['width_cqr']:.1f} -> {w['width_mondrian']:.1f} min)")
    print(f"Mondrian CQR (traffic, post-hoc) on that segment: {w['coverage_cqr']:.1%} -> "
          f"{w['coverage_mondrian_traffic']:.1%} (width {w['width_cqr']:.1f} -> {w['width_mondrian_traffic']:.1f} min)")
    print(f"MLflow run: {summary['run_id']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", choices=["order", "pickup"], default="order")
    ap.add_argument("--confidence", type=float, default=0.8)
    ap.add_argument("--no-save", action="store_true", help="don't overwrite the bundle used by the app")
    a = ap.parse_args()
    run(a.feature_set, a.confidence, save_bundle=not a.no_save and a.feature_set == "order")
