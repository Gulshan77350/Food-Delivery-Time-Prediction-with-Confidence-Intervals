"""Optuna hyperparameter search for the quantile models.

Objective: mean pinball loss of the P10/P50/P90 models on the VAL split.
One shared parameter set for all three quantiles keeps the search small and
the models comparable. Each trial is an MLflow child run.

The CALIB and TEST splits are never touched here. Tuning on calib would make
the conformal residuals optimistic; tuning on test would make the reported
numbers optimistic.

Usage:
    uv run python -m src.models.tune --trials 30
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import optuna

from src.models.evaluate import pinball
from src.models.train_quantile import (
    BEST_PARAMS_JSON, DEFAULT_PARAMS, QUANTILES, load_splits, predict_quantiles, train_quantile_models,
)
from src.tracking import mlflow_utils as mu


TUNED_KEYS = ["learning_rate", "num_leaves", "min_child_samples", "subsample",
              "colsample_bytree", "reg_lambda", "max_cat_to_onehot"]


def search_space(trial: optuna.Trial) -> dict:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        "max_cat_to_onehot": trial.suggest_int("max_cat_to_onehot", 4, 32),
    }


def main(trials: int = 30, feature_set: str = "order") -> dict:
    import mlflow

    _, data = load_splits(feature_set)
    (X_tr, y_tr), (X_val, y_val) = data["train"], data["val"]
    mu.setup()

    def objective(trial: optuna.Trial) -> float:
        space = search_space(trial)
        params = {**DEFAULT_PARAMS, **space}
        models = train_quantile_models(X_tr, y_tr, X_val, y_val, params)
        preds, _ = predict_quantiles(models, X_val)
        losses = {q: pinball(y_val, preds[:, i], q) for i, q in enumerate(QUANTILES)}
        score = float(np.mean(list(losses.values())))
        with mu.start_run(f"trial-{trial.number}", nested=True, tags={"stage": "tuning"}):
            mu.log_params_flat(space)
            mu.log_metrics_flat({f"val_pinball_q{q}": v for q, v in losses.items()} | {"val_pinball_mean": score})
        return score

    with mu.start_run("optuna-tuning", tags={"stage": "tuning", "feature_set": feature_set}):
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial({k: DEFAULT_PARAMS[k] for k in TUNED_KEYS})  # trial 0 = defaults: tuning can only help
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_val_pinball_mean", study.best_value)
        default_score = study.trials[0].value
        mlflow.log_metric("default_val_pinball_mean", default_score)

    BEST_PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    out = {"params": study.best_params, "val_pinball_mean": study.best_value,
           "default_val_pinball_mean": default_score, "n_trials": trials}
    BEST_PARAMS_JSON.write_text(json.dumps(out, indent=2))
    print(f"[tune] default val pinball {default_score:.4f} -> best {study.best_value:.4f}")
    print(f"[tune] best params: {study.best_params}\n[tune] wrote {BEST_PARAMS_JSON}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--feature-set", choices=["order", "pickup"], default="order")
    a = ap.parse_args()
    main(a.trials, a.feature_set)
