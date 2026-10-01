"""LightGBM quantile models (P10 / P50 / P90) and a point-estimate baseline.

Design notes
------------
* One LightGBM model per quantile, objective="quantile" (pinball loss). The
  alternative, one model with a multi-quantile head, isn't supported natively by
  LightGBM, and separate models let each quantile pick its own tree count.
* Early stopping uses the VAL split only. The CALIB split is never seen here:
  conformal guarantees require calibration residuals from data that no model
  was fit or tuned on (see src/models/conformal.py).
* Categoricals are passed as integer codes with `categorical_feature=` so the
  same float matrix flows through LightGBM, MAPIE and the Streamlit app
  without dtype surprises (MAPIE validates inputs as arrays).
* Quantile crossing (P10 > P50 or P50 > P90) can happen because the models
  are independent. We fix it with monotone rearrangement (sorting the three
  predictions per row; Chernozhukov et al., 2010), which never increases
  pinball loss, and we report how often it was needed.

Usage:
    uv run python -m src.models.train_quantile          # quick standalone check
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.features import (
    CATEGORICAL, FEATURES_AT_ORDER, FEATURES_AT_PICKUP, FEATURES_PARQUET, TARGET,
)

ROOT = Path(__file__).resolve().parents[2]
BEST_PARAMS_JSON = ROOT / "reports" / "best_params.json"

QUANTILES = (0.1, 0.5, 0.9)
SPLITS = ("train", "val", "calib", "test")

DEFAULT_PARAMS: dict = {
    "n_estimators": 3000,          # upper bound; early stopping picks the real count
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "max_cat_to_onehot": 8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS = 150


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def feature_list(feature_set: str = "order") -> list[str]:
    return {"order": FEATURES_AT_ORDER, "pickup": FEATURES_AT_PICKUP}[feature_set]


def to_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Feature frame -> all-float matrix; categoricals become codes (NaN = missing/unseen)."""
    X = pd.DataFrame(index=df.index)
    for c in features:
        if c in CATEGORICAL:
            codes = df[c].cat.codes.astype("float64")
            X[c] = codes.where(codes >= 0)
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return X


def load_splits(feature_set: str = "order", path: Path = FEATURES_PARQUET):
    """Returns (frames, data) where data[split] = (X, y) and frames[split] is the full row frame."""
    df = pd.read_parquet(path)
    feats = feature_list(feature_set)
    frames = {s: df[df["split"] == s].reset_index(drop=True) for s in SPLITS}
    data = {s: (to_matrix(f, feats), f[TARGET].astype("float64").to_numpy()) for s, f in frames.items()}
    return frames, data


def categorical_in(features: list[str]) -> list[str]:
    return [c for c in features if c in CATEGORICAL]


def load_params() -> dict:
    params = dict(DEFAULT_PARAMS)
    if BEST_PARAMS_JSON.exists():
        params.update(json.loads(BEST_PARAMS_JSON.read_text())["params"])
    return params


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _fit(model: lgb.LGBMRegressor, X_tr, y_tr, X_val, y_val) -> lgb.LGBMRegressor:
    model.fit(
        X_tr, y_tr,
        eval_X=X_val, eval_y=y_val,
        categorical_feature=categorical_in(list(X_tr.columns)),
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return model


def train_quantile_model(alpha: float, X_tr, y_tr, X_val, y_val, params: dict | None = None) -> lgb.LGBMRegressor:
    p = {**(params or DEFAULT_PARAMS), "objective": "quantile", "alpha": alpha}
    return _fit(lgb.LGBMRegressor(**p), X_tr, y_tr, X_val, y_val)


def train_quantile_models(X_tr, y_tr, X_val, y_val, params: dict | None = None,
                          quantiles=QUANTILES) -> dict[float, lgb.LGBMRegressor]:
    return {q: train_quantile_model(q, X_tr, y_tr, X_val, y_val, params) for q in quantiles}


def train_baseline(X_tr, y_tr, X_val, y_val, params: dict | None = None) -> lgb.LGBMRegressor:
    """Conventional point-estimate model (L2 loss): the 'before' in the story."""
    p = {**(params or DEFAULT_PARAMS), "objective": "regression"}
    return _fit(lgb.LGBMRegressor(**p), X_tr, y_tr, X_val, y_val)


def predict_quantiles(models: dict[float, lgb.LGBMRegressor], X) -> tuple[np.ndarray, float]:
    """(n, len(quantiles)) predictions with monotone rearrangement + crossing rate."""
    qs = sorted(models)
    raw = np.column_stack([models[q].predict(X) for q in qs])
    crossing_rate = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
    return np.sort(raw, axis=1), crossing_rate


def main() -> None:
    from src.models.evaluate import pinball

    _, data = load_splits()
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = data["train"], data["val"], data["test"]
    params = load_params()
    models = train_quantile_models(X_tr, y_tr, X_val, y_val, params)
    preds, crossing = predict_quantiles(models, X_te)
    for i, q in enumerate(QUANTILES):
        print(f"q={q}: best_iter={models[q].best_iteration_:4d}  test pinball={pinball(y_te, preds[:, i], q):.3f}")
    print(f"quantile crossing rate (before rearrangement): {crossing:.2%}")


if __name__ == "__main__":
    main()
