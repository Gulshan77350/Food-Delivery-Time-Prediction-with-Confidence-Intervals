"""Thin MLflow helpers: one place that decides where runs are stored and how
DataFrames / figures / dicts are logged.

Storage: SQLite backend (mlflow.db) + local artifact folder (mlruns/), both at
the repo root and gitignored. View with:

    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
ARTIFACT_ROOT = (ROOT / "mlruns").as_uri()
EXPERIMENT = "eta-prediction"


def setup(experiment: str = EXPERIMENT) -> str:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
    mlflow.set_experiment(experiment)
    return experiment


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - not a git repo / git missing
        return "no-git"


@contextmanager
def start_run(run_name: str, tags: dict | None = None, nested: bool = False):
    with mlflow.start_run(run_name=run_name, nested=nested) as run:
        mlflow.set_tags({"git_sha": git_sha(), **(tags or {})})
        yield run


def log_params_flat(params: dict, prefix: str = "") -> None:
    mlflow.log_params({f"{prefix}{k}": v for k, v in params.items()})


def log_metrics_flat(metrics: dict, prefix: str = "") -> None:
    mlflow.log_metrics({f"{prefix}{k}".replace("@", "_"): float(v) for k, v in metrics.items()
                        if isinstance(v, (int, float)) and v == v})  # skip NaN


def log_df(df: pd.DataFrame, name: str, artifact_path: str = "tables") -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / name
        (df.to_csv(p, index=False) if name.endswith(".csv") else df.to_markdown(p))
        mlflow.log_artifact(str(p), artifact_path)


def log_figure(fig, name: str, artifact_path: str = "figures") -> None:
    mlflow.log_figure(fig, f"{artifact_path}/{name}")
