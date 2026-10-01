#!/usr/bin/env bash
# End-to-end reproduction on macOS / Linux:  ./run_all.sh   (TUNE=1 to re-run Optuna, FRESH=1 to wipe MLflow)
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
[ -d .venv ] || uv sync --python 3.11
[ "${FRESH:-0}" = 1 ] && rm -rf mlflow.db mlruns
run() { echo -e "\n=== $1 ==="; shift; uv run "$@"; }

run "1/6 download"  python -m src.data.download
run "2/6 clean"     python -m src.data.clean
run "3/6 features"  python -m src.data.features
if [ "${TUNE:-0}" = 1 ] || [ ! -f reports/best_params.json ]; then
  run "4/6 tune" python -m src.models.tune --trials "${TRIALS:-30}"
fi
run "5/6 train + conformal + slices" python -m src.pipeline
run "5b/6 comparison run (pickup features)" python -m src.pipeline --feature-set pickup --no-save
run "6/6 tests" python -m pytest -q

echo -e "\nMLflow UI : uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
echo      "Demo app  : uv run streamlit run app/streamlit_app.py"
