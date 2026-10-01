# End-to-end reproduction on Windows PowerShell:  .\run_all.ps1   (add -Tune to re-run Optuna, -Fresh to wipe MLflow)
param([switch]$Tune, [switch]$Fresh, [int]$Trials = 30)
# "Continue", not "Stop": Windows PowerShell 5.1 turns any native stderr line (MLflow/LightGBM INFO logs)
# into an error record when output is redirected. Failures are detected via $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
$py = ".\.venv\Scripts\python.exe"

function Step($name, [scriptblock]$cmd) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "step failed: $name" }
}

if (-not (Test-Path $py)) { Step "env" { python -m uv sync --python 3.11 } }
if ($Fresh) { Remove-Item -Recurse -Force mlflow.db, mlruns -ErrorAction SilentlyContinue }

Step "1/6 download"  { & $py -m src.data.download }
Step "2/6 clean"     { & $py -m src.data.clean }
Step "3/6 features"  { & $py -m src.data.features }
if ($Tune -or -not (Test-Path reports\best_params.json)) { Step "4/6 tune" { & $py -m src.models.tune --trials $Trials } }
Step "5/6 train + conformal + slices (at-order features)" { & $py -m src.pipeline }
Step "5b/6 comparison run (at-pickup features)" { & $py -m src.pipeline --feature-set pickup --no-save }
Step "6/6 tests"     { & $py -m pytest -q }

Write-Host "`nDone. Next:" -ForegroundColor Green
Write-Host "  MLflow UI : .\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
Write-Host "  Demo app  : .\.venv\Scripts\streamlit run app\streamlit_app.py"
