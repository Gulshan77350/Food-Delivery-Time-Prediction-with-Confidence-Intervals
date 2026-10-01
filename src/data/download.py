"""Download the food-delivery dataset from Kaggle into data/raw/.

Candidate datasets that were evaluated (all public on Kaggle):

1. gauravmalik26/food-delivery-dataset          <-- CHOSEN
   ~45.6k labelled orders (train.csv). Restaurant + delivery lat/long, order date,
   order time AND pickup time (-> prep time), weather, road-traffic density,
   vehicle type/condition, multiple deliveries, festival flag, city tier,
   rider age/rating, and target `Time_taken(min)`. Raw and messy (strings like
   "conditions Sunny", "(min) 24", "NaN "), which is good for showing cleaning.
   NOTE: its test.csv has NO target, so we only use train.csv and make our own
   time-based train/val/calib/test split.

2. bhanupratapbiswas/zomato-delivery-operations-analytics-dataset
   Same underlying source (Indian food-delivery hackathon data), pre-cleaned.
   Kept as a fallback slug; less interesting cleaning story.

3. denkuznetz/food-delivery-time-prediction
   Rejected: ~1k synthetic rows, distance given directly, no coordinates/timestamps.

Auth: kagglehub/kaggle read credentials from ~/.kaggle/kaggle.json
(Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json) or the KAGGLE_USERNAME /
KAGGLE_KEY environment variables. Create a token at
https://www.kaggle.com/settings -> "API" -> "Create New Token".

Usage:
    uv run python -m src.data.download
    uv run python -m src.data.download --slug bhanupratapbiswas/zomato-delivery-operations-analytics-dataset
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DEFAULT_SLUG = "gauravmalik26/food-delivery-dataset"
FALLBACK_SLUGS = ["bhanupratapbiswas/zomato-delivery-operations-analytics-dataset"]


def _download_kagglehub(slug: str) -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(slug))


def _download_kaggle_api(slug: str, dest: Path) -> Path:
    # The classic `kaggle` package authenticates on import and raises if no creds.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(slug, path=str(dest), unzip=False, quiet=False)
    return dest


def _unzip_all(folder: Path) -> None:
    for z in folder.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        z.unlink()


def download(slug: str = DEFAULT_SLUG, dest: Path = RAW_DIR) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for candidate in [slug, *[s for s in FALLBACK_SLUGS if s != slug]]:
        try:
            print(f"[download] trying kagglehub: {candidate}")
            src = _download_kagglehub(candidate)
            for f in src.rglob("*"):
                if f.is_file():
                    shutil.copy2(f, dest / f.name)
        except Exception as e:  # noqa: BLE001 - try the next backend
            errors.append(f"kagglehub({candidate}): {e!r}")
            try:
                print(f"[download] trying kaggle API: {candidate}")
                _download_kaggle_api(candidate, dest)
            except Exception as e2:  # noqa: BLE001
                errors.append(f"kaggle-api({candidate}): {e2!r}")
                continue
        _unzip_all(dest)
        files = sorted(p for p in dest.iterdir() if p.suffix.lower() == ".csv")
        if files:
            (dest / "SOURCE.txt").write_text(f"https://www.kaggle.com/datasets/{candidate}\n")
            print(f"[download] OK from {candidate}: {[f.name for f in files]}")
            return files
    print("[download] FAILED. Errors:\n  " + "\n  ".join(errors), file=sys.stderr)
    print(
        "\nFix: put kaggle.json in ~/.kaggle/ (or set KAGGLE_USERNAME/KAGGLE_KEY),\n"
        f"or download manually from https://www.kaggle.com/datasets/{slug}\n"
        f"and unzip the CSVs into {dest}",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", default=DEFAULT_SLUG)
    args = ap.parse_args()
    download(args.slug)
