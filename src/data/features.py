"""Feature engineering + chronological train/val/calib/test split.

Distance: haversine vs GeoPandas projected CRS
---------------------------------------------
* `haversine_km` — great-circle distance on a sphere. Error vs the WGS84
  ellipsoid is <0.5%, i.e. a few metres at food-delivery scale (1-20 km).
  Pure numpy, vectorised, no system dependencies.
* `projected_distance_km` — GeoPandas: project both points into a metric CRS
  (EPSG:7755, "WGS 84 / India NSF LCC") and take Euclidean distance. That gives
  you *planar* geometry, useful for spatial joins, buffers and polygons (e.g.
  "is the customer inside this delivery zone"). For plain point-to-point distance
  it is NOT more accurate: a single pan-India projection distorts scale by up to
  ~1-2% far from its standard parallels, which is worse than haversine. It is
  also ~55x slower (110 ms vs 2 ms on 42k rows). We keep it for comparison and use haversine in the model.
* Neither is road distance. Riders follow streets, typically 1.2-1.5x the
  straight-line distance. `manhattan_km` (|dlat| + |dlon| in km) is a cheap
  proxy for grid-like road networks. A production system would call a routing
  engine (OSRM / Google Directions) for real road distance and live traffic.

Why a time-based split, not random
----------------------------------
In production the model is trained on the past and predicts the future. A random
split lets the model "see" orders from the same day/hour/rider as the test rows
(shared weather events, traffic jams, festival days), which inflates test scores
and, critically, makes conformal calibration look better than it will be live.
So we sort by order timestamp and cut on *day boundaries* (an order day never
straddles two splits):

    train (~60%) | val (~10%) | calib (~15%) | test (~15%)
      fit models   early stop    conformal      final, untouched
                   / tuning      calibration    report

The calibration set is disjoint from train AND val: conformal guarantees only
hold if the calibration residuals come from data the models never fit or tuned on.

Usage:
    uv run python -m src.data.features
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CLEAN_PARQUET = ROOT / "data" / "processed" / "clean.parquet"
FEATURES_PARQUET = ROOT / "data" / "processed" / "features.parquet"
SCHEMA_JSON = ROOT / "data" / "processed" / "feature_schema.json"
ARTIFACTS_PATH = ROOT / "data" / "processed" / "feature_artifacts.joblib"

EARTH_RADIUS_KM = 6371.0088
INDIA_LCC_CRS = "EPSG:7755"

# Rush-hour windows (local time, [start, end) hours). Indian food-delivery peaks:
RUSH_WINDOWS = {"lunch": (12, 14), "dinner": (19, 21)}  # 12:00-14:00, 19:00-21:00

TRAFFIC_ORDINAL = {"low": 0, "medium": 1, "high": 2, "jam": 3}
WEATHER_BUCKET = {
    "sunny": "clear",
    "cloudy": "mild", "windy": "mild",
    "fog": "severe", "stormy": "severe", "sandstorms": "severe",
}

SPLIT_FRACTIONS = {"train": 0.60, "val": 0.10, "calib": 0.15, "test": 0.15}
N_GEO_CLUSTERS = 20

TARGET = "eta_min"

# Features available when the customer PLACES the order (rider assignment is
# assumed near-instant, so rider/vehicle attributes are included).
#
# DATASET CAVEAT: in this Kaggle data every customer location is the restaurant
# location + (k*0.01 deg, k*0.01 deg) for one of 12 values of k (verified in
# notebooks/01_eda.ipynb). So bearing is constant (~43 deg) and manhattan_km is
# a fixed multiple of haversine. Both are still computed (they'd matter on real
# data) but are excluded here because they carry zero extra information.
FEATURES_AT_ORDER = [
    "distance_km",
    "hour", "minute_of_day", "day_of_week", "is_weekend", "is_rush_hour",
    "traffic_level", "weather", "weather_bucket", "festival",
    "city", "city_tier", "geo_cluster",
    "order_type", "vehicle_type", "vehicle_condition", "multiple_deliveries",
    "rider_age", "rider_rating", "order_time_imputed",
]
# prep_min = pickup - order is only known AFTER pickup. Including it in an
# "at order" ETA is target leakage-by-timing. It's legitimate for a re-estimate
# issued at pickup, so we expose it as a separate feature set.
FEATURES_AT_PICKUP = FEATURES_AT_ORDER + ["prep_min"]

CATEGORICAL = ["weather", "weather_bucket", "festival", "city", "city_tier",
               "geo_cluster", "order_type", "vehicle_type", "rush_window"]


# --------------------------------------------------------------------------- #
# Distances
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def manhattan_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """|north-south| + |east-west| leg lengths: a road-grid distance proxy."""
    ns = haversine_km(lat1, lon1, lat2, lon1)
    ew = haversine_km(lat2, lon1, lat2, lon2)
    return ns + ew


def bearing_deg(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    y = np.sin(lon2 - lon1) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def projected_distance_km(df: pd.DataFrame) -> pd.Series:
    """GeoPandas alternative (see module docstring for why it isn't the default)."""
    import geopandas as gpd

    a = gpd.GeoSeries(gpd.points_from_xy(df["rest_lon"], df["rest_lat"]), crs="EPSG:4326").to_crs(INDIA_LCC_CRS)
    b = gpd.GeoSeries(gpd.points_from_xy(df["cust_lon"], df["cust_lat"]), crs="EPSG:4326").to_crs(INDIA_LCC_CRS)
    return pd.Series(a.distance(b, align=False).to_numpy() / 1000.0, index=df.index)


# --------------------------------------------------------------------------- #
# Row-wise (stateless) features — safe to compute before splitting
# --------------------------------------------------------------------------- #
def rush_window(hour: pd.Series) -> pd.Series:
    out = pd.Series("off_peak", index=hour.index, dtype="object")
    for name, (start, end) in RUSH_WINDOWS.items():
        out[(hour >= start) & (hour < end)] = name
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    args = (df["rest_lat"], df["rest_lon"], df["cust_lat"], df["cust_lon"])
    df["distance_km"] = haversine_km(*args)
    df["manhattan_km"] = manhattan_km(*args)
    df["bearing_deg"] = bearing_deg(*args)

    ts = df["order_ts"]
    df["hour"] = ts.dt.hour
    df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["rush_window"] = rush_window(df["hour"])
    df["is_rush_hour"] = (df["rush_window"] != "off_peak").astype(int)

    # Ordinal: traffic has a natural order, so a single monotone feature lets
    # trees split "jam vs not" with one threshold. NaN stays NaN (LightGBM routes it).
    df["traffic_level"] = df["traffic"].map(TRAFFIC_ORDINAL).astype("float64")
    df["weather_bucket"] = df["weather"].map(WEATHER_BUCKET)

    df["distance_bucket"] = pd.cut(
        df["distance_km"], [0, 3, 6, 10, 15, np.inf], labels=["0-3km", "3-6km", "6-10km", "10-15km", "15km+"]
    ).astype("object")
    df["order_time_imputed"] = df["order_time_imputed"].astype(int)
    return df


# --------------------------------------------------------------------------- #
# Split + stateful features (fit on TRAIN ONLY)
# --------------------------------------------------------------------------- #
def time_split(df: pd.DataFrame, fractions: dict[str, float] = SPLIT_FRACTIONS) -> pd.Series:
    """Assign splits by cumulative row share, snapping cut points to whole days."""
    day = df["order_ts"].dt.normalize()
    share = day.value_counts().sort_index().cumsum() / len(df)
    cuts, acc = [], 0.0
    for f in list(fractions.values())[:-1]:
        acc += f
        cuts.append(share.index[np.searchsorted(share.to_numpy(), acc)])  # first day reaching acc
    names = list(fractions)
    split = pd.Series(names[-1], index=df.index, dtype="object")
    for name, cut in reversed(list(zip(names[:-1], cuts))):
        split[day <= cut] = name
    return split


def fit_geo_clusters(train: pd.DataFrame, k: int = N_GEO_CLUSTERS):
    """KMeans on restaurant coords, fit on train only (fitting on all rows would
    leak test-set geography into the training features)."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(train[["rest_lat", "rest_lon"]].to_numpy())


def _as_str(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), "unknown").astype(str)


def apply_categories(df: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast categoricals to a FIXED vocabulary. Unseen levels -> NaN (LightGBM: missing)."""
    df = df.copy()
    for c, cats in categories.items():
        df[c] = pd.Categorical(_as_str(df[c]), categories=cats)
    return df


def build(clean: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full offline path. Returns the feature table and the fitted artifacts
    (KMeans + category vocabularies, both learned on TRAIN only) that the
    online path (`featurize`) reuses, so training and serving share one code path."""
    df = add_features(clean.sort_values("order_ts").reset_index(drop=True))
    df["split"] = time_split(df)
    train_mask = df["split"] == "train"
    km = fit_geo_clusters(df[train_mask])
    df["geo_cluster"] = km.predict(df[["rest_lat", "rest_lon"]].to_numpy()).astype(str)
    categories = {c: sorted(_as_str(df.loc[train_mask, c]).unique()) for c in CATEGORICAL}
    return apply_categories(df, categories), {"kmeans": km, "categories": categories}


def featurize(orders: pd.DataFrame, artifacts: dict) -> pd.DataFrame:
    """Online/serving path: raw-ish order rows (clean schema) -> model features."""
    df = add_features(orders)
    df["geo_cluster"] = artifacts["kmeans"].predict(df[["rest_lat", "rest_lon"]].to_numpy()).astype(str)
    return apply_categories(df, artifacts["categories"])


def main() -> None:
    import joblib

    clean = pd.read_parquet(CLEAN_PARQUET)
    df, artifacts = build(clean)
    df.to_parquet(FEATURES_PARQUET, index=False)
    joblib.dump(artifacts, ARTIFACTS_PATH)
    SCHEMA_JSON.write_text(json.dumps({
        "target": TARGET,
        "features_at_order": FEATURES_AT_ORDER,
        "features_at_pickup": FEATURES_AT_PICKUP,
        "categorical": CATEGORICAL,
        "rush_windows": RUSH_WINDOWS,
        "split_fractions": SPLIT_FRACTIONS,
    }, indent=2))

    print(f"[features] wrote {FEATURES_PARQUET}  shape={df.shape}")
    summary = df.groupby("split", observed=True).agg(
        rows=("order_id", "size"), start=("order_ts", "min"), end=("order_ts", "max"),
        eta_mean=(TARGET, "mean"),
    ).reindex(list(SPLIT_FRACTIONS))
    print(summary.to_string())


if __name__ == "__main__":
    main()
