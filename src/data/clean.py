"""Clean the raw Kaggle food-delivery CSV into a typed, analysis-ready table.

Raw quirks handled here (found by profiling data/raw/train.csv):
  * Nearly every string has trailing whitespace ("Urban ", "Jam ", "NaN ").
  * Missing values are encoded as the *string* "NaN" (sometimes "conditions NaN").
  * Weather is prefixed: "conditions Sunny" -> "sunny".
  * Target is a string: "(min) 24" -> 24.
  * Some coordinates are negative (India is N/E, so a sign-flip data-entry error)
    and some restaurant coordinates are ~0 (impossible -> dropped).
  * Rider ratings > 5 and riders aged < 18 co-occur in junk rows -> set to NaN.
  * Order time is missing for some rows; pickup time is always present.
  * Pickup can be "earlier" than order when the order crosses midnight -> +1 day.
  * The `City` column is a *tier* (Urban / Metropolitian / Semi-Urban), not a
    city. The actual city is encoded in the rider ID prefix: "INDORES13DEL02"
    -> "INDO" (Indore). We extract it as `city`.

Usage:
    uv run python -m src.data.clean
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import haversine_km

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "data" / "raw" / "train.csv"
CLEAN_PARQUET = ROOT / "data" / "processed" / "clean.parquet"

NA_TOKENS = {"nan", "na", "none", "null", ""}

# Rider-ID prefix -> city name (prefixes observed in the data).
CITY_CODES = {
    "AGR": "Agra", "ALH": "Allahabad", "AURG": "Aurangabad", "BANG": "Bangalore",
    "BHP": "Bhopal", "CHEN": "Chennai", "COIMB": "Coimbatore", "DEH": "Dehradun",
    "GOA": "Goa", "HYD": "Hyderabad", "INDO": "Indore", "JAP": "Jaipur",
    "KNP": "Kanpur", "KOC": "Kochi", "KOL": "Kolkata", "LUDH": "Ludhiana",
    "MUM": "Mumbai", "MYS": "Mysore", "PUNE": "Pune", "RANCHI": "Ranchi",
    "SUR": "Surat", "VAD": "Vadodara",
}

# Plausible target range. Real food delivery rarely < 5 min or > 3 h; anything
# outside is a logging error, not signal.
MIN_MINUTES, MAX_MINUTES = 5, 180
# Food delivery radius sanity cap. Beyond this the coordinates are wrong.
MAX_DISTANCE_KM = 50.0

RENAME = {
    "ID": "order_id",
    "Delivery_person_ID": "rider_id",
    "Delivery_person_Age": "rider_age",
    "Delivery_person_Ratings": "rider_rating",
    "Restaurant_latitude": "rest_lat",
    "Restaurant_longitude": "rest_lon",
    "Delivery_location_latitude": "cust_lat",
    "Delivery_location_longitude": "cust_lon",
    "Order_Date": "order_date",
    "Time_Orderd": "time_ordered",
    "Time_Order_picked": "time_picked",
    "Weatherconditions": "weather",
    "Road_traffic_density": "traffic",
    "Vehicle_condition": "vehicle_condition",
    "Type_of_order": "order_type",
    "Type_of_vehicle": "vehicle_type",
    "multiple_deliveries": "multiple_deliveries",
    "Festival": "festival",
    "City": "city_tier",
    "Time_taken(min)": "eta_min",
}


def _to_na(s: pd.Series) -> pd.Series:
    """Strip whitespace and convert NaN-like tokens to real missing values."""
    s = s.astype("string").str.strip()
    return s.mask(s.str.lower().isin(NA_TOKENS))


def _parse_clock(date: pd.Series, clock: pd.Series) -> pd.Series:
    """Combine a date with an 'HH:MM[:SS]' string. Anything else -> NaT.

    A handful of raw rows store time as an Excel day-fraction (e.g. "0.458333");
    we convert those rather than silently dropping them.
    """
    clock = clock.copy()
    frac = pd.to_numeric(clock, errors="coerce")
    is_frac = frac.notna() & (frac >= 0) & (frac < 1)
    secs = (frac[is_frac] * 86400).round().astype("int64")
    clock.loc[is_frac] = (
        (secs // 3600).astype(str).str.zfill(2) + ":"
        + ((secs % 3600) // 60).astype(str).str.zfill(2) + ":00"
    )
    ok = clock.str.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?").fillna(False)
    ts = pd.Series(pd.NaT, index=clock.index, dtype="datetime64[ns]")
    ts.loc[ok] = pd.to_datetime(
        date[ok].dt.strftime("%Y-%m-%d") + " " + clock[ok], errors="coerce"
    )
    return ts


def clean(raw: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    log: list[str] = []
    n0 = len(raw)
    df = raw.rename(columns=RENAME)[list(RENAME.values())].copy()

    # --- 1. strings: strip + real NaN -------------------------------------------
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = _to_na(df[c])

    df["weather"] = df["weather"].str.replace(r"^conditions\s*", "", regex=True)
    df["weather"] = _to_na(df["weather"]).str.lower()
    for c in ["traffic", "order_type", "vehicle_type", "festival", "city_tier"]:
        df[c] = df[c].str.lower()
    df["city_tier"] = df["city_tier"].replace({"metropolitian": "metropolitan"})

    # --- 2. numeric types --------------------------------------------------------
    df["eta_min"] = pd.to_numeric(
        df["eta_min"].astype("string").str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce"
    )
    for c in ["rider_age", "rider_rating", "multiple_deliveries", "vehicle_condition",
              "rest_lat", "rest_lon", "cust_lat", "cust_lon"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    # Junk rider attributes (age 15 / rating 6 rows) -> missing, not dropped:
    # the order itself (coords, times, target) is still usable.
    bad_rider = (df["rider_rating"] > 5) | (df["rider_age"] < 18)
    df.loc[bad_rider, ["rider_rating", "rider_age"]] = np.nan
    log.append(f"rider age/rating set to NaN (invalid): {int(bad_rider.sum())}")

    # --- 3. coordinates ----------------------------------------------------------
    neg = (df[["rest_lat", "rest_lon", "cust_lat", "cust_lon"]] < 0).any(axis=1)
    for c in ["rest_lat", "rest_lon", "cust_lat", "cust_lon"]:
        df[c] = df[c].abs()
    log.append(f"negative coordinates sign-flipped: {int(neg.sum())}")

    # India bounding box (approx). Outside -> coordinates are garbage.
    in_india = (
        df["rest_lat"].between(6, 37) & df["rest_lon"].between(68, 98)
        & df["cust_lat"].between(6, 37) & df["cust_lon"].between(68, 98)
    )
    log.append(f"dropped: coordinates outside India (e.g. ~0.0): {int((~in_india).sum())}")
    df = df[in_india].copy()

    dist = haversine_km(df["rest_lat"], df["rest_lon"], df["cust_lat"], df["cust_lon"])
    too_far = dist > MAX_DISTANCE_KM
    log.append(f"dropped: restaurant->customer > {MAX_DISTANCE_KM:.0f} km: {int(too_far.sum())}")
    df = df[~too_far].copy()

    # --- 4. timestamps -----------------------------------------------------------
    df["order_date"] = pd.to_datetime(df["order_date"], format="%d-%m-%Y", errors="coerce")
    df = df[df["order_date"].notna()].copy()
    df["order_ts"] = _parse_clock(df["order_date"], df["time_ordered"])
    df["pickup_ts"] = _parse_clock(df["order_date"], df["time_picked"])

    # Midnight crossover: pickup "before" order -> pickup is next day.
    cross = df["pickup_ts"] < df["order_ts"]
    df.loc[cross, "pickup_ts"] += pd.Timedelta(days=1)
    log.append(f"midnight crossover fixed (+1 day to pickup): {int(cross.sum())}")

    df["prep_min"] = (df["pickup_ts"] - df["order_ts"]).dt.total_seconds() / 60
    bad_prep = df["prep_min"] > 120
    df.loc[bad_prep, ["prep_min", "order_ts"]] = np.nan
    log.append(f"implausible prep time (>2h) -> order_ts NaN: {int(bad_prep.sum())}")

    # Missing order time: impute from pickup - median prep, and FLAG it so the
    # model/analysis can tell. Needed because order_ts drives the time split and
    # the hour-of-day features.
    df["order_time_imputed"] = df["order_ts"].isna()
    med_prep = pd.Timedelta(minutes=float(df["prep_min"].median()))
    df.loc[df["order_time_imputed"], "order_ts"] = df.loc[df["order_time_imputed"], "pickup_ts"] - med_prep
    log.append(f"order time missing -> imputed from pickup (flagged): {int(df['order_time_imputed'].sum())}")
    df = df[df["order_ts"].notna()].copy()

    # --- 5. target ---------------------------------------------------------------
    bad_y = df["eta_min"].isna() | ~df["eta_min"].between(MIN_MINUTES, MAX_MINUTES)
    log.append(f"dropped: missing/absurd target (outside {MIN_MINUTES}-{MAX_MINUTES} min): {int(bad_y.sum())}")
    df = df[~bad_y].copy()

    # --- 6. city from rider ID ---------------------------------------------------
    code = df["rider_id"].str.extract(r"^([A-Z]+?)RES", flags=re.I)[0].str.upper()
    df["city"] = code.map(CITY_CODES).fillna(code).fillna("unknown").astype("string")

    # --- 7. de-dup ---------------------------------------------------------------
    dup = df["order_id"].duplicated()
    log.append(f"dropped: duplicate order_id: {int(dup.sum())}")
    df = df[~dup]

    df = df.drop(columns=["time_ordered", "time_picked"]).sort_values("order_ts").reset_index(drop=True)
    if verbose:
        print(f"[clean] rows in: {n0:,}")
        for line in log:
            print(f"[clean]   {line}")
        print(f"[clean] rows out: {len(df):,} ({len(df) / n0:.1%} kept)")
    return df


def main() -> None:
    raw = pd.read_csv(RAW_CSV, dtype=str)
    df = clean(raw)
    CLEAN_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CLEAN_PARQUET, index=False)
    print(f"[clean] wrote {CLEAN_PARQUET}")


if __name__ == "__main__":
    main()
