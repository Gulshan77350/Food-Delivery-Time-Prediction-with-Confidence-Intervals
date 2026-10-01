import numpy as np
import pandas as pd
import pytest
from haversine import haversine

from src.data.clean import clean
from src.data.features import (
    build, featurize, haversine_km, manhattan_km, rush_window, time_split,
)


def _raw_row(**overrides):
    row = {
        "ID": "0x1", "Delivery_person_ID": "INDORES13DEL02", "Delivery_person_Age": "37 ",
        "Delivery_person_Ratings": "4.9 ", "Restaurant_latitude": "22.745049",
        "Restaurant_longitude": "75.892471", "Delivery_location_latitude": "22.765049",
        "Delivery_location_longitude": "75.912471", "Order_Date": "19-03-2022",
        "Time_Orderd": "11:30:00 ", "Time_Order_picked": "11:45:00 ",
        "Weatherconditions": "conditions Sunny", "Road_traffic_density": "High ",
        "Vehicle_condition": "2", "Type_of_order": "Snack ", "Type_of_vehicle": "motorcycle ",
        "multiple_deliveries": "0 ", "Festival": "No ", "City": "Urban ", "Time_taken(min)": "(min) 24",
    }
    row.update(overrides)
    return row


def test_haversine_matches_reference_library():
    a, b = (12.9716, 77.5946), (13.0827, 80.2707)  # Bangalore -> Chennai
    assert haversine_km(*a, *b) == pytest.approx(haversine(a, b), rel=1e-3)


def test_manhattan_ge_haversine():
    d = haversine_km(22.7, 75.8, 22.8, 75.9)
    assert manhattan_km(22.7, 75.8, 22.8, 75.9) >= d


def test_clean_parses_messy_strings():
    df = clean(pd.DataFrame([_raw_row()]), verbose=False)
    r = df.iloc[0]
    assert r["eta_min"] == 24
    assert r["weather"] == "sunny" and r["traffic"] == "high" and r["city_tier"] == "urban"
    assert r["city"] == "Indore"
    assert r["prep_min"] == 15


def test_clean_handles_nan_strings_negatives_and_midnight():
    raw = pd.DataFrame([
        _raw_row(ID="a", Weatherconditions="conditions NaN", Road_traffic_density="NaN "),
        _raw_row(ID="b", Restaurant_latitude="-22.745049"),
        _raw_row(ID="c", Time_Orderd="23:55:00", Time_Order_picked="00:05:00"),
        _raw_row(ID="d", Restaurant_latitude="0.0", Restaurant_longitude="0.0"),   # dropped
        _raw_row(ID="e", **{"Time_taken(min)": "(min) 999"}),                       # dropped
        _raw_row(ID="f", Time_Orderd="NaN "),                                       # imputed
    ])
    df = clean(raw, verbose=False).set_index("order_id")
    assert set(df.index) == {"a", "b", "c", "f"}
    assert pd.isna(df.loc["a", "weather"]) and pd.isna(df.loc["a", "traffic"])
    assert df.loc["b", "rest_lat"] > 0
    assert df.loc["c", "prep_min"] == 10
    assert bool(df.loc["f", "order_time_imputed"])


def test_rush_window_boundaries():
    h = pd.Series([11, 12, 13, 14, 18, 19, 20, 21])
    assert rush_window(h).tolist() == [
        "off_peak", "lunch", "lunch", "off_peak", "off_peak", "dinner", "dinner", "off_peak"]


def test_time_split_is_chronological_and_day_aligned():
    ts = pd.date_range("2022-03-01", periods=40 * 24, freq="h")
    df = pd.DataFrame({"order_ts": ts})
    split = time_split(df)
    order = ["train", "val", "calib", "test"]
    ranks = split.map({s: i for i, s in enumerate(order)})
    assert ranks.is_monotonic_increasing                     # no going back in time
    assert set(split) == set(order)
    per_day = df.assign(s=split).groupby(df["order_ts"].dt.date)["s"].nunique()
    assert (per_day == 1).all()                              # no day straddles two splits


def test_categories_fixed_from_train():
    rng = np.random.default_rng(0)
    n = 400
    raw = pd.DataFrame([_raw_row(ID=str(i)) for i in range(n)])
    raw["Order_Date"] = pd.date_range("2022-03-01", periods=n, freq="3h").strftime("%d-%m-%Y")
    raw["Restaurant_latitude"] = (22.7 + rng.normal(0, 0.05, n)).astype(str)
    df, artifacts = build(clean(raw, verbose=False))
    for split in ["train", "val", "calib", "test"]:
        part = df[df["split"] == split]
        assert list(part["weather"].cat.categories) == list(df["weather"].cat.categories)
    assert artifacts["categories"]["weather"] == list(df["weather"].cat.categories)


def test_online_featurize_matches_offline_build():
    """Train/serve skew guard: the serving path must reproduce offline features."""
    rng = np.random.default_rng(1)
    n = 300
    raw = pd.DataFrame([_raw_row(ID=str(i)) for i in range(n)])
    raw["Order_Date"] = pd.date_range("2022-03-01", periods=n, freq="3h").strftime("%d-%m-%Y")
    raw["Restaurant_latitude"] = (22.7 + rng.normal(0, 0.05, n)).astype(str)
    cleaned = clean(raw, verbose=False)
    offline, artifacts = build(cleaned)
    online = featurize(cleaned.sort_values("order_ts").reset_index(drop=True), artifacts)
    for c in ["distance_km", "hour", "is_rush_hour", "traffic_level", "geo_cluster", "weather"]:
        pd.testing.assert_series_equal(offline[c], online[c], check_names=False)
