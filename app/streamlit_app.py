"""Streamlit demo: delivery ETA as a calibrated interval.

    uv run streamlit run app/streamlit_app.py

Requires the artifacts produced by `python -m src.pipeline`:
    models/eta_bundle.joblib, reports/test_predictions.parquet,
    reports/error_slices.csv, reports/calibration_curve_order.csv, reports/comparison_order.csv
"""

from __future__ import annotations

import json
import sys
from datetime import date, time
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURES_PARQUET, featurize, haversine_km  # noqa: E402
from src.models.train_quantile import to_matrix  # noqa: E402

BUNDLE = ROOT / "models" / "eta_bundle.joblib"
REPORTS = ROOT / "reports"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"
SLICE_SEGMENTS = ["city", "weather", "traffic", "rush_window", "distance_bucket", "festival", "city_tier"]

st.set_page_config(page_title="ETA with Uncertainty", page_icon="🛵", layout="wide")


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE)


@st.cache_data
def load_reports():
    return {
        "pred": pd.read_parquet(REPORTS / "test_predictions.parquet"),
        "slices": pd.read_csv(REPORTS / "error_slices.csv"),
        "curve": pd.read_csv(REPORTS / "calibration_curve_order.csv"),
        "comparison": pd.read_csv(REPORTS / "comparison_order.csv", index_col=0),
        "metrics": json.loads((REPORTS / "metrics_order.json").read_text()),
    }


@st.cache_data
def load_restaurants() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PARQUET, columns=["split", "city", "city_tier", "rest_lat", "rest_lon"])
    df = df[df["split"] == "train"]
    df["city"] = df["city"].astype(str)
    df["city_tier"] = df["city_tier"].astype(str)
    g = df.groupby(["city", "rest_lat", "rest_lon"]).agg(orders=("city_tier", "size"),
                                                         city_tier=("city_tier", lambda s: s.mode().iat[0]))
    return g.reset_index().sort_values(["city", "orders"], ascending=[True, False])


if not BUNDLE.exists():
    st.error("No trained model found. Run `uv run python -m src.pipeline` first.")
    st.stop()

bundle = load_bundle()
R = load_reports()
conf = bundle["confidence_level"]
P_LO, P_HI = round((1 - conf) / 2 * 100), round((1 + conf) / 2 * 100)
cats = bundle["feature_artifacts"]["categories"]


# --------------------------------------------------------------------------- #
# Prediction helpers
# --------------------------------------------------------------------------- #
def predict_frame(feat_df: pd.DataFrame) -> pd.DataFrame:
    X = to_matrix(feat_df, bundle["features"])
    out = bundle["cqr"].predict(X)
    qm = bundle["quantile_models"]
    qs = sorted(qm)
    raw = np.sort(np.column_stack([qm[q].predict(X) for q in qs]), axis=1)
    out["raw_lo"], out["raw_hi"] = raw[:, 0], raw[:, -1]
    out["baseline"] = bundle["baseline"].predict(X)
    return out


def customer_point(lat: float, lon: float, km: float) -> tuple[float, float]:
    """Customer at `km` along a NE diagonal (the dataset's customer points are
    restaurant + equal lat/lon offsets, so we stay inside the training geometry)."""
    lo, hi = 0.0, 0.5
    for _ in range(50):
        mid = (lo + hi) / 2
        if haversine_km(lat, lon, lat + mid, lon + mid) < km:
            lo = mid
        else:
            hi = mid
    return lat + lo, lon + lo


def segment_notes(row: dict) -> pd.DataFrame:
    s = R["slices"]
    keys = {"city": row["city"], "weather": row["weather"], "traffic": row["traffic"],
            "rush_window": row["rush_window"], "distance_bucket": row["distance_bucket"],
            "festival": row["festival"], "city_tier": row["city_tier"]}
    rows = []
    for seg, val in keys.items():
        m = s[(s["segment"] == seg) & (s["value"].astype(str) == str(val))]
        if len(m):
            r = m.iloc[0]
            status = ("⚠️ under-covered" if (r["gap_z"] < -2 and r["n"] >= 100)
                      else "ℹ️ small sample" if r["n"] < 100 else "✅ on target")
            rows.append({"segment": seg, "value": val, "test orders": int(r["n"]),
                         "coverage": r["coverage"], "MAE (min)": r["mae_p50"], "status": status})
    return pd.DataFrame(rows)


def deck(rest: tuple[float, float], cust: tuple[float, float], context: pd.DataFrame | None = None):
    pts = pd.DataFrame([
        {"name": "Restaurant", "lat": rest[0], "lon": rest[1], "color": [42, 120, 214], "r": 120},
        {"name": "Customer", "lat": cust[0], "lon": cust[1], "color": [235, 104, 52], "r": 120},
    ])
    layers = []
    if context is not None and len(context):
        layers.append(pdk.Layer("ScatterplotLayer", context, get_position=["rest_lon", "rest_lat"],
                                get_fill_color=[138, 137, 132, 110], get_radius=60,
                                radius_min_pixels=3, pickable=False))
    layers += [
        pdk.Layer("LineLayer", pd.DataFrame([{"s": [rest[1], rest[0]], "t": [cust[1], cust[0]]}]),
                  get_source_position="s", get_target_position="t", get_color=[82, 81, 78], get_width=3),
        pdk.Layer("ScatterplotLayer", pts, get_position=["lon", "lat"], get_fill_color="color",
                  get_radius="r", radius_min_pixels=7, pickable=True, stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2),
    ]
    km = float(haversine_km(*rest, *cust))
    zoom = float(np.clip(13.6 - np.log2(max(km, 0.5)), 9, 15))  # fit the trip in the frame
    view = pdk.ViewState(latitude=(rest[0] + cust[0]) / 2, longitude=(rest[1] + cust[1]) / 2, zoom=zoom)
    return pdk.Deck(layers=layers, initial_view_state=view, map_provider="carto", map_style="light",
                    tooltip={"text": "{name}"})


def interval_chart(p: pd.Series, actual: float | None = None) -> alt.Chart:
    rows = [
        {"method": f"Raw quantile P{P_LO}–P{P_HI}",
         "lo": p["raw_lo"], "hi": p["raw_hi"]},
        {"method": f"Conformal {conf:.0%} interval", "lo": p["lo"], "hi": p["hi"]},
    ]
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y("method:N", title=None, sort=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(height=18, cornerRadius=4, opacity=0.85).encode(
        x=alt.X("lo:Q", title="minutes", scale=alt.Scale(zero=False, padding=12)), x2="hi:Q",
        color=alt.Color("method:N", scale=alt.Scale(range=[MUTED, BLUE]), legend=None),
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")])
    marks = [bars]
    p50 = alt.Chart(pd.DataFrame({"x": [p["p50"]], "label": ["P50"]})).mark_rule(color="#0b0b0b", strokeWidth=2) \
        .encode(x="x:Q", tooltip=[alt.Tooltip("x:Q", title="P50", format=".1f")])
    marks.append(p50)
    if actual is not None:
        marks.append(alt.Chart(pd.DataFrame({"x": [actual]})).mark_point(
            shape="diamond", size=180, filled=True, color=ORANGE).encode(
            x="x:Q", tooltip=[alt.Tooltip("x:Q", title="actual", format=".0f")]))
    return alt.layer(*marks).properties(height=110)


def show_prediction(p: pd.Series, row: dict, actual: float | None = None):
    lo, hi, p50 = p["lo"], p["hi"], p["p50"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Promised window", f"{int(np.floor(lo))}–{int(np.ceil(hi))} min", help=f"{conf:.0%} conformal interval")
    c2.metric("Most likely (P50)", f"{p50:.0f} min")
    if actual is not None:
        inside = lo <= actual <= hi
        c3.metric("Actual", f"{actual:.0f} min", "inside window ✅" if inside else "outside window ❌",
                  delta_color="normal" if inside else "inverse")
    else:
        c3.metric("Point-estimate baseline", f"{p['baseline']:.0f} min", help="Plain LightGBM regression (L2)")
    st.altair_chart(interval_chart(p, actual), width="stretch")
    st.caption("Grey = raw quantile model (before conformal); blue = conformal interval; black rule = P50"
               + ("; orange diamond = actual." if actual is not None else "."))
    st.markdown("**Which segments does this order fall into, and how reliable are intervals there (test set)?**")
    notes = segment_notes(row)
    st.dataframe(notes.style.format({"coverage": "{:.1%}", "MAE (min)": "{:.1f}"}), hide_index=True,
                 width="stretch")
    weak = notes[notes["status"].str.startswith("⚠️")]
    if len(weak):
        st.warning("Heads-up: intervals under-cover in " + ", ".join(f"{a}={b}" for a, b in zip(weak.segment, weak.value))
                   + ". A dispatcher should pad this promise.")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("🛵 Delivery ETA with calibrated uncertainty")
st.markdown(
    f"LightGBM quantile regression + **conformalized quantile regression (CQR)**: every order gets a "
    f"*{conf:.0%} window* with a statistical coverage guarantee, not a single number.")
cmp = R["comparison"]
cqr_row = cmp.loc["4. CQR (MAPIE)"]
raw_row = cmp.loc["3. Raw quantile regression"]
base_row = cmp.loc["1. Point baseline (L2 LightGBM)"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Test coverage (target 80%)", f"{cqr_row['coverage']:.1%}", f"{(cqr_row['coverage'] - raw_row['coverage']) * 100:+.1f} pts vs raw quantiles")
k2.metric("Mean window width", f"{cqr_row['mean_width']:.1f} min")
k3.metric("P50 MAE", f"{cqr_row['p50_mae']:.2f} min", f"{cqr_row['p50_mae'] - base_row['mae']:+.2f} vs point baseline", delta_color="inverse")
k4.metric("Segments under-covered", f"{int(cqr_row['n_segments_under'])} / {int(cqr_row['n_segments_tested'])}",
          f"{int(cqr_row['n_segments_under'] - cmp.loc['2. Baseline + split conformal', 'n_segments_under']):+d} vs constant-width",
          delta_color="inverse")

tab_pred, tab_cal, tab_seg, tab_about = st.tabs(["🔮 Predict an order", "📏 Calibration", "🧩 Segments", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# Predict tab
# --------------------------------------------------------------------------- #
with tab_pred:
    mode = st.radio("Order source", ["Pick a real test-set order", "Simulate a new order"], horizontal=True)
    pred = R["pred"]

    if mode == "Pick a real test-set order":
        c1, c2 = st.columns([1, 2])
        with c1:
            city = st.selectbox("City", ["(any)"] + sorted(pred["city"].astype(str).unique()))
            sub = pred if city == "(any)" else pred[pred["city"].astype(str) == city]
            only_miss = st.checkbox("Only orders that fell outside their window")
            if only_miss:
                sub = sub[(sub["y"] < sub["lo"]) | (sub["y"] > sub["hi"])]
            if "order_idx" not in st.session_state or st.session_state.get("order_pool") != (city, only_miss):
                st.session_state.order_idx = 0
                st.session_state.order_pool = (city, only_miss)
            if st.button("🎲 Random order"):
                st.session_state.order_idx = int(np.random.default_rng().integers(len(sub)))
            idx = st.number_input("Order #", 0, max(len(sub) - 1, 0), st.session_state.order_idx, key="order_num")
            r = sub.iloc[int(idx)]
            st.caption(f"{r['order_ts']:%a %d %b %Y, %H:%M} · {r['city']} · {r['weather']} · traffic {r['traffic']} · "
                       f"{r['distance_km']:.1f} km · {r['vehicle_type']} · {int(r['multiple_deliveries']) if pd.notna(r['multiple_deliveries']) else '?'} extra drops")
            st.pydeck_chart(deck((r["rest_lat"], r["rest_lon"]), (r["cust_lat"], r["cust_lon"]),
                                 pred[pred["city"] == r["city"]][["rest_lat", "rest_lon"]].drop_duplicates()),
                            height=330)
        with c2:
            p = pd.Series({"lo": r["lo"], "hi": r["hi"], "p50": r["p50"], "raw_lo": r["raw_p10"],
                           "raw_hi": r["raw_p90"], "baseline": r["baseline"]})
            show_prediction(p, r.to_dict(), actual=float(r["y"]))

    else:
        rests = load_restaurants()
        c1, c2 = st.columns([1, 2])
        with c1:
            city = st.selectbox("City", sorted(rests["city"].unique()), index=sorted(rests["city"].unique()).index("Bangalore")
                                if "Bangalore" in rests["city"].values else 0)
            cr = rests[rests["city"] == city].reset_index(drop=True)
            ri = st.selectbox("Restaurant", cr.index, format_func=lambda i: f"#{i + 1}  ({cr.loc[i, 'rest_lat']:.4f}, "
                              f"{cr.loc[i, 'rest_lon']:.4f}) · {cr.loc[i, 'orders']} past orders")
            km = st.slider("Delivery distance (km)", 1.5, 20.0, 6.0, 0.5)
            d = st.date_input("Order date", date(2022, 4, 4))
            t = st.time_input("Order time", time(19, 30), step=900)
            a, b = st.columns(2)
            weather = a.selectbox("Weather", [c for c in cats["weather"] if c != "unknown"], index=4)
            traffic = b.selectbox("Traffic", ["low", "medium", "high", "jam"], index=3)
            vehicle = a.selectbox("Vehicle", [c for c in cats["vehicle_type"] if c != "unknown"], index=1)
            vcond = b.select_slider("Vehicle condition", [0, 1, 2, 3], 1)
            multi = a.select_slider("Multiple deliveries", [0, 1, 2, 3], 1)
            festival = b.selectbox("Festival", ["no", "yes"])
            order_type = a.selectbox("Order type", [c for c in cats["order_type"] if c != "unknown"])
            tier = b.selectbox("City tier", [c for c in cats["city_tier"] if c != "unknown"],
                               index=[c for c in cats["city_tier"] if c != "unknown"].index(cr.loc[ri, "city_tier"])
                               if cr.loc[ri, "city_tier"] in cats["city_tier"] else 0)
            age = a.slider("Rider age", 20, 39, 30)
            rating = b.slider("Rider rating", 2.5, 5.0, 4.6, 0.1)

        rest = (float(cr.loc[ri, "rest_lat"]), float(cr.loc[ri, "rest_lon"]))
        cust = customer_point(*rest, km)
        order = pd.DataFrame([{
            "rest_lat": rest[0], "rest_lon": rest[1], "cust_lat": cust[0], "cust_lon": cust[1],
            "order_ts": pd.Timestamp.combine(d, t), "weather": weather, "traffic": traffic, "festival": festival,
            "city": city, "city_tier": tier, "order_type": order_type, "vehicle_type": vehicle,
            "vehicle_condition": float(vcond), "multiple_deliveries": float(multi), "rider_age": float(age),
            "rider_rating": float(rating), "order_time_imputed": False, "prep_min": np.nan,
        }])
        feat = featurize(order, bundle["feature_artifacts"])
        p = predict_frame(feat).iloc[0]
        with c1:
            st.pydeck_chart(deck(rest, cust, cr[["rest_lat", "rest_lon"]]), height=300)
        with c2:
            row = feat.iloc[0].to_dict()
            row = {k: (str(v) if k in SLICE_SEGMENTS else v) for k, v in row.items()}
            show_prediction(p, row)
            st.caption("Customer location is placed on the dataset's synthetic NE-diagonal geometry (see About).")

# --------------------------------------------------------------------------- #
# Calibration tab
# --------------------------------------------------------------------------- #
with tab_cal:
    curve = R["curve"]
    st.subheader("Does a promised X% window contain the real delivery time X% of the time?")
    unit = alt.Scale(domain=[0.45, 1], zero=False)
    diag = alt.Chart(pd.DataFrame({"x": [0.45, 1.0], "y": [0.45, 1.0]})).mark_line(strokeDash=[4, 4], color=MUTED) \
        .encode(x=alt.X("x:Q", scale=unit), y=alt.Y("y:Q", scale=unit))
    color = alt.Color("method:N", scale=alt.Scale(domain=["CQR (conformal)", "Raw quantile regression"],
                                                  range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None))
    pts = alt.Chart(curve).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2).encode(
        x=alt.X("confidence_level:Q", title="target coverage", scale=alt.Scale(domain=[0.45, 1])),
        y=alt.Y("coverage:Q", title="empirical coverage (test set)", scale=unit),
        color=color,
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("confidence_level:Q", title="target", format=".0%"),
                 alt.Tooltip("coverage:Q", title="actual", format=".1%"),
                 alt.Tooltip("mean_width:Q", title="mean width (min)", format=".1f")])
    w = alt.Chart(curve).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2).encode(
        x=alt.X("confidence_level:Q", title="target coverage", scale=alt.Scale(domain=[0.45, 1])),
        y=alt.Y("mean_width:Q", title="mean interval width (min)", scale=alt.Scale(zero=False)), color=color,
        tooltip=[alt.Tooltip("method:N"), alt.Tooltip("confidence_level:Q", format=".0%"),
                 alt.Tooltip("mean_width:Q", format=".1f")])
    a, b = st.columns(2)
    a.altair_chart((diag + pts).properties(height=360, title="Calibration curve"), width="stretch")
    b.altair_chart(w.properties(height=360, title="Price of coverage"), width="stretch")
    st.markdown("Raw quantile models are **over-confident at every level** (points below the diagonal). "
                "One conformal correction per level, estimated on a held-out calibration week, moves them onto it.")
    st.subheader("Method comparison (test set)")
    show = cmp[["coverage", "mean_width", "interval_score", "miss_above_rate", "worst_segment_coverage",
                "n_segments_under", "mean_z2", "p50_mae", "mae"]].astype(float)
    fmt = {"coverage": "{:.1%}", "mean_width": "{:.2f}", "interval_score": "{:.2f}", "miss_above_rate": "{:.1%}",
           "worst_segment_coverage": "{:.1%}", "n_segments_under": "{:.0f}", "mean_z2": "{:.2f}",
           "p50_mae": "{:.2f}", "mae": "{:.2f}"}
    # format to strings ourselves: st.dataframe ignores Styler.na_rep and would print "None"
    st.dataframe(pd.DataFrame({c: show[c].map(lambda v, f=f: "–" if pd.isna(v) else f.format(v))
                               for c, f in fmt.items()}, index=show.index), width="stretch")
    st.caption("mean_z2 ≈ 1 means per-segment coverage gaps look like pure sampling noise; ≫1 means systematic "
               "mis-calibration across segments. n_segments_under = segments (n≥100) significantly below target (z<−2).")

# --------------------------------------------------------------------------- #
# Segments tab
# --------------------------------------------------------------------------- #
with tab_seg:
    s = R["slices"]
    st.info(R["metrics"]["headline"])
    seg = st.selectbox("Break down by", SLICE_SEGMENTS + ["multiple_deliveries", "vehicle_type"], index=2)
    t = s[s["segment"] == seg].copy()
    t["lo95"] = (t["coverage"] - 1.96 * t["coverage_se"]).clip(0, 1)
    t["hi95"] = (t["coverage"] + 1.96 * t["coverage_se"]).clip(0, 1)
    t["flag"] = np.where((t["gap_z"] < -2) & (t["n"] >= 100), "significantly below target", "within noise / above")
    order = t.sort_values("coverage")["value"].astype(str).tolist()
    yenc = alt.Y("value:N", sort=order, title=None)
    rule = alt.Chart(pd.DataFrame({"x": [conf]})).mark_rule(strokeDash=[4, 4], color="#0b0b0b").encode(x="x:Q")
    ci = alt.Chart(t).mark_rule(color=MUTED).encode(y=yenc, x=alt.X("lo95:Q", scale=alt.Scale(zero=False),
                                                                     title="coverage (95% CI)"), x2="hi95:Q")
    raw = alt.Chart(t).mark_tick(color=MUTED, thickness=2, size=14).encode(y=yenc, x="raw_coverage:Q")
    dots = alt.Chart(t).mark_circle(size=110, opacity=1).encode(
        y=yenc, x="coverage:Q",
        color=alt.Color("flag:N", scale=alt.Scale(domain=["within noise / above", "significantly below target"],
                                                  range=[BLUE, ORANGE]), legend=alt.Legend(orient="top", title=None)),
        tooltip=[alt.Tooltip("value:N", title=seg), alt.Tooltip("n:Q", title="test orders"),
                 alt.Tooltip("coverage:Q", format=".1%"), alt.Tooltip("raw_coverage:Q", title="raw (before conformal)", format=".1%"),
                 alt.Tooltip("mae_p50:Q", title="MAE", format=".2f"), alt.Tooltip("mean_width:Q", title="width", format=".1f"),
                 alt.Tooltip("gap_z:Q", title="z", format=".1f")])
    mae = alt.Chart(t).mark_bar(color=BLUE, cornerRadiusEnd=4, height=14).encode(
        y=alt.Y("value:N", sort=order, title=None, axis=None), x=alt.X("mae_p50:Q", title="MAE of P50 (min)"),
        tooltip=[alt.Tooltip("value:N", title=seg), alt.Tooltip("mae_p50:Q", format=".2f"),
                 alt.Tooltip("mean_width:Q", title="width", format=".1f")])
    h = max(160, 28 * len(t))
    st.altair_chart(alt.hconcat((rule + ci + raw + dots).properties(width=430, height=h, title=f"Coverage by {seg}"),
                                mae.properties(width=260, height=h, title="Error")), width="content")
    st.caption("Dot = conformal coverage, grey tick = raw quantile coverage before conformal, whisker = 95% CI. "
               "With ~42 segments tested, ~1 will fall below z = −2 by chance alone.")
    st.dataframe(t[["value", "n", "coverage", "raw_coverage", "late_rate", "mean_width", "mae_p50", "pinball_mean", "gap_z"]]
                 .sort_values("coverage").style.format({"coverage": "{:.1%}", "raw_coverage": "{:.1%}", "late_rate": "{:.1%}",
                                                        "mean_width": "{:.1f}", "mae_p50": "{:.2f}", "pinball_mean": "{:.3f}",
                                                        "gap_z": "{:.1f}"}), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# About tab
# --------------------------------------------------------------------------- #
with tab_about:
    m = R["metrics"]
    st.markdown(f"""
**Pipeline.** Kaggle food-delivery data → cleaning → features (haversine distance, time-of-day, rush-hour windows,
traffic ordinal, weather, city from rider ID, KMeans geo-cluster) → chronological split
(train {m['split_rows']['train']:,} · val {m['split_rows']['val']:,} · calib {m['split_rows']['calib']:,} · test {m['split_rows']['test']:,} orders)
→ LightGBM P{P_LO}/P50/P{P_HI} → MAPIE CQR on the calibration week → evaluation on the final week.

**Why the calibration set is separate.** Conformal coverage holds only if the residuals used to size the correction
come from data no model was trained or early-stopped on. Train → fit, val → early stopping & tuning, calib → conformal, test → report.

**Conformal correction.** CQR widened each side of the raw interval by **{m['cqr_correction_min']:+.2f} min**; that
small shift is what moved coverage from {raw_row['coverage']:.1%} to {cqr_row['coverage']:.1%}.

**Data caveats (honest).** This Kaggle dataset is semi-synthetic: customer locations are the restaurant plus an equal
lat/lon offset (12 values), weather is uniformly distributed across six classes in every city, prep time is always
5/10/15 min, and the target is clipped to 10–54 min. The *method* transfers to real data; the *numbers* describe this dataset.

**Guarantee scope.** Coverage is guaranteed on average (marginally) under exchangeability. A chronological split only
approximates that: drift (new city, monsoon) would need rolling recalibration or adaptive conformal inference.

MLflow run: `{m['run_id']}`
""")
