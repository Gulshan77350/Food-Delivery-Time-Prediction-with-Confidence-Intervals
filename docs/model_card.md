# Model card: ETA interval model (CQR-LightGBM, 80%)

| | |
|---|---|
| **Task** | Predict food-delivery duration (order placed → delivered, minutes) as an 80% prediction interval plus a median |
| **Prediction moment** | When the order is placed (rider assumed assigned). Prep/pickup time is *not* used |
| **Model** | 3 × LightGBM (`objective="quantile"`, α = 0.1 / 0.5 / 0.9), Optuna-tuned on val; conformalized with MAPIE `ConformalizedQuantileRegressor` (prefit, symmetric) |
| **Artifacts** | `models/eta_bundle.joblib` (models + conformal object + feature artifacts), LightGBM text models in `models/run_order/`, MLflow run in `mlflow.db` |
| **Owner / version** | Portfolio project · v0.1 · trained on data up to 2022-03-20, calibrated on 2022-03-26 → 31 |

## Data
Kaggle `gauravmalik26/food-delivery-dataset`, 45,593 orders in 22 Indian cities, Feb 11 – Apr 6 2022. After cleaning: 41,953
(3,640 rows with (0,0) restaurant coordinates dropped). Split chronologically by whole days:

| split | dates | orders | use |
|---|---|---|---|
| train | Feb 11 – Mar 20 | 26,052 | model fitting |
| val | Mar 21 – 25 | 3,991 | early stopping, Optuna, Mondrian segment choice |
| calib | Mar 26 – 31 | 5,945 | conformal scores only |
| test | Apr 1 – 6 | 5,965 | final evaluation |

**Known data issues:** semi-synthetic geometry (customer = restaurant + equal lat/lon offset), uniformly random weather, prep time in
{5, 10, 15}, target clipped to 10–54 min, gaps (Feb 19–27, Mar 22 missing). Details: `notebooks/01_eda.ipynb`.

## Features (22)
distance_km · hour · minute_of_day · day_of_week · is_weekend · is_rush_hour (12–14h, 19–21h) · traffic_level (ordinal) · weather ·
weather_bucket · festival · city (from rider ID) · city_tier · geo_cluster (KMeans k=20, train-fit) · order_type · vehicle_type ·
vehicle_condition · multiple_deliveries · rider_age · rider_rating · order_time_imputed.
Excluded on purpose: prep_min (leaky at order time), bearing and manhattan distance (constant / redundant in this data).

## Performance (test week)
| metric | value |
|---|---|
| coverage of 80% interval | **79.9%** (raw quantiles before conformal: 75.0%) |
| mean / median width | 10.0 / 9.4 min |
| interval (Winkler) score | 12.06 |
| late beyond upper bound | 10.5% |
| P50 MAE / RMSE | 3.11 / 3.93 min (L2 baseline MAE 3.10) |
| calibration at 50/60/70/90/95% | 49.7 / 60.3 / 69.7 / 90.6 / 95.7% |
| segments significantly below target | 3 of 42 (n ≥ 100); mean z² 1.34 |

## Where it is weak
- **Semi-urban orders: 54% coverage (n = 24)**, ~2× longer deliveries, rare in training. Treat intervals there as unreliable.
- **traffic = high: 76.4% (n = 618)**, borderline (z = −2.3, not significant after multiple-testing correction).
- Unknown traffic/weather (missing inputs) give wide intervals (~19 min) but are still covered.

## Intended use / not intended
- ✅ Demonstrating calibrated uncertainty for ETAs; offline evaluation of interval methods.
- ❌ Real customer promises: the data is semi-synthetic and a week old relative to calibration. No live traffic, kitchen load or road distance.

## Assumptions & risks
- **Exchangeability** of calibration and future orders (approximated by calibrating on the most recent week). Requires rolling recalibration
  or adaptive conformal inference under drift, plus a coverage monitor (alert if 7-day coverage leaves 77–83%).
- **rider_rating** may be computed with future information (undocumented in the source); verify point-in-time correctness before production.
- Guarantee is **marginal**; per-segment coverage is monitored, not guaranteed.

## Reproduce
`.\run_all.ps1 -Fresh` (or `./run_all.sh`). All metrics above come from `reports/metrics_order.json`.
