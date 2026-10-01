# Interview prep: questions this project invites (and honest answers)

Numbers below refer to the main run (`reports/metrics_order.json`, test week Apr 1–6 2022, 5,965 orders).

---

## 1. The 60-second pitch

> "Delivery apps show one ETA number, but delivery time is uncertain, and a late order costs more than an early
> one. I built a model that outputs a **calibrated window**, e.g. 'arrives in 15–25 min', where the window is
> guaranteed to contain the true time about 80% of the time.
> Three LightGBM quantile models give P10/P50/P90; on their own they were over-confident (75% coverage instead of 80%).
> I added **conformalized quantile regression** with MAPIE on a separate calibration week, which brought coverage to
> 80% on the unseen final week. Then I checked whether it holds *within* segments, not just on average: a
> constant-width conformal baseline also hits 80% overall but fails 8 of 42 segments (rush-hour jams down to 73%), while CQR
> fails 3, roughly what chance alone produces. Everything is tracked in MLflow and demoed in Streamlit."

---

## 2. Conformal prediction

**Q: What guarantee does conformal prediction give, exactly?**
If calibration and test points are *exchangeable*, then P(y ∈ interval) ≥ 1 − α, for any model and any
distribution. It's **marginal** (averaged over all orders) and **finite-sample** (no asymptotics). It is *not*
conditional coverage, and it doesn't hold for every order or every segment.

**Q: Walk me through CQR.**
Fit q̂_lo, q̂_hi on train. On a held-out calibration set compute E_i = max(q̂_lo(x_i) − y_i, y_i − q̂_hi(x_i)).
Take Q = the ⌈(n+1)(1−α)⌉-th smallest E. Predict [q̂_lo(x) − Q, q̂_hi(x) + Q]. If the raw quantiles are over-confident, Q > 0
and the interval widens; if they're too wide, Q < 0 and it shrinks. It keeps the *adaptive* width of quantile
regression and adds the *coverage guarantee* of conformal. Our Q was ≈ +0.3 min per side, small but enough to move 75% → 80%.

**Q: Why the (n+1) and the ceiling?**
Finite-sample correction: the test point is the (n+1)-th exchangeable point; its rank among the n+1 scores is uniform.
I implemented it by hand in `conformal_quantile` and unit-tested it against MAPIE. That test also exposed that
`np.quantile(..., method="higher")` at level k/n is *not* exactly the k-th order statistic (n=10, k=9 returns the 10th),
which is slightly conservative. I pinned the exact definition.

**Q: How did you keep the calibration set clean?** *(the most likely probe)*
Four chronological, day-aligned splits: train (fit) → val (early stopping, Optuna, choosing the Mondrian segment) →
calib (conformal scores only) → test (reported once). The calibration rows are never used by `fit`, early stopping, or tuning.
If they were, residuals would be optimistic, Q too small, and coverage would silently drop below target.
The KMeans geo-clusters and category vocabularies are also fit on train only.

**Q: Your split is chronological. Doesn't that break exchangeability?**
Yes, strictly. Exchangeability is an approximation for time-ordered data. I mitigate by calibrating on the week
*immediately before* test. The calibration curve on test lands on the diagonal (0.50 → 0.497, 0.90 → 0.906), so drift between
those weeks was small. In production I'd recalibrate on a rolling window or use **Adaptive Conformal Inference**
(Gibbs & Candès 2021), which updates α online from observed misses and keeps long-run coverage even under drift.

**Q: Symmetric vs asymmetric CQR?**
Symmetric uses one Q for both sides (original paper). Asymmetric calibrates each side at α/2, which is useful when
lateness and earliness errors differ. Asymmetric gave a lower late-rate (9.6% vs 10.5%) at +0.05 min width. For a
customer-facing ETA where lateness hurts more, that trade is arguably worth it.

**Q: What is Mondrian conformal, and did it help?**
Calibrate separately per group → coverage guarantee per group. I chose the group column on **val** (most significantly
under-covered group), which picked `city`. The weakest *test* segment was `traffic=high`, so I also ran Mondrian-by-traffic
as an explicitly labelled post-hoc check. It did **not** fix it (76.4% → 75.1%), because traffic=high was well covered in the
calibration week. Combined with mean z² ≈ 1.3, the honest conclusion is that the remaining gap is mostly sampling noise, and
global CQR is already about as conditionally calibrated as this data allows.

---

## 3. Quantile regression

**Q: Why pinball loss?** It's the proper scoring rule whose minimiser is the τ-quantile: it penalises under-prediction by τ and
over-prediction by 1−τ. LightGBM `objective="quantile", alpha=τ` optimises it directly.

**Q: Why were the raw quantile models under-covering?** Gradient boosting with early stopping regularises toward the
conditional median and shrinks tails; quantile estimates at 0.1/0.9 from finite trees are biased inward. That's
typical, and it's exactly the failure conformal fixes.

**Q: Quantile crossing?** Independent models can give P10 > P50. I apply monotone rearrangement (sort per row), which can only
reduce pinball loss, and log the crossing rate (~0.2%).

**Q: Why not a single model with a distribution (NGBoost, a Gaussian NLL head)?** Parametric heads assume a shape (often
symmetric), while delivery times are right-skewed. Quantile regression is distribution-free, and conformal gives the guarantee regardless.

---

## 4. Evaluation

**Q: How do you compare two interval methods?** Only at equal coverage, compare width. A wide interval always covers.
I also report the **interval (Winkler) score** = width + (2/α)·miss distance, one proper score that balances both,
and **conditional coverage** across 42 segments (worst segment, # significantly below target, mean z²).

**Q: The point-estimate baseline has the same MAE as your P50. So what did you gain?**
Point accuracy is unchanged (3.10 vs 3.11 min MAE). The gain is a decision-ready *window* with a verified guarantee.
That supports SLA promises ("90% of orders inside the promise"), dynamic buffers in jams, and rider batching decisions,
none of which a point estimate can support.

**Q: Isn't the constant-width baseline narrower (9.8 vs 10.0 min)?** Yes, marginally, and I say so. But it achieves 80% by
over-covering easy orders and under-covering hard ones: 8 of 42 segments significantly below target, worst 72.9%, mean z² 8.7.
CQR spends its width where the uncertainty is: 3/42 segments, mean z² 1.3. On real data with stronger heteroscedasticity,
the width advantage of CQR would also show up.

**Q: So what IS the weakest segment?** Not the one my rule flags. **Semi-urban orders: 54% coverage, but only 24 test orders**
(13 inside), below my n ≥ 100 flagging threshold. They average ~50 min vs 26 overall, and there were only 84 in training, so the
quantile models never learned that tail and one global correction can't fix it. That's the classic conformal failure mode: marginal
coverage hides rare, hard subgroups. Fixes: Mondrian calibration once enough data exists, a conservative fallback interval for rare
segments, or reweighted/group-conditional conformal. I'd rather volunteer this than have an interviewer find it.

**Q: Multiple-testing on 42 segments?** With 42 tests at z < −2 (~2.3% one-sided), about 1 false flag is expected by chance. The
Bonferroni threshold for 42 tests at 5% is |z| ≈ 3.0, and no segment exceeds it. That's why I call `traffic=high` (z = −2.3) a
"watch item", not a proven failure.

---

## 5. Data & features

**Q: Tell me about the data quality.** I audited it in `notebooks/01_eda.ipynb` and it is semi-synthetic:
- every customer point = restaurant + an equal lat/lon offset (12 values), so bearing is constant and there are only 388 restaurants
- weather is ~16.5% per class in *every* city (sandstorms in Kochi)
- prep time ∈ {5, 10, 15}, uncorrelated with the target
- target clipped to 10–54 min; ETA is a step function of distance
I say this up front: the *method* is the deliverable; the numbers describe this dataset.

**Q: Haversine vs GeoPandas?** For point-to-point distance haversine is more accurate (<0.5% spherical error) than projecting
to one pan-India CRS (measured −1.7% scale distortion) and ~55× faster (2 ms vs 110 ms for 42k rows, measured). GeoPandas earns its place for polygons (zones, geofences),
spatial joins and buffers. Neither is road distance; production needs a routing engine (OSRM).

**Q: Any leakage risks?**
1. **Prep time** (pickup − order) isn't known at order time, so it's excluded from the at-order model. A comparison run with it
   shows no gain anyway.
2. **Rider rating** is the top feature (22% of gain). If ratings are aggregated *after* deliveries, including this one, that's leakage;
   the dataset doesn't document it. In production I'd use the rating as of order time from a feature store.
3. **Random splits** would leak shared days, weather and traffic events into test. Hence the day-aligned chronological split.
4. KMeans and category vocabularies are fit on train only, and `featurize()` is shared by training and serving
   (unit test guards train/serve skew).

**Q: Why is rush hour defined as 12–14h and 19–21h?** That's a domain prior for Indian food delivery. In this data the evening is busy
from 17h, so the model also gets `hour`/`minute_of_day` and can learn the actual shape.

---

## 6. Engineering

- **MLflow**: every run logs params, data hash, split dates, 100+ metrics, figures, slice tables, LightGBM model files and the bundle;
  each Optuna trial is a nested run.
- **Tests**: CQR maths vs MAPIE, coverage guarantee on synthetic exchangeable data, adaptivity, Mondrian per-group coverage,
  small-group fallback, metric identities, cleaning edge cases, split chronology, train/serve parity, Streamlit smoke tests.
- **Reproducibility**: `run_all.ps1` / `run_all.sh` rebuild everything from the Kaggle download; CI runs the tests.

## 7. "What would you do next / in production?"

1. Real-time features: live traffic speed, restaurant queue length / kitchen load, rider location and current batch.
2. Road distance and ETA from a routing engine instead of haversine.
3. Online recalibration (rolling window or ACI), with a coverage monitor that alerts when the 7-day coverage leaves [77%, 83%].
4. Asymmetric loss aligned with business cost: promise the P85 upper bound, show P50 as "likely".
5. Mondrian/conditional conformal per city once each city has enough recent data; conformal for *re-estimates* after pickup.
6. Model the target as a sum of stages (prep, wait, travel) with stage-level uncertainty.
