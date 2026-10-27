# Selected-observation AR(1) estimation

`fmri_reg::selected_estimation::estimate_selected_ar1` prepares **one-step stationary
AR(1) feasible GLS** after observation exclusion. It selects the original design
and representative noise data using the same `ObservationSelection`, fits OLS on
those retained rows, estimates corrected residual lag moments, and returns a
`GlsModel<SelectedArPlan>` plus diagnostics. Subsequent fits accept selected rows.

```sh
cargo run -p fmri-reg --features dataset --example selected_ar1
```

The example estimates a common coefficient from 24 synthetic feature series,
then uses the original-ID-checked lazy assay view and two-worker block fitting.
The generating coefficient is 0.4; the recorded example estimates 0.41225. That
single realization demonstrates operation, not general estimator accuracy.

## Scientific contract

The selected design determines the OLS projector M=I−QQ'. After residualization,
C subtracts each original run's retained-sample mean (not a mean for every gap
segment). R=CM is therefore the residual operator actually used by the moments.
For each covariance group and lag k, S_k describes selected observations separated
by k **original scan intervals**, in the same run and, under Restart, the same
contiguous segment. The moment correction is

```text
A[h,k] = mean over observed lag-h pairs of (R S_k R')[a,b]
E[raw moments] = A * covariance_lags
```

`GapPolicy::Preserve` includes endpoint pairs across missing interior scans when
their original lag matches. `Restart` excludes cross-gap pairs and covariance.
The same policy determines the resulting stationary whitening plan. Neither
policy treats the compressed sequence as equally spaced observations.

`Pooling::Global` solves one common stationary autocovariance vector across runs,
using pair-weighted moments pooled across runs. This assumes a common ACVF,
including its scale, rather than averaging independently estimated coefficients.
A completely excluded run is allowed because the coefficient is shared.
`Pooling::Run` solves one covariance vector per original run in a **coupled**
system: shared regressors can mix residual covariance between runs. Every run
must have at least three retained observations. Empty runs are not assigned a
made-up coefficient.

Moment contributions are averaged over representative features without
standardizing each feature's variance. Larger-variance features contribute more
to the pooled covariance. Choose that representative data deliberately. The
fit retains the existing common innovation-scale model; it does not implement
runwise heteroscedastic GLS even when coefficients are estimated per run.

The correction is exact only within the finite covariance-lag model. AR(1) has
an infinite tail, so truncation is an explicit approximation. `max_lag` is required
and never silently capped. Every requested lag/group needs observed pairs; the
number of moment parameters must be below the retained row count. Preparation
also requires at least three residual degrees of freedom and a correction
condition number at most 1e6. Unsupported geometry or an ill-conditioned solve
is an error. A positive variance and a stationary gamma[1]/gamma[0] are required:
there is no coefficient clipping, order reduction or IID fallback.

Startup is stationary in this API. Zero-history covariance is nonstationary and
needs a different estimation derivation; its availability for supplied-coefficient
whitening does not qualify it for this estimator. The fixed stationary AR(p) extension below now reuses this correction. Order
selection, iterative GLS and zero-history estimation remain outside scope.

## Implementation and diagnostics

The production correction uses R=I−UV' with run-indicator and design factors.
It applies sparse original-lag operators and forms only small correction matrices;
it does not materialize a retained-time-by-time projector or covariance. The
estimator is reusable across residual feature blocks with the same design and
selection. Memory includes retained-row low-rank factors, lag-pair indices, and
a G*(L+1) square correction matrix, where G is one for Global or the original
run count for Run. Preparation cost grows with the lag budget and number of runs.
No new performance benchmark is claimed.

`SelectedAr1Diagnostics` exposes per-original-run coefficients, corrected ACVFs,
lag-pair counts, the chosen lag budget, correction condition number and design
rank. ACVFs are in units of residuals divided by `residual_scale`, a common maximum
absolute residual used for stable accumulation. Multiplying by residual_scale²
recovers physical units when representable; coefficient ratios are unchanged.
Global diagnostics contain one ACVF/pair-count vector; Run diagnostics contain
one per run. No diagnostic implies uncertainty in the coefficient estimates.

For callers that already have retained OLS residuals,
`fmri_noise::selected_estimation::SelectedAr1Estimator` exposes preparation,
`bias_matrix()` and `fit()`. Residual dimensions, finiteness, nonzero variance and
orthogonality to the prepared design are checked. Orthogonality cannot establish
provenance; callers must supply the matching residualization and observation IDs.
The convenience regression function constructs those residuals itself.

## Verification and limits

The derivation follows the operator/moment principle documented in local
fmriAR `R/acvf_bias.R`, but changes the projection geometry to the retained design
and supports a coupled runwise system. It is not a claim of equality to fmriAR's
full-grid censor-mask workflow. A standalone R generator constructs dense C, M,
S_k and the correction matrix independently, then solves dense GLS with the
estimated covariance. Four fixtures cover both pooling and gap policies, shared
regressors, unequal run lengths and 64 response features. Tests compare correction
matrices (2e-11 relative-plus-absolute), ACVFs/coefficients (2e-10), and GLS
coefficients, variance and t statistics (2e-9). The same fitted plans are checked
through ID-bound streaming with tail blocks. An adversarial smooth residual has
corrected coefficient 1.000805 and must error rather than clip.

The deterministic simulation test uses 160 replicates per scenario, 120+100
original scans, patterned exclusion, lag budget 10, 16 representative noise
features, and one independent held-out inference target. Global true coefficients
are 0.4; runwise coefficients are 0.45 and −0.25.

| Policy / pooling | Mean coefficient bias | Mean whitened residual lag-1 correlation | Nominal 95% interval coverage |
| --- | --- | --- | --- |
| Preserve / Global | −0.00072 | −0.01021 | 96.25% |
| Restart / Global | −0.00282 | −0.01602 | 92.50% |
| Preserve / Run | −0.00265, −0.00182 | −0.00983 | 96.25% |
| Restart / Run | −0.00372, −0.00075 | −0.01062 | 96.25% |

There were no estimator failures in these 640 replicates. These finite Monte Carlo
results support only the stated moderate-correlation cases. They do not establish
calibration for near-unit-root noise, a small representative feature set, a target
used to estimate its own AR coefficient, long-memory truncation, or real datasets.
GLS t/F inference remains conditional plug-in inference: estimated-coefficient
uncertainty is not included. Seeds, logs, source hashes and check results are in
`evidence/selected-estimation/verification.json`.

## Fixed stationary AR(p) extension

`SelectedArOptions` and `estimate_selected_ar` extend the same workflow to an
explicit positive order p. Require `order <= max_lag`. Preparation corrects the
entire requested lag budget before using lags 0..p in the Yule-Walker solve;
missing lag support never silently lowers p. The existing AR(1) API is preserved.
For reusable residual estimation, construct `SelectedArEstimator::new(correction,
order)` from a `SelectedAr1Estimator` correction preparation. Despite that legacy
name, its moment correction is independent of the AR order.

The normalized Toeplitz covariance must be positive definite and have condition
number at most 1e6. Cholesky solves the Yule-Walker equations. Innovation variance
must be positive and the resulting AR polynomial stationary. Failures return
errors; no shrinkage, clipping, coefficient repair, or IID fallback occurs.
Diagnostics include coefficient vectors per original run, the Yule-Walker
condition number, innovation variance divided by lag-zero variance, and the
minimum `1 - abs(reflection coefficient)` across the stationarity recursion.
These diagnostics supplement the correction condition, pair counts, rank, and
scaled ACVFs already described above. Stability margin is not a confidence bound.

```sh
cargo run -p fmri-reg --features dataset --example selected_arp
```

The example generates independent run histories for a moderate AR(2), estimates
its coefficients, and fits ID-bound feature blocks with two workers. It uses a
256-step burn-in for synthetic run initialization. No fitting or whitening kernel
optimization is introduced by this extension.

Qualification covers orders 2–4, each with Preserve/Restart and Global/Run pooling:
12 independent R fixtures use dense residual projectors, direct Yule-Walker solves,
`stats::ARMAacf` covariance, and dense GLS. Checks cover coefficients, corrected
moments, solve diagnostics, GLS variance and t statistics, and streamed fits with
tail blocks. Additional tests exercise AR(1) compatibility, zero/excessive order,
unsupported lag geometry, singular/ill-conditioned/indefinite Yule-Walker systems,
and nonpositive innovation variance. The earlier AR(1) adverse residual test
continues to check rejection of nonstationary estimates.

There are 1,200 deterministic simulation replicates (100 per combination), with
120+100 original scans, patterned exclusions, correction lag 10, 16 noise features,
and an independent held-out inference target. Generating vectors are
(0.45, −0.18), (0.4, −0.16, 0.08), and (0.4, −0.15, 0.08, −0.04); Run scenarios use
−0.7 times that vector in run two. Independent segments use 256 burn-in steps.
Across the stated scenarios there were no estimator failures; maximum absolute
mean coefficient bias was 0.00613, maximum absolute mean whitened-residual lag
correlation (lags 1..p) was 0.03215, and nominal 95% interval coverage ranged from
92% to 98%. These are finite Monte Carlo results for moderate stationary cases,
not broad calibration guarantees. The exact scenario metrics, seeds, Wilson
coverage intervals, logs, and source hashes are in
`evidence/selected-arp-estimation/verification.json`.

The finite-lag approximation, shared innovation-scale fit, feature-variance
weighting, and conditional plug-in uncertainty limitations remain unchanged.
Orders above four are accepted by the general algorithm but not qualified by
these fixtures/simulations. Zero-history estimation, automatic order selection,
near-unit-root calibration, and AR-parameter uncertainty remain unqualified.
