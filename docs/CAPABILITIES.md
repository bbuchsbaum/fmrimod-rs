# Capability boundaries

This is a capability assessment, not a percentage of R exports ported.

| Area | Implementation and evidence | Remaining scope |
| --- | --- | --- |
| Independent-subject group analysis | Identified cohorts; rank-aware OLS; FE/DL/PM meta-regression; normal/mKH policies; requested group-contrast covariance; whole-family BH/BY; exact deletion with fixed/refitted tau; scalar portable disk bundles and explicit shared-weight FE batching | Sampling variances treated as fixed; small-sample plug-in inference has measured calibration limitations; full frame serialization and arbitrary correlated-estimand interchange remain outside scalar bundle v1 |
| Repeated group models | Separate observation/feature/participant-bound SPD sampling blocks; RI or correlated RI+slope; optional extra observation variance; featurewise ML/REML; exact PSD boundaries, covariance-basis/REML identifiability checks, conditional normal covariance; dense R/analytic/gradient witnesses (REPEATED_GROUP.md) | Complete declared ragged layouts; no automatic featurewise omission, Satterthwaite/KR, pooled covariance fitting, or general global-optimizer guarantee; input covariance is resident; numerical fitting workspace budget is explicit |
| Group resampling | Independent-subject joint sign flips and blocked Freedman–Lane residual permutations; exhaustive or seeded sampling; one whole-family max-T distribution (GROUP_RESAMPLING.md) | Caller establishes symmetry/exchangeability; nuisance residual permutation is generally approximate; no arbitrary repeated-measures permutation, cluster/TFCE or spatial resampling |
| HRF/event core | Current-source SPMG1/2/3, raw scale, temporal/dispersion bases, normalization, endpoint-constrained spline/tent, fractional impulses/blocks, multibasis and run boundaries; regenerated R fixtures | Empirical, Fourier/FIR endpoints, derivatives, reconstruction, penalties and supported portable recipes audited in HRF_DESIGN_AUDIT.md; run-aware continuous numeric features are implemented and checked against R (SAMPLED_FEATURES.md); additional family/decorator recipes and higher-level feature workflows remain deferred |
| Experimental design | Existing formula/basis/contrast frontend retained; current-source event and baseline matrix fixtures; explicit condition-major R ↔ basis-major Rust mapping | Core contrast alignment/selection, relative estimability and time-grid composition audited; explicit polynomial/DCT drift, run intercepts, named covariates and spike/row-selection censoring now covered in NUISANCE.md; automatic nuisance cleaning, formula-extension breadth and sampling-type unification remain deferred |
| Regression | Reusable rank-aware SVD preparation, block OLS, rank/df/variance, estimable t/F contrasts, known AR(p) GLS, one-step corrected AR(p) GLS | Known-parameter AR(p) GLS on selected observations now supports explicit preserve-gap/restart covariance (SELECTED_GLS.md). Stationary selected-row AR(p) estimation has finite-lag correction and limited orders 1–4 simulation evidence (SELECTED_ESTIMATION.md); zero-history estimation, design-corrected automatic order selection, robust/WLS and LSS follow later; no reduced-rank/latent/group models in this scope |
| AR noise | Per-run/global estimation, censor-separated lag products, matched X/Y filtering; fixed AR(p), explicit zero-history or stationary startup, uncorrected and prepared design-corrected estimation; orders 1–4 checked against current R | Conditional common-sample AIC/BIC selection, strict Hannan–Rissanen estimation, known zero-history ARMA whitening, per-feature pooled parcel ACVF with explicit shrinkage, and known/plugin run scales are implemented (NOISE_EXTENSIONS.md). Stationary ARMA startup, corrected/censored order selection and sandwich covariance remain outside scope |
| Design-bias correction | Finite-lag maps computed using low-rank factors; current fmriAR dense maps and coefficient fixtures provide independent checks | Long-memory truncation remains an approximation; no automatic lag tuning; ill-conditioned systems error |
| Datasets | Existing f32/f64 in-memory/index/volume frame foundation and bounded readers; f64 OLS/GLS block adapters and exact observation-ID/order binding with feature-ID delivery | Optional NIfTI ingress and feature-major f64 file sources are available; the full R backend ecosystem and acquisition-metadata joins are not implemented |
| Efficiency | No formula/dataframe dependency in regression; one preparation per design; compact fit summaries; reusable corrections; opt-in plotting | 16 earlier R AR-GLS comparisons plus 28 selected-row release workloads and block/worker/process-RSS sweeps; see PERFORMANCE.md and QUALIFICATION.md. Portable bundle warm I/O/shared-weight/RSS measurements are in GROUP_PORTABLE_EXECUTION.md; new model timings and calibration are in COMPLETION_REPORT.md. Real-data acceptance and universal speed/memory guarantees remain unclaimed |

Noise censor masks exclude rows from noise estimation and restart filtering after
excluded rows. They do **not** remove rows from the final regression objective.
Censor-excluded known-parameter fitting is available through ObservationSelection
and SelectedArPlan (SELECTED_GLS.md); stationary selected-row estimation is
separately qualified in SELECTED_ESTIMATION.md;
this release does not silently reinterpret the mask. The established AR estimators use one innovation
scale estimated in the transformed fit. `InnovationScaled` separately supplies
explicit run-specific scales; their plug-in estimation is not a new uncertainty correction.
AR parameter uncertainty is not included in plug-in t/F inference.

Numerical rank is relative to the unscaled design's largest singular value and
an explicit tolerance. Scale predictors sensibly. F contrasts reject dependent
rows. Zero residual variance makes contrast inference undefined and errors.
The dataset working-buffer budget excludes resident source/model storage,
allocator overhead, numerical-backend packing/custom-transform scratch, and values retained by user sinks. It is not a total-process RSS bound.

Fixed-order AR estimation errors on unsupported lag pairs, nonpositive variance
or ill-conditioned correction systems. It never silently reduces order or
substitutes IID. The older full-grid estimators expose ACVF shrinkage and stability-adjustment
diagnostics. Selected-row estimators use strict positive-definiteness,
conditioning and stationarity checks and reject invalid estimates without repair. Stationary startup uses unit-innovation stationary
covariance, including segments shorter than the AR order. Zero-history startup
matches the current R conditional filter. Orders beyond four are supported by
the general implementation but are not covered by the current R fixture suite.


See [HRF/design audit](HRF_DESIGN_AUDIT.md) for the source-by-capability matrix,
ScalaFIM design references, deliberate compatibility changes and exact limits.

See [workspace qualification](QUALIFICATION.md) for current integrated checks,
selected-row timings, adverse statistical scenarios and the complete walkthrough.
