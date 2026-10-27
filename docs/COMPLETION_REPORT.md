# Core capability completion and qualification

This tranche delivers the five areas in [the execution plan](COMPLETION_PLAN.md):
repeated group models, broader synthetic qualification, permutation/influence
methods, noise extensions, and a reproducible local workspace checkpoint.
The user selected synthetic data. No real-data, hosted-CI or publication claim is made.

## Delivered behavior

| Area | Public entry points | Scope |
| --- | --- | --- |
| Repeated models | `RepeatedCohort`, `PreparedRepeatedModel` | Identified within-subject SPD sampling covariance; random intercept or correlated intercept/slope; optional extra observation variance; featurewise ML/REML; conditional normal covariance |
| Resampling | `permutation_max_t` | Joint subject sign flips or explicitly blocked Freedman–Lane; complete declared feature family; exhaustive or seeded Monte Carlo; marginal and max-T probabilities |
| Influence | `meta_influence` | Exact FE/DL/PM subject deletion, fixed or refitted heterogeneity; missingness/rank/failure statuses and descriptive effect changes |
| Noise | `select_ar_order_conditional`, `estimate_parcel_ar`, `ArmaPlan`, `estimate_hannan_rissanen`, `InnovationScaled` | Common-sample conditional order scores, per-feature pooled ACVFs with explicit shrinkage, strict ARMA estimation/filtering, known or explicit residual-RMS run scales |
| Delivery | Cargo workspace, CI configuration, source/check receipts | Local consolidation preserves original repositories and concurrent cohort/viewer changes; Linux/macOS/Windows CI configuration awaits hosted execution |

Detailed contracts are in [repeated models](REPEATED_GROUP.md),
[permutations](GROUP_RESAMPLING.md), [influence](GROUP_INFLUENCE.md),
and [noise extensions](NOISE_EXTENSIONS.md). Portable bundle v1 remains the
independent-subject scalar profile; it does not serialize repeated covariance.

## Numerical acceptance

Independent evidence includes dense R Cholesky/QR likelihood and covariance,
an independent covariance optimizer, analytic RI ML/REML variance boundaries,
finite-difference covariance scores, unit/offset/order invariance, and explicit
nonidentifiability/evaluation-limit tests. Balanced repeated layouts also have an
independent eigenspace REML reference: subject mean/slope scores and the remaining
within-subject residual direction separate when the unconstrained covariance is PSD.

Complete enumeration compares Rust and R maximum distributions, including a
non-axis contrast. Fresh metafor checks cover both deletion modes and fixed-tau
mKH. Full-precision R fixtures check every AR criterion score, the HR estimate,
pooled ACVFs and zero-history ARMA recursion. Consulted upstream source hashes
are retained; these new noise fixtures use independent base R, not an unexecuted
claim about compiled fmriAR parity.

Independent review exposed and the final implementation repairs:

- A fixed positive-tau deletion being gated by an ill-conditioned zero-tau fit.
  `MetaDiagnostics.q_fixed` is now optional and absent for that fixed positive-tau fit.
- Contrast normalization overflow changing a permutation null space. Scaled
  design/contrast coordinates now form the nuisance projector directly.
- Overflow admitting negative random covariance. PSD admission uses scaled eigenvalues.
- Repeated likelihood cancellation and design-unit-dependent optimizer stopping.
  Whitened SVD/residual products and dimensionless stopping replace those drafts.

Failed development checks remain in the local evidence. They are not successful
qualification receipts.

## Synthetic calibration and performance

Recorded studies cover estimated sampling variances, high leverage/missingness,
skewed errors, repeated covariance boundaries, same-feature AR estimation, and
joint-feature permutation error rates. Failures remain in the denominators;
interval coverage is explicitly conditional on an estimate being delivered.
Wilson intervals quantify Monte Carlo uncertainty. Numerical agreement does not
imply nominal coverage.

All intervals below are 95% Wilson intervals for the simulation proportion.
The nominal model interval is 95%; coverage uses successful fits as its denominator.
Attempted fits and failures are reported separately. Seeds and inputs are retained.

**AR estimation: 4,000 attempts, 2,998 delivered fits.** Each row has 500 attempts per
noise-estimation strategy. The heavy-censor pattern leaves no lag-2 pairs for the
declared lag-12 estimator, so all 1,000 attempts explicitly fail; no IID fit replaces it.

| Scenario | Delivered per strategy | Same-feature coverage | Held-out-feature coverage |
| --- | --- | --- | --- |
| moderate | 500/500, 500/500 | 95.6% [93.4, 97.1] | 93.8% [91.3, 95.6] |
| persistent | 499/500, 499/500 | 85.6% [82.2, 88.4] | 85.6% [82.2, 88.4] |
| short_runs | 500/500, 500/500 | 90.2% [87.3, 92.5] | 89.6% [86.6, 92.0] |
| heavy_censor | 0/500, 0/500 | unavailable | unavailable |

**Independent-subject models: 36,000/36,000 delivered fits.** The following
Paule–Mandel results illustrate sensitivity to estimated sampling variances and
small-cohort inference. FE/DL/PM, normal/mKH, all six scenarios and both sizes
are retained in [the full summary](../evidence/completion/calibration-summary.json).

| Subjects | Scenario | Normal coverage | Modified Knapp–Hartung coverage |
| --- | --- | --- | --- |
| 8 | known_heterogeneous | 92.0% [89.3, 94.1] | 96.2% [94.1, 97.6] |
| 8 | estimated_df4 | 88.4% [85.3, 90.9] | 92.8% [90.2, 94.8] |
| 8 | high_leverage_missing | 89.8% [86.8, 92.2] | 95.2% [93.0, 96.8] |
| 30 | known_heterogeneous | 94.0% [91.6, 95.8] | 94.8% [92.5, 96.4] |
| 30 | estimated_df4 | 94.0% [91.6, 95.8] | 95.0% [92.7, 96.6] |
| 30 | high_leverage_missing | 94.4% [92.0, 96.1] | 94.8% [92.5, 96.4] |

**Repeated models: 7,986/8,000 delivered fits.** Each case has 1,000 attempts
and three observations per subject. There are 14 explicit `NonConvergence` results,
all in the correlated-slope interior cases; these contain no usable estimates.
Coverage remains conditional on successful delivery.

| Subjects | Scenario | Delivered | Intercept coverage | Slope coverage | Mean fit ms/attempt |
| --- | --- | --- | --- | --- | --- |
| 8 | ri_boundary | 1000/1000 | 94.4% [92.8, 95.7] | — | 0.238 |
| 8 | ri_interior | 1000/1000 | 90.7% [88.7, 92.3] | — | 1.332 |
| 8 | slope_boundary | 1000/1000 | 96.5% [95.2, 97.5] | 96.3% [94.9, 97.3] | 1.439 |
| 8 | slope_interior | 998/1000 | 91.5% [89.6, 93.1] | 92.2% [90.4, 93.7] | 5.356 |
| 30 | ri_boundary | 1000/1000 | 96.0% [94.6, 97.0] | — | 0.632 |
| 30 | ri_interior | 1000/1000 | 93.9% [92.2, 95.2] | — | 1.505 |
| 30 | slope_boundary | 1000/1000 | 95.5% [94.0, 96.6] | 97.0% [95.7, 97.9] | 4.785 |
| 30 | slope_interior | 988/1000 | 92.8% [91.0, 94.3] | 94.2% [92.6, 95.5] | 18.877 |

Fitting times are warm, single-thread, portable-backend measurements on this
shared ARM64 macOS machine. Preparation is measured separately in the summary
(0.018–0.086 ms per attempt); compilation and preparation are excluded from fit
time, while failed fitting attempts remain included. These are bounded synthetic
workloads, not a cross-language speedup or whole-brain throughput claim.

**Permutation calibration: 800 null experiments.** Each has four correlated
Gaussian features and 999 Monte Carlo transformations. Max-T covers that full
four-feature family. These limited cases do not establish universal error control,
especially for nuisance-adjusted Freedman–Lane inference.

| Subjects | Method | Family rejection rate |
| --- | --- | --- |
| 8 | sign flip | 4.0% [2.0, 7.7] |
| 8 | Freedman-Lane | 5.5% [3.1, 9.6] |
| 30 | sign flip | 5.5% [3.1, 9.6] |
| 30 | Freedman-Lane | 3.0% [1.4, 6.4] |

Fresh independent R checks passed: 1,800 metafor comparisons, maximum normalized
error `5.91e-13`; 80 dense repeated-model likelihood/covariance checks, `1.51e-14`;
and 55 analytic eigenspace REML parameter comparisons, `1.01e-8`. These verify
numerical estimates; the coverage shortfalls above remain scientific limitations.

## Executed checks and receipts

- [Default workspace](../evidence/completion/run-20261027T201836Z/receipt.json):
  1,024 passed, 0 failed, 17 ignored. The later unequal-visit witness passed in a
  focused default run and is included in the all-feature count below.
- [All-feature checks](../evidence/completion/run-20261027T203157Z/receipt.json):
  1,054 passed, 0 failed, 21 ignored; 58 example targets built. Targets ran one at
  a time because shared disk is limited. The same run found a formatting-only
  module-import ordering difference, which was corrected.
- [Final checks and studies](../evidence/completion/run-20261027T203534Z/receipt.json):
  workspace formatting, numerical-library Clippy with warnings denied, public
  repeated-model example, four synthetic studies, and both independent R audits
  all exited zero. No source changes occurred during either source-bound final run.
- [Source reconciliation](../evidence/completion/qualification-summary.json) records
  the transition between these receipts: test/fixture additions, runner changes
  and formatting only; no numerical implementation changed between them.
- [Upstream drift audit](../evidence/completion/upstream-drift.log) exited zero for
  the five original local R reference checkouts. Group-model references have
  their separate provenance in the fixture generators and group audit.

The failed eager all-feature build and failed formatting run remain visible.
The [disk-failure recovery note](../evidence/completion/run-20261027T201836Z/recovery.json)
distinguishes its passed default suite from its incomplete all-feature evidence.
The new summary generator is postprocessing only and hashes all four study inputs.

## Reproduction and delivery boundaries

Run `python3 scripts/qualify_completion.py` for default/all-feature workspace
checks, formatting, numerical-library lint, the public repeated example, seeded
studies and independent R audits. R needs `jsonlite` and `metafor`; the Rust
tests themselves consume committed reference fixtures and require no R runtime.
Use `--checks` or `--calibration` to run either half. Run
`python3 scripts/summarize_completion.py` afterward to reproduce all rates and
Wilson intervals. `--feature-set all` selects only the all-feature suite;
`--feature-set post-only` runs only formatting, lint and the public example.

Each run records commands, exit status, runtime, thread settings, source hashes
before/after, and artifact hashes. Source drift must be reconciled before treating
a run as an immutable checkpoint. Existing legacy frontend warnings are not
equivalent to numerical-library Clippy failures.

Repeated fitting holds supplied covariance in memory and bounds numeric fitting
workspace. Its current optimizer is local, with explicit nonconvergence and no
usable output on failure. Normal plug-in intervals omit covariance-estimation
uncertainty. AR plug-in intervals likewise retain material short-run/persistence
limits. Conditional AR order selection and residual RMS run scales do not add a
new design-corrected automatic noise-selection pipeline.

Source archives, numerical payloads and shared build caches remain local where
already designated. No original repository was removed, and nothing was pushed,
published or remotely migrated.
