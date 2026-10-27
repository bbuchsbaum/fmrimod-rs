# fmrimod-rs: consolidation and scientific delivery

Current continuation: [core completion plan](COMPLETION_PLAN.md) and
[qualification report](COMPLETION_REPORT.md). Earlier checkpoints below retain
their historical scope; the capability matrix records current boundaries.

## Outcome and scope

One standalone Cargo workspace with efficient, idiomatic Rust capabilities for
HRFs, experimental designs, bounded dataset access, and core mass-univariate
estimation. R source behavior is a numerical reference, not a mandate to reproduce
S3 dispatch, formula metaprogramming, or data-frame allocation in numerical loops.

User scope correction: regression prioritizes fast core estimation. Reduced-rank
regression, latent models, meta-analysis, spatial FDR, plotting/reporting,
distributed job orchestration, BIDS workflow wrappers and optimization dashboards
are deferred. Preserve existing plotting/formula adapters, but do not make new
regression kernels depend on them. LSS, robust estimation, sophisticated covariance
estimators and ARMA are follow-on capabilities, not substitutes for a qualified
OLS/AR-GLS path. No unsupported option may silently fall back to IID.

The initial exclusions above applied to first-level regression. The subsequent
fmrigds audit and user authorization add independent-subject group inference in
a separate `fmri-group` crate, without expanding the first-level solver's role.
Its first tranche is implemented; see GROUP_ANALYSIS.md and GROUP_BENCHMARK.md.

## Audit findings (2026-10-26)

* There was no fmrireg-rs; the user confirmed regression is a fresh implementation.
* HRF and design contain extensive uncommitted September work. Dataset has no Git
  history. Import preserves the working sources, not just their older HEADs.
* `evidence/source-manifest.json` records SHA-256 per imported file, original Git
  HEAD/status and current R source fingerprints. Local history bundles preserve
  prior commits; originals remain intact. Agent state and build products are not
  imported. `splines` is an existing local numerical dependency, copied with its
  provenance to avoid a nonportable sibling path.
* September HRF/design contract tests are valuable but stale: R fmrihrf changed
  SPMG A1 from .0833 to 1/120, made the undershoot coefficient exact, changed
  SPMG3 from a second temporal derivative to a dispersion derivative, and now uses
  B-spline HRF basis functions that vanish at both endpoints. Design gained
  estimability checks, naming/contrast fixes and corrected multi-basis ordering.
  Historical percentage-complete claims are not evidence of current capability.
* Dataset is a typed in-memory/index/volume foundation with bounded readers. It
  does not implement the entire current R dataset backend ecosystem.
* Current fmriAR explicitly distinguishes run centering, censor-separated lag
  segments, pooled autocovariance, design-corrected residual estimation, AR order
  selection, ARMA, and parcel pooling. Matching only an AR(1) recurrence is not
  full fmriAR equivalence.
* Installed R packages differ from checkout versions (notably fmriAR). Generate
  fixtures from isolated builds of the recorded sources, not ambient libraries.

## Initial architecture decisions (historical)

Package names and layout now follow [REPOSITORY.md](REPOSITORY.md); the original
choices below preserve the consolidation record.

Keep crates `fmrihrf`, `fmridesign`, `fmridataset`; add `fmri-noise`, `fmrireg`
and a small `fmrimod` facade. Keep scientific kernels independent of file formats
and dataframe/formula frontends. Consolidation is local and non-destructive;
remote creation, migration of consumers and retirement of originals are separate.
Retain existing license declarations, including GPL-3.0-or-later on dataset;
do not relicense the workspace under the HRF/design licenses.

Use time-by-feature borrowed dense views at compute boundaries. Fit a design once
and reuse its factorization across voxel blocks. Validate dimensions, finiteness,
rank, row/run identity and residual degrees of freedom before compute. Noise plans
are reusable, validated objects. Apply the same transform to X and Y; restart at
run/segment boundaries. Known noise parameters and estimated noise are distinct
APIs. Censoring for noise estimation and exclusion from the regression objective
are distinct operations. Return rank/df/variance diagnostics explicitly.

Prefer stable QR/SVD-based solves to inversion of X'X. Contrasts require an
estimability check. Fit outputs should allow dropping residuals/fitted data when
only coefficients, residual sums of squares and contrast summaries are needed.
Parallelize over independent feature blocks with explicit thread settings;
avoid oversubscribing nested numerical kernels. Optimize only after equivalent
outputs pass. Existing multiple sampling-frame types need explicit, tested
conversion before any forced public-API unification.

## Ordered execution and acceptance

| Stage | Deliverable | Acceptance evidence |
| --- | --- | --- |
| 1 | Standalone workspace and provenance | Cargo metadata paths all internal; original imported files still match initial hashes; existing crate tests run; preserve failing baselines |
| 2 | Current HRF/design contracts | Fresh R source builds; SPMG and endpoint spline fixtures; fractional timing/multibasis/run isolation; naming and estimability cases; no blind historical column permutation |
| 3 | Core regression | Reusable multi-response OLS factorization, known-noise GLS, t/F contrasts, rank and df diagnostics; independent R lm/explicit covariance solutions; singular, ill-conditioned and invalid inputs |
| 4 | AR noise | Explicit run-aware AR filtering and boundary policy, stable estimation, reusable plans; current fmriAR references and independent covariance whitening checks; run/censor isolation, zero noise, order limits and rank correction |
| 5 | Dataset-to-fit pipeline | Borrowed/bounded blocks, temporal and feature alignment, end-to-end HRF→design→fit→contrast; dense vs streamed equivalence; budget/reader failure checks |
| 6 | Efficiency and qualification | Release workloads for 300/1200 timepoints, 8/32 regressors, 1k/50k features; throughput, peak memory, block-size and thread scaling; equivalent R baseline; minimal-feature and full-feature builds |

Stage 4 first qualifies fixed AR(1) with stationary first-row scaling and global
or per-run parameters, then fixed AR(p) estimation and current fmriAR residual-bias
correction. Unknown-noise GLS must remain explicitly unqualified until estimation
is independently checked. Automatic order selection, parcel shrinkage and ARMA
follow only after that core. This order avoids presenting a large option surface
whose scientific behavior has not been established.

Each stage has a Mote issue with evidence and an honest disposition. Passing an
old fixture is baseline compatibility; passing a regenerated fixture establishes
only its recorded cases. No universal parity percentage or unmeasured speed claim.

## Checks and risks

Run `cargo test --workspace` for the default suite and focused package tests while
iterating. Run `cargo test --workspace --all-features` before claiming full-feature
qualification. Record actual exit status and unexecuted/ignored tests. Use
`cargo clippy -p <changed-crate> --lib -- -D warnings` for new kernels; existing
legacy warnings are separately recorded. R fixture generation is repeatable via
scripts, with source revision and hashes in fixtures/evidence.

Tolerances are based on scale and conditioning: analytic tiny cases near 1e-12,
R numerical references with absolute plus relative bounds; do not loosen a
tolerance merely to accommodate a changed coefficient or mislabeled column.
Performance measurements are not correctness evidence. Budget peaks include
source blocks, transformed workspaces and outputs, not just final coefficient size.

Major risks: stale successful fixtures, loss of dirty source work, incompatible
licenses/consumer paths during publication, historical R-vs-Rust conventions,
rank-deficient contrasts, invalid whitening at run/censor boundaries, and large
mandatory dataframe dependencies. Local changes are reversible; external
publication and retirement of source repositories require explicit authorization.


## Execution checkpoint

The initial core implementation is in place: source-preserving workspace,
current SPMG/spline contracts, OLS and known/estimated AR1 GLS, reusable t/F
contrasts, a bounded OLS reader adapter and a facade example. Design-corrected
AR1 uses an algebraic low-rank expansion independently checked against fmriAR's
dense bias matrices. The main Mote issue remains open for the broader capability
programme. See CAPABILITIES.md for explicit remaining gaps and evidence/
verification.json for actual completed checks, rather than treating this plan
as proof that every stage is complete.


### Fixed AR(p) and aligned GLS checkpoint (2026-10-26)

Delivered known and estimated fixed AR(p), explicit zero-history/stationary
startup, reuse of finite-lag design correction, and coefficient repair diagnostics.
Current-source R cases cover orders 1–4, run/global pooling and censor gaps.
Independent dense covariance cases cover stationary startup, short segments,
near-boundary coefficients and full GLS estimates. The correction system now uses
pivoted LU after conditioning checks: the prior SVD inverse produced a detectable
coefficient discrepancy in the expanded fixture. No tolerances were relaxed.

The reader adapter now supports GLS and exact declared observation-ID/order
binding, with feature IDs delivered to result sinks. It checks identity, not the
accuracy of acquisition metadata. Working-buffer estimates include the additional
GLS transform; backend scratch and total RSS remain outside that estimate.
Evidence for this slice is in evidence/arp-verification.json and its named logs.
The earlier full-workspace receipt remains a historical baseline, not a rerun
of every imported crate after these edits.

The AR-GLS performance tranche is now qualified on the stated synthetic workloads;
see docs/PERFORMANCE.md and docs/GLS_RESULTS.md. All 16 R comparisons and eight
block/worker comparisons passed. A contiguous residual-variance traversal was
measured and validated; fixed-block RSS remained nearly flat from 50k to 200k
features. No universal speed or RSS guarantee is implied.

### HRF/design audit and repair checkpoint

The requested core audit and repair tranche is complete and locally verified; see
HRF_DESIGN_AUDIT.md for the evidence-backed matrix and ScalaFIM review. Changes
correct empirical/Fourier/FIR semantics, structural penalty selection, derivative
dispatch and batched evaluation, portable HRF recipes, multibasis contrast
alignment, exact legacy selection, relative estimability, and time-grid composition.
A new end-to-end R covariance oracle covers event construction through streamed
AR(2) GLS inference. Final whole-workspace results are recorded separately in
`evidence/hrf-design-verification.json`, not inferred from earlier passing suites.

Remaining programme scope includes additional portable/decorator recipes and
formula breadth, followed by separately
scoped automatic AR-order selection, ARMA and parcel pooling when needed. No
remote migration, retirement of originals or downstream-consumer acceptance is
claimed by these local qualification stages.


### Continuous sampled features and named designs checkpoint

Delivered uniform/irregular zero-order-hold feature bins, multiple columns and
runs, masked run-local centering/sample-SD scaling and reusable prepared features.
The fmridesign adapter adds checked feature/basis labels and acquisition times,
with direct composition and exact named contrast examples. Sixteen current-source
R cases and independent mass, run-isolation and naming/time checks qualify this
scope. See SAMPLED_FEATURES.md and the two evidence/sampled-feature* receipts.
AR/OLS optimization remains separately owned; this slice changes neither kernel.

### Nuisance and explicit censoring checkpoint

Added a numeric run-aware nuisance API: polynomial/DCT drift with explicit
intercepts, validated named block-diagonal covariates, and separate spike-column
and original-row-selection censor operations. R matrices and lm.fit references,
DCT invariants and whole-run rank-loss cases qualify the documented scope.
See NUISANCE.md and evidence/nuisance/verification.json. Automatic nuisance
cleaning and AR-aware row-exclusion integration remain separate work; no AR/OLS
kernel changes or general correlated-error censor equivalence are claimed.

### Selected-observation AR-GLS checkpoint

Added known-parameter AR(p) innovations whitening on retained original scans,
with explicit preserve-gap/restart covariance and stationary/zero-history startup.
Lazy assay views validate original IDs before selecting and reuse existing GLS
block execution. Twenty-four independent R dense-covariance cases qualify
whitening, coefficients, covariance, t/F inference and dense/streamed equivalence.
See SELECTED_GLS.md and evidence/selected-gls/verification.json. Censored AR
estimation and correction remain unqualified; no implicit estimator or IID
fallback is introduced. Optimized fit and contiguous-whitening kernels remain
unchanged apart from exposing internal coefficient/covariance validation helpers.

### Selected-observation stationary AR(1) estimation checkpoint

Added explicit finite-lag correction on the retained OLS design, with global
common-ACVF or coupled runwise estimation and matching Preserve/Restart policy.
One-step feasible GLS exposes diagnostics and reuses ID-bound streaming. Dense
R operator references and 640 moderate-correlation simulations pass the stated
checks; observed nominal 95% coverage ranges from 92.5% to 96.25%, with independent
inference targets. See SELECTED_ESTIMATION.md and its evidence receipt. This is
conditional plug-in inference, not general calibration. Censored AR(p),
zero-history estimation and parameter-uncertainty inference remain unqualified.

### Selected-observation fixed AR(p) estimation checkpoint

Extended retained-design correction to stationary fixed AR(p) using a strict
conditioned positive-definite Yule-Walker solve. Diagnostics expose stability
margins, innovation fractions and solve conditioning. Orders 2–4 pass 12 dense
R reference cases and 1,200 moderate-correlation simulations across both gap and
pooling policies; the existing AR(1) API is retained. Nominal 95% coverage spans
92–98% for independent held-out targets. Streaming and a runnable AR(2) example
are covered. See SELECTED_ESTIMATION.md and evidence/selected-arp-estimation.
Zero-history estimation, automatic order selection and parameter uncertainty
remain unqualified; no coefficient repair or order reduction is introduced.

### Integrated qualification checkpoint

The workspace consolidation is now checked as a complete local system, with
default/all-feature target receipts, source-drift audit, selected-row preparation
and block/worker/RSS workloads, and retained statistical stress outcomes. A new
complete_first_level facade example combines events, sampled features, nuisance
design, observation exclusion, estimated AR(2) and ID-bound streamed contrasts.
See QUALIFICATION.md and evidence/consolidation/verification.json for actual
results and limits; statistical acceptance is not inferred from build success.

### Independent-subject group checkpoint (2026-10-27)

Implemented the five-deliverable first tranche from GROUP_ANALYSIS_AUDIT.md:
typed estimate/cohort identity and t-contrast bridge; bounded OLS with explicit
missingness and rank-aware named contrasts; scalar common-effect and DL
meta-analysis without hidden variance floors; typed block sinks and global BH/BY;
independent reference fixtures, end-to-end example and equivalent-output benchmarks.
The facade exports `fmrimod::group`. First-level output and AR optimization changes
from the concurrent workstream were preserved. Four upstream R defects were
reported separately (fmrigds #36–39); Rust does not reproduce those defects.
Qualification and remaining statistical/storage limits are in GROUP_ANALYSIS.md.

### Group performance tranche (2026-10-27)

The staged analytical plan and qualification are in GROUP_PERFORMANCE.md.
Implemented duplicate-free common-effect/zero-tau DL weighting, bounded tiled
meta-analysis scratch, compensated centered intercept OLS, and reusable
`PreparedOls` with general-fit workspaces and a complete-block path that avoids
gathering. First-level AR/OLS kernels are unchanged. Independent Decimal/R
references and group/facade checks cover the changed paths; release comparisons
retain raw trials, allocation diagnostics, source/binary hashes and adverse
whole-suite timing evidence. The subsequent tranche implements Paule–Mandel
derivatives as described below. Nuisance-subspace reuse, fit downdates, permutation
algebra and repeated-measures shortcuts remain later stages with explicit
mathematical and numerical acceptance gates.

### Weighted group regression checkpoint (2026-10-27)

The five deliverables are implemented in `fmri-group`: FE/DL covariate regression,
safeguarded PM with an analytic derivative, independent normal/mKH inference,
requested effects/marginal/joint covariance and p-values/intervals, and independent
numerical, calibration and performance evidence. `PreparedMetaRegression` binds
metadata and reuses row buffers; each feature/tau uses its own column-scaled,
pivoted weighted QR. Extreme-weight residual refinement and explicit arithmetic
failure boundaries preserve the estimand without variance floors. This full-rank
API leaves legacy group OLS/scalar meta-analysis and first-level AR/OLS kernels
intact. Contracts, reproducible commands, measured results and limitations are in
[GROUP_META_REGRESSION.md](GROUP_META_REGRESSION.md).

### Portable estimates and prepared execution checkpoint (2026-10-27)

Implemented version-1 scalar estimate bundles with checked identities, exact feature
geometry, optional genuine variance and provenance; bounded feature-major disk
sources and incremental writers; retained response/residual/contrast scratch; and
explicit shared-weight common-effect batching with exact eligibility checks and
general-solver fallback. Rust/R round trips, fresh metafor references, numerical
stress cases, matched baseline/candidate timings, and fresh-process peak RSS are
recorded in [GROUP_PORTABLE_EXECUTION.md](GROUP_PORTABLE_EXECUTION.md). The wire
profile deliberately supports one scalar estimand per independent subject;
within-subject joint estimands and repeated measures remain separate work.
