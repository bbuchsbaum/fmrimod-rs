# HRF and design capability audit

The target is efficient Rust modeling capability. R supplies independently
computed numerical cases; ScalaFIM supplies useful typed contracts and algorithms.
This audit does not equate API count with scientific parity.

## Reference review before implementation

ScalaFIM working source at HEAD `ac2ec8b706fce08596a943d79f71a140aad05948`
was consulted read-only. `evidence/scala-design-reference.json` fingerprints the
specific files; no Scala compilation or runtime parity is claimed.

* `HrfDescriptor` and `HrfIdentity` distinguish structural parameters from names.
  Rust now uses explicit portable recipes for supported HRFs and structural
  penalty policies. Arbitrary function pointers are not serialized by guessing
  their names. A complete Scala descriptor hierarchy was not transplanted.
* `HrfFunctions.empirical` uses linear interpolation and binary interval lookup.
  Rust adopts these defaults, with explicit opt-in cubic interpolation.
* `Deriv` selects analytic SPMG derivatives and treats support boundaries
  separately. Rust exposes available analytic derivatives through trait objects,
  shares one derivative implementation, and batches numerical evaluations.
* `Penalty` selects a policy rather than inspecting a display name. Rust adopts
  that separation; default custom HRFs get identity regularization.
* `ConvolvedContrastWeights` aligns contrast rows to convolved columns. Rust
  standardizes expansion in basis-major order, matching its event design.

## Capability and acceptance matrix

| Capability | Status after this tranche | Evidence or remaining limit |
|---|---|---|
| SPMG1/2/3 evaluation and temporal/dispersion columns | Implemented; current R verified | Existing regenerated contracts plus `hrf_audit`; nondefault parameters preserved by typed recipes and design specifications |
| Fractional impulses, durations, variable TR, run offsets, empty runs, run isolation | Implemented; bounded cases verified | Existing convolution/current-design contracts; three-run multibasis end-to-end pipeline |
| Empirical HRF interpolation | Corrected and verified | Default linear agrees with current R; sorted distinct finite samples; binary search; cubic explicit, knot-exact, no silent fallback |
| Derivatives | Analytic SPMG1/2 object dispatch; batched numerical fallback | Current R derivative values plus analytic endpoint slope check; SPMG3 falls back to numerical differentiation; discontinuous interior knots have no claimed unique derivative |
| Fourier HRF | Corrected and verified | Alternating unnormalized sin/cos columns, no intercept, any positive count, compact support; R checks odd/even counts and endpoints |
| FIR, endpoint spline, tent | Implemented and verified for sampled cases | FIR is half-open at final endpoint; reconstructed values and roughness penalties match R |
| Reconstruction and coefficient weighting | Matrix path verified | R basis-matrix × coefficients; dimension, shape and nonfinite guards; no new penalized-estimation API claimed |
| Penalty policies | Corrected and verified | R matrices for SPMG/FIR/Fourier/spline/tent; FIR previously missed roughness dispatch because its name differed; arbitrary names no longer select model behavior |
| Naming and order | Existing sanitization retained; core design names checked | Current R names mapped by documented redundant-prefix removal; basis-major order explicit; no new universal Unicode naming parity claim |
| Contrast selection/alignment | Core bugs fixed | Exact legacy simple/pair selectors; modern pairs use exact names or final dot-separated levels; explicit regex alternative, central basis-major expansion, missing-group rejection; A/AB and multibasis unit/pair/poly/F regressions |
| Estimability and rank | Corrected relative-tolerance diagnostics | SVD row-space projection; design and contrast scaling to avoid overflow/underflow; singular-design and extreme-scale counterexamples |
| Horizontal design composition | Time-grid safety fixed | Mismatched/nonfinite grids error; matching times do not prove run or acquisition identity—use explicit observation IDs at fitting boundary |
| Portable serialization | Supported recipes verified; unsafe restoration rejected | Nondefault SPMG, canonical normalization, empirical samples/interpolation, FIR/spline/Fourier recipes; full evaluated-regressor roundtrip tests. Legacy name-only and heterogeneous/custom recipe restoration errors |
| Design `HrfSpec` serialization | Implemented and checked | Tagged structured data preserves configured SPMG and sampled values; validation occurs at construction; arbitrary closures remain outside this schema |
| Feature-regressor families | Partial equivalent capability; broader API deferred | Borrowed multi-column event amplitudes handle event features; no claim of full R `feature_regressor` object/workflow parity |
| Nuisance/formula breadth | Existing implementation retained; partial current-source qualification | Run baselines, polynomial/spline nuisance cases and core formulas tested; broad external covariate joins and every R formula extension not claimed |
| End-to-end estimation | Independently checked | Formula events → multibasis design → known stationary AR(2) → ID-bound streamed GLS → t inference; R uses explicit dense covariance solves; aliased-design contrast rejection tested |

## Deliberate compatibility changes

Empirical interpolation no longer changes implicitly when a fourth sample is
added. Request `empirical_hrf_with_method(..., InterpolationMethod::Cubic)` for
cubic interpolation. Fourier HRFs now match the current R sine/cosine convention;
consumers relying on the old normalized constant-plus-harmonics basis must build
that basis explicitly. FIR evaluation at the final endpoint is zero.

Legacy `Contrast::simple/pair` means exact column names. Use `Contrast::column`
for explicitly requested patterns or metadata-based contrast APIs. Unsupported
legacy polynomial degrees and vector-valued F requests error instead of returning
linear/zero weights. The richer polynomial/F API remains available.

`Regressor::try_to_data` checks portable recipe support before writing data.
`to_data` remains available but may contain no recipe for unsupported HRFs;
`from_data` rejects those payloads and legacy name-only records. Registry lookup
cannot recover an unrecorded normalization choice, sampled curve or parameter.
Event-specific HRF serialization and decorator/custom closure recipes remain
explicitly unsupported rather than silently losing information.

## Numerical and performance boundaries

R fixtures are regenerated against the isolated source builds. Source/binary
provenance from prior checks and current-source hashes are retained. Tolerances
are 1e-12 for simple interpolation/penalties, 2e-10 for HRF numerical references,
and 2e-8 for the complete GLS inference path. Failures are retained, not waived.

Batch derivatives require two vector evaluations rather than two allocations and
evaluations per sample. Binary search replaces a linear scan per empirical query.
These are implementation complexity improvements, not new measured speedup claims.
The previous AR-GLS benchmarks remain scoped to their recorded sources/workloads.

Acceptance uses focused independent references and a fresh bounded all-feature
workspace campaign, with a separate receipt label to preserve the initial suite.
The master programme still tracks deferred capabilities; completion of this
tranche is not a claim of every R/ScalaFIM feature or downstream consumer parity.


## Completed verification

The all-feature workspace campaign passed 891 tests in 81 bounded commands,
with 21 existing ignored doctests. A final missing-pair-group guard was followed
by affected design/facade rechecks: 508 passed, 20 existing ignored doctests.
The minimal HRF build passed 168 tests. Formatting passed. Clippy completed,
with legacy warnings retained; this is not a warning-free workspace claim.
Counts from overlapping rechecks are not added to the workspace total.

`evidence/hrf-design-verification.json` binds those receipts to the final source
manifest. `scripts/verify_hrf_design.py` also checks R fixture/reference and Scala
source fingerprints. The original-source preservation check confirmed 921 files
unchanged. The initial all-feature compile error, obsolete Fourier DC test,
swapped empirical arguments and substring-selector assumptions remain recorded
in the adverse receipts; their corrected tests and implementation pass.

Reproduction:

```sh
LC_ALL=en_US.UTF-8 R_LIBS="$PWD/evidence/r-library" Rscript --vanilla scripts/generate_hrf_audit_contract.R
Rscript --vanilla scripts/generate_pipeline_contract.R
python3 scripts/check_bounded.py --label hrf-design-recheck
cargo test -p fmridesign -p fmrimod --all-features
cargo test -p fmrihrf --no-default-features --lib --test hrf_audit
```

`--resume` on the bounded runner reuses successful targets and is appropriate
only when edits affect failed or unrun targets; use a fresh label for a new
source-wide campaign. Historical receipts are preserved separately.
