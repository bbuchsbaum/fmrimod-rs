# Independent-subject group analysis

`fmri-group` implements the first group-analysis tranche in the fmrimod workspace:
identified scalar estimates, independent cohorts, group OLS, common-effect and
DerSimonian–Laird meta-analysis, block result sinks, and whole-family BH/BY.
The architectural assessment remains in [GROUP_ANALYSIS_AUDIT.md](GROUP_ANALYSIS_AUDIT.md).

Run the first-level-to-group example:

```sh
cargo run -p fmri-group --example group_pipeline
```

It fits synthetic first-level responses, extracts a named scalar t contrast for
each participant, assembles an admitted cohort, fits group OLS in two-feature
blocks, and corrects the complete six-parcel family. The fixture illustrates
plumbing; its deterministic noise is not an inferential calibration experiment.

## Contracts

A cohort contains one scalar scientific estimand over an ordered `FeatureAxis`.
Each subject has a dataset-qualified participant ID and a distinct observation
and measurement ID. Repeated participant rows are rejected even when their run
names differ. An `Estimand` carries a stable ID, definition and units; equal
labels do not automatically establish equivalence. `Admission` records the
caller's common-space target and supporting evidence. It does not claim that
coordinates alone establish registration.

`IndependentCohort::new` consumes a validated `FmriFrame`. Effects and optional
variance are identified assay roles on exactly the same observation and feature
axes. OLS can consume effects alone. Meta-analysis requires a separately supplied
variance assay and per-subject uncertainty origins: known, estimated with scalar
or featurewise degrees of freedom, or explicitly unknown. It treats those
variances as fixed **for inference**, while preserving their original provenance.
No unit variance is fabricated. Declaring an assay an effect remains the caller's
scientific responsibility; an arbitrary statistic map cannot be converted into
a scientifically meaningful effect by naming it.

`FirstLevelEstimate::from_t` offers a typed bridge from `fmrireg::TContrast`.
It retains effect, squared SE, residual df, the caller's noise/selection/model
receipt and approximate/conditional uncertainty flag. Assembly checks estimand
identity, units and exact feature-space compatibility. This bridge currently
requires valid finite marginals; manually constructed cohort frames support
per-cell missingness. F statistics and joint contrast covariance have no implicit
conversion. The concurrent requested-output producer work is preserved; this
tranche uses the stable t-contrast interface.

`GroupDesign` is explicitly bound to ordered observation IDs. Named contrast
terms resolve once, rejecting duplicate or unknown terms. Contrasts also bind the
design's ordered column names. Design cells must be finite. A singular design can
still estimate contrasts within its retained row space; residual df use numerical
rank. The returned `RunReceipt` retains the numeric design, contrast weights,
rank tolerance, original fit/variance provenance and missingness policy.

## Estimation and unavailable results

- OLS estimates residual scale and uses two-sided Student t inference. It reuses
  `fmrireg::LinearModel`'s SVD preparation for features with the same retained rows.
  A bounded FIFO cache prevents unbounded growth when missingness patterns differ.
- Common-effect meta-analysis uses inverse sampling variance and normal inference.
- DL additionally estimates nonnegative between-subject heterogeneity and uses
  inverse `(variance + tau²)` weights with plug-in normal inference. Q, I² and
  tau² are explicit outputs. No small-sample correction is implied.

Zero effects are valid. Nonfinite effects represent unavailable observations;
meta-analysis additionally requires finite positive variance. The default
`RejectFeature` policy reports unavailable features when any required observation
is invalid. `OmitObservation` explicitly performs featurewise deletion, with
per-feature n/rank/df. This computational policy makes no assumption that real
missing data are statistically ignorable.

Results carry status per feature and requested contrast: estimated, missing
observation, insufficient observations, unusable retained design, nonestimable contrast, degenerate
variance, or numerical failure. A constant/overflowing feature does not invalidate
neighboring features. Invalid global identity/design/options are request errors.
There is no hidden variance floor. Extremely ill-scaled arithmetic that cannot
produce finite outputs reports numerical failure rather than changing the model.

## Bounded execution

`fit_ols` / `fit_meta` read contiguous feature blocks through `fmridataset` and
transfer each `ResultBlock` to a fallible callback. Sink failure stops execution.
The dataset foundation currently supplies memory-backed f32/f64 sources and lazy
views; this is not a new disk-format implementation. Input is accumulated in f64.

`FitOptions` controls feature-block size, pattern-cache size, minimum observations,
rank tolerance and a conservative numeric working-buffer allowance. The receipt
reports the preflight estimate. This is **not** a hard process-RSS ceiling:
resident sources, metadata, allocator/backend scratch, and results retained by a
sink are outside it. Featurewise uncertainty provenance is shared with the input
rather than copied into every receipt. Global identity/provenance validation may
use metadata storage proportional to the feature/subject axes.

The callback can write/discard blocks to keep numeric fit storage bounded. The
benchmark intentionally retains its comparison outputs; those outputs are included
in process RSS, not in the fitting buffer estimate. BH/BY is a separate terminal
operation with O(F) family storage and sorting; it is not secretly block-local.

`TestFamily` has an explicit family ID, contrast ID, unique feature IDs and optional
p-values. Its denominator includes available tests only; absent tests remain absent.
Supply the complete scientific family, not one family per execution block. Rust
cannot infer whether the caller omitted scientific hypotheses from that list.

## Qualification

The integration suite covers:

- base-R QR references for covariate contrasts and feature-specific missingness;
- independent `metafor::rma.uni` FE/DL outputs and source-built fmrigds comparison;
- base-R BH/BY with ties, absent tests, zero and unit p-values;
- exact effect/variance axis binding and duplicate participant rejection;
- first-level bridge rejection for differing units or feature identity;
- per-feature rank loss, nonestimability and constant-feature isolation;
- block-size and row-order invariance, cache eviction, f32 sources;
- meta-analysis unit rescaling, DL analytic/boundary cases and valid zeros;
- budget rejection before sink output and immediate sink-error propagation;
- synthetic first-level → cohort → group estimation with uncertainty receipts.

Fixtures are generated by `scripts/group_reference.R`, using the isolated
fmrigds build from the source audit and installed `metafor`. Session/version and
source-hash receipts are under `evidence/group-tranche1/`. A fresh fixture-generation
run verifies fmrigds FE/DL against metafor before saving the Rust test inputs.
The live R checkout changed to `31e26ba5bd0bc088c71395077be1a13765512dab`
during this work. Numerical comparisons intentionally retain the isolated
`273b029bbe9c6c9024bfc12fe4ff6cdb98e18639` build: all 93 recorded source files
still match its build copy. `reference-drift.json` records the difference;
these measurements are not a qualification of the newer R checkout.
Independent numerical tests use 2e-10 scaled absolute tolerance on these small
well-conditioned cases; benchmark comparison uses 2e-9 for all requested outputs.

The null experiment uses 2,000 independent features, 20 observations, standard
normal errors and genuine unit sampling variance, R seed 2710. Reported rejection
rates are a limited calibration check, not qualification for estimated variances,
small heterogeneous cohorts, selection, correlated subjects or repeated measures.

## Reproduce benchmarks

```sh
LC_ALL=C Rscript --vanilla scripts/group_reference.R
cargo test -p fmri-group
cargo clippy -p fmri-group --all-targets --no-deps -- -D warnings
python3 crates/fmri-group/benchmarks/run.py r-benchmark \
  Rscript --vanilla crates/fmri-group/benchmarks/reference.R
cargo build --release -p fmri-group --example group_benchmark
python3 crates/fmri-group/benchmarks/run.py rust-benchmark \
  target/release/examples/group_benchmark
```

The runners set `FMRIGDS_THREADS`, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
`VECLIB_MAXIMUM_THREADS` to one. Rust uses the portable backend, not Accelerate.
Each workload records setup, first execution and five warmed executions. Timings
compare requested mean effect, SE, statistic, p, Q/tau² where applicable and n;
R's public pipeline also materializes its standard extra assays and provenance.
Thus these are application-path comparisons, not pure arithmetic microbenchmarks.
Preparation inside a fit, source reads, fitting and collection of selected outputs
are timed. Fixture generation, cohort construction, oracle loading, file writing
and FDR are outside the fit timings. Setup is recorded separately.

Every requested scalar is compared against fmrigds and block-size invariance is
checked at 256 versus 113 features. Raw records, five-run distributions, maximum
errors, null rejection counts and process-level peak RSS accompany the benchmark.
The original `/usr/bin/time -l` attempt finished R computation but failed on a
sandboxed `kern.clockrate` query; that log is preserved. The successful measurements
use child-process resource usage through Python and retain exit status.

See [GROUP_BENCHMARK.md](GROUP_BENCHMARK.md) for the measured results.

## Deferred boundaries

This tranche does not implement weighted meta-regression, PM/mKH, repeated LMM,
joint first-level covariance, spatial mapping, permutation tests, examination,
file-format interchange or experimental LDR. They retain their ordered acceptance
gates in the audit. No AR/OLS producer kernel was changed by this work. The next
numerical tranche is weighted meta-regression and explicitly selectable richer
heterogeneity/inference policies, with matching metafor reference fixtures.
