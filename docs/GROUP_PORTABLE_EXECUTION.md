# Portable estimates and prepared weighted execution

Mote: `bd-01M5Z8P1HFTVRDA30R7B9BQDCF`.

Outcome: deliver the five authorized items: versioned estimate interchange,
bounded disk-backed input/output, reusable weighted fitting scratch, an explicit
shared-weight batch path, and independent end-to-end qualification.

Scope: scalar first-level estimands for independent subjects, existing Index and
Volume feature spaces, genuine optional marginal variance and its provenance.
Preserve the first-level solvers, rank policy, missingness and extreme-weight
failure boundaries. Publication and new statistical models are outside this tranche.

Contract: a JSON manifest declares version, subject and estimand identities,
exact geometry/order, variance origin, fit/admission provenance, shape-derived payload lengths
and SHA256 hashes. Fixed-name little-endian f64 files use feature-major storage.
Publish the manifest only after all contiguous blocks are written. Reads validate
before fitting and obey byte budgets. Files must remain immutable while in use.
This is an explicit estimate profile, not serialization of arbitrary frame state.

Prepared execution: reuse response, residual and contrast buffers without changing
the qualified weighted QR algorithm. Retain nalgebra's factorization allocations
where its public interface requires ownership. Shared-weight execution explicitly
binds row order, design and relative weights and uses one QR to solve response
blocks. Check eligibility against actual finite observations and supplied
variances; incompatible features and heterogeneity refits use the general path.
No rounded weight keys, variance bins or approximate factorization reuse.

Acceptance: Rust and independent R round trips; metadata/geometry/identity and
corruption rejection; budget, incomplete-write and selection tests; disk/memory
fit parity; shared/general and block-size parity including rank and missingness;
all existing weighted numerical witnesses; matched release timings and allocation
counts against the preserved baseline, plus separate I/O/preparation and fresh
process peak-memory measurements. Numerical output equivalence precedes timing.

Paths: `fmri-group::{portable,meta_regression}`, the fmridataset file source,
`scripts/group_portable_*`, and `evidence/group-portable-execution/`.

## Delivered API

`PortableMetadata` carries subjects, one estimand, the feature axis, optional
variance origins, first-level fit receipts and the explicit scientific admission.
It can be built directly without an in-memory cohort, or copied with
`PortableMetadata::from_cohort`. `EstimateBundleWriter::create` requires a new
directory; its separate effect/variance block methods accept contiguous feature
ranges. `finish` publishes a complete manifest atomically without replacing one.
`write_cohort_bundle` is the convenience wrapper for an existing f64 cohort.
`read_cohort_bundle` returns an `IndependentCohort` backed by file sources.

The [versioned format](GROUP_ESTIMATE_FORMAT.md) deliberately excludes arbitrary
frame annotations and additional assays. It supports Index and Volume geometry,
scalar or featurewise degrees of freedom, and known/estimated/unknown variance
provenance. Missing values remain data; availability and positive-variance rules
are applied by the chosen estimator. Unknown versions and nested fields are
rejected. R rewrite uses 17-digit JSON precision, and Rust uses round-trip float
parsing to preserve nontrivial affine/df values.

`SourceSpec::feature_major_f64_file` also works independently of the bundle. It
checks the file size and uses bounded reads with row-major output. Contiguous
feature blocks use chunked sequential I/O; arbitrary reordered selections retain
an exact gather path. Scratch is at most 64 KiB and included in reader preflight.
Reader metadata caps, writer accepted-block limits and fit workspace limits are
separate. Writer I/O buffers and metadata are additional bounded overhead; the
limits are not a promise about total process RSS. Payload immutability is a
requirement while a file-backed cohort is used.

The [first-level example](../crates/fmri-group/examples/portable_first_level.rs)
follows the actual numerical producer path: fit each subject's response block,
extract a t contrast and its variance, write one cohort feature block, then run
PM/mKH group inference from the completed files. It holds neither the whole
cohort's estimates nor the whole first-level response collection in RAM. Its
synthetic OLS noise model is explicit; it is not an AR fallback.

```sh
cargo run -p fmri-group --example portable_first_level -- /tmp/new-estimate-bundle
```

### Shared-weight execution

Build the usual prepared plan, then opt in with
`plan.with_shared_variances(reference_variances)`. This vector contains relative
**variances**, in the bound design's observation order. Eligibility requires
complete finite rows and bit-identical normalized f64 weights calculated from
actual input variances. Proportionality is not guessed with a tolerance. Difficult
geometries and any failed numerical check use the qualified general path. DL and
PM always use the general path in this tranche.

Eligible common-effect features share one column-scaled pivoted QR and matrix
right-hand-side solves. Their own total variance scale, residual Q, inference
policy and requested outputs are preserved. `receipt.shared_features` reports
actual use, and `shared_reference_variances` records the declaration. General
factorization counts exclude the shared factorization performed during plan
preparation; preparation is timed separately. Repeated fits reread their data.

Response, residual, coefficient and contrast geometry scratch is reused in the
general path. nalgebra still allocates its owned QR factors, and returned feature
results still own their vectors. This is not a zero-allocation API.

## Qualification

All **94 tests passed** across fmridataset, fmri-group and the facade with all features/targets; scoped clippy and formatting pass. Doctest targets succeed with zero current doctests. These are local macOS arm64 checks, not hosted or cross-platform qualification.

Independent evidence includes the existing weighted regression/metafor stress suite, the analytic weighted-mean shared-path witness, exact fallback tests, raw little-endian file fixtures, a non-column-aligned 64 KiB scratch-boundary test, incomplete/corrupt bundle rejection, Index/Volume and origin/df round trips, a fractional-affine/df R rewrite, and fresh metafor comparisons of both the synthetic shared-weight fixture and actual first-level outputs. The prior 12,000-fit calibration comparisons continue to pass; no additional population calibration is claimed.

### General weighted fitting

At 100 subjects × 10,000 features × five predictors with marginal uncertainty:

| Method | Previous allocations | Current allocations | Reduction | Current / previous runtime |
|---|---:|---:|---:|---:|
| FE | 291,143 | 91,183 | 68.7% | 0.976 |
| DL | 440,503 | 156,528 | 64.5% | 1.003 |

The memory improvement is clear; the general runtime improvement is modest and workload-dependent. At 20 × 200 × 3, some FE/DL trials remain slower. The comparison pools two five-trial runs per version in whole-suite ABBA order on a shared workstation; temporal variability limits causal attribution. It is not an isolated-host microbenchmark. All requested outputs were checked against the independent reference payloads before timing.

### Shared-weight and disk-backed fitting

Common-effect regression, all coefficients and marginal SEs, block size 256, five warm fits per fresh process. The fixture has explicitly proportional sampling variances. Times include bounded reads, fitting and a streaming checksum sink, but exclude initial checksum verification and preparation.

| Subjects × features × predictors | Storage | General (ms) | Shared (ms) | Speedup | General / shared peak RSS (MiB) |
|---|---|---:|---:|---:|---:|
| 20 × 1000 × 3 | disk | 1.32 | 0.59 | 2.23× | 3.5 / 2.6 |
| 20 × 1000 × 3 | memory | 1.21 | 0.49 | 2.48× | 3.4 / 3.0 |
| 100 × 100000 × 5 | disk | 614.69 | 267.34 | 2.30× | 38.9 / 44.6 |
| 100 × 100000 × 5 | memory | 569.70 | 225.54 | 2.53× | 205.0 / 203.2 |

All 100,000 large-fixture features use the declared shared path. Actual first-level subject variances generally need not share this geometry: these speedups are conditional on eligibility. Every output was compared across general/shared and disk/memory execution before timing.

The large bundle has 160 MB of numeric payloads. Opening and checksum verification took about 0.38 s; materializing both assays added about 0.05 s. General/shared numeric workspace estimates were 2,081,760 / 2,932,960 bytes. The larger measured RSS includes feature IDs, parsed metadata, allocator state, code and—in memory mode—the resident arrays. Memory comparison mode explicitly permits 256 MiB per full-assay read; streaming fits retain their 64 MiB numeric budget. Allocation totals are cumulative requests, not peak RSS.

Warm fits use operating-system file caching; no cold-storage or filesystem-cache eviction claim is made. Raw first/preparation/warm times and all process results are retained.

## Reproduction and evidence

```sh
python3 scripts/group_portable_benchmark.py --run
```

The runner uses the preserved previous executable and prior independent reference
inputs under `evidence/group-portable-execution/`. It creates a new timestamped
run, records source and binary hashes, and rejects source changes during execution.
It runs affected all-feature tests, doctests, scoped clippy and formatting;
executes the real first-level example; checks Rust/R round trips and fresh metafor
references; compares general/shared disk/memory results before timing; and records
warm trials, cumulative allocations, preparation/checksum/materialization times,
and per-process peak RSS. Builds do not overlap its timed runs.

Final evidence is linked by [latest.json](../evidence/group-portable-execution/latest.json).
Each run retains `commands.json`, `source.json`, `binary.json`, `artifacts.json`,
`workspace-comparison.json` and `process-results.json`. The input bundles are
synthetic and regenerable from the example. There is no publication step.

Earlier receipts are retained: the first measurement exposed unnecessary clones
inside nalgebra's unchanged-shape resize calls; the corrected code avoids those
calls. The macOS `time -l` wrapper failed to read a sandbox-restricted sysctl after
its child had run, so it supplied no valid RSS result. Final measurements use
`getrusage(RUSAGE_SELF)` inside fresh benchmark processes. The low-allocation
implementation's slower small-workload timings remain in the reported evidence.

