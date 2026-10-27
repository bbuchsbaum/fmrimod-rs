# Plan a cohort analysis before reading voxel data

`fmrimod::analysis` (feature `nifti`) joins immutable NIfTI sources, explicit
models, named consumer requirements and an execution budget. It orchestrates
existing sessions and regression kernels; the low-level APIs remain available.

Construct one `AnalysisSubject { source, request }` per subject. Its
`SubjectRequest` contains a `ModelSpec`, a model ID and `ConsumerRequirement`s.
Each consumer identifies its subject/model, named linear estimands, and required
uncertainty. See [the executable example](examples/analysis_viewer.rs) for complete
source, design, nuisance, run and AR specification.

```rust,ignore
use fmrimod::analysis::{CohortAnalysis, ExecutionPolicy, ExpectedUse, RetentionPolicy};

// subjects: Vec<AnalysisSubject>, already bound to named model columns.
let policy = ExecutionPolicy {
    expected_use: ExpectedUse::ContrastExploration,
    retention: RetentionPolicy::CoefficientsAndScale,
    external_reserved_bytes: group_collector_bytes,
    ..ExecutionPolicy::new(3 * 1024 * 1024 * 1024)
};
let mut analysis = CohortAnalysis::new(subjects, policy)?;
let plan = analysis.inspect()?; // no voxel reads, preparation or fit
// Display plan.storage, workers, reservations and per-subject decisions.
let receipt = analysis.run(|subject_index, output, compiled| {
    // Use compiled consumer maps to select rows in the identified output.
    consume(subject_index, output, compiled)
})?;

// New contrasts or a changed design; one request per original source, in order.
analysis.update(next_requests)?;
let next_plan = analysis.inspect()?;
```

## Outputs and retained state are separate decisions

| Consumer requirement | Returned result |
|---|---|
| `Uncertainty::None` | Requested estimates only |
| `Uncertainty::Marginal` | Estimates and marginal variances; standard errors are their square roots |
| `Uncertainty::Joint` | Estimates and joint covariance information |

The compiler unions named estimands, preserves each consumer's row mapping, and
uses the strongest requested uncertainty. A viewer may request estimates while a
mixed-effects consumer requires variances. Joint uncertainty is available when a
consumer needs covariance between contrasts. A coefficient is requested as a
named unit-weight estimand; requesting contrasts does not imply returning every
coefficient. Scientific semantics and column bindings must agree.

`RetentionPolicy::None` avoids retaining a full coefficient fit. Explicit
`CoefficientsAndScale` retains coefficients and residual scale internally so
later contrasts and their uncertainty need no time-series scan. It does not add
coefficients to the returned result. Even an estimates-only request pays this
retention cost if retention is explicitly selected. Retention requires resident
storage and is never silently dropped to meet a budget.

## Inspectable execution choices

`StoragePolicy::Auto` uses file batches for `ExpectedUse::OneShot`. For
`ContrastExploration`, it attempts resident storage, reducing the worker count
if necessary, then falls back to file batches. Explicit `Resident` or
`FileBatches` selects that storage constraint. Worker count is bounded by the
subject count and budget; block width is caller supplied.

The plan reports storage, workers, retained bytes, total numerical reservation,
reason, source version, model ID, noise policy, rank tolerance, uncertainty and
estimand IDs. Each subject also has data (`Load`, `ReuseResident`, `Reload`),
model (`Prepare`, `EstimateNoise`, `Reuse`) and fit (`ScanData`,
`BuildRetainedFit`, `UseRetainedFit`) actions. Requested AR errors, startup,
run boundaries, estimation sample and uncertainty are preserved. There is no
implicit IID fallback or automatic statistical approximation.

Reuse compares the actual model specification, including design and noise
policy, rather than trusting the model label. Contrast changes reuse preparation;
design changes prepare again and re-estimate noise when requested. A storage
transition discards the old execution state and is reflected in the next plan.
The execution receipt records actual reads, preparations, reuse and retained-fit
builds. No process-wide BLAS threading settings are changed by the library.

## Budget and lifecycle contract

The budget covers known numerical allocations, including conservative loading,
preparation, retained-fit and execution peaks. It is **not an RSS cap**: source and
request metadata, allocator excess capacity, backend scratch and unreserved sink
copies are excluded. Reserve a caller-owned numerical collector with
`external_reserved_bytes`. File batching bounds concurrent subject loads; it
still loads a subject's time series and does not promise streaming decompression
with constant memory. The inspection and file executor share their preflight.

Construction, inspection and request updates do not read voxel values. Source
construction reads metadata. Rank, estimability and data-dependent checks occur
when executing. A failed `update` preserves previous requests and fit state.
An execution error may leave prepared state and already delivered sink outputs;
execution is not a transaction over external sinks.

Sources and policy are fixed for the object's lifetime. Construct a new analysis
when either changes. Source versions are caller assertions, not automatic file
content verification: do not mutate admitted files behind a live analysis.

## Synthetic browser qualification

[The retained evidence](../fmrireg/evidence/analysis-plan/README.md) covers 25
synthetic subjects, two 120-frame runs, 21,824 masked voxels, six design columns
and estimated AR(2) per run using 64 sampled voxels. Fixed AR is also exercised.
The qualification viewer computes an unthresholded group OLS mean; it supplies
first-level variances when requested but does not claim mixed-effects fitting.
The grid is synthetic and is not evidence of MNI registration.

Run from the workspace root (NumPy and installed Playwright/Chromium required):

```sh
cargo build -p fmrimod --release --features nifti,accelerate --example analysis_viewer
python3 crates/fmrimod/scripts/analysis_viewer.py generate /tmp/my-analysis-input
python3 crates/fmrimod/scripts/analysis_viewer.py serve /tmp/my-analysis-input \
  --binary target/release/examples/analysis_viewer \
  --info /tmp/my-analysis-server.json --receipts /tmp/my-analysis-responses
# Open the localhost port recorded in my-analysis-server.json, or in another shell:
node crates/fmrimod/scripts/analysis_browser.cjs /tmp/my-analysis-server.json /tmp/my-analysis-browser
```

The server binds only to localhost. Stop it when finished; it closes its Rust
child. Automated browser runs create and close their own Chromium context.
On non-macOS systems omit `accelerate` to use the portable backend. The server
requests one BLAS thread per worker; callers must choose their own thread policy.
