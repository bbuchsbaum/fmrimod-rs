# NIfTI session/cohort qualification — 2026-10-27

The output foundation was committed as `6db03b1`. This subsequent tranche adds
NIfTI ingress, immutable resident subjects, exact model/noise invalidation,
bounded cohort batches and a genuine compact-summary bridge to `fmri-group`.
The new tranche remains uncommitted. Source hashes and check receipts are in
`manifest.json`; usage and limits are in `crates/fmrimod/SESSIONS.md`.

## End-to-end measurements

Synthetic on-disk cohort: **25 subjects × 300 timepoints × 8,192 voxels**, eight
design columns, float32 uncompressed NIfTI-1 input converted explicitly to f64,
fixed zero-history AR(4), one requested contrast with marginal uncertainty.
Group analysis is an intercept-only OLS model of the independent subject effects.
Apple M3 Max, release profile, Accelerate, four subject workers, one BLAS thread.

| Scenario | Four workers, final run | Four workers, repeat | One worker |
|---|---:|---:|---:|
| New session through HTML file write | 2,020 ms | 2,092 ms | 2,233 ms |
| Warm design edit through HTML file write | 95 ms | 85 ms | 250 ms |
| Contrast edit through HTML file write | 101 ms | 89 ms | 249 ms |

The final run spent 1,925 ms loading files, 2 ms preparing initial models,
86 ms in first-level execution and about 5 ms in the group reducer. The exact
component timings are in `latency-w4.json`. Warm contrast edits performed
**zero source reads and zero model preparations**; the whole three-stage run
made 25 source reads and 50 model preparations (initial + changed design).

These are descriptive runs on a shared developer host, not timing guarantees.
An unrelated project build was active during part of qualification. The OS file
cache was uncontrolled: “new session” means new decoder/data/model objects, not
guaranteed physical cold storage. Browser painting is not included. Gzip is
covered by input tests, but the cohort timing uses `.nii`, not `.nii.gz`.
Estimated-AR invalidation has a separate regression witness; noise estimation
was not part of this timed workload.

The numeric pipeline reservation was **536,770,688 bytes**, including resident
subject/model storage and the scalar collector, under a 1 GiB configured limit.
This is a conservative reservation for documented numerical allocations, not
measured peak RSS. Metadata, allocator excess capacity, backend packing,
decomposition-private scratch and arbitrary external sink copies are excluded.
All subjects must fit the resident budget; there is no automatic eviction yet.

## Numerical and lifecycle evidence

- Affected-package test run: **87 passed, zero failures/warnings**. After two
  equivalent lint simplifications, all eight affected NIfTI/cohort tests passed
  again. Scoped Clippy (`-D warnings`) and formatting checks passed.
- NIfTI tests cover hand-assembled values, x-fastest support/order, float/integer
  scaling, little/big endian, gzip, qform/reflection, sform, header mutation,
  unused NaNs, truncation and budget rejection before voxel loading.
- Session tests cover model reuse, row/column/noise identity, fixed startup/reset
  invalidation, estimated-AR design changes versus a fresh fit, deliberately stale
  whitening, tolerance constraints, transactional failures, overflow and slices.
- No-copy loading is witnessed by transferring a reader's numerical buffer under
  its exact peak budget and checking retained pointer identity. Logical clones
  avoid oversized backing-array copies after `slice_move`.
- Cohort tests cover deterministic order, aggregate one-byte-under/exact budgets,
  sink error/recovery, estimates-only and variance bridges, original df/noise
  provenance, and analytic group means/standard errors.
- All **8,192** group effects in each of the three scenarios agree exactly between
  one and four workers in the measured fixture.
- Independent NumPy SVD least squares on eight scattered coordinates across all
  25 subjects agrees with Rust group maps within **8.88e-13**; tolerance 2e-10.
  `verify.py` independently constructs the AR whitening and group mean. It does
  not call Rust kernels. See `numpy-reference.json`.

The generated [contrast map](contrast-edit.html) is an unthresholded synthetic
axial slice browser. Its JavaScript initial render and endpoint slider updates
were exercised with a mock DOM/canvas. This is **not** a visual browser acceptance
claim or a clinical/registration validation.

Independent review audited explicit allocation phases and found/fixed omitted
AR/correction/receipt buffers, premature output preparation, a duplicate full
reader copy, and ndarray clones retaining hidden backing elements. The final
accounting includes these known arrays and the original residual degrees of
freedom; the scratch/metadata exclusions remain explicit.

## Reproduction and storage

Run `crates/fmrimod/scripts/generate_cohort.py` to regenerate the synthetic
cohort, then the `cohort_latency` example as documented in `SESSIONS.md`.
`fixture-manifest.json` records source hashes, dimensions, seeds and NumPy version.
Inputs were generated at `/private/tmp/fmrimod-cohort-25` and removed after
hash/reference verification to recover disk space. No research or user data was
used. The HTML and receipts are retained here; restoring inputs requires only
the generator and NumPy, not an archived subject dataset.

The older output benchmark caches and this task's own temporary build artifacts
are regenerable. User-approved cleanup of older UV/virtualenv caches preserves
their checkout, source edits and evidence. No remote publication was performed.

Temporal projection remains deferred: these measurements identify file loading
as the dominant new-session cost, while warm edits already complete near 100 ms
on this workload. Larger real masks and compressed ingress should be measured
before choosing the next cache or algebraic optimization.
