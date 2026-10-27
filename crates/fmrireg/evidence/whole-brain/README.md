# Whole-brain-sized cohort qualification — 2026-10-27

The preceding session/cohort tranche is committed as `f95d53f`. This performance
tranche adds typed float32 frame decoding, parallel subject loading, bounded
file sessions, and explicit coefficient/scale retention for contrast edits.
The Git commit containing these receipts binds this tranche. `manifest.json`
binds final source hashes and the measured binary; intermediate benchmark
provenance is identified separately.

## Workload and measured limits

25 synthetic subjects × 300 frames × 131,072 features (64 × 64 × 32), eight
regressors, float32 NIfTI-1 input converted to f64, one contrast with marginal
uncertainty, and the existing intercept-only group OLS reducer. This is a
whole-brain-sized feature count, not an anatomical mask or verified MNI dataset.
No research data was used. All map times end at HTML file write, not browser paint.

Apple M3 Max, 36 GiB RAM, Rust 1.91.1, release/Accelerate, one BLAS thread.
Four subject workers requested; the 1 GiB numeric-budget file path admits three.
The shared host and OS file cache were uncontrolled. Results are descriptive
runs, not isolated performance guarantees or physical cold-disk measurements.

| Path | New session → HTML | Design edit → HTML | Contrast edit → HTML | Peak RSS |
|---|---:|---:|---:|---:|
| Previous serial loader, fixed AR4 | 26.24 s | 1.11 s | 0.98 s | 8.47 GB |
| Parallel loader, original scalar decoding | 12.92 s | 0.94 s | 1.18 s | 8.68 GB |
| Typed decoding + optional retention, run 1 | 4.87 s | 1.98 s | 0.22 s | 8.71 GB |
| Same, run 2 | 5.19 s | 1.17 s | 0.85 s | 8.41 GB |
| Typed decoding + retention, estimated AR4 | 5.29 s | 1.33 s | 0.25 s | 8.54 GB |
| File batches, fixed AR4, 1 GiB numeric budget | 3.56 s | 2.19 s | 2.13 s | 2.16 GB |
| File batches, estimated AR4, same budget | 2.98 s | 2.12 s | 2.08 s | 2.15 GB |
| Gzip resident + retention, fixed AR4 | 8.40 s | 1.16 s | 0.14 s | 8.73 GB |
| Gzip file batches, fixed AR4 | 7.52 s | 6.47 s | 6.35 s | 2.15 GB |

Do not infer that estimated AR is faster from single noisy runs. Its policy is
explicitly global AR(4), correction lag 8, estimated using 256 spatially spread
features; this is not voxelwise AR or a recommended sampling policy for all data.
Design changes re-estimate it; contrast edits reuse it. Original residual df and
conditional/estimated-noise provenance remain attached to the summaries.

Typed decoding reduced resident input loading from 23.95 s in the baseline to
2.98–3.48 s in the final uncompressed runs. The implementation uses the NIfTI
provider's endian-aware typed frame decoder, then gathers masked/reordered
positions and scales in f64. It removes repeated coordinate/datatype dispatch,
without changing values or introducing a new file parser. Other datatypes retain
the scalar accessor path.

Retention is an explicit tradeoff, not a default. It stores 235,929,600 bytes
(225 MiB) of coefficients/scales for this workload and cost 3.01–4.58 s to build
in the two final uncompressed runs. **That cost is reported separately and is not
included in the contrast-edit column.** Requested outputs still contain only
the requested contrast and uncertainty. The cache saves subsequent time-series
scans; it need not pay off for one or two edits, particularly on a busy host.

The file executor pipelines load/prepare/fit, releases data, and keeps prepared
models between requests. It read 75 subjects over the three passes, prepared 50
models and reused 25. Resident execution read 25 subjects total, prepared 50
models, and performed no contrast-edit preparation. All fixed-noise paths share
the same inference; no requested AR model is replaced by IID errors.

## Memory is measured separately from reservations

The 1 GiB configuration bounds the documented numerical reservations, including
active subject data and the example's scalar collector. Final fixed-AR batches
reserved about 1.04 GB. Source/axis metadata, allocator excess capacity, stacks,
backend scratch, and caller-owned model specifications are excluded. The bound
is **not** an RSS cap. Final bounded runs measured 2.15–2.16 GB maximum RSS;
macOS peak memory footprint was 1.23–1.74 GB. Resident RSS was 8.41–8.73 GB.
Both OS measures, raw `/usr/bin/time -l` receipts, and reservations are retained.
The earlier `bounded.json` omits initial source admission from its cold timer;
it is superseded by `bounded-final.json`, not used in the table.

## Correctness and API evidence

- 93 affected-package tests passed, including the existing R/precision references.
- Final scoped Clippy with `-D warnings`, formatting, and default-feature portable
  facade compilation passed. Logs and commands are recorded in `manifest.json`.
- All 131,072 group effects at all three stages agree **bit-for-bit** across the
  baseline, parallel, typed, retained, bounded and gzip fixed-AR paths.
- Estimated-AR resident/file maps also agree bit-for-bit at all three stages.
- Independent NumPy whitening + SVD least squares and group means, at eight
  scattered coordinates across all subjects/stages, agree within **8.82e-13**
  against a tolerance of 2e-10 (`verify.py`, `numpy-reference.json`).
- Float32 endian/scaling/gzip/reordered-support tests agree bit-for-bit with both
  analytic values and the provider's scalar accessor. Integer ingress tests remain.
- Analytic retained-fit covariance, exact-model identity after moves, rejection by
  another model, model-change invalidation, failed-preparation preservation,
  exact/one-byte-under budgets, ordered loading, adaptive worker admission,
  source reattachment and sink error recovery have targeted regression coverage.

Usage and failure semantics are documented in
[the session interface](../../../fmrimod/SESSIONS.md). New APIs are
`ingress::{NiftiSubject, load_subjects, FileSession, run_file_cohort}`,
`SubjectSession::{retain_for_edits, clear_retained, detach}`, and the model-bound
`PreparedOutputs::summarize_retained` seam. OutputSpec and retention remain
separate decisions. File contents must remain immutable under their supplied
version; the library validates headers, but does not automatically hash voxels.

The [synthetic map](contrast-edit.html) is an unthresholded slice browser. No new
visual-browser or clinical/registration acceptance is claimed.

## Reproduction, custody and deferred cache choice

See `SESSIONS.md` for generator/example commands and environment controls.
`fixture-manifest.json` binds the uncompressed 3,932,168,800-byte fixture;
`gzip-fixture-manifest.json` binds its 2,665,410,245-byte compressed equivalent.
Each gzip file was decompressed and SHA256-compared before its generated original
was removed. `compress_fixture.py` and `compress.log` preserve that operation.
NumPy 2.4.3 and the generator hash/seeds/dimensions are recorded in the manifests.
The generated inputs and temporary build cache are removed after qualification;
restore the inputs with the generator, not from a research-data archive.

A durable decoded cache was evaluated, not enabled: repeated gzip file batches
pay roughly four extra seconds per pass here; retaining a decompressed cohort
would require another 3.93 GB plus a managed version/invalidation/custody contract.
Resident sessions already avoid repeat decoding, and explicit fit retention avoids
contrast scans. Callers can currently choose uncompressed input or resident data
when that tradeoff fits. Automatic durable storage and temporal projection remain
outside this tranche; neither is needed to realize the measured improvements.
