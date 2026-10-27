# Scientific evidence and provenance

Current entry points are the [completion report](../docs/COMPLETION_REPORT.md),
[repository layout](../docs/REPOSITORY.md), and [capability matrix](../docs/CAPABILITIES.md).
Numerical test fixtures live with their crates. This directory retains reviewed
receipts, reference inputs, oracle summaries, source hashes and benchmark results.

## Generated artifacts and recovery

Raw logs, duplicate payloads, source snapshots and nested Git bundles are not
part of the current public source tree. The local files are preserved. The
[archive index](repository/archive-index.json) records each original/current
path, SHA-256, size, reason and any retained replacement. The
[preservation receipt](repository/preservation.json) records the complete verified
pre-cleanup archive. Historical receipt bodies remain unchanged; their paths and
source hashes describe their original checkpoints.

To recover a single archived file in a fresh clone, use its `original_path`:

```sh
git show 9a0fc9f:ORIGINAL_PATH > RESTORED_FILE
```

Or extract the verified archive into an empty directory. On this machine it is
stored in the sibling `fmrimod-archives` directory; the receipt gives its exact
path and hash. The archive contains all previously tracked files, not ignored
runtime payloads. Reproduction scripts regenerate their own outputs. Large
ignored data that already existed locally remain in place.

New generated logs and dated run directories are ignored by default. Add compact,
reviewed evidence explicitly when it supports a scientific or performance claim.
Do not change old receipt contents to match renamed crates. First-level evidence
formerly under `crates/fmrireg/evidence` now lives under `evidence/regression`.

## Earlier stage index

The following index is historical; paths may refer to the archive or original
checkpoint layout. Use the archive index when a raw file is not tracked here.

# Evidence

`source-manifest.json` captures original working files (SHA-256), original Git
HEAD/status and R source revisions. `scripts/check_source_preservation.py`
checks the originals against that capture. Local `*-history.bundle` files retain
Git history, while originals also remain in place. Builds and agent state were
not imported. Rust dependencies are pinned by the root Cargo.lock.

Reference builds live in ignored `r-source/` and `r-library/`. They were copied
from the recorded checkouts and compiled with `R CMD INSTALL --preclean`;
ambient installed fmriAR was a different version and was not used as the oracle.
Regenerate from the workspace root with `R_LIBS=$PWD/evidence/r-library` and
`LC_ALL=en_US.UTF-8`:

* `Rscript --vanilla scripts/generate_estimation_contract.R`
* `Rscript --vanilla scripts/generate_spline_contract.R`
* `Rscript --vanilla crates/fmrihrf/scripts/generate_convolution_contract.R crates/fmrihrf/tests/fixtures/convolution_contract.json`
* `FMRIHRF_FIXTURE_OUT=$PWD/crates/fmrihrf/tests/fixtures Rscript --vanilla crates/fmrihrf/scripts/generate_r_fixtures.R`
* `Rscript --vanilla crates/fmridesign/scripts/generate_r_core_contract.R crates/fmridesign/tests/fixtures/r_core_contract.json`

Fixtures include provenance; ordinary Rust tests do not require R. Legacy CSV
HRF fixtures share the source build recorded by the convolution and spline JSON.

Retained adverse receipts: `workspace-baseline.log` records disk-exhaustion
linker errors; `workspace-baseline-compact.log` records two inherited spline
README/doctest defects; `current-r-before.log` records HRF failures against fresh
R fixtures; `current-contract-tests.log` and the first all-features receipt
record stale design scaling and SPMG3 derivative semantics. Those failures were
not waived. Final check receipts are summarized separately in verification.json.


The full all-features command exceeded available disk space while retaining
all linked test binaries. `python3 scripts/check_bounded.py` executes the same
workspace libraries, each integration-test target, each example test harness,
and all doctests with all features, removing only successful Cargo-reported
executables in this workspace's target directory. `bounded-checks.json` records
exact commands, exit codes, counts and log hashes. It is the initial full-workspace suite receipt;
earlier monolithic failures are retained as environmental evidence.

`qualified-source.json` fingerprints the initial qualified Rust sources, tests/fixtures and
manifests. `verification.json` binds the receipts to that manifest and toolchain.
Legacy lints remain; strict Clippy is scoped to the new noise/regression kernels
with --no-deps. The earlier facade-inclusive Clippy log exposes inherited spline
lints and is not claimed as passing.


AR(p) continuation: `arp-verification.json` separately fingerprints the affected
sources and records actual exit codes for the full-feature noise/regression/facade
checks (19 passed), minimal regression (8 passed), strict kernel Clippy and
workspace formatting. Earlier full-workspace receipts were not relabelled as a
fresh run. Regenerate the new oracle with:

```sh
LC_ALL=en_US.UTF-8 R_LIBS="$PWD/evidence/r-library" Rscript --vanilla scripts/generate_arp_contract.R
```

The fixture records current fmriAR source/binary hashes. It includes 16 fixed-order
estimation cases, five independent stationary covariance/GLS cases and two ACVF
repair cases. The expanded cases exposed correction inverse error: one AR(1)
coefficient was 0.972385315455001 instead of R's 0.9725493687616034.
Pivoted LU for this full-rank correction system resolved the discrepancy;
coefficient tolerance remains 2e-9 and bias-map tolerance 1e-12. These checks do
not establish AR-GLS performance or automatic-order-selection behavior.


AR-GLS performance tranche: `gls-baseline/`, `gls-final/`, and `gls-memory/`
retain per-process phase timings, output hashes, peak RSS and source/executable
metadata. `gls-validation.json` records all-output R comparisons, block/worker
invariance, the 200k-feature prefix check, and R source/binary provenance.
See docs/PERFORMANCE.md for exact scope and reproduction. Raw binary outputs
are retained locally and ignored in Git. `gls-checks.json` records the affected
crate checks after the contiguous residual-variance optimization.


HRF/design audit tranche: `hrf-design-final.json` is the fresh all-feature
workspace receipt (81 batches, 891 passed, 21 ignored), with named per-target
logs. `hrf-design-verification.json` adds affected-package post-checks after the
last guard, minimal HRF checks and formatting/Clippy disposition, bound to
`hrf-design-source.json`. `scala-design-reference.json` records the read-only
ScalaFIM source consultation. The new fixtures are `hrf_audit.json` and the
facade's `pipeline.json`, regenerated by their scripts and verified by
`verify_hrf_design.py`. See docs/HRF_DESIGN_AUDIT.md for compatibility and scope.
Do not reinterpret old source snapshots or benchmark manifests as current
whole-workspace qualification; each receipt identifies its own evidence.
