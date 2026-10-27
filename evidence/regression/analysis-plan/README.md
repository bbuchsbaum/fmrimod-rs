# Cohort planning and browser qualification

Completed 2026-10-27. User approved representative synthetic data. The API is
documented in [ANALYSIS.md](../../../fmrimod/ANALYSIS.md).

The fixture has 25 float32 gzip NIfTI subjects, 48×48×32 voxels, an ellipsoid
mask of 21,824 voxels, and two 120-frame runs. Six design columns include run
intercepts, a task, a nuisance regressor and run drifts. Generated AR(2) noise
resets at each run. `fixture.json` records seeds, generator version and all input
hashes. This is a synthetic grid, not MNI registration or clinical evidence.
Compressed input size was 392,984,926 bytes; generated inputs were removed after
verification. Regenerate them with the committed Python script and NumPy 2.4.3.

## Observed latency

Apple M3 Max, macOS 14.3, release Rust with Accelerate; four requested cohort
workers and one BLAS/OMP thread per worker. The planner selected four workers
for resident execution and two for file batches under a 128 MiB budget.
Resident execution explicitly retained coefficients and residual scale. Its
numerical reservation was about 1,054 MiB (1,084 MiB during a design update).
File batching reserved about 100 MiB. Reservations are not measured peak RSS;
metadata, backend scratch and browser/Python memory are outside that contract.

| Request | Browser render opportunity (ms) | Screenshot-confirmed upper bound (ms) | Reads / model preparations |
|---|---:|---:|---:|
| First estimated-AR result | 1641.4 | 1689.8 | 25 / 25 |
| Contrast edit | 73.8 | 108.9 | 0 / 0 |
| Design edit | 397.0 | 432.3 | 0 / 25 |
| Repeat | 73.3 | 108.7 | 0 / 0 |
| Estimates only | 73.4 | 109.6 | 0 / 0 |
| File batches | 1881.9 | 1917.8 | 25 / 25 |
| File contrast edit | 1582.5 | 1619.1 | 25 / 0 |
| File fixed-AR request | 1565.0 | 1600.1 | 25 / 25 |

These are single observed runs, not a distribution or a backend speed ranking.
The first request includes source admission, I/O, noise estimation, fitting,
group reduction, JSON transport and canvas update. "First" means cold analysis
state, not an emptied OS disk cache. The browser clock ends after canvas update
and two animation frames; screenshot completion provides a separately measured
render-confirmed upper bound. Headless Chromium is not physical display latency.
Estimates-only uses the already retained fit, so it does not measure savings
from avoiding initial residual-scale computation.

The localhost viewer runs the real Rust backend and renders a browsable axial
map. Playwright 1.62.1 / Chromium 151.0.7922.34 exercised eight cases, inspection
without fitting, and both slice-slider endpoints. Screenshots were captured for
each case; the design-edit screenshot was visually inspected. No page errors.
The group result is an unthresholded OLS mean; marginal first-level uncertainty
is supplied when requested, but this demo does not fit mixed effects.

## Correctness and build evidence

- 102 affected-package tests passed, including four new planner tests. Some
  existing tests cover unrelated selected-observation work in the live workspace.
- Strict scoped Clippy, formatting, release example build and the portable
  no-default-features build passed. Commands and exit status are recorded below
  and in `checks.json`; complete logs are retained.
- `verify.py` independently reconstructs the design and run-block whitening,
  uses NumPy SVD least squares, and checks effects, residual variance and group
  mean/SE at four spatial probes for all 25 subjects and eight requests. Maximum
  effect error was 5.76e-12; maximum variance error 7.43e-16, against an absolute
  tolerance of 2e-9. This is conditional GLS given the reported AR coefficients;
  it is not an independent AR-estimator oracle.
- Full group effect maps match exactly for resident, repeat, estimates-only and
  file-batch requests with the same model and contrast. Design edits change
  estimated AR coefficients (maximum change 0.0034154).
- Planning tests remove admitted input files before constructing/inspecting a
  plan, proving that these operations do not read voxel values. They also cover
  consumer union/binding, failed-update preservation, reuse, budget rejection,
  worker reduction and file execution at its advertised numerical budget.

```sh
cargo test -p fmridataset -p fmrimod -p fmrireg --features fmrimod/nifti,fmrimod/accelerate
cargo clippy -p fmridataset -p fmrimod -p fmrireg --features fmrimod/nifti,fmrimod/accelerate --all-targets --no-deps -- -D warnings
cargo fmt -p fmridataset -p fmrimod -p fmrireg --check
cargo check -p fmrimod --no-default-features
```

`responses/` contains losslessly compressed HTTP command/result receipts,
including full maps and probe values. To rerun the independent witness,
decompress these into a temporary response directory, regenerate the fixture,
then run `python3 verify.py INPUT RESPONSES OUTPUT.json`.

`source-hashes.json` binds the live workspace's crate sources, tests, examples,
scripts and Cargo files; `checks.json` records base HEAD and release binary
hash. The workspace contains pre-existing uncommitted provider and
selected-observation code. This evidence does not claim that this isolated
commit reconstructs those providers from a clean checkout.

Browser ownership audits found no automated top-level browsers before or after
the test. Chromium/context were closed in `finally`; `cleanup.json` confirms
that the task-owned HTTP server and Rust child exited. The shared Cargo target
was preserved for the other active workspace task. No service was published.
