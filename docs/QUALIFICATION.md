# Integrated qualification, 2026-10-27

This checkpoint evaluates the assembled workspace and selected-observation
stationary AR estimation/GLS. It does not add a new estimator or silently repair
statistically unsupported cases. The R source-drift audit reports no changes to
the five recorded reference checkouts. Original imported source preservation is
checked separately from the modified consolidated workspace.

## Complete first-level walkthrough

```sh
cargo run -p fmrimod --features formula --example complete_first_level
```

The `formula` feature exposes fmridesign through the facade; this example uses
numeric constructors, without parsing a formula. Its two runs have 80 and 70
scans, TRs 0.8 and 1 second, and default acquisition offsets TR/2.

1. Construct run-local duration-bearing events and an SPMG1 HRF. Convolve them
   on the original acquisition frame, labeling the task column.
2. Prepare sampled sound features as half-second bins, with per-run centering,
   and generate named feature columns on the same frame.
3. Add DCT drift with a 32-second cutoff, separate run intercepts, and named
   run-specific motion covariates. Verify the shared acquisition grid while
   composing the 14 columns.
4. Mark eight original scans as censored. Preserve original row indices and
   times, and create an `ObservationSelection` from the original run lengths.
5. Generate 32 synthetic AR(2) responses with known task/feature effects, then
   estimate a shared stationary AR(2) coefficient vector from retained OLS
   residuals with correction lag 10 and `GapPolicy::Preserve`.
6. Validate the full original observation-ID sequence before constructing a lazy
   selected assay. Bind selected IDs to the model and stream seven-feature blocks
   with two workers and a 2 MB working-buffer budget.
7. Resolve the exact named task contrast and assert every streamed t statistic
   matches the dense result. Feature IDs and total delivery are checked too.

Recorded output: 142 retained observations, design rank 14, residual df 128,
32 delivered features, and estimated AR coefficients approximately
(0.42745, −0.13343), compared with generating coefficients (0.4, −0.15).
The output retains actual effects and statistics, including weak/noisy estimates:
raw HRF scale and this small event set do not imply a powerful experiment. This
is an executable integration example, not empirical inference validation.

The generating noise uses a fixed seed and a 256-step burn-in for each run. The
same response set supplies the representative noise data in this example, so
its t statistics are plug-in results, not evidence for same-feature calibration.
The simulation stress cases below instead use independent inference targets.

## Measurement contract

Build and rerun the release driver and collection script:

```sh
cargo build --release -p fmrireg --features dataset --example selected_qualification
python3 scripts/qualify_selected.py
```

The build uses the portable default backend, not the optional Accelerate feature.
External BLAS/OpenMP thread environment variables are set to one. Cases run
serially after workspace compilation finishes. Each fresh process measures:

- One preparation, including retained OLS, finite-lag AR correction and GLS
  preparation, using 32 representative noise features and lag budget 12.
- A first dense fit, a first streamed fit, and three subsequent streamed fits.
  Reported throughput uses the median of those three warm fits.
- Blocks of 64 or 256 features with one worker, and 256 with four workers.
  The reader/fitting working-buffer budget is 256 MiB.

Workloads use two equal runs, eight regressors, roughly 9% original-row exclusion,
Preserve covariance, global estimation and fixed orders 1–4. They cover 300 or
1200 original scans with 1000 features, plus 300 scans with 10000 features.
Every delivered coefficient and residual variance is compared to an allocating
dense fit at relative-plus-absolute tolerance 2e-10 before accepting a timing.
This checks execution equivalence, not an independent statistical oracle; earlier
R/covariance fixtures provide the latter. No new R speedup comparison is claimed.

Peak RSS comes from `wait4` child-process resource usage in macOS bytes. It
includes original/selected arrays, source copies, dense reference outputs, plan
storage and runtime overhead. It is **not** a streaming-only peak or proof of a
fixed-memory bound. `/usr/bin/time -l` was blocked from reading a kernel clock
setting in the sandbox; that failed instrumentation receipt is retained.
A single fresh-process preparation timing and three short warm repetitions are
local measurements, not a broad hardware or latency guarantee.

## Statistical stress contract

There are 60 fixed-seed replicates for each of five scenarios and four requested
orders (1,200 attempted fits). All use Preserve/global stationary estimation,
lag budget 12 and three design columns. Baseline uses 180 original scans, 16
noise features and moderate AR coefficients. Variations are:

- Persistence: true coefficient vector (0.95, 0, …), fitted at the requested
  order. Higher-order cases are intentionally over-specified persistent AR(1).
- Heavy censoring: keep only two scans in every five, leaving 72/180 scans.
- Short runs: two runs of 20 original scans each, retaining 36 in total.
- Few features: use only one representative feature to estimate noise.

Inference uses a separate held-out feature. Noise is initialized with 2048
burn-in steps per run. Coverage is conditional on successful estimation and uses
nominal 95% Student-t intervals. Rejections, their reasons, and the denominator
are reported; unsupported cases are not removed from the record. Finite Monte
Carlo uncertainty remains substantial at 60 replicates. These runs do not qualify
real datasets, arbitrary censor masks, or same-feature inference.

The heavy pattern provides no observed lag-two pairs, so the explicit lag-12
correction rejects all attempts for all orders. The API does not guess a smaller
budget or substitute IID. Short and persistent scenarios show undercoverage in
some configurations. Successful computation therefore must not be read as
calibrated inference. The existing finite-lag, stationarity and conditional
parameter-uncertainty limitations remain material.

## Receipts and remaining scope

`evidence/consolidation/verification.json` binds commands, source hashes, target
results, runtime, measurements and statistical outcomes. Target-by-target workspace
execution bounds build disk usage and retains failed artifacts; successful test
executables may be removed after execution. Ignored tests are listed separately.
Building an example under Cargo's test harness does not execute its `main`; the
complete walkthrough is explicitly run and recorded separately.

No automatic order selection, ARMA, zero-history estimation, per-run innovation
scale model or AR-estimation uncertainty correction is added. No publication,
remote migration, blanket upstream parity, real-data acceptance, or new universal
speed/memory claim follows from this checkpoint.

## Executed results

Default workspace: **910 passed, 17 ignored**, across 100 target batches.
All-feature workspace: **924 passed, 21 ignored**, across 102 target batches.
All ignored cases are doctests. No failed target was suppressed. The default
metadata inventory preceded addition of the measurement and complete-walkthrough
examples: the measurement example was checked separately with dataset enabled;
the walkthrough requires formula and was explicitly run and covered in the
all-feature inventory. Strict regression/noise library Clippy passes. The broader
legacy frontend lint debt is not claimed resolved by test success.

All 28 release workloads pass dense/streamed output equivalence; the maximum
reported relative coefficient/variance discrepancy was zero on this run. The
following rows use 1,000 features and block size 256. Preparation is a single
measurement; throughput is the median of three warm streamed fits.

| Original scans | AR order | Preparation, ms (one worker run) | Features/s, 1 worker | Features/s, 4 workers | Peak MiB, 1 / 4 workers |
| --- | --- | --- | --- | --- | --- |
| 300 | 1 | 0.75 | 405,104 | 620,684 | 13.4 / 37.9 |
| 300 | 2 | 0.87 | 291,882 | 552,448 | 13.0 / 38.6 |
| 300 | 3 | 0.82 | 231,844 | 470,155 | 16.6 / 40.4 |
| 300 | 4 | 0.92 | 190,951 | 497,595 | 15.3 / 38.9 |
| 1200 | 1 | 3.01 | 86,055 | 158,445 | 56.1 / 118.9 |
| 1200 | 2 | 3.31 | 63,430 | 140,436 | 68.4 / 119.3 |
| 1200 | 3 | 3.42 | 54,604 | 134,122 | 71.2 / 98.3 |
| 1200 | 4 | 3.49 | 50,620 | 128,948 | 70.8 / 93.5 |

Block-64 and 10,000-feature runs, all repetitions, first-fit times and process
peaks are retained in benchmarks.json. Four workers helped these workloads but
increased peak memory; the best block size varied. These results do not establish
a universal worker or block-size default.

Stress results below show successful fits out of 60 attempts and empirical
coverage conditional on success. All scenarios use independent inference targets.

| Scenario | AR1 success / coverage | AR2 success / coverage | AR3 success / coverage | AR4 success / coverage |
| --- | --- | --- | --- | --- |
| baseline | 60/60 / 98.3% | 60/60 / 91.7% | 60/60 / 96.7% | 60/60 / 93.3% |
| persistent | 60/60 / 93.3% | 60/60 / 86.7% | 60/60 / 95.0% | 60/60 / 86.7% |
| heavy_censor | 0/60 / undefined | 0/60 / undefined | 0/60 / undefined | 0/60 / undefined |
| short_runs | 60/60 / 91.7% | 60/60 / 90.0% | 60/60 / 98.3% | 60/60 / 86.7% |
| few_features | 60/60 / 95.0% | 60/60 / 91.7% | 60/60 / 95.0% | 60/60 / 91.7% |

The 240 heavy-censor rejections are retained results, not silently skipped fits.
For the successful scenarios, the lowest observed coverage is 86.7%, including
persistent AR2/AR4 and short-run AR4. At only 60 replicates, sampling uncertainty
is wide; Wilson intervals are supplied in the receipt rather than treating a
single coverage percentage as a precise estimate. No interval adjustment or
coefficient repair was introduced to make these results appear calibrated.
