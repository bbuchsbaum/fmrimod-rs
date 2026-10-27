# Local estimation measurements

Release-mode, single-process multiresponse OLS; five warmed repetitions, median
fit time. Both methods prepare the design outside the timed fit loop. R reuses
QR; Rust reuses an SVD-derived factorization. Both compute all coefficients and
RSS-based residual variances, without retaining fitted/residual matrices.

| Timepoints × regressors × features | Rust (ms) | R (ms) |
| --- | ---: | ---: |
| 300 × 8 × 1000 | 0.905 | 2.000 |
| 300 × 8 × 10000 | 7.249 | 20.000 |
| 1200 × 32 × 1000 | 4.974 | 33.000 |

Peak resident memory over the complete three-workload process was
90.9 MiB for Rust and 290.0 MiB for R.
This includes data generation, runtime and allocator behavior; it is not a
per-fit allocation bound. R used Accelerate BLAS, with OPENBLAS_NUM_THREADS,
OMP_NUM_THREADS and VECLIB_MAXIMUM_THREADS set to 1. Rust used ndarray's pure
Rust matrixmultiply backend without a parallel feature. System load was not
isolated. R elapsed timing has millisecond resolution on these short cases.

Coefficient and residual-variance checksums agreed within 1e-9 (observed errors
around 1e-12 or smaller). Checksums are a benchmark guard, not a full numerical
oracle: the R contract tests separately compare complete outputs, t/F statistics,
whitening, and dense-covariance GLS on small cases. These measurements do not
establish speedups for AR fitting, I/O, full formula workflows, other hardware,
or arbitrary dimensions. No automatic performance gate is derived from them.

Raw receipts: evidence/rust-throughput-memory.json, evidence/r-throughput.json,
evidence/rust-memory.log, evidence/r-memory.log. Reproduce with
`cargo run --release -p fmrireg --example throughput` and
`scripts/benchmark_estimation.R`; use `/usr/bin/time -l` for process RSS.


## AR-GLS qualification (2026-10-26 local session)

The complete measurements are in [GLS_RESULTS.md](GLS_RESULTS.md). On this
Apple M3 Max, final single-worker AR(4) compute for 50,000 features took
92–382 ms across the four tested timepoint/regressor combinations, compared
with 262–2,306 ms in R: observed ratios of 2.9–6.0. Compute includes whitening,
coefficients, residual variances and one t contrast with p values. Generation
and one-time preparation are reported separately, not hidden inside a speedup.

### Workload and equivalence

The matrix is the Cartesian product of 300/1,200 timepoints, 8/32 regressors,
1,000/50,000 features and AR orders 1/4. Both implementations use f64, two runs,
three noise-censor rows (including consecutive exclusions), zero-history startup,
run pooling, correction lag 10 and 32 representative features for noise estimation.
Noise masks restart filtering and exclude lag pairs; regression retains all rows.
Both prepare the design once and reuse it for all feature blocks. The R baseline
uses the source-built fmriAR package, prepared QR and explicit residual variance;
it does not re-fit the noise model or design per block. No formula or file I/O
cost is included. The synthetic harmonic data exercise coefficient repair but
are not a study of noise-estimator accuracy on real fMRI.

All coefficients, variances, effects, standard errors, t statistics and p values
are compared, not just checksums. All 16 final R comparisons passed the existing
GLS bound `abs(Rust-R)/(1+abs(R)) < 2e-8`; the largest observed discrepancy was
9.1e-10. AR coefficients, rank and residual df also agree. Eight additional
block/worker configurations agree with the canonical Rust result. The
200,000-feature result has a bit-identical first 50,000-feature prefix.
Current R source and compiled-library hashes are verified against the oracle
fixture; local reference checkouts showed no drift.

### Timing and memory interpretation

Each configuration runs in a fresh process, with one first block pass followed
by three warmed repetitions. Full-output validation is another untimed pass.
Preparation is one cold observation per process; do not treat its ratios as
precise warm microbenchmarks. R's elapsed timer has approximately 1 ms resolution.
The raw records retain each phase/repetition and first-pass timing. Processes run
serially. CPU load and frequency are not isolated; these are local observations,
not confidence intervals or a universal speed claim.

R uses Accelerate BLAS with VECLIB_MAXIMUM_THREADS, OPENBLAS_NUM_THREADS and
OMP_NUM_THREADS set to 1; fmriAR filtering sets parallel=FALSE. Rust uses ndarray's
matrixmultiply backend without BLAS or internal parallelism. Scoped benchmark
workers share immutable preparation and process disjoint feature blocks. Thread
creation is included in wall time. This is an explicit caller-side strategy,
not a new parallel dataset-reader API. Phase sums across workers are aggregate
worker time, never presented as parallel wall time.

The best measured 50,000-feature configuration was block 128 with four workers:
283 ms including synthetic generation, with 44.7 MiB process peak RSS. Block
4,096/four workers used 581 MiB and took 321 ms. Treat this as guidance for this
workload; benchmark a consumer's own data source and sink before fixing defaults.
At block 1,000/one worker, increasing features from 50,000 to 200,000 changed peak
RSS from 61.8 to 62.2 MiB, while wall time grew from 1.08 to 4.36 s. This supports
bounded streaming for these workloads. It does not establish a universal process
memory cap. RSS includes runtime, allocation caching, model preparation (including
R's dense correction), repetitions and output validation. No full Y or complete
fit is retained; synthetic blocks feed an incremental sink. Storage/backend I/O
and consumer-retained outputs require their own budgets.

### Optimization and reproduction

The measured fit bottleneck motivated one kernel change: accumulate squared
residuals across contiguous rows instead of strided columns. Explicit residuals
are retained to avoid cancellation near perfect fits. For the four large AR(4)
cases, fit-phase ratios were 1.19, 0.98, 1.33 and 1.21; the small apparent regression
in the 300×32 case is retained in the report. Large-block fit phases improved
more. Independent R contracts and an analytic large-offset/small-residual test
pass, including extreme feature scales and block equivalence.

```sh
cargo build --release -p fmrireg --example gls_benchmark
python3 scripts/benchmark_gls.py --label baseline
python3 scripts/benchmark_gls.py --label final --rust-only
python3 scripts/benchmark_gls.py --label memory --memory-only
python3 scripts/verify_gls_benchmarks.py
python3 scripts/report_gls.py
```

The runner requires Python NumPy and macOS `/usr/bin/time -l` resource-statistic
access. R requires the isolated installation in `evidence/r-library`. Avoid
reusing evidence labels when preserving an earlier comparison. The recorded
pre-optimization source is `evidence/gls-baseline/fmrireg-lib.rs`; rerunning the
commands on current sources measures current performance, not that historical
baseline. Metadata records source/executable hashes, hardware, compiler and
thread settings. Binary outputs are ignored by Git but retained locally for
full comparison; results, hashes and source snapshots are portable receipts.
