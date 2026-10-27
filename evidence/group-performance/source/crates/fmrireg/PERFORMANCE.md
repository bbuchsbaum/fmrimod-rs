# Fitting performance qualification

The optional Accelerate backend, reusable fit/whitening buffers, and ordered
parallel dataset executor are implemented. This qualification measures specific
CPU workloads; it does not establish a universal optimum or GPU performance.

## Single-worker AR-GLS

Apple M3 Max, f64, release build (`opt-level=3`, thin LTO, one codegen unit),
BLAS thread limits set to one. Each configuration ran in a fresh process with
one first pass and five warmed repetitions. Medians below exclude synthetic
input generation and preparation. Compute includes whitening, coefficients,
explicit residual variance, and one t contrast with p values.

Before is the preserved executable from the audit. After enables `accelerate`
and reuses whitening and fitting buffers. Both use blocks of 1,000 features,
50,000 total features, two runs, fixed AR(4), the same censor resets, and the
same design-corrected noise estimator prepared from 32 representative features.

| Timepoints | Regressors | Fit before → after (ms) | Whitening before → after (ms) | Compute before → after (ms) |
|---:|---:|---:|---:|---:|
| 300 | 8 | 34.32 → 10.50 | 46.87 → 29.18 | 87.82 → 46.46 |
| 300 | 32 | 67.66 → 15.88 | 48.73 → 30.60 | 128.40 → 58.32 |
| 1,200 | 8 | 105.30 → 47.82 | 176.51 → 127.48 | 293.61 → 187.21 |
| 1,200 | 32 | 215.68 → 60.89 | 176.98 → 126.55 | 403.58 → 198.49 |

Observed fitting improvements are 2.20–4.26×; whitening improves 1.38–1.61×;
complete compute improves 1.57–2.20×. Phase medians need not sum to the median
of each repetition's compute total. Synthetic generation still dominates the
largest phase benchmark: its full wall time improves from 1.083 s to 0.884 s.

The raw results retain intermediate portable, Accelerate allocating, and
Accelerate reusable variants. Buffer reuse is not always a large timing win:
on the 1,200 × 8 case, allocating and reusable Accelerate wall medians are
866 and 868 ms. These observations are retained rather than suppressed.
System load and CPU frequency were not isolated.

All 16 final reusable workloads (300/1,200 timepoints × 8/32 regressors ×
1,000/50,000 features × AR(1)/AR(4)) compare every coefficient, variance,
effect, standard error, t statistic, and p value against the recorded
source-built R reference, using the existing scaled bound
`abs(Rust-R)/(1+abs(R)) < 2e-8`. The R source and compiled installation hashes
were checked before running. See `evidence/speed/qualified-1` for per-process
records, full-output hashes, commands, hardware and source fingerprints.
The reference suite also covers rank-deficient and ill-conditioned designs,
independent dense covariance GLS, and analytic small residuals on large offsets.

## Production executor and process memory

The public `fit_blocks_with_options` API was measured on known AR(4) GLS with
1,200 timepoints, 32 regressors, 50,000 features, and the same serial synthetic
reader and ordered sink for every configuration. Wall time includes input
generation, validation, fitting, thread launch and sink delivery; coefficients
and residual variances are produced, without t contrasts in this benchmark.

| Block features | Workers | Wall median (ms) | Peak RSS (MiB) |
|---:|---:|---:|---:|
| 1,000 | 1 | 231.48 | 37.6 |
| 1,000 | 2 | 122.31 | 67.0 |
| 1,000 | 4 | 73.19 | 123.4 |
| 128 | 1 | 210.50 | 34.8 |
| 128 | 4 | 70.69 | 92.7 |
| 4,096 | 4 | 110.22 | 523.2 |

At block 1,000, four workers improve wall time by 3.16× over one worker. The
legacy allocating serial adapter takes 239.50 ms on this same workload, versus
231.48 ms for the reusable serial path. Every coefficient and variance agrees
across the tested block/worker configurations within the recorded tolerance.

With four workers and block 1,000, increasing total features from 50,000 to
200,000 changes peak RSS from 123.4 to 123.9 MiB. This supports bounded streaming
for this reader and sink; it is not a universal RSS cap. Sources and results
are streamed and no complete volume/output is retained. The 1,000-feature case
uses only one active worker and is not an equal-worker memory comparison.

Final executor records are in `evidence/speed/executor-qualified`. Earlier
executor timings in `qualified-1` are retained as diagnostic evidence only:
that synthetic reader grew a vector while declaring only final output bytes
as its peak. The corrected reader reserves its exact declared output capacity;
the final measurements above use that implementation.

## Correctness and review

- Focused all-feature tests: 33 passed, none ignored. Portable minimal tests:
  13 passed, none ignored. Strict Clippy on both changed crates/all targets passes.
- Full workspace: default features 880 passed/17 ignored; all features
  894 passed/21 ignored. The ignored cases are existing HRF/design doctests,
  not skipped numerical tests for this change.
- New checks cover analytic variance at large offsets, independent GLS
  precision quadratic forms, strided layouts, buffer reuse, overflow and
  invalid inputs, retained result lifetimes, exact aggregate budget rejection,
  non-Send readers, ordered sink delivery, and returned-error worker shutdown.
- An instrumented allocator confirms that a large broadcast NaN input is
  rejected before allocating fitting buffers. Independent source review found
  and resolved that preallocation regression and a callback lifetime issue;
  no findings remain in the reviewed scope.

Commands, exit statuses, counts and source fingerprints are in
`evidence/speed/verification.json`. Local results do not establish Linux/Windows
runtime qualification or provide a general process RSS guarantee.

## Implementation and use

- Enable `accelerate` on macOS to select Accelerate by default; select
  `MatrixBackend::Portable` explicitly for comparison. Other targets retain
  the portable backend. Noncontiguous inputs use the portable fallback.
- Use `model.workspace(max_features)` and `model.fit_into(data, &mut workspace)`
  to reuse explicit buffers. Fit results borrow the workspace; `to_owned()`
  retains compact outputs when needed. Backend packing scratch is separate.
- Built-in AR plans implement `apply_into` with row-wise feature traversal.
  Lag accumulation order, run boundaries, resets and startup policies remain.
- Use `dataset::fit_blocks_with_options` or `BoundModel::fit_assay_with_options`
  with an explicit worker count and aggregate numeric memory budget. Reads and
  sinks stay serial and ordered; worker fits run concurrently. See README.md
  for exact memory accounting and custom-plan exclusions.

## Reproduction

From the workspace root, run focused checks and build the measurement programs:

```sh
cargo test -p fmrireg --no-default-features --locked
cargo test -p fmrireg -p fmri-noise --all-features --locked
cargo clippy -p fmrireg -p fmri-noise --all-features --all-targets --locked -- -D warnings
cargo build --release -p fmrireg --all-features --examples --locked
python3 crates/fmrireg/scripts/qualify_speed.py --target "$PWD/target" --label new-run
```

Use a fresh label to preserve previous evidence. The runner requires NumPy,
macOS resource-statistic access, the recorded R output binaries, and the
source-built reference installation. It runs processes serially and fixes
BLAS thread limits. `--executor-only` repeats just the production executor
measurements. The preserved baseline executable is local/ignored; rebuilding
that baseline requires the recorded source snapshot and dependency lockfile.
