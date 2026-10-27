# Further fitting performance hill climb

Two additional optimizations are retained: vectorizable finite-value validation
and fused feature-row kernels for AR orders 1–4. A residual-tiling candidate was
rejected. The baseline is the immediately preceding Accelerate/workspace/parallel
implementation, not the original unoptimized fitter.

## Measured streaming GLS gains

Apple M3 Max, f64, release build, Accelerate with BLAS thread limits set to one.
Fresh processes, first pass recorded separately, five warmed repetitions.
Each row has 50,000 features, block size 1,000, one worker, two runs and AR(4).
Compute includes whitening, fitting and one t contrast with p values. Preparation
and synthetic generation are excluded from compute and recorded separately.

| Timepoints | Regressors | Whitening before → after (ms) | Fitting before → after (ms) | Compute before → after (ms) | Compute speedup |
|---:|---:|---:|---:|---:|---:|
| 300 | 8 | 31.00 → 8.77 | 12.56 → 7.94 | 50.82 → 23.52 | 2.16× |
| 300 | 32 | 30.71 → 9.09 | 16.31 → 12.43 | 58.59 → 33.08 | 1.77× |
| 1,200 | 8 | 129.73 → 44.13 | 50.74 → 38.06 | 193.21 → 94.07 | 2.05× |
| 1,200 | 32 | 127.07 → 44.01 | 61.41 → 50.84 | 199.78 → 106.32 | 1.88× |

The largest case's wall time, including expensive synthetic generation, changes
from 877 to 784 ms. The compute gain must not be represented as that entire
pipeline's wall-time gain. System load and CPU frequency were not isolated.

## Retained and rejected trials

The kernel probes exclude input generation and preparation and include all
fitting/whitening validation. They cover eight shapes: 300/1,200 timepoints,
8/32 regressors, and blocks from 32 to 4,096 features. Every candidate was
compared in A/B/B/A process order, with seven warmed batches per phase in each
process. Full raw timings and ranges are preserved, not just selected medians.
The checksum guard in these probes is supplemented by the full-output reference
qualification below.

| Candidate | Observed complete GLS kernel speedup | Decision |
|---|---:|---|
| Chunked, vectorizable finite validation | 1.17–1.31× across eight shapes | Retain |
| Fused AR(1–4) row kernels, beyond validation improvement | 1.46–2.01× across eight AR(4) shapes | Retain |
| 128-row residual tiles for large matrices | 0.895–1.039× across active cases | Reject |

Tiling slowed the large eight-regressor case by about 10%; other gains were
small or within local timing variation. Its source and executable are retained
under `evidence/hill-climb/tile`; production fitting was restored byte-for-byte
to the preceding candidate. There is no new tiling policy or workspace shape.

Additional A/B/B/A probes of the final candidate against the preceding delivery:
AR(1) gains 1.56–2.21×; AR(8), which keeps the generic lag kernel, gains
1.13–1.18×; portable-backend AR(4) gains 1.34–1.85×. These are Mac CPU
observations, not measurements of Linux/Windows or a claim of universal optimality.

## Production executor and memory

The public executor uses the same serial synthetic reader and ordered sink in
both versions; known AR(4), 1,200 timepoints, 32 regressors, 50,000 features,
block 1,000. Wall time includes reader generation, thread launch, coefficients
and residual variances; this executor benchmark does not compute t contrasts.

| Workers | Previous implementation (ms) | Retained candidate (ms) | Speedup |
|---:|---:|---:|---:|
| 1 | 233.87 | 134.68 | 1.74× |
| 4 | 70.54 | 50.52 | 1.40× |

With four workers and fixed block 1,000, process peak RSS is 125.7 MiB for
50,000 features and 126.6 MiB for 200,000. Explicit numeric workspace sizes
are unchanged. RSS still includes backend/runtime/allocator behavior and is
not the API's numeric-buffer budget. No complete source or output volume is
retained in this measurement.

## Numerical and implementation boundaries

Finite validation still checks every selected value and rejects NaN/infinity.
Contiguous storage uses small boolean reductions; arbitrary strided and broadcast
views keep the logical-iterator path. Invalid data is rejected before output
allocation in the allocating APIs. No validation is removed or made optional.

The fused AR path traverses each feature row once, maintaining ascending-lag
accumulation from zero followed by subtraction. Stationary startup, run/censor
resets, mixed run coefficients, overflow detection, arbitrary strides, and
higher-order generic filtering retain their existing contracts. No f32,
fast-math, explicit fused multiply-add, or alternative noise model was introduced.
SVD rank policy, residual degrees of freedom and contrasts are unchanged.

All 16 complete reusable GLS workloads agree with the source-built R reference
within the existing scaled tolerance `2e-8`; the largest error remains
`9.0712e-10`. Every coefficient, variance, effect, standard error, t statistic
and p value is compared. The R source and compiled-library hashes are verified.
Executor outputs also agree across all measured block/worker configurations.

New regressions cover every invalid-value position across chunk boundaries,
finite extremes/subnormals, both memory orders, excluded strided elements,
broadcasts, and AR orders 0–8 with short and long reset segments. Reversed input
and output views cover full AR4 and the stationary startup transition. A
mixed-scale/signed-zero case checks bitwise agreement with a scalar reference
for orders 0–4; this is a bounded test, not a universal bitwise-equivalence claim.
Independent source review identified a short-segment coverage gap, which was
fixed; no review findings remain.

## Executed checks

- Focused all-feature tests: **37 passed, none ignored**.
- Minimal portable fitting tests: **14 passed, none ignored**.
- Full workspace commands: **891 passed/17 ignored** with default features;
  **905 passed/21 ignored** with all features. Ignored cases are existing
  HRF/design doctests. These commands also see concurrent design work in the
  shared checkout; the retained fitting/noise sources are hash-bound separately.
- Strict Clippy and formatting checks for both changed crates pass.
- Workspace-wide formatting reports only unrelated `sampling`/`sampled_features`
  module/prelude ordering in `crates/fmri-design/src/lib.rs`. That file was not
  changed by this hill climb. The failed global receipt is preserved alongside
  the successful scoped check.
- Final benchmark executables and all recorded fitting/noise source hashes
  were rechecked after qualification.

## Reproduction and receipts

From the workspace root:

```sh
cargo test -p fmri-reg -p fmri-noise --all-features --locked
cargo test -p fmri-reg --no-default-features --locked
cargo clippy -p fmri-reg -p fmri-noise --all-features --all-targets --locked -- -D warnings
cargo build --release -p fmri-reg --all-features --examples --locked
python3 crates/fmri-reg/scripts/qualify_hill.py --target "$PWD/target" --label new-run
```

The qualifier requires NumPy, macOS resource-statistic access, the locally
preserved baseline executables and recorded R binaries. Use a fresh label;
it refuses to overwrite an existing run. `scripts/hill_compare.py` records
A/B/B/A kernel trials; its default target directory is the isolated local target
used during this session. Source and executable hashes bind the evidence.

Receipts: `evidence/hill-climb/decisions.json`, each candidate's `summary.json`
and source snapshot, `qualified/results.json`, `qualified/metadata.json`, and
`verification.json`. Earlier results are retained in `PERFORMANCE.md` and
`evidence/speed` as historical measurements.
