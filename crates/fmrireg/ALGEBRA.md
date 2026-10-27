# Algebraic fitting experiments

This round retained **no new production optimization**. The existing two-product
fit remains the fastest qualified implementation among these candidates. This
is a result for the measured workloads and machine, not a universal optimality
claim. The prior accepted improvements remain described in `HILL_CLIMB.md`.

## Decisions

| Candidate | Final warm timing versus current fit | Decision |
|---|---|---|
| Cache `(WX)+ W`; fit original data, whiten original-space residuals and accumulate RSS | 1.05–1.28× speedup over eight GLS workloads | Reject unrestricted use: native variance-accuracy regressions |
| Keep `WY` and coefficients; overwrite `WY` with `WY - WX beta` using GEMM | 1.00–1.07× speedup over the same workloads | Reject unrestricted use: exact residual-kernel cancellation |
| Implicit Householder QR using LAPACK `DORMQR` | 2.85–13.30× slower over 16 OLS workloads | Reject for throughput |
| Force block Householder application with compact WY (`DLARFT`/`DLARFB`) | 1.12–1.81× slower over the same OLS workloads | Reject for throughput |

The GLS microbenchmark uses known zero-history AR(4), two runs and four reset
segments. It compares coefficients and residual variances, with all entries
checked against the current implementation at `2e-8 * (1 + abs(reference))`.
The ordinary-data errors are much smaller than this tolerance. The two GLS
candidates reduce their explicit numeric fit buffers from `(2n+p+1)*B*8` to
`(n+p+1)*B*8` bytes. These are workspace counts, not measured peak RSS, and exclude
prepared operators, backend scratch and input data.

The QR probes use full-rank OLS only. They include input validation, copying,
implicit transformation, triangular solve, variance accumulation and output
validation. Row-major responses are viewed as their column-major transpose so
no layout conversion is required. DORMQR uses queried LAPACK workspace; the
compact WY path uses a `B × p` work buffer and forces matrix-matrix reflector application even for small
designs. Coefficients and variances are checked entry by entry against the
current solver. An independent exactly representable large-offset problem has
known coefficients and RSS `0.625`. Its measured relative variance error is
about `1e-10`; the recorded scale-aware acceptance allowance is a heuristic,
not a certified forward-error bound.

## Numerical rejection evidence

### Cached GLS map

The identity `D (WY) = (D W) Y`, where `D=(WX)+`, holds in exact arithmetic.
Likewise `WY - WX beta = W(Y - X beta)`. Their floating-point reassociations
need not preserve small residuals.

`scripts/qualify_algebra.py` independently solves small full-rank GLS problems
using Python Decimal at 90-digit arithmetic precision. It converts the supplied
binary64 inputs exactly and independently constructs AR(1), including stationary
startup and resets. High-precision normal equations are used only for this
three-column oracle; they are not a proposed production solver. The precision
setting does not assert 90 certified correct digits.

The grid has 108 cases spanning nearly dependent columns, large offsets,
positive/negative AR parameters, both startup policies, and segments as short
as one sample. Ninety cases retain full rank in the NumPy stress calculation;
rank-truncated cases are explicitly excluded from the full-rank oracle claim.
Native Rust probes also exercise both matrix backends, use the same SVD
construction as production, and compose the map through the sparse AR adjoint.

A decisive native case has 32 observations, three retained columns, column
separation `2^-20`, response offset `2^20`, AR coefficient `0.999999` and
zero-history startup:

| Result | Residual variance | Relative error against Decimal |
|---|---:|---:|
| Independent reference | 0.00840820991236256 | — |
| Current Accelerate fit | 0.008408209936515997 | 2.87e-9 |
| Composed candidate | 0.008408213373854029 | 4.12e-7 |

The adverse-case acceptance threshold is **relative variance error `2e-8`**,
separate from the ordinary benchmark's scaled absolute tolerance. The portable
backend also has cases where the current fit meets this threshold and the
candidate fails. This native witness evaluates the two reassociations together;
the NumPy probe additionally separates them. Some more extreme inputs defeat
both paths; those adverse results are retained rather than claiming universal
accuracy for the current solver.

A finite-result check does not detect these errors. A future automatic path
would need a defensible cancellation/error estimate and fallback, or a narrower
explicit contract. No such guard or opt-in API was added in this round.

### In-place residual GEMM

The exact kernel witness uses `A` filled with `2^50`, coefficient rows `+1` for
the first half and `-1` for the second, and observed values `1`. The exact matrix
product is zero, so every residual should be one. Separate multiplication and
subtraction preserve this result. `C <- C - AB` loses it when internal partial
products erase `C` before cancelling:

- Accelerate fails at inner dimension 32 and feature width 128.
- Portable fails at inner dimension 512 for all three tested feature widths.

The observed residual is zero instead of one. This is a residual-kernel
counterexample, not an end-to-end fitted-model counterexample: its coefficient
matrix is supplied directly. It disproves unrestricted kernel equivalence and
rules out promoting this shortcut based only on ordinary-data fit agreement.

## Reproduction and evidence

Final timings used Apple M3 Max, macOS 14.3, Rust 1.91.1, release builds, one
application worker, and `VECLIB_MAXIMUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=1`.
Warm batches exclude data generation and model preparation. GLS uses mirrored
three-method order; QR uses ABBA. Input dimensions are `n=300/1200`, `p=8/32`,
and feature blocks `32/128/1000/4096`, with the exact subsets in each receipt.
The machine was not isolated from external load or frequency changes. These are
prepared-kernel timings, not new whole-pipeline, I/O or R throughput claims.

```sh
CARGO_TARGET_DIR=/private/tmp/fmrimod-fit-speed-target cargo build -p fmrireg \
  --release --features accelerate,dataset --example algebra_gls --example algebra_qr
VECLIB_MAXIMUM_THREADS=1 /private/tmp/fmrimod-fit-speed-target/release/examples/algebra_gls
VECLIB_MAXIMUM_THREADS=1 /private/tmp/fmrimod-fit-speed-target/release/examples/algebra_qr
FMRIREG_QR_BLOCKED=1 VECLIB_MAXIMUM_THREADS=1 \
  /private/tmp/fmrimod-fit-speed-target/release/examples/algebra_qr
```

The GLS example also accepts `--stress-product` and `--stress-composition`.
Set `FMRIREG_BACKEND=portable` to exercise the portable kernels. The oracle script
reads the native composition receipts when present and otherwise reports its
independent NumPy/Decimal comparison.

`evidence/algebra/` contains:

- `decisions.json`: per-shape timing medians, ratios and decisions.
- `metadata.json`, `build-inputs.json`, `source/` and the two executables: source
  and binary bindings, environment, commands and limitations. Relevant build
  inputs were hashed before and after the final build and were unchanged.
- `prototype-accelerate.json`, `qr-accelerate.json` and
  `qr-blocked-accelerate.json`: final raw warm samples.
- `composition-*.json`, `inplace-cancellation-*.json`, and
  `reassociation-stress.json`: native and independent numerical witnesses.
- `pilot/`: earlier exploratory receipts. Their older example revisions were
  not individually snapshotted; they are not bound to the final source hashes
  and do not supply the headline timings.

Validation: 30 current `fmrireg` tests passed with all features; strict Clippy
passed for the crate's examples. Both new examples also compile with no default
features, and their formatting checks pass. No additional production R fixture
generation or full-workspace qualification was required because no candidate
was integrated. Concurrent observation-selection changes belong to a separate
workspace task and were preserved.
