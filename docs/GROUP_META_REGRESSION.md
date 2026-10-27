# Weighted meta-regression

Mote: `bd-01M5YYH13D02EYKG9ZVJFN7NDE`. The five deliverables are:
FE/DL regression, analytic-derivative PM, selectable normal/mKH inference,
requested outputs, and independent numerical/performance qualification.

## Numerical contract

One independent subject effect and genuine positive sampling variance per row.
Rows and contrasts bind to `GroupDesign`; missingness/minimum n are explicit.
This API requires full column rank and n>p on the retained rows. It does not
silently drop predictors. Legacy rank-aware group OLS and scalar FE/DL remain.

Use column-scaled, column-pivoted QR of sqrt(W)X; numerical rank is based on
relative R-diagonal magnitudes after scaling. Weights are feature-dependent.
Reuse retained-row buffers and design metadata, never a factorization across unequal weights.
Center responses when a nonzero constant design column permits it. Project explicit
residuals into the QR residual space; never subtract large squared norms. Contrast covariance
comes from triangular solves and Gram products, without normal-equation inversion.

FE fixes tau²=0. DL uses max(0,(Q(0)-(n-p))/tr(P)), where
P=W-WX(X'WX)^-1X'W. A QR-complement computation provides a positive denominator
fallback when leverage subtraction is ill-conditioned. No variance floor.

PM solves Q(t)=n-p, t=tau²>=0. Boundary Q(0)<=n-p returns zero. The analytic
derivative is Q'(t)=-sum(r_i(t)²/(v_i+t)²). Use a bracket and safeguarded Newton
steps with bisection fallback, a dimensionless Q tolerance, and a finite evaluation
limit. Nonconvergence yields diagnostics and no usable estimates. Initial/final
Q, tau², bracket, evaluation/step counts and rank diagnostics are observable.

Normal inference uses plug-in covariance. Modified Knapp–Hartung multiplies it by
max(1,Q(t)/(n-p)) and uses t(n-p). These policies are independent of tau estimation.
Common-effect plus mKH is allowed as an explicit policy combination; it is not
conventional common-effect inference. Metafor warns about this combination. The
default remains common-effect plus normal inference. The `adhoc` scaling and
reference distributions match [metafor's documented policies](https://wviechtb.github.io/metafor/reference/rma.uni.html).
Sampling variances are treated as fixed for fitting/inference; their first-level
provenance is retained. This is not a parameter-uncertainty or general coverage
guarantee. Metafor oracle settings: FE/DL/PM, test="z" or test="adhoc".

Effects-only requests omit contrast covariance and distribution evaluation.
Marginal uncertainty returns requested SEs; joint uncertainty additionally returns
their covariance. P-values/statistics and confidence intervals are explicit optional
requests, with validation rather than NaN placeholders. Joint output size is
included in the numeric preflight budget. Existing scalar result types stay valid.

### Numerical failure boundary

Weights are normalized by the smallest total variance, and predictor columns are
scaled before QR. At extreme weight spread or poor QR diagonal ratio, two residual
refinement steps use the original weighted design. Refined residual-basis vectors
also repair the derivative and the DL complement denominator without an n-by-n
projector. An unresolved derivative disables Newton and uses bisection. An unstable
DL denominator or failed residual-norm consistency check returns `NumericalFailure`.

Normalized weights below `f64::MIN_POSITIVE` are rejected, including ratios that
underflow to zero. This is an arithmetic support boundary, not a variance floor:
observations are neither removed nor reweighted to force a fit. Uniformly small
variances can still be supported. Very ill-conditioned inputs outside the recorded
witnesses have no general forward-accuracy guarantee.

### Using the API

Create a `MetaOutputSpec` from named `GroupContrast`s bound to a `GroupDesign`.
`MetaOutputSpec::new` requests effects only. Set `uncertainty` to `Marginal` or
`Joint`, then request p-values and/or a confidence level as needed. Joint output
is covariance between group contrasts of the one first-level estimand; it does
not model several correlated first-level estimands per subject.

`PreparedMetaRegression::new` validates and retains design metadata;
`fit(&cohort, sink)` rereads the data on every call and delivers bounded feature
blocks. Check each feature's `status` before using estimates. Missingness, retained
rank loss, insufficient observations and nonconvergence are explicit outcomes.
The one-shot `fit_meta_regression` wrapper uses the same implementation.

Run the [complete example](../crates/fmri-group/examples/weighted_group.rs):

```sh
cargo run -p fmri-group --example weighted_group
```

It generates first-level contrasts, admits independent subjects, and requests PM
with mKH intervals and joint covariance for an adjusted mean and a covariate slope.
This is a usage example with synthetic data, not a scientific calibration exercise.

The general prepared path reuses numeric buffers, but nalgebra still allocates each
weighted QR. The later [portable execution tranche](GROUP_PORTABLE_EXECUTION.md)
adds an explicit shared-weight batch path for eligible common-effect fits.
Feature blocking bounds fit workspace, not caller-owned inputs, sink-retained
outputs or total process RSS. `numeric_working_bytes` is a conservative preflight
estimate; the inherited pattern-cache allowance is reserved even though this API
does not cache weighted factorizations. `weighted_evaluations` counts attempts;
`run.factorizations` counts successful weighted fits, excluding preparation's
design-validation factorization.

## Acceptance and execution

1. Fresh metafor fixtures with matched options and a dense QR/uniroot witness;
   fmrigds compiled FE/DL regression comparisons where estimands match. Record
   package versions/source identity and source hashes.
2. Covariate, contrast, covariance, p/CI, missingness and tile/block parity; zero
   heterogeneity, rank loss, rescaling/offset/row-order, extreme weights, degenerate
   responses, solver exhaustion, budget and sink failures. Independent derivative
   finite differences and analytic equal-variance PM cases.
3. Bounded simulations with seed, model, coverage, rejection rates and Monte Carlo
   uncertainty; retain adverse outcomes. No universal calibration claim.
4. Equivalent-output release benchmarks with separate preparation/first/warm times, iterations,
   allocations and output profiles. Compare real compiled fmrigds paths and fresh
   metafor fits; report any slower workloads. No builds overlap timed Rust trials.
5. Affected group/facade tests, doctests, scoped clippy and formatting. Preserve
   concurrent/unrelated work. No publication is authorized by this tranche.

Portable interchange, repeated-measures likelihoods and permutation inference
remain separate stages. Evidence: `evidence/group-meta-regression/`.

## Results

Numerical fixtures compare all three estimators, both inference choices, named
contrasts, their covariance, SEs, statistics, p-values and intervals. Measurement
scales are 1, 1e-8 and 1e8; tests undo the units before comparing at a scaled
tolerance of 2e-9. Missingness and three block sizes are included. Fresh metafor
5.0.1 references are cross-checked with independent base-R QR/uniroot and freshly
compiled fmrigds FE/DL regression (`eps=0`). The fmrigds tracked-source snapshot is
`31e26ba5bd0bc088c71395077be1a13765512dab`, with per-file hashes and isolated
installation retained. R 4.5.1 reports that metafor/metadat were built under R
4.5.2; those warnings are retained and the executed comparisons pass.

Analytic tests cover equal-variance PM/DL, an offset of 2^40, all 24 row permutations
of an extreme-weight problem, unequal pair variances, reversed/rescaled predictor
columns, independent derivative finite differences, and near-zero versus large
contrast statistics. They also verify rank/missingness failures, solver exhaustion,
request omissions, binding, budget and sink errors. The numerical development
counterexamples and rejection boundary are retained in
[numerical-repairs.md](../evidence/group-meta-regression/numerical-repairs.md).

The reproducible benchmark and simulation results are below. Machine-readable
source hashes, binary hashes, actual commands, exit codes and comparison records
are in [verification.json](../evidence/group-meta-regression/verification.json).

### Executed checks

All 51 tests passed across the group crate and facade with all features and targets. Scoped clippy with warnings denied, formatting, and the runnable example passed. Doctest targets completed successfully but currently contain zero examples. These are local macOS/arm64 results, not hosted or cross-platform qualification.

### Runtime and allocation measurements

Single-thread release medians over five trials on macOS 14.3 arm64, Rust 1.91.1. Both paths return coefficients, marginal SEs, Q, tau², n and df. Rust includes cohort reads and output collection; the FE/DL reference calls compiled fmrigds directly. PM calls `metafor::rma.uni` per feature, including its R-level model setup. Preparation and first-fit times are recorded separately; these are not cold-process or cold-filesystem measurements.

| Subjects × features × predictors | Method | Rust (ms) | R (ms) | R / Rust |
|---|---|---:|---:|---:|
| 20 × 200 × 3 | PM | 1.024 | 955.000 | 932.62 |
| 20 × 1000 × 3 | FE | 1.298 | 1.020 | 0.79 |
| 20 × 1000 × 3 | DL | 3.027 | 3.400 | 1.12 |
| 100 × 200 × 5 | PM | 7.401 | 2797.000 | 377.93 |
| 100 × 10000 × 5 | FE | 58.040 | 59.333 | 1.02 |
| 100 × 10000 × 5 | DL | 112.691 | 182.000 | 1.62 |

DL is faster on these matched workloads. Common-effect regression is slower on small workloads and approximately tied at 100 × 10,000 × 5. The PM ratios describe these complete execution paths, not a language-only comparison or an isolated Newton-versus-bisection speedup. All raw trials are retained. The earlier passing qualification is in `qualification-initial/`; a source guard detected concurrent producer edits, so the final checks and benchmark were refreshed. Timing varies between trials and runs. This was a shared workstation, not an isolated performance host. The runner did not overlap its own builds with timing.

At 100 × 10,000 × 5, effects-only FE took 53.56 ms versus 58.04 ms with marginal uncertainty; effects-only DL took 107.61 ms versus 112.69 ms. Omitting outputs reduces work and allocation, but does not guarantee a measurable speedup when weighted fitting dominates. Marginal FE requested 291,143 allocations / 145.8 MB cumulatively; DL requested 440,503 / 292.9 MB. Those totals include output collection and repeated QR allocations, not peak live memory. The conservative numeric workspace budget was 2,081,760 bytes for these workloads.

PM needed 998 weighted evaluations over 200 features at n=20, and 1,076 at n=100: means 4.99 and 5.38, including boundary features. Newton/bisection counts were 581/33 and 668/14 respectively. These fixtures establish convergence and actual iteration counts; they are not a worst-case bound.

### Finite-sample calibration

The prespecified seed was 271027. Four panels contain 600 independent Gaussian datasets each, with known heterogeneous sampling variances, cosine covariates, and a zero target coefficient. Every rejection and coverage count across 12,000 fits matches metafor. The largest scaled numerical discrepancy was 3.34e-11 (tolerance 2e-8). Nominal interval coverage is 95%:

| n / predictors / true tau² | FE normal | DL normal | DL mKH | PM normal | PM mKH |
|---|---:|---:|---:|---:|---:|
| 8 / 2 / 0 | 94.00% | 95.50% | 99.17% | 95.50% | 99.17% |
| 8 / 2 / 0.5 | 77.17% | 91.67% | 96.83% | 91.67% | 96.83% |
| 20 / 3 / 0 | 96.17% | 97.33% | 98.17% | 97.33% | 98.17% |
| 20 / 3 / 0.5 | 72.83% | 93.67% | 95.50% | 93.50% | 95.33% |

FE with true heterogeneity is an intentional negative control. At n=8 and tau²=0.5, PM-normal coverage was 91.67% (exact binomial 95% interval 89.16–93.75%); PM-mKH was 96.83% (95.10–98.08%). At n=8 and tau²=0, mKH coverage was 99.17%, demonstrating conservatism. Exact intervals and Monte Carlo standard errors for every panel are in `r-calibration.json`. Matching an oracle is numerical evidence; these finite panels do not establish universal coverage, validity under dependent participants, or propagation of first-level variance-estimation uncertainty.


### Reproduction

From the workspace root, with the isolated reference library already built:

```sh
# Regenerate oracles, batched R benchmarks and all simulation panels; then qualify Rust.
python3 scripts/group_meta_benchmark.py --r experiments --run
# Recheck Rust and summarize the existing independent R artifacts.
python3 scripts/group_meta_benchmark.py --run
# Validate receipts/source hashes and summarize without rerunning experiments.
python3 scripts/group_meta_benchmark.py
```

The fmrigds snapshot and its manifest are under
`evidence/group-meta-regression/r-source/fmrigds` and `fmrigds-source.json`.
Rebuild the isolated library from that snapshot with:

```sh
R CMD INSTALL --preclean --no-multiarch \
  -l evidence/group-meta-regression/r-library \
  evidence/group-meta-regression/r-source/fmrigds
```

The runner sets OpenBLAS, OMP and Accelerate thread limits to one. It runs release
timings after builds/checks have completed. It never modifies the original R
repository, installs dependencies, or publishes results. The original unbatched R
timings are retained as adverse measurement evidence: sub-millisecond workloads
hit the elapsed timer's resolution limit. Final R trials batch repeated fits.
