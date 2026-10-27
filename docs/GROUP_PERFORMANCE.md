# Group performance plan and qualification

Scope: improve the existing independent-subject OLS, common-effect and DL
estimators without changing their estimands, variance policy, rank policy,
missingness rules or first-level AR/OLS kernels. Mote:
`bd-01M5YSR0THZH656APQQ7GA7937`.

## Tranche 1: implement and measure now

1. Preserve a release baseline executable and source hashes. Use the same
   benchmark inputs, output requests, threads and reference payloads for both
   versions. Record allocation calls/bytes separately from timings, and measure
   standalone input reads and p-value evaluation to locate remaining costs.
2. Reuse the common-effect estimate/SE in FE and zero-heterogeneity DL fits.
   Compute the stable DL denominator only when DL needs positive heterogeneity.
   Replace per-feature row/pair allocations with bounded reusable tiled scratch.
3. Specialize an actual all-ones, single-column design with compensated,
   centered two-pass moments. Retain signed/scaled contrasts and feature-local
   failure handling. Never form variance by subtracting large raw moments.
4. Add a reusable `PreparedOls` that owns the bound design, contrasts, bounded
   missingness-pattern cache, gathering buffer and fit workspace. Keep `fit_ols`
   as a convenience wrapper. Complete blocks bypass missing-pattern grouping and
   gathering; arbitrary designs continue to use the existing rank-aware solver.
5. Qualify observable outputs with existing base-R QR/metafor/fmrigds fixtures,
   independent high-precision centered-moment witnesses, cancellation and extreme
   weight cases, row/block invariance, missingness/rank failures, budget and sink
   errors, and prepared reuse with changed data. Run affected facade checks,
   formatting and scoped clippy. Retain adverse benchmark results.

Acceptance: all applicable numerical/status checks pass; no hidden regularization
or IID substitution; bounded memory remains enforced. Report per-workload medians
and raw trials, not an extrapolated universal speedup. New paths whose cost or
accuracy is worse should be revised or explicitly deferred. Baseline R reference
is the isolated fmrigds source recorded in GROUP_BENCHMARK.md, not the changing
live R checkout. No remote publication is part of this work.

## Later analytical stages

These are planned extensions, not claims of implemented or measured performance.

* **Paule–Mandel/meta-regression:** for t=tau² and profiled weighted residual sum
  Q(t), use Q'(t)=-sum(r_i(t)²/(v_i+t)²) in a bracketed Newton method. Compare
  against independent bracketed roots, including zero-boundary and ill-conditioned
  cases. Preserve the estimator; iteration reduction is an empirical question.
* **Shared nuisance subspaces:** prepare QR-based residualization when many
  target models share nuisance regressors. Bind reuse to row selection and
  weights; retain full-model degrees of freedom. Benchmark repeated-model work.
* **Missing-row/influence updates:** stable QR/Cholesky downdates with rank and
  leverage guards and refactorization fallback. Compare every deletion with an
  independent refit. Re-estimated random-effects weights require a new tau² fit.
* **Permutation inference:** batched signed sums and, for feasible exhaustive
  tests, Gray-code sign enumeration; stream family maxima. Specify symmetry,
  exchangeability blocks and nuisance strategy before optimizing. Qualify rounding
  drift and small residual variance against direct enumeration.
* **Repeated measures:** exploit mean/within-subject eigenspaces for declared
  homogeneous random-intercept models and small Woodbury/determinant-lemma systems
  for low-rank random effects. Compare likelihoods/inference with dense covariance
  oracles; do not apply homogeneous shortcuts to unequal sampling variances.

Precision-reducing variance binning, mixed precision and sketches require separate
accuracy contracts. Existing rejected reassociations and residual-kernel failures
in `crates/fmri-reg/ALGEBRA.md` remain acceptance counterexamples.

## Implemented API and numerical scope

All five items in tranche 1 are implemented. `fit_ols` remains the one-shot entry
point. For repeated execution on the same ordered subjects and design:

```rust,ignore
let mut prepared = PreparedOls::new(&design, &contrasts, options)?;
let first = prepared.fit(&cohort, |block| write_block(block))?;
let next = prepared.fit(&updated_cohort, |block| write_block(block))?;
```

The plan owns its design/contrasts/options and rereads each cohort. Different
feature axes and new response values are supported; different observation IDs
are rejected. The FIFO pattern cache is bounded by `max_cached_patterns`.
`RunReceipt::factorizations` counts successful initial preparation on first use
and newly prepared patterns thereafter. Intercept fits have zero factorizations.
The numeric budget estimate retains the largest block capacity used by a plan,
so smaller subsequent inputs do not hide retained workspace storage.

An intercept is recognized from its actual all-ones single-column matrix, not
its column name. Its mean and residual variance use compensated sums of deviations
from an observed anchor. The residual pass stays in centered coordinates, avoiding
loss of small variance around a large offset. Constant features report degenerate
variance; unsupported overflow remains a feature-local numerical failure. This
improves the numerical realization of the same model, not bitwise equivalence to
SVD residuals. General designs retain the existing solver and contrast rank check.

Meta-analysis gathers at most 32 features into reused contiguous pair buffers.
Weights remain scaled and means compensated. Positive DL heterogeneity retains
the stable positive-cross-product denominator and requires a second weighting
pass. Common-effect and zero-tau² DL reuse their first estimate and SE exactly.

The output schema remains unchanged. Numeric workspaces are reused, but feature
IDs, owned status/estimate vectors and sink-retained outputs still allocate.
General incomplete blocks still group row-index vectors; no bitset cache or
parallel executor is claimed. A changed row count can resize the single general
fit workspace. Contrast covariance validation is still performed per block.

## Qualification and measurement

The group suite has 16 passing integration tests, including base-R QR/metafor
fixtures, strict unit/offset tests against a 256-digit Python Decimal oracle,
tile boundaries 31/32/33, missingness, rank loss, signed/scaled contrasts,
constant/overflow isolation, prepared reuse, and budget/sink failures. The
all-feature group/facade check passed 27 tests. Scoped clippy and formatting are
recorded separately. No new population-level inference calibration is claimed.

`scripts/group_performance.py` regenerates the independent Decimal fixtures.
The benchmark checks every requested output against the preserved fmrigds
references before timing and also checks feature blocks 256 versus 113. Its
allocation diagnostic is outside timed trials. Allocation bytes are cumulative
requested bytes, not live memory or peak RSS. Standalone input-read and CDF loops
are cost probes, not additive phase timings.

Frozen baseline/candidate sources share dependencies and the instrumented harness;
only `fmri-group/src/fit.rs` differs in runtime library code. Executable hashes and
an intercept-factorization witness guard against comparing the same executable.
An initial shared Cargo target reused an executable incorrectly; those rejected
runs are retained under `rejected-shared-target-run/` and excluded from results.
Whole-suite ABBA trials showed substantial temporal variability, including an
adverse 0.93x large-DL result. They remain under `variable-whole-suite/`; final
comparisons pair each individual workload in ABBA order to reduce time drift.

Evidence directory: `evidence/group-performance/`. This is a shared arm64 macOS
workstation with one-thread environment settings and portable release Rust, not
a controlled benchmark host. The timed public calls include reads, preparation,
fitting, inference and output collection. This tranche measures one-shot fits;
repeat-plan speedup is not separately quantified.

Final medians below combine ten warm trials per implementation (two runs of five),
paired ABBA separately for each workload. Allocation counting is disabled during
those trials. These speedups compare previous Rust with revised Rust, not R.

| Subjects | Features | Columns | Missing | Method | Previous Rust ms | Revised Rust ms | Speedup |
|---:|---:|---:|---|---|---:|---:|---:|
| 20 | 1,000 | 1 | complete | OLS | 0.491 | 0.506 | 0.97x |
| 20 | 1,000 | 1 | complete | FE | 0.685 | 0.348 | 1.97x |
| 20 | 1,000 | 1 | complete | DL | 0.700 | 0.476 | 1.47x |
| 100 | 10,000 | 1 | complete | OLS | 13.889 | 9.801 | 1.42x |
| 100 | 10,000 | 1 | complete | FE | 26.702 | 14.066 | 1.90x |
| 100 | 10,000 | 1 | complete | DL | 26.466 | 20.826 | 1.27x |
| 100 | 10,000 | 1 | patterned | OLS | 13.751 | 9.864 | 1.39x |
| 100 | 10,000 | 1 | patterned | FE | 26.765 | 14.341 | 1.87x |
| 100 | 10,000 | 1 | patterned | DL | 26.508 | 20.734 | 1.28x |
| 100 | 100,000 | 1 | complete | OLS | 138.929 | 96.642 | 1.44x |
| 100 | 100,000 | 1 | complete | FE | 264.499 | 144.526 | 1.83x |
| 100 | 100,000 | 1 | complete | DL | 264.715 | 210.463 | 1.26x |
| 100 | 10,000 | 5 | complete | OLS | 11.499 | 6.593 | 1.74x |
| 100 | 10,000 | 20 | complete | OLS | 13.235 | 8.685 | 1.52x |

The smallest OLS case is 3% slower in this paired run; it is not counted as a
speedup. Larger OLS, FE and DL cases improve. All trials remain available, including
the earlier adverse large-DL result. Shared-machine timing noise limits precision;
these are workload-specific observations, not a universal performance guarantee.

For 100 subjects x 100,000 features, OLS allocation calls fall from 911,404 to
302,504 and FE/DL from 1,002,777 to 302,779. Cumulative requested allocation bytes
fall from about 555 MB to 103 MB for OLS and 545 MB to 183 MB for meta-analysis.
These include reader/output/receipt work and repeated allocation of memory; they
are not peak-memory ratios. Remaining output objects account for many calls.

Standalone CDF and input-read probes are in `comparison.json`. They support
investigating optional inference outputs and reader/output overhead next; they do
not justify replacing the qualified CDF or assigning additive percentages to phases.

Reproduce the independent fixture with `python3 scripts/group_performance.py`.
To reproduce timings, build the two frozen workspaces' `group_benchmark` examples
in release mode, copy each resulting executable to `baseline_benchmark` or
`candidate_benchmark` in the evidence directory, and run
`python3 scripts/group_performance.py --compare` from the repository root.
Use separate Cargo target directories, or explicitly force the changed library to
recompile before copying an executable; the runner rejects identical binaries and
checks an estimator-path witness. No test/build jobs should overlap timed trials.

