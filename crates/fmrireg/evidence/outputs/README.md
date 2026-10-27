# Requested-output qualification, 2026-10-27

Scope: requirement-driven first-level summaries and their upstream compiler.
`manifest.json` binds source/reference hashes, hardware, commands and exit
statuses. No commit or publication was made for this implementation.

## Correctness

- All-feature tests across fmrireg, fmri-noise and fmrimod: **71 passed**.
- fmrireg without default features: **33 passed**.
- After removing a redundant finite scan, final affected output/dataset/full-fit
  executor tests: **16 passed**, exercising Portable and Accelerate.
- Final scoped Clippy (`--no-deps`) and formatting checks: **passed**.
- Dependency-inclusive Clippy: **failed** on five existing splines lints;
  preserved in `clippy-dependency-failure.log`. No lints were disabled for that
  dependency and no unrelated source was repaired.

Eight new numerical tests cover analytic Walsh OLS effects/RSS/covariance,
independent stationary AR(1) precision, near-collinear estimands, zero/dependent
contrasts, nonestimability, both backends, strided/tail buffers, custom-transform
semantics, owned lifetimes and plan-clone identity. Three new block tests cover
ordering, bounds, reader/sink errors and assay identity. Seven upstream tests
cover metadata/weight conflicts, union mapping, reuse capabilities and an
end-to-end compile/prepare/run path. Existing regression tests also ran.

Ordinary analytic comparisons use relative-plus-absolute tolerances around
2e-12; existing R effects/SEs use 2e-11. All 20 source MD5s recorded in the R
fixture match the current fmriAR source files. R was not rebuilt this turn;
new numerical changes have independent analytic checks.

The near-collinear witness uses X=[u,u+2^-e v], e=20/28/30, with orthogonal
Walsh u,v and a third orthogonal unit residual vector. The estimand [1,1]
has unit variance1/8 and observed variance1/6 at df6. Its 2e-6 tolerance allows
roundoff from cancellation at condition numbers of order1e9. Marginal/joint
agreement is checked at 1e-12. Preparation now uses H=L X^+, G=H Hᵀ instead
of expanding L V0 Lᵀ. This addresses the identified new-output cancellation;
it does not qualify legacy contrast covariance or every ill-conditioned model.

Independent read-only review found the covariance issue and an ineffective
derived Clone bound; both were fixed and re-reviewed. Final review confirmed
Gram geometry, context-sharing Clone and validated-kernel preconditions, with
no remaining blocker in that bounded review. Reviewer did not run Rust tests.

## Performance

`benchmark.csv` contains 96 production timing cases and 48 experimental direct
projection cases, plus 18 adversarial comparisons. The latter records have
variable-width diagnostic fields after `adverse`; parse by record kind.

Apple M3 Max, Rust1.91.1, release profile, one worker and BLAS threads constrained
to one. n=300/1200, p=8/32, q=1/4/16, 2,048 features per block, OLS or fixed
zero-history AR(4) across four runs. The generator is deterministic and is in
`examples/output_benchmark.rs`; contrasts combine two coefficient rows.

Baseline: reusable full fit plus L beta and residual scale. New path: requested
effects, with or without residual scale. Effects are checked entrywise and
uncertainty paths check residual scales against the full fit. Variance geometry
and accessors are covered independently by the numerical tests. Timing excludes
preparation and output copying/materialization; both paths expose compact values,
without t/p-value computation. Each sample uses three warmups, five adaptive
batches and their median; production comparison reverses baseline/request order
and averages the two medians. Ratios are descriptive, not confidence intervals.

Representative Accelerate case, n=1,200, p=32, q=1, b=2,048:

| Model/request | Full fit + effects ms | Requested ms | Speedup |
|---|---:|---:|---:|
| OLS estimates | 2.491 | 1.192 | 2.09x |
| AR4 estimates | 4.269 | 2.888 | 1.48x |
| OLS marginal | 2.507 | 2.472 | 1.01x |
| AR4 marginal | 4.288 | 4.245 | 1.01x |

Across the tested grid, Accelerate estimates-only speedups were1.83–3.15x OLS
and1.35–1.63x AR4; marginal ratios were0.957–1.020x. Portable estimates-only
ratios were1.77–4.02x OLS and1.53–2.49x AR4. Marginal primarily reduces retained
outputs, not full-model work; the small timing changes should not be oversold.

For the representative case, OLS estimates-only explicit workspace falls from
20,201,472 to540,672 bytes; AR4 from39,862,272 to20,201,472 bytes. Marginal adds
16,384 bytes to full-fit workspace for the output buffer. Retained uncertainty
results are32,768 bytes (one effect + scale), versus540,672 for32 betas + scale,
excluding shared geometry/specification. These are allocation counts, not OS
peak RSS. Backend packing, preparation, model storage and sink retention are
outside workspace counts.

No disk I/O, NIfTI decompression, noise estimation, 25-subject scheduling, group
reduction or viewer delivery was timed. Cold/end-to-end latency is unmeasured.

## Direct-projection candidate: not enabled

The example probes K=L(WX)^+ applied to explicitly whitened Y. Normal-case
maximum absolute discrepancy against the established path was1.599e-14, but
near-collinear/large-offset cases had severe discrepancies: Accelerate with
near-collinearity1e-9 and offset1e12 differed by about1.0e5. This comparison
alone does not prove which arithmetic order is closer to truth on every adverse
input; it fails to establish a safe general dispatch rule. The candidate remains
outside production. OLS candidate timings include an explicit identity copy,
so they are conservative probes, not an optimized OLS implementation.

Persistent sessions, projection caches and NIfTI/cohort orchestration remain
the later stage of the recorded plan.
