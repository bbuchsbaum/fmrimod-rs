# v0.1.0-alpha.1 developer preview

This first preview provides composable fMRI modelling through `fmrimod` and seven
component crates. Package versions and the repository tag agree. Rust 1.91.1 or
newer is required. Distribution is through pinned Git dependencies; crates.io
publication remains disabled. APIs may change before a stable release.

## Included capabilities

- HRFs, event convolution, sampled regressors, experimental designs and nuisance terms.
- Identified observations/features, spatial metadata, bounded data readers and optional NIfTI ingress.
- Reusable OLS/GLS preparation, selected-observation fitting, named estimands and requested marginal/joint covariance.
- AR(p) and ARMA transforms/estimation, conditional order selection, parcel ACVF pooling/shrinkage and run scales.
- Independent-subject OLS/meta-regression, repeated ML/REML models, explicit test families, permutation max-T and influence diagnostics.
- A task-oriented guide with an executable events-to-AR-GLS-to-group analysis,
  plus generated API documentation for all eight crates.

Start with the [published guide](https://bbuchsbaum.github.io/fmrimod-rs/),
[API](https://bbuchsbaum.github.io/fmrimod-rs/api/fmrimod/index.html), and
[Git dependency instructions](GIT_CONSUMERS.md). Per-crate licensing is retained
in the [license map](../LICENSE.md).

## Qualification and inference boundaries

The preview is gated on Linux, macOS and Windows CI, including default/all-feature
Rust tests, formatting, numerical-library Clippy, package/guide checks and
external Git consumer applications. The companion guide runs from an external
project as well as from the source workspace. Exact release evidence is attached
to the GitHub prerelease and recorded under `evidence/preview`.

All qualification data are synthetic. Existing independent R and analytic
references cover their recorded numerical cases. Plug-in AR and repeated-model
intervals have measured coverage shortfalls; first-level estimated variance is
not automatically integrated out by group meta-regression. Nonconvergence and
unsupported inputs remain explicit failures. Read the
[calibration report](COMPLETION_REPORT.md) for attempted/failed fits and conditional
coverage, and the [capability matrix](CAPABILITIES.md) for scope.

## Bounded benchmark snapshot

Measured on this shared ARM64 macOS workstation with Rust 1.91.1 release builds,
portable Rust backend, one worker/backend thread and 250-feature blocks. Both
paths use the same synthetic design/data, estimated AR(2), and requested outputs.
Five warm workload repetitions follow preparation. Times include data generation,
whitening, fitting and contrasts; they exclude compilation, process startup,
model preparation, reference I/O and the untimed validation export.

| Rows | Columns | Features | Rust warm median | R/fmriAR warm median | Maximum scaled output error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 300 | 8 | 1,000 | 5.49 ms | 10.0 ms | 2.05e-13 |
| 1,200 | 32 | 5,000 | 89.95 ms | 358.0 ms | 8.44e-11 |

The independent R path uses its source-built, hash-checked fmriAR installation
and QR regression. Every exported coefficient, residual variance, effect, SE,
statistic and p-value was compared before publishing these timings. The existing
scaled-error tolerance is 2e-8. First-workload and preparation measurements,
raw trials, source/binary hashes, R provider provenance and output checksums are
in [benchmark.json](../evidence/preview/benchmark.json).

These are two bounded application workloads, not a universal speedup, cold-start
or disk-throughput claim. R's elapsed timer is coarse at the smaller scale and
this was not a controlled or randomized benchmark host. Historical broader
comparisons remain separately source-bound in [the earlier benchmark guide](GROUP_BENCHMARK.md).

Reproduce with `python3 scripts/preview_benchmark.py` after preparing the isolated
R provider recorded in the AR fixture provenance; it requires NumPy and R's
`fmriAR`/`jsonlite`. The normal Rust build and guide need neither R nor NumPy.

For a relocated R source checkout, use `--r-source PATH`. A freshly rebuilt
reference provider needs an explicitly recorded `--reference-provenance FILE`
with `source_md5` and root-relative `binary_md5` entries. The harness checks both;
it never silently accepts another installed R package.
