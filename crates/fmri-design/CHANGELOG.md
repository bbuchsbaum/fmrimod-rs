> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# Changelog

## Unreleased — core port audit, 2026-09-05

- Route categorical, continuous, matrix, basis, and interaction event amplitudes
  through one duration-aware convolution engine; retain fractional event timing.
- Preserve all SPMG columns, align column metadata, and isolate runs including
  empty runs. Honor sampling-frame precision and HRF basis spans.
- Preserve raw continuous amplitudes unless centering/scaling is requested.
- Orthogonalize polynomial terms against the constant even when it is omitted.
  Replace approximate spline code with fitted B-splines and constrained natural
  cubic splines; match R quantiles, extrapolation, and baseline degree semantics.
- Compute explicit and linear contrast formulas, reject unsupported expressions,
  and reject interaction contrasts outside the supported four-cell case.
- Remove incomplete cross-model cache identities; event convolution caching now
  includes exact timing, amplitudes, options, and immutable HRF identity.
- Add source-backed R fixtures for complete event/baseline matrices, signed
  modulation, and spline fits/predictions, plus independent regression checks.
- Reject irregular or inconsistent time grids in the separate sampled-design
  convolution API instead of silently substituting a median interval.
- The dependent fmrihrf-rs core now uses pure Rust linear algebra and chooses
  direct or FFT convolution from input density. See the parity audit for evidence.
