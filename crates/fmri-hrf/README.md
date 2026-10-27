# fmri-hrf

Hemodynamic response functions, sampled regressors, convolution, spline bases,
and reconstruction utilities for fMRI modelling. Rust imports use `fmri_hrf`.

Use this crate directly for HRF work, or use `fmrimod::hrf` in an analysis pipeline.
The R package `fmrihrf` supplies scientific references; its object model is not
this crate's interface. Current numerical contracts and known differences are in
[the HRF/design audit](../../docs/HRF_DESIGN_AUDIT.md).

Default builds omit plotting and parallel execution. Optional features are
`serde`, `parallel`, and `plotting`.

```sh
cargo test -p fmri-hrf
cargo doc -p fmri-hrf --no-deps
```

See the [workspace guide](../../README.md) for integration and qualification.
License: MIT OR Apache-2.0; both license texts are included here.
