# Model fMRI data in Rust

fmrimod connects experimental events, typed observation/feature identities,
reusable first-level estimation and group inference. It is a Rust library for
analyses you compose in your own application.

This guide follows one runnable synthetic analysis: convolve task events with
an HRF, fit 12 independent participants with estimated AR(1) noise, retain one
named effect and its sampling variance, then estimate a population effect in
four regions. The same executable supplies the code excerpts throughout.

```sh
cargo run -p fmrimod --example guide_workflow --locked
```

The analysis is small enough to inspect and runs without R or external data.
Its assertions check successful execution, identities and finite results; the
larger numerical suite and calibration studies establish their own bounded
correctness and inference claims.

This is **developer preview 0.1.0-alpha.1**. APIs may change. Inference is
conditional on the fitted noise/covariance models, with measured calibration
limits. Read [the interpretation chapter](interpret.md) before adapting the
example to substantive data.

Start with [installation](install.md), or go directly to the
[generated Rust API](api/fmrimod/index.html). The site records its exact source
revision in [build metadata](build-info.json).
