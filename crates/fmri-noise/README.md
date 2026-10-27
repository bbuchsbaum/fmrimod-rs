# fmri-noise

Reusable run-aware whitening plans and noise estimators. Rust imports use
`fmri_noise`; `fmrimod` also exposes this crate as `noise`.

Capabilities include AR(p), stationary/conditional starts, selected-observation
transforms, design-corrected estimation, conditional AR order selection, parcel
ACVF pooling with explicit shrinkage, ARMA filtering/estimation, and run scales.
Apply matched transforms to both design and response; requested AR models never
silently become IID fits.

See [noise extension contracts](../../docs/NOISE_EXTENSIONS.md) and
[qualification limits](../../docs/COMPLETION_REPORT.md), including undercoverage
for short runs and persistent noise under plug-in inference.

```sh
cargo test -p fmri-noise
```

License: MIT OR Apache-2.0; both license texts are included here.
