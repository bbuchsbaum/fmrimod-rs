# fmri-design

Experimental design matrices, event and sampled-feature regressors, nuisance
terms, contrasts and design diagnostics. Rust imports use `fmri_design`.

This crate contains the dataframe/formula frontend and depends on Polars. It is
optional in `fmrimod`: enable `formula` when those interfaces are needed. The
borrowed-event first-level path in `fmrimod` does not require this dependency.

```sh
cargo run -p fmrimod --features formula --example complete_first_level
cargo test -p fmri-design
```

Optional features are `visualization` and `parallel`; neither is enabled by default.
See [current design contracts](../../docs/HRF_DESIGN_AUDIT.md), the
[capability matrix](../../docs/CAPABILITIES.md), and the [workspace guide](../../README.md).
Historical porting notes are kept under [docs/history](../../docs/history/README.md).
License: MIT OR Apache-2.0; both license texts are included here.
