# fmri-group

Independent-subject fMRI group estimation with identified cohorts, bounded feature
blocks, OLS, common-effect and DL meta-analysis, and explicit BH/BY families.
Uses `fmridataset` axes/readers and `fmrireg`'s rank-aware least squares.

See [the usage and qualification guide](../../docs/GROUP_ANALYSIS.md).

```sh
cargo run -p fmri-group --example group_pipeline
cargo test -p fmri-group
```

License: GPL-3.0-or-later. This crate depends on the existing GPL dataset foundation;
other workspace crates retain their own licenses.
