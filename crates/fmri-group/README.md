# fmri-group

Identified fMRI group estimation with bounded feature processing: OLS, FE/DL/PM
meta-regression, repeated ML/REML models, permutation max-T, exact deletion
diagnostics, and explicitly defined multiple-testing families.

Uses `fmri-data` for identities/readers and `fmri-reg` for numerical regression.
The independent-subject scalar bundle and repeated covariance model have separate
contracts; arbitrary repeated-measures permutations are not implied.

Start with [group analysis](../../docs/GROUP_ANALYSIS.md),
[repeated models](../../docs/REPEATED_GROUP.md), and
[qualification limits](../../docs/COMPLETION_REPORT.md).

```sh
cargo run -p fmri-group --example group_pipeline
cargo run -p fmri-group --example repeated_group
cargo test -p fmri-group
```

License: GPL-3.0-or-later; the license text is included here.
