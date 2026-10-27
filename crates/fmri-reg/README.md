# fmri-reg

Reusable mass-univariate OLS/GLS factorization, contrasts, uncertainty, and
bounded block execution. Rust imports use `fmri_reg`.

Prepare a model once and fit multiple response blocks. `LinearModel` and
`GlsModel` separate model preparation from fitting; named output specifications
retain only the estimates and covariance required by downstream consumers.
See [output contracts](OUTPUTS.md) and [the workspace example](../../README.md).

Features are `dataset` for typed data adapters and `accelerate` for the optional
macOS system backend. Both are off by default. `fmrimod` enables `dataset` and
provides the identified analysis/session layer above these numeric solvers.

```sh
cargo test -p fmri-reg --all-features
cargo run -p fmri-reg --features dataset --example selected_arp
```

Performance evidence and numerical tradeoffs are documented in
[PERFORMANCE.md](PERFORMANCE.md), [ALGEBRA.md](ALGEBRA.md), and
[HILL_CLIMB.md](HILL_CLIMB.md). Raw historical artifacts are indexed under
[evidence](../../evidence/README.md).

License: MIT OR Apache-2.0; both license texts are included here. The optional
data dependency retains its own GPL-3.0-or-later declaration.
