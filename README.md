# fmrimod-rs

Composable fMRI modelling in Rust: HRFs and event designs, spatially typed
bounded data access, and reusable mass-univariate estimation.

Group OLS, FE/DL/PM meta-regression, and covariance-bound repeated ML/REML models
are available through `fmri-group` and `fmrimod::group`. Portable estimate bundles,
whole-family permutation max-T and exact deletion diagnostics have separate APIs.
Start with the [group guide](docs/GROUP_ANALYSIS.md), [repeated models](docs/REPEATED_GROUP.md),
and [current qualification and limits](docs/COMPLETION_REPORT.md).

This workspace consolidates the working sources of `fmrihrf-rs`,
`fmridesign-rs`, and `fmridataset-rs`. `fmrireg` and `fmri-noise` are new crates.
The aim is scientific capabilities with efficient Rust interfaces; R package
syntax and object systems are not the API blueprint.

```rust
use fmrireg::LinearModel;
use ndarray::array;

let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.], [1., 4.]];
let y = array![[1., 2.], [2., 1.], [2., 4.], [4., 3.], [6., 6.]];
let model = LinearModel::new(x.view(), 1e-12)?;
let fit = model.fit(y.view())?;
let contrast = fit.t_contrast(array![0., 1.].view())?;
# Ok::<(), fmrireg::FitError>(())
```

Prepare the design once and reuse it across feature blocks. A fit stores
coefficients and residual variances; t/F contrasts reuse that fit. Rank-deficient
models require estimable contrasts. Known AR(p) GLS and one-step feasible GLS
are explicit APIs; the latter estimates design-corrected noise from a supplied
representative matrix and reuses the plan for subsequent feature blocks.

`fmrireg::dataset::fit_blocks` (feature `dataset`) streams f64 readers into a
result sink with an explicit working-buffer budget. Row identity/order must
already match the design. `BoundModel::fit_assay` additionally verifies exact
observation IDs and order before opening the reader, and supplies feature IDs
to the sink. Both OLS and GLS support block execution. `fmrimod` reexports the components; its optional
`formula` feature enables the legacy dataframe/design frontend. Plotting is
opt-in in `fmridesign`.

Run the event-to-fit example:

```sh
cargo run -p fmrimod --example first_level
# Events + sampled features + nuisance + censoring + estimated AR(2) + streamed contrasts
cargo run -p fmrimod --features formula --example complete_first_level
cargo test --workspace --all-features -j 2
```

See [the execution plan](docs/PLAN.md) for scope, architecture and acceptance
criteria, [capability status](docs/CAPABILITIES.md) and the [HRF/design audit](docs/HRF_DESIGN_AUDIT.md) for limitations and compatibility changes, and
[integrated qualification](docs/QUALIFICATION.md) for current workspace checks,
selected-row performance and statistical limitations. [Earlier evidence](evidence/README.md)
records prior stages. Existing crate audit documents
are historical; their percentages and September source references do not certify
current R behavior.

Original sibling repositories remain intact. Source hashes and original Git
HEAD/status are in `evidence/source-manifest.json`; local Git history bundles are
kept in `evidence/` but excluded from version control. Each crate retains its
license. Dataset and the facade declare GPL-3.0-or-later; the workspace has no
blanket relicensing declaration. Nothing has been published or remotely migrated.

Run `python3 scripts/audit_upstream.py` to detect drift in the five local R
reference checkouts. A drift report requires review and fresh numerical
qualification; it never automatically blesses new upstream behavior.
