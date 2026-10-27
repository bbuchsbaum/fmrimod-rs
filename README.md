# fmrimod-rs

Composable fMRI modelling in Rust: HRFs and experimental designs, typed data,
fast reusable OLS/GLS fitting, AR noise models, and group inference.

Start with **`fmrimod`** for an analysis pipeline. Use a component directly when
you need a smaller dependency set. R packages supply scientific references;
the Rust APIs use explicit identities, reusable plans and bounded data blocks.

| Crate | Responsibility |
| --- | --- |
| [fmrimod](crates/fmrimod) | Pipeline entry point, sessions, named outputs and group adapters |
| [fmri-data](crates/fmri-data) | Observation/feature identities, assays, spatial metadata and readers |
| [fmri-hrf](crates/fmri-hrf) | HRFs, convolution, sampled regressors and reconstruction |
| [fmri-design](crates/fmri-design) | Design matrices, formulas, nuisance terms and contrasts |
| [fmri-noise](crates/fmri-noise) | AR/ARMA estimation, whitening, censoring and run scales |
| [fmri-reg](crates/fmri-reg) | Reusable mass-univariate OLS/GLS and contrast estimation |
| [fmri-group](crates/fmri-group) | Group regression, meta-analysis, repeated models and resampling |
| [fmri-splines](crates/fmri-splines) | Numerical spline support for the workspace |

Cargo names use hyphens; Rust imports use underscores, for example `fmri_reg`.
The [architecture guide](docs/REPOSITORY.md) explains dependencies, naming changes
and repository conventions.

## Run from a checkout

Rust 1.91.1 or newer is required. Registry publication is disabled for this
initial source distribution; use the workspace directly.

```sh
cargo run -p fmrimod --example first_level
cargo run -p fmrimod --features formula --example complete_first_level
cargo run -p fmri-group --example repeated_group
```

The default `fmrimod` build omits the Polars formula frontend and NIfTI ingress.
Enable `formula`, `nifti`, or `accelerate` explicitly when needed. Plotting remains
an optional feature of the HRF/design components.

For direct numeric fitting, prepare a model once and reuse it across blocks:

```rust
use fmri_reg::LinearModel;
use ndarray::array;

fn main() -> Result<(), fmri_reg::FitError> {
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.], [1., 4.]];
    let y = array![[1., 2.], [2., 1.], [2., 4.], [4., 3.], [6., 6.]];
    let model = LinearModel::new(x.view(), 1e-12)?;
    let fit = model.fit(y.view())?;
    let contrast = fit.t_contrast(array![0., 1.].view())?;
    println!("{:?}", contrast);
    Ok(())
}
```

## Documentation and qualification

Read [the documentation index](docs/README.md), [capability matrix](docs/CAPABILITIES.md),
and [scientific qualification report](docs/COMPLETION_REPORT.md). That report
separates numerical agreement from coverage: estimated-noise and repeated-model
plug-in intervals have documented limitations. Synthetic qualification does not
establish real-data acceptance.

For development, see [CONTRIBUTING.md](CONTRIBUTING.md). CI is configured for
Linux, macOS and Windows; hosted execution is separate from local checks.

The [evidence index](evidence/README.md) explains retained receipts, fixtures,
archived raw experiments and reproduction. Original sibling repositories remain
intact. Historical porting notes are in [docs/history](docs/history/README.md).

Licenses are retained per crate; see [the license map](LICENSE.md). The data,
group and facade crates declare GPL-3.0-or-later. There is no blanket relicensing
of the workspace.
