> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# fmridesign-rs

Build fMRI event and nuisance design matrices in Rust. The authoritative behavioral
references are the local R packages `fmridesign` and `fmrihrf`; this crate uses
`../fmrihrf-rs` for hemodynamic response functions and convolution.

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmridesign::{formula::event_model, sampling::SamplingFrame, DataFrame};

    let mut data = DataFrame::new();
    data.add_column("onset", vec![0.13, 8.27])?;
    data.add_column("duration", vec![0.0, 0.15])?;
    data.add_str_column("condition", vec!["A".into(), "B".into()])?;
    let frame = SamplingFrame::new(vec![100], vec![2.0], None, 0.05)?;
    let model = event_model("onset ~ hrf(condition, basis=\"spmg3\")", &data, frame, 2.0)?;
    let matrix = model.generate_design_matrix()?;
    assert_eq!(matrix.dim(), (100, 6));
    assert_eq!(model.column_names().len(), matrix.ncols());
    Ok(())
}
```

Events retain fractional onsets and durations in seconds. Positive durations are
integrated blocks, including durations below the convolution precision. Signed
amplitudes are preserved; centering and scaling must be requested. Multiple HRF
bases produce multiple columns with matching names.

For a multi-run sampling frame, provide a numeric `run` or `block` column with
one-based run indices and run-local onsets. An explicit `run` column takes
precedence. In single-run formulas, `block` remains available as an ordinary
experimental factor. The lower-level `fmrihrf::regressor::EventTable` uses zero-based
run indices and can retain explicitly supplied predictors absent from the events.

Polynomial, B-spline, and natural-spline baselines are evaluated separately per
run. Spline `df` counts returned columns, intercepts are omitted by default, and
the first successful spline evaluation fixes knots for later prediction.
For baseline B-splines, `degree` is the polynomial degree, matching R.

Use `contrast(name, Vec<f64>)` for exact weights, pair contrasts for condition
group comparisons, or linear formulas with exact names and numeric coefficients.
Unsupported expressions return errors. Contrast weights are not silently replaced.

Run `cargo test --all-features` for the full suite and
`cargo test --no-default-features --test r_core_contract` for the current R contracts.
The numerical core builds without BLAS or a Fortran compiler.

See [PORT_PARITY_AUDIT.md](PORT_PARITY_AUDIT.md) for verified scope, R discrepancies,
performance evidence, reproducible fixture generation, and remaining gaps.
