# fmri-splines

Spline bases used by the fmrimod workspace, including B-splines, natural cubic
splines and interpolation splines. Rust imports use `fmri_splines`.

This is the workspace's numerical support crate. It is distinct from the
unrelated `splines` package on crates.io. HRF users can generally use the spline
reexports in `fmri_hrf` instead of adding a direct dependency.

```rust
use fmri_splines::BSpline;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let spline = BSpline::new(&x, None, None, 3, false, None)?;
    let values = spline.design_matrix(&[1.5, 2.5, 3.5])?;
    assert_eq!(values.nrows(), 3);
    Ok(())
}
```

`parallel` is optional. R `splines` supplies numerical references; qualification
covers the checked contracts rather than a blanket R parity claim.
License: MIT OR Apache-2.0; both license texts are included here.
