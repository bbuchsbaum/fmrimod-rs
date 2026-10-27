use fmrimod::{
    hrf::{hrf::spm_canonical, Hrf},
    regression::LinearModel,
};
use ndarray::{array, Array1};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.], [1., 4.]];
    let y = array![[1.], [2.], [2.], [4.], [6.]];
    let model = LinearModel::new(x.view(), 1e-12)?;
    let fit = model.fit(y.view())?;
    assert!((fit.coefficients()[(0, 0)] - 0.6).abs() < 1e-12);
    assert!((fit.coefficients()[(1, 0)] - 1.2).abs() < 1e-12);
    let c = fit.t_contrast(array![0., 1.].view())?;
    let _ = c;
    let hrf = spm_canonical(None, None, None, None, None, None);
    let values = hrf.evaluate(&Array1::linspace(0., 24., 49))?;
    assert!(values.iter().all(|v| v.is_finite()));
    println!("PASS facade: analytic OLS and HRF evaluation");
    Ok(())
}
