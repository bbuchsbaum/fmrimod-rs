use fmrireg::LinearModel;
use ndarray::{array, s};
fn main() -> Result<(), fmrireg::FitError> {
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.]];
    let y = array![[1., 4.], [2., 2.], [2., 3.], [4., 1.]];
    let model = LinearModel::new(x.view(), 1e-12)?;
    let mut workspace = model.workspace(2)?;
    let fit = model.fit_into(y.view(), &mut workspace)?;
    let contrast = fit.t_contrast(array![0., 1.].view())?;
    let saved = fit.to_owned();
    let tail = model.fit_into(y.slice(s![.., 0..1]), &mut workspace)?;
    println!(
        "backend={:?}, retained={:?}, tail={:?}, t={:?}",
        model.backend(),
        saved.coefficients(),
        tail.coefficients(),
        contrast.statistic
    );
    Ok(())
}
