use fmri_noise::{Ar1Plan, FirstSample};
use fmri_splines::BSpline;
use ndarray::Array2;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let phi: f64 = 0.6;
    let w =
        Ar1Plan::new(vec![7], vec![phi], FirstSample::Stationary)?.apply(Array2::eye(7).view())?;
    let covariance = Array2::from_shape_fn((7, 7), |(i, j)| {
        phi.powi(i.abs_diff(j) as i32) / (1. - phi * phi)
    });
    let identity = w.dot(&covariance).dot(&w.t());
    for ((i, j), v) in identity.indexed_iter() {
        assert!((v - f64::from(i == j)).abs() < 1e-12);
    }
    let spline = BSpline::new(&[1., 2., 3., 4., 5.], None, None, 3, false, None)?;
    assert_eq!(spline.design_matrix(&[1.5, 2.5, 3.5])?.nrows(), 3);
    let _ = fmri_data::ObservationId::new("external-subject/visit1")?;
    println!("PASS components: covariance whitening, splines and identities");
    Ok(())
}
