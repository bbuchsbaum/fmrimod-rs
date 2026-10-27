use fmrimod::{hrf, regression::LinearModel};
use ndarray::{array, Array2};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = hrf::SamplingFrame::uniform(100, 1.5, 0.1)?;
    let amplitudes = array![[1.], [1.], [1.], [1.]];
    let events = hrf::regressor::EventTable {
        onsets: &[5., 30., 65., 95.],
        durations: &[1.],
        amplitudes: amplitudes.view(),
        run_ids: &[0, 0, 0, 0],
    };
    let response = events.evaluate(
        &frame,
        &hrf::hrf::SpmG1Hrf::new(5.0, 15.0, 1.0 / 120.0),
        hrf::regressor::ConvolutionMethod::Auto,
    )?;
    let x = Array2::from_shape_fn(
        (100, 2),
        |(i, j)| if j == 0 { 1. } else { response[(i, 0)] },
    );
    let y = Array2::from_shape_fn((100, 3), |(i, j)| {
        10. + (j + 1) as f64 * x[(i, 1)] + 0.1 * ((i * 7 + j) as f64).sin()
    });
    let model = LinearModel::new(x.view(), 1e-12)?;
    let fit = model.fit(y.view())?;
    let contrast = fit.t_contrast(array![0., 1.].view())?;
    println!(
        "rank={}, df={}, task effects={:?}",
        model.rank(),
        fit.residual_df(),
        contrast.effect
    );
    Ok(())
}
