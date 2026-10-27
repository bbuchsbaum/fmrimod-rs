//! Two feature columns, independently preprocessed and convolved within two runs.
use fmrihrf::{hrf::SpmG2Hrf, regressor::*, SamplingFrame};
use ndarray::array;

fn main() -> fmrihrf::Result<()> {
    let first = array![[1., 3.], [2., 1.], [9., 9.], [4., 2.]];
    let second = array![[3., 2.], [5., 4.], [2., 1.]];
    let mask = [true, true, false, true];
    let times = [0., 0.3, 0.8];
    let features = SampledFeatures::new(
        &[
            FeatureRun {
                values: first.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 0.25,
                },
                mask: Some(&mask),
            },
            FeatureRun {
                values: second.view(),
                grid: FeatureGrid::Times(&times),
                mask: None,
            },
        ],
        FeatureOptions {
            center: true,
            scale: FeatureScale::SampleSd,
        },
    )?;
    let frame = SamplingFrame::new(vec![30, 25], vec![1.], None, 0.025)?;
    let hrf = SpmG2Hrf::new(5., 15., 1. / 120.);
    let design = features.evaluate(&frame, &hrf, ConvolutionMethod::Auto)?;
    assert_eq!(design.dim(), (55, 4));
    println!(
        "{} scans × {} columns: f1/b1, f2/b1, f1/b2, f2/b2",
        design.nrows(),
        design.ncols()
    );
    println!("Run preprocessing: {:?}", features.statistics());
    Ok(())
}
