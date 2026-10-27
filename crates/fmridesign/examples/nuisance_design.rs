//! Events + sampled features + drift + motion, with two explicit censor choices.
use fmridesign::{
    baseline::InterceptType,
    design::{hstack_design_matrices, DesignMatrix},
    nuisance::*,
    sampled_features::sampled_feature_design,
};
use fmrihrf::{hrf::SpmG1Hrf, regressor::*, SamplingFrame};
use ndarray::{array, Array1, Array2};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = SamplingFrame::new(vec![40, 32], vec![0.8, 1.], None, 0.025)?;
    let hrf = SpmG1Hrf::new(5., 15., 1. / 120.);
    let amps = array![[1.], [1.], [1.], [1.]];
    let events = EventTable {
        onsets: &[2., 12., 3., 15.],
        durations: &[0.5],
        amplitudes: amps.view(),
        run_ids: &[0, 0, 1, 1],
    };
    let task = DesignMatrix::with_times(
        events.evaluate(&frame, &hrf, ConvolutionMethod::Auto)?,
        vec!["task".into()],
        Array1::from_vec(frame.samples()),
    );
    let f1 = Array2::from_shape_fn((64, 1), |(i, _)| (i as f64 * 0.4).sin());
    let f2 = Array2::from_shape_fn((64, 1), |(i, _)| (i as f64 * 0.3).cos());
    let features = SampledFeatures::new(
        &[
            FeatureRun {
                values: f1.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 0.5,
                },
                mask: None,
            },
            FeatureRun {
                values: f2.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 0.5,
                },
                mask: None,
            },
        ],
        FeatureOptions::default(),
    )?;
    let continuous =
        sampled_feature_design(&features, &["sound"], &frame, &hrf, ConvolutionMethod::Auto)?;
    let drift = drift_design(
        &frame,
        DriftBasis::Dct {
            cutoff_seconds: 16.,
        },
        InterceptType::Runwise,
    )?;
    let motion1 = Array2::from_shape_fn((40, 1), |(i, _)| 0.1 * (i as f64 * 0.7).sin());
    let motion2 = Array2::from_shape_fn((32, 1), |(i, _)| 0.1 * (i as f64 * 0.9).sin());
    let motion = nuisance_design(
        &frame,
        &[
            NuisanceRun {
                values: motion1.view(),
                names: &["motion_x"],
            },
            NuisanceRun {
                values: motion2.view(),
                names: &["motion_x"],
            },
        ],
    )?;
    let design = hstack_design_matrices(&[task, continuous, drift, motion])?;
    let mut mask = vec![false; 72];
    mask[5] = true;
    mask[43] = true;
    let censor = CensorPlan::new(&frame, &mask)?;
    let spikes = hstack_design_matrices(&[design.clone(), censor.spikes()?])?;
    let selected = censor.select_design(&design)?;
    let response = Array2::from_shape_fn((72, 2), |(i, j)| ((i + j) as f64).sin());
    let selected_response = censor.select_rows(response.view())?;
    assert_eq!(spikes.nrows(), 72);
    assert_eq!(selected.nrows(), 70);
    assert_eq!(selected_response.nrows(), selected.nrows());
    println!("Original: {design}\nSpike design: {spikes}\nExcluded-row design: {selected}");
    println!(
        "Retained original row indices: {:?}",
        censor.retained_indices()
    );
    Ok(())
}
