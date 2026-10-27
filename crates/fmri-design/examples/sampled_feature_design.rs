//! Build named continuous-feature columns and select a basis-specific contrast.
use fmri_design::{
    contrast::Contrast,
    design::{hstack_design_matrices, DesignMatrix},
    sampled_features::sampled_feature_design,
    types::ContrastSpec,
};
use fmri_hrf::{hrf::SpmG2Hrf, regressor::*, SamplingFrame};
use ndarray::{array, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = array![[1., 3.], [2., 1.], [4., 2.], [3., 4.]];
    let features = SampledFeatures::new(
        &[FeatureRun {
            values: values.view(),
            grid: FeatureGrid::Uniform {
                start: 0.,
                step: 0.25,
            },
            mask: None,
        }],
        FeatureOptions::default(),
    )?;
    let frame = SamplingFrame::uniform(40, 0.8, 0.025)?;
    let hrf = SpmG2Hrf::new(5., 15., 1. / 120.);
    let task = sampled_feature_design(
        &features,
        &["sound", "motion"],
        &frame,
        &hrf,
        ConvolutionMethod::Auto,
    )?;
    let nuisance = DesignMatrix::with_times(
        Array2::ones((40, 1)),
        vec!["intercept".into()],
        task.row_times.clone(),
    );
    let design = hstack_design_matrices(&[task, nuisance])?;
    let weights = Contrast::pair("sound-minus-motion", "sound.b1", "motion.b1")
        .compute_weights(&design.column_names)?;
    assert_eq!(weights, array![1., -1., 0., 0., 0.]);
    println!("{design}");
    println!("Canonical-basis contrast: {weights}");
    Ok(())
}
