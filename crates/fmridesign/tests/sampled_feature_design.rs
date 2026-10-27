use fmridesign::{
    contrast::Contrast,
    design::{hstack_design_matrices, DesignMatrix},
    sampled_features::sampled_feature_design,
    types::ContrastSpec,
};
use fmrihrf::{hrf::SpmG2Hrf, regressor::*, SamplingFrame};
use ndarray::{array, Array2};
use serde_json::Value;
fn vector(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn matrix(v: &Value) -> Array2<f64> {
    let a = v.as_array().unwrap();
    Array2::from_shape_fn((a.len(), a[0].as_array().unwrap().len()), |(i, j)| {
        a[i][j].as_f64().unwrap()
    })
}
fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < tol * (1. + b.abs()), "{a} != {b}");
    }
}
#[test]
fn current_r_uniform_irregular_mask_center_scale_and_multibasis() {
    let f: Value = serde_json::from_str(include_str!(
        "../../fmrihrf/tests/fixtures/sampled_features.json"
    ))
    .unwrap();
    let values: Vec<_> = f["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| matrix(&r["values"]))
        .collect();
    let masks: Vec<Vec<bool>> = f["inputs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r["mask"]
                .as_array()
                .unwrap()
                .iter()
                .map(|b| b.as_bool().unwrap())
                .collect()
        })
        .collect();
    let frame = SamplingFrame::new(
        vec![16, 13],
        vector(&f["tr"]),
        Some(vector(&f["start"])),
        f["precision"].as_f64().unwrap(),
    )
    .unwrap();
    let hrf = SpmG2Hrf::new(5., 15., 1. / 120.);
    for case in f["cases"].as_array().unwrap() {
        let times: Vec<_> = case["onsets"]
            .as_array()
            .unwrap()
            .iter()
            .map(vector)
            .collect();
        let runs: Vec<_> = (0..2)
            .map(|r| FeatureRun {
                values: values[r].view(),
                grid: if case["irregular"].as_bool().unwrap() {
                    FeatureGrid::Times(&times[r])
                } else {
                    FeatureGrid::Uniform {
                        start: 0.1,
                        step: 0.2,
                    }
                },
                mask: if case["masked"].as_bool().unwrap() {
                    Some(masks[r].as_slice())
                } else {
                    None
                },
            })
            .collect();
        let prepared = SampledFeatures::new(
            &runs,
            FeatureOptions {
                center: case["center"].as_bool().unwrap(),
                scale: if case["scale"] == "sd" {
                    FeatureScale::SampleSd
                } else {
                    FeatureScale::None
                },
            },
        )
        .unwrap();
        close(
            &prepared.amplitudes().to_owned(),
            &matrix(&case["amplitudes"]),
            2e-12,
        );
        let design = sampled_feature_design(
            &prepared,
            &["sound", "motion"],
            &frame,
            &hrf,
            ConvolutionMethod::Auto,
        )
        .unwrap();
        close(&design.data, &matrix(&case["expected"]), 2e-9);
        assert_eq!(
            design.column_names,
            ["sound.b1", "motion.b1", "sound.b2", "motion.b2"]
        );
        let expected_times: Vec<_> = (0..16)
            .map(|i| 0.15 + i as f64 * 0.7)
            .chain((0..13).map(|i| 11.2 + 0.2 + i as f64 * 1.1))
            .collect();
        for (a, b) in design.row_times.iter().zip(expected_times) {
            assert!((a - b).abs() < 1e-13);
        }
        let nuisance = DesignMatrix::with_times(
            Array2::ones((29, 1)),
            vec!["intercept".into()],
            design.row_times.clone(),
        );
        let combined = hstack_design_matrices(&[design, nuisance]).unwrap();
        let weights = Contrast::pair("sound-minus-motion", "sound.b2", "motion.b2")
            .compute_weights(&combined.column_names)
            .unwrap();
        assert_eq!(weights, array![0., 0., 1., -1., 0.]);
        let result = combined.data.dot(&weights);
        let reference = matrix(&case["expected"]);
        for i in 0..29 {
            assert!((result[i] - (reference[(i, 2)] - reference[(i, 3)])).abs() < 4e-9);
        }
        assert_eq!(prepared.as_events().onsets.len(), 11);
    }
}

#[test]
fn invalid_names_and_mutated_frames_are_checked() {
    let values = array![[1., 2.], [3., 4.]];
    let features = SampledFeatures::new(
        &[FeatureRun {
            values: values.view(),
            grid: FeatureGrid::Uniform {
                start: 0.,
                step: 0.5,
            },
            mask: None,
        }],
        FeatureOptions::default(),
    )
    .unwrap();
    let mut frame = SamplingFrame::uniform(12, 1., 0.1).unwrap();
    let hrf = SpmG2Hrf::new(5., 15., 1. / 120.);
    for names in [&["a"][..], &["a", "a"], &[" ", "b"]] {
        assert!(
            sampled_feature_design(&features, names, &frame, &hrf, ConvolutionMethod::Auto)
                .is_err()
        );
    }
    frame.tr.clear();
    assert!(sampled_feature_design(
        &features,
        &["a", "b"],
        &frame,
        &hrf,
        ConvolutionMethod::Auto
    )
    .is_err());
}

#[test]
fn single_basis_names_and_broadcast_frame_fields_are_preserved() {
    let values = array![[1.], [3.]];
    let run = FeatureRun {
        values: values.view(),
        grid: FeatureGrid::Uniform {
            start: 0.,
            step: 0.5,
        },
        mask: None,
    };
    let features = SampledFeatures::new(&[run, run], FeatureOptions::default()).unwrap();
    let mut frame = SamplingFrame::new(vec![12, 10], vec![1.], Some(vec![0.]), 0.1).unwrap();
    // Public fields may contain constructor-supported broadcast values.
    frame.tr = vec![1.];
    frame.start_time = vec![0.];
    let hrf = fmrihrf::hrf::SpmG1Hrf::new(5., 15., 1. / 120.);
    let design = sampled_feature_design(
        &features,
        &["sound.energy"],
        &frame,
        &hrf,
        ConvolutionMethod::Auto,
    )
    .unwrap();
    assert_eq!(design.column_names, ["sound.energy"]);
    assert_eq!(
        design.row_times.to_vec(),
        (0..22).map(|i| i as f64).collect::<Vec<_>>()
    );
}
