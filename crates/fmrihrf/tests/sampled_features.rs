use fmrihrf::{hrf::SpmG2Hrf, regressor::*, Hrf, SamplingFrame};
use ndarray::{array, s, Array1, Array2};
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
    let f: Value = serde_json::from_str(include_str!("fixtures/sampled_features.json")).unwrap();
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
        close(
            &prepared
                .evaluate(&frame, &hrf, ConvolutionMethod::Auto)
                .unwrap(),
            &matrix(&case["expected"]),
            2e-9,
        );
        assert_eq!(prepared.as_events().onsets.len(), 11);
    }
}
#[derive(Debug)]
struct Ramp;
impl Hrf for Ramp {
    fn evaluate(&self, t: &Array1<f64>) -> fmrihrf::Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((t.len(), 1), |(i, _)| {
            if (0.0..=10.).contains(&t[i]) {
                t[i]
            } else {
                0.
            }
        }))
    }
    fn name(&self) -> &str {
        "ramp"
    }
    fn nbasis(&self) -> usize {
        1
    }
    fn span(&self) -> f64 {
        10.
    }
    fn params(&self) -> Vec<(&str, f64)> {
        vec![]
    }
}
#[test]
fn zoh_mass_is_invariant_to_resampling_and_not_impulse_count() {
    let frame = SamplingFrame::new(vec![9], vec![0.25], Some(vec![0.]), 0.01).unwrap();
    for n in [5, 10, 20] {
        let values = Array2::from_elem((n, 1), 2.);
        let prepared = SampledFeatures::new(
            &[FeatureRun {
                values: values.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 1. / n as f64,
                },
                mask: None,
            }],
            FeatureOptions {
                center: false,
                scale: FeatureScale::None,
            },
        )
        .unwrap();
        let actual = prepared
            .evaluate(&frame, &Ramp, ConvolutionMethod::Auto)
            .unwrap();
        for i in 0..9 {
            let t = i as f64 * 0.25;
            let expected = if t <= 1. { t * t } else { 2. * t - 1. };
            assert!(
                (actual[(i, 0)] - expected).abs() < 1e-10,
                "n={n} i={i} actual={} expected={expected}",
                actual[(i, 0)]
            );
        }
    }
}
#[test]
fn mask_centering_identity_and_run_isolation() {
    let values = array![[1.], [3.], [20.], [5.]];
    let mask = [true, true, false, true];
    let run = FeatureRun {
        values: values.view(),
        grid: FeatureGrid::Uniform {
            start: 0.,
            step: 0.5,
        },
        mask: Some(&mask),
    };
    let raw = SampledFeatures::new(
        &[run],
        FeatureOptions {
            center: false,
            scale: FeatureScale::None,
        },
    )
    .unwrap();
    let centered = SampledFeatures::new(&[run], FeatureOptions::default()).unwrap();
    let presence = Array2::ones((4, 1));
    let on = SampledFeatures::new(
        &[FeatureRun {
            values: presence.view(),
            ..run
        }],
        FeatureOptions {
            center: false,
            scale: FeatureScale::None,
        },
    )
    .unwrap();
    let frame = SamplingFrame::uniform(20, 0.25, 0.025).unwrap();
    let y = raw
        .evaluate(&frame, &Ramp, ConvolutionMethod::Auto)
        .unwrap()
        - on.evaluate(&frame, &Ramp, ConvolutionMethod::Auto).unwrap() * 3.;
    close(
        &centered
            .evaluate(&frame, &Ramp, ConvolutionMethod::Auto)
            .unwrap(),
        &y,
        1e-10,
    );
    let zero = Array2::zeros((4, 1));
    let all = SampledFeatures::new(
        &[
            run,
            FeatureRun {
                values: zero.view(),
                mask: None,
                ..run
            },
        ],
        FeatureOptions {
            center: false,
            scale: FeatureScale::None,
        },
    )
    .unwrap();
    let frame2 = SamplingFrame::new(vec![20, 20], vec![0.25], None, 0.025).unwrap();
    let response = all
        .evaluate(&frame2, &Ramp, ConvolutionMethod::Auto)
        .unwrap();
    assert!(response.slice(s![20.., ..]).iter().all(|v| *v == 0.));
    close(
        &response.slice(s![..20, ..]).to_owned(),
        &raw.evaluate(&frame, &Ramp, ConvolutionMethod::Auto)
            .unwrap(),
        1e-12,
    );
}
#[test]
fn invalid_inputs_strides_and_scale_extremes() {
    let data = array![[1., 2., 3., 4.], [3., 4., 7., 8.], [5., 6., 11., 12.]];
    let run = FeatureRun {
        values: data.slice(s![..,..;2]),
        grid: FeatureGrid::Uniform {
            start: 0.,
            step: 0.5,
        },
        mask: None,
    };
    let opt = FeatureOptions {
        center: true,
        scale: FeatureScale::SampleSd,
    };
    let prepared = SampledFeatures::new(&[run], opt).unwrap();
    close(
        &prepared.amplitudes().to_owned(),
        &array![[-1., -1.], [0., 0.], [1., 1.]],
        1e-12,
    );
    for scale in [1e-200, 1e200] {
        let scaled = &data * scale;
        let p = SampledFeatures::new(
            &[FeatureRun {
                values: scaled.slice(s![..,..;2]),
                ..run
            }],
            opt,
        )
        .unwrap();
        close(
            &p.amplitudes().to_owned(),
            &prepared.amplitudes().to_owned(),
            1e-12,
        );
    }
    for times in [&[0., 0., 1.][..], &[0., f64::NAN, 1.][..], &[0., 1.][..]] {
        assert!(SampledFeatures::new(
            &[FeatureRun {
                grid: FeatureGrid::Times(times),
                ..run
            }],
            opt
        )
        .is_err());
    }
    assert!(SampledFeatures::new(
        &[FeatureRun {
            mask: Some(&[false; 3]),
            ..run
        }],
        opt
    )
    .is_err());
    assert!(SampledFeatures::new(
        &[FeatureRun {
            grid: FeatureGrid::Uniform {
                start: 0.,
                step: 0.
            },
            ..run
        }],
        opt
    )
    .is_err());
    let constant = Array2::ones((3, 2));
    assert!(SampledFeatures::new(
        &[FeatureRun {
            values: constant.view(),
            ..run
        }],
        opt
    )
    .is_err());
    let frame = SamplingFrame::uniform(4, 1., 1.).unwrap();
    assert!(prepared
        .evaluate(&frame, &Ramp, ConvolutionMethod::Auto)
        .is_err());
}
