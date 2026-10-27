use approx::assert_abs_diff_eq;
use fmri_hrf::hrf::{Hrf, HrfExt};
use fmri_hrf::regressor::{convolve_hrf, Regressor};
use ndarray::{array, Array1, Array2};
use std::sync::Arc;

#[derive(Debug)]
struct Ramp;
impl Hrf for Ramp {
    fn evaluate(&self, t: &Array1<f64>) -> fmri_hrf::Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((t.len(), 2), |(i, b)| {
            if (0.0..=4.0).contains(&t[i]) {
                t[i] * (b + 1) as f64
            } else {
                0.0
            }
        }))
    }
    fn name(&self) -> &str {
        "analytic_ramp"
    }
    fn nbasis(&self) -> usize {
        2
    }
    fn span(&self) -> f64 {
        4.0
    }
    fn params(&self) -> Vec<(&str, f64)> {
        vec![]
    }
}

#[test]
fn blocks_integrate_even_below_precision_and_average_by_duration() {
    for dt in [0.33, 0.1, 0.025] {
        for duration in [0.15, 0.7] {
            for summate in [true, false] {
                let got = convolve_hrf(
                    &array![2.0],
                    &[0.13],
                    &[duration],
                    &[1.5],
                    &Ramp,
                    4.0,
                    dt,
                    "conv",
                    summate,
                )
                .unwrap();
                let expected = 1.5 * (1.87 * duration - duration * duration / 2.0)
                    / if summate { 1.0 } else { duration };
                for b in 0..2 {
                    assert_abs_diff_eq!(got[[0, b]], expected * (b + 1) as f64, epsilon = 1e-12);
                }
            }
        }
    }
}

#[test]
fn off_grid_impulses_preserve_exact_time_and_signed_amplitude() {
    for method in ["conv", "direct", "fft", "loop"] {
        let got = convolve_hrf(
            &array![1.5, 2.0, 2.0, 3.1],
            &[0.13],
            &[0.0],
            &[-2.0],
            &Ramp,
            4.0,
            0.33,
            method,
            true,
        )
        .unwrap();
        for (i, t) in [1.5, 2.0, 2.0, 3.1].into_iter().enumerate() {
            assert_abs_diff_eq!(got[[i, 0]], -2.0 * (t - 0.13), epsilon = 1e-12);
            assert_abs_diff_eq!(got[[i, 1]], -4.0 * (t - 0.13), epsilon = 1e-12);
        }
    }
}

#[test]
fn block_evaluation_uses_trapezoid_and_duration_mean() {
    for d in [0.15, 0.7] {
        for summate in [true, false] {
            let got = Ramp
                .evaluate_block(&array![2.0, 3.0], d, 0.33, summate, false)
                .unwrap();
            for (i, t) in [2.0, 3.0].into_iter().enumerate() {
                assert_abs_diff_eq!(
                    got[[i, 0]],
                    (t * d - d * d / 2.0) / if summate { 1.0 } else { d },
                    epsilon = 1e-12
                );
            }
        }
    }
}

#[test]
fn long_blocks_starting_before_window_keep_their_tails() {
    let reg = Regressor::new(
        vec![0.0],
        Box::new(Ramp),
        Some(vec![100.0]),
        None,
        None,
        None,
    )
    .unwrap();
    let times = array![99.0, 100.0, 102.0, 104.5];
    let got = reg.evaluate(&times, Some(0.01), None).unwrap();
    // The sampled kernel includes t=4, so its trapezoid error at this jump is O(dt).
    for (i, expected) in [8.0, 8.0, 6.0, 0.0].into_iter().enumerate() {
        assert_abs_diff_eq!(got[[i, 0]], expected, epsilon = 0.05);
    }
}

#[test]
fn empty_trial_varying_events_and_explicit_span_are_valid() {
    let reg = Regressor::new_trial_varying(
        vec![],
        vec![Arc::new(Box::new(Ramp))],
        None,
        None,
        Some(40.0),
        None,
    )
    .unwrap();
    assert_eq!(reg.span, 40.0);
    assert_eq!(
        reg.evaluate(&array![0.0, 1.0], None, None).unwrap(),
        Array2::<f64>::zeros((2, 2))
    );
    let reg = Regressor::new(vec![0.0], Box::new(Ramp), None, None, Some(40.0), None).unwrap();
    assert_eq!(reg.span, 40.0);
}

#[test]
fn invalid_convolution_inputs_return_errors() {
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(convolve_hrf(
            &array![1.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &Ramp,
            4.0,
            dt,
            "conv",
            true
        )
        .is_err());
    }
    assert!(convolve_hrf(
        &array![1.0],
        &[0.0, 1.0],
        &[0.0],
        &[1.0],
        &Ramp,
        4.0,
        0.1,
        "conv",
        true
    )
    .is_err());
    assert!(convolve_hrf(
        &array![1.0],
        &[0.0],
        &[0.0],
        &[1.0],
        &Ramp,
        4.0,
        0.1,
        "typo",
        true
    )
    .is_err());
    assert!(convolve_hrf(
        &array![f64::INFINITY],
        &[0.0],
        &[0.0],
        &[1.0],
        &Ramp,
        4.0,
        0.1,
        "conv",
        true
    )
    .is_err());
}

#[test]
fn current_r_conv_and_reference_loop_fixtures() {
    use fmri_hrf::hrf::{spm_canonical, spm_hrf_with_derivatives};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/convolution_contract.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        // auto_unbox affects vectors of length one in R JSON.
        let vector = |key: &str| -> Vec<f64> {
            let x = &case[key];
            if let Some(a) = x.as_array() {
                a.iter().map(|v| v.as_f64().unwrap()).collect()
            } else {
                vec![x.as_f64().unwrap()]
            }
        };
        let hrf = if case["basis"] == "spmg1" {
            spm_canonical(None, None, None, None, None, Some(false))
        } else {
            spm_hrf_with_derivatives()
        };
        let reg = Regressor::new(
            vector("onsets"),
            hrf,
            Some(vector("durations")),
            Some(vector("amplitudes")),
            case["span"].as_f64(),
            case["summate"].as_bool(),
        )
        .unwrap();
        let got = reg
            .evaluate(
                &Array1::from(vector("grid")),
                case["precision"].as_f64(),
                case["method"].as_str(),
            )
            .unwrap();
        let rows = case["expected"].as_array().unwrap();
        assert_eq!(
            got.dim(),
            (rows.len(), case["ncols"].as_u64().unwrap() as usize)
        );
        for (i, row) in rows.iter().enumerate() {
            for (b, value) in row.as_array().unwrap().iter().enumerate() {
                let expected = value.as_f64().unwrap();
                let delta = (got[[i, b]] - expected).abs();
                assert!(
                    delta <= 2e-10 * (1.0 + expected.abs()),
                    "{} {} {} dt={} summate={} [{i},{b}]: {} vs {expected} (delta {delta})",
                    case["name"],
                    case["basis"],
                    case["method"],
                    case["precision"],
                    case["summate"],
                    got[[i, b]]
                );
            }
        }
    }
}

#[test]
fn condition_sets_preserve_custom_hrf_instead_of_reconstructing_by_name() {
    let set = fmri_hrf::regressor::regressor_set(
        vec![0.0, 0.0],
        vec!["A".into(), "B".into()],
        Some(Box::new(Ramp)),
        None,
        Some(vec![2.0, 3.0]),
        None,
        None,
    )
    .unwrap();
    assert_eq!(set.regressors.len(), 2);
    assert!(Arc::ptr_eq(&set.regressors[0].hrf, &set.regressors[1].hrf));
    for (r, scale) in set.regressors.iter().zip([2.0, 3.0]) {
        assert_eq!(r.hrf.nbasis(), 2);
        assert_eq!(r.span, 4.0);
        assert_eq!(
            r.evaluate(&array![1.0, 2.0], Some(0.1), Some("loop"))
                .unwrap(),
            array![[scale, 2.0 * scale], [2.0 * scale, 4.0 * scale]]
        );
    }
}

#[test]
fn causal_hrfs_and_fixed_normalization_are_invariant_to_query_partition() {
    use fmri_hrf::hrf::*;
    let hrfs = vec![
        gamma_hrf(6.0, 1.0, Some(true)),
        gaussian_hrf(6.0, 2.0, Some(true)),
        mexhat_hrf(2.0, 6.0),
        inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0),
        lwu_hrf(None, None, None, Some("height")).unwrap(),
        make_hrf("gamma(shape=6,rate=1,normalize=height)").unwrap(),
    ];
    let times = array![-2.0, -0.1, 0.0, 2.31, 5.0, 8.23];
    for hrf in hrfs {
        let whole = hrf.evaluate(&times).unwrap();
        assert_eq!(whole[[0, 0]], 0.0, "{}", hrf.name());
        assert_eq!(whole[[1, 0]], 0.0, "{}", hrf.name());
        for (i, &t) in times.iter().enumerate() {
            assert_abs_diff_eq!(
                hrf.evaluate(&array![t]).unwrap()[[0, 0]],
                whole[[i, 0]],
                epsilon = 1e-14
            );
        }
    }
    assert!(make_hrf("gamma(normalize=typo)").is_err());
}

#[test]
fn registered_factories_can_use_the_registry_without_holding_its_lock() {
    use fmri_hrf::hrf::{get_hrf, register_hrf};
    register_hrf(
        "nested_contract",
        || {
            register_hrf("nested_leaf", || Box::new(Ramp), true).unwrap();
            get_hrf("nested_leaf").unwrap()
        },
        true,
    )
    .unwrap();
    assert_eq!(get_hrf("nested_contract").unwrap().nbasis(), 2);
    assert_eq!(get_hrf("spmg2").unwrap().nbasis(), 2);
    assert_eq!(get_hrf("spmg3").unwrap().nbasis(), 3);
}

#[test]
fn fixed_normalization_modes_match_r_and_preserve_relative_derivative_scales() {
    use fmri_hrf::hrf::{normalize_hrf_with, spm_hrf_with_derivatives, Normalization};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/convolution_contract.json")).unwrap();
    for case in fixture["normalization_cases"].as_array().unwrap() {
        let mode = match case["mode"].as_str().unwrap() {
            "none" => Normalization::None,
            "spm" => Normalization::Spm,
            "unit_peak" => Normalization::UnitPeak,
            "unit_integral" => Normalization::UnitIntegral,
            _ => Normalization::UnitPeakPerBasis,
        };
        let hrf = normalize_hrf_with(spm_hrf_with_derivatives(), mode).unwrap();
        let times = Array1::from_iter(
            case["time"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap()),
        );
        let values = hrf.evaluate(&times).unwrap();
        for (i, row) in case["expected"].as_array().unwrap().iter().enumerate() {
            for (j, v) in row.as_array().unwrap().iter().enumerate() {
                assert_abs_diff_eq!(values[[i, j]], v.as_f64().unwrap(), epsilon = 1e-12);
                assert_abs_diff_eq!(
                    hrf.evaluate(&array![times[i]]).unwrap()[[0, j]],
                    values[[i, j]],
                    epsilon = 1e-14
                );
            }
        }
    }
}

#[test]
fn positive_semidefinite_validation_uses_the_actual_matrix() {
    use fmri_hrf::penalty::is_positive_semidefinite;
    let gram = array![[5.0, 2.0, -1.0], [2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]];
    assert!(is_positive_semidefinite(&gram));
    assert!(is_positive_semidefinite(&gram.t().to_owned()));
    assert!(!is_positive_semidefinite(&array![[1.0, 2.0], [2.0, 1.0]]));
    assert!(!is_positive_semidefinite(&array![[1.0, 2.0], [0.0, 1.0]]));
    assert!(!is_positive_semidefinite(&array![[f64::NAN]]));
}

#[test]
fn normalization_reference_count_matches_r_ties_to_even_rounding() {
    use fmri_hrf::hrf::{normalize_hrf_with, Normalization, SimpleHrf};
    let hrf = SimpleHrf::new("square", 1, 17.21, vec![], |t, _| {
        Ok(t.mapv(|x| x * x).insert_axis(ndarray::Axis(1)))
    });
    let normalized = normalize_hrf_with(Box::new(hrf), Normalization::UnitIntegral).unwrap();
    // Exact composite-trapezoid integral of x^2 with 860 intervals.
    let integral = 17.21_f64.powi(3) * (1.0 / 3.0 + 1.0 / (6.0 * 860.0_f64.powi(2)));
    assert_abs_diff_eq!(
        normalized.evaluate(&array![2.0]).unwrap()[[0, 0]],
        4.0 / integral,
        epsilon = 1e-15
    );
}
