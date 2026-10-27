use fmri_noise::{
    selected_estimation::SelectedAr1Estimator, GapPolicy, ObservationSelection, Pooling,
    WhiteningPlan,
};
use fmri_reg::{
    selected_estimation::{estimate_selected_ar1, SelectedAr1Options},
    GlsModel, LinearModel,
};
use ndarray::{array, s, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/selected_estimation.json")).unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn matrix(v: &Value) -> Array2<f64> {
    let r = v.as_array().unwrap();
    Array2::from_shape_fn((r.len(), r[0].as_array().unwrap().len()), |(i, j)| {
        r[i][j].as_f64().unwrap()
    })
}
fn selection() -> ObservationSelection {
    ObservationSelection::new(
        vec![36, 31],
        fixture()["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect(),
    )
    .unwrap()
}
fn close(a: impl IntoIterator<Item = f64>, b: impl IntoIterator<Item = f64>, tol: f64) {
    let a: Vec<_> = a.into_iter().collect();
    let b: Vec<_> = b.into_iter().collect();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < tol * (1. + b.abs()), "{a} != {b}");
    }
}
#[test]
fn selected_correction_and_fgls_match_independent_dense_r() {
    let f = fixture();
    let x = matrix(&f["x"]);
    let selected = selection();
    let xs = selected.select(x.view()).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let gaps = if case["policy"] == "preserve" {
            GapPolicy::Preserve
        } else {
            GapPolicy::Restart
        };
        let pooling = if case["pool"] == "global" {
            Pooling::Global
        } else {
            Pooling::Run
        };
        let estimator =
            SelectedAr1Estimator::new(xs.view(), selected.clone(), gaps, pooling, 5, 1e-12)
                .unwrap();
        close(
            estimator.bias_matrix().iter().copied(),
            matrix(&case["bias"]).iter().copied(),
            2e-11,
        );
        let (plan, d) = estimator.fit(matrix(&case["residual"]).view()).unwrap();
        close(
            d.coefficients.iter().copied(),
            vector(&case["coefficients"]),
            2e-10,
        );
        close(
            d.corrected_acvf
                .iter()
                .flatten()
                .map(|g| g * d.residual_scale * d.residual_scale),
            vector(&case["gamma"]),
            2e-10,
        );
        close(
            d.pair_counts.iter().flatten().map(|&n| n as f64),
            vector(&case["pair_counts"]),
            0.1e-12,
        );
        let y = matrix(&case["y"]);
        let fit = estimate_selected_ar1(
            x.view(),
            y.view(),
            SelectedAr1Options {
                selection: selected.clone(),
                gaps,
                pooling,
                max_lag: 5,
                relative_tolerance: 1e-12,
            },
        )
        .unwrap();
        close(
            fit.diagnostics.coefficients.iter().copied(),
            d.coefficients.iter().copied(),
            2e-10,
        );
        let ys = selected.select(y.view()).unwrap();
        let fitted = fit.model.fit(ys.view()).unwrap();
        close(
            fitted.coefficients().iter().copied(),
            matrix(&case["beta"]).iter().copied(),
            2e-9,
        );
        close(
            fitted.residual_variance().iter().copied(),
            vector(&case["variance"]),
            2e-9,
        );
        close(
            fitted
                .t_contrast(array![0., 0., 1.].view())
                .unwrap()
                .statistic
                .iter()
                .copied(),
            vector(&case["t"]),
            2e-9,
        );
        for scale in [1e-150, 1e150] {
            let (_, other) = estimator
                .fit((matrix(&case["residual"]) * scale).view())
                .unwrap();
            close(other.coefficients, d.coefficients.iter().copied(), 2e-10);
        }
        assert!(plan.apply(ys.view()).is_ok());
    }
}
#[test]
fn unsupported_geometry_bad_residuals_and_singular_correction_error() {
    let sparse_selection =
        ObservationSelection::new(vec![20], (0..20).step_by(2).collect()).unwrap();
    let x = Array2::ones((10, 1));
    assert!(SelectedAr1Estimator::new(
        x.view(),
        sparse_selection,
        GapPolicy::Preserve,
        Pooling::Global,
        2,
        1e-12
    )
    .is_err());
    let f = fixture();
    let selection = selection();
    let x = selection.select(matrix(&f["x"]).view()).unwrap();
    let e = SelectedAr1Estimator::new(
        x.view(),
        selection.clone(),
        GapPolicy::Preserve,
        Pooling::Run,
        5,
        1e-12,
    )
    .unwrap();
    assert!(e.fit(Array2::zeros((x.nrows(), 1)).view()).is_err());
    assert!(e.fit(Array2::ones((x.nrows(), 1)).view()).is_err());
    let mut residual = matrix(&f["cases"][0]["residual"]);
    residual[(0, 0)] = f64::NAN;
    assert!(e.fit(residual.view()).is_err());
    assert!(SelectedAr1Estimator::new(
        x.view(),
        selection,
        GapPolicy::Preserve,
        Pooling::Run,
        50,
        1e-12
    )
    .is_err());
    let absent = ObservationSelection::new(vec![10, 10], (10..20).collect()).unwrap();
    assert!(SelectedAr1Estimator::new(
        Array2::ones((10, 1)).view(),
        absent.clone(),
        GapPolicy::Restart,
        Pooling::Run,
        2,
        1e-12
    )
    .is_err());
    assert!(SelectedAr1Estimator::new(
        Array2::ones((10, 1)).view(),
        absent,
        GapPolicy::Restart,
        Pooling::Global,
        2,
        1e-12
    )
    .is_ok());
    let full = ObservationSelection::new(vec![8], (0..8).collect()).unwrap();
    assert!(SelectedAr1Estimator::new(
        Array2::eye(8).view(),
        full,
        GapPolicy::Preserve,
        Pooling::Global,
        2,
        1e-12
    )
    .is_err());
}
// Fixed deterministic generator: simulations are reproducible without an external RNG.
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / 9007199254740992.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (2. * std::f64::consts::PI * self.uniform()).cos()
    }
}
#[test]
fn stationary_simulations_check_recovery_whitening_and_conditional_coverage() {
    let lengths = vec![120, 100];
    let n = 220;
    let keep: Vec<_> = (0..n)
        .filter(|i| i % 13 != 0 && i % 13 != 1 && *i != 119)
        .collect();
    let selected = ObservationSelection::new(lengths, keep.clone()).unwrap();
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => {
            if i < 120 {
                1.
            } else {
                0.
            }
        }
        1 => {
            if i >= 120 {
                1.
            } else {
                0.
            }
        }
        _ => ((i as f64) * 0.19).sin(),
    });
    let xs = selected.select(x.view()).unwrap();
    let ols = LinearModel::new(xs.view(), 1e-12).unwrap();
    for (scenario, (gaps, pooling)) in [
        (GapPolicy::Preserve, Pooling::Global),
        (GapPolicy::Restart, Pooling::Global),
        (GapPolicy::Preserve, Pooling::Run),
        (GapPolicy::Restart, Pooling::Run),
    ]
    .into_iter()
    .enumerate()
    {
        let phi: [f64; 2] = if matches!(pooling, Pooling::Global) {
            [0.4, 0.4]
        } else {
            [0.45, -0.25]
        };
        let e = SelectedAr1Estimator::new(xs.view(), selected.clone(), gaps, pooling, 10, 1e-12)
            .unwrap();
        let mut rng = Rng(71931 + scenario as u64);
        let reps = 160;
        let mut bias = [0.; 2];
        let mut covered = 0;
        let mut corr = 0.;
        let mut failures = 0;
        for _ in 0..reps {
            let mut errors = Array2::zeros((n, 17));
            for j in 0..17 {
                let mut previous = 0.;
                for i in 0..n {
                    let r = usize::from(i >= 120);
                    let reset = i == 0
                        || i == 120
                        || (matches!(gaps, GapPolicy::Restart)
                            && keep.contains(&i)
                            && !keep.contains(&(i - 1)));
                    previous = if reset {
                        rng.normal() / (1. - phi[r] * phi[r]).sqrt()
                    } else {
                        phi[r] * previous + rng.normal()
                    };
                    errors[(i, j)] = previous;
                }
            }
            let errors = selected.select(errors.view()).unwrap();
            let noise = errors.slice(s![.., ..16]);
            let initial = ols.fit(noise).unwrap();
            let residual = &noise - &xs.dot(initial.coefficients());
            let Ok((plan, d)) = e.fit(residual.view()) else {
                failures += 1;
                continue;
            };
            for r in 0..2 {
                bias[r] += d.coefficients[r] - phi[r];
            }
            let model = GlsModel::new(xs.view(), plan.clone(), 1e-12).unwrap();
            // Independent held-out target: calibration here does not assume the same
            // feature was used to estimate its own AR coefficient.
            let target = errors.slice(s![.., 16..17]).to_owned()
                + xs.column(2).to_owned().insert_axis(ndarray::Axis(1)) * 0.7;
            let fit = model.fit(target.view()).unwrap();
            let t = fit.t_contrast(array![0., 0., 1.].view()).unwrap();
            let critical = statrs::distribution::ContinuousCDF::inverse_cdf(
                &statrs::distribution::StudentsT::new(0., 1., t.df as f64).unwrap(),
                0.975,
            );
            if (t.effect[0] - 0.7).abs() <= critical * t.standard_error[0] {
                covered += 1;
            }
            let wr = plan
                .apply((&target - &xs.dot(fit.coefficients())).view())
                .unwrap();
            let numerator = (1..wr.nrows())
                .map(|i| wr[(i, 0)] * wr[(i - 1, 0)])
                .sum::<f64>();
            let denom = wr.iter().map(|v| v * v).sum::<f64>();
            corr += numerator / denom;
        }
        let success = reps - failures;
        let coverage = covered as f64 / success as f64;
        let mean_corr = corr / success as f64;
        let b = bias.map(|v| v / success as f64);
        println!("SIM scenario={scenario} reps={reps} failures={failures} bias={b:?} residual_lag1={mean_corr:.6} coverage={coverage:.6}");
        assert_eq!(failures, 0);
        assert!(b.iter().all(|b| b.abs() < 0.07));
        assert!(mean_corr.abs() < 0.08);
        assert!((0.87..=0.995).contains(&coverage));
    }
}

#[cfg(feature = "dataset")]
#[test]
fn estimated_plan_runs_through_id_bound_streaming() {
    use fmri_data::{
        Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
    };
    use fmri_reg::{
        dataset::{BlockOptions, BoundModel},
        selection::select_assay,
    };
    let f = fixture();
    let selection = selection();
    let x = matrix(&f["x"]);
    let ids: Vec<_> = (0..67)
        .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
        .collect();
    let features: Vec<_> = (0..64)
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    for case in f["cases"].as_array().unwrap() {
        let y = matrix(&case["y"]);
        let estimated = estimate_selected_ar1(
            x.view(),
            y.view(),
            SelectedAr1Options {
                selection: selection.clone(),
                gaps: if case["policy"] == "preserve" {
                    GapPolicy::Preserve
                } else {
                    GapPolicy::Restart
                },
                pooling: if case["pool"] == "global" {
                    Pooling::Global
                } else {
                    Pooling::Run
                },
                max_lag: 5,
                relative_tolerance: 1e-12,
            },
        )
        .unwrap();
        let assay = Assay::new(
            "bold",
            SourceSpec::memory(
                NumericBlock::new([67, 64], Values::F64(y.iter().copied().collect())).unwrap(),
            ),
            ids.clone(),
            features.clone(),
            AssayAnnotations::default(),
        )
        .unwrap();
        let view = select_assay(&assay, &ids, &selection).unwrap();
        let bound = BoundModel::new(&estimated.model, view.observation_ids().to_vec()).unwrap();
        let expected = matrix(&case["beta"]);
        let mut offset = 0;
        bound
            .fit_assay_with_options(
                &view,
                BlockOptions {
                    block_features: 7,
                    workers: 2,
                    working_bytes: 200_000,
                },
                |ids, fit| {
                    assert_eq!(ids, &features[offset..offset + ids.len()]);
                    close(
                        fit.coefficients().iter().copied(),
                        expected
                            .slice(s![.., offset..offset + ids.len()])
                            .iter()
                            .copied(),
                        2e-9,
                    );
                    offset += ids.len();
                    Ok(())
                },
            )
            .unwrap();
        assert_eq!(offset, 64);
    }
}

#[test]
fn nonstationary_corrected_estimates_are_errors_not_clipped() {
    let f = fixture();
    assert!(f["adverse"]["coefficient"].as_f64().unwrap() > 1.);
    let selected = selection();
    let xs = selected.select(matrix(&f["x"]).view()).unwrap();
    let e = SelectedAr1Estimator::new(
        xs.view(),
        selected,
        GapPolicy::Preserve,
        Pooling::Global,
        5,
        1e-12,
    )
    .unwrap();
    assert!(matches!(
        e.fit(matrix(&f["adverse"]["residual"]).view()),
        Err(fmri_noise::NoiseError::Invalid(
            "nonstationary corrected selected AR1 estimate; no clipping or fallback"
        ))
    ));
}
