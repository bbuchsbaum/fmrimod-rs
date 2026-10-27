use fmri_noise::{
    selected_estimation::{SelectedAr1Estimator, SelectedArEstimator},
    GapPolicy, ObservationSelection, Pooling, WhiteningPlan,
};
use fmrireg::{
    selected_estimation::{estimate_selected_ar, SelectedArOptions},
    GlsModel, LinearModel,
};
use ndarray::{array, s, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/selected_arp_estimation.json")).unwrap()
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
fn choices(case: &Value) -> (GapPolicy, Pooling) {
    (
        if case["policy"] == "preserve" {
            GapPolicy::Preserve
        } else {
            GapPolicy::Restart
        },
        if case["pool"] == "global" {
            Pooling::Global
        } else {
            Pooling::Run
        },
    )
}
fn estimator(
    x: &Array2<f64>,
    selected: ObservationSelection,
    gaps: GapPolicy,
    pooling: Pooling,
    lag: usize,
    order: usize,
) -> SelectedArEstimator {
    SelectedArEstimator::new(
        SelectedAr1Estimator::new(x.view(), selected, gaps, pooling, lag, 1e-12).unwrap(),
        order,
    )
    .unwrap()
}
#[test]
fn independent_r_orders_two_to_four_and_diagnostics() {
    let f = fixture();
    let x = matrix(&f["x"]);
    let selected = selection();
    let xs = selected.select(x.view()).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let (gaps, pooling) = choices(case);
        let order = case["order"].as_u64().unwrap() as usize;
        let e = estimator(&xs, selected.clone(), gaps, pooling, 5, order);
        let residual = matrix(&case["residual"]);
        close(
            e.bias_matrix().iter().copied(),
            matrix(&case["bias"]).iter().copied(),
            2e-11,
        );
        let (_, d) = e.fit(residual.view()).unwrap();
        let expected: Vec<_> = case["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(vector)
            .collect();
        close(d.coefficients.iter().flatten().copied(), expected, 2e-10);
        close(
            d.corrected_acvf
                .iter()
                .flatten()
                .map(|g| g * d.residual_scale * d.residual_scale),
            vector(&case["gamma"]),
            2e-10,
        );
        close(
            d.yule_walker_condition.iter().copied(),
            vector(&case["conditions"]),
            2e-10,
        );
        close(
            d.innovation_variance_fraction.iter().copied(),
            vector(&case["innovation_fraction"]),
            2e-10,
        );
        assert!(d.stability_margin.iter().all(|m| *m > 0. && *m <= 1.));
        let y = matrix(&case["y"]);
        let estimated = estimate_selected_ar(
            x.view(),
            y.view(),
            SelectedArOptions {
                selection: selected.clone(),
                gaps,
                pooling,
                order,
                max_lag: 5,
                relative_tolerance: 1e-12,
            },
        )
        .unwrap();
        let ys = selected.select(y.view()).unwrap();
        let fit = estimated.model.fit(ys.view()).unwrap();
        close(
            fit.coefficients().iter().copied(),
            matrix(&case["beta"]).iter().copied(),
            2e-9,
        );
        close(
            fit.residual_variance().iter().copied(),
            vector(&case["variance"]),
            2e-9,
        );
        close(
            fit.t_contrast(array![0., 0., 1.].view())
                .unwrap()
                .statistic
                .iter()
                .copied(),
            vector(&case["t"]),
            2e-9,
        );
        for scale in [1e-150, 1e150] {
            let (_, scaled) = e.fit((&residual * scale).view()).unwrap();
            close(
                scaled.coefficients.iter().flatten().copied(),
                d.coefficients.iter().flatten().copied(),
                2e-10,
            );
        }
    }
}
#[test]
fn order_contracts_and_ar1_compatibility() {
    let f = fixture();
    let selected = selection();
    let x = selected.select(matrix(&f["x"]).view()).unwrap();
    for order in [0, 6] {
        let c = SelectedAr1Estimator::new(
            x.view(),
            selected.clone(),
            GapPolicy::Preserve,
            Pooling::Global,
            5,
            1e-12,
        )
        .unwrap();
        assert!(SelectedArEstimator::new(c, order).is_err());
    }
    let c = SelectedAr1Estimator::new(
        x.view(),
        selected.clone(),
        GapPolicy::Preserve,
        Pooling::Global,
        5,
        1e-12,
    )
    .unwrap();
    let r = matrix(&f["cases"][0]["residual"]);
    let (_, old) = c.fit(r.view()).unwrap();
    let (_, new) = SelectedArEstimator::new(c, 1)
        .unwrap()
        .fit(r.view())
        .unwrap();
    close(
        new.coefficients.iter().flatten().copied(),
        old.coefficients,
        2e-12,
    );
    let sparse = ObservationSelection::new(vec![20], (0..20).step_by(2).collect()).unwrap();
    assert!(SelectedAr1Estimator::new(
        Array2::ones((10, 1)).view(),
        sparse,
        GapPolicy::Preserve,
        Pooling::Global,
        4,
        1e-12
    )
    .is_err());
}
#[cfg(feature = "dataset")]
#[test]
fn estimated_plan_runs_through_id_bound_streaming() {
    use fmridataset::{
        Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
    };
    use fmrireg::{
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
        let estimated = estimate_selected_ar(
            x.view(),
            y.view(),
            SelectedArOptions {
                order: case["order"].as_u64().unwrap() as usize,
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
    fn step(&mut self, state: &mut [f64], phi: &[f64]) -> f64 {
        let value = state.iter().zip(phi).map(|(s, p)| s * p).sum::<f64>() + self.normal();
        state.rotate_right(1);
        state[0] = value;
        value
    }
}
#[test]
fn arp_simulations_recovery_whitening_and_independent_target_coverage() {
    let n = 220;
    let keep: Vec<_> = (0..n)
        .filter(|i| i % 13 != 0 && i % 13 != 1 && *i != 119)
        .collect();
    let selected = ObservationSelection::new(vec![120, 100], keep.clone()).unwrap();
    let mut retained = vec![false; n];
    for &i in &keep {
        retained[i] = true;
    }
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
    for order in 2..=4 {
        for (policy, (gaps, pooling)) in [
            (GapPolicy::Preserve, Pooling::Global),
            (GapPolicy::Restart, Pooling::Global),
            (GapPolicy::Preserve, Pooling::Run),
            (GapPolicy::Restart, Pooling::Run),
        ]
        .into_iter()
        .enumerate()
        {
            let base = match order {
                2 => vec![0.45, -0.18],
                3 => vec![0.4, -0.16, 0.08],
                _ => vec![0.4, -0.15, 0.08, -0.04],
            };
            let other = if matches!(pooling, Pooling::Global) {
                base.clone()
            } else {
                base.iter().map(|p| -0.7 * p).collect()
            };
            let phi = [base, other];
            let e = estimator(&xs, selected.clone(), gaps, pooling, 10, order);
            let reps = 100;
            let mut rng = Rng(92000 + 10 * order as u64 + policy as u64);
            let mut bias = vec![0.; 2 * order];
            let mut corr = vec![0.; order];
            let mut covered = 0;
            let mut failures = 0;
            for _ in 0..reps {
                let mut errors = Array2::zeros((n, 17));
                for j in 0..17 {
                    let mut state = vec![0.; order];
                    for i in 0..n {
                        let run = usize::from(i >= 120);
                        let reset = i == 0
                            || i == 120
                            || (matches!(gaps, GapPolicy::Restart)
                                && retained[i]
                                && !retained[i - 1]);
                        if reset {
                            state.fill(0.);
                            for _ in 0..256 {
                                rng.step(&mut state, &phi[run]);
                            }
                        }
                        errors[(i, j)] = rng.step(&mut state, &phi[run]);
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
                    for k in 0..order {
                        bias[r * order + k] += d.coefficients[r][k] - phi[r][k];
                    }
                }
                let model = GlsModel::new(xs.view(), plan.clone(), 1e-12).unwrap();
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
                let denom = wr.iter().map(|v| v * v).sum::<f64>();
                for lag in 1..=order {
                    corr[lag - 1] += (lag..wr.nrows())
                        .map(|i| wr[(i, 0)] * wr[(i - lag, 0)])
                        .sum::<f64>()
                        / denom;
                }
            }
            let successes = (reps - failures) as f64;
            for b in &mut bias {
                *b /= successes;
            }
            for c in &mut corr {
                *c /= successes;
            }
            let coverage = covered as f64 / successes;
            println!("ARP_SIM order={order} policy={policy} reps={reps} failures={failures} bias={bias:?} residual_lags={corr:?} coverage={coverage:.6}");
            assert_eq!(failures, 0);
            assert!(bias.iter().all(|b| b.abs() < 0.07));
            assert!(corr.iter().all(|c| c.abs() < 0.08));
            assert!((0.85..=0.995).contains(&coverage));
        }
    }
}
