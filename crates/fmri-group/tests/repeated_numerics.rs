use fmri_data::*;
use fmri_group::*;
use ndarray::{array, Array2};

pub(crate) fn prepared(
    y: Vec<f64>,
    visits: usize,
    x: Array2<f64>,
    s: Array2<f64>,
    model: RepeatedModel,
) -> fmri_group::Result<PreparedRepeatedModel> {
    let sizes = vec![visits; y.len() / visits];
    let blocks = vec![s; sizes.len()];
    prepared_blocks(y, &sizes, x, blocks, model)
}
fn prepared_blocks(
    y: Vec<f64>,
    sizes: &[usize],
    x: Array2<f64>,
    sampling: Vec<Array2<f64>>,
    model: RepeatedModel,
) -> fmri_group::Result<PreparedRepeatedModel> {
    let n = y.len();
    let ids = (0..n)
        .map(|i| ObservationId::new(format!("o{i}")).unwrap())
        .collect::<Vec<_>>();
    let fid = FeatureId::new("feature").unwrap();
    let fs = FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new("fixture", vec![fid.clone()]).unwrap(),
    ))
    .unwrap();
    let assay = Assay::new(
        "effect",
        SourceSpec::memory(NumericBlock::new([n, 1], Values::F64(y)).unwrap()),
        ids.clone(),
        vec![fid.clone()],
        AssayAnnotations::default(),
    )
    .unwrap();
    let frame = FmriFrame::new(
        Axis::new(ids.clone(), vec![]).unwrap(),
        fs,
        vec![assay],
        "effect",
        Metadata::new(),
    )
    .unwrap();
    let membership = sizes
        .iter()
        .enumerate()
        .flat_map(|(subject, &size)| (0..size).map(move |visit| (subject, visit)))
        .collect::<Vec<_>>();
    let subjects = (0..n)
        .map(|i| RepeatedSubject {
            observation: ids[i].clone(),
            dataset: "d".into(),
            participant: format!("s{:04}", membership[i].0),
            measurement: format!("visit{}", membership[i].1),
        })
        .collect();
    let cov = RepeatedFeatureCovariance {
        feature: fid,
        blocks: sizes
            .iter()
            .zip(sampling)
            .scan(0, |start, (&size, values)| {
                let observations = ids[*start..*start + size].to_vec();
                *start += size;
                Some(RepeatedCovarianceBlock {
                    observations,
                    values,
                })
            })
            .collect(),
    };
    let cohort = RepeatedCohort::new(
        frame,
        subjects,
        Estimand {
            id: "signal".into(),
            definition: "synthetic response".into(),
            units: "a.u.".into(),
        },
        Admission {
            target: "synthetic".into(),
            evidence: "explicit independent subject blocks".into(),
        },
        "effect".into(),
        vec![cov],
        CovarianceProvenance::Known {
            method: "simulation".into(),
        },
    )?;
    let cols = (0..x.ncols()).map(|j| format!("x{j}")).collect::<Vec<_>>();
    let design = GroupDesign::new(ids, cols.clone(), x)?;
    let contrasts = cols
        .iter()
        .map(|c| GroupContrast::named(c, &design, &[(c.as_str(), 1.)]))
        .collect::<fmri_group::Result<Vec<_>>>()?;
    PreparedRepeatedModel::new(cohort, design, contrasts, model)
}
fn ri(model: RepeatedModel, scale: f64, offset: f64) -> PreparedRepeatedModel {
    let y = [-1., 0., 1., 2.]
        .iter()
        .flat_map(|v| vec![v * scale + offset; 3])
        .collect();
    prepared(
        y,
        3,
        Array2::ones((12, 1)),
        Array2::eye(3) * (0.3 * scale * scale),
        model,
    )
    .unwrap()
}
#[test]
fn fitted_ml_reml_exact_boundary_and_variance_witness() {
    for (likelihood, tau, variance) in [
        (RepeatedLikelihood::Ml, 1.15, 0.3125),
        (RepeatedLikelihood::Reml, 47. / 30., 5. / 12.),
    ] {
        let plan = ri(
            RepeatedModel {
                likelihood,
                residual_variance: ResidualVariance::Estimate,
                ..Default::default()
            },
            1.,
            0.,
        );
        let r = plan.fit_feature(0).unwrap();
        assert_eq!(r.status, RepeatedStatus::Estimated, "{:?}", r.diagnostics);
        let e = r.estimates.unwrap();
        assert_eq!(e.eta, 0.);
        assert!((e.g[(0, 0)] - tau).abs() < 2e-5, "{} {tau}", e.g[(0, 0)]);
        assert!((e.beta[0] - 0.5).abs() < 1e-11);
        assert!((e.beta_covariance[(0, 0)] - variance).abs() < 6e-6);
        assert!(r.diagnostics.unwrap().eta_boundary);
    }
}
#[test]
fn fitted_response_units_and_large_intercept_offset_are_invariant() {
    for scale in [1e-8, 1., 1e8] {
        let p = ri(RepeatedModel::default(), scale, 1e8 * scale);
        let r = p.fit_feature(0).unwrap();
        assert_eq!(r.status, RepeatedStatus::Estimated, "{:?}", r.diagnostics);
        let e = r.estimates.unwrap();
        assert!((e.g[(0, 0)] / (scale * scale) - 47. / 30.).abs() < 3e-5);
        assert!(((e.beta[0] - 1e8 * scale) / scale - 0.5).abs() < 1e-7);
        assert!((e.beta_covariance[(0, 0)] / (scale * scale) - 5. / 12.).abs() < 1e-5);
    }
}
#[test]
fn exact_zero_random_covariance_and_singular_fixed_g_are_valid() {
    let times = (0..12).map(|i| (i % 3) as f64 - 1.).collect::<Vec<_>>();
    let x = Array2::from_shape_fn((12, 2), |(i, j)| if j == 0 { 1. } else { times[i] });
    let model = RepeatedModel {
        random_effects: RandomEffects::InterceptSlope,
        residual_variance: ResidualVariance::Estimate,
        time: Some(times.clone()),
        ..Default::default()
    };
    let p = prepared(
        times.iter().map(|t| 2. + 0.3 * t).collect(),
        3,
        x,
        Array2::eye(3) * 0.3,
        model,
    )
    .unwrap();
    let r = p.fit_feature(0).unwrap();
    assert_eq!(r.status, RepeatedStatus::Estimated, "{:?}", r.diagnostics);
    let e = r.estimates.unwrap();
    assert!(e.g.iter().all(|v| *v == 0.));
    assert_eq!(e.eta, 0.);
    let fixed = p.evaluate_fixed(0, array![[1., 1.], [1., 1.]], 0.).unwrap();
    assert_eq!(fixed.status, RepeatedStatus::Estimated);
    assert_eq!(fixed.diagnostics.unwrap().g_rank, 1);
}
#[test]
fn true_two_visit_confounding_and_reml_subject_intercepts_are_rejected() {
    let model = RepeatedModel {
        random_effects: RandomEffects::InterceptSlope,
        residual_variance: ResidualVariance::Estimate,
        time: Some((0..8).map(|i| if i % 2 == 0 { -1. } else { 1. }).collect()),
        ..Default::default()
    };
    assert!(prepared(
        (0..8).map(|i| (i as f64).sin()).collect(),
        2,
        Array2::ones((8, 1)),
        Array2::eye(2) * 0.3,
        model
    )
    .is_err());
    let x = Array2::from_shape_fn((12, 4), |(i, j)| f64::from(i / 3 == j));
    assert!(prepared(
        (0..12).map(|i| (i as f64).sin()).collect(),
        3,
        x,
        Array2::eye(3) * 0.3,
        RepeatedModel::default()
    )
    .is_err());
}
#[test]
fn limits_invalid_covariance_and_invalid_indices_never_return_estimates() {
    let p = ri(
        RepeatedModel {
            max_evaluations: 1,
            ..Default::default()
        },
        1.,
        0.,
    );
    let r = p.fit_feature(0).unwrap();
    assert_eq!(r.status, RepeatedStatus::EvaluationLimit);
    assert!(r.estimates.is_none());
    assert!(p.fit_feature(1).is_err());
    assert!(p
        .evaluate_fixed(0, array![[-1.]], 0.)
        .unwrap()
        .estimates
        .is_none());
    assert!(p
        .evaluate_fixed(0, array![[1.]], 0.2)
        .unwrap()
        .estimates
        .is_none());
    let mut s = Array2::eye(3);
    s[(0, 1)] = 0.2;
    assert!(prepared(
        vec![1.; 12],
        3,
        Array2::ones((12, 1)),
        s,
        RepeatedModel::default()
    )
    .is_err());
    let tiny = RepeatedModel {
        max_working_bytes: 1,
        ..Default::default()
    };
    assert!(matches!(
        prepared(vec![1.; 12], 3, Array2::ones((12, 1)), Array2::eye(3), tiny),
        Err(Error::Budget)
    ));
}
#[test]
fn correlated_slope_objective_matches_independent_r_optimizer() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/repeated_reference.json")).unwrap();
    let r = &f["interior_slope"];
    let vec = |name: &str| {
        r[name]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let y = vec("y");
    let t = vec("time");
    let s = array![[0.25, 0.04, 0.01], [0.04, 0.30, 0.03], [0.01, 0.03, 0.28]];
    let plan = prepared(
        y,
        3,
        Array2::ones((15, 1)),
        s,
        RepeatedModel {
            random_effects: RandomEffects::InterceptSlope,
            residual_variance: ResidualVariance::Estimate,
            likelihood: RepeatedLikelihood::Ml,
            time: Some(t),
            max_evaluations: 10000,
            ..Default::default()
        },
    )
    .unwrap();
    let fit = plan.fit_feature(0).unwrap();
    assert_eq!(
        fit.status,
        RepeatedStatus::Estimated,
        "{:?}",
        fit.diagnostics
    );
    let e = fit.estimates.unwrap();
    let expected = r["optimum"].as_f64().unwrap();
    assert!(
        (e.objective - expected).abs() < 2e-5,
        "Rust {} R {expected}; {:?}",
        e.objective,
        fit.diagnostics
    );
}

#[test]
fn analytic_covariance_gradients_match_central_differences_for_ml_and_reml() {
    let t = (0..12).map(|i| (i % 3) as f64 - 1.).collect::<Vec<_>>();
    let x = Array2::from_shape_fn((12, 2), |(i, j)| if j == 0 { 1. } else { t[i] });
    for likelihood in [RepeatedLikelihood::Ml, RepeatedLikelihood::Reml] {
        let plan = prepared(
            (0..12).map(|i| ((i + 1) as f64 * 0.7).sin()).collect(),
            3,
            x.clone(),
            array![[0.3, 0.03, 0.01], [0.03, 0.3, 0.02], [0.01, 0.02, 0.3]],
            RepeatedModel {
                random_effects: RandomEffects::InterceptSlope,
                residual_variance: ResidualVariance::Estimate,
                likelihood,
                time: Some(t.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        let par = [0.5, 0.1 * std::f64::consts::SQRT_2, 0.2, 0.15];
        let eval = |p: &[f64; 4]| {
            plan.evaluate_fixed(
                0,
                array![
                    [p[0], p[1] / std::f64::consts::SQRT_2],
                    [p[1] / std::f64::consts::SQRT_2, p[2]]
                ],
                p[3],
            )
            .unwrap()
        };
        let d = eval(&par).diagnostics.unwrap();
        for k in 0..4 {
            let h = 1e-5;
            let mut plus = par;
            let mut minus = par;
            plus[k] += h;
            minus[k] -= h;
            let fd = (eval(&plus).estimates.unwrap().objective
                - eval(&minus).estimates.unwrap().objective)
                / (2. * h);
            let analytic = d.covariance_gradient[k] / d.response_variance_scale;
            assert!(
                (fd - analytic).abs() < 2e-7 * (1. + analytic.abs()),
                "{likelihood:?} k{k}: {fd} {analytic}"
            );
        }
    }
}
#[test]
fn huge_negative_random_variance_is_rejected_even_when_total_covariance_is_spd() {
    let p = prepared(
        vec![1., 2., 3., 2., 4., 1.],
        3,
        Array2::ones((6, 1)),
        Array2::eye(3) * 1e200,
        RepeatedModel::default(),
    )
    .unwrap();
    let r = p.evaluate_fixed(0, array![[-1e199]], 0.).unwrap();
    assert_eq!(r.status, RepeatedStatus::InvalidCovariance);
    assert!(r.estimates.is_none());
}

#[test]
fn high_heterogeneity_fit_is_invariant_to_fixed_design_units() {
    for amplitude in [100., 10000.] {
        let expected = 5. * amplitude * amplitude / 3. - 0.1;
        for design_scale in [1., 1e150] {
            let y = [-1., 0., 1., 2.]
                .iter()
                .flat_map(|v| vec![v * amplitude; 3])
                .collect();
            let p = prepared(
                y,
                3,
                Array2::ones((12, 1)) * design_scale,
                Array2::eye(3) * 0.3,
                RepeatedModel::default(),
            )
            .unwrap();
            let r = p.fit_feature(0).unwrap();
            assert_eq!(r.status, RepeatedStatus::Estimated, "{:?}", r.diagnostics);
            let e = r.estimates.unwrap();
            assert!(
                (e.g[(0, 0)] - expected).abs() / expected < 1e-5,
                "{} {expected}",
                e.g[(0, 0)]
            );
        }
    }
}
#[test]
fn covariance_coordinate_and_subject_order_changes_preserve_fit() {
    let y = vec![
        -2.3, -0.7, 1.1, -1., -0.4, 0.5, -0.8, 0.8, 2.5, 1.8, 1.6, 1.5, 1.1, 3.6, 5.,
    ];
    let t = (0..15).map(|i| (i % 3) as f64 - 1.).collect::<Vec<_>>();
    let s = array![[0.25, 0.04, 0.01], [0.04, 0.30, 0.03], [0.01, 0.03, 0.28]];
    let base = RepeatedModel {
        random_effects: RandomEffects::InterceptSlope,
        residual_variance: ResidualVariance::Estimate,
        likelihood: RepeatedLikelihood::Ml,
        time: Some(t.clone()),
        max_evaluations: 10000,
        ..Default::default()
    };
    let p = prepared(y.clone(), 3, Array2::ones((15, 1)), s.clone(), base.clone()).unwrap();
    let a = p.fit_feature(0).unwrap().estimates.unwrap();
    let changed = RepeatedModel {
        time: Some(t.iter().map(|t| 10. + 2. * t).collect()),
        ..base.clone()
    };
    let b = prepared(y.clone(), 3, Array2::ones((15, 1)), s.clone(), changed)
        .unwrap()
        .fit_feature(0)
        .unwrap()
        .estimates
        .unwrap();
    assert!((a.objective - b.objective).abs() < 1e-9);
    assert!((b.g[(1, 1)] * 4. - a.g[(1, 1)]).abs() < 1e-8);
    assert!((b.g[(0, 1)] * 2. + 5. * a.g[(1, 1)] - a.g[(0, 1)]).abs() < 5e-6);
    let reordered = (0..5)
        .rev()
        .flat_map(|subject| (0..3).rev().map(move |visit| subject * 3 + visit))
        .collect::<Vec<_>>();
    let changed = RepeatedModel {
        time: Some(reordered.iter().map(|i| t[*i]).collect()),
        ..base
    };
    let s = Array2::from_shape_fn((3, 3), |(i, j)| s[(2 - i, 2 - j)]);
    let b = prepared(
        reordered.iter().map(|i| y[*i]).collect(),
        3,
        Array2::ones((15, 1)),
        s,
        changed,
    )
    .unwrap()
    .fit_feature(0)
    .unwrap()
    .estimates
    .unwrap();
    assert!((a.objective - b.objective).abs() < 1e-8);
    assert!((a.beta[0] - b.beta[0]).abs() < 1e-7);
}

#[test]
fn unequal_visits_match_dense_gls_and_independent_reml_optimizer() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/repeated_reference.json")).unwrap();
    let f = &fixture["ragged"];
    let numbers = |v: &serde_json::Value| {
        v.as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect::<Vec<_>>()
    };
    let sizes = f["sizes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect::<Vec<_>>();
    let time = numbers(&f["time"]);
    let y = numbers(&f["y"]);
    let x = Array2::from_shape_fn((y.len(), 2), |(i, j)| if j == 0 { 1. } else { time[i] });
    let sampling = f["sampling_blocks"]
        .as_array()
        .unwrap()
        .iter()
        .zip(&sizes)
        .map(|(block, &n)| {
            Array2::from_shape_vec(
                (n, n),
                block.as_array().unwrap().iter().flat_map(numbers).collect(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let model = RepeatedModel {
        random_effects: RandomEffects::InterceptSlope,
        time: Some(time),
        residual_variance: ResidualVariance::Estimate,
        ..Default::default()
    };
    let plan = prepared_blocks(y.clone(), &sizes, x.clone(), sampling.clone(), model).unwrap();
    let fitted = plan
        .evaluate_fixed(0, array![[0.7, 0.12], [0.12, 0.25]], 0.08)
        .unwrap();
    assert_eq!(fitted.status, RepeatedStatus::Estimated);
    let e = fitted.estimates.unwrap();
    let reference = &f["fixed"];
    for (a, b) in e.beta.iter().zip(numbers(&reference["beta"])) {
        assert!((a - b).abs() < 1e-11);
    }
    let covariance = reference["covariance"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(numbers)
        .collect::<Vec<_>>();
    for (a, b) in e.beta_covariance.iter().zip(covariance) {
        assert!((a - b).abs() < 1e-11);
    }
    assert!((e.objective - reference["objective"].as_f64().unwrap()).abs() < 1e-10);
    let plan = prepared_blocks(y, &sizes, x, sampling, RepeatedModel::default()).unwrap();
    let fitted = plan.fit_feature(0).unwrap();
    assert_eq!(
        fitted.status,
        RepeatedStatus::Estimated,
        "{:?}",
        fitted.diagnostics
    );
    let e = fitted.estimates.unwrap();
    assert!((e.g[(0, 0)] - f["ri_tau"].as_f64().unwrap()).abs() < 2e-5);
    assert!((e.objective - f["ri_objective"].as_f64().unwrap()).abs() < 1e-9);
}
