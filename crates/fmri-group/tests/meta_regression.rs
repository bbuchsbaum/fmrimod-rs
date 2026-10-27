use fmri_group::*;
use ndarray::{array, Array2};
use serde_json::Value;
mod common;
use common::cohort;
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect()
}
fn near(a: f64, b: f64) {
    assert!(a.is_finite() && b.is_finite());
    assert!((a - b).abs() < 2e-9 * (1. + b.abs()), "{a} != {b}");
}
fn design(c: &IndependentCohort, x: Array2<f64>) -> GroupDesign {
    GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        (0..x.ncols()).map(|i| format!("x{i}")).collect(),
        x,
    )
    .unwrap()
}
fn spec(d: &GroupDesign, c: Array2<f64>) -> MetaOutputSpec {
    let contrasts = c
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, row)| {
            let terms = d
                .columns()
                .iter()
                .zip(row)
                .map(|(s, w)| (s.as_str(), *w))
                .collect::<Vec<_>>();
            GroupContrast::named(format!("c{i}"), d, &terms).unwrap()
        })
        .collect();
    MetaOutputSpec {
        contrasts,
        uncertainty: MetaUncertainty::Joint,
        p_values: true,
        confidence_level: Some(0.95),
    }
}
fn options(
    method: HeterogeneityEstimator,
    inference: MetaInference,
    block: usize,
) -> MetaRegressionOptions {
    MetaRegressionOptions {
        heterogeneity: method,
        inference,
        execution: FitOptions {
            block_features: block,
            missingness: Missingness::OmitObservation,
            ..Default::default()
        },
        ..Default::default()
    }
}
fn run(
    c: &IndependentCohort,
    d: &GroupDesign,
    s: MetaOutputSpec,
    o: MetaRegressionOptions,
) -> Vec<MetaRegressionFeature> {
    let mut rows = vec![];
    fit_meta_regression(c, d, s, o, |b| {
        rows.extend(b.features);
        Ok(())
    })
    .unwrap();
    rows
}
#[test]
fn fresh_metafor_covariance_and_inference_oracles() {
    let reference: Value = serde_json::from_str(include_str!("meta_reference.json")).unwrap();
    let mut positive_pm = 0;
    for case in reference["cases"].as_array().unwrap() {
        let n = case["n"].as_u64().unwrap() as usize;
        let p = case["p"].as_u64().unwrap() as usize;
        let nf = case["features"].as_u64().unwrap() as usize;
        let scale = case["scale"].as_f64().unwrap();
        let c = cohort(
            Array2::from_shape_vec((n, nf), nums(&case["effects"])).unwrap(),
            Some(Array2::from_shape_vec((n, nf), nums(&case["variance"])).unwrap()),
        );
        let d = design(
            &c,
            Array2::from_shape_vec((n, p), nums(&case["X"])).unwrap(),
        );
        let s = spec(
            &d,
            Array2::from_shape_vec((3, p), nums(&case["contrasts"])).unwrap(),
        );
        for (method, key) in [
            (HeterogeneityEstimator::CommonEffect, "FE"),
            (HeterogeneityEstimator::DerSimonianLaird, "DL"),
            (HeterogeneityEstimator::PauleMandel, "PM"),
        ] {
            for (inference, label) in [
                (MetaInference::Normal, "z"),
                (MetaInference::ModifiedKnappHartung, "adhoc"),
            ] {
                for block in [1, 3, 256] {
                    for (row, expected) in run(&c, &d, s.clone(), options(method, inference, block))
                        .iter()
                        .zip(
                            case["results"][format!("{key}-{label}")]
                                .as_array()
                                .unwrap(),
                        )
                    {
                        assert_eq!(row.status, MetaStatus::Estimated, "{key} {label} {row:?}");
                        assert_eq!(row.n, expected["n"].as_u64().unwrap() as usize);
                        assert_eq!(row.rank, p);
                        let e = row.estimates.as_ref().unwrap();
                        let diag = row.diagnostics.as_ref().unwrap();
                        for j in 0..3 {
                            near(
                                e.effects[j] / scale,
                                expected["effects"][j].as_f64().unwrap() / scale,
                            );
                            near(
                                e.standard_errors.as_ref().unwrap()[j] / scale,
                                expected["se"][j].as_f64().unwrap() / scale,
                            );
                            near(
                                e.statistics.as_ref().unwrap()[j],
                                expected["statistic"][j].as_f64().unwrap(),
                            );
                            near(
                                e.p_values.as_ref().unwrap()[j],
                                expected["p"][j].as_f64().unwrap(),
                            );
                            for k in 0..2 {
                                near(
                                    e.confidence_intervals.as_ref().unwrap()[j][k] / scale,
                                    expected["intervals"][2 * j + k].as_f64().unwrap() / scale,
                                );
                            }
                            for k in 0..3 {
                                near(
                                    e.joint_covariance.as_ref().unwrap()[(j, k)] / (scale * scale),
                                    expected["covariance"][3 * j + k].as_f64().unwrap()
                                        / (scale * scale),
                                );
                            }
                        }
                        near(
                            diag.tau_squared / (scale * scale),
                            expected["tau"].as_f64().unwrap() / (scale * scale),
                        );
                        near(
                            diag.q_fixed.expect("estimated/common-effect Q"),
                            expected["q_fixed"].as_f64().unwrap(),
                        );
                        near(diag.q_final, expected["q_final"].as_f64().unwrap());
                        near(diag.covariance_scale, expected["scale"].as_f64().unwrap());
                        if method == HeterogeneityEstimator::PauleMandel && diag.tau_squared > 0. {
                            positive_pm += 1;
                            assert_eq!(diag.convergence, TauConvergence::Converged);
                            near(diag.q_final, diag.residual_df as f64);
                            let bracket = diag.bracket.unwrap();
                            assert!(
                                diag.tau_squared >= bracket[0] && diag.tau_squared <= bracket[1]
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(positive_pm > 0);
}
#[test]
fn analytic_equal_variance_pm_large_offset_and_row_order() {
    let x = array![[1., -1.], [1., -1.], [1., 0.], [1., 0.], [1., 1.], [1., 1.]];
    let residual = [-2., 2., -4., 4., -6., 6.];
    for offset in [0., 2_f64.powi(40)] {
        let y = Array2::from_shape_fn((6, 1), |(i, _)| offset + 2. * x[(i, 1)] + residual[i]);
        for reverse in [false, true] {
            let (xx, yy) = if reverse {
                (
                    x.slice(ndarray::s![..;-1,..]).to_owned(),
                    y.slice(ndarray::s![..;-1,..]).to_owned(),
                )
            } else {
                (x.clone(), y.clone())
            };
            let c = cohort(yy, Some(Array2::ones((6, 1))));
            let d = design(&c, xx);
            for method in [
                HeterogeneityEstimator::DerSimonianLaird,
                HeterogeneityEstimator::PauleMandel,
            ] {
                let out = run(
                    &c,
                    &d,
                    spec(&d, Array2::eye(2)),
                    options(method, MetaInference::ModifiedKnappHartung, 1),
                );
                let e = out[0].estimates.as_ref().unwrap();
                let h = out[0].diagnostics.as_ref().unwrap();
                near(e.effects[0] - offset, 0.);
                near(e.effects[1], 2.);
                near(h.tau_squared, 27.);
                near(h.q_fixed.expect("estimated/common-effect Q"), 112.);
                near(h.q_final, 4.);
                near(e.joint_covariance.as_ref().unwrap()[(0, 0)], 28. / 6.);
                near(e.joint_covariance.as_ref().unwrap()[(1, 1)], 7.);
                near(h.covariance_scale, 1.);
            }
        }
    }
}
#[test]
fn positive_qr_complement_denominator_survives_extreme_weights() {
    // Two independent pairs: each group's dominant row has leverage nearly 1.
    // As epsilon->0, Q=4+36=40, df=2 and tr(P)=2+2=4; tau_DL=9.5.
    for epsilon in [1e-30, 1e-310] {
        for a in 0..4 {
            for b in 0..4 {
                for cc in 0..4 {
                    for dd in 0..4 {
                        let order = [a, b, cc, dd];
                        if order
                            .iter()
                            .copied()
                            .collect::<std::collections::HashSet<_>>()
                            .len()
                            != 4
                        {
                            continue;
                        }
                        let y = [1., 3., 2., 8.];
                        let v = [epsilon, 1., epsilon, 1.];
                        let c = cohort(
                            Array2::from_shape_fn((4, 1), |(i, _)| y[order[i]]),
                            Some(Array2::from_shape_fn((4, 1), |(i, _)| v[order[i]])),
                        );
                        let d = design(
                            &c,
                            Array2::from_shape_fn(
                                (4, 2),
                                |(i, j)| if order[i] / 2 == j { 1. } else { 0. },
                            ),
                        );
                        for method in [
                            HeterogeneityEstimator::DerSimonianLaird,
                            HeterogeneityEstimator::PauleMandel,
                        ] {
                            let out = run(
                                &c,
                                &d,
                                spec(&d, Array2::eye(2)),
                                options(method, MetaInference::Normal, 1),
                            );
                            if epsilon < f64::MIN_POSITIVE {
                                assert_eq!(out[0].status, MetaStatus::NumericalFailure);
                                assert!(out[0].estimates.is_none());
                                continue;
                            }
                            assert_eq!(
                                out[0].status,
                                MetaStatus::Estimated,
                                "epsilon={epsilon}, order={order:?}, method={method:?}: {:?}",
                                out[0]
                            );
                            let h = out[0].diagnostics.as_ref().unwrap();
                            assert!(h.dl_complement_fallback);
                            near(h.q_fixed.expect("estimated/common-effect Q"), 40.);
                            near(h.tau_squared, 9.5);
                            near(h.q_final, 2.);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn requested_outputs_are_real_omissions_and_joint_matches_marginal() {
    let c = cohort(
        array![[1.], [3.], [2.], [8.]],
        Some(array![[1.], [2.], [3.], [1.]]),
    );
    let d = design(&c, array![[1., -1.], [1., 0.], [1., 1.], [1., 2.]]);
    let full = spec(&d, array![[1., 0.], [0., 1.], [1., -1.]]);
    let o = options(
        HeterogeneityEstimator::PauleMandel,
        MetaInference::ModifiedKnappHartung,
        1,
    );
    let reference = run(&c, &d, full.clone(), o);
    for uncertainty in [
        MetaUncertainty::None,
        MetaUncertainty::Marginal,
        MetaUncertainty::Joint,
    ] {
        let mut request = full.clone();
        request.uncertainty = uncertainty;
        request.p_values = false;
        request.confidence_level = None;
        let out = run(&c, &d, request, o);
        let e = out[0].estimates.as_ref().unwrap();
        let r = reference[0].estimates.as_ref().unwrap();
        assert_eq!(e.effects, r.effects);
        assert!(e.statistics.is_none() && e.p_values.is_none() && e.confidence_intervals.is_none());
        assert_eq!(
            e.standard_errors.is_some(),
            uncertainty != MetaUncertainty::None
        );
        assert_eq!(
            e.joint_covariance.is_some(),
            uncertainty == MetaUncertainty::Joint
        );
        if let Some(se) = &e.standard_errors {
            assert_eq!(se, r.standard_errors.as_ref().unwrap());
        }
    }
    let mut invalid = full.clone();
    invalid.uncertainty = MetaUncertainty::None;
    assert!(PreparedMetaRegression::new(&d, invalid, o).is_err());
    let mut invalid = full;
    invalid.confidence_level = Some(1.);
    assert!(PreparedMetaRegression::new(&d, invalid, o).is_err());
}
#[test]
fn feature_local_rank_missingness_and_nonconvergence() {
    let c = cohort(
        array![
            [1., 1., 1.],
            [2., 3., 2.],
            [f64::NAN, 2., f64::NAN],
            [f64::NAN, 8., f64::NAN]
        ],
        Some(Array2::ones((4, 3))),
    );
    let d = design(&c, array![[1., 0.], [1., 0.], [1., 1.], [1., 2.]]);
    let s = spec(&d, Array2::eye(2));
    let mut o = options(
        HeterogeneityEstimator::PauleMandel,
        MetaInference::Normal,
        3,
    );
    let a = run(&c, &d, s.clone(), o);
    assert_eq!(a[0].status, MetaStatus::InsufficientObservations);
    assert_eq!(a[1].status, MetaStatus::Estimated);
    o.execution.missingness = Missingness::RejectFeature;
    let a = run(&c, &d, s.clone(), o);
    assert_eq!(a[0].status, MetaStatus::MissingObservation);
    o.execution.missingness = Missingness::OmitObservation;
    o.paule_mandel.max_evaluations = 1;
    let a = run(&c, &d, s, o);
    assert_eq!(a[1].status, MetaStatus::NonConvergence);
    assert!(a[1].estimates.is_none());
    assert_eq!(
        a[1].diagnostics.as_ref().unwrap().convergence,
        TauConvergence::EvaluationLimit
    );
    let c = cohort(
        array![[1.], [2.], [4.], [f64::NAN]],
        Some(Array2::ones((4, 1))),
    );
    let d = design(&c, array![[1., 0.], [1., 0.], [1., 0.], [1., 1.]]);
    assert_eq!(
        run(
            &c,
            &d,
            spec(&d, Array2::eye(2)),
            options(
                HeterogeneityEstimator::CommonEffect,
                MetaInference::Normal,
                1
            )
        )[0]
        .status,
        MetaStatus::RankDeficient
    );
    let bad = design(&c, Array2::ones((4, 2)));
    assert!(PreparedMetaRegression::new(&bad, spec(&bad, Array2::eye(2)), o).is_err());
}
#[test]
fn zero_heterogeneity_constant_effects_and_negative_contrasts() {
    let c = cohort(Array2::from_elem((5, 2), 3.), Some(Array2::ones((5, 2))));
    let d = GroupDesign::intercept(&c);
    for method in [
        HeterogeneityEstimator::CommonEffect,
        HeterogeneityEstimator::DerSimonianLaird,
        HeterogeneityEstimator::PauleMandel,
    ] {
        let rows = run(
            &c,
            &d,
            spec(&d, array![[1.], [-2.]]),
            options(method, MetaInference::ModifiedKnappHartung, 1),
        );
        let e = rows[0].estimates.as_ref().unwrap();
        let h = rows[0].diagnostics.as_ref().unwrap();
        near(e.effects[0], 3.);
        near(e.effects[1], -6.);
        near(e.standard_errors.as_ref().unwrap()[0], 1. / 5_f64.sqrt());
        near(h.tau_squared, 0.);
        near(h.q_final, 0.);
        assert_eq!(h.evaluations, 1);
        near(e.joint_covariance.as_ref().unwrap()[(0, 1)], -0.4);
        near(
            e.p_values.as_ref().unwrap()[0],
            e.p_values.as_ref().unwrap()[1],
        );
    }
}
#[test]
fn prepared_reads_new_data_and_enforces_binding_budget_and_sink() {
    let c = cohort(
        array![[1., 2.], [3., 4.], [2., 7.], [8., 1.]],
        Some(Array2::ones((4, 2))),
    );
    let d = design(&c, array![[1., -1.], [1., 0.], [1., 1.], [1., 2.]]);
    let s = spec(&d, Array2::eye(2));
    let o = options(
        HeterogeneityEstimator::CommonEffect,
        MetaInference::Normal,
        1,
    );
    let mut plan = PreparedMetaRegression::new(&d, s.clone(), o).unwrap();
    let mut a = vec![];
    plan.fit(&c, |b| {
        a.extend(b.features);
        Ok(())
    })
    .unwrap();
    let changed = cohort(
        array![[2., 4.], [6., 8.], [4., 14.], [16., 2.]],
        Some(Array2::ones((4, 2))),
    );
    let mut b = vec![];
    plan.fit(&changed, |r| {
        b.extend(r.features);
        Ok(())
    })
    .unwrap();
    for (a, b) in a.iter().zip(b) {
        for (x, y) in a
            .estimates
            .as_ref()
            .unwrap()
            .effects
            .iter()
            .zip(&b.estimates.as_ref().unwrap().effects)
        {
            near(*x, *y / 2.);
        }
    }
    let mut calls = 0;
    assert!(plan
        .fit(&c, |_| {
            calls += 1;
            Err(Error::Invalid("sink"))
        })
        .is_err());
    assert_eq!(calls, 1);
    assert!(plan
        .fit(
            &cohort(Array2::ones((3, 1)), Some(Array2::ones((3, 1)))),
            |_| Ok(())
        )
        .is_err());
    let mut limited = o;
    limited.execution.max_working_bytes = 1;
    assert!(matches!(
        fit_meta_regression(&c, &d, s, limited, |_| panic!(
            "budget must precede reads/sink"
        )),
        Err(Error::Budget)
    ));
}

#[test]
fn extreme_weight_near_zero_and_large_statistics_are_distinguished() {
    for delta in [0., 2_f64.powi(-40)] {
        let c = cohort(
            array![[1. + delta], [3.], [2.], [8.]],
            Some(array![[1e-30], [1.], [1e-30], [1.]]),
        );
        let d = design(&c, array![[1., 0.], [1., 0.], [0., 1.], [0., 1.]]);
        let mut s = spec(&d, array![[2., -1.]]);
        s.confidence_level = Some(1. - f64::EPSILON);
        let result = run(
            &c,
            &d,
            s,
            options(
                HeterogeneityEstimator::CommonEffect,
                MetaInference::Normal,
                1,
            ),
        );
        assert_eq!(result[0].status, MetaStatus::Estimated);
        let e = result[0].estimates.as_ref().unwrap();
        near(
            e.statistics.as_ref().unwrap()[0],
            2. * delta / (5e-30_f64).sqrt(),
        );
        assert_eq!(
            e.p_values.as_ref().unwrap()[0],
            if delta == 0. { 1. } else { 0. }
        );
        assert!(e.confidence_intervals.as_ref().unwrap()[0]
            .iter()
            .all(|v| v.is_finite()));
    }
}
