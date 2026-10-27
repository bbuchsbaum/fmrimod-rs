use fmri_data::*;
use fmri_group::*;
use ndarray::{array, Array2};
use serde_json::Value;
mod common;
use common::*;
fn near(a: f64, b: f64) {
    assert!((a - b).abs() <= 2e-10 * (1. + b.abs()), "{a} != {b}");
}
fn nums(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap_or(f64::NAN))
        .collect()
}
fn options(block: usize) -> FitOptions {
    FitOptions {
        block_features: block,
        missingness: Missingness::OmitObservation,
        ..Default::default()
    }
}
fn collect_meta(c: &IndependentCohort, m: MetaMethod, o: FitOptions) -> Vec<FeatureResult> {
    let mut all = vec![];
    fit_meta(c, m, o, |b| {
        all.extend(b.features);
        Ok(())
    })
    .unwrap();
    all
}
#[test]
fn independent_r_oracles_and_block_invariance() {
    let r: Value = serde_json::from_str(include_str!("reference.json")).unwrap();
    let n = r["n"].as_u64().unwrap() as usize;
    let f = r["features"].as_u64().unwrap() as usize;
    let c = cohort(
        Array2::from_shape_vec((n, f), nums(&r["effects"])).unwrap(),
        Some(Array2::from_shape_vec((n, f), nums(&r["variance"])).unwrap()),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "age".into()],
        Array2::from_shape_vec((n, 2), nums(&r["X"])).unwrap(),
    )
    .unwrap();
    let contrast =
        GroupContrast::named("adjusted", &d, &[("intercept", 1.), ("age", -0.4)]).unwrap();
    for block in [1, 3, 256] {
        let mut out = vec![];
        fit_ols(
            &c,
            &d,
            std::slice::from_ref(&contrast),
            options(block),
            |b| {
                out.extend(b.features);
                Ok(())
            },
        )
        .unwrap();
        for (j, row) in out.iter().enumerate() {
            let a = row.estimates[0].as_ref().unwrap();
            let expected = &r["ols"][j];
            near(a.effect, expected["effect"].as_f64().unwrap());
            near(a.standard_error, expected["se"].as_f64().unwrap());
            near(a.statistic, expected["statistic"].as_f64().unwrap());
            near(a.p_value, expected["p"].as_f64().unwrap());
            assert_eq!(row.n, expected["n"].as_u64().unwrap() as usize);
            assert_eq!(
                a.distribution,
                Distribution::StudentT {
                    df: expected["df"].as_u64().unwrap() as usize
                }
            );
        }
        for (method, key) in [
            (MetaMethod::CommonEffect, "fe"),
            (MetaMethod::DerSimonianLaird, "dl"),
        ] {
            for (j, row) in collect_meta(&c, method, options(block)).iter().enumerate() {
                let a = row.estimates[0].as_ref().unwrap();
                let e = &r[key][j];
                near(a.effect, e["effect"].as_f64().unwrap());
                near(a.standard_error, e["se"].as_f64().unwrap());
                near(a.statistic, e["statistic"].as_f64().unwrap());
                near(a.p_value, e["p"].as_f64().unwrap());
                let h = row.heterogeneity.as_ref().unwrap();
                near(h.q, e["Q"].as_f64().unwrap());
                near(h.tau_squared, e["tau2"].as_f64().unwrap());
            }
        }
    }
}
#[test]
fn identity_validation_and_variance_requirement() {
    let c = cohort(array![[1., 2.], [2., 3.], [4., 5.]], None);
    assert!(fit_meta(&c, MetaMethod::CommonEffect, options(2), |_| Ok(())).is_err());
    let mut rows = c.subjects().to_vec();
    rows[1].participant = rows[0].participant.clone();
    assert!(IndependentCohort::new(
        c.frame().clone(),
        rows,
        estimand(),
        "effect".into(),
        None,
        vec![fit_receipt(); 3],
        admission()
    )
    .is_err());
    let mut ids = c.frame().observations().ids().to_vec();
    ids.reverse();
    let d = GroupDesign::new(ids, vec!["x".into()], Array2::ones((3, 1))).unwrap();
    let ct = GroupContrast::named("x", &d, &[("x", 1.)]).unwrap();
    assert!(fit_ols(&c, &d, &[ct], options(2), |_| Ok(())).is_err());
    let d = GroupDesign::intercept(&c);
    let other = GroupDesign::new(dumm_ids(&c), vec!["other".into()], Array2::ones((3, 1))).unwrap();
    let wrong = GroupContrast::named("wrong", &other, &[("other", 1.)]).unwrap();
    assert!(fit_ols(&c, &d, &[wrong], options(2), |_| Ok(())).is_err());
}
fn dumm_ids(c: &IndependentCohort) -> Vec<ObservationId> {
    c.frame().observations().ids().to_vec()
}
#[test]
fn missingness_rank_and_constant_feature_are_local() {
    let y = array![
        [0., 1., 1., f64::NAN],
        [0., 2., 2., f64::NAN],
        [0., 4., f64::NAN, f64::NAN],
        [0., 5., f64::NAN, f64::NAN]
    ];
    let c = cohort(y, None);
    let d = GroupDesign::new(
        dumm_ids(&c),
        vec!["intercept".into(), "group".into()],
        array![[1., 0.], [1., 0.], [1., 1.], [1., 1.]],
    )
    .unwrap();
    let contrasts = [
        GroupContrast::named("baseline", &d, &[("intercept", 1.)]).unwrap(),
        GroupContrast::named("group", &d, &[("group", 1.)]).unwrap(),
    ];
    let mut out = vec![];
    let r = fit_ols(&c, &d, &contrasts, options(4), |b| {
        out.extend(b.features);
        Ok(())
    })
    .unwrap();
    assert_eq!(out[0].status[0], Status::DegenerateVariance);
    assert_eq!(out[1].status[0], Status::Estimated);
    assert_eq!(out[2].status[0], Status::Estimated);
    assert_eq!(out[2].rank, 1);
    assert_eq!(out[2].status[1], Status::NonEstimable);
    assert_eq!(out[3].status[0], Status::InsufficientObservations);
    assert_eq!(r.factorizations, 2);
    let mut strict = vec![];
    fit_ols(&c, &d, &contrasts, FitOptions::default(), |b| {
        strict.extend(b.features);
        Ok(())
    })
    .unwrap();
    assert_eq!(strict[2].status[0], Status::MissingObservation);
}
#[test]
fn unit_invariance_dl_boundary_and_valid_zero() {
    for method in [MetaMethod::CommonEffect, MetaMethod::DerSimonianLaird] {
        let base = cohort(
            array![[0.], [2.], [4.], [6.]],
            Some(array![[1.], [1.], [4.], [4.]]),
        );
        let a = collect_meta(&base, method, options(1));
        let a = a[0].estimates[0].as_ref().unwrap();
        for scale in [1e-8, 1e8] {
            let c = cohort(
                array![[0.], [2.], [4.], [6.]] * scale,
                Some(array![[1.], [1.], [4.], [4.]] * (scale * scale)),
            );
            let b = collect_meta(&c, method, options(1));
            assert_eq!(b[0].n, 4);
            let b = b[0].estimates[0].as_ref().unwrap();
            near(a.effect, b.effect / scale);
            near(a.standard_error, b.standard_error / scale);
            near(a.statistic, b.statistic);
        }
    }
    let c = cohort(array![[-1.], [1.]], Some(Array2::ones((2, 1))));
    let r = collect_meta(&c, MetaMethod::DerSimonianLaird, options(1));
    near(r[0].heterogeneity.as_ref().unwrap().tau_squared, 1.);
    near(r[0].estimates[0].as_ref().unwrap().standard_error, 1.);
    let c = cohort(array![[0.], [0.], [0.]], Some(Array2::ones((3, 1))));
    let r = collect_meta(&c, MetaMethod::DerSimonianLaird, options(1));
    near(r[0].heterogeneity.as_ref().unwrap().tau_squared, 0.);
    near(r[0].estimates[0].as_ref().unwrap().p_value, 1.);
}
#[test]
fn budget_and_sink_failure_are_enforced() {
    let c = cohort(Array2::ones((4, 8)), Some(Array2::ones((4, 8))));
    let mut calls = 0;
    assert!(matches!(
        fit_meta(
            &c,
            MetaMethod::CommonEffect,
            FitOptions {
                max_working_bytes: 1,
                ..options(1)
            },
            |_| {
                calls += 1;
                Ok(())
            }
        ),
        Err(Error::Budget)
    ));
    assert_eq!(calls, 0);
    assert!(fit_meta(&c, MetaMethod::CommonEffect, options(1), |_| {
        calls += 1;
        Err(Error::Invalid("sink stopped"))
    })
    .is_err());
    assert_eq!(calls, 1);
}
#[test]
fn fdr_matches_base_r_and_preserves_missing_family_members() {
    let r: Value = serde_json::from_str(include_str!("reference.json")).unwrap();
    let values = r["p_values"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, p)| (FeatureId::new(format!("f{i}")).unwrap(), p.as_f64()))
        .collect();
    let f = TestFamily::new("whole-map", "mean", values).unwrap();
    for (method, key) in [
        (FdrMethod::BenjaminiHochberg, "bh"),
        (FdrMethod::BenjaminiYekutieli, "by"),
    ] {
        let a = f.adjust(method);
        assert_eq!(a.tested, 7);
        for (i, (_, p)) in a.values.iter().enumerate() {
            match (p, r[key][i].as_f64()) {
                (Some(a), Some(b)) => near(*a, b),
                (None, None) => (),
                _ => panic!("missing mismatch"),
            }
        }
    }
    let a = TestFamily::new(
        "empty-support",
        "mean",
        vec![(FeatureId::new("f").unwrap(), None)],
    )
    .unwrap()
    .adjust(FdrMethod::BenjaminiHochberg);
    assert_eq!(a.tested, 0);
    assert!(a.values[0].1.is_none());
    assert!(TestFamily::new(
        "bad",
        "mean",
        vec![(FeatureId::new("f").unwrap(), Some(-0.1))]
    )
    .is_err());
}
#[test]
fn first_level_to_group_and_explicit_provenance() {
    let x = array![[1., -1.], [1., 0.], [1., 1.], [1., 2.], [1., 3.]];
    let model = fmri_reg::LinearModel::new(x.view(), 1e-12).unwrap();
    let mut rows = vec![];
    let mut expected = vec![];
    for (i, subject) in subjects(5).into_iter().enumerate() {
        let y = Array2::from_shape_fn((5, 2), |(t, j)| {
            1. + (0.3 + i as f64 * 0.1) * x[(t, 1)] + ((t + j * 2 + i) as f64).sin() * 0.2
        });
        let fit = model.fit(y.view()).unwrap();
        let t = fit.t_contrast(array![0., 1.].view()).unwrap();
        expected.push(t.effect[0]);
        rows.push(
            FirstLevelEstimate::from_t(subject, estimand(), features(2), &t, fit_receipt(), false)
                .unwrap(),
        );
    }
    let c = IndependentCohort::from_first_level(rows.clone(), admission()).unwrap();
    let d = GroupDesign::intercept(&c);
    let ct = GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap();
    let mut output = vec![];
    let receipt = fit_ols(&c, &d, &[ct], options(1), |b| {
        output.extend(b.features);
        Ok(())
    })
    .unwrap();
    near(
        output[0].estimates[0].as_ref().unwrap().effect,
        expected.iter().sum::<f64>() / 5.,
    );
    assert_eq!(receipt.variance_origins.len(), 5);
    assert_eq!(receipt.blocks, 2);
    let mut duplicate = rows.clone();
    duplicate.push(rows[0].clone());
    assert!(IndependentCohort::from_first_level(duplicate, admission()).is_err());
}
#[test]
fn dataset_rejects_permuted_variance_axis() {
    let ids = subjects(3)
        .iter()
        .map(|s| s.observation.clone())
        .collect::<Vec<_>>();
    let mut wrong = ids.clone();
    wrong.reverse();
    let fs = features(1);
    let source = SourceSpec::memory(NumericBlock::new([3, 1], Values::F64(vec![1.; 3])).unwrap());
    let a = Assay::new(
        "effect",
        source.clone(),
        ids.clone(),
        fs.axis().ids().to_vec(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let v = Assay::new(
        "variance",
        source,
        wrong,
        fs.axis().ids().to_vec(),
        AssayAnnotations::default(),
    )
    .unwrap();
    assert!(FmriFrame::new(
        Axis::new(ids, vec![]).unwrap(),
        fs,
        vec![a, v],
        "effect",
        Metadata::new()
    )
    .is_err());
}
#[test]
fn bounded_pattern_cache_and_row_permutation() {
    let y = Array2::from_shape_fn((8, 17), |(i, j)| {
        if i == j % 8 {
            f64::NAN
        } else {
            (i as f64 * 0.4 + j as f64).sin() + i as f64
        }
    });
    let c = cohort(y.clone(), Some(Array2::ones(y.dim())));
    // Exercise the general solver/cache, not the factorization-free mean path.
    let d = GroupDesign::new(
        dumm_ids(&c),
        vec!["a".into(), "b".into()],
        Array2::ones((8, 2)),
    )
    .unwrap();
    let ct = GroupContrast::named("mean", &d, &[("a", 1.), ("b", 1.)]).unwrap();
    let mut a = vec![];
    let r = fit_ols(
        &c,
        &d,
        std::slice::from_ref(&ct),
        FitOptions {
            max_cached_patterns: 1,
            ..options(3)
        },
        |b| {
            a.extend(b.features);
            Ok(())
        },
    )
    .unwrap();
    assert!(r.factorizations > 8);
    let reversed = cohort(
        y.slice(ndarray::s![..;-1,..]).to_owned(),
        Some(Array2::ones(y.dim())),
    );
    let rd = GroupDesign::new(
        dumm_ids(&reversed),
        d.columns().to_vec(),
        Array2::ones((8, 2)),
    )
    .unwrap();
    let mut b = vec![];
    fit_ols(&reversed, &rd, &[ct], options(17), |r| {
        b.extend(r.features);
        Ok(())
    })
    .unwrap();
    for (a, b) in a.iter().zip(&b) {
        near(
            a.estimates[0].as_ref().unwrap().statistic,
            b.estimates[0].as_ref().unwrap().statistic,
        );
    }
}
#[test]
fn f32_source_and_invalid_variance_policy() {
    let c = cohort(
        array![[0., 1.], [2., 2.], [4., 4.]],
        Some(array![[1., 1.], [0., -1.], [1., f64::NAN]]),
    );
    let a = collect_meta(&c, MetaMethod::CommonEffect, options(1));
    assert_eq!(a[0].n, 2);
    near(a[0].estimates[0].as_ref().unwrap().effect, 2.);
    assert_eq!(a[1].status[0], Status::InsufficientObservations);
    let strict = collect_meta(&c, MetaMethod::CommonEffect, FitOptions::default());
    assert_eq!(strict[0].status[0], Status::MissingObservation);
    let ids = c.frame().observations().ids().to_vec();
    let fs = features(2);
    let source = SourceSpec::memory(
        NumericBlock::new([3, 2], Values::F32(vec![0., 1., 2., 2., 4., 4.])).unwrap(),
    );
    let frame =
        FmriFrame::from_source("effect", source, Axis::new(ids, vec![]).unwrap(), fs).unwrap();
    let c32 = IndependentCohort::new(
        frame,
        subjects(3),
        estimand(),
        "effect".into(),
        None,
        vec![fit_receipt(); 3],
        admission(),
    )
    .unwrap();
    let d = GroupDesign::intercept(&c32);
    let ct = GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap();
    let mut out = vec![];
    fit_ols(&c32, &d, &[ct], options(1), |b| {
        out.extend(b.features);
        Ok(())
    })
    .unwrap();
    near(out[0].estimates[0].as_ref().unwrap().effect, 2.);
}
#[test]
fn first_level_bridge_rejects_different_estimands_and_feature_identity() {
    let t = fmri_reg::TContrast {
        effect: array![1.],
        standard_error: array![1.],
        statistic: array![1.],
        p_value: array![0.5],
        df: 8,
    };
    let ss = subjects(2);
    let a = FirstLevelEstimate::from_t(
        ss[0].clone(),
        estimand(),
        features(1),
        &t,
        fit_receipt(),
        true,
    )
    .unwrap();
    let mut different = estimand();
    different.units = "raw signal".into();
    let b = FirstLevelEstimate::from_t(
        ss[1].clone(),
        different,
        features(1),
        &t,
        fit_receipt(),
        true,
    )
    .unwrap();
    assert!(IndependentCohort::from_first_level(vec![a.clone(), b], admission()).is_err());
    let other = FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new("different reference", vec![FeatureId::new("f0").unwrap()]).unwrap(),
    ))
    .unwrap();
    let b = FirstLevelEstimate::from_t(ss[1].clone(), estimand(), other, &t, fit_receipt(), true)
        .unwrap();
    assert!(IndependentCohort::from_first_level(vec![a, b], admission()).is_err());
}
#[test]
fn retained_zero_rank_is_not_reported_as_too_few_observations() {
    let c = cohort(array![[1.], [2.], [f64::NAN], [f64::NAN]], None);
    let d = GroupDesign::new(
        dumm_ids(&c),
        vec!["x".into()],
        array![[0.], [0.], [1.], [2.]],
    )
    .unwrap();
    let contrast = GroupContrast::named("x", &d, &[("x", 1.)]).unwrap();
    let mut output = vec![];
    fit_ols(&c, &d, &[contrast], options(1), |b| {
        output.extend(b.features);
        Ok(())
    })
    .unwrap();
    assert_eq!(output[0].n, 2);
    assert_eq!(output[0].status[0], Status::UnusableDesign);
}

// Relative errors matter for unit-scaled uncertainty; an absolute tolerance of
// 1e-10 would silently accept zero for the small-unit witnesses.
fn relative_near(a: f64, b: f64, tolerance: f64) {
    assert!(a.is_finite() && b.is_finite());
    if b == 0. {
        assert_eq!(a, 0.);
    } else {
        assert!(
            (a - b).abs() <= tolerance * b.abs(),
            "{a} != {b}, relative error {}",
            (a - b).abs() / b.abs()
        );
    }
}
#[test]
fn centered_intercept_matches_decimal_across_offsets_units_and_contrasts() {
    let reference: Value =
        serde_json::from_str(include_str!("performance_reference.json")).unwrap();
    for case in reference["ols"].as_array().unwrap() {
        let ys = nums(&case["y"]);
        for reverse in [false, true] {
            let mut input = ys.clone();
            if reverse {
                input.reverse();
            }
            // Same data, neighboring degenerate features, and varied row support.
            let y = Array2::from_shape_fn((ys.len(), 4), |(i, j)| match j {
                0 => input[i],
                1 => 7.,
                2 => {
                    if i == 0 {
                        f64::NAN
                    } else {
                        input[i]
                    }
                }
                _ => f64::NAN,
            });
            let c = cohort(y, None);
            let d = GroupDesign::intercept(&c);
            let contrasts = [1., -2., 1e-30]
                .map(|w| GroupContrast::named(format!("w{w}"), &d, &[("intercept", w)]).unwrap());
            for block in [1, 3, 256] {
                let mut out = vec![];
                let receipt = fit_ols(&c, &d, &contrasts, options(block), |b| {
                    out.extend(b.features);
                    Ok(())
                })
                .unwrap();
                assert_eq!(receipt.factorizations, 0);
                for (index, weight) in [1_f64, -2., 1e-30].iter().enumerate() {
                    let e = out[0].estimates[index].as_ref().unwrap();
                    relative_near(e.effect, weight * case["mean"].as_f64().unwrap(), 3e-15);
                    relative_near(
                        e.standard_error,
                        weight.abs() * case["se"].as_f64().unwrap(),
                        3e-15,
                    );
                    relative_near(
                        e.statistic,
                        weight.signum() * case["t"].as_f64().unwrap(),
                        4e-15,
                    );
                }
                assert_eq!(out[1].status[0], Status::DegenerateVariance);
                assert_eq!(out[2].n, ys.len() - 1);
                assert_eq!(out[3].status[0], Status::InsufficientObservations);
            }
        }
    }
}
#[test]
fn tiled_meta_matches_decimal_and_preserves_zero_tau_bits() {
    let reference: Value =
        serde_json::from_str(include_str!("performance_reference.json")).unwrap();
    for case in reference["meta"].as_array().unwrap() {
        let ys = nums(&case["y"]);
        let vs = nums(&case["v"]);
        let c = cohort(
            Array2::from_shape_fn((ys.len(), 67), |(i, _)| ys[i]),
            Some(Array2::from_shape_fn((ys.len(), 67), |(i, _)| vs[i])),
        );
        let method = if case["method"] == "fe" {
            MetaMethod::CommonEffect
        } else {
            MetaMethod::DerSimonianLaird
        };
        for block in [1, 31, 32, 33, 67] {
            let out = collect_meta(&c, method, options(block));
            for row in &out {
                let a = row.estimates[0].as_ref().unwrap();
                let h = row.heterogeneity.as_ref().unwrap();
                relative_near(a.effect, case["effect"].as_f64().unwrap(), 2e-13);
                relative_near(a.standard_error, case["se"].as_f64().unwrap(), 2e-13);
                relative_near(a.statistic, case["t"].as_f64().unwrap(), 2e-13);
                relative_near(h.q, case["q"].as_f64().unwrap(), 2e-13);
                relative_near(h.tau_squared, case["tau"].as_f64().unwrap(), 2e-13);
            }
        }
        if case["name"] == "zero_tau" {
            let fe = collect_meta(&c, MetaMethod::CommonEffect, options(33));
            let dl = collect_meta(&c, MetaMethod::DerSimonianLaird, options(33));
            for (a, b) in fe.iter().zip(dl) {
                let a = a.estimates[0].as_ref().unwrap();
                let b = b.estimates[0].as_ref().unwrap();
                assert_eq!(a.effect.to_bits(), b.effect.to_bits());
                assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
                assert_eq!(a.p_value.to_bits(), b.p_value.to_bits());
            }
        }
    }
}
#[test]
fn prepared_ols_reuses_geometry_but_rereads_data_and_feature_axes() {
    let y = Array2::from_shape_fn((9, 7), |(i, j)| {
        (i as f64 * 0.3 + j as f64).sin() + i as f64
    });
    let c = cohort(y.clone(), None);
    let d = GroupDesign::new(
        dumm_ids(&c),
        vec!["intercept".into(), "x".into()],
        Array2::from_shape_fn((9, 2), |(i, j)| if j == 0 { 1. } else { i as f64 }),
    )
    .unwrap();
    let ct = GroupContrast::named("slope", &d, &[("x", 1.)]).unwrap();
    let mut plan = PreparedOls::new(&d, std::slice::from_ref(&ct), options(4)).unwrap();
    let first = plan.fit(&c, |_| Ok(())).unwrap();
    assert_eq!(first.factorizations, 1);
    for nf in [2, 11, 7] {
        let changed = cohort(
            Array2::from_shape_fn((9, nf), |(i, j)| 2. * y[(i, j % 7)]),
            None,
        );
        let mut reused = vec![];
        let mut fresh = vec![];
        let r = plan
            .fit(&changed, |b| {
                reused.extend(b.features);
                Ok(())
            })
            .unwrap();
        assert_eq!(r.factorizations, 0);
        fit_ols(&changed, &d, std::slice::from_ref(&ct), options(4), |b| {
            fresh.extend(b.features);
            Ok(())
        })
        .unwrap();
        assert_eq!(reused.len(), nf);
        for (a, b) in reused.iter().zip(fresh) {
            assert_eq!(a.feature, b.feature);
            let a = a.estimates[0].as_ref().unwrap();
            let b = b.estimates[0].as_ref().unwrap();
            assert_eq!(a.effect.to_bits(), b.effect.to_bits());
            assert_eq!(a.standard_error.to_bits(), b.standard_error.to_bits());
            assert_eq!(a.p_value.to_bits(), b.p_value.to_bits());
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
        .fit(&cohort(Array2::ones((8, 1)), None), |_| Ok(()))
        .is_err());
    // A failed sink leaves no stale result state in subsequent executions.
    assert_eq!(plan.fit(&c, |_| Ok(())).unwrap().blocks, 2);
}
#[test]
fn intercept_failure_policy_and_budget_are_feature_local() {
    let c = cohort(
        array![
            [f64::MAX, 1., f64::NAN],
            [f64::MAX, -1., 2.],
            [-f64::MAX, 0., 3.]
        ],
        None,
    );
    let d = GroupDesign::intercept(&c);
    let ct = GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap();
    let mut out = vec![];
    fit_ols(
        &c,
        &d,
        std::slice::from_ref(&ct),
        FitOptions::default(),
        |b| {
            out.extend(b.features);
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(out[0].status[0], Status::NumericalFailure);
    assert_eq!(out[1].status[0], Status::Estimated);
    assert_eq!(out[1].estimates[0].as_ref().unwrap().p_value, 1.);
    assert_eq!(out[2].status[0], Status::MissingObservation);
    let mut called = false;
    assert!(matches!(
        fit_ols(
            &c,
            &d,
            &[ct],
            FitOptions {
                max_working_bytes: 1,
                ..Default::default()
            },
            |_| {
                called = true;
                Ok(())
            }
        ),
        Err(Error::Budget)
    ));
    assert!(!called);
}
