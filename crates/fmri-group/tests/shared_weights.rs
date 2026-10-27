mod common;
use common::cohort;
use fmri_group::*;
use ndarray::{array, Array2};

fn setup(n: usize, nf: usize) -> (IndependentCohort, GroupDesign, Vec<f64>) {
    let pattern = (0..n).map(|i| 1. + (i % 5) as f64).collect::<Vec<_>>();
    let y = Array2::from_shape_fn((n, nf), |(i, j)| {
        2. + 0.3 * i as f64 + (i as f64 * (j + 1) as f64 * 0.71).sin()
    });
    let v = Array2::from_shape_fn((n, nf), |(i, j)| pattern[i] * 2_f64.powi(j as i32 % 7 - 3));
    let c = cohort(y, Some(v));
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "slope".into()],
        Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                1.
            } else {
                (i as f64 - 3.) / n as f64
            }
        }),
    )
    .unwrap();
    (c, d, pattern)
}
fn spec(d: &GroupDesign) -> MetaOutputSpec {
    MetaOutputSpec {
        contrasts: vec![
            GroupContrast::named("slope", d, &[("slope", 1.)]).unwrap(),
            GroupContrast::named("mix", d, &[("intercept", 1.), ("slope", -0.5)]).unwrap(),
        ],
        uncertainty: MetaUncertainty::Joint,
        p_values: true,
        confidence_level: Some(0.95),
    }
}
fn collect(
    c: &IndependentCohort,
    p: &mut PreparedMetaRegression,
) -> (Vec<MetaRegressionFeature>, MetaRegressionReceipt) {
    let mut v = vec![];
    let r = p
        .fit(c, |b| {
            v.extend(b.features);
            Ok(())
        })
        .unwrap();
    (v, r)
}
fn near(x: f64, y: f64) {
    assert!((x - y).abs() < 2e-10 * (1. + y.abs()), "{x} != {y}");
}
fn equivalent(a: &MetaRegressionFeature, b: &MetaRegressionFeature) {
    assert_eq!(a.feature, b.feature);
    assert_eq!(a.status, b.status);
    assert_eq!(a.n, b.n);
    assert_eq!(a.rank, b.rank);
    if let (Some(a), Some(b)) = (&a.estimates, &b.estimates) {
        for (a, b) in a.effects.iter().zip(&b.effects) {
            near(*a, *b);
        }
        for (a, b) in a
            .standard_errors
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.standard_errors.as_ref().unwrap())
        {
            near(*a, *b);
        }
        for (a, b) in a
            .joint_covariance
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.joint_covariance.as_ref().unwrap())
        {
            near(*a, *b);
        }
        for (a, b) in a
            .statistics
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.statistics.as_ref().unwrap())
        {
            near(*a, *b);
        }
        for (a, b) in a
            .p_values
            .as_ref()
            .unwrap()
            .iter()
            .zip(b.p_values.as_ref().unwrap())
        {
            near(*a, *b);
        }
        for (a, b) in a
            .confidence_intervals
            .as_ref()
            .unwrap()
            .iter()
            .flatten()
            .zip(b.confidence_intervals.as_ref().unwrap().iter().flatten())
        {
            near(*a, *b);
        }
    }
    if let (Some(a), Some(b)) = (&a.diagnostics, &b.diagnostics) {
        near(a.q_final, b.q_final);
        near(a.tau_squared, b.tau_squared);
    }
}
#[test]
fn batched_common_effect_matches_general_across_units_blocks_inference_and_reuse() {
    let (c, d, w) = setup(20, 33);
    for inference in [MetaInference::Normal, MetaInference::ModifiedKnappHartung] {
        let mut baseline = PreparedMetaRegression::new(
            &d,
            spec(&d),
            MetaRegressionOptions {
                inference,
                ..Default::default()
            },
        )
        .unwrap();
        let (expected, _) = collect(&c, &mut baseline);
        for block_features in [1, 7, 32, 64] {
            let mut fast = PreparedMetaRegression::new(
                &d,
                spec(&d),
                MetaRegressionOptions {
                    inference,
                    execution: FitOptions {
                        block_features,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap()
            .with_shared_variances(w.clone())
            .unwrap();
            for _ in 0..2 {
                let (got, receipt) = collect(&c, &mut fast);
                assert_eq!(receipt.shared_features, 33);
                assert_eq!(receipt.run.factorizations, 0);
                assert_eq!(
                    receipt.shared_reference_variances.as_deref(),
                    Some(w.as_slice())
                );
                for (a, b) in got.iter().zip(&expected) {
                    equivalent(a, b);
                }
            }
        }
    }
}
#[test]
fn mismatching_weights_missing_rows_and_random_effects_fall_back() {
    let (base, d, w) = setup(20, 5);
    let mut y = Array2::from_shape_fn((20, 5), |(i, j)| {
        2. + (i as f64 * (j + 1) as f64 * 0.71).sin()
    });
    let mut v = Array2::from_shape_fn((20, 5), |(i, _)| w[i]);
    v[(3, 1)] *= 1. + 1e-14;
    y[(4, 2)] = f64::NAN;
    v[(5, 3)] = 0.;
    let c = cohort(y, Some(v));
    assert_eq!(base.subjects(), c.subjects());
    for heterogeneity in [
        HeterogeneityEstimator::CommonEffect,
        HeterogeneityEstimator::DerSimonianLaird,
        HeterogeneityEstimator::PauleMandel,
    ] {
        for missingness in [Missingness::RejectFeature, Missingness::OmitObservation] {
            let o = MetaRegressionOptions {
                heterogeneity,
                execution: FitOptions {
                    missingness,
                    ..Default::default()
                },
                ..Default::default()
            };
            let (expected, _) = collect(
                &c,
                &mut PreparedMetaRegression::new(&d, spec(&d), o).unwrap(),
            );
            let mut fast = PreparedMetaRegression::new(&d, spec(&d), o)
                .unwrap()
                .with_shared_variances(w.clone())
                .unwrap();
            let (got, r) = collect(&c, &mut fast);
            assert_eq!(
                r.shared_features,
                if heterogeneity == HeterogeneityEstimator::CommonEffect {
                    2
                } else {
                    0
                }
            );
            for (a, b) in got.iter().zip(&expected) {
                equivalent(a, b);
            }
        }
    }
}
#[test]
fn analytic_weighted_mean_and_difficult_geometry_keep_inference() {
    let c = cohort(
        array![[0., 0.], [2., 4.], [4., 8.], [6., 12.]],
        Some(array![[1., 4.], [1., 4.], [4., 16.], [4., 16.]]),
    );
    let d = GroupDesign::intercept(&c);
    let s = MetaOutputSpec {
        contrasts: vec![GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap()],
        uncertainty: MetaUncertainty::Marginal,
        p_values: true,
        confidence_level: Some(0.95),
    };
    let mut p = PreparedMetaRegression::new(&d, s, Default::default())
        .unwrap()
        .with_shared_variances(vec![1., 1., 4., 4.])
        .unwrap();
    let (r, receipt) = collect(&c, &mut p);
    assert_eq!(receipt.shared_features, 2);
    near(r[0].estimates.as_ref().unwrap().effects[0], 1.8);
    near(
        r[0].estimates
            .as_ref()
            .unwrap()
            .standard_errors
            .as_ref()
            .unwrap()[0],
        0.4_f64.sqrt(),
    );
    near(r[1].estimates.as_ref().unwrap().effects[0], 3.6);
    near(
        r[1].estimates
            .as_ref()
            .unwrap()
            .standard_errors
            .as_ref()
            .unwrap()[0],
        1.6_f64.sqrt(),
    );
    let c = cohort(
        array![[1.], [3.], [2.], [8.]],
        Some(array![[1e-30], [1.], [1e-30], [1.]]),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "slope".into()],
        array![[1., 0.], [1., 0.], [0., 1.], [0., 1.]],
    )
    .unwrap();
    let mut p = PreparedMetaRegression::new(&d, spec(&d), Default::default())
        .unwrap()
        .with_shared_variances(vec![1e-30, 1., 1e-30, 1.])
        .unwrap();
    let (r, receipt) = collect(&c, &mut p);
    assert_eq!(receipt.shared_features, 0);
    near(r[0].diagnostics.as_ref().unwrap().q_final, 40.);
}
#[test]
fn shared_declaration_validates_and_budget_includes_retained_buffers() {
    let (c, d, w) = setup(20, 33);
    let s = spec(&d);
    for bad in [vec![1.; 19], vec![0.; 20], vec![f64::NAN; 20]] {
        assert!(
            PreparedMetaRegression::new(&d, s.clone(), Default::default())
                .unwrap()
                .with_shared_variances(bad)
                .is_err()
        );
    }
    let mut p = PreparedMetaRegression::new(&d, s.clone(), Default::default())
        .unwrap()
        .with_shared_variances(w.clone())
        .unwrap();
    let (_, r) = collect(&c, &mut p);
    let (small, _, _) = setup(20, 3);
    let (_, rsmall) = collect(&small, &mut p);
    assert_eq!(
        rsmall.run.numeric_working_bytes,
        r.run.numeric_working_bytes
    );
    let mut p = PreparedMetaRegression::new(
        &d,
        s,
        MetaRegressionOptions {
            execution: FitOptions {
                max_working_bytes: r.run.numeric_working_bytes - 1,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap()
    .with_shared_variances(w)
    .unwrap();
    assert!(matches!(
        p.fit(&c, |_| panic!("budget precedes sink")),
        Err(Error::Budget)
    ));
}
