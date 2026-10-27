use fmri_noise::{Ar1Plan, FirstSample, WhiteningPlan};
use fmrireg::{
    output::{EstimandSet, OutputSpec, Summary, SummaryView, Uncertainty},
    FitError, GlsModel, LinearModel, MatrixBackend,
};
use ndarray::{array, s, Array2, ArrayView2, ShapeBuilder};

fn spec(weights: Array2<f64>, uncertainty: Uncertainty) -> OutputSpec {
    let names = (0..weights.nrows())
        .map(|i| format!("effect-{i}"))
        .collect();
    OutputSpec::new(EstimandSet::contrasts(names, weights).unwrap()).with_uncertainty(uncertainty)
}
fn backends() -> Vec<MatrixBackend> {
    #[allow(unused_mut)]
    let mut b = vec![MatrixBackend::Portable];
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    b.push(MatrixBackend::Accelerate);
    b
}
fn close(a: f64, b: f64, tol: f64) {
    assert!(
        a.is_finite() && (a - b).abs() <= tol * (1. + b.abs()),
        "{a} != {b}"
    );
}

#[test]
fn analytic_joint_covariance_dependent_and_zero_estimands() {
    // Three orthogonal columns with norm sqrt(8). The fourth Walsh column
    // supplies exactly orthogonal residuals. No production formula is used to
    // derive the independent coefficient/RSS/covariance expectations.
    let x = Array2::from_shape_fn((8, 3), |(i, j)| match j {
        0 => 1.,
        1 => {
            if i % 2 == 0 {
                1.
            } else {
                -1.
            }
        }
        _ => {
            if (i / 2) % 2 == 0 {
                1.
            } else {
                -1.
            }
        }
    });
    let truth = array![[2., -1.], [3., 2.], [-4., 5.]];
    let y = Array2::from_shape_fn((8, 2), |(i, j)| {
        (0..3).map(|k| x[(i, k)] * truth[(k, j)]).sum::<f64>()
            + (j + 1) as f64 * if i < 4 { 0.25 } else { -0.25 }
    });
    let l = array![
        [0., 1., 0.],
        [0., 0., 1.],
        [0., 1., -1.],
        [0., 2., -2.],
        [0., 0., 0.]
    ];
    let units = array![
        [1., 0., 1., 2., 0.],
        [0., 1., -1., -2., 0.],
        [1., -1., 2., 4., 0.],
        [2., -2., 4., 8., 0.],
        [0., 0., 0., 0., 0.]
    ];
    for backend in backends() {
        let model = LinearModel::new(x.view(), 1e-12)
            .unwrap()
            .with_backend(backend);
        for level in [Uncertainty::None, Uncertainty::Marginal, Uncertainty::Joint] {
            let request = spec(l.clone(), level);
            let plan = model.prepare_outputs(&request).unwrap();
            let mut workspace = plan.workspace(2).unwrap();
            let summary = plan.run_into(y.view(), &mut workspace).unwrap();
            assert_eq!(summary.residual_df(), 5);
            let expected = l.dot(&truth);
            for (a, b) in summary.estimates().iter().zip(expected.iter()) {
                close(*a, *b, 2e-12);
            }
            for j in 0..2 {
                let scale = 0.1 * ((j + 1) * (j + 1)) as f64;
                for i in 0..5 {
                    let variance = units[(i, i)] * scale / 8.;
                    match &summary {
                        SummaryView::Estimates(_) => {}
                        SummaryView::Marginal(v) => {
                            close(v.variance(i, j).unwrap(), variance, 2e-12);
                            close(v.standard_error(i, j).unwrap(), variance.sqrt(), 2e-12);
                            assert!(v.variance(5, j).is_err());
                        }
                        SummaryView::Joint(v) => {
                            close(v.variance(i, j).unwrap(), variance, 2e-12);
                            for k in 0..5 {
                                close(
                                    v.covariance(i, k, j).unwrap(),
                                    units[(i, k)] * scale / 8.,
                                    2e-12,
                                );
                            }
                            assert!(v.covariance(i, 5, j).is_err());
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn gls_marginal_and_joint_match_independent_precision_solution() {
    // Two stationary AR(1), phi=1/2 runs, each length four. Precision sums to
    // three, giving intercept covariance factor 1/3 and explicit tridiagonal RSS.
    let x = Array2::ones((8, 1));
    let y = array![
        [1., 8.],
        [2., 7.],
        [4., 6.],
        [5., 5.],
        [2., 4.],
        [3., 3.],
        [7., 2.],
        [8., 1.]
    ];
    for backend in backends() {
        let model = GlsModel::new(
            x.view(),
            Ar1Plan::new(vec![4, 4], vec![0.5, 0.5], FirstSample::Stationary).unwrap(),
            1e-12,
        )
        .unwrap()
        .with_backend(backend);
        let request = spec(array![[1.], [2.], [0.]], Uncertainty::Joint);
        let plan = model.prepare_outputs(&request).unwrap();
        let mut workspace = plan.workspace(2).unwrap();
        let summary = plan.run_into(y.view(), &mut workspace).unwrap();
        assert!(summary.conditional_noise());
        let SummaryView::Joint(joint) = &summary else {
            panic!("joint required")
        };
        for j in 0..2 {
            let weights = [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5];
            let beta = (0..8).map(|i| weights[i] * y[(i, j)]).sum::<f64>() / 3.;
            let r: Vec<_> = (0..8).map(|i| y[(i, j)] - beta).collect();
            let mut rss = 0.;
            for start in [0, 4] {
                rss += r[start].powi(2) + r[start + 3].powi(2);
                for v in &r[start + 1..start + 3] {
                    rss += 1.25 * v * v;
                }
                for i in start..start + 3 {
                    rss -= r[i] * r[i + 1];
                }
            }
            close(summary.estimates()[(0, j)], beta, 2e-12);
            close(joint.variance(0, j).unwrap(), rss / 21., 2e-12);
            close(joint.covariance(0, 1, j).unwrap(), 2. * rss / 21., 2e-12);
        }
    }
}

#[test]
fn minimal_workspace_tail_strides_and_owned_summary_outlive_models() {
    let retained = {
        let x = Array2::ones((128, 1));
        let model = LinearModel::new(x.view(), 1e-12).unwrap();
        let request = spec(array![[1.]], Uncertainty::None);
        let plan = model.prepare_outputs(&request).unwrap();
        assert!(!plan.requires_residual_scale());
        assert_eq!(plan.retained_numeric_bytes(5).unwrap(), 5 * 8);
        assert!(plan.retained_numeric_bytes(usize::MAX).is_err());
        let clone = plan.clone();
        let mut workspace = plan.workspace(5).unwrap();
        clone
            .run_into(Array2::ones((128, 1)).view(), &mut workspace)
            .unwrap();
        assert_eq!(workspace.working_bytes(), plan.workspace_bytes(5).unwrap());
        assert!(workspace.working_bytes() < model.workspace(5).unwrap().working_bytes() / 4);
        let y = Array2::from_shape_fn((128, 5).f(), |(i, j)| {
            1048576. * (j + 1) as f64 + if i % 2 == 0 { 0.25 } else { -0.25 }
        });
        let mut kept = None;
        for width in [5, 1, 3, 5] {
            let summary = plan
                .run_into(y.slice(s![..;-1,..width]), &mut workspace)
                .unwrap();
            for j in 0..width {
                close(
                    summary.estimates()[(0, j)],
                    1048576. * (j + 1) as f64,
                    1e-13,
                );
            }
            kept = Some(summary.to_owned());
        }
        let bad = Array2::from_elem((128, 5), f64::NAN);
        assert!(plan.run_into(bad.view(), &mut workspace).is_err());
        assert!(plan.run_into(y.view(), &mut workspace).is_ok());
        let another = model
            .prepare_outputs(&spec(array![[2.]], Uncertainty::None))
            .unwrap();
        assert!(another.run_into(y.view(), &mut workspace).is_err());
        kept.unwrap()
    };
    close(retained.estimates()[(0, 4)], 5. * 1048576., 1e-13);
}

#[test]
fn requests_preserve_estimability_and_allow_zero_scale() {
    let x = array![[1., 0., 0.], [1., 1., 1.], [1., 2., 2.], [1., 3., 3.]];
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let invalid = OutputSpec::new(
        EstimandSet::coefficients(vec!["unidentified".into()], vec![1], 3).unwrap(),
    );
    assert!(matches!(
        model.prepare_outputs(&invalid),
        Err(FitError::NotEstimable)
    ));
    let plan = model
        .prepare_outputs(&spec(
            array![[0., 1., 1.], [0., 0., 0.]],
            Uncertainty::Joint,
        ))
        .unwrap();
    let mut ws = plan.workspace(1).unwrap();
    let summary = plan
        .run_into(Array2::zeros((4, 1)).view(), &mut ws)
        .unwrap();
    let SummaryView::Joint(v) = summary else {
        panic!("joint")
    };
    assert_eq!(v.standard_error(0, 0).unwrap(), 0.);
    assert_eq!(v.covariance(0, 1, 0).unwrap(), 0.);
    assert!(
        EstimandSet::contrasts(vec!["same".into(), "same".into()], array![[1.], [2.]]).is_err()
    );
    assert!(EstimandSet::contrasts(vec!["  ".into()], array![[1.]]).is_err());
    assert!(EstimandSet::contrasts(vec!["bad".into()], array![[f64::NAN]]).is_err());
    assert!(EstimandSet::coefficients(vec!["bad".into()], vec![3], 3).is_err());
    assert!(plan.workspace(usize::MAX).is_err());
    assert!(plan.workspace(0).is_err());
}

#[test]
fn estimates_only_does_not_compute_overflowing_residual_squares() {
    let x = Array2::ones((8, 1));
    let y = Array2::from_shape_fn((8, 1), |(i, _)| if i % 2 == 0 { 1e200 } else { -1e200 });
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    assert!(model.fit(y.view()).is_err());
    let plan = model
        .prepare_outputs(&spec(array![[1.]], Uncertainty::None))
        .unwrap();
    let mut ws = plan.workspace(1).unwrap();
    assert!(plan.run_into(y.view(), &mut ws).unwrap().estimates()[(0, 0)].is_finite());
}

struct OffsetPlan;
impl WhiteningPlan for OffsetPlan {
    fn observations(&self) -> usize {
        8
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> fmri_noise::Result<Array2<f64>> {
        // Custom plan contract permits preprocessing that accepts invalid raw
        // entries; output validation belongs after the transform in GLS.
        Ok(values.mapv(|v| if v.is_finite() { v * 2. } else { 0. }))
    }
}
#[test]
fn custom_transform_compatibility_and_transformed_validation() {
    let x = Array2::ones((8, 1));
    let y = Array2::from_shape_fn(
        (8, 2),
        |(i, j)| if i == 0 { f64::NAN } else { (i + j) as f64 },
    );
    let model = GlsModel::new(x.view(), OffsetPlan, 1e-12).unwrap();
    let expected = model.fit(y.view()).unwrap();
    for u in [Uncertainty::None, Uncertainty::Marginal] {
        let plan = model.prepare_outputs(&spec(array![[1.]], u)).unwrap();
        let mut ws = plan.workspace(2).unwrap();
        let summary = plan.run_into(y.view(), &mut ws).unwrap();
        for j in 0..2 {
            close(
                summary.estimates()[(0, j)],
                expected.coefficients()[(0, j)],
                1e-12,
            );
        }
    }
}

#[test]
fn current_r_fixture_matches_requested_effects_and_standard_errors() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/estimation.json")).unwrap();
    let matrix = |v: &serde_json::Value| {
        let rows = v.as_array().unwrap();
        Array2::from_shape_fn((rows.len(), rows[0].as_array().unwrap().len()), |(i, j)| {
            rows[i][j].as_f64().unwrap()
        })
    };
    let x = matrix(&f["X"]);
    let y = matrix(&f["Y"]);
    let c = matrix(&f["C"]);
    let model = GlsModel::new(
        x.view(),
        Ar1Plan::new(vec![12, 12], vec![0.4, -0.25], FirstSample::Stationary).unwrap(),
        1e-12,
    )
    .unwrap();
    let plan = model
        .prepare_outputs(&spec(c, Uncertainty::Marginal))
        .unwrap();
    let mut ws = plan.workspace(y.ncols()).unwrap();
    let summary = plan.run_into(y.view(), &mut ws).unwrap();
    let SummaryView::Marginal(v) = &summary else {
        panic!("marginal")
    };
    for j in 0..y.ncols() {
        close(
            summary.estimates()[(0, j)],
            f["gls"]["effect"][j].as_f64().unwrap(),
            2e-11,
        );
        close(
            v.standard_error(0, j).unwrap(),
            f["gls"]["se"][j].as_f64().unwrap(),
            2e-11,
        );
    }
}

#[test]
fn near_collinear_estimand_variance_uses_stable_geometry() {
    // X=[u,u+epsilon*v], with orthogonal Walsh u,v. The estimand beta0+beta1
    // is the coefficient of u: unit variance 1/8 even when individual betas
    // have enormous variance. A third orthogonal Walsh vector supplies RSS=8.
    // Expanded L V L' loses this small variance by subtracting O(epsilon^-2).
    for exponent in [20, 28, 30] {
        let epsilon = 2f64.powi(-exponent);
        let x = Array2::from_shape_fn((8, 2), |(i, j)| {
            if j == 0 {
                1.
            } else {
                1. + epsilon * if i % 2 == 0 { 1. } else { -1. }
            }
        });
        let y = Array2::from_shape_fn((8, 1), |(i, _)| if (i / 2) % 2 == 0 { 1. } else { -1. });
        for backend in backends() {
            let model = LinearModel::new(x.view(), 1e-12)
                .unwrap()
                .with_backend(backend);
            assert_eq!(model.rank(), 2);
            let mut marginal_variance = 0.;
            for level in [Uncertainty::Marginal, Uncertainty::Joint] {
                let plan = model
                    .prepare_outputs(&spec(array![[1., 1.], [2., 2.]], level))
                    .unwrap();
                let mut ws = plan.workspace(1).unwrap();
                let result = plan.run_into(y.view(), &mut ws).unwrap();
                match result {
                    SummaryView::Marginal(v) => {
                        marginal_variance = v.variance(0, 0).unwrap();
                        close(marginal_variance, 1. / 6., 2e-6);
                    }
                    SummaryView::Joint(v) => {
                        close(v.variance(0, 0).unwrap(), marginal_variance, 1e-12);
                        close(
                            v.covariance(0, 1, 0).unwrap(),
                            2. * marginal_variance,
                            1e-12,
                        );
                        close(v.variance(1, 0).unwrap(), 4. * marginal_variance, 1e-12);
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn output_context_copies_only_logical_sliced_weights() {
    let mut backing = Array2::zeros((10_000, 2));
    backing[(5000, 0)] = 1.;
    let weights = backing.slice_move(s![5000..5001, ..]);
    let request = spec(weights, Uncertainty::None);
    assert_eq!(
        request
            .clone()
            .estimands()
            .weights()
            .clone()
            .into_raw_vec_and_offset()
            .0
            .len(),
        2
    );
    let model = LinearModel::new(array![[1., 0.], [1., 1.], [1., 2.]].view(), 1e-12).unwrap();
    let plan = model.prepare_outputs(&request).unwrap();
    assert_eq!(
        plan.spec()
            .estimands()
            .weights()
            .clone()
            .into_raw_vec_and_offset()
            .0
            .len(),
        2
    );
}

#[test]
fn retained_fit_is_model_bound_and_matches_analytic_uncertainty() {
    let x = ndarray::array![[1., -1.], [1., -1.], [1., 1.], [1., 1.]];
    let y = ndarray::array![[0.5, 1.], [1.5, 3.], [2.5, 5.], [3.5, 7.]];
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let mut workspace = model.workspace(2).unwrap();
    let retained = model
        .fit_into(y.view(), &mut workspace)
        .unwrap()
        .retain_for_outputs()
        .unwrap();
    assert_eq!(retained.numeric_bytes(), 48);
    // Move the model: opaque identity must remain valid.
    let moved = Box::new(model);
    for level in [Uncertainty::None, Uncertainty::Marginal, Uncertainty::Joint] {
        let request = OutputSpec::new(
            EstimandSet::contrasts(
                vec!["slope".into(), "twice".into(), "zero".into()],
                ndarray::array![[0., 1.], [0., 2.], [0., 0.]],
            )
            .unwrap(),
        )
        .with_uncertainty(level);
        let plan = moved.prepare_outputs(&request).unwrap();
        let result = plan.summarize_retained(&retained, 0..2).unwrap();
        assert!((result.estimates()[(0, 0)] - 1.).abs() < 1e-12);
        assert!((result.estimates()[(0, 1)] - 2.).abs() < 1e-12);
        assert_eq!(result.estimates()[(2, 0)], 0.);
        match result {
            Summary::Marginal(r) => assert!((r.variance(0, 0).unwrap() - 0.125).abs() < 1e-12),
            Summary::Joint(r) => assert!((r.covariance(0, 1, 0).unwrap() - 0.25).abs() < 1e-12),
            _ => {}
        }
        let other = LinearModel::new(x.view(), 1e-12).unwrap();
        assert!(other
            .prepare_outputs(&request)
            .unwrap()
            .summarize_retained(&retained, 0..2)
            .is_err());
        assert!(plan.summarize_retained(&retained, 0..3).is_err());
    }
}
