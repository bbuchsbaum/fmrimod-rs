use fmri_noise::{
    estimate_hannan_rissanen, estimate_parcel_ar, estimate_residual_innovation_scale,
    select_ar_order_conditional, ArStart, ArmaPlan, InnovationScaled, OrderCriterion,
    WhiteningPlan,
};
use ndarray::{array, Array2, ArrayView2, ArrayViewMut2};
use serde_json::Value;

fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol * (1. + y.abs()), "{x} != {y}");
    }
}

#[test]
fn conditional_order_selection_is_comparable_and_auditable() {
    let y: Vec<f64> = (0..80)
        .scan((0.0, 0.0), |state, t| {
            let e = ((t * 37 % 17) as f64 - 8.0) / 9.;
            let next = 0.65 * state.0 - 0.2 * state.1 + e;
            state.1 = state.0;
            state.0 = next;
            Some(next)
        })
        .collect();
    for criterion in [OrderCriterion::Aic, OrderCriterion::Bic] {
        let selected = select_ar_order_conditional(&y, 5, criterion).unwrap();
        assert!(selected.selected <= 5);
        assert!(selected.candidates.iter().all(|c| c.observations == 75));
        assert_eq!(selected.candidates[0].order, 0); // IID is visible, not a fallback.
        assert!(selected
            .candidates
            .iter()
            .any(|c| c.order == 2 && c.coefficients.is_some()));
    }
    assert!(select_ar_order_conditional(&[1., 1., 1., 1., 1.], 2, OrderCriterion::Bic).is_err());
}

#[test]
fn conditional_selection_is_scale_relative_and_rejects_giant_orders() {
    let y: Vec<f64> = (0..80)
        .map(|t| ((t * 19 % 31) as f64 - 15.) / 11.)
        .collect();
    let base = select_ar_order_conditional(&y, 4, OrderCriterion::Bic).unwrap();
    let scaled: Vec<_> = y.iter().map(|x| x * 1e-100).collect();
    let tiny = select_ar_order_conditional(&scaled, 4, OrderCriterion::Bic).unwrap();
    assert_eq!(base.selected, tiny.selected);
    assert!(select_ar_order_conditional(&y, usize::MAX, OrderCriterion::Bic).is_err());
    assert!(estimate_hannan_rissanen(&y, usize::MAX, 1, 4, 0).is_err());
    assert!(estimate_hannan_rissanen(&y, 1, usize::MAX, 4, 0).is_err());
    assert!(estimate_hannan_rissanen(&y, 1, 1, usize::MAX, 0).is_err());
}

#[test]
fn arma_zero_history_is_independent_by_run_and_matches_scalar_recursion() {
    let plan = ArmaPlan::new(
        vec![4, 4],
        vec![vec![0.4], vec![-0.2]],
        vec![vec![0.3], vec![0.15]],
    )
    .unwrap()
    .with_resets(&[2])
    .unwrap();
    let x = array![[1.], [2.], [4.], [8.], [3.], [6.], [12.], [24.]];
    let actual = plan.apply(x.view()).unwrap();
    let expected = array![
        [1.],
        [1.3],
        [4.],
        [5.2],
        [3.],
        [6.15],
        [12.2775],
        [24.558375]
    ];
    close(&actual, &expected, 1e-13);
    let (dx, dy) = plan.apply_pair(x.view(), x.view()).unwrap();
    close(&dx, &dy, 0.);
    assert!(ArmaPlan::new(vec![4], vec![vec![1.]], vec![vec![]]).is_err());
    assert!(ArmaPlan::new(vec![4], vec![vec![]], vec![vec![1.]]).is_err());
}

#[test]
fn hannan_rissanen_is_explicit_single_series_and_refuses_bad_orders() {
    let y: Vec<f64> = (0..120)
        .scan((0., 0.), |s, t| {
            let e = ((t * 13 % 23) as f64 - 11.) / 8.;
            let v = 0.35 * s.0 + e + 0.25 * s.1;
            s.1 = e;
            s.0 = v;
            Some(v)
        })
        .collect();
    let fit = estimate_hannan_rissanen(&y, 1, 1, 8, 1).unwrap();
    assert_eq!(fit.observations, 112);
    assert!(fit.innovation_variance > 0.);
    assert!(estimate_hannan_rissanen(&y, 1, 1, 119, 0).is_err());
    assert!(estimate_hannan_rissanen(&[1.; 20], 1, 1, 4, 0).is_err());
    let explosive: Vec<_> = (0..40).map(|t| 1.2_f64.powi(t)).collect();
    assert!(estimate_hannan_rissanen(&explosive, 1, 1, 4, 0).is_err());
}

#[test]
fn parcel_plan_requires_matched_design_and_preserves_columns() {
    let residuals = Array2::from_shape_fn((16, 4), |(t, c)| ((t * 5 + c * 3) % 11) as f64 - 5.);
    let plan = estimate_parcel_ar(
        residuals.view(),
        vec![8, 8],
        vec![7, 7, 9, 9],
        1,
        0.25,
        ArStart::ZeroHistory,
    )
    .unwrap();
    let x = Array2::from_shape_fn((16, 2), |(t, c)| (t + c) as f64 / 7.);
    let out = plan.apply_matched(x.view(), residuals.view()).unwrap();
    assert_eq!(out.parcel_ids, &[7, 9]);
    assert_eq!(out.designs.len(), 2);
    assert_eq!(out.response.dim(), residuals.dim());
    assert!(plan
        .apply_matched(x.slice(ndarray::s![..15, ..]), residuals.view())
        .is_err());
    assert!(estimate_parcel_ar(
        residuals.view(),
        vec![8, 8],
        vec![7, 7, 9, 9],
        1,
        1.1,
        ArStart::ZeroHistory
    )
    .is_err());
}

#[test]
fn parcel_acvf_pools_features_not_means_and_weights_global_by_feature_count() {
    let base: Vec<f64> = (0..24).map(|t| ((t + 1) as f64 * 1.3).sin()).collect();
    let residuals = Array2::from_shape_fn((24, 3), |(t, c)| match c {
        0 => base[t],
        1 => -base[t],
        _ => ((t + 1) as f64 * 2.2).sin(),
    });
    // Columns 0 and 1 cancel exactly in their parcel mean but each has positive
    // variance; feature pooling therefore remains estimable.
    let local = estimate_parcel_ar(
        residuals.view(),
        vec![24],
        vec![1, 1, 2],
        1,
        0.,
        ArStart::ZeroHistory,
    )
    .unwrap();
    let global = estimate_parcel_ar(
        residuals.view(),
        vec![24],
        vec![1, 1, 2],
        1,
        1.,
        ArStart::ZeroHistory,
    )
    .unwrap();
    let phi_a = local.coefficients(1).unwrap()[0][0];
    let phi_b = local.coefficients(2).unwrap()[0][0];
    let f: Value = serde_json::from_str(include_str!("fixtures/noise_extensions.json")).unwrap();
    let g1 = f["r_pooled_feature_acvf"]["parcel_1"].as_array().unwrap();
    let gg = f["r_pooled_feature_acvf"]["global_feature_weighted"]
        .as_array()
        .unwrap();
    assert!((phi_a - g1[1].as_f64().unwrap() / g1[0].as_f64().unwrap()).abs() < 2e-12);
    assert!((phi_a - phi_b).abs() > 1e-4);
    // lambda = 1 is one feature-count weighted global ACVF for every parcel.
    assert!(
        (global.coefficients(1).unwrap()[0][0] - global.coefficients(2).unwrap()[0][0]).abs()
            < 1e-12
    );
    assert!(
        (global.coefficients(1).unwrap()[0][0] - gg[1].as_f64().unwrap() / gg[0].as_f64().unwrap())
            .abs()
            < 2e-12
    );
    assert!((global.coefficients(1).unwrap()[0][0] - phi_a).abs() > 1e-4);
}

#[test]
fn innovation_scale_is_known_or_explicit_plugin_only() {
    let plan = ArmaPlan::new(vec![3, 3], vec![vec![], vec![]], vec![vec![], vec![]]).unwrap();
    let scaled = InnovationScaled::new(plan.clone(), vec![3, 3], vec![2., 4.]).unwrap();
    let x = Array2::from_elem((6, 1), 8.);
    let expected = array![[4.], [4.], [4.], [2.], [2.], [2.]];
    close(&scaled.apply(x.view()).unwrap(), &expected, 0.);
    let diag = estimate_residual_innovation_scale(&plan, x.view(), &[3, 3]).unwrap();
    assert_eq!(diag.innovation_sd, vec![8., 8.]);
    assert_eq!(diag.method, "residual RMS plug-in; no uncertainty estimate");
}

#[derive(Clone)]
struct NonFinitePlan;
impl WhiteningPlan for NonFinitePlan {
    fn observations(&self) -> usize {
        2
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> fmri_noise::Result<Array2<f64>> {
        Ok(Array2::from_elem(values.raw_dim(), f64::NAN))
    }
    fn apply_into(
        &self,
        _values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> fmri_noise::Result<()> {
        out.fill(f64::NAN);
        Ok(())
    }
}

#[test]
fn specialized_apply_into_reuses_output_and_rejects_bad_inner_or_scaling() {
    let plan = ArmaPlan::new(vec![2], vec![vec![0.2]], vec![vec![0.1]]).unwrap();
    let x = array![[2.], [4.]];
    let mut out = Array2::zeros((2, 1));
    plan.apply_into(x.view(), out.view_mut()).unwrap();
    close(&out, &plan.apply(x.view()).unwrap(), 0.);
    let scaled = InnovationScaled::new(plan, vec![2], vec![2.]).unwrap();
    scaled.apply_into(x.view(), out.view_mut()).unwrap();
    close(&out, &scaled.apply(x.view()).unwrap(), 0.);
    let bad = InnovationScaled::new(NonFinitePlan, vec![2], vec![1.]).unwrap();
    assert!(bad.apply_into(x.view(), out.view_mut()).is_err());
    assert!(estimate_residual_innovation_scale(&NonFinitePlan, x.view(), &[2]).is_err());
    let overflow = InnovationScaled::new(
        ArmaPlan::new(vec![2], vec![vec![]], vec![vec![]]).unwrap(),
        vec![2],
        vec![f64::MIN_POSITIVE],
    )
    .unwrap();
    assert!(overflow
        .apply(Array2::from_elem((2, 1), f64::MAX).view())
        .is_err());
}

#[test]
fn fixture_has_fresh_source_fingerprints() {
    let f: Value = serde_json::from_str(include_str!("fixtures/noise_extensions.json")).unwrap();
    assert_eq!(
        f["reference"]["commit"],
        "ca27f2772bb4675573fae717a992c993c2f986c8"
    );
    assert!(
        f["reference"]["sha256"]["R/hr_arma.R"]
            .as_str()
            .unwrap()
            .len()
            == 64
    );
}

#[test]
fn independent_r_zero_history_witness_matches_known_arma_plan() {
    let f: Value = serde_json::from_str(include_str!("fixtures/noise_extensions.json")).unwrap();
    let expected: Vec<f64> = f["r_arma_zero_history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let x = Array2::from_shape_vec((8, 1), vec![1., 2., 4., 8., 3., 6., 12., 24.]).unwrap();
    let plan = ArmaPlan::new(vec![8], vec![vec![0.4]], vec![vec![0.3]])
        .unwrap()
        .with_resets(&[2, 4])
        .unwrap();
    for (actual, expected) in plan.apply(x.view()).unwrap().iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-12);
    }
}

#[test]
fn full_hr_estimate_and_all_order_scores_match_independent_r_qr() {
    let f: Value = serde_json::from_str(include_str!("fixtures/noise_extensions.json")).unwrap();
    let y = f["series"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect::<Vec<_>>();
    for (criterion, key) in [
        (OrderCriterion::Aic, "r_scores_aic"),
        (OrderCriterion::Bic, "r_scores_bic"),
    ] {
        let selection = select_ar_order_conditional(&y, 5, criterion).unwrap();
        for (row, c) in f[key].as_array().unwrap().iter().zip(selection.candidates) {
            assert!(c.failure.is_none());
            assert!((c.innovation_variance - row[2].as_f64().unwrap()).abs() < 1e-12);
            assert!((c.score - row[3].as_f64().unwrap()).abs() < 1e-10);
        }
    }
    let fit = estimate_hannan_rissanen(&y, 1, 1, 8, 1).unwrap();
    let expected = &f["r_hr"];
    assert!((fit.ar[0] - expected["phi"].as_f64().unwrap()).abs() < 1e-10);
    assert!((fit.ma[0] - expected["theta"].as_f64().unwrap()).abs() < 1e-10);
    assert!((fit.innovation_variance - expected["sigma2"].as_f64().unwrap()).abs() < 1e-10);
}
