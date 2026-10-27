use fmri_noise::{Ar1Plan, FirstSample};
use ndarray::{array, Array2};
#[test]
fn independent_stationary_covariance_whitens_to_innovation_identity() {
    for phi in [-0.8, 0., 0.6, 0.99] {
        let plan = Ar1Plan::new(vec![7], vec![phi], FirstSample::Stationary).unwrap();
        let w = plan.apply(Array2::eye(7).view()).unwrap();
        let sigma = Array2::from_shape_fn((7, 7), |(i, j)| {
            phi.powi(i.abs_diff(j) as i32) / (1. - phi * phi)
        });
        let whitened = w.dot(&sigma).dot(&w.t());
        for ((i, j), v) in whitened.indexed_iter() {
            assert!((v - if i == j { 1. } else { 0. }).abs() < 1e-12);
        }
    }
}
#[test]
fn boundaries_and_resets_do_not_leak_and_do_not_drop_rows() {
    let plan = Ar1Plan::new(vec![2, 3], vec![0.5, -0.5], FirstSample::ZeroHistory)
        .unwrap()
        .with_resets(&[4])
        .unwrap();
    let y = array![[10.], [2.], [100.], [4.], [8.]];
    assert_eq!(
        plan.apply(y.view()).unwrap(),
        array![[10.], [-3.], [100.], [54.], [8.]]
    );
    assert!(Ar1Plan::new(vec![0], vec![0.5], FirstSample::Stationary).is_err());
    assert!(Ar1Plan::new(vec![2], vec![1.], FirstSample::Stationary).is_err());
    assert!(plan.apply(array![[1.]].view()).is_err());
    assert!(plan.with_resets(&[5]).is_err());
}
#[test]
fn source_built_fmriar_ar1_estimation_matches_with_censoring_and_offsets() {
    use fmri_noise::{estimate_ar1_uncorrected, Pooling};
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/estimation.json")).unwrap();
    let rows = f["noise"].as_array().unwrap();
    let noise = Array2::from_shape_fn((rows.len(), 2), |(i, j)| rows[i][j].as_f64().unwrap());
    let mut valid = vec![true; rows.len()];
    for i in f["censor"].as_array().unwrap() {
        valid[i.as_u64().unwrap() as usize - 1] = false;
    }
    for case in f["cases"].as_array().unwrap() {
        let pool = if case["pooling"] == "run" {
            Pooling::Run
        } else {
            Pooling::Global
        };
        let plan = estimate_ar1_uncorrected(
            noise.view(),
            vec![12, 12],
            Some(&valid),
            pool,
            FirstSample::Stationary,
        )
        .unwrap();
        let expected: Vec<f64> = if let Some(v) = case["phi"].as_array() {
            v.iter().map(|x| x.as_f64().unwrap()).collect()
        } else {
            vec![case["phi"].as_f64().unwrap(); 2]
        };
        for (a, b) in plan.coefficients().iter().zip(expected) {
            assert!((a - b).abs() < 1e-12, "{a} != {b}");
        }
    }
    assert!(estimate_ar1_uncorrected(
        Array2::ones((4, 1)).view(),
        vec![4],
        None,
        Pooling::Run,
        FirstSample::Stationary
    )
    .is_err());
}
#[test]
fn low_rank_bias_maps_and_corrected_estimates_match_independent_dense_r() {
    use fmri_noise::{Ar1Estimator, Pooling};
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/correction.json")).unwrap();
    let matrix = |v: &serde_json::Value| {
        let rows = v.as_array().unwrap();
        Array2::from_shape_fn((rows.len(), rows[0].as_array().unwrap().len()), |(i, j)| {
            rows[i][j].as_f64().unwrap()
        })
    };
    let x = matrix(&f["design"]);
    let residuals = matrix(&f["residuals"]);
    let mut valid = vec![true; x.nrows()];
    for c in f["censor"].as_array().unwrap() {
        valid[c.as_u64().unwrap() as usize - 1] = false;
    }
    let estimator =
        Ar1Estimator::new(x.view(), vec![60, 60], Some(&valid), 8, Pooling::Run).unwrap();
    for (run, a) in estimator.bias_matrices().iter().enumerate() {
        let expected = matrix(&f["bias"][(run + 1).to_string()]);
        assert_eq!(a.dim(), expected.dim());
        for (a, b) in a.iter().zip(&expected) {
            assert!((a - b).abs() < 3e-12, "{a} != {b}");
        }
    }
    let plan = estimator.fit(residuals.view()).unwrap();
    for (run, p) in plan.coefficients().iter().enumerate() {
        let expected = f["phi"][run].as_f64().unwrap();
        assert!((p - expected).abs() < 2e-11, "{p} != {expected}");
    }
    assert!(estimator.fit(x.view()).is_err());
}
