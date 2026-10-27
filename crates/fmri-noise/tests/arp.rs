use fmri_noise::{estimate_ar_uncorrected, ArEstimator, ArPlan, ArStart, Pooling};
use ndarray::{s, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/arp.json")).unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    if let Some(a) = v.as_array() {
        a.iter().map(|v| v.as_f64().unwrap()).collect()
    } else {
        vec![v.as_f64().unwrap()]
    }
}
fn matrix(v: &Value) -> Array2<f64> {
    let rows = v.as_array().unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].as_array().unwrap().len()), |(i, j)| {
        rows[i][j].as_f64().unwrap()
    })
}
fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < tol * (1.0 + b.abs()), "{a} != {b}");
    }
}
#[test]
fn current_fmriar_fixed_orders_corrected_and_uncorrected() {
    let f = fixture();
    let x = matrix(&f["X"]);
    let noise = matrix(&f["noise"]);
    let resid = matrix(&f["residuals"]);
    let mut valid = vec![true; x.nrows()];
    for i in f["censor"].as_array().unwrap() {
        valid[i.as_u64().unwrap() as usize - 1] = false;
    }
    for case in f["cases"].as_array().unwrap() {
        let p = case["order"].as_u64().unwrap() as usize;
        let pooling = if case["pooling"] == "run" {
            Pooling::Run
        } else {
            Pooling::Global
        };
        let plan = if case["corrected"].as_bool().unwrap() {
            ArEstimator::new(
                x.view(),
                vec![80, 80],
                Some(&valid),
                p,
                10,
                pooling,
                ArStart::ZeroHistory,
            )
            .unwrap()
            .fit(resid.view())
            .unwrap()
        } else {
            estimate_ar_uncorrected(
                noise.view(),
                vec![80, 80],
                Some(&valid),
                p,
                pooling,
                ArStart::ZeroHistory,
            )
            .unwrap()
        };
        let expected = case["phi"].as_array().unwrap();
        for (run, phi) in plan.coefficients().iter().enumerate() {
            let expected = vector(&expected[if expected.len() == 1 { 0 } else { run }]);
            for (a, b) in phi.iter().zip(expected) {
                assert!(
                    (a - b).abs() < 2e-9,
                    "p={p}, corrected={}, pooling={}, {a} != {b}",
                    case["corrected"],
                    case["pooling"]
                );
            }
        }
        close(&plan.apply(x.view()).unwrap(), &matrix(&case["Xw"]), 2e-9);
        close(
            &plan.apply(noise.view()).unwrap(),
            &matrix(&case["Yw"]),
            2e-9,
        );
        assert_eq!(plan.diagnostics().unwrap().len(), 2);
    }
}
#[test]
fn startup_matches_dense_stationary_covariance_and_r_conditional_filter() {
    let f = fixture();
    for c in f["known"].as_array().unwrap() {
        let phi = vector(&c["phi"]);
        let n = c["n"].as_u64().unwrap() as usize;
        let starts: Vec<_> = c["resets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as usize)
            .collect();
        let stationary = ArPlan::new(vec![n], vec![phi.clone()], ArStart::Stationary)
            .unwrap()
            .with_resets(&starts)
            .unwrap();
        let w = stationary.apply(Array2::eye(n).view()).unwrap();
        close(&w, &matrix(&c["stationary"]), 2e-10);
        close(
            &w.dot(&matrix(&c["covariance"])).dot(&w.t()),
            &Array2::eye(n),
            2e-10,
        );
        let zero = ArPlan::new(vec![n], vec![phi], ArStart::ZeroHistory)
            .unwrap()
            .with_resets(&starts)
            .unwrap();
        close(
            &zero.apply(Array2::eye(n).view()).unwrap(),
            &matrix(&c["zero"]),
            2e-12,
        );
    }
}
#[test]
fn invalid_order_stationarity_and_no_boundary_leakage() {
    assert!(ArPlan::new(vec![4], vec![vec![1.1]], ArStart::Stationary).is_err());
    assert!(ArPlan::new(vec![4], vec![vec![0.0, 1.0]], ArStart::ZeroHistory).is_err());
    assert!(ArPlan::new(vec![4], vec![vec![f64::NAN]], ArStart::Stationary).is_err());
    let y = Array2::from_shape_fn((12, 2), |(i, j)| ((i + 1) * (j + 1)) as f64);
    let plan = ArPlan::new(
        vec![6, 6],
        vec![vec![0.6, -0.2], vec![-0.4, 0.1]],
        ArStart::Stationary,
    )
    .unwrap();
    let full = plan.apply(y.view()).unwrap();
    let second = ArPlan::new(vec![6], vec![vec![-0.4, 0.1]], ArStart::Stationary)
        .unwrap()
        .apply(y.slice(s![6.., ..]))
        .unwrap();
    close(&full.slice(s![6.., ..]).to_owned(), &second, 1e-12);
    let mask: Vec<_> = (0..12).map(|i| i % 2 == 0).collect();
    assert!(estimate_ar_uncorrected(
        y.view(),
        vec![12],
        Some(&mask),
        2,
        Pooling::Run,
        ArStart::ZeroHistory
    )
    .is_err());
    assert!(estimate_ar_uncorrected(
        Array2::ones((12, 2)).view(),
        vec![12],
        None,
        2,
        Pooling::Run,
        ArStart::ZeroHistory
    )
    .is_err());
    let x = Array2::ones((12, 1));
    assert!(ArEstimator::new(
        x.view(),
        vec![12],
        None,
        5,
        2,
        Pooling::Run,
        ArStart::Stationary
    )
    .is_err());
    assert!(ArEstimator::new(
        x.view(),
        vec![12],
        Some(&mask),
        2,
        5,
        Pooling::Run,
        ArStart::Stationary
    )
    .is_err());
}
