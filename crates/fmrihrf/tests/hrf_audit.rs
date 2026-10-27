use fmrihrf::{hrf::*, penalty::penalty_matrix, reconstruction::*, Hrf};
use ndarray::{array, Array1, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/hrf_audit.json")).unwrap()
}
fn vector(v: &Value) -> Array1<f64> {
    if let Some(a) = v.as_array() {
        Array1::from_iter(a.iter().map(|v| v.as_f64().unwrap()))
    } else {
        array![v.as_f64().unwrap()]
    }
}
fn matrix(v: &Value) -> Array2<f64> {
    let a = v.as_array().unwrap();
    Array2::from_shape_fn((a.len(), a[0].as_array().unwrap().len()), |(i, j)| {
        a[i][j].as_f64().unwrap()
    })
}
fn close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
    assert_eq!(a.dim(), b.dim());
    for (a, b) in a.iter().zip(b) {
        assert!((a - b).abs() < tol * (1.0 + b.abs()), "{a} != {b}");
    }
}
fn basis(name: &str) -> Box<dyn Hrf> {
    match name {
        "spmg1" => Box::new(SpmG1Hrf::new(5., 15., 1. / 120.)),
        "spmg2" => Box::new(SpmG2Hrf::new(5., 15., 1. / 120.)),
        "spmg3" => Box::new(SpmG3Hrf::new(5., 15., 1. / 120.)),
        "fourier5" => fourier_hrf(5, Some(24.)),
        "fourier6" => fourier_hrf(6, Some(24.)),
        "fir" => fir_hrf(5, Some(24.)),
        "bspline" => bspline_hrf(5, Some(3), Some(24.)),
        "tent" => tent_hrf(5, Some(24.)),
        _ => panic!(),
    }
}
#[test]
fn current_r_basis_penalty_and_reconstruction_contracts() {
    let f = fixture();
    let t = vector(&f["times"]);
    for c in f["cases"].as_array().unwrap() {
        let h = basis(c["name"].as_str().unwrap());
        let m = reconstruction_matrix(h.as_ref(), &t).unwrap();
        close(&m, &matrix(&c["values"]), 2e-10);
        close(
            &penalty_matrix(h.as_ref(), 2, None).unwrap(),
            &matrix(&c["penalty"]),
            1e-12,
        );
        let got = reconstruct_hrf(&m, &vector(&c["coefficients"])).unwrap();
        for (a, b) in got.iter().zip(vector(&c["reconstruction"])) {
            assert!((a - b).abs() < 2e-10 * (1.0 + b.abs()));
        }
    }
}
#[test]
fn empirical_linear_default_and_explicit_cubic() {
    let f = fixture();
    let e = &f["empirical"];
    let t = vector(&e["times"]);
    let y = vector(&e["values"]);
    let h = empirical_hrf(&t, &y, Some("spmg2"), None).unwrap();
    let actual = h.evaluate(&vector(&f["times"])).unwrap();
    for (a, b) in actual.column(0).iter().zip(vector(&e["expected"])) {
        assert!((a - b).abs() < 1e-12);
    }
    // A deceptive display name must not select a derivative penalty.
    assert_eq!(penalty_matrix(h.as_ref(), 2, None).unwrap(), array![[1.]]);
    let cubic = empirical_hrf_with_method(&t, &y, None, None, InterpolationMethod::Cubic).unwrap();
    close(
        &cubic.evaluate(&t).unwrap(),
        &y.clone().insert_axis(ndarray::Axis(1)),
        1e-10,
    );
    assert!(
        (cubic.evaluate(&array![2.]).unwrap()[(0, 0)] - h.evaluate(&array![2.]).unwrap()[(0, 0)])
            .abs()
            > 1e-3
    );
    assert!(empirical_hrf(&array![0., 1., 1.], &array![0., 1., 2.], None, None).is_err());
    assert!(empirical_hrf(&t, &y, None, Some(f64::NAN)).is_err());
    assert!(h.evaluate(&array![f64::NAN]).is_err());
}
#[test]
fn derivatives_dispatch_through_trait_objects_and_respect_endpoint() {
    let f = fixture();
    let t = vector(&f["derivative_times"]);
    for name in ["spmg1", "spmg2"] {
        let h = basis(name);
        let expected = if name == "spmg1" {
            vector(&f["derivatives"][name]).insert_axis(ndarray::Axis(1))
        } else {
            matrix(&f["derivatives"][name])
        };
        close(
            &deriv(h.as_ref(), &t, DerivativeMethod::Analytic).unwrap(),
            &expected,
            2e-10,
        );
        close(
            &h.evaluate_deriv(&t, DerivativeMethod::Auto).unwrap(),
            &expected,
            2e-10,
        );
    }
    let h = empirical_hrf(&array![0., 1., 2.], &array![0., 1., 2.], None, None).unwrap();
    close(
        &numeric_derivative(h.as_ref(), &array![0., 0.5, 2.]).unwrap(),
        &Array2::ones((3, 1)),
        1e-9,
    );
    assert!(reconstruct_hrf(&array![[1.]], &array![f64::NAN]).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn portable_regressor_roundtrip_preserves_actual_response() {
    use fmrihrf::regressor::Regressor;
    for h in [
        Box::new(SpmG3Hrf::new(6., 17., 0.01)) as Box<dyn Hrf>,
        empirical_hrf(
            &array![0., 1., 3., 8.],
            &array![0., 2., 0.4, 0.],
            None,
            None,
        )
        .unwrap(),
        spm_canonical(None, None, None, None, None, Some(true)),
    ] {
        let reg =
            Regressor::new(vec![0.3, 10.7], h, Some(vec![0., 1.2]), None, None, None).unwrap();
        let encoded = serde_json::to_string(&reg.to_data()).unwrap();
        let restored = Regressor::from_data(serde_json::from_str(&encoded).unwrap()).unwrap();
        let t = Array1::linspace(0., 32., 129);
        close(
            &reg.evaluate(&t, None, None).unwrap(),
            &restored.evaluate(&t, None, None).unwrap(),
            1e-12,
        );
        let mut legacy = reg.to_data();
        legacy.hrf_recipe = None;
        assert!(Regressor::from_data(legacy).is_err());
    }
}
