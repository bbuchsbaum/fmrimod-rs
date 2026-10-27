use fmrihrf::hrf::bspline_hrf;
use ndarray::Array1;
#[test]
fn current_r_endpoint_constrained_splines_across_degrees_and_sizes() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/current_splines.json")).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let n = c["n"].as_u64().unwrap() as usize;
        let degree = c["degree"].as_u64().unwrap() as usize;
        let times = Array1::from_iter(
            c["time"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap()),
        );
        let values = bspline_hrf(n, Some(degree), Some(24.))
            .evaluate(&times)
            .unwrap();
        assert_eq!(values.dim(), (times.len(), n));
        for ((i, j), actual) in values.indexed_iter() {
            let expected = c["expected"][i][j].as_f64().unwrap();
            assert!(
                (actual - expected).abs() < 2e-12,
                "degree={degree}, n={n}, ({i},{j}): {actual} != {expected}"
            );
            if times[i] <= 0. || times[i] >= 24. {
                assert_eq!(*actual, 0.);
            }
        }
    }
    assert!(bspline_hrf(1, Some(4), None)
        .evaluate(&Array1::from_vec(vec![0., 1.]))
        .is_err());
}
