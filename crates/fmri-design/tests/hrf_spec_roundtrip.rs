use fmri_design::hrf::{create_hrf, HrfSpec};
use ndarray::array;
use std::collections::HashMap;
#[test]
fn roundtrip_preserves_spmg_parameters_and_custom_samples() {
    let params = HashMap::from([("p1".into(), 6.0), ("p2".into(), 17.0), ("a1".into(), 0.01)]);
    for spec in [
        HrfSpec::Named {
            name: "spmg3".into(),
            params,
        },
        HrfSpec::custom(array![0., 1., 0.4, 0.], 1.5),
        HrfSpec::fourier(6, 24., 1.5),
    ] {
        let encoded = serde_json::to_string(&spec).unwrap();
        let decoded: HrfSpec = serde_json::from_str(&encoded).unwrap();
        let t = array![0., 0.7, 2.5, 6.1, 13.];
        assert_eq!(
            create_hrf(&spec).unwrap().evaluate(&t).unwrap(),
            create_hrf(&decoded).unwrap().evaluate(&t).unwrap()
        );
    }
    let custom = HrfSpec::Named {
        name: "spmg1".into(),
        params: HashMap::from([("p1".into(), 7.)]),
    };
    assert_ne!(
        create_hrf(&custom).unwrap().evaluate(&array![5.]).unwrap(),
        create_hrf(&HrfSpec::canonical())
            .unwrap()
            .evaluate(&array![5.])
            .unwrap()
    );
    assert!(create_hrf(&HrfSpec::Named {
        name: "spmg1".into(),
        params: HashMap::from([("typo".into(), 7.)])
    })
    .is_err());
    assert!(create_hrf(&HrfSpec::fir(0, 24., 1.)).is_err());
    assert!(create_hrf(&HrfSpec::fourier(5, f64::NAN, 1.)).is_err());
}
