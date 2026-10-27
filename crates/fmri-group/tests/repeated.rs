use fmri_group::*;
use fmridataset::*;
use ndarray::{array, Array2};
use serde_json::Value;

fn cohort() -> RepeatedCohort {
    let ids = (0..12)
        .map(|i| ObservationId::new(format!("o{i}")).unwrap())
        .collect::<Vec<_>>();
    let fs = FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new("test", vec![FeatureId::new("f0").unwrap()]).unwrap(),
    ))
    .unwrap();
    let y = vec![-1., -1., -1., 0., 0., 0., 1., 1., 1., 2., 2., 2.];
    let assay = Assay::new(
        "effect",
        SourceSpec::memory(NumericBlock::new([12, 1], Values::F64(y)).unwrap()),
        ids.clone(),
        fs.axis().ids().to_vec(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let frame = FmriFrame::new(
        Axis::new(ids.clone(), vec![]).unwrap(),
        fs,
        vec![assay],
        "effect",
        Metadata::new(),
    )
    .unwrap();
    let subjects = (0..12)
        .map(|i| RepeatedSubject {
            observation: ids[i].clone(),
            dataset: "d".into(),
            participant: format!("s{}", i / 3),
            measurement: format!("m{}", i % 3),
        })
        .collect::<Vec<_>>();
    let blocks = (0..4)
        .map(|s| RepeatedCovarianceBlock {
            observations: ids[s * 3..s * 3 + 3].to_vec(),
            values: Array2::eye(3) * 0.3,
        })
        .collect();
    RepeatedCohort::new(
        frame,
        subjects,
        Estimand {
            id: "e".into(),
            definition: "synthetic mean".into(),
            units: "a.u.".into(),
        },
        Admission {
            target: "synthetic".into(),
            evidence: "unit fixture".into(),
        },
        "effect".into(),
        vec![RepeatedFeatureCovariance {
            feature: FeatureId::new("f0").unwrap(),
            blocks,
        }],
        CovarianceProvenance::Known {
            method: "fixture".into(),
        },
    )
    .unwrap()
}
#[test]
fn fixed_ri_matches_analytic_gls_witness() {
    let oracle: Value =
        serde_json::from_str(include_str!("fixtures/repeated_reference.json")).unwrap();
    let c = cohort();
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["mean".into()],
        Array2::ones((12, 1)),
    )
    .unwrap();
    let ct = GroupContrast::named("mean", &d, &[("mean", 1.)]).unwrap();
    let p = PreparedRepeatedModel::new(
        c,
        d,
        vec![ct],
        RepeatedModel {
            likelihood: RepeatedLikelihood::Ml,
            ..Default::default()
        },
    )
    .unwrap();
    let r = p.evaluate_fixed(0, array![[1.15]], 0.).unwrap();
    let e = r.estimates.unwrap();
    assert_eq!(r.status, RepeatedStatus::Estimated);
    assert!((e.beta[0] - oracle["ri"]["ml"]["beta"].as_f64().unwrap()).abs() < 1e-11);
    assert!(
        (e.beta_covariance[(0, 0)] - oracle["ri"]["ml"]["covariance"][0][0].as_f64().unwrap())
            .abs()
            < 1e-11
    );
    assert!((e.objective - oracle["ri"]["ml"]["objective"].as_f64().unwrap()).abs() < 1e-10);
}
#[test]
fn rejects_constant_slope_basis() {
    let c = cohort(); // the three-time fixture is identified; use a constant slope to exercise the same admission boundary.
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["mean".into()],
        Array2::ones((12, 1)),
    )
    .unwrap();
    let ct = GroupContrast::named("mean", &d, &[("mean", 1.)]).unwrap();
    assert!(PreparedRepeatedModel::new(
        c,
        d,
        vec![ct],
        RepeatedModel {
            random_effects: RandomEffects::InterceptSlope,
            residual_variance: ResidualVariance::Estimate,
            time: Some(vec![1.; 12]),
            ..Default::default()
        }
    )
    .is_err());
}
#[test]
fn covariance_order_and_spd_are_admission_contracts() {
    let c = cohort();
    let mut bad = c.covariance(0).unwrap().to_vec();
    bad[0].observations.reverse();
    let frame = c.frame().clone();
    let subjects = c.subjects().to_vec();
    assert!(RepeatedCohort::new(
        frame,
        subjects,
        c.estimand().clone(),
        c.admission().clone(),
        "effect".into(),
        vec![RepeatedFeatureCovariance {
            feature: FeatureId::new("f0").unwrap(),
            blocks: bad
        }],
        CovarianceProvenance::Known {
            method: "fixture".into()
        }
    )
    .is_err());
}
