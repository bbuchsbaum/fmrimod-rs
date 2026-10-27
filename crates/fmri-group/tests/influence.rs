mod common;
use common::cohort;
use fmri_group::*;
use ndarray::{array, Array2};
fn numbers(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn check(f: &MetaRegressionFeature, r: &serde_json::Value) {
    assert_eq!(f.status, MetaStatus::Estimated);
    let e = f.estimates.as_ref().unwrap();
    for (a, b) in e.effects.iter().zip(numbers(&r["effect"])) {
        assert!((a - b).abs() < 1e-8 * (1. + b.abs()), "effect {a} {b}");
    }
    for (a, b) in e
        .standard_errors
        .as_ref()
        .unwrap()
        .iter()
        .zip(numbers(&r["se"]))
    {
        assert!((a - b).abs() < 1e-8 * (1. + b.abs()), "se {a} {b}");
    }
    let tau = r["tau"].as_f64().unwrap();
    assert!((f.diagnostics.as_ref().unwrap().tau_squared - tau).abs() < 2e-8 * (1. + tau));
}
#[test]
fn exact_deletions_match_fresh_metafor_in_both_tau_modes() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/influence_reference.json")).unwrap();
    let c = cohort(
        Array2::from_shape_vec((10, 1), numbers(&f["y"])).unwrap(),
        Some(Array2::from_shape_vec((10, 1), numbers(&f["v"])).unwrap()),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "slope".into()],
        Array2::from_shape_vec((10, 2), numbers(&f["x"])).unwrap(),
    )
    .unwrap();
    let spec = MetaOutputSpec {
        contrasts: vec![
            GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap(),
            GroupContrast::named("slope", &d, &[("slope", 1.)]).unwrap(),
        ],
        uncertainty: MetaUncertainty::Marginal,
        p_values: false,
        confidence_level: None,
    };
    for (name, method) in [
        ("FE", HeterogeneityEstimator::CommonEffect),
        ("DL", HeterogeneityEstimator::DerSimonianLaird),
        ("PM", HeterogeneityEstimator::PauleMandel),
    ] {
        let mut mkh = vec![];
        meta_influence(
            &c,
            &d,
            spec.clone(),
            MetaRegressionOptions {
                heterogeneity: method,
                inference: MetaInference::ModifiedKnappHartung,
                ..Default::default()
            },
            DeletionHeterogeneity::FixedAtFullFit,
            |b| {
                mkh.extend(b.features);
                Ok(())
            },
        )
        .unwrap();
        check(&mkh[0].full, &f["results"][name]["full_mkh"]);
        for (i, r) in mkh[0].deletions.iter().enumerate() {
            check(r.fit.as_ref().unwrap(), &f["results"][name]["fixed_mkh"][i]);
        }
        for mode in [
            DeletionHeterogeneity::Refit,
            DeletionHeterogeneity::FixedAtFullFit,
        ] {
            let mut result = vec![];
            let receipt = meta_influence(
                &c,
                &d,
                spec.clone(),
                MetaRegressionOptions {
                    heterogeneity: method,
                    ..Default::default()
                },
                mode,
                |b| {
                    result.extend(b.features);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(receipt.fits_attempted, 11);
            let f0 = &result[0];
            check(&f0.full, &f["results"][name]["full"]);
            for (i, deleted) in f0.deletions.iter().enumerate() {
                let r = &f["results"][name][if mode == DeletionHeterogeneity::Refit {
                    "refit"
                } else {
                    "fixed"
                }][i];
                check(deleted.fit.as_ref().unwrap(), r);
                assert_eq!(deleted.omitted, c.subjects()[i].observation);
                for k in 0..2 {
                    let delta = deleted
                        .fit
                        .as_ref()
                        .unwrap()
                        .estimates
                        .as_ref()
                        .unwrap()
                        .effects[k]
                        - f0.full.estimates.as_ref().unwrap().effects[k];
                    assert_eq!(deleted.effect_change.as_ref().unwrap()[k], delta);
                }
            }
        }
    }
}
#[test]
fn missingness_rank_loss_and_sink_failure_stay_explicit() {
    let c = cohort(
        array![[1., 1.], [2., f64::NAN], [3., 4.], [4., 2.]],
        Some(Array2::ones((4, 2))),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "rare".into()],
        array![[1., 1.], [1., 0.], [1., 0.], [1., 0.]],
    )
    .unwrap();
    let spec = MetaOutputSpec::new(vec![
        GroupContrast::named("rare", &d, &[("rare", 1.)]).unwrap()
    ]);
    let mut o = MetaRegressionOptions::default();
    o.execution.missingness = Missingness::OmitObservation;
    o.execution.block_features = 1;
    let mut r = vec![];
    meta_influence(&c, &d, spec.clone(), o, DeletionHeterogeneity::Refit, |b| {
        r.extend(b.features);
        Ok(())
    })
    .unwrap();
    assert_eq!(
        r[0].deletions[0].fit.as_ref().unwrap().status,
        MetaStatus::RankDeficient
    );
    assert!(r[1].deletions[1].fit.is_none());
    assert_eq!(
        r[1].deletions[2].fit.as_ref().unwrap().status,
        MetaStatus::InsufficientObservations
    );
    assert!(
        meta_influence(&c, &d, spec, o, DeletionHeterogeneity::Refit, |_| Err(
            Error::Invalid("sink refused")
        ))
        .is_err()
    );
}

#[test]
fn positive_fixed_tau_does_not_require_an_estimable_zero_tau_fit() {
    let c = cohort(
        array![[0.], [0.], [10.], [-10.], [20.]],
        Some(array![[1e-30], [1e-30], [1.], [1.], [1.]]),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "slope".into()],
        array![[1., 1.], [1., 0.], [1., 1.], [1., 2.], [1., 3.]],
    )
    .unwrap();
    let spec = MetaOutputSpec {
        contrasts: vec![
            GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap(),
            GroupContrast::named("slope", &d, &[("slope", 1.)]).unwrap(),
        ],
        uncertainty: MetaUncertainty::Marginal,
        p_values: false,
        confidence_level: None,
    };
    let mut r = vec![];
    meta_influence(
        &c,
        &d,
        spec,
        MetaRegressionOptions {
            heterogeneity: HeterogeneityEstimator::PauleMandel,
            ..Default::default()
        },
        DeletionHeterogeneity::FixedAtFullFit,
        |b| {
            r.extend(b.features);
            Ok(())
        },
    )
    .unwrap();
    let full = r[0].full.diagnostics.as_ref().unwrap();
    assert!((full.tau_squared - 141.3289646825866).abs() < 1e-7);
    let deleted = r[0].deletions[1].fit.as_ref().unwrap();
    assert_eq!(deleted.status, MetaStatus::Estimated);
    let e = deleted.estimates.as_ref().unwrap();
    assert!((e.effects[0] + 4.55011769).abs() < 1e-7);
    assert!((e.effects[1] - 5.45629413).abs() < 1e-7);
    assert!(deleted.diagnostics.as_ref().unwrap().q_fixed.is_none());
}
