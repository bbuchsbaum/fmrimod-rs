mod common;
use common::cohort;
use fmri_group::*;
use ndarray::{array, Array2};

fn options(method: PermutationMethod, block: usize) -> PermutationOptions {
    PermutationOptions {method,schedule:PermutationSchedule::Exhaustive {max_transformations:1000},
        justification:"synthetic symmetric independent subjects; nuisance null and within-block exchangeability declared".into(),
        block_features:block,max_working_bytes:1<<20,rank_tolerance:1e-12}
}
fn numbers(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
#[test]
fn fresh_r_complete_enumeration_and_whole_family_block_invariance() {
    let f: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/resampling_reference.json")).unwrap();
    let c = cohort(
        Array2::from_shape_vec((6, 3), numbers(&f["y"])).unwrap(),
        None,
    );
    for mode in 0..3 {
        let fl = mode > 0;
        let d = if fl {
            GroupDesign::new(
                c.frame().observations().ids().to_vec(),
                vec!["intercept".into(), "slope".into()],
                Array2::from_shape_vec((6, 2), numbers(&f["x"])).unwrap(),
            )
            .unwrap()
        } else {
            GroupDesign::intercept(&c)
        };
        let contrast = if mode == 2 {
            GroupContrast::named("test", &d, &[("intercept", 1.), ("slope", 0.5)]).unwrap()
        } else {
            GroupContrast::named("test", &d, &[(if fl { "slope" } else { "intercept" }, 1.)])
                .unwrap()
        };
        let method = if fl {
            PermutationMethod::FreedmanLane {
                blocks: vec![
                    "a".into(),
                    "a".into(),
                    "a".into(),
                    "b".into(),
                    "b".into(),
                    "b".into(),
                ],
            }
        } else {
            PermutationMethod::SignFlip
        };
        let expected = &f[if mode == 2 {
            "freedman_lane_general"
        } else if fl {
            "freedman_lane"
        } else {
            "sign"
        }];
        let mut reference = None;
        for block in [1, 2, 7] {
            let result =
                permutation_max_t(&c, &d, &contrast, options(method.clone(), block)).unwrap();
            assert_eq!(result.transformations, if fl { 36 } else { 64 });
            for (j, r) in result.features.iter().enumerate() {
                assert!((r.statistic - expected["observed"][j].as_f64().unwrap()).abs() < 1e-11);
                assert!((r.p_uncorrected - expected["p"][j].as_f64().unwrap()).abs() < 1e-14);
                assert!((r.p_max_t - expected["p_max"][j].as_f64().unwrap()).abs() < 1e-14);
                assert!(r.p_max_t >= r.p_uncorrected);
            }
            let mut maxima = result.max_abs_t;
            maxima.sort_by(f64::total_cmp);
            let mut expected_max = numbers(&expected["maxima"]);
            expected_max.sort_by(f64::total_cmp);
            for (a, b) in maxima.iter().zip(expected_max) {
                assert!((a - b).abs() < 1e-10 * (1. + b.abs()), "{a} {b}");
            }
            if let Some(ref prior) = reference {
                assert_eq!(prior, &maxima);
            } else {
                reference = Some(maxima);
            }
        }
    }
}
#[test]
fn seeded_draws_are_reproducible_and_never_zero() {
    let c = cohort(
        array![[1., 0.2], [0.8, -0.4], [1.2, 0.7], [0.9, -0.3], [0.5, 0.2]],
        None,
    );
    let d = GroupDesign::intercept(&c);
    let k = GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap();
    let mut o = options(PermutationMethod::SignFlip, 1);
    o.schedule = PermutationSchedule::MonteCarlo {
        draws: 199,
        seed: 77,
    };
    let a = permutation_max_t(&c, &d, &k, o.clone()).unwrap();
    o.block_features = 3;
    let b = permutation_max_t(&c, &d, &k, o).unwrap();
    assert_eq!(a.max_abs_t, b.max_abs_t);
    for (a, b) in a.features.iter().zip(b.features) {
        assert_eq!(a.p_max_t, b.p_max_t);
        assert!(a.p_uncorrected >= 1. / 200.);
    }
}
#[test]
fn rejects_missing_invalid_exchangeability_budget_and_undefined_statistic() {
    let c = cohort(array![[1.], [2.], [3.], [4.]], None);
    let d = GroupDesign::intercept(&c);
    let k = GroupContrast::named("mean", &d, &[("intercept", 1.)]).unwrap();
    let mut o = options(PermutationMethod::SignFlip, 1);
    o.max_working_bytes = 1;
    assert!(matches!(
        permutation_max_t(&c, &d, &k, o),
        Err(Error::Budget)
    ));
    let mut o = options(PermutationMethod::SignFlip, 1);
    o.schedule = PermutationSchedule::Exhaustive {
        max_transformations: 8,
    };
    assert!(permutation_max_t(&c, &d, &k, o).is_err());
    assert!(permutation_max_t(
        &c,
        &d,
        &k,
        options(
            PermutationMethod::FreedmanLane {
                blocks: vec!["a".into()]
            },
            1
        )
    )
    .is_err());
    for y in [
        array![[1.], [1.], [1.], [1.]],
        array![[1.], [2.], [f64::NAN], [4.]],
    ] {
        let c = cohort(y, None);
        assert!(permutation_max_t(&c, &d, &k, options(PermutationMethod::SignFlip, 1)).is_err());
    }
}

#[test]
fn column_and_contrast_units_cannot_change_the_null_space() {
    let c = cohort(array![[0.2], [1.1], [1.7], [2.1]], None);
    let values = array![[1., -1.5], [1., -0.5], [1., 0.5], [1., 1.5]];
    let method = PermutationMethod::FreedmanLane {
        blocks: vec!["all".into(); 4],
    };
    for (intercept_scale, slope_scale, contrast_scale) in [
        (1., 1., 1.),
        (1e150, 1e150, 1e160),
        (1e-150, 1e150, 1e160),
        (1e150, 1e-150, 1e-160),
    ] {
        let x = Array2::from_shape_fn((4, 2), |(i, j)| {
            values[(i, j)] * if j == 0 { intercept_scale } else { slope_scale }
        });
        let d = GroupDesign::new(
            c.frame().observations().ids().to_vec(),
            vec!["intercept".into(), "slope".into()],
            x,
        )
        .unwrap();
        let ct = GroupContrast::named("slope", &d, &[("slope", contrast_scale)]).unwrap();
        let r = permutation_max_t(&c, &d, &ct, options(method.clone(), 1)).unwrap();
        assert!((r.features[0].p_max_t - 2. / 24.).abs() < 1e-14);
    }
}
