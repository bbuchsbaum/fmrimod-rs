use fmri_design::contrast::Contrast;
use fmri_design::design::{hstack_design_matrices, DesignMatrix};
use fmri_design::types::ContrastSpec;
use fmri_design::validation::validate_contrasts;
use ndarray::array;

#[test]
fn legacy_selectors_match_exact_column_names() {
    let columns = vec!["A".to_string(), "AB".to_string(), "B".to_string()];

    assert_eq!(
        Contrast::simple("a", "A")
            .compute_weights(&columns)
            .unwrap(),
        array![1.0, 0.0, 0.0]
    );
    assert_eq!(
        Contrast::pair("a_vs_b", "A", "B")
            .compute_weights(&columns)
            .unwrap(),
        array![1.0, 0.0, -1.0]
    );
    assert!(Contrast::pair("same", "A", "A")
        .compute_weights(&columns)
        .is_err());
    assert!(Contrast::column("overlap", "A", Some("AB"))
        .compute_weights(&columns)
        .is_err());
}

#[test]
fn unsupported_legacy_contrasts_do_not_become_zero_vectors() {
    let columns = vec!["dose.low".to_string(), "dose.high".to_string()];

    assert!(Contrast::f_contrast("f", vec![])
        .compute_weights(&columns)
        .is_err());
    assert!(Contrast::polynomial("quadratic", "dose", 2)
        .compute_weights(&columns)
        .is_err());
    assert!(Contrast::simple("nonfinite", "dose.low")
        .with_weights(array![f64::NAN, 1.0])
        .compute_weights(&columns)
        .is_err());
}

#[test]
fn hstack_requires_identical_finite_time_grid() {
    let first = DesignMatrix::with_times(
        array![[1.0], [2.0]],
        vec!["a".to_string()],
        array![0.0, 1.0],
    );
    let shifted = DesignMatrix::with_times(
        array![[3.0], [4.0]],
        vec!["b".to_string()],
        array![0.5, 1.5],
    );
    let nonfinite = DesignMatrix::with_times(
        array![[3.0], [4.0]],
        vec!["b".to_string()],
        array![0.0, f64::NAN],
    );

    assert!(hstack_design_matrices(&[first.clone(), shifted]).is_err());
    assert!(hstack_design_matrices(&[first, nonfinite]).is_err());
}

#[test]
fn estimability_is_relative_and_scale_invariant() {
    let full_rank = array![[1.0, 0.0], [0.0, 1.0]];
    let names = vec!["a".to_string(), "b".to_string()];
    for scale in [1e-200, 1.0, 1e200] {
        let weights = array![[scale], [-scale]];
        assert!(validate_contrasts(&full_rank, &weights, &names, 1e-8)[0].estimable);
        assert!(
            validate_contrasts(&(full_rank.clone() * scale), &weights, &names, 1e-8)[0].estimable
        );
    }

    let rank_deficient = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
    let estimable = array![[1.0], [1.0]];
    for design_scale in [1e-200, 1.0, 1e200] {
        for contrast_scale in [1e-200, 1.0, 1e200] {
            let non_estimable = array![[contrast_scale], [-contrast_scale]];
            assert!(
                !validate_contrasts(
                    &(rank_deficient.clone() * design_scale),
                    &non_estimable,
                    &names,
                    1e-8,
                )[0]
                .estimable
            );
        }
    }
    assert!(validate_contrasts(&rank_deficient, &estimable, &names, 1e-8)[0].estimable);

    let invalid_weights = array![[f64::INFINITY], [0.0]];
    assert!(!validate_contrasts(&full_rank, &invalid_weights, &names, 1e-8)[0].estimable);
}
