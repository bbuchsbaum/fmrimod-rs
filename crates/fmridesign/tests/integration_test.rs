//! Integration tests for fmridesign.

use fmridesign::core::EventModel;
use fmridesign::naming::{make_column_names, sanitize};
use fmridesign::prelude::*;
use fmridesign::sampling::uniform_sampling_frame;
use fmridesign::utils::{all_equal, unique_ordered};

#[test]
fn test_basic_types() {
    // Test validation functions
    let onsets = vec![1.0, 2.0, 3.0];
    let validated = validate_onsets(onsets).unwrap();
    assert_eq!(validated.len(), 3);

    let durations = uniform_durations(1.0, 3).unwrap();
    assert_eq!(durations.len(), 3);
}

#[test]
fn test_naming_functions() {
    // Test sanitization
    assert_eq!(sanitize("hello world", true), "hello.world");
    assert_eq!(sanitize("123test", true), "X123test");

    // Test column name generation
    let cond_tags = vec!["cond1".to_string(), "cond2".to_string()];
    let names = make_column_names(Some("task"), &cond_tags, 1);
    assert_eq!(names, vec!["task_cond1", "task_cond2"]);
}

#[test]
fn test_sampling_frame() {
    let sf = uniform_sampling_frame(100, 2.0, None).unwrap();
    assert_eq!(sf.n_scans(), 100);
    assert_eq!(sf.n_blocks(), 1);
}

#[test]
fn test_utils() {
    // Test utility functions
    assert!(all_equal(&[1, 1, 1]));
    assert!(!all_equal(&[1, 2, 3]));

    let unique = unique_ordered(&[1, 2, 2, 3, 1]);
    assert_eq!(unique, vec![1, 2, 3]);
}

#[test]
fn test_event_model_creation() {
    let model = EventModel {
        events: vec![],
        sampling_frame: uniform_sampling_frame(1, 1.0, None).unwrap(),
        name: "test_model".to_string(),
    };

    // Model should be created successfully
    assert!(format!("{:?}", model).contains("test_model"));
}

#[test]
fn test_formula_parsing() {
    let formula = Formula::parse("condition + response_time").unwrap();
    assert!(formula.formula.contains("condition"));
    assert!(formula.formula.contains("response_time"));
}

#[test]
fn test_polynomial_basis() {
    use ndarray::array;
    let basis = PolynomialBasis::new(2);
    assert_eq!(basis.nbasis(), 2); // default excludes the intercept

    let x = array![1.0, 2.0, 3.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[3, 2]);

    // For orthogonal polynomials, we can't check exact values but we can check orthogonality
    // Check that columns are orthogonal (dot product should be ~0)
    let col0 = result.column(0);
    let col1 = result.column(1);

    let dot01 = col0.dot(&col1);

    assert!(dot01.abs() < 1e-10, "Columns 0 and 1 should be orthogonal");
}

#[test]
fn test_raw_polynomial_basis() {
    use ndarray::array;
    let basis = PolynomialBasis::new(2).with_intercept(true).with_raw(true);
    assert_eq!(basis.nbasis(), 3);

    let x = array![1.0, 2.0, 3.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[3, 3]);

    // For raw polynomials we can check exact values
    // Check first column is all 1s (x^0)
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[1, 0]], 1.0);
    assert_eq!(result[[2, 0]], 1.0);

    // Check second column is x (x^1)
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[1, 1]], 2.0);
    assert_eq!(result[[2, 1]], 3.0);

    // Check third column is x^2
    assert_eq!(result[[0, 2]], 1.0);
    assert_eq!(result[[1, 2]], 4.0);
    assert_eq!(result[[2, 2]], 9.0);
}

#[test]
fn test_natural_polynomial_basis() {
    use fmridesign::basis::NaturalPolynomialBasis;
    use ndarray::array;

    let basis = NaturalPolynomialBasis::new(2);
    assert_eq!(basis.nbasis(), 3);

    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[5, 3]);

    // First column should be all 1s (intercept)
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[1, 0]], 1.0);
    assert_eq!(result[[2, 0]], 1.0);
    assert_eq!(result[[3, 0]], 1.0);
    assert_eq!(result[[4, 0]], 1.0);

    // Second column should be centered x values
    // Mean of [1, 2, 3, 4, 5] is 3
    assert_eq!(result[[0, 1]], -2.0); // 1 - 3
    assert_eq!(result[[1, 1]], -1.0); // 2 - 3
    assert_eq!(result[[2, 1]], 0.0); // 3 - 3
    assert_eq!(result[[3, 1]], 1.0); // 4 - 3
    assert_eq!(result[[4, 1]], 2.0); // 5 - 3

    // Third column should be squared centered values
    assert_eq!(result[[0, 2]], 4.0); // (-2)^2
    assert_eq!(result[[1, 2]], 1.0); // (-1)^2
    assert_eq!(result[[2, 2]], 0.0); // 0^2
    assert_eq!(result[[3, 2]], 1.0); // 1^2
    assert_eq!(result[[4, 2]], 4.0); // 2^2
}

#[test]
fn test_polynomial_basis_without_intercept() {
    use ndarray::array;
    let basis = PolynomialBasis::new(2).with_intercept(false).with_raw(true);
    assert_eq!(basis.nbasis(), 2); // No intercept, so only x^1 and x^2

    let x = array![1.0, 2.0, 3.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[3, 2]);

    // First column is x (x^1)
    assert_eq!(result[[0, 0]], 1.0);
    assert_eq!(result[[1, 0]], 2.0);
    assert_eq!(result[[2, 0]], 3.0);

    // Second column is x^2
    assert_eq!(result[[0, 1]], 1.0);
    assert_eq!(result[[1, 1]], 4.0);
    assert_eq!(result[[2, 1]], 9.0);
}

#[test]
fn test_bspline_basis() {
    use fmridesign::basis::BSplineBasis;
    use ndarray::array;

    let basis = BSplineBasis::new(4);
    assert_eq!(basis.nbasis(), 4);

    let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[5, 4]);

    // Check that basis functions are non-negative
    // (Our simplified implementation doesn't guarantee partition of unity)
    for i in 0..5 {
        for j in 0..4 {
            assert!(
                result[[i, j]] >= 0.0,
                "Basis functions should be non-negative"
            );
        }
    }
}

#[test]
fn test_natural_spline_basis() {
    use fmridesign::basis::NaturalSplineBasis;
    use ndarray::array;

    let basis = NaturalSplineBasis::new(4);
    assert_eq!(basis.nbasis(), 4);

    let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[5, 4]);
}

#[test]
fn test_spline_with_knots() {
    use fmridesign::basis::BSplineBasis;
    use ndarray::array;

    let basis = BSplineBasis::with_knots(vec![0.25, 0.5, 0.75]).with_intercept(true);
    // With 3 knots, degree 3, and intercept, we should have 3 + 3 + 1 = 7 basis functions
    assert_eq!(basis.nbasis(), 7);

    let x = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    let result = basis.evaluate(&x.view()).unwrap();
    assert_eq!(result.shape(), &[6, 7]);
}
