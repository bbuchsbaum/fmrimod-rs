//! Tests for ScaleWithin transform

use approx::assert_abs_diff_eq;
use fmridesign::basis::{Scale, ScaleWithin};
use fmridesign::types::BasisFunction;
use ndarray::Array1;

#[test]
fn test_scale_within_basic() {
    // Create data with two distinct groups
    let groups = vec![
        "A".to_string(),
        "A".to_string(),
        "A".to_string(),
        "B".to_string(),
        "B".to_string(),
        "B".to_string(),
    ];

    // Group A: values around 10, Group B: values around 50
    let values = Array1::from(vec![9.0, 10.0, 11.0, 49.0, 50.0, 51.0]);

    let transform = ScaleWithin::new(groups, true, true);
    let result = transform.evaluate(&values.view()).unwrap();

    // Check shape
    assert_eq!(result.shape(), &[6, 1]);

    // Group A should be standardized: mean=0, std=1
    let group_a_mean = (result[[0, 0]] + result[[1, 0]] + result[[2, 0]]) / 3.0;
    assert_abs_diff_eq!(group_a_mean, 0.0, epsilon = 1e-10);

    // Group B should be standardized: mean=0, std=1
    let group_b_mean = (result[[3, 0]] + result[[4, 0]] + result[[5, 0]]) / 3.0;
    assert_abs_diff_eq!(group_b_mean, 0.0, epsilon = 1e-10);

    // Check that values are properly scaled
    // Note: ScaleWithin uses population std (n) not sample std (n-1)
    // Group A: mean = 10, values = [9, 10, 11], variance = 2/3, std = sqrt(2/3)
    let expected_std = (2.0_f64 / 3.0).sqrt();
    assert_abs_diff_eq!(result[[0, 0]], -1.0 / expected_std, epsilon = 1e-10); // (9-10)/std
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10); // (10-10)/std
    assert_abs_diff_eq!(result[[2, 0]], 1.0 / expected_std, epsilon = 1e-10); // (11-10)/std

    // Same for Group B
    assert_abs_diff_eq!(result[[3, 0]], -1.0 / expected_std, epsilon = 1e-10); // (49-50)/std
    assert_abs_diff_eq!(result[[4, 0]], 0.0, epsilon = 1e-10); // (50-50)/std
    assert_abs_diff_eq!(result[[5, 0]], 1.0 / expected_std, epsilon = 1e-10); // (51-50)/std
}

#[test]
fn test_scale_within_center_only() {
    let groups = vec![
        "G1".to_string(),
        "G1".to_string(),
        "G2".to_string(),
        "G2".to_string(),
    ];
    let values = Array1::from(vec![10.0, 20.0, 100.0, 200.0]);

    let transform = ScaleWithin::new(groups, true, false); // center only
    let result = transform.evaluate(&values.view()).unwrap();

    // Group 1: mean = 15, so values become [-5, 5]
    assert_abs_diff_eq!(result[[0, 0]], -5.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[1, 0]], 5.0, epsilon = 1e-10);

    // Group 2: mean = 150, so values become [-50, 50]
    assert_abs_diff_eq!(result[[2, 0]], -50.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[3, 0]], 50.0, epsilon = 1e-10);
}

#[test]
fn test_scale_within_scale_only() {
    let groups = vec![
        "G1".to_string(),
        "G1".to_string(),
        "G2".to_string(),
        "G2".to_string(),
    ];
    let values = Array1::from(vec![2.0, 4.0, 10.0, 20.0]);

    let transform = ScaleWithin::new(groups, false, true); // scale only
    let result = transform.evaluate(&values.view()).unwrap();

    // Group 1: std = 1.0, values unchanged
    assert_abs_diff_eq!(result[[0, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[1, 0]], 4.0, epsilon = 1e-10);

    // Group 2: std = 5.0, values scaled
    assert_abs_diff_eq!(result[[2, 0]], 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[3, 0]], 4.0, epsilon = 1e-10);
}

#[test]
fn test_scale_within_single_group() {
    let groups = vec!["A".to_string(), "A".to_string(), "A".to_string()];
    let values = Array1::from(vec![1.0, 2.0, 3.0]);

    let transform = ScaleWithin::new(groups, true, true);
    let result = transform.evaluate(&values.view()).unwrap();

    // Should behave like regular standardization
    let expected_mean = 0.0;
    let actual_mean = result.column(0).mean().unwrap();
    assert_abs_diff_eq!(actual_mean, expected_mean, epsilon = 1e-10);
}

#[test]
fn test_scale_within_vs_global() {
    let groups = vec![
        "run1".to_string(),
        "run1".to_string(),
        "run2".to_string(),
        "run2".to_string(),
    ];
    let values = Array1::from(vec![1.0, 2.0, 10.0, 20.0]);

    // ScaleWithin
    let within_transform = ScaleWithin::new(groups, true, true);
    let within_result = within_transform.evaluate(&values.view()).unwrap();

    // Global Scale
    let global_transform = Scale::new(true, true);
    let global_result = global_transform.evaluate(&values.view()).unwrap();

    // Results should be different
    assert!(within_result[[0, 0]] != global_result[[0, 0]]);
    assert!(within_result[[2, 0]] != global_result[[2, 0]]);

    // Within-group means should both be 0
    let run1_mean = (within_result[[0, 0]] + within_result[[1, 0]]) / 2.0;
    let run2_mean = (within_result[[2, 0]] + within_result[[3, 0]]) / 2.0;
    assert_abs_diff_eq!(run1_mean, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(run2_mean, 0.0, epsilon = 1e-10);
}

#[test]
fn test_scale_within_empty_group() {
    // Test with groups that have no actual data points
    let groups = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let values = Array1::from(vec![1.0, 2.0, 3.0]);

    let transform = ScaleWithin::new(groups, true, true);
    let result = transform.evaluate(&values.view()).unwrap();

    // Group B has only one value, so std = 0, value should remain at 0 after centering
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_scale_within_error_handling() {
    let groups = vec!["A".to_string(), "B".to_string()];
    let values = Array1::from(vec![1.0, 2.0, 3.0]); // Length mismatch

    let transform = ScaleWithin::new(groups, true, true);
    let result = transform.evaluate(&values.view());

    assert!(result.is_err());
}
