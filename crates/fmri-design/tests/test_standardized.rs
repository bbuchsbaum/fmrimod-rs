//! Tests for Standardized transform

use approx::assert_abs_diff_eq;
use fmri_design::basis::{Scale, Standardized};
use fmri_design::types::BasisFunction;
use ndarray::Array1;

#[test]
fn test_standardized_basic_fit_transform() {
    let mut transform = Standardized::new(true, true);

    // Fit on training data
    let train_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    transform.fit(&train_data).unwrap();

    // Check fitted parameters
    assert!(transform.is_fitted());
    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 3.0, epsilon = 1e-10);
    assert_abs_diff_eq!(
        transform.fitted_std().unwrap(),
        2.0_f64.sqrt(),
        epsilon = 1e-10
    );

    // Transform test data
    let test_data = Array1::from(vec![3.0, 6.0, 0.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    // Check transformations
    assert_eq!(result.shape(), &[3, 1]);
    assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10); // (3-3)/sqrt(2) = 0
    assert_abs_diff_eq!(result[[1, 0]], 3.0 / 2.0_f64.sqrt(), epsilon = 1e-10); // (6-3)/sqrt(2)
    assert_abs_diff_eq!(result[[2, 0]], -3.0 / 2.0_f64.sqrt(), epsilon = 1e-10);
    // (0-3)/sqrt(2)
}

#[test]
fn test_standardized_center_only() {
    let mut transform = Standardized::new(true, false);

    let train_data = vec![10.0, 20.0, 30.0];
    transform.fit(&train_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 20.0, epsilon = 1e-10);
    assert_abs_diff_eq!(transform.fitted_std().unwrap(), 1.0, epsilon = 1e-10); // No scaling

    let test_data = Array1::from(vec![25.0, 15.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    assert_abs_diff_eq!(result[[0, 0]], 5.0, epsilon = 1e-10); // 25 - 20
    assert_abs_diff_eq!(result[[1, 0]], -5.0, epsilon = 1e-10); // 15 - 20
}

#[test]
fn test_standardized_scale_only() {
    let mut transform = Standardized::new(false, true);

    let train_data = vec![2.0, 4.0, 6.0, 8.0];
    transform.fit(&train_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 0.0, epsilon = 1e-10); // No centering

    // Compute expected std using population formula
    // mean = 5, values = [2,4,6,8]
    // variance = [(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2] / 4 = [9+1+1+9]/4 = 20/4 = 5
    // But wait, the mean when not centering is 0, not 5!
    // So variance = [4 + 16 + 36 + 64] / 4 = 120/4 = 30
    // std = sqrt(30)
    let expected_std = 30.0_f64.sqrt();
    assert_abs_diff_eq!(
        transform.fitted_std().unwrap(),
        expected_std,
        epsilon = 1e-10
    );

    let test_data = Array1::from(vec![10.0, 5.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    assert_abs_diff_eq!(result[[0, 0]], 10.0 / expected_std, epsilon = 1e-10);
    assert_abs_diff_eq!(result[[1, 0]], 5.0 / expected_std, epsilon = 1e-10);
}

#[test]
fn test_standardized_auto_fit() {
    // When not pre-fitted, should auto-fit on input data
    let transform = Standardized::new(true, true);

    let data = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let result = transform.evaluate(&data.view()).unwrap();

    // Should behave like regular Scale when not pre-fitted
    let scale = Scale::new(true, true);
    let scale_result = scale.evaluate(&data.view()).unwrap();

    for i in 0..5 {
        assert_abs_diff_eq!(result[[i, 0]], scale_result[[i, 0]], epsilon = 1e-10);
    }
}

#[test]
fn test_standardized_constant_data() {
    let mut transform = Standardized::new(true, true);

    // Fit on constant data
    let const_data = vec![7.0, 7.0, 7.0, 7.0];
    transform.fit(&const_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 7.0, epsilon = 1e-10);
    assert_abs_diff_eq!(transform.fitted_std().unwrap(), 1.0, epsilon = 1e-10); // Should be 1.0 to avoid division by zero

    // Transform new data
    let test_data = Array1::from(vec![7.0, 10.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10); // (7-7)/1 = 0
    assert_abs_diff_eq!(result[[1, 0]], 3.0, epsilon = 1e-10); // (10-7)/1 = 3
}

#[test]
fn test_standardized_cross_validation_scenario() {
    // Simulate a cross-validation scenario
    let all_data = vec![
        1.0, 2.0, 3.0, // Fold 1
        4.0, 5.0, 6.0, // Fold 2
        7.0, 8.0, 9.0, // Fold 3
    ];

    // Leave fold 2 out for testing
    let train_data: Vec<f64> = all_data[0..3]
        .iter()
        .chain(all_data[6..9].iter())
        .cloned()
        .collect();
    let test_data = Array1::from(vec![4.0, 5.0, 6.0]);

    let mut transform = Standardized::new(true, true);
    transform.fit(&train_data).unwrap();

    // Train data is [1,2,3,7,8,9], mean = 5
    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 5.0, epsilon = 1e-10);

    // Calculate population std: variance = sum((x-mean)^2)/n
    // [(1-5)^2 + (2-5)^2 + (3-5)^2 + (7-5)^2 + (8-5)^2 + (9-5)^2] / 6
    // [16 + 9 + 4 + 4 + 9 + 16] / 6 = 58/6 = 29/3
    let expected_std = (29.0_f64 / 3.0).sqrt();
    assert_abs_diff_eq!(
        transform.fitted_std().unwrap(),
        expected_std,
        epsilon = 1e-10
    );

    let result = transform.evaluate(&test_data.view()).unwrap();

    // Check that test data is transformed using training statistics
    assert_abs_diff_eq!(result[[0, 0]], -1.0 / expected_std, epsilon = 1e-10); // (4-5)/std
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10); // (5-5)/std
    assert_abs_diff_eq!(result[[2, 0]], 1.0 / expected_std, epsilon = 1e-10); // (6-5)/std
}

#[test]
fn test_standardized_empty_data_error() {
    let mut transform = Standardized::new(true, true);

    let empty_data: Vec<f64> = vec![];
    let result = transform.fit(&empty_data);

    assert!(result.is_err());
}

#[test]
fn test_standardized_single_value() {
    let mut transform = Standardized::new(true, true);

    let single_data = vec![42.0];
    transform.fit(&single_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 42.0, epsilon = 1e-10);
    assert_abs_diff_eq!(transform.fitted_std().unwrap(), 1.0, epsilon = 1e-10); // Should be 1.0 for single value

    let test_data = Array1::from(vec![50.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    assert_abs_diff_eq!(result[[0, 0]], 8.0, epsilon = 1e-10); // (50-42)/1 = 8
}

#[test]
fn test_standardized_large_values() {
    let mut transform = Standardized::new(true, true);

    let large_data = vec![1e6, 2e6, 3e6, 4e6, 5e6];
    transform.fit(&large_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 3e6, epsilon = 1e-4);

    let test_data = Array1::from(vec![3e6, 6e6]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    // First value should be 0 (at mean)
    assert_abs_diff_eq!(result[[0, 0]], 0.0, epsilon = 1e-10);
    // Second value should be positive
    assert!(result[[1, 0]] > 0.0);
}

#[test]
fn test_standardized_negative_values() {
    let mut transform = Standardized::new(true, true);

    let neg_data = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
    transform.fit(&neg_data).unwrap();

    assert_abs_diff_eq!(transform.fitted_mean().unwrap(), 0.0, epsilon = 1e-10);

    let test_data = Array1::from(vec![-10.0, 0.0, 10.0]);
    let result = transform.evaluate(&test_data.view()).unwrap();

    // Check symmetry
    assert_abs_diff_eq!(result[[0, 0]], -result[[2, 0]], epsilon = 1e-10);
    assert_abs_diff_eq!(result[[1, 0]], 0.0, epsilon = 1e-10);
}

#[test]
fn test_standardized_predict() {
    let mut transform = Standardized::new(true, true);

    let train_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    transform.fit(&train_data).unwrap();

    let x = Array1::from(vec![3.0, 6.0]);
    let coef = Array1::from(vec![2.0]); // Single coefficient for single basis

    let result = transform.predict(&x.view(), Some(&coef.view())).unwrap();

    // Predictions should be scaled values * coefficient
    let scaled_3 = (3.0 - 3.0) / 2.0_f64.sqrt(); // 0
    let scaled_6 = (6.0 - 3.0) / 2.0_f64.sqrt(); // 3/sqrt(2)

    assert_abs_diff_eq!(result[0], scaled_3 * 2.0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[1], scaled_6 * 2.0, epsilon = 1e-10);
}

#[test]
fn test_standardized_predict_no_coef() {
    let mut transform = Standardized::new(true, true);

    let train_data = vec![10.0, 20.0, 30.0];
    transform.fit(&train_data).unwrap();

    let x = Array1::from(vec![20.0, 35.0]);
    let result = transform.predict(&x.view(), None).unwrap();

    // Without coefficients, should return the standardized values directly
    let expected_0 = (20.0 - 20.0) / (200.0_f64 / 3.0).sqrt(); // 0
    let expected_1 = (35.0 - 20.0) / (200.0_f64 / 3.0).sqrt();

    assert_abs_diff_eq!(result[0], expected_0, epsilon = 1e-10);
    assert_abs_diff_eq!(result[1], expected_1, epsilon = 1e-10);
}
