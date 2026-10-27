//! Tests for EventBasis continuous event modeling

use approx::assert_abs_diff_eq;
use fmri_design::basis::{BSplineBasis, PolynomialBasis, Scale, ScaleWithin};
use fmri_design::events::EventBasis;
use fmri_design::types::{BasisFunction, EventSpec, EventType};
use ndarray::{array, Array1};

#[test]
fn test_event_basis_linear() {
    let onsets = vec![0.0, 10.0, 20.0];
    let values = vec![1.0, 2.0, 3.0];
    let durations = vec![2.0, 2.0, 2.0];

    let basis = PolynomialBasis::new(1).with_intercept(false);
    let event = EventBasis::new("param", onsets, values, Box::new(basis), durations).unwrap();

    assert_eq!(event.name(), "param");
    assert_eq!(event.event_type(), EventType::Basis);
    assert_eq!(event.n_basis(), 1);
    assert_eq!(event.basis_names(), vec!["param_basis1"]);
}

#[test]
fn test_event_basis_quadratic() {
    let onsets = vec![0.0, 5.0, 10.0];
    let values = vec![1.0, 2.0, 3.0];
    let durations = vec![1.0; 3];

    let basis = PolynomialBasis::new(2).with_intercept(false);
    let event = EventBasis::new("quadratic", onsets, values, Box::new(basis), durations).unwrap();

    assert_eq!(event.n_basis(), 2); // Linear and quadratic terms
    assert_eq!(
        event.basis_names(),
        vec!["quadratic_basis1", "quadratic_basis2"]
    );
}

#[test]
fn test_event_basis_spline() {
    let onsets = vec![0.0, 2.0, 4.0, 6.0];
    let values = vec![0.0, 0.5, 0.8, 1.0];
    let durations = vec![0.0; 4]; // Impulse events

    let basis = BSplineBasis::new(3); // 3 basis functions
    let event =
        EventBasis::new("spline_param", onsets, values, Box::new(basis), durations).unwrap();

    assert_eq!(event.n_basis(), 3);
    assert_eq!(event.basis_names().len(), 3);
}

#[test]
fn test_event_basis_scaled_values() {
    let onsets = vec![0.0, 5.0, 10.0];
    let raw_values = vec![10.0, 20.0, 30.0];
    let durations = vec![1.0; 3];

    // Scale values first
    let scale = Scale::new(true, true);
    let scaled = scale
        .evaluate(&Array1::from(raw_values.clone()).view())
        .unwrap();
    let scaled_values: Vec<f64> = scaled.column(0).to_vec();

    let basis = PolynomialBasis::new(1).with_intercept(false);
    let _event = EventBasis::new(
        "scaled_param",
        onsets,
        scaled_values.clone(),
        Box::new(basis),
        durations,
    )
    .unwrap();

    // Check that values are standardized
    let mean: f64 = scaled_values.iter().sum::<f64>() / 3.0;
    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
}

#[test]
fn test_event_basis_design_matrix() {
    let onsets = vec![0.0, 4.0];
    let values = vec![1.0, 2.0];
    let durations = vec![2.0, 2.0];

    let basis = PolynomialBasis::new(1).with_intercept(true).with_raw(true);
    let event = EventBasis::new("test", onsets, values, Box::new(basis), durations).unwrap();

    let time_points = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dm = event.design_matrix(&time_points).unwrap();

    assert_eq!(dm.shape(), &[7, 2]); // 7 time points, 2 basis functions

    // First event (value=1) at t=0-2
    assert_eq!(dm[[0, 0]], 1.0); // Intercept
    assert_eq!(dm[[0, 1]], 1.0); // Value
    assert_eq!(dm[[1, 0]], 1.0); // Intercept
    assert_eq!(dm[[1, 1]], 1.0); // Value
    assert_eq!(dm[[2, 0]], 0.0); // Outside duration
    assert_eq!(dm[[2, 1]], 0.0);

    // Second event (value=2) at t=4-6
    assert_eq!(dm[[4, 0]], 1.0); // Intercept
    assert_eq!(dm[[4, 1]], 2.0); // Value
    assert_eq!(dm[[5, 0]], 1.0); // Intercept
    assert_eq!(dm[[5, 1]], 2.0); // Value
}

#[test]
fn test_event_basis_impulse_events() {
    let onsets = vec![1.0, 3.0, 5.0];
    let values = vec![0.5, 1.0, 1.5];
    let durations = vec![0.0; 3]; // All impulse events

    let basis = PolynomialBasis::new(1).with_intercept(false).with_raw(true); // Use raw values
    let event = EventBasis::new("impulse", onsets, values, Box::new(basis), durations).unwrap();

    let time_points = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let dm = event.design_matrix(&time_points).unwrap();

    // Check that impulses appear at correct time points
    assert_eq!(dm[[1, 0]], 0.5); // t=1
    assert_eq!(dm[[3, 0]], 1.0); // t=3
    assert_eq!(dm[[5, 0]], 1.5); // t=5

    // Check that other time points are zero
    assert_eq!(dm[[0, 0]], 0.0);
    assert_eq!(dm[[2, 0]], 0.0);
    assert_eq!(dm[[4, 0]], 0.0);
    assert_eq!(dm[[6, 0]], 0.0);
}

#[test]
fn test_event_basis_multirun_scaling() {
    let onsets = vec![0.0, 5.0, 100.0, 105.0]; // Two runs
    let values = vec![1.0, 2.0, 5.0, 6.0]; // Different scales per run
    let runs = vec!["run1", "run1", "run2", "run2"];
    let durations = vec![1.0; 4];

    // Apply within-run scaling
    let scale_within = ScaleWithin::new(runs.iter().map(|s| s.to_string()).collect(), true, true);
    let scaled = scale_within
        .evaluate(&Array1::from(values.clone()).view())
        .unwrap();
    let scaled_values: Vec<f64> = scaled.column(0).to_vec();

    let basis = PolynomialBasis::new(1).with_intercept(false);
    let _event = EventBasis::new(
        "multirun",
        onsets,
        scaled_values.clone(),
        Box::new(basis),
        durations,
    )
    .unwrap();

    // Check that each run is centered
    let run1_mean = (scaled_values[0] + scaled_values[1]) / 2.0;
    let run2_mean = (scaled_values[2] + scaled_values[3]) / 2.0;

    assert_abs_diff_eq!(run1_mean, 0.0, epsilon = 1e-10);
    assert_abs_diff_eq!(run2_mean, 0.0, epsilon = 1e-10);
}

#[test]
fn test_event_basis_to_dataframe() {
    let onsets = vec![1.0, 2.0, 3.0];
    let values = vec![10.0, 20.0, 30.0];
    let durations = vec![0.5, 0.5, 0.5];

    let basis = PolynomialBasis::new(2).with_intercept(false);
    let event = EventBasis::new("rating", onsets, values, Box::new(basis), durations).unwrap();

    let df = event.to_dataframe();

    // Check columns
    assert!(df.column_names().contains(&"onset".to_string()));
    assert!(df.column_names().contains(&"duration".to_string()));
    assert!(df.column_names().contains(&"rating_value".to_string()));
    assert!(df.column_names().contains(&"rating_basis1".to_string()));
    assert!(df.column_names().contains(&"rating_basis2".to_string()));

    // Check values
    let onset_col = df.get_column("onset").unwrap();
    assert_eq!(onset_col[0], 1.0);
    assert_eq!(onset_col[1], 2.0);

    let value_col = df.get_column("rating_value").unwrap();
    assert_eq!(value_col[0], 10.0);
    assert_eq!(value_col[1], 20.0);
}

#[test]
fn test_event_basis_validation() {
    let onsets = vec![1.0, 2.0];
    let values = vec![1.0]; // Wrong length
    let durations = vec![1.0, 1.0];

    let basis = PolynomialBasis::new(1).with_intercept(false);
    let result = EventBasis::new("invalid", onsets, values, Box::new(basis), durations);

    assert!(result.is_err());
}

#[test]
fn test_event_basis_nonfinite_values() {
    let onsets = vec![1.0, 2.0];
    let values = vec![1.0, f64::NAN];
    let durations = vec![1.0, 1.0];

    let basis = PolynomialBasis::new(1).with_intercept(false);
    let result = EventBasis::new("invalid", onsets, values, Box::new(basis), durations);

    assert!(result.is_err());
}

#[test]
fn test_event_basis_high_degree_polynomial() {
    let onsets = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let values = vec![0.0, 0.25, 0.5, 0.75, 1.0];
    let durations = vec![0.5; 5];

    let basis = PolynomialBasis::new(3).with_intercept(true); // Cubic with intercept
    let event = EventBasis::new("cubic", onsets, values, Box::new(basis), durations).unwrap();

    assert_eq!(event.n_basis(), 4); // intercept + x + x^2 + x^3

    let basis_names = event.basis_names();
    assert_eq!(basis_names.len(), 4);
    assert!(basis_names[0].contains("cubic_basis1"));
}
