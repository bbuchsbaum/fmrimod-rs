//! Core tests ported from Python to ensure feature parity.

use approx::assert_abs_diff_eq;
use fmri_design::basis::{BSplineBasis, PolynomialBasis};
use fmri_design::contrast::spec::{oneway_contrast, pair_contrast, unit_contrast};
use fmri_design::contrast::{contrast_weights, ContrastSpecEnum};
use fmri_design::core::{DataFrame, EventModel};
use fmri_design::events::{EventFactor, EventMatrix, EventTerm, EventVariable};
use fmri_design::formula::{event_model, parse_formula};
use fmri_design::sampling::{SamplingFrame, SamplingFrameExt};
use fmri_design::types::BasisFunction;
use fmri_design::types::{EventSpec, EventType};
use ndarray::{array, Array1, Array2};

// ============================================================================
// Core Event Creation Tests (from test_events.py)
// ============================================================================

#[test]
fn test_event_factor_creation_and_methods() {
    // Simple factor creation
    let event = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0],
        vec!["A", "B", "A"],
        vec![2.0, 2.0, 2.0],
    )
    .unwrap();

    assert_eq!(event.name(), "condition");
    assert_eq!(event.n_events(), 3);
    assert_eq!(
        event.levels(),
        Some(&["A".to_string(), "B".to_string()][..])
    );

    // Test level methods
    assert_eq!(event.get_level_indices("A"), vec![true, false, true]);
    assert_eq!(event.get_level_indices("B"), vec![false, true, false]);
    assert_eq!(event.get_level_indices("D"), vec![false, false, false]); // Non-existent

    assert_eq!(event.get_level_onsets("A"), vec![1.0, 10.0]);
    assert_eq!(event.get_level_durations("A"), vec![2.0, 2.0]);
}

#[test]
fn test_event_factor_design_matrix() {
    let event = EventFactor::new(
        "condition",
        vec![0.0, 2.0, 4.0],
        vec!["A", "B", "A"],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let sampling_points = Array1::linspace(0.0, 6.0, 7);
    let dm = event.design_matrix(&sampling_points).unwrap();

    // Should have shape (7, 2) - 7 time points, 2 levels
    assert_eq!(dm.shape(), &[7, 2]);

    // Check that events are active at correct times
    assert_eq!(dm[[0, 0]], 1.0); // A at t=0
    assert_eq!(dm[[4, 0]], 1.0); // A at t=4
    assert_eq!(dm[[2, 1]], 1.0); // B at t=2
}

#[test]
fn test_event_variable_creation_and_centering() {
    let mut event = EventVariable::new(
        "rating",
        vec![1.0, 5.0, 10.0],
        vec![2.0, 4.0, 6.0],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    assert_eq!(event.name(), "rating");
    assert_eq!(event.event_type(), EventType::Continuous);

    // Test centering
    event = event.with_center(true);
    let mean = (2.0 + 4.0 + 6.0) / 3.0; // 4.0

    let sampling_points = array![1.0, 5.0, 10.0];
    let dm = event.design_matrix(&sampling_points).unwrap();

    // Values should be centered around mean
    assert_abs_diff_eq!(dm[[0, 0]], 2.0 - mean, epsilon = 1e-10);
    assert_abs_diff_eq!(dm[[1, 0]], 4.0 - mean, epsilon = 1e-10);
    assert_abs_diff_eq!(dm[[2, 0]], 6.0 - mean, epsilon = 1e-10);
}

#[test]
fn test_event_matrix_creation() {
    let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let event = EventMatrix::new(
        "parametric",
        vec![1.0, 5.0, 10.0],
        data.clone(),
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    assert_eq!(event.name(), "parametric");
    assert_eq!(event.n_events(), 3);
    assert_eq!(event.n_columns(), 2);
}

// ============================================================================
// Design Matrix Generation Tests
// ============================================================================

#[test]
fn test_event_model_design_matrix_generation() {
    // Create simple events
    let event1 = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0],
        vec!["A", "B", "A"],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let event2 = EventVariable::new(
        "rating",
        vec![2.0, 6.0, 11.0],
        vec![3.5, 4.2, 2.8],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap()
    .with_center(false);

    // Create event terms
    let term1 = EventTerm::new(vec![Box::new(event1)], false).unwrap();
    let term2 = EventTerm::new(vec![Box::new(event2)], false).unwrap();

    // Create model
    let sampling_frame = SamplingFrame::from_duration(15.0, 1.0);
    let model = EventModel::new(vec![Box::new(term1), Box::new(term2)], sampling_frame).unwrap();

    // Generate design matrix
    let dm = model.generate_design_matrix().unwrap();

    // Should have 15 time points and 3 columns (2 for condition + 1 for rating)
    assert_eq!(dm.shape(), &[15, 3]);
    assert_eq!(model.column_names().len(), 3);
}

// ============================================================================
// Contrast Computation Tests
// ============================================================================

#[test]
fn test_contrast_computations() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0, 15.0],
        vec!["A", "B", "C", "A"],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    // Test unit contrast
    let unit_spec = unit_contrast("main_effect", None);
    let contrast = contrast_weights(&ContrastSpecEnum::Unit(unit_spec), &term).unwrap();
    assert_eq!(contrast.weights.shape(), &[3, 1]); // 3 levels, 1 contrast

    // Test pair contrast
    let pair_spec = pair_contrast("A_vs_B", "condition.A", "condition.B", None);
    let contrast = contrast_weights(&ContrastSpecEnum::Pair(pair_spec), &term).unwrap();
    assert_eq!(contrast.weights.shape(), &[3, 1]);
    assert_abs_diff_eq!(contrast.weights[[0, 0]], 1.0, epsilon = 1e-10); // A
    assert_abs_diff_eq!(contrast.weights[[1, 0]], -1.0, epsilon = 1e-10); // B
    assert_abs_diff_eq!(contrast.weights[[2, 0]], 0.0, epsilon = 1e-10); // C

    // Test oneway contrast
    let oneway_spec = oneway_contrast("group_effect", "group", None);
    let contrast = contrast_weights(&ContrastSpecEnum::Oneway(oneway_spec), &term).unwrap();
    assert_eq!(contrast.weights.shape(), &[3, 2]); // 3 levels, 2 contrasts
}

// ============================================================================
// Formula Parsing Tests
// ============================================================================

#[test]
fn test_formula_parsing() {
    // Simple formula
    let terms = parse_formula("hrf(condition)").unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert!(terms[0].hrf.is_some());

    // Multiple terms
    let terms = parse_formula("hrf(condition) + rating").unwrap();
    assert_eq!(terms.len(), 2);

    // Interaction
    let terms = parse_formula("condition:block").unwrap();
    assert_eq!(terms.len(), 1);
    assert!(terms[0].is_interaction());

    // With keyword arguments
    let terms = parse_formula("hrf(condition, hrf=\"gamma\")").unwrap();
    assert_eq!(terms.len(), 1);
}

// ============================================================================
// Basis Function Tests
// ============================================================================

#[test]
fn test_polynomial_basis() {
    let x = Array1::linspace(0.0, 1.0, 10);

    // Linear basis with intercept
    let linear_basis = PolynomialBasis::new(1).with_intercept(true);
    let basis_matrix = linear_basis.evaluate(&x.view()).unwrap();
    assert_eq!(basis_matrix.shape(), &[10, 2]); // intercept + linear

    // Quadratic without intercept
    let quad_basis = PolynomialBasis::new(2).with_intercept(false);
    let basis_matrix = quad_basis.evaluate(&x.view()).unwrap();
    assert_eq!(basis_matrix.shape(), &[10, 2]); // linear + quadratic
}

#[test]
fn test_bspline_basis() {
    let x = Array1::linspace(0.0, 1.0, 20);

    // B-spline with df=4
    let bs = BSplineBasis::new(4).with_intercept(true);
    let basis_matrix = bs.evaluate(&x.view()).unwrap();
    assert_eq!(basis_matrix.shape(), &[20, 4]);

    // Check partition of unity property
    for i in 0..20 {
        let row_sum: f64 = (0..4).map(|j| basis_matrix[[i, j]]).sum();
        assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline_from_formula() {
    // Create data (only event-related data, not time series)
    let mut data = DataFrame::new();
    data.insert("onset", vec!["1.0", "5.0", "10.0", "15.0"]);
    data.insert("condition", vec!["A", "B", "A", "B"]);
    data.insert("duration", vec!["2.0", "2.0", "2.0", "2.0"]);

    // Create model from formula
    let sampling_frame = SamplingFrame::from_duration(20.0, 1.0);
    let model = event_model("hrf(condition)", &data, sampling_frame, 1.0).unwrap();

    // Generate design matrix
    let dm = model.generate_design_matrix().unwrap();

    // Check shape and columns
    assert_eq!(dm.nrows(), 20); // 20 time points
    assert_eq!(dm.ncols(), 2); // 2 condition levels (A and B)

    let column_names = model.column_names();
    assert!(column_names.contains(&"condition.A".to_string()));
    assert!(column_names.contains(&"condition.B".to_string()));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_error_handling() {
    // Empty events
    let result = EventFactor::new("empty", vec![], Vec::<&str>::new(), vec![]);
    assert!(result.is_err());

    // Mismatched lengths
    let result = EventFactor::new(
        "bad",
        vec![1.0, 2.0],
        vec!["A", "B", "C"], // Wrong length
        vec![1.0, 1.0],
    );
    assert!(result.is_err());

    // Negative durations
    let result = EventVariable::new(
        "bad",
        vec![1.0],
        vec![3.5],
        vec![-1.0], // Negative duration
    );
    assert!(result.is_err());
}
