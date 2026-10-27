//! Tests ported from Python to ensure feature parity.

use approx::assert_abs_diff_eq;
use fmri_design::basis::{BSplineBasis, PolynomialBasis, RobustScale, RobustScaleMethod, Scale};
use fmri_design::contrast::weights::TermInfo;
use fmri_design::contrast::{contrast_weights, oneway_contrast, pair_contrast, unit_contrast};
use fmri_design::core::{DataFrame, EventModel};
use fmri_design::events::{EventFactor, EventMatrix, EventTerm, EventVariable};
use fmri_design::formula::{event_model, parse_formula};
use fmri_design::sampling::{SamplingFrame, SamplingFrameExt};
use fmri_design::types::{EventSpec, EventType};
use ndarray::{array, Array1, Array2};

// ============================================================================
// Core Event Creation Tests (from test_events.py)
// ============================================================================

#[test]
fn test_event_factor_simple() {
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
}

#[test]
fn test_event_factor_level_methods() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0, 15.0],
        vec!["A", "B", "A", "C"],
        vec![2.0, 2.0, 2.0, 2.0],
    )
    .unwrap();

    // Test get_level_indices
    assert_eq!(event.get_level_indices("A"), vec![true, false, true, false]);
    assert_eq!(
        event.get_level_indices("B"),
        vec![false, true, false, false]
    );
    assert_eq!(
        event.get_level_indices("C"),
        vec![false, false, false, true]
    );
    assert_eq!(
        event.get_level_indices("D"),
        vec![false, false, false, false]
    ); // Non-existent level

    // Test get_level_onsets
    assert_eq!(event.get_level_onsets("A"), vec![1.0, 10.0]);
    assert_eq!(event.get_level_onsets("B"), vec![5.0]);

    // Test get_level_durations
    assert_eq!(event.get_level_durations("A"), vec![2.0, 2.0]);
    assert_eq!(event.get_level_durations("C"), vec![2.0]);
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

    // Check that A is active at t=0 and t=4
    assert_eq!(dm[[0, 0]], 1.0); // A at t=0
    assert_eq!(dm[[4, 0]], 1.0); // A at t=4
    assert_eq!(dm[[2, 1]], 1.0); // B at t=2
}

#[test]
fn test_event_variable_simple() {
    let event = EventVariable::new(
        "rating",
        vec![1.0, 5.0, 10.0],
        vec![3.5, 4.2, 2.8],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    assert_eq!(event.name(), "rating");
    assert_eq!(event.event_type(), EventType::Continuous);
    assert_eq!(event.n_events(), 3);
}

#[test]
fn test_event_variable_centering() {
    let mut event = EventVariable::new(
        "rating",
        vec![1.0, 5.0, 10.0],
        vec![2.0, 4.0, 6.0],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    // Default is centered
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
fn test_event_variable_design_matrix() {
    let event = EventVariable::new(
        "rating",
        vec![1.0, 3.0, 5.0],
        vec![3.5, 4.2, 2.8],
        vec![0.5, 0.5, 0.5], // Short durations
    )
    .unwrap()
    .with_center(false);

    let sampling_points = Array1::linspace(0.0, 6.0, 7);
    let dm = event.design_matrix(&sampling_points).unwrap();

    // Should have shape (7, 1) - 7 time points, 1 continuous variable
    assert_eq!(dm.shape(), &[7, 1]);

    // Check values at event times
    assert_eq!(dm[[1, 0]], 3.5); // t=1
    assert_eq!(dm[[3, 0]], 4.2); // t=3
    assert_eq!(dm[[5, 0]], 2.8); // t=5
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
    // Column names are auto-generated as parametric_1, parametric_2
}

// ============================================================================
// Design Matrix Generation Tests (from test_design_matrix.py)
// ============================================================================

#[test]
fn test_event_model_design_matrix() {
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

    // Should have 15 time points and 3 columns (2 for condition levels + 1 for rating)
    assert_eq!(dm.shape(), &[15, 3]);
    assert_eq!(model.column_names().len(), 3);
}

// ============================================================================
// Contrast Computation Tests (from test_contrast.py)
// ============================================================================

#[test]
fn test_unit_contrast() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0],
        vec!["A", "B", "A"],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    let spec = unit_contrast("main_effect", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    // Unit contrast should have equal positive weights
    assert_eq!(contrast.weights.shape(), &[2, 1]); // 2 levels, 1 contrast
    assert_abs_diff_eq!(contrast.weights[[0, 0]], 0.5, epsilon = 1e-10);
    assert_abs_diff_eq!(contrast.weights[[1, 0]], 0.5, epsilon = 1e-10);
}

#[test]
fn test_pair_contrast() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 5.0, 10.0, 15.0],
        vec!["A", "B", "C", "A"],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    let spec = pair_contrast("A_vs_B", "condition.A", "condition.B", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    // Should contrast A vs B
    assert_eq!(contrast.weights.shape(), &[3, 1]); // 3 levels, 1 contrast
    assert_abs_diff_eq!(contrast.weights[[0, 0]], 1.0, epsilon = 1e-10); // A
    assert_abs_diff_eq!(contrast.weights[[1, 0]], -1.0, epsilon = 1e-10); // B
    assert_abs_diff_eq!(contrast.weights[[2, 0]], 0.0, epsilon = 1e-10); // C
}

#[test]
fn test_oneway_contrast() {
    let event = EventFactor::new(
        "group",
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec!["ctrl", "low", "high", "ctrl", "low", "high"],
        vec![1.0; 6],
    )
    .unwrap();

    let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

    let spec = oneway_contrast("group_effect", "group", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    // For 3 levels, should generate 2 orthogonal contrasts
    assert_eq!(contrast.weights.shape(), &[3, 2]); // 3 levels, 2 contrasts

    // Check that contrasts sum to zero
    for j in 0..2 {
        let col_sum: f64 = (0..3).map(|i| contrast.weights[[i, j]]).sum();
        assert_abs_diff_eq!(col_sum, 0.0, epsilon = 1e-10);
    }
}

// ============================================================================
// Formula Parsing Tests (from test_formula.py)
// ============================================================================

#[test]
fn test_parse_simple_formula() {
    let terms = parse_formula("hrf(condition)").unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert!(terms[0].hrf.is_some());
}

#[test]
fn test_parse_multiple_terms() {
    let terms = parse_formula("hrf(condition) + rating").unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert_eq!(terms[1].events, vec!["rating"]);
}

#[test]
fn test_parse_interaction() {
    let terms = parse_formula("condition:block").unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition", "block"]);
    assert!(terms[0].is_interaction());
}

#[test]
fn test_parse_with_kwargs() {
    let terms = parse_formula("hrf(condition, hrf=\"gamma\")").unwrap();
    assert_eq!(terms.len(), 1);

    match &terms[0].hrf {
        Some(fmri_design::formula::HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
        _ => panic!("Expected named HRF"),
    }
}

// ============================================================================
// Basis Function Tests (from test_basis.py)
// ============================================================================

#[test]
fn test_polynomial_basis() {
    let x = Array1::linspace(0.0, 1.0, 10);

    // Linear basis (degree=1)
    let linear_basis = PolynomialBasis::new(1).with_intercept(true);
    let basis_matrix = linear_basis.design_matrix(&x).unwrap();
    assert_eq!(basis_matrix.shape(), &[10, 2]); // intercept + linear

    // Quadratic basis without intercept
    let quad_basis = PolynomialBasis::new(2).with_intercept(false);
    let basis_matrix = quad_basis.design_matrix(&x).unwrap();
    assert_eq!(basis_matrix.shape(), &[10, 2]); // linear + quadratic (no intercept)
}

#[test]
fn test_bspline_basis() {
    let x = Array1::linspace(0.0, 1.0, 20);

    // B-spline with df=4
    let bs = BSplineBasis::new(4).with_intercept(true);
    let basis_matrix = bs.design_matrix(&x).unwrap();
    assert_eq!(basis_matrix.shape(), &[20, 4]);

    // Check that basis functions sum to 1 (partition of unity)
    for i in 0..20 {
        let row_sum: f64 = (0..4).map(|j| basis_matrix[[i, j]]).sum();
        assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_scale_transform() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

    // Z-score scaling (center and scale)
    let scale = Scale::new(true, true);
    let scaled = scale.design_matrix(&x).unwrap();

    // Check mean is 0
    let mean: f64 = scaled.column(0).sum() / scaled.len() as f64;
    assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);

    // Check std is 1
    let variance = scaled.column(0).mapv(|v| v.powi(2)).sum() / scaled.len() as f64;
    let std = variance.sqrt();
    assert_abs_diff_eq!(std, 1.0, epsilon = 1e-10);
}

#[test]
fn test_robust_scale_transform() {
    let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 10.0]; // 10.0 is an outlier

    // Robust scaling with MAD
    let scale = RobustScale::new(true, RobustScaleMethod::Mad);
    let scaled = scale.design_matrix(&x).unwrap();

    // The outlier should have less impact on scaling
    assert!(scaled[[5, 0]].abs() > 2.0); // Outlier should still be far from center
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_pipeline_formula_to_design_matrix() {
    // Create data
    let mut data = DataFrame::new();
    data.insert("onset", vec!["1.0", "5.0", "10.0", "15.0"]);
    data.insert("condition", vec!["A", "B", "A", "B"]);
    data.insert("rating", vec!["3.5", "4.2", "2.8", "4.5"]);
    data.insert("duration", vec!["2.0", "2.0", "2.0", "2.0"]);

    // Create model from formula
    let sampling_frame = SamplingFrame::from_duration(20.0, 1.0);
    let model = event_model("hrf(condition) + rating", &data, sampling_frame, 1.0).unwrap();

    // Generate design matrix
    let dm = model.generate_design_matrix().unwrap();

    // Should have correct shape
    assert_eq!(dm.nrows(), 20); // 20 time points
    assert_eq!(dm.ncols(), 3); // 2 condition levels + 1 rating

    // Check column names
    let column_names = model.column_names();
    assert!(column_names.contains(&"condition.A".to_string()));
    assert!(column_names.contains(&"condition.B".to_string()));
    assert!(column_names.contains(&"rating".to_string()));
}

#[test]
fn test_interaction_term_design_matrix() {
    // Create factorial design
    let condition = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["A", "B", "A", "B"],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let block = EventFactor::new(
        "block",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["1", "1", "2", "2"],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    // Create interaction term
    let interaction = EventTerm::new(
        vec![Box::new(condition), Box::new(block)],
        true, // interaction=true
    )
    .unwrap();

    // Create model
    let sampling_frame = SamplingFrame::from_duration(5.0, 1.0);
    let model = EventModel::new(vec![Box::new(interaction.clone())], sampling_frame).unwrap();

    // Generate design matrix
    let dm = model.generate_design_matrix().unwrap();

    // Should have 2x2 = 4 columns for full interaction
    assert_eq!(dm.ncols(), 4);

    // Check conditions
    let conditions = interaction.conditions(false, false).unwrap();
    assert_eq!(conditions.len(), 4);
    assert!(conditions.contains(&"condition.A_block.1".to_string()));
    assert!(conditions.contains(&"condition.B_block.2".to_string()));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

#[test]
fn test_empty_events() {
    let result = EventFactor::new("empty", vec![], Vec::<&str>::new(), vec![]);
    assert!(result.is_err());
}

#[test]
fn test_mismatched_lengths() {
    let result = EventFactor::new(
        "bad",
        vec![1.0, 2.0],
        vec!["A", "B", "C"], // Wrong length
        vec![1.0, 1.0],
    );
    assert!(result.is_err());
}

#[test]
fn test_negative_durations() {
    let result = EventVariable::new(
        "bad",
        vec![1.0],
        vec![3.5],
        vec![-1.0], // Negative duration
    );
    assert!(result.is_err());
}

#[test]
fn test_non_finite_values() {
    let result = EventVariable::new(
        "bad",
        vec![1.0],
        vec![f64::NAN], // Non-finite value
        vec![1.0],
    );
    assert!(result.is_err());
}
