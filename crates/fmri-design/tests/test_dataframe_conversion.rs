//! Tests for DataFrame conversion functionality.

use fmri_design::basis::PolynomialBasis;
use fmri_design::events::{EventBasis, EventFactor, EventMatrix, EventVariable};
use fmri_design::types::EventSpec;
use ndarray::Array2;

#[test]
fn test_event_factor_to_dataframe() {
    let event = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["A", "B", "A", "B"],
        vec![1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();

    let df = event.to_dataframe();

    // Check columns exist
    assert!(df.get_column("onset").is_some());
    assert!(df.get_column("duration").is_some());
    assert!(df.get_str_column("condition").is_some());

    // Check values
    let onsets = df.get_column("onset").unwrap();
    assert_eq!(onsets, vec![1.0, 2.0, 3.0, 4.0]);

    let conditions = df.get_str_column("condition").unwrap();
    assert_eq!(
        conditions,
        vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string()
        ]
    );

    // Check column order
    assert_eq!(
        df.column_names(),
        vec![
            "onset".to_string(),
            "duration".to_string(),
            "condition".to_string()
        ]
    );
}

#[test]
fn test_event_variable_to_dataframe() {
    let event = EventVariable::new(
        "rating",
        vec![1.0, 2.0, 3.0],
        vec![2.5, 3.5, 4.5],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let df = event.to_dataframe();

    // Check columns
    assert!(df.get_column("onset").is_some());
    assert!(df.get_column("duration").is_some());
    assert!(df.get_column("rating").is_some());

    // Raw amplitudes are preserved unless a transform is explicitly requested.
    assert_eq!(df.get_column("rating").unwrap(), vec![2.5, 3.5, 4.5]);
    assert!(df.get_column("rating_raw").is_none());
}

#[test]
fn test_event_variable_with_transform_to_dataframe() {
    // Create event with values that will be centered
    let event = EventVariable::new(
        "rating",
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap()
    .with_center(true);

    // Centering was explicitly requested
    let df = event.to_dataframe();

    // Should have both transformed and raw values
    assert!(df.get_column("rating").is_some());
    assert!(df.get_column("rating_raw").is_some());

    let transformed = df.get_column("rating").unwrap();
    assert_eq!(transformed, vec![-1.0, 0.0, 1.0]);

    let raw = df.get_column("rating_raw").unwrap();
    assert_eq!(raw, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_event_matrix_to_dataframe() {
    let values = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let event = EventMatrix::new("motion", vec![1.0, 2.0, 3.0], values, vec![1.0, 1.0, 1.0])
        .unwrap()
        .with_column_names(vec!["motion_x".to_string(), "motion_y".to_string()])
        .unwrap();

    let df = event.to_dataframe();

    // Check columns
    assert!(df.get_column("onset").is_some());
    assert!(df.get_column("duration").is_some());
    assert!(df.get_column("motion_x").is_some());
    assert!(df.get_column("motion_y").is_some());

    // Check values
    let x_values = df.get_column("motion_x").unwrap();
    assert_eq!(x_values, vec![1.0, 3.0, 5.0]);

    let y_values = df.get_column("motion_y").unwrap();
    assert_eq!(y_values, vec![2.0, 4.0, 6.0]);

    // Check column order
    assert_eq!(
        df.column_names(),
        vec![
            "onset".to_string(),
            "duration".to_string(),
            "motion_x".to_string(),
            "motion_y".to_string()
        ]
    );
}

#[test]
fn test_event_basis_to_dataframe() {
    let basis = PolynomialBasis::new(2).with_intercept(true).with_raw(true);
    let event = EventBasis::new(
        "rating",
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0],
        Box::new(basis),
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    let df = event.to_dataframe();

    // Check basic columns
    assert!(df.get_column("onset").is_some());
    assert!(df.get_column("duration").is_some());
    assert!(df.get_column("rating_value").is_some());

    // Check basis columns
    assert!(df.get_column("rating_basis1").is_some());
    assert!(df.get_column("rating_basis2").is_some());
    assert!(df.get_column("rating_basis3").is_some());

    // Check values
    let values = df.get_column("rating_value").unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);

    // Basis should have intercept column (all 1s)
    let basis1 = df.get_column("rating_basis1").unwrap();
    assert_eq!(basis1, vec![1.0, 1.0, 1.0]);
}

#[test]
fn test_dataframe_builder() {
    use fmri_design::core::DataFrameBuilder;

    let df = DataFrameBuilder::new()
        .with_column("x", vec![1.0, 2.0, 3.0])
        .unwrap()
        .with_column("y", vec![4.0, 5.0, 6.0])
        .unwrap()
        .with_str_column("label", vec!["A", "B", "C"])
        .unwrap()
        .build();

    assert_eq!(df.len(), 3);
    assert_eq!(
        df.column_names(),
        vec!["x".to_string(), "y".to_string(), "label".to_string()]
    );
}

#[test]
fn test_dataframe_length_validation() {
    use fmri_design::core::DataFrameBuilder;

    let result = DataFrameBuilder::new()
        .with_column("x", vec![1.0, 2.0, 3.0])
        .unwrap()
        .with_column("y", vec![4.0, 5.0]); // Wrong length

    assert!(result.is_err());
}
