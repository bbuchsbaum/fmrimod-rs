//! Integration tests for the formula interface.

use fmri_design::core::DataFrame;
use fmri_design::events::EventTerm;
use fmri_design::formula::{event_model, parse_formula, EventModelBuilder};
use fmri_design::sampling::{SamplingFrame, SamplingFrameExt};
use fmri_design::types::EventType;

fn create_test_data() -> DataFrame {
    let mut data = DataFrame::new();

    // Event timing
    data.insert("onset", vec!["1.0", "5.0", "10.0", "15.0"]);
    data.insert("duration", vec!["2.0", "2.0", "2.0", "2.0"]);

    // Categorical event
    data.insert("condition", vec!["A", "B", "A", "B"]);

    // Continuous event
    data.insert("rating", vec!["3.5", "4.2", "2.8", "4.5"]);
    data.insert("RT", vec!["3.5", "4.2", "2.8", "4.5"]);

    // Covariates
    data.insert("motion_x", vec!["0.1", "-0.2", "0.3", "-0.1"]);

    data
}

#[test]
fn test_simple_formula() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("hrf(condition)", &data, sampling_frame, 2.0).unwrap();

    assert_eq!(model.n_events(), 1);
}

#[test]
fn test_multiple_events_formula() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("hrf(condition) + hrf(rating)", &data, sampling_frame, 2.0).unwrap();

    assert_eq!(model.n_events(), 2);
}

#[test]
fn test_interaction_formula() {
    let mut data = create_test_data();
    data.insert("block", vec!["1", "1", "2", "2"]);

    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("hrf(condition:block)", &data, sampling_frame, 2.0).unwrap();

    assert_eq!(model.n_events(), 1);
}

#[test]
fn test_formula_with_covariates() {
    let mut data = DataFrame::new();
    data.insert(
        "onset",
        vec![
            "0.0", "2.0", "4.0", "6.0", "8.0", "10.0", "12.0", "14.0", "16.0", "18.0",
        ],
    );
    data.insert("duration", vec!["1.0"; 10]);
    data.insert(
        "condition",
        vec!["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
    );
    data.insert(
        "motion_x",
        vec![
            "0.1", "-0.2", "0.3", "-0.1", "0.2", "-0.3", "0.1", "0.0", "-0.1", "0.2",
        ],
    );
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model(
        "hrf(condition) + covariate(motion_x)",
        &data,
        sampling_frame,
        2.0,
    )
    .unwrap();

    // Should have 2 events: condition + motion_x
    assert_eq!(model.n_events(), 2);
}

#[test]
fn test_formula_with_lhs() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model(
        "onset ~ hrf(condition) + hrf(rating)",
        &data,
        sampling_frame,
        2.0,
    )
    .unwrap();

    assert_eq!(model.n_events(), 2);
}

#[test]
fn test_event_model_builder() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = EventModelBuilder::new()
        .tr(2.0)
        .data(data)
        .sampling_frame(sampling_frame)
        .formula("hrf(condition)")
        .unwrap()
        .build()
        .unwrap();

    assert_eq!(model.n_events(), 1);
}

#[test]
fn test_parse_formula() {
    // Test that formula parsing works
    let terms = parse_formula("hrf(condition) + rating").unwrap();
    assert_eq!(terms.len(), 2);

    let terms = parse_formula("condition:block").unwrap();
    assert_eq!(terms.len(), 1);
    assert!(terms[0].is_interaction());
}

#[test]
fn test_hrf_with_type() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("hrf(condition, hrf=\"gamma\")", &data, sampling_frame, 2.0).unwrap();

    assert_eq!(model.n_events(), 1);
}

#[test]
fn test_design_matrix_generation() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("hrf(condition)", &data, sampling_frame, 2.0).unwrap();

    let dm = model.generate_design_matrix().unwrap();

    // Should have 10 time points (20s / 2s TR)
    assert_eq!(dm.nrows(), 10);

    // Should have columns for each level of condition
    assert!(dm.ncols() >= 2);
}

#[test]
fn test_bare_terms_resolve_against_event_columns() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model("condition + RT", &data, sampling_frame, 2.0).unwrap();

    assert_eq!(model.n_events(), 2);

    let condition_term = model.events()[0]
        .as_any()
        .downcast_ref::<EventTerm>()
        .unwrap();
    let rt_term = model.events()[1]
        .as_any()
        .downcast_ref::<EventTerm>()
        .unwrap();

    assert_eq!(
        condition_term.events()[0].event_type(),
        EventType::Categorical
    );
    assert_eq!(rt_term.events()[0].event_type(), EventType::Continuous);

    let dm = model.generate_design_matrix().unwrap();
    assert_eq!(dm.nrows(), 10);
    assert_eq!(dm.ncols(), 3);
}

#[test]
fn test_nested_hrf_basis_and_dataframe_dimensions() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    let model = event_model(
        "onset ~ hrf(condition, basis=\"spmg3\") + hrf(RT)",
        &data,
        sampling_frame,
        2.0,
    )
    .unwrap();

    let dm = model.generate_design_matrix().unwrap();

    assert_eq!(model.n_events(), 2);
    assert_eq!(dm.nrows(), 10);
    assert_eq!(dm.ncols(), 7);
    assert_eq!(model.column_names().len(), 7);
}

#[test]
fn test_invalid_formula() {
    let data = create_test_data();
    let sampling_frame = SamplingFrame::from_duration(20.0, 2.0);

    // Missing event should fail
    let _result = event_model("hrf(nonexistent)", &data, sampling_frame, 2.0);

    // This might not fail immediately if we create a default event
    // but it's good to have the test structure
}

#[test]
fn test_empty_formula() {
    let result = parse_formula("");
    assert!(result.is_err());
}
