//! Tests for EventModel design matrix generation

use fmridesign::core::EventModel;
use fmridesign::events::{EventFactor, EventVariable};
use fmridesign::sampling::uniform_sampling_frame;
use fmridesign::types::EventSpec;

fn build_model(events: Vec<Box<dyn EventSpec>>, n_scans: usize, tr: f64) -> EventModel {
    EventModel::new(events, uniform_sampling_frame(n_scans, tr, None).unwrap()).unwrap()
}

#[test]
fn test_event_model_design_matrix_generation() {
    // Create events
    let event1 = EventVariable::new(
        "stimulus",
        vec![1.0, 5.0, 10.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
    )
    .unwrap();

    let event2 = EventFactor::new(
        "condition",
        vec![2.0, 7.0, 12.0],
        vec!["A", "B", "A"],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();

    // Create sampling frame
    let model =
        build_model(vec![Box::new(event1), Box::new(event2)], 50, 2.0).with_name("test_model");

    // Generate design matrix
    let dm = model.generate_design_matrix().unwrap();

    // Check dimensions
    assert_eq!(dm.nrows(), 50); // 50 time points
    assert_eq!(dm.ncols(), 3); // 1 column for EventVariable + 2 for EventFactor

    // Check column names
    let col_names = model.column_names();
    assert_eq!(col_names, vec!["stimulus", "condition.A", "condition.B"]);
}

#[test]
fn test_event_model_empty() {
    let sampling_frame = uniform_sampling_frame(50, 2.0, None).unwrap();

    // Empty event models are rejected at construction time.
    let result = EventModel::new(vec![], sampling_frame);
    assert!(result.is_err());
}

#[test]
fn test_event_model_no_sampling_frame() {
    let event = EventVariable::new(
        "stimulus",
        vec![1.0, 5.0, 10.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
    )
    .unwrap();

    let result = fmridesign::core::EventModelBuilder::new()
        .add_event(Box::new(event))
        .build();
    assert!(result.is_err());
}

#[test]
fn test_event_model_caching() {
    let event = EventVariable::new(
        "stimulus",
        vec![1.0, 5.0, 10.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 2.0],
    )
    .unwrap();

    let model = build_model(vec![Box::new(event)], 50, 2.0);

    // Generate design matrix
    let dm1 = model.generate_design_matrix().unwrap();

    // Check it has the expected properties
    assert_eq!(dm1.nrows(), 50);
    assert_eq!(dm1.ncols(), 1);
}
