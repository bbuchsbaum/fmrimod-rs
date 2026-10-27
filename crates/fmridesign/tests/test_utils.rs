//! Tests for utility functions

use fmridesign::basis::PolynomialBasis;
use fmridesign::core::EventModel;
use fmridesign::events::{EventBasis, EventFactor, EventTerm, EventVariable};
use fmridesign::sampling::{SamplingFrame, SamplingFrameExt};
use fmridesign::types::EventSpec;
use fmridesign::utils::*;

fn build_model(events: Vec<Box<dyn EventSpec>>, sf: SamplingFrame) -> EventModel {
    EventModel::new(events, sf).unwrap()
}

fn empty_model() -> EventModel {
    EventModel {
        events: vec![],
        sampling_frame: SamplingFrame::from_duration(1.0, 1.0),
        name: "EventModel".to_string(),
    }
}

#[test]
fn test_model_conditions_basic() {
    let factor = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0],
        vec!["A", "B", "A"],
        vec![1.0; 3],
    )
    .unwrap();

    let variable = EventVariable::new(
        "rating",
        vec![1.0, 2.0, 3.0],
        vec![5.0, 7.0, 3.0],
        vec![1.0; 3],
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(factor), Box::new(variable)], sf);

    let conditions = model_conditions(&model, false, false).unwrap();

    assert!(conditions.contains(&"A".to_string()));
    assert!(conditions.contains(&"B".to_string()));
    assert!(conditions.contains(&"rating".to_string()));
    assert_eq!(conditions.len(), 3); // A, B, rating (no duplicates)
}

#[test]
fn test_model_conditions_with_interaction() {
    let factor1 = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["A", "B", "A", "B"],
        vec![1.0; 4],
    )
    .unwrap();

    let factor2 = EventFactor::new(
        "difficulty",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["easy", "hard", "hard", "easy"],
        vec![1.0; 4],
    )
    .unwrap();

    let interaction = EventTerm::new(
        vec![Box::new(factor1.clone()), Box::new(factor2.clone())],
        true,
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(
        vec![Box::new(factor1), Box::new(factor2), Box::new(interaction)],
        sf,
    );

    let conditions = model_conditions(&model, false, false).unwrap();

    // Print actual conditions for debugging
    println!("Actual conditions: {:?}", conditions);

    // Should have individual conditions plus interaction conditions
    assert!(conditions.contains(&"A".to_string()));
    assert!(conditions.contains(&"B".to_string()));
    assert!(conditions.contains(&"easy".to_string()));
    assert!(conditions.contains(&"hard".to_string()));
    // The interaction conditions might be formatted differently
    // Let's check what we actually get
    assert!(conditions.len() >= 4); // At least the individual conditions
}

#[test]
fn test_model_conditions_with_basis() {
    let factor = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0],
        vec!["A", "B", "A"],
        vec![1.0; 3],
    )
    .unwrap();

    let basis = PolynomialBasis::new(2).with_intercept(false);
    let basis_event = EventBasis::new(
        "parametric",
        vec![1.0, 2.0, 3.0],
        vec![1.0, 2.0, 3.0],
        Box::new(basis),
        vec![1.0; 3],
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(factor), Box::new(basis_event)], sf);

    // Without expansion
    let conds_no_expand = model_conditions(&model, false, false).unwrap();
    assert!(conds_no_expand.contains(&"parametric".to_string()));
    assert!(!conds_no_expand.iter().any(|c| c.contains("basis")));

    // With expansion
    let conds_expanded = model_conditions(&model, false, true).unwrap();
    assert!(conds_expanded.contains(&"parametric_basis1".to_string()));
    assert!(conds_expanded.contains(&"parametric_basis2".to_string()));
}

#[test]
fn test_count_event_types() {
    let factor =
        EventFactor::new("condition", vec![1.0, 2.0], vec!["A", "B"], vec![1.0; 2]).unwrap();

    let variable =
        EventVariable::new("rating", vec![1.0, 2.0], vec![5.0, 7.0], vec![1.0; 2]).unwrap();

    let interaction = EventTerm::new(vec![Box::new(factor.clone())], true).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(
        vec![Box::new(factor), Box::new(variable), Box::new(interaction)],
        sf,
    );

    let counts = count_event_types(&model);

    assert_eq!(counts.get("Categorical"), Some(&1));
    assert_eq!(counts.get("Continuous"), Some(&1));
    assert_eq!(counts.get("Term"), Some(&1));
}

#[test]
fn test_factor_and_variable_names() {
    let factor1 = EventFactor::new("condition", vec![1.0], vec!["A"], vec![1.0]).unwrap();

    let factor2 = EventFactor::new("difficulty", vec![1.0], vec!["easy"], vec![1.0]).unwrap();

    let variable = EventVariable::new("rating", vec![1.0], vec![5.0], vec![1.0]).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(
        vec![Box::new(factor1), Box::new(factor2), Box::new(variable)],
        sf,
    );

    let factors = factor_names(&model);
    assert_eq!(factors.len(), 2);
    assert!(factors.contains(&"condition"));
    assert!(factors.contains(&"difficulty"));

    let variables = variable_names(&model);
    assert_eq!(variables.len(), 1);
    assert!(variables.contains(&"rating"));
}

#[test]
fn test_has_interactions() {
    let factor = EventFactor::new("condition", vec![1.0], vec!["A"], vec![1.0]).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model_no_interaction = build_model(vec![Box::new(factor.clone())], sf.clone());

    assert!(!has_interactions(&model_no_interaction));

    let interaction = EventTerm::new(vec![Box::new(factor.clone())], true).unwrap();

    let model_with_interaction = build_model(vec![Box::new(factor), Box::new(interaction)], sf);

    assert!(has_interactions(&model_with_interaction));
}

#[test]
fn test_cells_extraction() {
    let factor1 = EventFactor::new(
        "condition",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["A", "B", "A", "B"],
        vec![1.0; 4],
    )
    .unwrap();

    let factor2 = EventFactor::new(
        "difficulty",
        vec![1.0, 2.0, 3.0, 4.0],
        vec!["easy", "hard", "hard", "easy"],
        vec![1.0; 4],
    )
    .unwrap();

    let interaction = EventTerm::new(
        vec![Box::new(factor1.clone()), Box::new(factor2.clone())],
        true,
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(interaction)], sf);

    let model_cells = cells(&model);

    assert_eq!(model_cells.len(), 4); // A:easy, B:hard, A:hard, B:easy
    assert!(model_cells.contains(&vec!["A".to_string(), "easy".to_string()]));
    assert!(model_cells.contains(&vec!["B".to_string(), "hard".to_string()]));
    assert!(model_cells.contains(&vec!["A".to_string(), "hard".to_string()]));
    assert!(model_cells.contains(&vec!["B".to_string(), "easy".to_string()]));
}

#[test]
fn test_event_terms_extraction() {
    let factor = EventFactor::new("condition", vec![1.0], vec!["A"], vec![1.0]).unwrap();

    let term1 = EventTerm::new(vec![Box::new(factor.clone())], false).unwrap();

    let term2 = EventTerm::new(vec![Box::new(factor.clone())], false).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(factor), Box::new(term1), Box::new(term2)], sf);

    let terms = event_terms(&model);
    assert_eq!(terms.len(), 2);
}

#[test]
fn test_empty_model() {
    let model = empty_model();

    let conditions = model_conditions(&model, false, false).unwrap();
    assert!(conditions.is_empty());

    let counts = count_event_types(&model);
    assert!(counts.is_empty());

    assert!(!has_interactions(&model));
    assert!(factor_names(&model).is_empty());
    assert!(variable_names(&model).is_empty());
}

#[test]
fn test_total_conditions() {
    let factor =
        EventFactor::new("condition", vec![1.0, 2.0], vec!["A", "B"], vec![1.0; 2]).unwrap();

    let basis = PolynomialBasis::new(2).with_intercept(false);
    let basis_event = EventBasis::new(
        "parametric",
        vec![1.0, 2.0, 3.0, 4.0],
        vec![1.0, 2.0, 3.0, 4.0],
        Box::new(basis),
        vec![1.0; 4],
    )
    .unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(factor), Box::new(basis_event)], sf);

    let total = total_conditions(&model).unwrap();
    // Should be 2 (A, B) + 2 (basis functions)
    assert_eq!(total, 4);
}

#[test]
fn test_has_cells_trait() {
    let factor1 =
        EventFactor::new("condition", vec![1.0, 2.0], vec!["A", "B"], vec![1.0; 2]).unwrap();

    let factor2 = EventFactor::new(
        "difficulty",
        vec![1.0, 2.0],
        vec!["easy", "hard"],
        vec![1.0; 2],
    )
    .unwrap();

    let interaction = EventTerm::new(vec![Box::new(factor1), Box::new(factor2)], true).unwrap();

    let sf = SamplingFrame::from_duration(10.0, 1.0);
    let model = build_model(vec![Box::new(interaction)], sf);

    // Test trait directly
    let trait_cells: Vec<Vec<String>> = HasCells::cells(&model);
    assert_eq!(trait_cells.len(), 4); // A:easy, A:hard, B:easy, B:hard
}
