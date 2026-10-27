//! Utility functions for EventModel and related types.
//!
//! This module provides implementations of generic traits for EventModel
//! and convenience functions for extracting information from models.

use crate::core::EventModel;
use crate::events::{EventBasis, EventFactor, EventTerm};
use crate::types::{EventSpec, EventType, ValidationError};
use crate::utils::{HasCells, HasConditions};
use std::collections::HashMap;

/// Extract cells (unique factor level combinations) from an EventModel
impl HasCells for EventModel {
    fn cells(&self) -> Vec<Vec<String>> {
        let mut all_cells = Vec::new();

        // Iterate through all events and collect cells from EventTerms
        for event in self.events() {
            if let Some(term) = event.as_any().downcast_ref::<EventTerm>() {
                if let Ok(cells_obj) = term.cells(false) {
                    // Convert Cells to Vec<Vec<String>>
                    let cells_vec = cells_obj.into_cells();
                    all_cells.extend(cells_vec);
                }
            }
        }

        // Remove duplicates while preserving order
        let mut unique_cells = Vec::new();
        for cell in all_cells {
            if !unique_cells.contains(&cell) {
                unique_cells.push(cell);
            }
        }

        unique_cells
    }
}

/// Extract conditions from an EventModel
impl HasConditions for EventModel {
    fn conditions(&self) -> &[String] {
        // This is a bit tricky since we need to return a reference
        // In practice, we might need to store conditions in EventModel
        // For now, return empty slice
        &[]
    }
}

/// Convenience function to extract all cells from an EventModel
pub fn cells(model: &EventModel) -> Vec<Vec<String>> {
    HasCells::cells(model)
}

/// Convenience function to extract conditions from various event types
pub fn conditions<T: EventSpec>(event: &T) -> Result<Vec<String>, ValidationError> {
    match event.event_type() {
        EventType::Categorical => {
            if let Some(factor) = event.as_any().downcast_ref::<EventFactor>() {
                if let Some(levels) = factor.levels() {
                    Ok(levels.to_vec())
                } else {
                    Ok(vec![])
                }
            } else {
                Err(ValidationError::InvalidType {
                    expected: "EventFactor".to_string(),
                    actual: "Unknown".to_string(),
                })
            }
        }
        EventType::Term => {
            if let Some(term) = event.as_any().downcast_ref::<EventTerm>() {
                term.conditions(false, false)
            } else {
                Err(ValidationError::InvalidType {
                    expected: "EventTerm".to_string(),
                    actual: "Unknown".to_string(),
                })
            }
        }
        _ => Ok(vec![event.name().to_string()]),
    }
}

/// Extract all condition names from an EventModel
pub fn model_conditions(
    model: &EventModel,
    drop_empty: bool,
    expand_basis: bool,
) -> Result<Vec<String>, ValidationError> {
    let mut all_conditions = Vec::new();

    for event in model.events() {
        match event.event_type() {
            EventType::Categorical => {
                if let Some(factor) = event.as_any().downcast_ref::<EventFactor>() {
                    if let Some(levels) = factor.levels() {
                        all_conditions.extend(levels.iter().cloned());
                    }
                }
            }
            EventType::Term => {
                if let Some(term) = event.as_any().downcast_ref::<EventTerm>() {
                    if let Ok(conds) = term.conditions(drop_empty, expand_basis) {
                        all_conditions.extend(conds);
                    }
                }
            }
            EventType::Continuous | EventType::Matrix => {
                all_conditions.push(event.name().to_string());
            }
            EventType::Basis => {
                if let Some(basis_event) = event.as_any().downcast_ref::<EventBasis>() {
                    if expand_basis {
                        all_conditions.extend(basis_event.basis_names());
                    } else {
                        all_conditions.push(event.name().to_string());
                    }
                }
            }
        }
    }

    // Remove duplicates while preserving order
    let mut unique_conditions = Vec::new();
    for cond in all_conditions {
        if !unique_conditions.contains(&cond) {
            unique_conditions.push(cond);
        }
    }

    Ok(unique_conditions)
}

/// Get event terms from an EventModel
pub fn event_terms(model: &EventModel) -> Vec<&EventTerm> {
    model
        .events()
        .iter()
        .filter_map(|e| e.as_any().downcast_ref::<EventTerm>())
        .collect()
}

/// Count the number of each event type in the model
pub fn count_event_types(model: &EventModel) -> HashMap<String, usize> {
    let mut counts = HashMap::new();

    for event in model.events() {
        let type_name = format!("{:?}", event.event_type());
        *counts.entry(type_name).or_insert(0) += 1;
    }

    counts
}

/// Get all factor names from the model
pub fn factor_names(model: &EventModel) -> Vec<&str> {
    model
        .events()
        .iter()
        .filter(|e| e.event_type() == EventType::Categorical)
        .map(|e| e.name())
        .collect()
}

/// Get all continuous variable names from the model
pub fn variable_names(model: &EventModel) -> Vec<&str> {
    model
        .events()
        .iter()
        .filter(|e| {
            matches!(
                e.event_type(),
                EventType::Continuous | EventType::Matrix | EventType::Basis
            )
        })
        .map(|e| e.name())
        .collect()
}

/// Check if the model has any interactions
pub fn has_interactions(model: &EventModel) -> bool {
    model
        .events()
        .iter()
        .any(|e| e.event_type() == EventType::Term)
}

/// Get the total number of conditions/columns in the design matrix
pub fn total_conditions(model: &EventModel) -> Result<usize, ValidationError> {
    model_conditions(model, false, true).map(|c| c.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::EventModelBuilder;
    use crate::events::{EventFactor, EventVariable};
    use crate::sampling::{SamplingFrame, SamplingFrameExt};

    #[test]
    fn test_model_conditions() {
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
        let model = EventModelBuilder::new()
            .sampling_frame(sf)
            .add_event(Box::new(factor))
            .add_event(Box::new(variable))
            .build()
            .unwrap();

        let conds = model_conditions(&model, false, false).unwrap();
        assert!(conds.contains(&"A".to_string()));
        assert!(conds.contains(&"B".to_string()));
        assert!(conds.contains(&"rating".to_string()));
    }

    #[test]
    fn test_count_event_types() {
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
        let model = EventModelBuilder::new()
            .sampling_frame(sf)
            .add_event(Box::new(factor))
            .add_event(Box::new(variable))
            .build()
            .unwrap();

        let counts = count_event_types(&model);
        assert_eq!(counts.get("Categorical"), Some(&1));
        assert_eq!(counts.get("Continuous"), Some(&1));
    }
}
