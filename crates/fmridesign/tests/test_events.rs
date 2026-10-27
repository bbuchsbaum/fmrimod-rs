//! Tests for event specifications

use approx::assert_relative_eq;
use fmridesign::events::{EventFactor, EventVariable};
use fmridesign::types::{EventSpec, ValidationError};
use ndarray::{array, Array1, Array2};
use std::collections::HashMap;

#[test]
fn test_event_variable_as_impulse() {
    let onsets = vec![1.0, 3.0, 5.0];
    let values = vec![1.0, 1.0, 1.0]; // All ones for impulse
    let durations = vec![0.0, 0.0, 0.0]; // Zero duration for impulse
    let event = EventVariable::new("impulse", onsets, values, durations).unwrap();

    assert_eq!(event.name(), "impulse");
    assert_eq!(event.n_events(), 3);
    assert_eq!(event.onsets()[0], 1.0);
    assert_eq!(event.onsets()[1], 3.0);
    assert_eq!(event.onsets()[2], 5.0);
}

#[test]
fn test_event_variable_with_amplitudes() {
    let onsets = vec![1.0, 3.0, 5.0];
    let amplitudes = vec![1.0, 2.0, 0.5];
    let durations = vec![0.0, 0.0, 0.0];
    let event = EventVariable::new("impulse", onsets, amplitudes, durations).unwrap();

    // EventVariable centers values by default, so check raw values
    assert_eq!(event.raw_values[0], 1.0);
    assert_eq!(event.raw_values[1], 2.0);
    assert_eq!(event.raw_values[2], 0.5);
}

#[test]
fn test_event_variable_invalid() {
    // Empty onsets
    let empty_onsets = vec![];
    let empty_values = vec![];
    let empty_durations = vec![];
    assert!(EventVariable::new("test", empty_onsets, empty_values, empty_durations).is_err());

    // Mismatched lengths
    let onsets = vec![1.0, 2.0];
    let values = vec![1.0, 2.0, 3.0];
    let durations = vec![0.0, 0.0];
    assert!(EventVariable::new("test", onsets, values, durations).is_err());
}

#[test]
fn test_event_variable_as_block() {
    let onsets = vec![1.0, 5.0, 10.0];
    let durations = vec![2.0, 3.0, 1.0];
    let values = vec![1.0, 1.0, 1.0]; // All ones for block
    let event = EventVariable::new("block", onsets, values, durations).unwrap();

    assert_eq!(event.name(), "block");
    assert_eq!(event.n_events(), 3);
    assert_eq!(event.durations()[0], 2.0);
    assert_eq!(event.durations()[1], 3.0);
    assert_eq!(event.durations()[2], 1.0);
}

#[test]
fn test_event_variable_block_with_amplitudes() {
    let onsets = vec![1.0, 5.0];
    let durations = vec![2.0, 3.0];
    let amplitudes = vec![1.0, 0.5];
    let event = EventVariable::new("block", onsets, amplitudes, durations).unwrap();

    assert_eq!(event.raw_values[0], 1.0);
    assert_eq!(event.raw_values[1], 0.5);
}

#[test]
fn test_event_factor() {
    let factor = EventFactor::new(
        "factor",
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec!["A", "B", "A", "C", "B"],
        vec![0.0, 0.0, 0.0, 0.0, 0.0], // impulse events
    )
    .unwrap();

    assert_eq!(factor.name(), "factor");
    assert_eq!(factor.n_events(), 5);
    assert_eq!(factor.n_levels(), 3);

    let levels = &factor.levels;
    assert!(levels.contains(&"A".to_string()));
    assert!(levels.contains(&"B".to_string()));
    assert!(levels.contains(&"C".to_string()));
}

#[test]
fn test_event_factor_split() {
    let factor = EventFactor::new(
        "factor",
        vec![1.0, 2.0, 3.0, 4.0, 5.0],
        vec!["A", "B", "A", "C", "B"],
        vec![0.0, 0.0, 0.0, 0.0, 0.0],
    )
    .unwrap();

    // Split by level
    let split = factor.split_by_level();

    // Check level "A"
    let level_a = split.get("A").unwrap();
    assert_eq!(level_a.n_events(), 2);
    assert_eq!(level_a.onsets()[0], 1.0);
    assert_eq!(level_a.onsets()[1], 3.0);

    // Check level "B"
    let level_b = split.get("B").unwrap();
    assert_eq!(level_b.n_events(), 2);
    assert_eq!(level_b.onsets()[0], 2.0);
    assert_eq!(level_b.onsets()[1], 5.0);
}

#[test]
fn test_event_factor_invalid() {
    // Empty levels
    let values: Vec<&str> = vec![];
    let onsets = vec![1.0];
    let durations = vec![0.0];
    assert!(EventFactor::new("test", onsets, values, durations).is_err());

    // Mismatched lengths
    let values = vec!["A", "B"];
    let onsets = vec![1.0, 2.0, 3.0];
    let durations = vec![0.0, 0.0, 0.0];
    assert!(EventFactor::new("test", onsets, values, durations).is_err());
}

#[test]
fn test_event_spec_durations() {
    let onsets = vec![1.0, 5.0, 10.0];
    let durations = vec![2.0, 3.0, 1.0];
    let values = vec![1.0, 1.0, 1.0];
    let event = EventVariable::new("block", onsets, values, durations).unwrap();

    // Check durations
    let event_durations = event.durations();
    assert_eq!(event_durations.len(), 3);
    assert_relative_eq!(event_durations[0], 2.0);
    assert_relative_eq!(event_durations[1], 3.0);
    assert_relative_eq!(event_durations[2], 1.0);
}

#[test]
fn test_event_spec_onsets() {
    let onsets = vec![1.0, 5.0, 10.0];
    let values = vec![1.0, 1.0, 1.0];
    let durations = vec![0.0, 0.0, 0.0];
    let event = EventVariable::new("impulse", onsets, values, durations).unwrap();

    // Check onsets
    let event_onsets = event.onsets();
    assert_eq!(event_onsets.len(), 3);
    assert_relative_eq!(event_onsets[0], 1.0);
    assert_relative_eq!(event_onsets[1], 5.0);
    assert_relative_eq!(event_onsets[2], 10.0);
}

#[test]
fn test_event_design_matrix() {
    let onsets = vec![0.0, 2.0, 4.0];
    let values = vec![1.0, 1.0, 1.0];
    let durations = vec![0.0, 0.0, 0.0];
    let event = EventVariable::new("impulse", onsets, values, durations)
        .unwrap()
        .with_center(false); // Don't center for easier testing

    let times = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let dm = event.design_matrix(&times).unwrap();

    assert_eq!(dm.shape(), &[6, 1]);
    // For impulse events, we should see values at the exact times
    assert_relative_eq!(dm[[0, 0]], 1.0); // Event at time 0
    assert_relative_eq!(dm[[1, 0]], 0.0);
    assert_relative_eq!(dm[[2, 0]], 1.0); // Event at time 2
    assert_relative_eq!(dm[[3, 0]], 0.0);
    assert_relative_eq!(dm[[4, 0]], 1.0); // Event at time 4
    assert_relative_eq!(dm[[5, 0]], 0.0);
}
