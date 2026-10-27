//! Utility functions for working with events and onsets.

use crate::types::{DurationArray, OnsetArray};
use ndarray::Array1;
use std::collections::HashMap;

/// Split event onsets by a grouping criterion.
///
/// Splits the onsets of an event into groups based on a
/// categorical variable or custom function.
///
/// # Arguments
/// * `onsets` - Event onset times
/// * `labels` - Group labels for each onset
///
/// # Returns
/// HashMap mapping group labels to onset arrays
pub fn split_onsets_by_labels(
    onsets: &OnsetArray,
    labels: &[String],
) -> HashMap<String, OnsetArray> {
    assert_eq!(
        onsets.len(),
        labels.len(),
        "Onsets and labels must have same length"
    );

    let mut groups: HashMap<String, Vec<f64>> = HashMap::new();

    // Group onsets by label
    for (onset, label) in onsets.iter().zip(labels.iter()) {
        groups
            .entry(label.clone())
            .or_insert_with(Vec::new)
            .push(*onset);
    }

    // Convert to arrays
    groups
        .into_iter()
        .map(|(label, vec)| (label, Array1::from_vec(vec)))
        .collect()
}

/// Split event onsets by a custom function.
///
/// # Arguments
/// * `onsets` - Event onset times
/// * `grouper` - Function that assigns each onset to a group
///
/// # Returns
/// HashMap mapping group labels to onset arrays
pub fn split_onsets_by_fn<F, G>(onsets: &OnsetArray, grouper: F) -> HashMap<G, OnsetArray>
where
    F: Fn(f64) -> G,
    G: std::hash::Hash + Eq + Clone,
{
    let mut groups: HashMap<G, Vec<f64>> = HashMap::new();

    // Group onsets by function result
    for &onset in onsets.iter() {
        let group = grouper(onset);
        groups.entry(group).or_insert_with(Vec::new).push(onset);
    }

    // Convert to arrays
    groups
        .into_iter()
        .map(|(label, vec)| (label, Array1::from_vec(vec)))
        .collect()
}

/// Block assignment information
#[derive(Debug, Clone)]
pub struct BlockAssignment {
    /// Block label/identifier
    pub label: String,
    /// Start time of block
    pub onset: f64,
    /// Duration of block
    pub duration: f64,
}

/// Split events by experimental blocks based on timing.
///
/// # Arguments
/// * `onsets` - Event onset times
/// * `blocks` - Block definitions
///
/// # Returns
/// HashMap mapping block labels to indices of events in that block
pub fn split_by_block_timing(
    onsets: &OnsetArray,
    blocks: &[BlockAssignment],
) -> HashMap<String, Vec<usize>> {
    let mut block_indices: HashMap<String, Vec<usize>> = HashMap::new();

    // For each onset, find which block it belongs to
    for (i, &onset) in onsets.iter().enumerate() {
        for block in blocks {
            let block_end = block.onset + block.duration;
            if onset >= block.onset && onset < block_end {
                block_indices
                    .entry(block.label.clone())
                    .or_insert_with(Vec::new)
                    .push(i);
                break;
            }
        }
    }

    block_indices
}

/// Split events by block labels.
///
/// # Arguments
/// * `n_events` - Number of events
/// * `block_labels` - Block label for each event
///
/// # Returns
/// HashMap mapping block labels to indices of events in that block
pub fn split_by_block_labels(
    n_events: usize,
    block_labels: &[String],
) -> HashMap<String, Vec<usize>> {
    assert_eq!(
        n_events,
        block_labels.len(),
        "Number of events must match number of block labels"
    );

    let mut block_indices: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, label) in block_labels.iter().enumerate() {
        block_indices
            .entry(label.clone())
            .or_insert_with(Vec::new)
            .push(i);
    }

    block_indices
}

/// Extract a subset of events based on indices.
///
/// # Arguments
/// * `onsets` - All onset times
/// * `durations` - All durations
/// * `indices` - Indices to extract
///
/// # Returns
/// Tuple of (subset_onsets, subset_durations)
pub fn extract_event_subset(
    onsets: &OnsetArray,
    durations: &DurationArray,
    indices: &[usize],
) -> (OnsetArray, DurationArray) {
    let subset_onsets: Vec<f64> = indices.iter().map(|&i| onsets[i]).collect();

    let subset_durations: Vec<f64> = indices.iter().map(|&i| durations[i]).collect();

    (
        Array1::from_vec(subset_onsets),
        Array1::from_vec(subset_durations),
    )
}

/// Find events within a time window.
///
/// # Arguments
/// * `onsets` - Event onset times
/// * `start` - Start of time window
/// * `end` - End of time window
///
/// # Returns
/// Indices of events within the window
pub fn events_in_window(onsets: &OnsetArray, start: f64, end: f64) -> Vec<usize> {
    onsets
        .iter()
        .enumerate()
        .filter_map(|(i, &onset)| {
            if onset >= start && onset < end {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

/// Compute inter-onset intervals.
///
/// # Arguments
/// * `onsets` - Event onset times (assumed sorted)
///
/// # Returns
/// Array of inter-onset intervals
pub fn inter_onset_intervals(onsets: &OnsetArray) -> Array1<f64> {
    if onsets.len() < 2 {
        return Array1::zeros(0);
    }

    let mut intervals = Vec::with_capacity(onsets.len() - 1);
    for i in 1..onsets.len() {
        intervals.push(onsets[i] - onsets[i - 1]);
    }

    Array1::from_vec(intervals)
}

/// Group consecutive events based on a time threshold.
///
/// Events are grouped together if they occur within `threshold` seconds
/// of each other.
///
/// # Arguments
/// * `onsets` - Event onset times (assumed sorted)
/// * `threshold` - Maximum time between events in same group
///
/// # Returns
/// Vector of group indices for each event
pub fn group_consecutive_events(onsets: &OnsetArray, threshold: f64) -> Vec<usize> {
    if onsets.is_empty() {
        return Vec::new();
    }

    let mut groups = vec![0];
    let mut current_group = 0;

    for i in 1..onsets.len() {
        if onsets[i] - onsets[i - 1] > threshold {
            current_group += 1;
        }
        groups.push(current_group);
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_split_onsets_by_labels() {
        let onsets = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let labels = vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
        ];

        let groups = split_onsets_by_labels(&onsets, &labels);

        assert_eq!(groups.len(), 2);
        assert_eq!(groups["A"], array![1.0, 3.0, 5.0]);
        assert_eq!(groups["B"], array![2.0, 4.0]);
    }

    #[test]
    fn test_split_onsets_by_fn() {
        let onsets = array![1.0, 2.5, 3.0, 4.5, 5.0];
        let groups =
            split_onsets_by_fn(&onsets, |onset| if onset < 3.0 { "early" } else { "late" });

        assert_eq!(groups.len(), 2);
        assert_eq!(groups["early"], array![1.0, 2.5]);
        assert_eq!(groups["late"], array![3.0, 4.5, 5.0]);
    }

    #[test]
    fn test_split_by_block_timing() {
        let onsets = array![5.0, 15.0, 25.0, 35.0];
        let blocks = vec![
            BlockAssignment {
                label: "block1".to_string(),
                onset: 0.0,
                duration: 20.0,
            },
            BlockAssignment {
                label: "block2".to_string(),
                onset: 20.0,
                duration: 20.0,
            },
        ];

        let block_indices = split_by_block_timing(&onsets, &blocks);

        assert_eq!(block_indices["block1"], vec![0, 1]);
        assert_eq!(block_indices["block2"], vec![2, 3]);
    }

    #[test]
    fn test_inter_onset_intervals() {
        let onsets = array![0.0, 1.0, 3.0, 6.0];
        let intervals = inter_onset_intervals(&onsets);

        assert_eq!(intervals, array![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_group_consecutive_events() {
        let onsets = array![1.0, 1.5, 2.0, 5.0, 5.5, 10.0];
        let groups = group_consecutive_events(&onsets, 1.0);

        assert_eq!(groups, vec![0, 0, 0, 1, 1, 2]);
    }
}
