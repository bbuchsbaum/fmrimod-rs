//! Single-trial regressor functionality for fMRI design matrices.
//!
//! This module provides the trialwise() function which creates one regressor
//! per trial in the event model, optionally with a grand-mean column.

use crate::events::EventFactor;
use crate::formula::Term;
use crate::types::{DurationArray, OnsetArray, ValidationError};
use ndarray::{Array1, Array2};

/// Configuration for trialwise regressors
#[derive(Debug, Clone)]
pub struct TrialwiseConfig {
    /// The hemodynamic response function or name
    pub basis: String,
    /// Temporal offset in seconds
    pub lag: f64,
    /// Number of basis functions
    pub nbasis: usize,
    /// Whether to add a grand-mean column
    pub add_sum: bool,
    /// Term label/prefix for columns
    pub label: String,
}

impl Default for TrialwiseConfig {
    fn default() -> Self {
        Self {
            basis: "spmg1".to_string(),
            lag: 0.0,
            nbasis: 1,
            add_sum: false,
            label: "trial".to_string(),
        }
    }
}

/// Generate one regressor per trial (plus an optional grand-mean column).
///
/// This creates individual regressors for each trial/event in the data,
/// allowing for single-trial analysis. Each trial gets its own column
/// in the design matrix.
///
/// # Examples
///
/// ```ignore
/// // Basic single-trial model
/// let model = event_model("hrf(trialwise())", &data, tr, n_scans)?;
///
/// // Single-trial with grand mean
/// let model = event_model("hrf(trialwise(add_sum=true))", &data, tr, n_scans)?;
/// ```
pub fn trialwise() -> Term {
    trialwise_with_config(TrialwiseConfig::default())
}

/// Generate trialwise regressors with custom configuration
pub fn trialwise_with_config(config: TrialwiseConfig) -> Term {
    // Create a special term that will be expanded during model construction
    let mut term = Term::new("trialwise")
        .with_hrf(config.basis.clone())
        .with_name(config.label.clone());

    // Add special parameters for trialwise processing
    term = term
        .set_param("_is_trialwise", 1.0)
        .set_param("_add_sum", if config.add_sum { 1.0 } else { 0.0 })
        .set_param("_lag", config.lag)
        .set_param("_nbasis", config.nbasis as f64)
        .set_param("nbasis", config.nbasis as f64)
        .set_param("basis", config.basis.as_str());

    // Also store lag as a regular parameter for tests
    term = term.set_param("lag", config.lag);

    // Store the label as a string parameter (encoded as numeric for now)
    // In practice, we'll handle this differently
    term
}

/// Create an EventFactor with unique levels for each trial
pub fn create_trial_factor(
    n_trials: usize,
    onsets: Option<&OnsetArray>,
    durations: Option<&DurationArray>,
    label: &str,
) -> Result<EventFactor, ValidationError> {
    // Determine padding for nice formatting
    let pad = n_trials.to_string().len();

    // Create trial labels with zero-padding
    let trial_labels: Vec<String> = (0..n_trials)
        .map(|i| format!("{:0width$}", i + 1, width = pad))
        .collect();

    // Use provided onsets or create default
    let trial_onsets = if let Some(onsets) = onsets {
        onsets.clone()
    } else {
        Array1::range(0.0, n_trials as f64, 1.0)
    };

    // Use provided durations or default to 0
    let trial_durations = if let Some(durations) = durations {
        durations.clone()
    } else {
        Array1::zeros(n_trials)
    };

    // Create EventFactor with unique level for each trial
    let trial_values: Vec<&str> = trial_labels.iter().map(|s| s.as_str()).collect();
    let trial_onsets_vec: Vec<f64> = trial_onsets.to_vec();
    let trial_durations_vec: Vec<f64> = trial_durations.to_vec();

    EventFactor::new(
        label.to_string(),
        trial_onsets_vec,
        trial_values,
        trial_durations_vec,
    )
}

/// Check if a term is a trialwise term
pub fn is_trialwise_term(term: &Term) -> bool {
    term.get_param("_is_trialwise")
        .map(|v| v > 0.5)
        .unwrap_or(false)
}

/// Get trialwise configuration from a term
pub fn get_trialwise_config(term: &Term) -> Option<TrialwiseConfig> {
    if !is_trialwise_term(term) {
        return None;
    }

    Some(TrialwiseConfig {
        basis: term
            .hrf
            .as_ref()
            .and_then(|h| {
                if let crate::formula::HrfSpec::Named(name) = h {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| "spmg1".to_string()),
        lag: term.get_param("_lag").unwrap_or(0.0),
        nbasis: term.get_param("_nbasis").map(|v| v as usize).unwrap_or(1),
        add_sum: term.get_param("_add_sum").map(|v| v > 0.5).unwrap_or(false),
        label: term.name.clone().unwrap_or_else(|| "trial".to_string()),
    })
}

/// Process trialwise columns by optionally adding a grand mean
pub fn process_trialwise_columns(
    columns: Array2<f64>,
    add_sum: bool,
) -> (Array2<f64>, Vec<String>) {
    let n_trials = columns.ncols();
    let label = "trial"; // Will be passed in from config
    let pad = n_trials.to_string().len();

    // Generate column names
    let mut col_names: Vec<String> = (0..n_trials)
        .map(|i| format!("{}_{:0width$}", label, i + 1, width = pad))
        .collect();

    if add_sum {
        // Calculate grand mean across trials
        let mean_col = columns
            .mean_axis(ndarray::Axis(1))
            .unwrap()
            .insert_axis(ndarray::Axis(1));

        // Concatenate original columns with mean
        let extended =
            ndarray::concatenate(ndarray::Axis(1), &[columns.view(), mean_col.view()]).unwrap();

        col_names.push(format!("{}_mean", label));

        (extended, col_names)
    } else {
        (columns, col_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_create_trial_factor() {
        let onsets = array![0.0, 2.0, 4.0, 6.0];
        let durations = array![1.0, 1.0, 1.0, 1.0];

        let factor = create_trial_factor(4, Some(&onsets), Some(&durations), "trial").unwrap();

        assert_eq!(factor.name, "trial");
        assert_eq!(factor.levels.len(), 4);
        assert_eq!(factor.levels[0], "1");
        assert_eq!(factor.levels[3], "4");
    }

    #[test]
    fn test_trial_padding() {
        // Test padding for different numbers of trials
        let factor = create_trial_factor(100, None, None, "trial").unwrap();
        assert_eq!(factor.levels[0], "001");
        assert_eq!(factor.levels[99], "100");
    }

    #[test]
    fn test_process_trialwise_columns() {
        let cols = Array2::from_shape_fn((10, 3), |(i, j)| (i + j) as f64);

        // Without grand mean
        let (result, names) = process_trialwise_columns(cols.clone(), false);
        assert_eq!(result.shape(), &[10, 3]);
        assert_eq!(names.len(), 3);
        assert_eq!(names[0], "trial_1");

        // With grand mean
        let (result, names) = process_trialwise_columns(cols, true);
        assert_eq!(result.shape(), &[10, 4]);
        assert_eq!(names.len(), 4);
        assert_eq!(names[3], "trial_mean");
    }
}
