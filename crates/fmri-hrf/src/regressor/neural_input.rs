//! Neural input generation for fMRI regressors.

use crate::error::Result;
use ndarray::Array1;
use std::collections::HashMap;

/// Generate neural input time series from events.
///
/// This function creates a boxcar time series representing neural activity,
/// where each event contributes its amplitude during its duration.
///
/// # Arguments
/// * `onsets` - Event onset times
/// * `durations` - Event durations
/// * `amplitudes` - Event amplitudes
/// * `start` - Start time
/// * `end` - End time
/// * `resolution` - Time resolution
///
/// # Returns
/// Tuple of (time_points, neural_input_values)
pub fn neural_input_core(
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    start: f64,
    end: f64,
    resolution: f64,
) -> Result<(Array1<f64>, Array1<f64>)> {
    if resolution <= 0.0 {
        return Err(crate::error::HrfError::InvalidInput(
            "resolution must be positive".into(),
        ));
    }

    // Create time grid
    let n_points = ((end - start) / resolution) as usize + 1;
    let time = Array1::linspace(start, end, n_points);

    // Use fast algorithm for efficiency
    let neural = neural_input_fast(onsets, durations, amplitudes, start, end, resolution)?;

    Ok((time, neural))
}

/// Fast neural input generation using difference array method.
///
/// This is an optimized O(E + N) algorithm where E is number of events
/// and N is number of time points.
fn neural_input_fast(
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    start: f64,
    end: f64,
    resolution: f64,
) -> Result<Array1<f64>> {
    // Number of time bins
    let n_bins = ((end - start) / resolution) as usize + 1;

    // Initialize difference array with extra guard slot
    let mut diff = vec![0.0; n_bins + 1];

    // Process each event
    for ((onset, duration), amplitude) in onsets.iter().zip(durations.iter()).zip(amplitudes.iter())
    {
        // Calculate start bin index
        let start_idx = if *onset <= start {
            0
        } else {
            ((onset - start) / resolution) as usize
        };

        // Skip if event starts after our time range
        if start_idx >= n_bins {
            continue;
        }

        // Calculate end bin index
        let end_idx = if *duration > 0.0 {
            ((onset + duration - start) / resolution) as usize
        } else {
            // Impulse event
            start_idx
        };

        let end_idx = end_idx.min(n_bins - 1);

        // Skip if invalid range
        if start_idx > end_idx {
            continue;
        }

        // Add amplitude at start, subtract at end+1
        diff[start_idx] += amplitude;
        if end_idx + 1 < diff.len() {
            diff[end_idx + 1] -= amplitude;
        }
    }

    // Cumulative sum to get actual values (exclude guard slot)
    let mut neural = Array1::zeros(n_bins);
    let mut cumsum = 0.0;
    for i in 0..n_bins {
        cumsum += diff[i];
        neural[i] = cumsum;
    }

    Ok(neural)
}

/// Generate neural input time series from a Regressor.
///
/// # Arguments
/// * `regressor` - Regressor object
/// * `start` - Start time in seconds
/// * `end` - End time in seconds (if None, auto-determined)
/// * `resolution` - Time resolution in seconds
///
/// # Returns
/// HashMap with 'time' and 'neural_input' arrays
pub fn neural_input(
    regressor: &crate::regressor::core::Regressor,
    start: Option<f64>,
    end: Option<f64>,
    resolution: Option<f64>,
) -> Result<HashMap<String, Array1<f64>>> {
    let start = start.unwrap_or(0.0);
    let resolution = resolution.unwrap_or(0.33);

    let end = end.unwrap_or_else(|| {
        if regressor.onsets.is_empty() {
            start + regressor.span
        } else {
            let max_end: f64 = regressor
                .onsets
                .iter()
                .zip(regressor.durations.iter())
                .map(|(onset, duration)| onset + duration)
                .fold(f64::NEG_INFINITY, f64::max);
            max_end + 10.0
        }
    });

    let (time, neural) = neural_input_core(
        &regressor.onsets,
        &regressor.durations,
        &regressor.amplitudes,
        start,
        end,
        resolution,
    )?;

    let mut result = HashMap::new();
    result.insert("time".to_string(), time);
    result.insert("neural_input".to_string(), neural);

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_neural_input_impulse() {
        let onsets = vec![1.0, 2.0, 3.0];
        let durations = vec![0.0, 0.0, 0.0];
        let amplitudes = vec![1.0, 1.0, 1.0];

        let (_time, neural) =
            neural_input_core(&onsets, &durations, &amplitudes, 0.0, 5.0, 0.1).unwrap();

        // Check that we have spikes at the right times
        let spike_indices: Vec<usize> = neural
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.0)
            .map(|(idx, _)| idx)
            .collect();

        // Should have 3 spikes
        assert!(spike_indices.len() >= 3);
    }

    #[test]
    fn test_neural_input_blocks() {
        let onsets = vec![1.0, 3.0];
        let durations = vec![1.0, 1.0];
        let amplitudes = vec![1.0, 2.0];

        let (_time, neural) =
            neural_input_core(&onsets, &durations, &amplitudes, 0.0, 6.0, 0.1).unwrap();

        // Check values during blocks
        // First block (t=1 to t=2, amplitude=1)
        let idx_1_5 = ((1.5 / 0.1) as usize).min(neural.len() - 1);
        assert_relative_eq!(neural[idx_1_5], 1.0, epsilon = 0.1);

        // Second block (t=3 to t=4, amplitude=2)
        let idx_3_5 = ((3.5 / 0.1) as usize).min(neural.len() - 1);
        assert_relative_eq!(neural[idx_3_5], 2.0, epsilon = 0.1);

        // Outside blocks
        assert_eq!(neural[0], 0.0);
        assert_eq!(neural[neural.len() - 1], 0.0);
    }

    #[test]
    fn test_neural_input_overlapping() {
        let onsets = vec![1.0, 1.5];
        let durations = vec![1.0, 1.0];
        let amplitudes = vec![1.0, 1.0];

        let (_time, neural) =
            neural_input_core(&onsets, &durations, &amplitudes, 0.0, 4.0, 0.1).unwrap();

        // During overlap (t=1.5 to t=2), should sum to 2
        let idx_1_7 = ((1.7 / 0.1) as usize).min(neural.len() - 1);
        assert_relative_eq!(neural[idx_1_7], 2.0, epsilon = 0.1);
    }
}
