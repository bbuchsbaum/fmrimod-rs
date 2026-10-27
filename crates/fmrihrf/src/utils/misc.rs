//! Miscellaneous utility functions.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use crate::regressor::core::Regressor;
use crate::sampling::SamplingFrame;
use crate::types::TimeArray;
use ndarray::{Array1, Array2};

/// Recycle scalar to array or validate array length.
///
/// # Arguments
/// * `x` - Scalar or array values
/// * `n` - Expected length
/// * `name` - Parameter name for error messages
///
/// # Returns
/// Array of length n
///
/// # Errors
/// Returns error if array length doesn't match n
pub fn recycle_or_error(x: &[f64], n: usize, name: &str) -> Result<Vec<f64>> {
    match x.len() {
        0 => Err(HrfError::InvalidInput(format!("{} cannot be empty", name))),
        1 => Ok(vec![x[0]; n]),
        len if len == n => Ok(x.to_vec()),
        len => Err(HrfError::InvalidInput(format!(
            "`{}` must have length 1 or {}, not {}",
            name, n, len
        ))),
    }
}

/// Create Toeplitz matrix for HRF convolution.
///
/// A Toeplitz matrix has constant diagonals and is useful for
/// implementing convolution as matrix multiplication.
///
/// # Arguments
/// * `hrf` - HRF to create matrix for
/// * `time` - Time points for HRF evaluation
/// * `length` - Length of output matrix
///
/// # Returns
/// Toeplitz matrix for convolution
pub fn hrf_toeplitz(hrf: &dyn Hrf, time: &TimeArray, length: usize) -> Result<Array2<f64>> {
    // Evaluate HRF
    let hrf_values = hrf.evaluate(time)?;

    // Extract first basis function if multi-basis
    let hrf_vec = hrf_values.column(0).to_vec();

    // Create Toeplitz matrix
    let mut matrix = Array2::zeros((length, length));

    for i in 0..length {
        for j in 0..length {
            if i >= j && (i - j) < hrf_vec.len() {
                matrix[[i, j]] = hrf_vec[i - j];
            }
        }
    }

    Ok(matrix)
}

/// Create a single trial regressor helper.
///
/// This is a convenience function for creating a regressor with a single event.
/// Note: This returns the neural input array, not a full Regressor object.
///
/// # Arguments
/// * `onset` - Event onset in seconds
/// * `duration` - Event duration
/// * `amplitude` - Event amplitude
/// * `n_scans` - Number of scans
/// * `tr` - Repetition time
///
/// # Returns
/// Neural input array
pub fn single_trial_regressor(
    onset: f64,
    duration: f64,
    amplitude: f64,
    n_scans: usize,
    tr: f64,
) -> Array1<f64> {
    let mut neural = Array1::zeros(n_scans);

    // Calculate which scans are affected
    let start_scan = (onset / tr).floor() as usize;
    let end_scan = ((onset + duration) / tr).ceil() as usize;

    for i in start_scan..end_scan.min(n_scans) {
        neural[i] = amplitude;
    }

    neural
}

/// Get fMRI acquisition onset times.
///
/// Calculate the onset time in seconds for each fMRI volume
/// acquisition from the start of the experiment.
///
/// # Arguments
/// * `sframe` - SamplingFrame object
///
/// # Returns
/// Array of acquisition onset times in seconds
pub fn acquisition_onsets(sframe: &SamplingFrame) -> Vec<f64> {
    sframe.samples()
}

/// Apply a function element-wise to arrays, recycling shorter arrays.
///
/// # Arguments
/// * `a` - First array
/// * `b` - Second array  
/// * `f` - Function to apply
///
/// # Returns
/// Result array with length equal to the longer input
pub fn apply_binary_op<F>(a: &[f64], b: &[f64], f: F) -> Vec<f64>
where
    F: Fn(f64, f64) -> f64,
{
    let len = a.len().max(b.len());
    let mut result = Vec::with_capacity(len);

    for i in 0..len {
        let a_val = a[i % a.len()];
        let b_val = b[i % b.len()];
        result.push(f(a_val, b_val));
    }

    result
}

/// Check if a value is approximately zero.
pub fn is_zero(x: f64, epsilon: f64) -> bool {
    x.abs() < epsilon
}

/// Linearly interpolate between two arrays.
///
/// # Arguments
/// * `a` - First array (weight = 0)
/// * `b` - Second array (weight = 1)
/// * `weight` - Interpolation weight (0 to 1)
///
/// # Returns
/// Interpolated array
pub fn lerp_arrays(a: &Array1<f64>, b: &Array1<f64>, weight: f64) -> Result<Array1<f64>> {
    if a.len() != b.len() {
        return Err(HrfError::InvalidInput(
            "Arrays must have same length for interpolation".into(),
        ));
    }

    Ok(a * (1.0 - weight) + b * weight)
}

/// Create a regular time grid.
///
/// # Arguments
/// * `start` - Start time
/// * `end` - End time  
/// * `dt` - Time step
///
/// # Returns
/// Array of time points
pub fn time_grid(start: f64, end: f64, dt: f64) -> Array1<f64> {
    let n = ((end - start) / dt).ceil() as usize + 1;
    Array1::linspace(start, start + (n - 1) as f64 * dt, n)
}

/// Find peaks in a 1D array.
///
/// A peak is defined as a point that is greater than its neighbors.
///
/// # Arguments
/// * `data` - Input array
///
/// # Returns
/// Indices of peaks
pub fn find_peaks(data: &Array1<f64>) -> Vec<usize> {
    let mut peaks = Vec::new();
    let n = data.len();

    if n < 3 {
        return peaks;
    }

    for i in 1..n - 1 {
        if data[i] > data[i - 1] && data[i] > data[i + 1] {
            peaks.push(i);
        }
    }

    peaks
}

/// Normalize array to unit maximum.
///
/// # Arguments
/// * `data` - Input array
///
/// # Returns
/// Normalized array
pub fn normalize_max(data: &Array1<f64>) -> Array1<f64> {
    let max_val = data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x.abs()));

    if max_val > 0.0 {
        data / max_val
    } else {
        data.clone()
    }
}

/// Shift regressor onsets by a temporal offset.
///
/// This is a convenience function that creates a new regressor with
/// shifted event onsets.
///
/// # Arguments
/// * `reg` - The regressor to shift
/// * `shift_amount` - Amount to shift in seconds (positive = forward)
///
/// # Returns
/// New regressor with shifted onsets
pub fn shift_regressor(reg: &Regressor, shift_amount: f64) -> Regressor {
    reg.shift(shift_amount)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recycle_or_error() {
        // Test scalar recycling
        let result = recycle_or_error(&[1.0], 5, "test").unwrap();
        assert_eq!(result, vec![1.0; 5]);

        // Test exact length
        let result = recycle_or_error(&[1.0, 2.0, 3.0], 3, "test").unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);

        // Test error
        assert!(recycle_or_error(&[1.0, 2.0], 3, "test").is_err());
    }

    #[test]
    fn test_apply_binary_op() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0];

        let result = apply_binary_op(&a, &b, |x, y| x + y);
        assert_eq!(result, vec![11.0, 12.0, 13.0]);
    }

    #[test]
    fn test_time_grid() {
        let grid = time_grid(0.0, 2.0, 0.5);
        assert_eq!(grid.len(), 5);
        assert_eq!(grid[0], 0.0);
        assert_eq!(grid[4], 2.0);
    }

    #[test]
    fn test_find_peaks() {
        let data = Array1::from_vec(vec![1.0, 3.0, 2.0, 5.0, 4.0]);
        let peaks = find_peaks(&data);
        assert_eq!(peaks, vec![1, 3]);
    }

    #[test]
    fn test_normalize_max() {
        let data = Array1::from_vec(vec![1.0, -2.0, 3.0]);
        let normalized = normalize_max(&data);
        assert_eq!(normalized[2], 1.0);
        assert_eq!(normalized[1], -2.0 / 3.0);
    }

    #[test]
    fn test_shift_regressor() {
        use crate::hrf::functions::spm_canonical;
        use crate::regressor::core::regressor;

        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![10.0, 20.0, 30.0], Some(hrf), None, None, None, None).unwrap();

        let shifted = shift_regressor(&reg, 5.0);
        assert_eq!(shifted.onsets[0], 15.0);
        assert_eq!(shifted.onsets[1], 25.0);
        assert_eq!(shifted.onsets[2], 35.0);
    }
}
