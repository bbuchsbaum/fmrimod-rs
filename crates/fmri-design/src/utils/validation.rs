//! Validation utilities for event data.

use crate::types::{DurationArray, OnsetArray, ValidationError};
use ndarray::Array1;

/// Validate onset times.
///
/// Ensures onsets are non-negative and finite.
pub fn validate_onsets<T: IntoIterator<Item = f64>>(
    onsets: T,
) -> Result<OnsetArray, ValidationError> {
    let array: Array1<f64> = onsets.into_iter().collect();

    if array.is_empty() {
        return Err(ValidationError::EmptyArray);
    }

    for &val in array.iter() {
        if !val.is_finite() {
            return Err(ValidationError::NonFiniteValues);
        }
        if val < 0.0 {
            return Err(ValidationError::NegativeValues);
        }
    }

    Ok(array)
}

/// Validate duration values.
///
/// Ensures durations are non-negative and finite.
pub fn validate_durations<T: IntoIterator<Item = f64>>(
    durations: T,
) -> Result<DurationArray, ValidationError> {
    let array: Array1<f64> = durations.into_iter().collect();

    if array.is_empty() {
        return Err(ValidationError::EmptyArray);
    }

    for &val in array.iter() {
        if !val.is_finite() {
            return Err(ValidationError::NonFiniteValues);
        }
        if val < 0.0 {
            return Err(ValidationError::NegativeValues);
        }
    }

    Ok(array)
}

/// Create uniform durations.
///
/// Creates an array of the same duration value repeated n times.
pub fn uniform_durations(duration: f64, n: usize) -> Result<DurationArray, ValidationError> {
    if n == 0 {
        return Err(ValidationError::EmptyArray);
    }

    if !duration.is_finite() {
        return Err(ValidationError::NonFiniteValues);
    }

    if duration < 0.0 {
        return Err(ValidationError::NegativeValues);
    }

    Ok(Array1::from_elem(n, duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_onsets() {
        // Valid onsets
        let onsets = validate_onsets(vec![0.0, 1.0, 2.0]).unwrap();
        assert_eq!(onsets.len(), 3);

        // Empty array
        assert!(validate_onsets(vec![]).is_err());

        // Negative values
        assert!(validate_onsets(vec![0.0, -1.0, 2.0]).is_err());

        // Non-finite values
        assert!(validate_onsets(vec![0.0, f64::INFINITY, 2.0]).is_err());
    }

    #[test]
    fn test_uniform_durations() {
        let durations = uniform_durations(2.0, 5).unwrap();
        assert_eq!(durations.len(), 5);
        assert!(durations.iter().all(|&x| x == 2.0));

        // Zero length
        assert!(uniform_durations(2.0, 0).is_err());

        // Negative duration
        assert!(uniform_durations(-1.0, 5).is_err());
    }
}
