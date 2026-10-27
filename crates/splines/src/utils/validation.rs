//! Input validation utilities for spline operations

use crate::errors::{SplineError, SplineResult};
use ndarray::Array1;
use num_traits::Float;

/// Validate that input x values are within the specified range
pub fn validate_x_range<T: Float>(x: &[T], min: T, max: T, _name: &str) -> SplineResult<()> {
    for &val in x {
        if val < min || val > max {
            return Err(SplineError::DataOutOfRange {
                min: min.to_f64().unwrap_or(0.0),
                max: max.to_f64().unwrap_or(0.0),
            });
        }
    }
    Ok(())
}

/// Validate that knots are in non-decreasing order
pub fn validate_knots<T: Float>(knots: &Array1<T>) -> SplineResult<()> {
    if knots.len() < 2 {
        return Err(SplineError::InvalidKnots(
            "Need at least 2 knots".to_string(),
        ));
    }

    for i in 1..knots.len() {
        if knots[i] < knots[i - 1] {
            return Err(SplineError::InvalidKnots(
                "Knots must be in non-decreasing order".to_string(),
            ));
        }
    }
    Ok(())
}

/// Validate that boundary knots are valid
pub fn validate_boundary_knots<T: Float>(boundary_knots: (T, T)) -> SplineResult<()> {
    if boundary_knots.0 >= boundary_knots.1 {
        return Err(SplineError::InvalidBoundaryKnots(
            "Lower boundary must be less than upper boundary".to_string(),
        ));
    }
    Ok(())
}

/// Validate spline degree
pub fn validate_degree(degree: usize) -> SplineResult<()> {
    if degree == 0 {
        return Err(SplineError::InvalidDegree(degree as i32));
    }
    Ok(())
}

/// Validate derivative order against spline order
pub fn validate_derivative_order(deriv: usize, order: usize) -> SplineResult<()> {
    if deriv >= order {
        return Err(SplineError::InvalidDerivative {
            order: deriv as i32,
            max: (order - 1) as i32,
        });
    }
    Ok(())
}

/// Validate that input vectors have compatible lengths
pub fn validate_lengths<T>(x: &[T], y: &[T]) -> SplineResult<()> {
    if x.len() != y.len() {
        return Err(SplineError::dimension_mismatch(x.len(), y.len()));
    }
    Ok(())
}

/// Validate that x values are strictly increasing (for interpolation)
pub fn validate_strictly_increasing<T: Float>(x: &[T]) -> SplineResult<()> {
    if x.len() < 2 {
        return Err(SplineError::insufficient_data(2, x.len()));
    }

    for i in 1..x.len() {
        if x[i] <= x[i - 1] {
            return Err(SplineError::NonMonotonic);
        }
    }
    Ok(())
}

/// Validate minimum number of data points
pub fn validate_min_points<T>(data: &[T], min_required: usize) -> SplineResult<()> {
    if data.len() < min_required {
        return Err(SplineError::insufficient_data(min_required, data.len()));
    }
    Ok(())
}

/// Validate that there are no NA/NaN values in the data
pub fn validate_finite<T: Float>(data: &[T], name: &str) -> SplineResult<()> {
    for (i, &val) in data.iter().enumerate() {
        if !val.is_finite() {
            return Err(SplineError::ComputationError(format!(
                "Non-finite value found in {} at index {}",
                name, i
            )));
        }
    }
    Ok(())
}

/// Check if x values are sorted (needed for efficient evaluation)
pub fn is_sorted<T: Float>(x: &[T]) -> bool {
    x.windows(2).all(|w| w[0] <= w[1])
}

/// Sort x values and return permutation indices
pub fn sort_with_indices<T: Float>(x: &[T]) -> (Vec<T>, Vec<usize>) {
    let mut indexed: Vec<(usize, T)> = x.iter().enumerate().map(|(i, &val)| (i, val)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let sorted_x: Vec<T> = indexed.iter().map(|(_, val)| *val).collect();
    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();

    (sorted_x, indices)
}
