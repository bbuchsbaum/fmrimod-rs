//! Knot selection and manipulation utilities

use crate::errors::{SplineError, SplineResult};
use crate::utils::validation::{validate_finite, validate_min_points};
use ndarray::Array1;
use num_traits::Float;

/// Select knots based on degrees of freedom
pub fn select_knots_by_df<T: Float>(
    x: &[T],
    df: usize,
    degree: usize,
    intercept: bool,
    boundary_knots: Option<(T, T)>,
) -> SplineResult<Array1<T>> {
    validate_finite(x, "x")?;
    validate_min_points(x, 2)?;

    // Calculate number of interior knots needed
    let n_interior = if df > degree + (1 - intercept as usize) {
        df - degree - (1 - intercept as usize)
    } else {
        0
    };

    if n_interior == 0 {
        return Ok(Array1::from(vec![]));
    }

    // Determine the data range for quantile calculation
    let (min_x, max_x) = if let Some((min_bound, max_bound)) = boundary_knots {
        (min_bound, max_bound)
    } else {
        let mut sorted_x = x.to_vec();
        sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (sorted_x[0], sorted_x[sorted_x.len() - 1])
    };

    // Filter x values to be within boundaries for quantile calculation
    let filtered_x: Vec<T> = x
        .iter()
        .filter(|&&val| val >= min_x && val <= max_x)
        .cloned()
        .collect();

    if filtered_x.is_empty() {
        return Err(SplineError::ComputationError(
            "No data points within boundary knots".to_string(),
        ));
    }

    // Calculate quantiles for knot placement
    let quantiles = calculate_quantiles(&filtered_x, n_interior + 2)?;

    // Remove the first and last quantiles (these are the boundaries)
    let interior_knots = quantiles[1..quantiles.len() - 1].to_vec();

    Ok(Array1::from(interior_knots))
}

/// Calculate quantiles for knot placement
pub fn calculate_quantiles<T: Float>(x: &[T], n_quantiles: usize) -> SplineResult<Vec<T>> {
    if n_quantiles < 2 {
        return Err(SplineError::ComputationError(
            "Need at least 2 quantiles".to_string(),
        ));
    }

    let mut sorted_x = x.to_vec();
    sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut quantiles = Vec::with_capacity(n_quantiles);

    for i in 0..n_quantiles {
        let p = T::from(i).unwrap() / T::from(n_quantiles - 1).unwrap();
        let quantile = compute_quantile(&sorted_x, p)?;
        quantiles.push(quantile);
    }

    Ok(quantiles)
}

/// Compute a specific quantile value
fn compute_quantile<T: Float>(sorted_x: &[T], p: T) -> SplineResult<T> {
    if sorted_x.is_empty() {
        return Err(SplineError::ComputationError("Empty data".to_string()));
    }

    if p < T::zero() || p > T::one() {
        return Err(SplineError::ComputationError(
            "Quantile probability must be between 0 and 1".to_string(),
        ));
    }

    let n = sorted_x.len();

    if p == T::zero() {
        return Ok(sorted_x[0]);
    }
    if p == T::one() {
        return Ok(sorted_x[n - 1]);
    }

    // Linear interpolation for quantiles
    let h = T::from(n - 1).unwrap() * p;
    let h_floor = h.floor();
    let h_int = h_floor.to_usize().unwrap().min(n - 2);

    let x0 = sorted_x[h_int];
    let x1 = sorted_x[h_int + 1];
    let fraction = h - h_floor;

    Ok(x0 + fraction * (x1 - x0))
}

/// Create augmented knot vector (with multiplicities at boundaries)
pub fn create_augmented_knots<T: Float>(
    interior_knots: &Array1<T>,
    boundary_knots: (T, T),
    order: usize,
) -> SplineResult<Array1<T>> {
    let mut knots = Vec::new();

    // Add repeated boundary knots at the start
    for _ in 0..order {
        knots.push(boundary_knots.0);
    }

    // Add interior knots
    knots.extend_from_slice(interior_knots.as_slice().unwrap());

    // Add repeated boundary knots at the end
    for _ in 0..order {
        knots.push(boundary_knots.1);
    }

    Ok(Array1::from(knots))
}

/// Compute boundary knots from data range
pub fn compute_boundary_knots<T: Float>(x: &[T]) -> SplineResult<(T, T)> {
    if x.is_empty() {
        return Err(SplineError::insufficient_data(1, 0));
    }

    let min_x = x
        .iter()
        .fold(x[0], |acc, &val| if val < acc { val } else { acc });
    let max_x = x
        .iter()
        .fold(x[0], |acc, &val| if val > acc { val } else { acc });

    if min_x == max_x {
        return Err(SplineError::ComputationError(
            "All x values are identical".to_string(),
        ));
    }

    Ok((min_x, max_x))
}

/// Find the interval index for a given x value in the knot vector
pub fn find_knot_interval<T: Float>(knots: &Array1<T>, x: T) -> usize {
    // Binary search for the interval
    let knot_slice = knots.as_slice().unwrap();

    match knot_slice.binary_search_by(|&knot| knot.partial_cmp(&x).unwrap()) {
        Ok(idx) => idx.saturating_sub(1), // Exact match - use previous interval
        Err(idx) => idx.saturating_sub(1), // Insertion point - use previous interval
    }
}

/// Remove duplicate knots (within tolerance)
pub fn remove_duplicate_knots<T: Float>(knots: &Array1<T>, tolerance: T) -> Array1<T> {
    if knots.is_empty() {
        return Array1::from(vec![]);
    }

    let mut unique_knots = vec![knots[0]];

    for &knot in knots.iter().skip(1) {
        let last_knot = unique_knots[unique_knots.len() - 1];
        if (knot - last_knot).abs() > tolerance {
            unique_knots.push(knot);
        }
    }

    Array1::from(unique_knots)
}

/// Validate knot configuration for specific spline requirements
pub fn validate_knot_configuration<T: Float>(
    interior_knots: &Array1<T>,
    boundary_knots: (T, T),
    _degree: usize,
) -> SplineResult<()> {
    // Check boundary knots
    if boundary_knots.0 >= boundary_knots.1 {
        return Err(SplineError::InvalidBoundaryKnots(
            "Lower boundary must be less than upper boundary".to_string(),
        ));
    }

    // Check interior knots are within boundaries
    for &knot in interior_knots.iter() {
        if knot <= boundary_knots.0 || knot >= boundary_knots.1 {
            return Err(SplineError::InvalidKnots(
                "Interior knots must be within boundary knots".to_string(),
            ));
        }
    }

    // Check interior knots are sorted
    for i in 1..interior_knots.len() {
        if interior_knots[i] < interior_knots[i - 1] {
            return Err(SplineError::InvalidKnots(
                "Interior knots must be in non-decreasing order".to_string(),
            ));
        }
    }

    Ok(())
}
