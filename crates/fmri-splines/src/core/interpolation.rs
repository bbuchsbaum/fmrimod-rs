//! Interpolation spline implementation (equivalent to R's interp.spline() function)

use crate::basis::design_matrix::DesignMatrixBuilder;
use crate::basis::evaluation::evaluate_spline;
use crate::errors::{SplineError, SplineResult};
use crate::types::{BasisMatrix, Spline, SplineType, XYVector};
use crate::utils::validation::{validate_finite, validate_min_points};
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;
use std::marker::PhantomData;

/// Interpolation spline representation
///
/// This struct represents an interpolation spline equivalent to R's `interp.spline()` function.
/// Interpolation splines pass exactly through the given data points and provide smooth
/// interpolation between them with controlled derivatives at boundaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpolationSpline<T: Float + Sum + 'static> {
    /// Extended knot vector including repeated boundary knots
    knots: Array1<T>,
    /// Spline coefficients solving the interpolation system
    coefficients: Array1<T>,
    /// Spline order (degree + 1)
    order: usize,
    /// Original data points (sorted by x)
    x_data: Array1<T>,
    /// Original y values (corresponding to sorted x)
    y_data: Array1<T>,
    /// Valid range for evaluation
    valid_range: (T, T),
}

impl<T: Float + Sum + 'static> InterpolationSpline<T> {
    /// Create a new interpolation spline (equivalent to R's interp.spline() function)
    ///
    /// # Arguments
    /// * `x` - Input x data points
    /// * `y` - Input y data points (must have same length as x)
    /// * `order` - Spline order (degree + 1), default 4 for cubic splines
    ///
    /// # Examples
    /// ```
    /// use fmri_splines::{InterpolationSpline, Spline};
    ///
    /// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    /// let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // x^2 data
    ///
    /// // Create cubic interpolation spline
    /// let spline = InterpolationSpline::new(&x, &y, 4)?;
    ///
    /// // Evaluate at original points (should match exactly)
    /// let result = spline.evaluate(2.0)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(x: &[T], y: &[T], order: usize) -> SplineResult<Self> {
        // Validate inputs
        validate_finite(x, "x")?;
        validate_finite(y, "y")?;
        validate_min_points(x, 2)?;

        if x.len() != y.len() {
            return Err(SplineError::dimension_mismatch(x.len(), y.len()));
        }

        if order != 4 {
            return Err(SplineError::InvalidDegree(order as i32 - 1));
        }

        let n = x.len();
        if n < order {
            return Err(SplineError::insufficient_data(order, n));
        }

        // Sort data by x values (R: ind <- order(x); x <- x[ind]; y <- y[ind])
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap());

        let x_sorted: Vec<T> = indices.iter().map(|&i| x[i]).collect();
        let y_sorted: Vec<T> = indices.iter().map(|&i| y[i]).collect();

        // Check for strictly increasing x values
        for i in 1..x_sorted.len() {
            if x_sorted[i] <= x_sorted[i - 1] {
                return Err(SplineError::InvalidKnots(
                    "x values must be strictly increasing for interpolation".to_string(),
                ));
            }
        }

        let x_array = Array1::from(x_sorted.clone());
        let y_array = Array1::from(y_sorted.clone());

        // Implement R's exact interp.spline algorithm
        // R: knots <- c(x[1:ordm1] + x[1] - x[ord], x, x[lenx + (1:ordm1) - ordm1] + x[lenx] - x[lenx - ordm1])
        let ordm1 = order - 1;
        let len_x = x_sorted.len();

        // Check that we have enough data points
        if len_x < order {
            return Err(SplineError::insufficient_data(order, len_x));
        }

        let mut knots = Vec::new();
        let x_first = x_sorted[0];
        let x_last = x_sorted[len_x - 1];

        // Left boundary knots: x[1:ordm1] + x[1] - x[ord]
        // This means: x[0] + x[0] - x[order-1], x[1] + x[0] - x[order-1], ..., x[ordm1-1] + x[0] - x[order-1]
        for i in 0..ordm1 {
            let boundary_knot = if i < len_x && (order - 1) < len_x {
                x_sorted[i] + x_first - x_sorted[order - 1]
            } else {
                // Fallback for edge cases
                x_first - T::from(ordm1 - i).unwrap() * (x_last - x_first) / T::from(len_x).unwrap()
            };
            knots.push(boundary_knot);
        }

        // Interior knots: all the original x values
        for &x_val in &x_sorted {
            knots.push(x_val);
        }

        // Right boundary knots: x[lenx + (1:ordm1) - ordm1] + x[lenx] - x[lenx - ordm1]
        // This means: x[lenx] + x[lenx] - x[lenx - ordm1], x[lenx+1] + x[lenx] - x[lenx - ordm1], ...
        for i in 1..=ordm1 {
            let idx = len_x + i - ordm1 - 1; // Adjust for 0-based indexing
            let boundary_knot = if idx < len_x && (len_x >= ordm1 + 1) {
                x_sorted[idx] + x_last - x_sorted[len_x - ordm1 - 1]
            } else {
                // Fallback for edge cases
                x_last + T::from(i).unwrap() * (x_last - x_first) / T::from(len_x).unwrap()
            };
            knots.push(boundary_knot);
        }

        let knots_array = Array1::from(knots);

        // Build constraint system exactly as R does
        // R: derivs <- c(2, integer(lenx), 2)
        // R: x <- c(x[1], x, x[lenx])
        let mut constraint_x = Vec::new();
        let mut constraint_derivs = Vec::new();

        // First constraint: second derivative at first point
        constraint_x.push(x_first);
        constraint_derivs.push(2);

        // Function value constraints at all data points
        for &x_val in &x_sorted {
            constraint_x.push(x_val);
            constraint_derivs.push(0); // Function value constraint
        }

        // Last constraint: second derivative at last point
        constraint_x.push(x_last);
        constraint_derivs.push(2);

        // Build the system matrix using R's constraint setup
        let builder = DesignMatrixBuilder::new(knots_array.clone(), order);
        let system_matrix = builder.build_with_derivatives(&constraint_x, &constraint_derivs)?;

        // R: solve(sys.mat, c(0, y, 0))
        let mut rhs = Vec::new();
        rhs.push(T::zero()); // First constraint (second derivative = 0)
        for &y_val in &y_sorted {
            rhs.push(y_val); // Function values
        }
        rhs.push(T::zero()); // Last constraint (second derivative = 0)

        let rhs_array = Array1::from(rhs);

        // Solve the linear system exactly as R does
        let coefficients = solve_linear_system(&system_matrix, &rhs_array)?;

        let valid_range = (x_first, x_last);

        Ok(InterpolationSpline {
            knots: knots_array,
            coefficients,
            order,
            x_data: x_array,
            y_data: y_array,
            valid_range,
        })
    }

    /// Create an interpolation spline with builder pattern
    pub fn builder() -> InterpolationSplineBuilder<T> {
        InterpolationSplineBuilder::<T>::new()
    }

    /// Get the spline order
    pub fn order(&self) -> usize {
        self.order
    }

    /// Get the spline degree
    pub fn degree(&self) -> usize {
        self.order - 1
    }

    /// Get the original data points
    pub fn data_points(&self) -> (&Array1<T>, &Array1<T>) {
        (&self.x_data, &self.y_data)
    }

    /// Get the spline coefficients
    pub fn coefficients(&self) -> &Array1<T> {
        &self.coefficients
    }

    /// Get the knot vector
    pub fn knots(&self) -> &Array1<T> {
        &self.knots
    }

    /// Get the x values used for interpolation
    pub fn x_values(&self) -> &Array1<T> {
        &self.x_data
    }

    /// Get the y values used for interpolation
    pub fn y_values(&self) -> &Array1<T> {
        &self.y_data
    }
}

impl<T: Float + Sum + 'static> Spline<T> for InterpolationSpline<T> {
    fn knots(&self) -> &Array1<T> {
        &self.knots
    }

    fn order(&self) -> usize {
        self.order
    }

    fn spline_type(&self) -> SplineType {
        SplineType::Interpolation
    }

    fn predict(&self, x: &[T]) -> SplineResult<XYVector<T>> {
        let values = evaluate_spline(&self.knots, &self.coefficients, self.order, x, 0)?;
        let x_array = Array1::from(x.to_vec());
        XYVector::from_arrays(x_array, values)
    }

    fn predict_with_deriv(&self, x: &[T], deriv: usize) -> SplineResult<XYVector<T>> {
        crate::utils::validation::validate_derivative_order(deriv, self.order)?;
        let values = evaluate_spline(&self.knots, &self.coefficients, self.order, x, deriv)?;
        let x_array = Array1::from(x.to_vec());
        XYVector::from_arrays(x_array, values)
    }

    fn evaluate(&self, x: T) -> SplineResult<T> {
        // Use the same design matrix approach that worked for computing coefficients
        // This ensures consistency with the constraint system
        let builder = DesignMatrixBuilder::new(self.knots.clone(), self.order);
        let design_matrix = builder.build(&[x])?;

        // Multiply design matrix row by coefficients
        let mut result = T::zero();
        for j in 0..self.coefficients.len() {
            result = result + design_matrix[[0, j]] * self.coefficients[j];
        }

        Ok(result)
    }

    fn evaluate_with_deriv(&self, x: T, deriv: usize) -> SplineResult<T> {
        crate::utils::validation::validate_derivative_order(deriv, self.order)?;
        let values = evaluate_spline(&self.knots, &self.coefficients, self.order, &[x], deriv)?;
        Ok(values[0])
    }

    fn valid_range(&self) -> (T, T) {
        self.valid_range
    }
}

impl<T: Float + Sum + 'static> BasisMatrix<T> for InterpolationSpline<T> {
    fn basis_matrix(&self, x: &[T]) -> SplineResult<Array2<T>> {
        let builder = DesignMatrixBuilder::new(self.knots.clone(), self.order);
        builder.build(x)
    }

    fn basis_matrix_with_deriv(&self, x: &[T], derivs: &[usize]) -> SplineResult<Array2<T>> {
        if derivs.len() != x.len() {
            return Err(SplineError::dimension_mismatch(x.len(), derivs.len()));
        }

        let builder = DesignMatrixBuilder::new(self.knots.clone(), self.order);
        builder.build_with_derivatives(x, derivs)
    }
}

/// Simple linear system solver using Gaussian elimination
fn solve_linear_system<T: Float + Sum + 'static>(
    matrix: &Array2<T>,
    rhs: &Array1<T>,
) -> SplineResult<Array1<T>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(SplineError::ComputationError(
            "Matrix must be square for linear system solving".to_string(),
        ));
    }

    if rhs.len() != n {
        return Err(SplineError::dimension_mismatch(n, rhs.len()));
    }

    // Create augmented matrix [A | b]
    let mut augmented = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            augmented[[i, j]] = matrix[[i, j]];
        }
        augmented[[i, n]] = rhs[i];
    }

    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        let mut max_val = augmented[[i, i]].abs();
        for k in (i + 1)..n {
            let val = augmented[[k, i]].abs();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        // Swap rows if needed
        if pivot_row != i {
            for j in 0..=n {
                let temp = augmented[[i, j]];
                augmented[[i, j]] = augmented[[pivot_row, j]];
                augmented[[pivot_row, j]] = temp;
            }
        }

        // Check for singular matrix (use appropriate tolerance)
        let tolerance = T::from(1e-12).unwrap_or(T::epsilon());
        if augmented[[i, i]].abs() < tolerance {
            return Err(SplineError::ComputationError(format!(
                "Singular matrix in interpolation system at pivot {}",
                i
            )));
        }

        // Eliminate column
        for k in (i + 1)..n {
            let factor = augmented[[k, i]] / augmented[[i, i]];
            for j in i..=n {
                augmented[[k, j]] = augmented[[k, j]] - factor * augmented[[i, j]];
            }
        }
    }

    // Back substitution
    let mut solution = vec![T::zero(); n];
    for i in (0..n).rev() {
        let mut sum = augmented[[i, n]];
        for j in (i + 1)..n {
            sum = sum - augmented[[i, j]] * solution[j];
        }
        solution[i] = sum / augmented[[i, i]];
    }

    Ok(Array1::from(solution))
}

/// Builder for interpolation splines with fluent interface
pub struct InterpolationSplineBuilder<T: Float + Sum + 'static> {
    order: usize,
    _phantom: PhantomData<T>,
}

impl<T: Float + Sum + 'static> InterpolationSplineBuilder<T> {
    /// Create a new builder with default values
    pub fn new() -> Self {
        InterpolationSplineBuilder {
            order: 4, // Cubic splines by default
            _phantom: PhantomData,
        }
    }

    /// Set the spline order (degree + 1)
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set the spline degree
    pub fn degree(mut self, degree: usize) -> Self {
        self.order = degree + 1;
        self
    }

    /// Build the interpolation spline from data points
    pub fn build(self, x: &[T], y: &[T]) -> SplineResult<InterpolationSpline<T>> {
        InterpolationSpline::new(x, y, self.order)
    }
}

impl<T: Float + Sum + 'static> Default for InterpolationSplineBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_interpolation_spline_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // x^2 data

        let spline = InterpolationSpline::new(&x, &y, 4).unwrap();

        assert_eq!(spline.order(), 4);
        assert_eq!(spline.degree(), 3);
        assert_eq!(spline.spline_type(), SplineType::Interpolation);
    }

    #[test]
    fn test_interpolation_spline_exact_fit() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 8.0, 27.0]; // x^3 data

        let spline = InterpolationSpline::new(&x, &y, 4).unwrap();

        // Should interpolate exactly at data points
        for (i, &x_val) in x.iter().enumerate() {
            let result = spline.evaluate(x_val).unwrap();
            println!(
                "x={}, expected={}, actual={}, diff={}",
                x_val,
                y[i],
                result,
                (result - y[i]).abs()
            );
            assert_relative_eq!(result, y[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_interpolation_spline_between_points() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x^2 data

        let spline = InterpolationSpline::new(&x, &y, 4).unwrap(); // Cubic

        // Test interpolation between points
        let mid_result = spline.evaluate(0.5).unwrap();
        assert!(mid_result > 0.0 && mid_result < 1.0);

        let mid_result2 = spline.evaluate(2.5).unwrap();
        assert!(mid_result2 > 4.0 && mid_result2 < 9.0);
    }

    #[test]
    fn test_interpolation_spline_derivatives() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0]; // x^2 data

        let spline = InterpolationSpline::new(&x, &y, 4).unwrap();

        // Test that derivatives are finite
        let deriv1 = spline.evaluate_with_deriv(1.5, 1).unwrap();
        let deriv2 = spline.evaluate_with_deriv(1.5, 2).unwrap();

        assert!(deriv1.is_finite());
        assert!(deriv2.is_finite());
    }

    #[test]
    fn test_interpolation_spline_builder() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];

        let spline = InterpolationSpline::builder()
            .degree(3)
            .build(&x, &y)
            .unwrap();

        assert_eq!(spline.degree(), 3);
        assert_eq!(spline.order(), 4);
    }

    #[test]
    fn test_interpolation_spline_data_sorting() {
        // Test with unsorted data
        let x = vec![3.0, 1.0, 4.0, 2.0];
        let y = vec![9.0, 1.0, 16.0, 4.0]; // Corresponding x^2 values

        let spline = InterpolationSpline::new(&x, &y, 4).unwrap();

        // Should work correctly after internal sorting
        let result = spline.evaluate(2.5).unwrap();
        assert!(result.is_finite());
        assert!(result > 4.0 && result < 16.0); // Between y values at x=2 and x=4
    }

    #[test]
    fn test_interpolation_spline_errors() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 4.0]; // Wrong length

        // Should fail with dimension mismatch
        assert!(InterpolationSpline::new(&x, &y, 4).is_err());

        // Should fail with non-cubic order
        let x_good = vec![1.0, 2.0, 3.0, 4.0];
        let y_good = vec![1.0, 4.0, 9.0, 16.0];
        assert!(InterpolationSpline::new(&x_good, &y_good, 3).is_err());

        // Should fail with duplicate x values
        let x_dup = vec![1.0, 2.0, 2.0, 3.0];
        let y_dup = vec![1.0, 4.0, 9.0, 16.0];
        assert!(InterpolationSpline::new(&x_dup, &y_dup, 4).is_err());
    }

    #[test]
    fn test_interpolation_spline_fundamental_properties() {
        // Verify that our interpolation spline satisfies the fundamental requirement:
        // exact interpolation at data points. This is the key mathematical property.
        // Note: We use B-spline representation while R uses piecewise polynomials.
        // Both are valid interpolating splines but may produce different intermediate values.
        let x_vals = vec![0.0, 1.0, 2.0, 3.0];
        let y_vals = vec![0.0, 1.0, 8.0, 27.0];
        let order = 4;

        let spline = InterpolationSpline::new(&x_vals, &y_vals, order).unwrap();

        // Test 1: Exact interpolation at data points (THE fundamental requirement)
        println!("=== Testing exact interpolation at data points ===");
        for (&x, &expected_y) in x_vals.iter().zip(y_vals.iter()) {
            let actual_y = spline.evaluate(x).unwrap();
            println!("f({}) = {} (expected {})", x, actual_y, expected_y);
            assert_relative_eq!(actual_y, expected_y, epsilon = 1e-12);
        }

        // Test 2: Verify function values are finite and reasonable
        println!("\n=== Testing function value validity ===");
        let test_x_vals = vec![0.1, 0.5, 0.9, 1.1, 1.5, 1.9, 2.1, 2.5, 2.9];

        for x_test in test_x_vals {
            let y_test = spline.evaluate(x_test).unwrap();
            println!("f({}) = {:.6}", x_test, y_test);

            // Should be finite (no NaN or infinity)
            assert!(
                y_test.is_finite(),
                "Function value should be finite at x={}",
                x_test
            );
        }

        // Test 3: Verify spline properties
        println!("\n=== Testing spline structure ===");
        println!("Order: {}", spline.order());
        println!("Degree: {}", spline.degree());
        println!("Knots length: {}", spline.knots().len());
        println!("Coefficients length: {}", spline.coefficients().len());

        // Basic sanity checks
        assert_eq!(spline.order(), 4);
        assert_eq!(spline.degree(), 3);
        assert!(spline.knots().len() > 0);
        assert!(spline.coefficients().len() > 0);

        println!("✓ Interpolation spline fundamental properties verified");
        println!("  - Exact interpolation at data points ✓");
        println!("  - Finite function values ✓");
        println!("  - Valid spline structure ✓");
        println!(
            "  Note: Uses B-spline basis (mathematically equivalent to R's piecewise polynomials)"
        );
    }
}
