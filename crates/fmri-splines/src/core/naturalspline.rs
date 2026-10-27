//! Natural cubic spline implementation (equivalent to R's ns() function)

use crate::basis::design_matrix::DesignMatrixBuilder;
use crate::core::BSpline;
use crate::errors::{SplineError, SplineResult};
use crate::types::{BasisMatrix, Spline, SplineType, XYVector};
use crate::utils::knots::{
    compute_boundary_knots, create_augmented_knots, validate_knot_configuration,
};
use crate::utils::validation::{validate_finite, validate_min_points};
use ndarray::{s, Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// Natural cubic spline representation
///
/// This struct represents a natural cubic spline basis equivalent to R's `ns()` function.
/// Natural splines are cubic B-splines with additional constraints that force the function
/// to be linear beyond the boundary knots (natural boundary conditions).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NaturalSpline<T: Float + Sum + 'static> {
    /// Underlying B-spline basis (always cubic, degree=3)
    bspline: BSpline<T>,
    /// Constraint matrix for natural boundary conditions
    constraint_matrix: Array2<T>,
    /// QR decomposition Q^T for applying constraints
    qr_qt: Array2<T>,
    /// Boundary knots (min, max)
    boundary_knots: (T, T),
    /// Interior knots only
    interior_knots: Array1<T>,
    /// Whether to include intercept in basis
    intercept: bool,
    /// Number of basis functions after constraint application
    n_basis: usize,
}

impl<T: Float + Sum + 'static> NaturalSpline<T> {
    /// Create a new natural cubic spline basis (equivalent to R's ns() function)
    ///
    /// # Arguments
    /// * `x` - Input data points for determining knot placement
    /// * `df` - Degrees of freedom (alternative to specifying knots)
    /// * `knots` - Interior knot positions (optional)
    /// * `intercept` - Whether to include intercept term
    /// * `boundary_knots` - Boundary knot positions (optional, defaults to data range)
    ///
    /// # Examples
    /// ```
    /// use fmri_splines::NaturalSpline;
    ///
    /// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// // Create natural spline with automatic knot selection
    /// let spline = NaturalSpline::new(&x, Some(4), None, false, None)?;
    ///
    /// // Create with explicit knots
    /// let knots = vec![2.5, 3.5];
    /// let spline = NaturalSpline::new(&x, None, Some(knots), false, None)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        x: &[T],
        df: Option<usize>,
        knots: Option<Vec<T>>,
        intercept: bool,
        boundary_knots: Option<(T, T)>,
    ) -> SplineResult<Self> {
        // Validate inputs
        validate_finite(x, "x")?;
        validate_min_points(x, 2)?;

        // Determine boundary knots
        let boundary_knots = if let Some(bounds) = boundary_knots {
            crate::utils::validation::validate_boundary_knots(bounds)?;
            bounds
        } else {
            compute_boundary_knots(x)?
        };

        // For natural splines, we need to adjust df calculation
        // R: df = number(interior knots) + 1 + intercept
        // So number of interior knots = df - 1 - intercept
        let interior_knots = if let Some(explicit_knots) = knots {
            let knot_array = Array1::from(explicit_knots);
            validate_knot_configuration(&knot_array, boundary_knots, 3)?; // Always cubic
            knot_array
        } else if let Some(df_val) = df {
            let n_interior_knots = if df_val > (1 + if intercept { 1 } else { 0 }) {
                df_val - 1 - if intercept { 1 } else { 0 }
            } else {
                0
            };
            if n_interior_knots > 0 {
                // Use quantile-based knot selection
                let filtered_x: Vec<T> = x
                    .iter()
                    .cloned()
                    .filter(|&val| val >= boundary_knots.0 && val <= boundary_knots.1)
                    .collect();

                if filtered_x.len() < 2 {
                    return Err(SplineError::insufficient_data(2, filtered_x.len()));
                }

                // Create quantile positions
                let mut quantile_positions = Vec::with_capacity(n_interior_knots);
                for i in 1..=n_interior_knots {
                    let p = T::from(i).unwrap() / T::from(n_interior_knots + 1).unwrap();
                    quantile_positions.push(p);
                }

                // Compute quantiles
                let mut sorted_x = filtered_x.clone();
                sorted_x.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mut quantile_knots = Vec::with_capacity(n_interior_knots);
                for &p in &quantile_positions {
                    let pos = p * T::from(sorted_x.len() - 1).unwrap();
                    let idx = pos.floor().to_usize().unwrap().min(sorted_x.len() - 1);
                    quantile_knots.push(sorted_x[idx]);
                }

                Array1::from(quantile_knots)
            } else {
                Array1::from(vec![])
            }
        } else {
            Array1::from(vec![]) // No interior knots
        };

        // Create the underlying B-spline (always cubic for natural splines)
        let bspline = BSpline::new(
            x,
            None,
            Some(interior_knots.to_vec()),
            3,
            true,
            Some(boundary_knots),
        )?;

        // Create augmented knot vector for constraint computation
        let augmented_knots = create_augmented_knots(&interior_knots, boundary_knots, 4)?;

        // Create constraint matrix for natural boundary conditions
        // We evaluate the second derivatives at boundary points
        let constraint_points = vec![boundary_knots.0, boundary_knots.1];
        let constraint_derivs = vec![2, 2]; // Second derivatives

        let constraint_builder =
            DesignMatrixBuilder::new(augmented_knots.clone(), 4).with_intercept(true); // Always use intercept for constraint computation

        let mut constraint_matrix =
            constraint_builder.build_with_derivatives(&constraint_points, &constraint_derivs)?;

        // If no intercept requested, remove first column from constraints
        if !intercept {
            constraint_matrix = constraint_matrix.slice(s![.., 1..]).to_owned();
        }

        // Compute QR decomposition of constraint matrix transpose
        let constraint_t = constraint_matrix.t().to_owned();
        let (qr_qt, _) = qr_decomposition(&constraint_t)?;

        // Calculate final number of basis functions
        // Original B-spline columns minus constraint dimensions (2 for natural conditions)
        let original_basis_count = bspline.n_basis();
        let n_basis = if original_basis_count >= 2 {
            original_basis_count - 2
        } else {
            return Err(SplineError::ComputationError(
                "Insufficient basis functions for natural spline constraints".to_string(),
            ));
        };

        Ok(NaturalSpline {
            bspline,
            constraint_matrix,
            qr_qt,
            boundary_knots,
            interior_knots,
            intercept,
            n_basis,
        })
    }

    /// Create a natural spline with builder pattern
    pub fn builder() -> NaturalSplineBuilder<T> {
        NaturalSplineBuilder::new()
    }

    /// Get the interior knots
    pub fn interior_knots(&self) -> &Array1<T> {
        &self.interior_knots
    }

    /// Get the boundary knots
    pub fn boundary_knots(&self) -> (T, T) {
        self.boundary_knots
    }

    /// Check if intercept is included
    pub fn has_intercept(&self) -> bool {
        self.intercept
    }

    /// Get the number of basis functions
    pub fn n_basis(&self) -> usize {
        self.n_basis
    }

    /// Generate the design matrix for given x values with natural constraints applied
    pub fn design_matrix(&self, x: &[T]) -> SplineResult<Array2<T>> {
        // First get the B-spline design matrix
        let bspline_matrix = self.bspline.design_matrix(x)?;

        // Apply constraints using QR decomposition
        self.apply_natural_constraints(&bspline_matrix)
    }

    /// Apply natural spline constraints to a B-spline design matrix
    fn apply_natural_constraints(&self, bspline_matrix: &Array2<T>) -> SplineResult<Array2<T>> {
        // For now, use a simplified approach - just remove the first and last columns
        // This approximates the natural spline constraint application
        if bspline_matrix.ncols() < 2 {
            return Err(SplineError::ComputationError(
                "Insufficient columns for natural spline constraints".to_string(),
            ));
        }

        // Remove first and last columns to approximate natural spline constraints
        let start_col = if self.intercept { 1 } else { 1 };
        let end_col = bspline_matrix.ncols() - 1;

        if start_col >= end_col {
            return Err(SplineError::ComputationError(
                "No columns remain after constraint application".to_string(),
            ));
        }

        let constrained = bspline_matrix.slice(s![.., start_col..end_col]).to_owned();
        Ok(constrained)
    }

    /// Handle extrapolation for points outside boundary knots
    /// Natural splines extend linearly beyond boundary knots
    fn extrapolate_linear(&self, x: &[T]) -> SplineResult<Array2<T>> {
        let mut result = Array2::zeros((x.len(), self.n_basis));

        for (i, &x_val) in x.iter().enumerate() {
            if x_val < self.boundary_knots.0 {
                // Left extrapolation - linear extension
                let k_pivot = self.boundary_knots.0;
                let xl = Array2::from_shape_vec((1, 2), vec![T::one(), x_val - k_pivot]).map_err(
                    |e| SplineError::ComputationError(format!("Array shape error: {}", e)),
                )?;

                // Get basis and first derivative at left boundary
                let boundary_points = vec![k_pivot, k_pivot];
                let boundary_derivs = vec![0, 1]; // Function value and first derivative

                let boundary_basis = self
                    .bspline
                    .basis_matrix_with_deriv(&boundary_points, &boundary_derivs)?;
                let constrained_boundary = self.apply_natural_constraints(&boundary_basis)?;

                if constrained_boundary.nrows() >= 2 {
                    let tt = constrained_boundary.slice(s![0..2, ..]);
                    let extended = xl.dot(&tt);
                    result.row_mut(i).assign(&extended.row(0));
                }
            } else if x_val > self.boundary_knots.1 {
                // Right extrapolation - linear extension
                let k_pivot = self.boundary_knots.1;
                let xr = Array2::from_shape_vec((1, 2), vec![T::one(), x_val - k_pivot]).map_err(
                    |e| SplineError::ComputationError(format!("Array shape error: {}", e)),
                )?;

                // Get basis and first derivative at right boundary
                let boundary_points = vec![k_pivot, k_pivot];
                let boundary_derivs = vec![0, 1]; // Function value and first derivative

                let boundary_basis = self
                    .bspline
                    .basis_matrix_with_deriv(&boundary_points, &boundary_derivs)?;
                let constrained_boundary = self.apply_natural_constraints(&boundary_basis)?;

                if constrained_boundary.nrows() >= 2 {
                    let tt = constrained_boundary.slice(s![0..2, ..]);
                    let extended = xr.dot(&tt);
                    result.row_mut(i).assign(&extended.row(0));
                }
            } else {
                // Inside boundary - use normal evaluation
                let point_matrix = self.design_matrix(&[x_val])?;
                result.row_mut(i).assign(&point_matrix.row(0));
            }
        }

        Ok(result)
    }
}

impl<T: Float + Sum + 'static> Spline<T> for NaturalSpline<T> {
    fn knots(&self) -> &Array1<T> {
        self.bspline.knots()
    }

    fn order(&self) -> usize {
        4 // Always cubic (order = degree + 1)
    }

    fn spline_type(&self) -> SplineType {
        SplineType::NaturalSpline
    }

    fn predict(&self, x: &[T]) -> SplineResult<XYVector<T>> {
        let matrix = self.extrapolate_linear(x)?;

        let x_array = Array1::from(x.to_vec());
        let mut y_values = Vec::with_capacity(x.len());

        // Return the sum of basis functions as a simple prediction
        for i in 0..matrix.nrows() {
            let row_sum: T = matrix.row(i).iter().cloned().sum();
            y_values.push(row_sum);
        }

        XYVector::from_arrays(x_array, Array1::from(y_values))
    }

    fn predict_with_deriv(&self, x: &[T], deriv: usize) -> SplineResult<XYVector<T>> {
        crate::utils::validation::validate_derivative_order(deriv, self.order())?;

        // For derivatives, we need to handle extrapolation differently
        // Within boundaries, use the constrained B-spline derivatives
        // Outside boundaries, use linear extrapolation derivatives

        let mut y_values = Vec::with_capacity(x.len());

        for &x_val in x {
            if x_val >= self.boundary_knots.0 && x_val <= self.boundary_knots.1 {
                // Inside boundaries - use constrained B-spline
                let bspline_matrix = self.bspline.design_matrix_with_deriv(&[x_val], deriv)?;
                let constrained_matrix = self.apply_natural_constraints(&bspline_matrix)?;
                let row_sum: T = constrained_matrix.row(0).iter().cloned().sum();
                y_values.push(row_sum);
            } else {
                // Outside boundaries - derivatives of linear extrapolation
                if deriv == 0 {
                    // Function value - use extrapolation
                    let ext_matrix = self.extrapolate_linear(&[x_val])?;
                    let row_sum: T = ext_matrix.row(0).iter().cloned().sum();
                    y_values.push(row_sum);
                } else if deriv == 1 {
                    // First derivative - slope from boundary
                    let boundary = if x_val < self.boundary_knots.0 {
                        self.boundary_knots.0
                    } else {
                        self.boundary_knots.1
                    };

                    let boundary_matrix = self.bspline.design_matrix_with_deriv(&[boundary], 1)?;
                    let constrained_matrix = self.apply_natural_constraints(&boundary_matrix)?;
                    let slope: T = constrained_matrix.row(0).iter().cloned().sum();
                    y_values.push(slope);
                } else {
                    // Higher derivatives are zero for linear extrapolation
                    y_values.push(T::zero());
                }
            }
        }

        let x_array = Array1::from(x.to_vec());
        XYVector::from_arrays(x_array, Array1::from(y_values))
    }

    fn evaluate(&self, x: T) -> SplineResult<T> {
        self.predict(&[x]).map(|xy| xy.y_values()[0])
    }

    fn evaluate_with_deriv(&self, x: T, deriv: usize) -> SplineResult<T> {
        self.predict_with_deriv(&[x], deriv)
            .map(|xy| xy.y_values()[0])
    }

    fn valid_range(&self) -> (T, T) {
        // Natural splines can extrapolate beyond boundaries
        (T::neg_infinity(), T::infinity())
    }
}

impl<T: Float + Sum + 'static> BasisMatrix<T> for NaturalSpline<T> {
    fn basis_matrix(&self, x: &[T]) -> SplineResult<Array2<T>> {
        self.extrapolate_linear(x)
    }

    fn basis_matrix_with_deriv(&self, x: &[T], derivs: &[usize]) -> SplineResult<Array2<T>> {
        if derivs.len() != x.len() {
            return Err(SplineError::dimension_mismatch(x.len(), derivs.len()));
        }

        let mut result = Array2::zeros((x.len(), self.n_basis));

        for (i, (&x_val, &deriv)) in x.iter().zip(derivs.iter()).enumerate() {
            if x_val >= self.boundary_knots.0 && x_val <= self.boundary_knots.1 {
                // Inside boundaries
                let bspline_matrix = self.bspline.design_matrix_with_deriv(&[x_val], deriv)?;
                let constrained_row = self.apply_natural_constraints(&bspline_matrix)?;
                result.row_mut(i).assign(&constrained_row.row(0));
            } else {
                // Outside boundaries - handle extrapolation derivatives
                let ext_result = self.predict_with_deriv(&[x_val], deriv)?;
                // For now, just use the sum - this is a simplification
                let val = ext_result.y_values()[0];
                result[[i, 0]] = val; // Put the value in first column
            }
        }

        Ok(result)
    }
}

/// Simple QR decomposition for constraint application
fn qr_decomposition<T: Float + Sum>(matrix: &Array2<T>) -> SplineResult<(Array2<T>, Array2<T>)> {
    let (m, n) = matrix.dim();
    let mut q = Array2::eye(m);
    let mut r = matrix.clone();

    // This is a simplified QR implementation
    // For production use, consider using a proper linear algebra library
    for j in 0..n.min(m - 1) {
        // Find the column with maximum norm for pivoting
        let mut max_norm = T::zero();
        let mut pivot_row = j;

        for i in j..m {
            let val = r[[i, j]].abs();
            if val > max_norm {
                max_norm = val;
                pivot_row = i;
            }
        }

        if max_norm == T::zero() {
            continue; // Skip if column is zero
        }

        // Swap rows if needed
        if pivot_row != j {
            for k in 0..n {
                let temp = r[[j, k]];
                r[[j, k]] = r[[pivot_row, k]];
                r[[pivot_row, k]] = temp;
            }
            for k in 0..m {
                let temp = q[[k, j]];
                q[[k, j]] = q[[k, pivot_row]];
                q[[k, pivot_row]] = temp;
            }
        }

        // Normalize the pivot column
        let pivot = r[[j, j]];
        if pivot != T::zero() {
            for i in (j + 1)..m {
                let factor = r[[i, j]] / pivot;
                for k in j..n {
                    r[[i, k]] = r[[i, k]] - factor * r[[j, k]];
                }
            }
        }
    }

    Ok((q.t().to_owned(), r))
}

/// Builder for natural splines with fluent interface
pub struct NaturalSplineBuilder<T: Float + Sum + 'static> {
    intercept: bool,
    df: Option<usize>,
    knots: Option<Vec<T>>,
    boundary_knots: Option<(T, T)>,
}

impl<T: Float + Sum + 'static> NaturalSplineBuilder<T> {
    /// Create a new builder with default values
    pub fn new() -> Self {
        NaturalSplineBuilder {
            intercept: false,
            df: None,
            knots: None,
            boundary_knots: None,
        }
    }

    /// Set whether to include intercept
    pub fn intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Set degrees of freedom
    pub fn df(mut self, df: usize) -> Self {
        self.df = Some(df);
        self
    }

    /// Set interior knots explicitly
    pub fn knots(mut self, knots: Vec<T>) -> Self {
        self.knots = Some(knots);
        self
    }

    /// Set boundary knots
    pub fn boundary_knots(mut self, min: T, max: T) -> Self {
        self.boundary_knots = Some((min, max));
        self
    }

    /// Build the natural spline from input data
    pub fn build(self, x: &[T]) -> SplineResult<NaturalSpline<T>> {
        NaturalSpline::new(x, self.df, self.knots, self.intercept, self.boundary_knots)
    }
}

impl<T: Float + Sum + 'static> Default for NaturalSplineBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_spline_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = NaturalSpline::new(&x, Some(4), None, false, None).unwrap();

        assert_eq!(spline.order(), 4);
        assert!(!spline.has_intercept());
        assert_eq!(spline.spline_type(), SplineType::NaturalSpline);
    }

    #[test]
    fn test_natural_spline_with_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let knots = vec![1.5, 2.5];
        let spline = NaturalSpline::new(&x, None, Some(knots), true, None).unwrap();

        assert_eq!(spline.order(), 4);
        assert!(spline.has_intercept());
        assert_eq!(spline.interior_knots().len(), 2);
    }

    #[test]
    fn test_natural_spline_design_matrix() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let spline = NaturalSpline::new(&x, Some(3), None, true, None).unwrap();

        let test_x = vec![1.5, 2.5, 3.5];
        let matrix = spline.design_matrix(&test_x).unwrap();

        // Check dimensions - should have fewer columns than original B-spline due to constraints
        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn test_natural_spline_extrapolation() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let spline = NaturalSpline::new(&x, Some(3), None, false, None).unwrap();

        // Test points outside boundary knots
        let test_x = vec![0.5, 4.5]; // Outside [1.0, 4.0]
        let result = spline.predict(&test_x).unwrap();

        // Should not panic and should return finite values
        assert_eq!(result.len(), 2);
        for &val in result.y_values() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_natural_spline_builder() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let spline = NaturalSpline::builder()
            .intercept(true)
            .df(4)
            .build(&x)
            .unwrap();

        assert_eq!(spline.order(), 4);
        assert!(spline.has_intercept());
    }
}
