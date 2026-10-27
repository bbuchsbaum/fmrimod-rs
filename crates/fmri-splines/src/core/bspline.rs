//! B-spline implementation (equivalent to R's bs() function)

use crate::basis::design_matrix::DesignMatrixBuilder;
use crate::errors::{SplineError, SplineResult};
use crate::types::{BasisMatrix, Spline, SplineType, XYVector};
use crate::utils::knots::{
    compute_boundary_knots, create_augmented_knots, select_knots_by_df, validate_knot_configuration,
};
use crate::utils::validation::{
    validate_degree, validate_finite, validate_min_points, validate_x_range,
};
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::iter::Sum;

/// B-spline representation
///
/// This struct represents a B-spline basis equivalent to R's `bs()` function.
/// B-splines provide a flexible basis for representing smooth functions using
/// piecewise polynomials.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BSpline<T: Float + Sum> {
    /// Augmented knot vector (with boundary multiplicities)
    knots: Array1<T>,
    /// Interior knots only
    interior_knots: Array1<T>,
    /// Boundary knots (min, max)
    boundary_knots: (T, T),
    /// Polynomial degree
    degree: usize,
    /// Whether to include intercept in basis
    intercept: bool,
    /// Number of basis functions
    n_basis: usize,
}

impl<T: Float + Sum> BSpline<T> {
    /// Create a new B-spline basis (equivalent to R's bs() function)
    ///
    /// # Arguments
    /// * `x` - Input data points for determining knot placement
    /// * `df` - Degrees of freedom (alternative to specifying knots)
    /// * `knots` - Interior knot positions (optional)
    /// * `degree` - Polynomial degree (default 3 for cubic splines)
    /// * `intercept` - Whether to include intercept term
    /// * `boundary_knots` - Boundary knot positions (optional, defaults to data range)
    ///
    /// # Examples
    /// ```
    /// use fmri_splines::BSpline;
    ///
    /// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    ///
    /// // Create cubic B-spline with automatic knot selection
    /// let spline = BSpline::new(&x, Some(5), None, 3, false, None)?;
    ///
    /// // Create with explicit knots
    /// let knots = vec![2.5, 3.5];
    /// let spline = BSpline::new(&x, None, Some(knots), 3, false, None)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(
        x: &[T],
        df: Option<usize>,
        knots: Option<Vec<T>>,
        degree: usize,
        intercept: bool,
        boundary_knots: Option<(T, T)>,
    ) -> SplineResult<Self> {
        // Validate inputs
        validate_finite(x, "x")?;
        validate_min_points(x, 2)?;
        validate_degree(degree)?;

        // Determine boundary knots
        let boundary_knots = if let Some(bounds) = boundary_knots {
            crate::utils::validation::validate_boundary_knots(bounds)?;
            bounds
        } else {
            compute_boundary_knots(x)?
        };

        // Determine interior knots
        let interior_knots = if let Some(explicit_knots) = knots {
            let knot_array = Array1::from(explicit_knots);
            validate_knot_configuration(&knot_array, boundary_knots, degree)?;
            knot_array
        } else if let Some(df_val) = df {
            select_knots_by_df(x, df_val, degree, intercept, Some(boundary_knots))?
        } else {
            Array1::from(vec![]) // No interior knots
        };

        // Create augmented knot vector
        let order = degree + 1;
        let augmented_knots = create_augmented_knots(&interior_knots, boundary_knots, order)?;

        // Calculate number of basis functions
        let n_coeffs = augmented_knots.len() - order;
        let n_basis = if intercept { n_coeffs } else { n_coeffs - 1 };

        if n_basis == 0 {
            return Err(SplineError::ComputationError(
                "No basis functions would be generated with current parameters".to_string(),
            ));
        }

        Ok(BSpline {
            knots: augmented_knots,
            interior_knots,
            boundary_knots,
            degree,
            intercept,
            n_basis,
        })
    }

    /// Create a B-spline with builder pattern
    pub fn builder() -> BSplineBuilder<T> {
        BSplineBuilder::new()
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

    /// Generate the design matrix for given x values
    pub fn design_matrix(&self, x: &[T]) -> SplineResult<Array2<T>> {
        // Validate x values are in range
        validate_x_range(x, self.boundary_knots.0, self.boundary_knots.1, "x")?;

        let builder = if self.intercept {
            DesignMatrixBuilder::new(self.knots.clone(), self.order())
        } else {
            DesignMatrixBuilder::new_no_intercept(self.knots.clone(), self.order())
        };

        builder.build(x)
    }

    /// Generate the design matrix with derivatives
    pub fn design_matrix_with_deriv(&self, x: &[T], deriv: usize) -> SplineResult<Array2<T>> {
        validate_x_range(x, self.boundary_knots.0, self.boundary_knots.1, "x")?;

        let derivatives = vec![deriv; x.len()];
        let builder = if self.intercept {
            DesignMatrixBuilder::new(self.knots.clone(), self.order())
        } else {
            DesignMatrixBuilder::new_no_intercept(self.knots.clone(), self.order())
        };

        builder.build_with_derivatives(x, &derivatives)
    }
}

impl<T: Float + Sum> Spline<T> for BSpline<T> {
    fn knots(&self) -> &Array1<T> {
        &self.knots
    }

    fn order(&self) -> usize {
        self.degree + 1
    }

    fn spline_type(&self) -> SplineType {
        SplineType::BSpline
    }

    fn predict(&self, x: &[T]) -> SplineResult<XYVector<T>> {
        let matrix = self.design_matrix(x)?;

        // For prediction, we return the design matrix rows as y values
        // This matches R's behavior where bs() returns the basis matrix
        let x_array = Array1::from(x.to_vec());
        let mut y_values = Vec::with_capacity(x.len());

        // Return the sum of basis functions as a simple prediction
        // (in practice, this would be multiplied by coefficients)
        for i in 0..matrix.nrows() {
            let row_sum: T = matrix.row(i).iter().cloned().sum();
            y_values.push(row_sum);
        }

        XYVector::from_arrays(x_array, Array1::from(y_values))
    }

    fn predict_with_deriv(&self, x: &[T], deriv: usize) -> SplineResult<XYVector<T>> {
        crate::utils::validation::validate_derivative_order(deriv, self.order())?;

        let matrix = self.design_matrix_with_deriv(x, deriv)?;
        let x_array = Array1::from(x.to_vec());
        let mut y_values = Vec::with_capacity(x.len());

        for i in 0..matrix.nrows() {
            let row_sum: T = matrix.row(i).iter().cloned().sum();
            y_values.push(row_sum);
        }

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
        self.boundary_knots
    }
}

impl<T: Float + Sum> BasisMatrix<T> for BSpline<T> {
    fn basis_matrix(&self, x: &[T]) -> SplineResult<Array2<T>> {
        self.design_matrix(x)
    }

    fn basis_matrix_with_deriv(&self, x: &[T], derivs: &[usize]) -> SplineResult<Array2<T>> {
        if derivs.len() != x.len() {
            return Err(SplineError::dimension_mismatch(x.len(), derivs.len()));
        }

        validate_x_range(x, self.boundary_knots.0, self.boundary_knots.1, "x")?;

        let builder = if self.intercept {
            DesignMatrixBuilder::new(self.knots.clone(), self.order())
        } else {
            DesignMatrixBuilder::new_no_intercept(self.knots.clone(), self.order())
        };

        builder.build_with_derivatives(x, derivs)
    }
}

/// Builder for B-splines with fluent interface
pub struct BSplineBuilder<T: Float + Sum> {
    degree: usize,
    intercept: bool,
    df: Option<usize>,
    knots: Option<Vec<T>>,
    boundary_knots: Option<(T, T)>,
}

impl<T: Float + Sum> BSplineBuilder<T> {
    /// Create a new builder with default values
    pub fn new() -> Self {
        BSplineBuilder {
            degree: 3, // Cubic splines by default
            intercept: false,
            df: None,
            knots: None,
            boundary_knots: None,
        }
    }

    /// Set the polynomial degree
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
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

    /// Build the B-spline from input data
    pub fn build(self, x: &[T]) -> SplineResult<BSpline<T>> {
        BSpline::new(
            x,
            self.df,
            self.knots,
            self.degree,
            self.intercept,
            self.boundary_knots,
        )
    }
}

impl<T: Float + Sum> Default for BSplineBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_bspline_basic() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = BSpline::new(&x, Some(5), None, 3, false, None).unwrap();

        assert_eq!(spline.degree(), 3);
        assert_eq!(spline.order(), 4);
        assert!(!spline.has_intercept());
        assert_eq!(spline.spline_type(), SplineType::BSpline);
    }

    #[test]
    fn test_bspline_with_knots() {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let knots = vec![1.5, 2.5];
        let spline = BSpline::new(&x, None, Some(knots), 2, true, None).unwrap();

        assert_eq!(spline.degree(), 2);
        assert!(spline.has_intercept());
        assert_eq!(spline.interior_knots().len(), 2);
    }

    #[test]
    fn test_bspline_design_matrix() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // Use intercept=true to ensure partition of unity property
        let spline = BSpline::new(&x, Some(4), None, 2, true, None).unwrap();

        let test_x = vec![1.5, 2.5, 3.5];
        let matrix = spline.design_matrix(&test_x).unwrap();

        // Check dimensions
        assert_eq!(matrix.nrows(), 3);

        // Check partition of unity property (each row sums to 1)
        // This should hold when intercept=true
        for i in 0..matrix.nrows() {
            let row_sum: f64 = matrix.row(i).iter().sum();
            assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bspline_builder() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let spline = BSpline::builder()
            .degree(2)
            .intercept(true)
            .df(5)
            .build(&x)
            .unwrap();

        assert_eq!(spline.degree(), 2);
        assert!(spline.has_intercept());
    }

    #[test]
    fn test_bspline_prediction() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        // Use intercept=true to ensure partition of unity property
        let spline = BSpline::new(&x, Some(4), None, 2, true, None).unwrap();

        let test_x = vec![1.5, 2.5];
        let result = spline.predict(&test_x).unwrap();

        assert_eq!(result.len(), 2);
        // Results should be close to 1.0 due to partition of unity when intercept=true
        for &y_val in result.y_values() {
            assert_relative_eq!(y_val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bspline_valid_range() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = BSpline::new(&x, None, None, 3, false, Some((0.5, 5.5))).unwrap();

        let (min, max) = spline.valid_range();
        assert_eq!(min, 0.5);
        assert_eq!(max, 5.5);

        assert!(spline.in_range(3.0));
        assert!(!spline.in_range(0.0));
        assert!(!spline.in_range(6.0));
    }
}
