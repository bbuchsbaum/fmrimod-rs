//! Utility functions for spline objects
//!
//! This module provides utility functions similar to R's spline utility functions:
//! - splineKnots() - extract knot vectors from spline objects
//! - splineOrder() - extract spline order from spline objects
//! - asVector() - convert spline objects to vector representations

use crate::core::{BSpline, InterpolationSpline, NaturalSpline};
use crate::errors::SplineResult;
use crate::types::Spline;
use ndarray::Array1;
use num_traits::Float;
use std::iter::Sum;

/// Extract the knot vector from a spline object
///
/// This function is equivalent to R's `splineKnots()` function.
/// It returns the knot vector for any spline object that implements the Spline trait.
///
/// # Arguments
/// * `spline` - A spline object that implements the Spline trait
///
/// # Returns
/// A reference to the knot vector as an Array1
///
/// # Examples
/// ```rust
/// use splines::{BSpline, spline_knots};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let spline = BSpline::new(&x, None, None, 3, false, None)?;
/// let knots = spline_knots(&spline);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn spline_knots<T: Float, S: Spline<T>>(spline: &S) -> &Array1<T> {
    spline.knots()
}

/// Extract the order of a spline object
///
/// This function is equivalent to R's `splineOrder()` function.
/// It returns the order (degree + 1) of the spline.
///
/// # Arguments
/// * `spline` - A spline object that implements the Spline trait
///
/// # Returns
/// The order of the spline as usize
///
/// # Examples
/// ```rust
/// use splines::{BSpline, spline_order};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let spline = BSpline::new(&x, None, None, 3, false, None)?;
/// let order = spline_order(&spline);
/// assert_eq!(order, 4); // degree 3 = order 4
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn spline_order<T: Float, S: Spline<T>>(spline: &S) -> usize {
    spline.order()
}

/// Convert a spline object to its vector representation
///
/// This function is similar to R's `asVector()` function.
/// For different spline types, it returns different vector representations:
/// - BSpline: returns the knot vector
/// - NaturalSpline: returns the knot vector
/// - InterpolationSpline: returns the x values used for interpolation
///
/// # Arguments
/// * `spline` - A spline object
///
/// # Returns
/// A cloned Array1 representing the spline's vector form
///
/// # Examples
/// ```rust
/// use splines::{BSpline, as_vector};
///
/// let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let spline = BSpline::new(&x, None, None, 3, false, None)?;
/// let vector = as_vector(&spline);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn as_vector<T: Float + Clone + Sum>(spline: &BSpline<T>) -> Array1<T> {
    spline.knots().clone()
}

/// Convert a NaturalSpline object to its vector representation
pub fn as_vector_natural<T: Float + Clone + Sum>(spline: &NaturalSpline<T>) -> Array1<T> {
    spline.knots().clone()
}

/// Convert an InterpolationSpline object to its vector representation
pub fn as_vector_interpolation<T: Float + Clone + Sum>(
    spline: &InterpolationSpline<T>,
) -> Array1<T> {
    // For interpolation splines, return the x values used for interpolation
    spline.x_values().clone()
}

/// A wrapper function for splineDesign functionality
///
/// This function is equivalent to R's `spline.des()` function.
/// It provides a convenient wrapper around the design matrix functionality.
///
/// # Arguments
/// * `knots` - The knot vector
/// * `x` - The x values at which to evaluate the spline basis
/// * `order` - The order of the spline (degree + 1)
/// * `derivs` - Optional derivative orders to compute
/// * `intercept` - Whether to include an intercept term
///
/// # Returns
/// A SplineResult containing the design matrix
///
/// # Examples
/// ```rust
/// use splines::spline_des;
/// use ndarray::Array1;
///
/// let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 2.0, 4.0, 4.0, 4.0, 4.0]);
/// let x = vec![1.0, 2.0, 3.0];
/// let matrix = spline_des(&knots, &x, 4, None, true)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn spline_des<T: Float + Clone>(
    knots: &Array1<T>,
    x: &[T],
    order: usize,
    derivs: Option<&[usize]>,
    intercept: bool,
) -> SplineResult<ndarray::Array2<T>> {
    use crate::basis::design_matrix::DesignMatrixBuilder;

    let builder = if intercept {
        DesignMatrixBuilder::new(knots.clone(), order)
    } else {
        DesignMatrixBuilder::new_no_intercept(knots.clone(), order)
    };

    match derivs {
        Some(deriv_orders) => {
            if deriv_orders.len() != x.len() {
                return Err(crate::errors::SplineError::dimension_mismatch(
                    x.len(),
                    deriv_orders.len(),
                ));
            }
            builder.build_with_derivatives(x, deriv_orders)
        }
        None => builder.build(x),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BSpline;

    #[test]
    fn test_spline_knots_bspline() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = BSpline::new(&x, None, None, 3, false, None).unwrap();
        let knots = spline_knots(&spline);

        // Should return the full knot vector
        assert!(knots.len() > 0);
        assert_eq!(knots.len(), spline.knots().len());
    }

    #[test]
    fn test_spline_order_bspline() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = BSpline::new(&x, None, None, 3, false, None).unwrap();
        let order = spline_order(&spline);

        // Order should be degree + 1 = 3 + 1 = 4
        assert_eq!(order, 4);
    }

    #[test]
    fn test_as_vector_bspline() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let spline = BSpline::new(&x, None, None, 3, false, None).unwrap();
        let vector = as_vector(&spline);

        // Should return the knot vector
        assert!(vector.len() > 0);
        assert_eq!(vector.len(), spline.knots().len());
    }

    #[test]
    fn test_spline_des_basic() {
        let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
        let x = vec![0.5, 1.5, 2.5];
        let matrix = spline_des(&knots, &x, 4, None, true).unwrap();

        // Should return a design matrix with correct dimensions
        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn test_spline_des_with_derivatives() {
        let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
        let x = vec![0.5, 1.5, 2.5];
        let derivs = vec![0, 1, 2];
        let matrix = spline_des(&knots, &x, 4, Some(&derivs), true).unwrap();

        // Should return a design matrix with derivatives
        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);
    }

    #[test]
    fn test_spline_des_no_intercept() {
        let knots = Array1::from(vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0]);
        let x = vec![0.5, 1.5, 2.5];
        let matrix = spline_des(&knots, &x, 4, None, false).unwrap();

        // Should return a design matrix without intercept
        assert_eq!(matrix.nrows(), 3);
        assert!(matrix.ncols() > 0);
    }
}
