//! Core traits for spline types

use crate::errors::SplineResult;
use crate::types::XYVector;
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// Types of splines supported
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SplineType {
    /// B-spline basis
    BSpline,
    /// Natural cubic splines
    NaturalSpline,
    /// Interpolation splines
    Interpolation,
    /// Periodic splines
    Periodic,
    /// Polynomial representation
    Polynomial,
}

/// Core trait for all spline types
pub trait Spline<T: Float> {
    /// Get the knot positions
    fn knots(&self) -> &Array1<T>;

    /// Get the spline order (degree + 1)
    fn order(&self) -> usize;

    /// Get the spline degree  
    fn degree(&self) -> usize {
        self.order() - 1
    }

    /// Get the spline type
    fn spline_type(&self) -> SplineType;

    /// Predict spline values at given x points
    fn predict(&self, x: &[T]) -> SplineResult<XYVector<T>>;

    /// Predict with derivatives
    fn predict_with_deriv(&self, x: &[T], deriv: usize) -> SplineResult<XYVector<T>>;

    /// Evaluate the spline at a single point
    fn evaluate(&self, x: T) -> SplineResult<T>;

    /// Evaluate with derivative at a single point
    fn evaluate_with_deriv(&self, x: T, deriv: usize) -> SplineResult<T>;

    /// Get the valid range for evaluation
    fn valid_range(&self) -> (T, T);

    /// Check if a point is in the valid range
    fn in_range(&self, x: T) -> bool {
        let (min, max) = self.valid_range();
        x >= min && x <= max
    }
}

/// Trait for splines that can be constructed from data
pub trait SplineBuilder<T: Float> {
    /// The output spline type produced by the builder
    type Output: Spline<T>;

    /// Build a spline from input data
    fn build(self) -> SplineResult<Self::Output>;
}

/// Trait for splines that can be converted to polynomial representation
pub trait ToPolynomial<T: Float> {
    /// The polynomial spline representation type
    type PolyOutput;

    /// Convert to polynomial spline representation
    fn to_polynomial(&self) -> SplineResult<Self::PolyOutput>;
}

/// Trait for splines that support basis matrix generation
pub trait BasisMatrix<T: Float> {
    /// Generate design/basis matrix for given x values
    fn basis_matrix(&self, x: &[T]) -> SplineResult<Array2<T>>;

    /// Generate basis matrix with derivatives
    fn basis_matrix_with_deriv(&self, x: &[T], derivs: &[usize]) -> SplineResult<Array2<T>>;
}
