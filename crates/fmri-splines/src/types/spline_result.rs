//! Result types for spline operations

use crate::errors::{SplineError, SplineResult};
use crate::types::XYVector;
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

/// Result of spline evaluation/prediction
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SplineOutput<T: Float> {
    /// Single value result
    Value(T),
    /// Vector of values  
    Vector(Array1<T>),
    /// XY paired result (common for predictions)
    XYVector(XYVector<T>),
    /// Matrix result (for basis functions)
    Matrix(Array2<T>),
}

impl<T: Float> SplineOutput<T> {
    /// Extract as a single value if possible
    pub fn as_value(&self) -> SplineResult<T> {
        match self {
            SplineOutput::Value(v) => Ok(*v),
            _ => Err(SplineError::ComputationError(
                "Expected single value result".to_string(),
            )),
        }
    }

    /// Extract as a vector if possible
    pub fn as_vector(&self) -> SplineResult<&Array1<T>> {
        match self {
            SplineOutput::Vector(v) => Ok(v),
            SplineOutput::XYVector(xy) => Ok(xy.y_array()),
            _ => Err(SplineError::ComputationError(
                "Expected vector result".to_string(),
            )),
        }
    }

    /// Extract as XYVector if possible
    pub fn as_xy_vector(&self) -> SplineResult<&XYVector<T>> {
        match self {
            SplineOutput::XYVector(xy) => Ok(xy),
            _ => Err(SplineError::ComputationError(
                "Expected XYVector result".to_string(),
            )),
        }
    }

    /// Extract as matrix if possible  
    pub fn as_matrix(&self) -> SplineResult<&Array2<T>> {
        match self {
            SplineOutput::Matrix(m) => Ok(m),
            _ => Err(SplineError::ComputationError(
                "Expected matrix result".to_string(),
            )),
        }
    }
}

/// Spline configuration parameters
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineConfig<T: Float> {
    /// Degrees of freedom (if specified)
    pub df: Option<usize>,
    /// Interior knot positions
    pub knots: Option<Array1<T>>,
    /// Polynomial degree
    pub degree: usize,
    /// Whether to include intercept
    pub intercept: bool,
    /// Boundary knots
    pub boundary_knots: Option<(T, T)>,
}

impl<T: Float> SplineConfig<T> {
    /// Create a new spline configuration
    pub fn new(degree: usize) -> Self {
        SplineConfig {
            df: None,
            knots: None,
            degree,
            intercept: false,
            boundary_knots: None,
        }
    }

    /// Set degrees of freedom
    pub fn with_df(mut self, df: usize) -> Self {
        self.df = Some(df);
        self
    }

    /// Set knots
    pub fn with_knots(mut self, knots: Array1<T>) -> Self {
        self.knots = Some(knots);
        self
    }

    /// Set intercept flag
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Set boundary knots
    pub fn with_boundary_knots(mut self, min: T, max: T) -> Self {
        self.boundary_knots = Some((min, max));
        self
    }

    /// Get the spline order (degree + 1)
    pub fn order(&self) -> usize {
        self.degree + 1
    }
}
