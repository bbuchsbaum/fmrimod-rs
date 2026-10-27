//! Error types for spline operations

use thiserror::Error;

/// Main error type for spline operations
#[derive(Error, Debug)]
pub enum SplineError {
    /// Invalid knot configuration
    #[error("Invalid knots: {0}")]
    InvalidKnots(String),

    /// Invalid degree specification
    #[error("Invalid degree: {0}. Degree must be >= 1")]
    InvalidDegree(i32),

    /// Dimension mismatch between inputs
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Invalid boundary knots
    #[error("Invalid boundary knots: {0}")]
    InvalidBoundaryKnots(String),

    /// Insufficient data points
    #[error("Insufficient data points: need at least {min}, got {actual}")]
    InsufficientData {
        /// Minimum required data points
        min: usize,
        /// Actual number of data points provided
        actual: usize,
    },

    /// Data points not in required range
    #[error("Data out of range: x values must be in range [{min}, {max}]")]
    DataOutOfRange {
        /// Minimum allowed value
        min: f64,
        /// Maximum allowed value
        max: f64,
    },

    /// Linear algebra error
    #[error("Linear algebra error: {0}")]
    LinearAlgebra(String),

    /// Invalid derivative order
    #[error("Invalid derivative order: {order}. Must be between 0 and {max}")]
    InvalidDerivative {
        /// The invalid derivative order requested
        order: i32,
        /// The maximum allowed derivative order
        max: i32,
    },

    /// Non-monotonic data for interpolation
    #[error("Non-monotonic data: x values must be strictly increasing")]
    NonMonotonic,

    /// Period-related error for periodic splines
    #[error("Period error: {0}")]
    PeriodError(String),

    /// General computation error
    #[error("Computation error: {0}")]
    ComputationError(String),
}

/// Result type for spline operations
pub type SplineResult<T> = Result<T, SplineError>;

impl SplineError {
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch(format!("expected {}, got {}", expected, actual))
    }

    /// Create an insufficient data error
    pub fn insufficient_data(min: usize, actual: usize) -> Self {
        Self::InsufficientData { min, actual }
    }

    /// Create a data out of range error  
    pub fn data_out_of_range(min: f64, max: f64) -> Self {
        Self::DataOutOfRange { min, max }
    }

    /// Create an invalid derivative error
    pub fn invalid_derivative(order: i32, max: i32) -> Self {
        Self::InvalidDerivative { order, max }
    }
}
