//! Error types for fmridesign.
//!
//! This module provides comprehensive error handling for all fMRI design operations,
//! mirroring the error patterns from the R fmridesign package's `stop()` calls.

use thiserror::Error;

/// Main error type for fmridesign operations.
///
/// This enum covers all error cases encountered in fMRI design matrix construction,
/// including validation, dimension mismatches, contrast errors, and more.
#[derive(Error, Debug)]
pub enum FmriError {
    // ============================================================================
    // Validation Errors
    // ============================================================================
    /// Generic validation error
    #[error("Validation error: {0}")]
    Validation(#[from] crate::types::ValidationError),

    /// Dimension mismatch error with context
    #[error("Dimension mismatch for {context}: got {actual}, expected {expected}")]
    DimensionMismatch {
        context: String,
        actual: usize,
        expected: usize,
    },

    /// Length mismatch error
    #[error("Length mismatch for {context}: got {actual}, expected {expected}")]
    LengthMismatch {
        context: String,
        actual: usize,
        expected: usize,
    },

    /// Empty input error
    #[error("{context} cannot be empty")]
    EmptyInput { context: String },

    /// Invalid parameter value
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },

    /// Missing required field
    #[error("Missing required field: {field}")]
    MissingField { field: String },

    // ============================================================================
    // Event System Errors
    // ============================================================================
    /// Event not found
    #[error("Event '{name}' not found in event table")]
    EventNotFound { name: String },

    /// Invalid event type
    #[error("Invalid event type for {context}: expected {expected}, got {actual}")]
    InvalidEventType {
        context: String,
        expected: String,
        actual: String,
    },

    /// Mismatched event lengths
    #[error("Event lengths must match: {details}")]
    EventLengthMismatch { details: String },

    /// Invalid onset times
    #[error("Invalid onset times: {reason}")]
    InvalidOnsets { reason: String },

    /// Invalid duration values
    #[error("Invalid duration values: {reason}")]
    InvalidDurations { reason: String },

    // ============================================================================
    // Basis Function Errors
    // ============================================================================
    /// Invalid basis specification
    #[error("Invalid basis specification for {context}: {reason}")]
    InvalidBasis { context: String, reason: String },

    /// Basis evaluation error
    #[error("Basis evaluation failed at x={x}: {reason}")]
    BasisEvaluation { x: f64, reason: String },

    /// Invalid degree for polynomial
    #[error("Invalid degree {degree} for {context}: must be >= {min}")]
    InvalidDegree {
        context: String,
        degree: usize,
        min: usize,
    },

    /// Invalid knot specification
    #[error("Invalid knots for B-spline: {reason}")]
    InvalidKnots { reason: String },

    // ============================================================================
    // Contrast Errors
    // ============================================================================
    /// Contrast specification error
    #[error("Invalid contrast specification: {reason}")]
    InvalidContrast { reason: String },

    /// Contrast levels not found
    #[error("Contrast levels {levels:?} not found in condition '{condition}'")]
    ContrastLevelsNotFound {
        levels: Vec<String>,
        condition: String,
    },

    /// Contrast weight mismatch
    #[error("Contrast weights length {actual} does not match design matrix columns {expected}")]
    ContrastWeightMismatch { actual: usize, expected: usize },

    /// Collinearity detected
    #[error("Collinearity detected between columns: {columns:?}")]
    Collinearity { columns: Vec<String> },

    /// Invalid contrast formula
    #[error("Invalid contrast formula: {reason}")]
    InvalidContrastFormula { reason: String },

    // ============================================================================
    // Convolution Errors
    // ============================================================================
    /// HRF error from fmrihrf
    #[error("HRF error: {0}")]
    Hrf(#[from] fmrihrf::HrfError),

    /// Convolution failed
    #[error("Convolution failed for {context}: {reason}")]
    ConvolutionFailed { context: String, reason: String },

    /// Sampling frame mismatch
    #[error("Sampling frame mismatch: {reason}")]
    SamplingFrameMismatch { reason: String },

    /// Invalid TR (repetition time)
    #[error("Invalid TR value {tr}: must be positive")]
    InvalidTR { tr: f64 },

    // ============================================================================
    // Design Matrix Errors
    // ============================================================================
    /// Column name conflict
    #[error("Column name conflict: '{name}' appears multiple times")]
    ColumnNameConflict { name: String },

    /// Column not found
    #[error("Column '{name}' not found in design matrix")]
    ColumnNotFound { name: String },

    /// Matrix construction error
    #[error("Design matrix construction failed: {reason}")]
    MatrixConstruction { reason: String },

    /// Rank deficiency
    #[error("Design matrix is rank deficient: rank {rank} < {ncol} columns")]
    RankDeficient { rank: usize, ncol: usize },

    // ============================================================================
    // Formula Parsing Errors
    // ============================================================================
    /// Formula parse error
    #[error("Failed to parse formula '{formula}': {reason}")]
    FormulaParse { formula: String, reason: String },

    /// Invalid formula syntax
    #[error("Invalid formula syntax at position {pos}: {reason}")]
    InvalidFormulaSyntax { pos: usize, reason: String },

    /// Undefined variable in formula
    #[error("Undefined variable '{var}' in formula")]
    UndefinedVariable { var: String },

    // ============================================================================
    // I/O and External Errors
    // ============================================================================
    /// IO error
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Polars DataFrame error
    #[error("DataFrame error: {0}")]
    DataFrame(#[from] polars::error::PolarsError),

    // ============================================================================
    // Baseline Model Errors
    // ============================================================================
    /// Invalid baseline specification
    #[error("Invalid baseline specification: {reason}")]
    InvalidBaseline { reason: String },

    /// Block specification error
    #[error("Invalid block specification: {reason}")]
    InvalidBlock { reason: String },

    /// Nuisance regressor error
    #[error("Invalid nuisance regressor '{name}': {reason}")]
    InvalidNuisance { name: String, reason: String },

    // ============================================================================
    // Miscellaneous Errors
    // ============================================================================
    /// Not implemented error (for features still being ported)
    #[error("Feature not yet implemented: {feature}")]
    NotImplemented { feature: String },

    /// Internal error (should not occur in normal operation)
    #[error("Internal error (please report): {details}")]
    Internal { details: String },

    /// Other errors
    #[error("{0}")]
    Other(String),
}

/// Convenience result type alias for fmridesign operations.
pub type Result<T> = std::result::Result<T, FmriError>;

/// Explicit alias to make the public name unambiguous at call sites.
pub type FmriResult<T> = std::result::Result<T, FmriError>;

impl FmriError {
    /// Create a dimension mismatch error
    pub fn dimension_mismatch(context: impl Into<String>, actual: usize, expected: usize) -> Self {
        Self::DimensionMismatch {
            context: context.into(),
            actual,
            expected,
        }
    }

    /// Create a length mismatch error
    pub fn length_mismatch(context: impl Into<String>, actual: usize, expected: usize) -> Self {
        Self::LengthMismatch {
            context: context.into(),
            actual,
            expected,
        }
    }

    /// Create an empty input error
    pub fn empty_input(context: impl Into<String>) -> Self {
        Self::EmptyInput {
            context: context.into(),
        }
    }

    /// Create an invalid parameter error
    pub fn invalid_parameter(param: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            param: param.into(),
            reason: reason.into(),
        }
    }

    /// Create a not implemented error
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
        }
    }
}
