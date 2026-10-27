//! Type definitions and traits for fmri_design.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use thiserror::Error;

// Re-export commonly used types from dependencies
pub use crate::core::DataFrame;
pub use fmri_hrf::{Hrf, HrfExt, HrfValues, TimeArray};

/// Type alias for f64 arrays
pub type Array = Array2<f64>;
pub type Vector = Array1<f64>;
pub type ArrayView<'a> = ArrayView2<'a, f64>;
pub type VectorView<'a> = ArrayView1<'a, f64>;

/// Type alias for integer arrays
pub type IntArray = Array1<i32>;
pub type IntVector = Array1<i32>;

/// Type alias for boolean arrays
pub type BoolArray = Array1<bool>;
pub type BoolVector = Array1<bool>;

/// Time-related types
pub type TimeType = f64;
pub type OnsetArray = Vector;
pub type DurationArray = Vector;
pub type AmplitudeArray = Vector;

/// Event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Categorical event (e.g., different conditions)
    Categorical,
    /// Continuous event (e.g., parametric modulation)
    Continuous,
    /// Basis function expansion
    Basis,
    /// Direct matrix specification
    Matrix,
    /// Event term (combination/interaction)
    Term,
}

/// Contrast types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContrastType {
    /// Simple contrast (e.g., A vs baseline)
    Simple,
    /// Pairwise contrast (e.g., A vs B)
    Pair,
    /// Polynomial contrast
    Polynomial,
    /// Interaction contrast
    Interaction,
    /// F-contrast (multiple contrasts)
    F,
}

/// Validation error types
#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Cannot convert to array: {0}")]
    ConversionError(String),

    #[error("Array must be {expected}-dimensional, got {actual}D")]
    DimensionError { expected: usize, actual: usize },

    #[error("Array cannot be empty")]
    EmptyArray,

    #[error("Values cannot be negative")]
    NegativeValues,

    #[error("Values must be finite")]
    NonFiniteValues,

    #[error("Length mismatch: expected {expected}, got {actual}")]
    LengthMismatch { expected: usize, actual: usize },

    #[error("Invalid value: {0}")]
    InvalidValue(String),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("File system error: {0}")]
    FileSystemError(String),

    #[error("Invalid type: expected {expected}, got {actual}")]
    InvalidType { expected: String, actual: String },

    #[error("Missing column: {0}")]
    MissingColumn(String),

    #[error("Not implemented: {0}")]
    NotImplemented(String),
}

/// Core trait for event specifications
pub trait EventSpec: Debug + Send + Sync {
    /// Name of the event
    fn name(&self) -> &str;

    /// Type of event
    fn event_type(&self) -> EventType;

    /// Event onset times
    fn onsets(&self) -> &OnsetArray;

    /// Event durations
    fn durations(&self) -> &DurationArray;

    /// Number of events
    fn n_events(&self) -> usize {
        self.onsets().len()
    }

    /// Event amplitudes before temporal sampling: one row per event and one
    /// column per predictor. HRF convolution uses these values at exact onsets.
    /// Custom event types must implement this method to support convolution.
    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        Err(ValidationError::NotImplemented(format!(
            "event amplitudes for {}",
            self.name()
        )))
    }

    /// Generate design matrix columns for this event at given sampling points
    fn design_matrix(&self, sampling_points: &TimeArray) -> Result<Array, ValidationError>;

    /// Get levels for categorical events (optional)
    fn levels(&self) -> Option<&[String]> {
        None
    }

    /// Get values for events (optional, mainly for categorical)
    fn values(&self) -> Option<&[String]> {
        None
    }

    /// Get number of columns for matrix events (optional)
    fn n_columns(&self) -> Option<usize> {
        None
    }

    /// Get column names for the design matrix
    fn column_names(&self) -> Vec<String> {
        vec![self.name().to_string()]
    }

    /// Convert to DataFrame representation
    fn to_dataframe(&self) -> DataFrame;

    /// Get as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;

    /// Clone as boxed event
    fn clone_box(&self) -> Box<dyn EventSpec>;

    /// Filter events by boolean mask
    fn filter(&self, mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError>;
}

impl Clone for Box<dyn EventSpec> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Trait for basis function objects
pub trait BasisFunction: Debug + Send + Sync {
    /// Name of the basis
    fn name(&self) -> &str;

    /// Number of basis functions
    fn nbasis(&self) -> usize;

    /// Evaluate basis functions at given points
    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError>;

    /// Predict using basis functions with given coefficients
    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError>;

    /// Get the levels/identifiers for each basis function
    ///
    /// For continuous bases (polynomial, spline), returns zero-padded indices like ["01", "02", ...].
    /// For identity basis, returns variable names.
    /// For transform bases (Scale, etc.), returns the wrapped basis levels.
    ///
    /// Corresponds to R's `levels()` S3 method for basis objects.
    fn levels(&self) -> Vec<String> {
        // Default implementation: use the same zero-padding logic as the
        // R helper `zeropad()`, with `n_total` equal to the number of basis
        // functions. This keeps basis identifiers stable and predictable.
        let n = self.nbasis();
        (1..=n).map(|i| crate::naming::zeropad(i, n)).collect()
    }

    /// Get column tokens for naming purposes
    ///
    /// Returns sanitized identifiers used when constructing design matrix column names.
    /// By default, returns the same as levels(), but can be overridden for bases that
    /// need different naming behavior.
    ///
    /// Corresponds to R's `columns()` S3 method for basis objects.
    fn columns(&self) -> Vec<String> {
        // Default: same as levels
        self.levels()
    }

    /// Extract a sub-basis with specified indices
    ///
    /// Returns a new basis function containing only the specified basis columns.
    /// Useful for contrast construction and selective inclusion of basis terms.
    ///
    /// Corresponds to R's `sub_basis()` S3 method.
    fn sub_basis(&self, _indices: &[usize]) -> Result<Box<dyn BasisFunction>, ValidationError> {
        Err(ValidationError::InvalidParameter(format!(
            "sub_basis not implemented for {}",
            self.name()
        )))
    }
}

/// Trait for contrast specifications
pub trait ContrastSpec: Debug + Send + Sync {
    /// Name of the contrast
    fn name(&self) -> &str;

    /// Type of contrast
    fn contrast_type(&self) -> ContrastType;

    /// Compute contrast weights for given columns
    fn compute_weights(&self, columns: &[String]) -> Result<Vector, ValidationError>;
}

/// Trait for model objects
pub trait Model: Debug + Send + Sync {
    /// Get the design matrix
    fn design_matrix(&self) -> &Array;

    /// Get column names of design matrix
    fn column_names(&self) -> &[String];

    /// Add a contrast to the model
    fn add_contrast(&self, contrast: Box<dyn ContrastSpec>) -> Result<(), ValidationError>;
}

/// Trait for objects that can produce design matrices
pub trait DesignMatrixProvider: Debug + Send + Sync {
    /// Get the design matrix
    fn design_matrix(&self) -> &Array;

    /// Get column names
    fn column_names(&self) -> Option<&[String]> {
        None
    }
}

/// Formula context for variable resolution
#[derive(Debug, Clone)]
pub struct FormulaContext {
    /// Variables stored as name-value pairs
    variables: std::collections::HashMap<String, FormulaValue>,
}

/// Values that can be stored in a formula context
#[derive(Debug, Clone)]
pub enum FormulaValue {
    /// Scalar value
    Scalar(f64),
    /// Vector of values
    Vector(Vector),
    /// String value
    String(String),
    /// Vector of strings
    StringVector(Vec<String>),
}

impl FormulaContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            variables: std::collections::HashMap::new(),
        }
    }

    /// Add a variable to the context
    pub fn add_variable<S: Into<String>>(&mut self, name: S, value: FormulaValue) {
        self.variables.insert(name.into(), value);
    }

    /// Get a variable from the context
    pub fn get(&self, name: &str) -> Option<&FormulaValue> {
        self.variables.get(name)
    }

    /// Check if a variable exists
    pub fn contains(&self, name: &str) -> bool {
        self.variables.contains_key(name)
    }
}

impl Default for FormulaContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate onset times
pub fn validate_onsets(onsets: Vec<f64>) -> Result<OnsetArray, ValidationError> {
    if onsets.is_empty() {
        return Err(ValidationError::EmptyArray);
    }

    let array = Array1::from_vec(onsets);

    if array.iter().any(|&x| x < 0.0) {
        return Err(ValidationError::NegativeValues);
    }

    if array.iter().any(|&x| !x.is_finite()) {
        return Err(ValidationError::NonFiniteValues);
    }

    Ok(array)
}

/// Validate durations
pub fn validate_durations(
    durations: Vec<f64>,
    n_events: usize,
) -> Result<DurationArray, ValidationError> {
    if durations.len() != n_events {
        return Err(ValidationError::LengthMismatch {
            expected: n_events,
            actual: durations.len(),
        });
    }

    let array = Array1::from_vec(durations);

    if array.iter().any(|&x| x < 0.0) {
        return Err(ValidationError::NegativeValues);
    }

    if array.iter().any(|&x| !x.is_finite()) {
        return Err(ValidationError::NonFiniteValues);
    }

    Ok(array)
}

/// Create uniform durations array
pub fn uniform_durations(duration: f64, n_events: usize) -> Result<DurationArray, ValidationError> {
    if duration < 0.0 {
        return Err(ValidationError::NegativeValues);
    }

    if !duration.is_finite() {
        return Err(ValidationError::NonFiniteValues);
    }

    Ok(Array1::from_elem(n_events, duration))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_onsets() {
        // Valid onsets
        let onsets = vec![0.0, 1.0, 2.0];
        assert!(validate_onsets(onsets).is_ok());

        // Empty onsets
        let onsets = vec![];
        assert!(matches!(
            validate_onsets(onsets),
            Err(ValidationError::EmptyArray)
        ));

        // Negative onsets
        let onsets = vec![0.0, -1.0, 2.0];
        assert!(matches!(
            validate_onsets(onsets),
            Err(ValidationError::NegativeValues)
        ));

        // Non-finite onsets
        let onsets = vec![0.0, f64::NAN, 2.0];
        assert!(matches!(
            validate_onsets(onsets),
            Err(ValidationError::NonFiniteValues)
        ));
    }

    #[test]
    fn test_validate_durations() {
        // Valid durations
        let durations = vec![1.0, 1.0, 1.0];
        assert!(validate_durations(durations, 3).is_ok());

        // Length mismatch
        let durations = vec![1.0, 1.0];
        assert!(matches!(
            validate_durations(durations, 3),
            Err(ValidationError::LengthMismatch {
                expected: 3,
                actual: 2
            })
        ));

        // Negative durations
        let durations = vec![1.0, -1.0, 1.0];
        assert!(matches!(
            validate_durations(durations, 3),
            Err(ValidationError::NegativeValues)
        ));
    }

    #[test]
    fn test_uniform_durations() {
        let durations = uniform_durations(1.0, 5).unwrap();
        assert_eq!(durations.len(), 5);
        assert!(durations.iter().all(|&x| x == 1.0));
    }
}
