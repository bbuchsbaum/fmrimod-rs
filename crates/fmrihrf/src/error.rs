//! Error types for the fmrihrf library.

use thiserror::Error;

/// Main error type for HRF operations.
#[derive(Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum HrfError {
    /// Invalid time points provided.
    #[error("Invalid time points: {0}")]
    InvalidTimePoints(String),

    /// Invalid parameters for HRF function.
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),

    /// Numerical computation error.
    #[error("Numerical error: {0}")]
    NumericalError(String),

    /// Invalid dimensions for array operations.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimensions or shape
        expected: String,
        /// Actual dimensions or shape encountered
        actual: String,
    },

    /// Invalid input values.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// HRF not found in registry.
    #[error("HRF '{0}' not found in registry")]
    HrfNotFound(String),

    /// Linear algebra operation failed.
    #[error("Linear algebra error: {0}")]
    LinAlgError(String),

    /// Plotting error.
    #[error("Plotting error: {0}")]
    PlottingError(String),

    /// IO error.
    #[error("IO error: {0}")]
    #[cfg_attr(feature = "serde", serde(with = "io_error_string"))]
    IoError(#[from] std::io::Error),
}

/// Alias for Result with HrfError.
pub type Result<T> = std::result::Result<T, HrfError>;

// Implement conversions from common error types
impl From<ndarray::ShapeError> for HrfError {
    fn from(err: ndarray::ShapeError) -> Self {
        HrfError::DimensionMismatch {
            expected: "valid shape".to_string(),
            actual: err.to_string(),
        }
    }
}

/// Custom serialization for IO errors.
#[cfg(feature = "serde")]
mod io_error_string {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::io;

    pub fn serialize<S>(error: &io::Error, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        error.to_string().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<io::Error, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(io::Error::other(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = HrfError::InvalidTimePoints("negative time".to_string());
        assert_eq!(err.to_string(), "Invalid time points: negative time");

        let err = HrfError::InvalidParameters("bad shape".to_string());
        assert_eq!(err.to_string(), "Invalid parameters: bad shape");

        let err = HrfError::NumericalError("overflow".to_string());
        assert_eq!(err.to_string(), "Numerical error: overflow");

        let err = HrfError::DimensionMismatch {
            expected: "(100, 3)".to_string(),
            actual: "(100, 2)".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "Dimension mismatch: expected (100, 3), got (100, 2)"
        );

        let err = HrfError::InvalidInput("empty vector".to_string());
        assert_eq!(err.to_string(), "Invalid input: empty vector");

        let err = HrfError::HrfNotFound("custom_hrf".to_string());
        assert_eq!(err.to_string(), "HRF 'custom_hrf' not found in registry");

        let err = HrfError::LinAlgError("singular matrix".to_string());
        assert_eq!(err.to_string(), "Linear algebra error: singular matrix");

        let err = HrfError::PlottingError("no backend".to_string());
        assert_eq!(err.to_string(), "Plotting error: no backend");
    }

    #[test]
    fn test_error_from_ndarray_shape() {
        use ndarray::ShapeBuilder;
        let _shape = (2, 3).strides((1, 2));
        let err = ndarray::ShapeError::from_kind(ndarray::ErrorKind::IncompatibleShape);
        let hrf_err: HrfError = err.into();

        match hrf_err {
            HrfError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, "valid shape");
                assert!(actual.contains("IncompatibleShape"));
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_error_from_io() {
        use std::io;
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let hrf_err: HrfError = io_err.into();

        match hrf_err {
            HrfError::IoError(e) => {
                assert_eq!(e.kind(), io::ErrorKind::NotFound);
            }
            _ => panic!("Expected IoError"),
        }
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_error_serialization() {
        let err = HrfError::InvalidInput("test error".to_string());
        let json = serde_json::to_string(&err).unwrap();
        let err2: HrfError = serde_json::from_str(&json).unwrap();

        // Both errors should have the same display string
        assert_eq!(err.to_string(), err2.to_string());

        // Test IO error serialization
        let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
        let err = HrfError::IoError(io_err);
        let json = serde_json::to_string(&err).unwrap();
        let err2: HrfError = serde_json::from_str(&json).unwrap();

        match err2 {
            HrfError::IoError(e) => {
                // The deserialized error will have ErrorKind::Other but same message
                assert_eq!(e.to_string(), "access denied");
            }
            _ => panic!("Expected IoError after deserialization"),
        }
    }
}
