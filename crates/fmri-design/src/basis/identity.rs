//! Identity basis function.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use ndarray::{Array2, ArrayView1};

/// Identity basis function (no transformation).
#[derive(Debug, Clone)]
pub struct IdentityBasis {
    name: String,
}

impl IdentityBasis {
    /// Create new identity basis.
    pub fn new() -> Self {
        Self {
            name: "identity".to_string(),
        }
    }
}

impl Default for IdentityBasis {
    fn default() -> Self {
        Self::new()
    }
}

impl BasisFunction for IdentityBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        // Identity transformation - return input as single column
        let n = x.len();
        let mut result = Array2::zeros((n, 1));
        result.column_mut(0).assign(x);
        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        if let Some(c) = coef {
            if c.len() != 1 {
                return Err(ValidationError::LengthMismatch {
                    expected: 1,
                    actual: c.len(),
                });
            }
            Ok(x * c[0])
        } else {
            Ok(x.to_owned())
        }
    }
}
