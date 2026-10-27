//! Basis subsetting functionality.
//!
//! This module provides functions for creating subsets of basis functions.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use ndarray::{s, Array2, ArrayView1};
use std::fmt::{self, Debug};
use std::sync::Arc;

/// A subset of basis functions from a parent basis.
///
/// This struct wraps an existing basis and provides access to only
/// a subset of its basis functions.
pub struct SubBasis {
    /// The parent basis to subset.
    parent: Arc<dyn BasisFunction>,
    /// Indices of basis functions to include.
    indices: Vec<usize>,
    /// Name for the sub-basis.
    name: String,
    /// Cached basis names (computed once).
    cached_names: Vec<String>,
}

impl SubBasis {
    /// Create a new SubBasis from a parent basis and indices.
    ///
    /// # Arguments
    /// * `parent` - The parent basis to subset
    /// * `indices` - Indices of basis functions to include
    ///
    /// # Examples
    /// ```ignore
    /// use fmridesign::basis::{PolynomialBasis, SubBasis};
    ///
    /// let poly = PolynomialBasis::new(5);
    /// let sub = SubBasis::new(Arc::new(poly), vec![0, 2, 4]).unwrap();
    /// ```
    pub fn new(
        parent: Arc<dyn BasisFunction>,
        indices: Vec<usize>,
    ) -> Result<Self, ValidationError> {
        if indices.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Must select at least one basis function".to_string(),
            ));
        }

        let parent_nbasis = parent.nbasis();

        // Validate indices
        for &idx in &indices {
            if idx >= parent_nbasis {
                return Err(ValidationError::InvalidParameter(format!(
                    "Index {} out of range for basis with {} functions",
                    idx, parent_nbasis
                )));
            }
        }

        // Generate name
        let indices_str = indices
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let name = format!("{}_sub[{}]", parent.name(), indices_str);

        // Cache basis names
        let cached_names = indices
            .iter()
            .map(|&i| format!("{}_basis_{}", parent.name(), i))
            .collect();

        Ok(Self {
            parent,
            indices,
            name,
            cached_names,
        })
    }

    /// Create a SubBasis from a slice specification.
    ///
    /// # Arguments
    /// * `parent` - The parent basis to subset
    /// * `start` - Starting index (inclusive)
    /// * `end` - Ending index (exclusive), None means to the end
    ///
    /// # Examples
    /// ```ignore
    /// // Select basis functions 1 through 3 (indices 1, 2, 3)
    /// let sub = SubBasis::from_slice(Arc::new(poly), 1, Some(4)).unwrap();
    ///
    /// // Select all but the first (remove intercept)
    /// let no_intercept = SubBasis::from_slice(Arc::new(poly), 1, None).unwrap();
    /// ```
    pub fn from_slice(
        parent: Arc<dyn BasisFunction>,
        start: usize,
        end: Option<usize>,
    ) -> Result<Self, ValidationError> {
        let parent_nbasis = parent.nbasis();
        let end = end.unwrap_or(parent_nbasis);

        if start >= parent_nbasis {
            return Err(ValidationError::InvalidParameter(format!(
                "Start index {} out of range for basis with {} functions",
                start, parent_nbasis
            )));
        }

        if end > parent_nbasis {
            return Err(ValidationError::InvalidParameter(format!(
                "End index {} out of range for basis with {} functions",
                end, parent_nbasis
            )));
        }

        if start >= end {
            return Err(ValidationError::InvalidParameter(format!(
                "Invalid slice: start {} >= end {}",
                start, end
            )));
        }

        let indices: Vec<usize> = (start..end).collect();
        Self::new(parent, indices)
    }

    /// Create a SubBasis with custom name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Get the selected indices.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Get reference to the parent basis.
    pub fn parent(&self) -> &Arc<dyn BasisFunction> {
        &self.parent
    }
}

impl BasisFunction for SubBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        self.indices.len()
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        // Get full basis evaluation from parent
        let full_basis = self.parent.evaluate(x)?;

        // Create output array
        let n = x.len();
        let n_selected = self.indices.len();
        let mut result = Array2::zeros((n, n_selected));

        // Copy selected columns
        for (j, &idx) in self.indices.iter().enumerate() {
            result
                .slice_mut(s![.., j])
                .assign(&full_basis.slice(s![.., idx]));
        }

        Ok(result)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        let basis = self.evaluate(x)?;
        if let Some(c) = coef {
            if c.len() != self.nbasis() {
                return Err(ValidationError::LengthMismatch {
                    expected: self.nbasis(),
                    actual: c.len(),
                });
            }
            Ok(basis.dot(c))
        } else {
            // Return first basis function if no coefficients provided
            Ok(basis.column(0).to_owned())
        }
    }
}

// Custom Debug implementation to avoid printing the entire parent
impl Debug for SubBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SubBasis")
            .field("parent", &self.parent.name())
            .field("indices", &self.indices)
            .field("n_basis", &self.nbasis())
            .finish()
    }
}

// Need to implement basis_names() method
impl SubBasis {
    /// Get the basis function names for the subset.
    pub fn basis_names(&self) -> Vec<String> {
        self.cached_names.clone()
    }
}

/// Create a subset of basis functions.
///
/// This function creates a new basis object that contains only
/// a subset of the original basis functions.
///
/// # Arguments
/// * `basis` - The basis to subset
/// * `indices` - Indices of basis functions to include
///
/// # Examples
/// ```ignore
/// use fmridesign::basis::{PolynomialBasis, sub_basis};
/// use std::sync::Arc;
///
/// // Select specific polynomial terms
/// let poly = PolynomialBasis::new(5);
/// let linear_cubic = sub_basis(Arc::new(poly), vec![1, 3]).unwrap(); // x and x^3 terms
/// ```
pub fn sub_basis(
    basis: Arc<dyn BasisFunction>,
    indices: Vec<usize>,
) -> Result<SubBasis, ValidationError> {
    SubBasis::new(basis, indices)
}

/// Create a subset of basis functions from a slice.
///
/// # Arguments
/// * `basis` - The basis to subset
/// * `start` - Starting index (inclusive)
/// * `end` - Ending index (exclusive), None means to the end
///
/// # Examples
/// ```ignore
/// use fmridesign::basis::{PolynomialBasis, sub_basis_slice};
/// use std::sync::Arc;
///
/// let poly = PolynomialBasis::new(5);
///
/// // Select all but the intercept
/// let no_intercept = sub_basis_slice(Arc::new(poly), 1, None).unwrap();
/// ```
pub fn sub_basis_slice(
    basis: Arc<dyn BasisFunction>,
    start: usize,
    end: Option<usize>,
) -> Result<SubBasis, ValidationError> {
    SubBasis::from_slice(basis, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::PolynomialBasis;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_sub_basis_creation() {
        let poly = PolynomialBasis::new(5);
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        let sub = SubBasis::new(poly_arc.clone(), vec![0, 2, 4]).unwrap();

        assert_eq!(sub.nbasis(), 3);
        assert_eq!(sub.indices(), &[0, 2, 4]);
    }

    #[test]
    fn test_sub_basis_from_slice() {
        let poly = PolynomialBasis::new(5).with_intercept(true); // Has 6 basis functions (0-5)
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        // Select functions 1 through 3
        let sub = SubBasis::from_slice(poly_arc.clone(), 1, Some(4)).unwrap();
        assert_eq!(sub.nbasis(), 3);
        assert_eq!(sub.indices(), &[1, 2, 3]);

        // Select all but first
        let no_intercept = SubBasis::from_slice(poly_arc.clone(), 1, None).unwrap();
        assert_eq!(no_intercept.nbasis(), 5); // degree 5 polynomial has 6 terms total
        assert_eq!(no_intercept.indices(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_sub_basis_evaluate() {
        let poly = PolynomialBasis::new(4).with_intercept(true).with_raw(true); // 1, x, x^2, x^3, x^4
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        // Select linear and cubic terms (indices 1 and 3)
        let sub = SubBasis::new(poly_arc.clone(), vec![1, 3]).unwrap();

        let x = Array1::from(vec![0.0, 1.0, 2.0]);
        let result = sub.evaluate(&x.view()).unwrap();

        assert_eq!(result.shape(), &[3, 2]);

        // Check values: column 0 should be x, column 1 should be x^3
        assert_eq!(result[[0, 0]], 0.0); // x=0
        assert_eq!(result[[0, 1]], 0.0); // x^3=0

        assert_eq!(result[[1, 0]], 1.0); // x=1
        assert_eq!(result[[1, 1]], 1.0); // x^3=1

        assert_eq!(result[[2, 0]], 2.0); // x=2
        assert_eq!(result[[2, 1]], 8.0); // x^3=8
    }

    #[test]
    fn test_sub_basis_empty_indices() {
        let poly = PolynomialBasis::new(3);
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        let result = SubBasis::new(poly_arc, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_basis_out_of_range() {
        let poly = PolynomialBasis::new(3); // Has 3 basis functions (0-2) without intercept
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        let result = SubBasis::new(poly_arc, vec![0, 1, 5]); // 5 is out of range
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_basis_predict() {
        let poly = PolynomialBasis::new(3).with_intercept(true).with_raw(true); // Has intercept + x + x^2 + x^3
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        // Select indices 0 and 2 (intercept and x^2)
        let sub = SubBasis::new(poly_arc, vec![0, 2]).unwrap();

        let x = Array1::from(vec![1.0, 2.0, 3.0]);
        let coef = Array1::from(vec![1.0, 0.5]); // 1.0 * intercept + 0.5 * x^2

        let pred = sub.predict(&x.view(), Some(&coef.view())).unwrap();

        // Expected: 1.0 + 0.5*x^2
        assert_relative_eq!(pred[0], 1.0 + 0.5 * 1.0, epsilon = 1e-10);
        assert_relative_eq!(pred[1], 1.0 + 0.5 * 4.0, epsilon = 1e-10);
        assert_relative_eq!(pred[2], 1.0 + 0.5 * 9.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sub_basis_names() {
        let poly = PolynomialBasis::new(3).with_intercept(true);
        let poly_arc = Arc::new(poly) as Arc<dyn BasisFunction>;

        let sub = SubBasis::new(poly_arc, vec![0, 2]).unwrap();
        let names = sub.basis_names();

        assert_eq!(names.len(), 2);
        // Check the generated names
        assert!(names[0].contains("poly3_basis_0"));
        assert!(names[1].contains("poly3_basis_2"));
    }
}
