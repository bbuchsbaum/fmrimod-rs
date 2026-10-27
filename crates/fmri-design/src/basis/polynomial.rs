//! Polynomial basis functions.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use nalgebra::DMatrix;
use ndarray::{Array1, Array2, ArrayView1};

/// Polynomial basis functions.
#[derive(Debug, Clone)]
pub struct PolynomialBasis {
    name: String,
    degree: usize,
    intercept: bool,
    raw: bool,
}

impl PolynomialBasis {
    /// Create new polynomial basis of given degree.
    pub fn new(degree: usize) -> Self {
        Self {
            name: format!("poly{}", degree),
            degree,
            intercept: false, // No intercept by default, consistent with R
            raw: false,
        }
    }

    /// Create polynomial basis with specified options.
    pub fn with_options(degree: usize, intercept: bool, raw: bool) -> Self {
        Self {
            name: format!("poly{}", degree),
            degree,
            intercept,
            raw,
        }
    }

    /// Set whether to include intercept term.
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Set whether to use raw polynomials (vs orthogonal).
    pub fn with_raw(mut self, raw: bool) -> Self {
        self.raw = raw;
        self
    }

    /// Set custom name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Compatibility alias used by older tests/examples.
    pub fn design_matrix(&self, x: &Array1<f64>) -> Result<Array, ValidationError> {
        self.evaluate(&x.view())
    }

    /// Generate orthogonal polynomials using QR decomposition.
    fn orthogonal_polynomials(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        let n = x.len();
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::InvalidParameter(
                "polynomial input must be finite".into(),
            ));
        }
        let mut unique = x.to_vec();
        unique.sort_by(f64::total_cmp);
        unique.dedup();
        if self.degree >= unique.len() {
            return Err(ValidationError::InvalidParameter(
                "orthogonal polynomial degree must be less than the number of distinct values"
                    .into(),
            ));
        }
        // Include the constant during orthogonalization even when it is omitted
        // from the returned basis (the stats::poly contract).
        let n_basis = self.degree + 1;
        let mut raw_basis = Array2::zeros((n, n_basis));
        raw_basis.column_mut(0).fill(1.0);
        let mut col = 1;

        // Center x for numerical stability
        let x_mean = x.mean().unwrap_or(0.0);
        let x_centered = x.mapv(|v| v - x_mean);
        let x_std = x_centered.std(0.0);
        let x_scale = if x_std > 0.0 { x_std } else { 1.0 };
        let x_norm = x_centered.mapv(|v| v / x_scale);

        // Generate polynomial terms
        for d in 1..=self.degree {
            for i in 0..n {
                raw_basis[[i, col]] = x_norm[i].powi(d as i32);
            }
            col += 1;
        }

        // QR decomposition for orthogonalization using nalgebra
        let na_mat = DMatrix::from_fn(n, n_basis, |i, j| raw_basis[[i, j]]);
        let qr = na_mat.qr();
        let q_na = qr.q();
        let r_na = qr.r();

        let first = usize::from(!self.intercept);
        let mut result = Array2::zeros((n, self.nbasis()));
        for j in first..n_basis {
            let sign = if r_na[(j, j)] < 0.0 { -1.0 } else { 1.0 };
            for i in 0..n {
                result[[i, j - first]] = q_na[(i, j)] * sign;
            }
        }
        Ok(result)
    }
}

impl BasisFunction for PolynomialBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        if self.intercept {
            self.degree + 1
        } else {
            self.degree
        }
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if self.degree < 1 {
            return Err(ValidationError::InvalidParameter(
                "Polynomial degree must be at least 1".to_string(),
            ));
        }

        if self.raw {
            // Raw polynomials
            let n = x.len();
            let n_basis = self.nbasis();
            let mut result = Array2::zeros((n, n_basis));

            let mut col = 0;

            if self.intercept {
                result.column_mut(col).fill(1.0);
                col += 1;
            }

            for d in 1..=self.degree {
                for i in 0..n {
                    result[[i, col]] = x[i].powi(d as i32);
                }
                col += 1;
            }

            Ok(result)
        } else {
            // Orthogonal polynomials
            self.orthogonal_polynomials(x)
        }
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
            // Return first basis function if no coefficients
            Ok(basis.column(0).to_owned())
        }
    }
}

/// Natural polynomial basis (centered at mean).
#[derive(Debug)]
pub struct NaturalPolynomialBasis {
    name: String,
    degree: usize,
    intercept: bool,
    x_mean: std::sync::RwLock<Option<f64>>,
}

impl NaturalPolynomialBasis {
    /// Create new natural polynomial basis of given degree.
    pub fn new(degree: usize) -> Self {
        Self {
            name: format!("npoly{}", degree),
            degree,
            intercept: true,
            x_mean: std::sync::RwLock::new(None),
        }
    }

    /// Set whether to include intercept term.
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self
    }

    /// Set custom name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Compatibility alias used by older tests/examples.
    pub fn design_matrix(&self, x: &Array1<f64>) -> Result<Array, ValidationError> {
        self.evaluate(&x.view())
    }
}

impl Clone for NaturalPolynomialBasis {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            degree: self.degree,
            intercept: self.intercept,
            // Don't clone the stored mean - each clone gets its own
            x_mean: std::sync::RwLock::new(None),
        }
    }
}

impl BasisFunction for NaturalPolynomialBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        if self.intercept {
            self.degree + 1
        } else {
            self.degree
        }
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if self.degree < 1 {
            return Err(ValidationError::InvalidParameter(
                "Polynomial degree must be at least 1".to_string(),
            ));
        }

        // Get or compute and store the mean
        let x_mean = {
            let mut stored_mean = self.x_mean.write().unwrap();
            match *stored_mean {
                Some(mean) => mean,
                None => {
                    // First call - compute and store the mean
                    let mean = x.mean().unwrap_or(0.0);
                    *stored_mean = Some(mean);
                    mean
                }
            }
        };

        // Center x at the stored mean
        let x_centered = x.mapv(|v| v - x_mean);

        let n = x.len();
        let n_basis = self.nbasis();
        let mut result = Array2::zeros((n, n_basis));

        let mut col = 0;

        if self.intercept {
            result.column_mut(col).fill(1.0);
            col += 1;
        }

        for d in 1..=self.degree {
            for i in 0..n {
                result[[i, col]] = x_centered[i].powi(d as i32);
            }
            col += 1;
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
            Ok(basis.column(0).to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_natural_polynomial_centering() {
        // Create natural polynomial basis
        let basis = NaturalPolynomialBasis::new(2);

        // First evaluation with mean = 5.0
        let x1 = array![2.0, 4.0, 6.0, 8.0];
        let result1 = basis.evaluate(&x1.view()).unwrap();

        // The mean should be 5.0, so centered values are [-3, -1, 1, 3]
        // First column is intercept (all 1s)
        assert_abs_diff_eq!(result1[[0, 0]], 1.0);
        assert_abs_diff_eq!(result1[[1, 0]], 1.0);

        // Second column is linear term (centered values)
        assert_abs_diff_eq!(result1[[0, 1]], -3.0);
        assert_abs_diff_eq!(result1[[1, 1]], -1.0);
        assert_abs_diff_eq!(result1[[2, 1]], 1.0);
        assert_abs_diff_eq!(result1[[3, 1]], 3.0);

        // Third column is quadratic term (centered values squared)
        assert_abs_diff_eq!(result1[[0, 2]], 9.0);
        assert_abs_diff_eq!(result1[[1, 2]], 1.0);
        assert_abs_diff_eq!(result1[[2, 2]], 1.0);
        assert_abs_diff_eq!(result1[[3, 2]], 9.0);

        // Second evaluation with different data
        // Should still use mean = 5.0 from first evaluation
        let x2 = array![3.0, 5.0, 7.0];
        let result2 = basis.evaluate(&x2.view()).unwrap();

        // Centered values should be [-2, 0, 2] (using stored mean of 5.0)
        assert_abs_diff_eq!(result2[[0, 1]], -2.0);
        assert_abs_diff_eq!(result2[[1, 1]], 0.0);
        assert_abs_diff_eq!(result2[[2, 1]], 2.0);

        // Quadratic terms
        assert_abs_diff_eq!(result2[[0, 2]], 4.0);
        assert_abs_diff_eq!(result2[[1, 2]], 0.0);
        assert_abs_diff_eq!(result2[[2, 2]], 4.0);
    }

    #[test]
    fn test_polynomial_orthogonal() {
        // Test orthogonal polynomial generation
        let basis = PolynomialBasis::new(3);
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = basis.evaluate(&x.view()).unwrap();

        // Check dimensions
        assert_eq!(result.shape(), &[5, 3]);

        // Check orthogonality (dot products of columns should be near zero)
        let col0 = result.column(0);
        let col1 = result.column(1);
        let col2 = result.column(2);

        let dot01 = col0.dot(&col1);
        let dot02 = col0.dot(&col2);
        let dot12 = col1.dot(&col2);

        assert_abs_diff_eq!(dot01, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dot02, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(dot12, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_polynomial_raw() {
        // Test raw polynomial generation
        let basis = PolynomialBasis::new(2).with_raw(true).with_intercept(true);
        let x = array![1.0, 2.0, 3.0];

        let result = basis.evaluate(&x.view()).unwrap();

        // Check dimensions
        assert_eq!(result.shape(), &[3, 3]);

        // First column is intercept
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 0]], 1.0);
        assert_eq!(result[[2, 0]], 1.0);

        // Second column is x
        assert_eq!(result[[0, 1]], 1.0);
        assert_eq!(result[[1, 1]], 2.0);
        assert_eq!(result[[2, 1]], 3.0);

        // Third column is x^2
        assert_eq!(result[[0, 2]], 1.0);
        assert_eq!(result[[1, 2]], 4.0);
        assert_eq!(result[[2, 2]], 9.0);
    }
}
