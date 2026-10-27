//! Discrete cosine transform (DCT) basis functions.
//!
//! This is a Rust port of the internal `dctbasis()` helper from the R
//! fmridesign package. It is typically used for low-frequency drift
//! modeling in baseline terms.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use ndarray::{Array1, Array2, ArrayView1};
use std::f64::consts::PI;

/// DCT (type-II) basis over a regularly sampled grid.
///
/// The basis is defined only by its length (taken from the input x) and the
/// number of cosine components. The actual values of `x` are ignored; only
/// `x.len()` matters, matching the R helper which uses `m <- 1:n`.
#[derive(Debug, Clone)]
pub struct DctBasis {
    /// Number of non-constant cosine components.
    n_cos: usize,
    /// Whether to include a constant (DC) column.
    include_constant: bool,
    /// Human-readable name.
    name: String,
}

impl DctBasis {
    /// Create a new DCT basis with `n_cos` cosine components.
    ///
    /// If `include_constant` is `true`, an additional DC column is added,
    /// so the total number of basis functions is `n_cos + 1`.
    pub fn new(n_cos: usize, include_constant: bool) -> Result<Self, ValidationError> {
        if n_cos == 0 {
            return Err(ValidationError::InvalidParameter(
                "DctBasis requires at least one cosine component (n_cos >= 1)".to_string(),
            ));
        }

        let name = if include_constant {
            format!("dct{}_c", n_cos)
        } else {
            format!("dct{}", n_cos)
        };

        Ok(Self {
            n_cos,
            include_constant,
            name,
        })
    }

    /// Convenience constructor for a DCT basis without a constant term.
    pub fn without_constant(n_cos: usize) -> Result<Self, ValidationError> {
        Self::new(n_cos, false)
    }

    /// Convenience constructor for a DCT basis with a constant term.
    pub fn with_constant(n_cos: usize) -> Result<Self, ValidationError> {
        Self::new(n_cos, true)
    }
}

impl BasisFunction for DctBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        self.n_cos + if self.include_constant { 1 } else { 0 }
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        let n = x.len();
        if n == 0 {
            return Err(ValidationError::EmptyArray);
        }

        let n_cos = self.n_cos;
        let n_basis = self.nbasis();
        let mut basis = Array2::zeros((n, n_basis));

        // Precompute (2*m - 1) for m = 1..n
        let m_vec: Array1<f64> = (1..=n).map(|m| (2 * m as isize - 1) as f64).collect();

        // Constant (DC) component if requested: (1/n)^{1/2}
        let mut col = 0;
        if self.include_constant {
            let c = (1.0 / n as f64).sqrt();
            for i in 0..n {
                basis[[i, col]] = c;
            }
            col += 1;
        }

        // Cosine components: match R's dctbasis(m, p, const) with p = n_cos + 1
        // k runs from 2..=p, so there are n_cos cosine columns.
        let p = n_cos + 1;
        let scale = (2.0 / n as f64).sqrt();

        for k in 2..=p {
            let k_f = k as f64;
            for (i, m_val) in m_vec.iter().enumerate() {
                let angle = (m_val * k_f * PI) / (2.0 * n as f64);
                basis[[i, col]] = scale * angle.cos();
            }
            col += 1;
        }

        Ok(basis)
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
            // By convention, return the first basis column if no coefficients are provided.
            Ok(basis.column(0).to_owned())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_dct_basis_shapes() {
        let basis = DctBasis::new(3, false).unwrap();
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = basis.evaluate(&x.view()).unwrap();

        assert_eq!(y.shape(), &[5, 3]); // n = 5, n_cos = 3, no constant
    }

    #[test]
    fn test_dct_basis_with_constant() {
        let basis = DctBasis::new(4, true).unwrap();
        let x = array![0.0, 1.0, 2.0, 3.0];
        let y = basis.evaluate(&x.view()).unwrap();

        assert_eq!(y.shape(), &[4, 5]); // n = 4, 1 const + 4 cosines

        // First column should be constant with norm 1
        let col0 = y.column(0);
        let mean = col0.mean().unwrap();
        assert_relative_eq!(mean, (1.0 / 4.0_f64).sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn test_dct_basis_orthogonality() {
        let basis = DctBasis::new(3, true).unwrap();
        let x = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = basis.evaluate(&x.view()).unwrap();

        // Columns should be approximately orthonormal
        for i in 0..basis.nbasis() {
            let col_i = y.column(i);
            let norm_i = col_i.dot(&col_i).sqrt();
            assert_relative_eq!(norm_i, 1.0, epsilon = 1e-10);

            for j in (i + 1)..basis.nbasis() {
                let col_j = y.column(j);
                let dot = col_i.dot(&col_j);
                assert_relative_eq!(dot, 0.0, epsilon = 1e-10);
            }
        }
    }
}
