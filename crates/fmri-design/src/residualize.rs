//! QR-based residualization utilities.
//!
//! Projects data onto the orthogonal complement of a design matrix,
//! returning OLS residuals. This is the Rust equivalent of R's
//! `residualize()` S3 methods from `fmri_design/R/residualize.R`.
//!
//! # Algorithm
//!
//! Given a design matrix **X** and data matrix **Y**:
//! 1. Compute the thin QR decomposition of **X**: `X = Q R`.
//! 2. Residuals = `Y - Q Q^T Y` (projection onto the column space is
//!    subtracted from **Y**).
//!
//! # Examples
//!
//! ```
//! use ndarray::array;
//! use fmri_design::residualize::residualize;
//!
//! let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
//! let y = array![[2.0], [3.0], [5.0], [4.0]];
//! let resid = residualize(&x, &y, None).unwrap();
//! assert_eq!(resid.shape(), &[4, 1]);
//! ```

use nalgebra::DMatrix;
use ndarray::Array2;

use crate::core::EventModel;
use crate::types::ValidationError;

// ---------------------------------------------------------------------------
// Core residualization
// ---------------------------------------------------------------------------

/// Project **Y** onto the orthogonal complement of **X** (optionally
/// subsetting columns of **X** first).
///
/// This mirrors R's `.residualize_with_matrix(X, Y)` combined with the
/// optional column selection found in the S3 methods.
///
/// # Arguments
///
/// * `x` - Design matrix (n x p).
/// * `y` - Data matrix (n x m).
/// * `cols` - Optional column indices to select from `x` before
///   residualizing. When `None`, all columns of `x` are used.
///
/// # Errors
///
/// Returns [`ValidationError::LengthMismatch`] when the row counts of
/// `x` and `y` differ, or [`ValidationError::InvalidParameter`] when a
/// column index is out of bounds.
pub fn residualize(
    x: &Array2<f64>,
    y: &Array2<f64>,
    cols: Option<&[usize]>,
) -> Result<Array2<f64>, ValidationError> {
    let x_sub = select_columns(x, cols)?;
    residualize_core(&x_sub, y)
}

/// Residualize using an [`EventModel`]'s design matrix.
///
/// Generates the design matrix from the model, optionally subsets its
/// columns, and returns the residuals of `data` with respect to that
/// design.
///
/// # Arguments
///
/// * `model` - An event model whose design matrix will be used as **X**.
/// * `data`  - Data matrix **Y** (n x m).
/// * `cols`  - Optional column indices to select from the design matrix.
pub fn residualize_with_event_model(
    model: &EventModel,
    data: &Array2<f64>,
    cols: Option<&[usize]>,
) -> Result<Array2<f64>, ValidationError> {
    let dm = model.generate_design_matrix()?;
    residualize(&dm, data, cols)
}

/// Residualize by selecting columns of **X** by name.
///
/// Matches entries in `select` against `col_names` to determine which
/// columns of `x` to keep, then residualizes `y`.
///
/// # Arguments
///
/// * `x`         - Design matrix (n x p).
/// * `y`         - Data matrix (n x m).
/// * `col_names` - Column names corresponding to the columns of `x`.
/// * `select`    - Column names to retain for residualization.
///
/// # Errors
///
/// Returns [`ValidationError::MissingColumn`] if any entry in `select`
/// is not found in `col_names`.
pub fn residualize_with_column_names(
    x: &Array2<f64>,
    y: &Array2<f64>,
    col_names: &[String],
    select: &[&str],
) -> Result<Array2<f64>, ValidationError> {
    let indices = resolve_column_indices(col_names, select)?;
    residualize(x, y, Some(&indices))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Core QR-based residualization: `Y - Q Q^T Y`.
fn residualize_core(x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>, ValidationError> {
    let n_x = x.nrows();
    let n_y = y.nrows();

    if n_x != n_y {
        return Err(ValidationError::LengthMismatch {
            expected: n_x,
            actual: n_y,
        });
    }

    let n = n_x;
    let p = x.ncols();
    let m = y.ncols();

    // Edge case: zero-column design -> residuals equal the data.
    if p == 0 {
        return Ok(y.clone());
    }

    // Convert ndarray -> nalgebra for QR.
    let na_x = DMatrix::from_fn(n, p, |i, j| x[[i, j]]);
    let na_y = DMatrix::from_fn(n, m, |i, j| y[[i, j]]);

    let qr = na_x.qr();
    // Thin Q has shape (n x p) when p <= n.
    let q = qr.q();
    // Only the first `min(n,p)` columns of Q matter for the projection.
    let q_thin = q.columns(0, p.min(n));

    // Projection: hat = Q_thin * Q_thin^T * Y
    let hat = &q_thin * (q_thin.transpose() * &na_y);
    let resid_na = na_y - hat;

    // Convert back to ndarray.
    let mut result = Array2::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            result[[i, j]] = resid_na[(i, j)];
        }
    }

    Ok(result)
}

/// Optionally subset columns of a matrix.
fn select_columns(x: &Array2<f64>, cols: Option<&[usize]>) -> Result<Array2<f64>, ValidationError> {
    match cols {
        None => Ok(x.clone()),
        Some(indices) => {
            let p = x.ncols();
            for &c in indices {
                if c >= p {
                    return Err(ValidationError::InvalidParameter(format!(
                        "Column index {} out of bounds for matrix with {} columns",
                        c, p
                    )));
                }
            }
            let n = x.nrows();
            let mut sub = Array2::zeros((n, indices.len()));
            for (out_j, &src_j) in indices.iter().enumerate() {
                sub.column_mut(out_j).assign(&x.column(src_j));
            }
            Ok(sub)
        }
    }
}

/// Map column names to indices.
fn resolve_column_indices(
    col_names: &[String],
    select: &[&str],
) -> Result<Vec<usize>, ValidationError> {
    select
        .iter()
        .map(|&name| {
            col_names
                .iter()
                .position(|cn| cn == name)
                .ok_or_else(|| ValidationError::MissingColumn(name.to_string()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Assert two matrices are element-wise equal within tolerance.
    fn assert_arrays_close(a: &Array2<f64>, b: &Array2<f64>, tol: f64) {
        assert_eq!(a.shape(), b.shape(), "Shape mismatch");
        for ((idx, &va), &vb) in a.indexed_iter().zip(b.iter()) {
            assert!(
                (va - vb).abs() < tol,
                "Mismatch at {:?}: {} vs {} (diff {:.2e})",
                idx,
                va,
                vb,
                (va - vb).abs(),
            );
        }
    }

    /// Assert scalar is near zero.
    fn assert_near_zero(val: f64, tol: f64, msg: &str) {
        assert!(val.abs() < tol, "{}: {:.2e}", msg, val);
    }

    /// Check that every column of `resid` is orthogonal to every
    /// column of `x` (within floating-point tolerance).
    fn assert_orthogonal(x: &Array2<f64>, resid: &Array2<f64>, tol: f64) {
        for j_x in 0..x.ncols() {
            for j_r in 0..resid.ncols() {
                let dot: f64 = x
                    .column(j_x)
                    .iter()
                    .zip(resid.column(j_r).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                assert!(
                    dot.abs() < tol,
                    "Column {} of X and column {} of residuals are not orthogonal: dot = {:.2e}",
                    j_x,
                    j_r,
                    dot,
                );
            }
        }
    }

    #[test]
    fn test_residualize_dimensions() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let y = array![[2.0, 10.0], [3.0, 20.0], [5.0, 30.0], [4.0, 40.0]];
        let resid = residualize(&x, &y, None).unwrap();
        assert_eq!(resid.shape(), &[4, 2]);
    }

    #[test]
    fn test_residuals_orthogonal_to_design() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let y = array![[2.0], [4.0], [5.0], [4.0], [5.0]];
        let resid = residualize(&x, &y, None).unwrap();
        assert_orthogonal(&x, &resid, 1e-10);
    }

    #[test]
    fn test_residualize_perfect_fit() {
        // Y lies entirely in the column space of X -> residuals should be ~0.
        let x = array![[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]];
        // y = 1*col0 + 1*col1
        let y = array![[3.0], [5.0], [7.0]];
        let resid = residualize(&x, &y, None).unwrap();
        for &val in resid.iter() {
            assert_near_zero(val, 1e-10, "residual of perfect fit");
        }
    }

    #[test]
    fn test_residualize_identity_with_zero_columns() {
        // Zero-column design: residuals equal data.
        let x = Array2::<f64>::zeros((4, 0));
        let y = array![[1.0], [2.0], [3.0], [4.0]];
        let resid = residualize(&x, &y, None).unwrap();
        assert_arrays_close(&resid, &y, 1e-14);
    }

    #[test]
    fn test_residualize_row_mismatch() {
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![[1.0], [2.0]];
        let err = residualize(&x, &y, None).unwrap_err();
        assert!(matches!(
            err,
            ValidationError::LengthMismatch {
                expected: 3,
                actual: 2
            }
        ));
    }

    #[test]
    fn test_residualize_with_column_subset() {
        // X has 3 columns; only use the first two for residualizing.
        let x = array![
            [1.0, 0.0, 100.0],
            [1.0, 1.0, 200.0],
            [1.0, 2.0, 300.0],
            [1.0, 3.0, 400.0]
        ];
        let y = array![[2.0], [3.0], [5.0], [4.0]];
        let resid = residualize(&x, &y, Some(&[0, 1])).unwrap();
        // Residuals should be orthogonal to the selected columns only.
        let x_sub = x.select(ndarray::Axis(1), &[0, 1]);
        assert_orthogonal(&x_sub, &resid, 1e-10);
    }

    #[test]
    fn test_residualize_col_index_out_of_bounds() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0]];
        let err = residualize(&x, &y, Some(&[5])).unwrap_err();
        assert!(matches!(err, ValidationError::InvalidParameter(_)));
    }

    #[test]
    fn test_residualize_with_column_names() {
        let x = array![[1.0, 0.0, 10.0], [1.0, 1.0, 20.0], [1.0, 2.0, 30.0]];
        let y = array![[5.0], [6.0], [7.0]];
        let names = vec![
            "intercept".to_string(),
            "slope".to_string(),
            "noise".to_string(),
        ];
        let resid = residualize_with_column_names(&x, &y, &names, &["intercept", "slope"]).unwrap();
        let x_sub = x.select(ndarray::Axis(1), &[0, 1]);
        assert_orthogonal(&x_sub, &resid, 1e-10);
    }

    #[test]
    fn test_residualize_missing_column_name() {
        let x = array![[1.0], [2.0]];
        let y = array![[1.0], [2.0]];
        let names = vec!["alpha".to_string()];
        let err = residualize_with_column_names(&x, &y, &names, &["beta"]).unwrap_err();
        assert!(matches!(err, ValidationError::MissingColumn(_)));
    }

    #[test]
    fn test_residualize_multi_column_y() {
        // Residualize multiple Y columns at once.
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
        let resid = residualize(&x, &y, None).unwrap();
        // After removing the mean (intercept), residuals should sum to ~0.
        for j in 0..resid.ncols() {
            let col_sum: f64 = resid.column(j).sum();
            assert_near_zero(col_sum, 1e-10, &format!("column {} sum", j));
        }
        assert_orthogonal(&x, &resid, 1e-10);
    }

    #[test]
    fn test_residualize_known_values() {
        // Manual example: X = [1; 1; 1; 1] (intercept only).
        // Residualizing removes the mean.
        let x = array![[1.0], [1.0], [1.0], [1.0]];
        let y = array![[2.0], [4.0], [6.0], [8.0]];
        let resid = residualize(&x, &y, None).unwrap();
        // Mean of y is 5.0, so residuals are y - 5.
        let expected = array![[-3.0], [-1.0], [1.0], [3.0]];
        assert_arrays_close(&resid, &expected, 1e-10);
    }
}
