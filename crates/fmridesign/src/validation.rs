//! Contrast validation and collinearity checks.
//!
//! Matches R's `validate_contrasts()` and `check_collinearity()` from
//! `fmridesign/R/validate.R`.

use nalgebra::DMatrix;
use ndarray::{Array1, Array2};

/// Result of validating a single contrast (or single column of an F-contrast).
#[derive(Debug, Clone)]
pub struct ContrastValidationResult {
    /// Contrast name.
    pub name: String,
    /// Type: "t" or "F".
    pub contrast_type: String,
    /// Whether the contrast is estimable (lies in row-space of X).
    pub estimable: bool,
    /// Whether the weights sum to zero (within tolerance).
    pub sum_to_zero: bool,
    /// Whether weights on intercept-like columns are zero.
    pub orthogonal_to_intercept: bool,
    /// Whether the F-contrast matrix has full column rank (None for t-contrasts).
    pub full_rank: Option<bool>,
    /// Number of non-zero weights.
    pub nonzero_weights: usize,
}

/// Validate contrast weights against a design matrix.
///
/// Checks estimability, sum-to-zero, intercept orthogonality, and full rank.
///
/// # Arguments
/// * `x` - Design matrix (n x p)
/// * `weights` - Contrast weight matrix (p x k). k=1 for t-contrast, k>1 for F-contrast.
/// * `col_names` - Column names for identifying intercept columns
/// * `tol` - Tolerance for zero checks (default: 1e-8)
pub fn validate_contrasts(
    x: &Array2<f64>,
    weights: &Array2<f64>,
    col_names: &[String],
    tol: f64,
) -> Vec<ContrastValidationResult> {
    let n_cols = weights.ncols();
    let contrast_type = if n_cols > 1 { "F" } else { "t" };
    let inputs_valid = tol.is_finite()
        && tol >= 0.0
        && x.ncols() == weights.nrows()
        && col_names.len() >= x.ncols()
        && x.iter().all(|value| value.is_finite())
        && weights.iter().all(|value| value.is_finite());

    // Find intercept-like columns
    let intercept_indices: Vec<usize> = col_names
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            let n = name.as_str();
            n == "(Intercept)" || n == "Intercept" || n == "constant" || n == "const"
        })
        .map(|(i, _)| i)
        .collect();

    // Full rank check for F-contrasts
    let full_rank = if n_cols > 1 {
        Some(inputs_valid && qr_rank(weights, tol) == n_cols)
    } else {
        None
    };

    let mut results = Vec::with_capacity(n_cols);

    for j in 0..n_cols {
        let cvec: Array1<f64> = weights.column(j).to_owned();

        let estimable = inputs_valid && is_estimable(x, &cvec, tol);
        let sum_to_zero = inputs_valid && cvec.sum().abs() < tol;

        let orthogonal_to_intercept = if !inputs_valid {
            false
        } else if intercept_indices.is_empty() {
            true
        } else {
            intercept_indices.iter().all(|&i| {
                if i < cvec.len() {
                    cvec[i].abs() < tol
                } else {
                    true
                }
            })
        };

        let nonzero_weights = if inputs_valid {
            cvec.iter().filter(|&&v| v.abs() > tol).count()
        } else {
            0
        };

        let name = if n_cols == 1 {
            "contrast".to_string()
        } else {
            format!("contrast#{}", j + 1)
        };

        results.push(ContrastValidationResult {
            name,
            contrast_type: contrast_type.to_string(),
            estimable,
            sum_to_zero,
            orthogonal_to_intercept,
            full_rank,
            nonzero_weights,
        });
    }

    results
}

/// Check if a contrast vector is estimable given design matrix X.
///
/// A contrast c is estimable if it lies in the row-space of X, i.e.,
/// its SVD row-space projection recovers c within the supplied relative tolerance.
fn is_estimable(x: &Array2<f64>, cvec: &Array1<f64>, tol: f64) -> bool {
    if cvec.len() != x.ncols()
        || !tol.is_finite()
        || tol < 0.0
        || x.iter().any(|value| !value.is_finite())
        || cvec.iter().any(|value| !value.is_finite())
    {
        return false;
    }

    let n = x.nrows();
    let p = x.ncols();
    if n == 0 || p == 0 {
        return cvec.is_empty();
    }

    let contrast_scale = cvec.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if contrast_scale == 0.0 {
        return true;
    }
    let design_scale = x.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if design_scale == 0.0 {
        return false;
    }

    let na_x = DMatrix::from_fn(n, p, |i, j| x[[i, j]] / design_scale);
    let svd = na_x.svd(false, true);
    let max_singular = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    if max_singular == 0.0 {
        return cvec.iter().all(|&value| value == 0.0);
    }
    let threshold = tol * max_singular;
    let v_t = match svd.v_t {
        Some(v_t) => v_t,
        None => return false,
    };
    let c = DMatrix::from_iterator(p, 1, cvec.iter().map(|value| value / contrast_scale));
    let mut projection = DMatrix::zeros(p, 1);
    for (index, &singular) in svd.singular_values.iter().enumerate() {
        if singular > threshold {
            let v = v_t.row(index).transpose();
            projection += &v * (v.transpose() * &c);
        }
    }
    let residual = (&c - projection).norm();
    residual <= tol * c.norm()
}

/// Compute QR rank of a matrix.
fn qr_rank(m: &Array2<f64>, tol: f64) -> usize {
    let (nrows, ncols) = m.dim();
    if !tol.is_finite() || tol < 0.0 || m.iter().any(|value| !value.is_finite()) {
        return 0;
    }
    let matrix_scale = m.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if matrix_scale == 0.0 {
        return 0;
    }
    let na_m = DMatrix::from_fn(nrows, ncols, |i, j| m[[i, j]] / matrix_scale);
    let singular_values = na_m.svd(false, false).singular_values;
    let max_singular = singular_values.iter().copied().fold(0.0_f64, f64::max);
    singular_values
        .iter()
        .filter(|&&singular| singular > tol * max_singular)
        .count()
}

/// Result of collinearity check.
#[derive(Debug, Clone)]
pub struct CollinearityResult {
    /// Whether the design is free of high collinearity.
    pub ok: bool,
    /// Pairs of regressors with correlation above threshold: (name1, name2, r).
    pub pairs: Vec<(String, String, f64)>,
}

/// Check a design matrix for multicollinearity.
///
/// Flags pairs of regressors with absolute correlation above `threshold`.
/// Intercept-like and zero-variance columns are excluded.
///
/// # Arguments
/// * `x` - Design matrix (n x p)
/// * `col_names` - Column names
/// * `threshold` - Correlation threshold (default: 0.9)
pub fn check_collinearity(
    x: &Array2<f64>,
    col_names: &[String],
    threshold: f64,
) -> CollinearityResult {
    let p = x.ncols();

    // Identify columns to keep (non-intercept, non-zero-variance)
    let intercept_names = ["(Intercept)", "Intercept", "constant", "const"];
    let mut keep: Vec<usize> = Vec::new();

    for j in 0..p {
        let name = col_names.get(j).map(|s| s.as_str()).unwrap_or("");
        if intercept_names.contains(&name) {
            continue;
        }

        // Check variance
        let col = x.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let var: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
        if var > 0.0 && var.is_finite() {
            keep.push(j);
        }
    }

    if keep.len() < 2 {
        return CollinearityResult {
            ok: true,
            pairs: vec![],
        };
    }

    // Compute correlation matrix for kept columns
    let mut pairs = Vec::new();

    for ii in 0..keep.len() {
        for jj in (ii + 1)..keep.len() {
            let ci = keep[ii];
            let cj = keep[jj];

            let r = pearson_correlation(&x.column(ci), &x.column(cj));

            if r.abs() > threshold {
                pairs.push((
                    col_names
                        .get(ci)
                        .cloned()
                        .unwrap_or_else(|| format!("col_{}", ci)),
                    col_names
                        .get(cj)
                        .cloned()
                        .unwrap_or_else(|| format!("col_{}", cj)),
                    r,
                ));
            }
        }
    }

    CollinearityResult {
        ok: pairs.is_empty(),
        pairs,
    }
}

/// Pearson correlation between two array views.
fn pearson_correlation(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
    let n = a.len() as f64;
    let mean_a = a.sum() / n;
    let mean_b = b.sum() / n;

    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai - mean_a;
        let db = bi - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }

    if var_a == 0.0 || var_b == 0.0 {
        0.0
    } else {
        cov / (var_a.sqrt() * var_b.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_validate_estimable_contrast() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let w = array![[1.0], [-1.0]]; // difference of two columns
        let names = vec!["a".to_string(), "b".to_string()];
        let results = validate_contrasts(&x, &w, &names, 1e-8);
        assert_eq!(results.len(), 1);
        assert!(results[0].estimable);
        assert!(results[0].sum_to_zero);
    }

    #[test]
    fn test_validate_sum_to_zero() {
        let x = array![[1.0, 0.0], [0.0, 1.0]];
        let w = array![[0.5], [0.5]]; // sums to 1, not 0
        let names = vec!["a".to_string(), "b".to_string()];
        let results = validate_contrasts(&x, &w, &names, 1e-8);
        assert!(!results[0].sum_to_zero);
    }

    #[test]
    fn test_validate_intercept_orthogonality() {
        let x = array![[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]];
        let w = array![[0.5], [1.0], [-1.0]]; // non-zero on intercept
        let names = vec!["Intercept".to_string(), "a".to_string(), "b".to_string()];
        let results = validate_contrasts(&x, &w, &names, 1e-8);
        assert!(!results[0].orthogonal_to_intercept);
    }

    #[test]
    fn test_validate_f_contrast_full_rank() {
        let x = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let w = array![[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]];
        let names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let results = validate_contrasts(&x, &w, &names, 1e-8);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].contrast_type, "F");
        assert_eq!(results[0].full_rank, Some(true));
    }

    #[test]
    fn test_collinearity_none() {
        // Uncorrelated columns
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 3.0], [4.0, 2.0]];
        let names = vec!["a".to_string(), "b".to_string()];
        let result = check_collinearity(&x, &names, 0.9);
        assert!(result.ok);
        assert!(result.pairs.is_empty());
    }

    #[test]
    fn test_collinearity_detected() {
        // Two nearly identical columns
        let x = array![[1.0, 1.01], [2.0, 2.01], [3.0, 3.01], [4.0, 4.01]];
        let names = vec!["a".to_string(), "b".to_string()];
        let result = check_collinearity(&x, &names, 0.9);
        assert!(!result.ok);
        assert_eq!(result.pairs.len(), 1);
        assert!(result.pairs[0].2 > 0.99);
    }

    #[test]
    fn test_collinearity_skips_intercept() {
        let x = array![[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]];
        let names = vec!["Intercept".to_string(), "slope".to_string()];
        let result = check_collinearity(&x, &names, 0.9);
        // Only one non-intercept column, can't have pairs
        assert!(result.ok);
    }
}
