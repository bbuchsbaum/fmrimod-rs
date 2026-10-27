//! Penalty matrix functions for HRF regularization.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2};

/// Structural regularization policy, independent of display names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyKind {
    /// Ridge on all coefficients.
    Identity,
    /// Unpenalized canonical component; derivative shrinkage.
    SpmDerivatives,
    /// Finite-difference roughness.
    Roughness,
    /// Increasing frequency penalty.
    FourierFrequency,
    /// Increasing Laguerre component penalty.
    DaguerreDecay,
}

/// Generate penalty matrix for regularizing HRF basis coefficients.
///
/// The penalty matrix encodes shape priors that discourage implausible
/// or overly wiggly HRF estimates. Different HRF types use different
/// penalty structures.
///
/// # Arguments
/// * `hrf` - HRF object
/// * `order` - Order of the penalty (default: 2)
/// * `options` - Additional options for specific HRF types
///
/// # Returns
/// Symmetric positive semidefinite penalty matrix of shape (nbasis, nbasis)
pub fn penalty_matrix(
    hrf: &dyn Hrf,
    order: usize,
    options: Option<PenaltyOptions>,
) -> Result<Array2<f64>> {
    let opts = options.unwrap_or_default();
    if hrf.nbasis() == 0 || !opts.shrink_deriv.is_finite() || opts.shrink_deriv < 0.0 {
        return Err(HrfError::InvalidInput(
            "invalid penalty dimension or shrinkage".into(),
        ));
    }
    match hrf.penalty_kind() {
        PenaltyKind::SpmDerivatives => penalty_spmg2(hrf, order, opts.shrink_deriv),
        PenaltyKind::Roughness => penalty_roughness(hrf, order),
        PenaltyKind::FourierFrequency => penalty_fourier(hrf, order),
        PenaltyKind::DaguerreDecay => penalty_daguerre(hrf, order),
        PenaltyKind::Identity => Ok(Array2::eye(hrf.nbasis())),
    }
}

/// Options for penalty matrix generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PenaltyOptions {
    /// Shrinkage factor for derivative terms
    pub shrink_deriv: f64,
}

impl Default for PenaltyOptions {
    fn default() -> Self {
        Self { shrink_deriv: 2.0 }
    }
}

/// Roughness penalty based on discrete derivatives.
///
/// Used for FIR, B-spline, and tent basis functions.
fn penalty_roughness(hrf: &dyn Hrf, order: usize) -> Result<Array2<f64>> {
    let nb = hrf.nbasis();

    if nb <= 1 {
        Ok(Array2::eye(nb))
    } else if nb > order {
        // Create difference matrix
        let d = create_difference_matrix(nb, order);
        // Return D'D
        Ok(d.t().dot(&d))
    } else {
        Ok(Array2::eye(nb))
    }
}

/// Create difference matrix of given order.
fn create_difference_matrix(n: usize, order: usize) -> Array2<f64> {
    if order == 0 || n == 0 {
        return Array2::eye(n);
    }

    // Start with identity matrix
    let mut mat = Array2::eye(n);

    // Apply differencing `order` times
    for _ in 0..order {
        let rows = mat.nrows();
        if rows <= 1 {
            break;
        }

        // Create new matrix with one fewer row
        let mut new_mat = Array2::zeros((rows - 1, mat.ncols()));

        // Compute differences
        for i in 0..rows - 1 {
            for j in 0..mat.ncols() {
                new_mat[[i, j]] = mat[[i + 1, j]] - mat[[i, j]];
            }
        }

        mat = new_mat;
    }

    mat
}

/// Penalty for SPM with temporal derivative.
///
/// Canonical term is not penalized, derivatives are shrunk.
fn penalty_spmg2(hrf: &dyn Hrf, _order: usize, shrink_deriv: f64) -> Result<Array2<f64>> {
    let nb = hrf.nbasis();
    let mut r = Array2::eye(nb);

    if nb >= 1 {
        r[[0, 0]] = 0.0; // Don't penalize canonical
    }
    // Shrink all derivatives
    for i in 1..nb {
        r[[i, i]] = shrink_deriv;
    }

    Ok(r)
}

/// Penalty for Fourier basis.
///
/// Higher frequencies are penalized more.
fn penalty_fourier(hrf: &dyn Hrf, order: usize) -> Result<Array2<f64>> {
    let nb = hrf.nbasis();

    // Frequency indices (1, 1, 2, 2, 3, 3, ...)
    let freqs: Vec<f64> = (1..=nb).map(|i| (i as f64 / 2.0).ceil()).collect();

    // Penalty proportional to frequency^order
    let penalties: Vec<f64> = freqs.iter().map(|&f| f.powi(order as i32)).collect();

    Ok(Array2::from_diag(&Array1::from_vec(penalties)))
}

/// Penalty for Daguerre basis.
///
/// Higher order terms are penalized more.
fn penalty_daguerre(hrf: &dyn Hrf, _order: usize) -> Result<Array2<f64>> {
    let nb = hrf.nbasis();

    // Penalty increases with basis index
    let penalties: Vec<f64> = (0..nb).map(|i| (i as f64).powi(2)).collect();

    Ok(Array2::from_diag(&Array1::from_vec(penalties)))
}

/// Check if a matrix is positive semi-definite.
pub fn is_positive_semidefinite(mat: &Array2<f64>) -> bool {
    if mat.nrows() != mat.ncols() {
        return false;
    }

    // Check symmetry
    for i in 0..mat.nrows() {
        for j in i + 1..mat.ncols() {
            if (mat[[i, j]] - mat[[j, i]]).abs() > 1e-10 {
                return false;
            }
        }
    }

    // Explicit indexing also supports sliced/non-contiguous ndarray storage.
    if mat.iter().any(|v| !v.is_finite()) {
        return false;
    }
    if mat.is_empty() {
        return true;
    }
    let dense = DMatrix::from_fn(mat.nrows(), mat.ncols(), |r, c| mat[[r, c]]);
    SymmetricEigen::try_new(dense, f64::EPSILON, 10_000)
        .is_some_and(|decomposition| decomposition.eigenvalues.iter().all(|&v| v >= -1e-10))
}

/// Generate penalty matrix with automatic order selection.
///
/// Chooses appropriate order based on HRF type and number of basis functions.
pub fn auto_penalty_matrix(hrf: &dyn Hrf) -> Result<Array2<f64>> {
    let order = match hrf.name().to_uppercase().as_str() {
        "FIR" | "FIR_HRF" | "TENT" => 2,
        "BSPLINE" | "BSPLINE_HRF" | "BS" => {
            // Use order = degree + 1
            if let Some(degree) = hrf
                .params()
                .iter()
                .find(|(name, _)| name == &"degree")
                .map(|(_, val)| *val as usize)
            {
                degree + 1
            } else {
                2
            }
        }
        "FOURIER" | "FOURIER_HRF" => 2,
        _ => 2,
    };

    penalty_matrix(hrf, order, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::functions::{bspline_hrf, fir_hrf, fourier_hrf, spm_hrf_with_derivatives};
    use approx::assert_relative_eq;

    #[test]
    fn test_penalty_identity() {
        // For unknown HRF types, should return identity
        use crate::hrf::functions::gamma_hrf;
        let hrf = gamma_hrf(6.0, 1.0, Some(true));
        let penalty = penalty_matrix(hrf.as_ref(), 2, None).unwrap();

        assert_eq!(penalty.shape(), &[1, 1]);
        assert_eq!(penalty[[0, 0]], 1.0);
    }

    #[test]
    fn test_penalty_spmg2() {
        let hrf = spm_hrf_with_derivatives();
        let penalty = penalty_matrix(hrf.as_ref(), 2, None).unwrap();

        assert_eq!(penalty.shape(), &[3, 3]);
        // Canonical not penalized
        assert_eq!(penalty[[0, 0]], 0.0);
        // Derivatives penalized
        assert_eq!(penalty[[1, 1]], 2.0);
        assert_eq!(penalty[[2, 2]], 2.0);
    }

    #[test]
    fn test_penalty_roughness_fir() {
        let hrf = fir_hrf(5, Some(10.0));
        let penalty = penalty_matrix(hrf.as_ref(), 2, None).unwrap();

        assert_eq!(penalty.shape(), &[5, 5]);

        // Should be symmetric
        for i in 0..5 {
            for j in i..5 {
                assert_relative_eq!(penalty[[i, j]], penalty[[j, i]], epsilon = 1e-10);
            }
        }

        // Should be positive semi-definite
        assert!(is_positive_semidefinite(&penalty));
    }

    #[test]
    fn test_penalty_bspline() {
        let hrf = bspline_hrf(4, Some(3), Some(10.0));
        let penalty = penalty_matrix(hrf.as_ref(), 2, None).unwrap();

        assert_eq!(penalty.shape(), &[4, 4]);
        assert!(is_positive_semidefinite(&penalty));
    }

    #[test]
    fn test_penalty_fourier() {
        let hrf = fourier_hrf(6, Some(10.0));
        let penalty = penalty_matrix(hrf.as_ref(), 2, None).unwrap();

        assert_eq!(penalty.shape(), &[6, 6]);

        // Check diagonal structure
        for i in 0..6 {
            for j in 0..6 {
                if i != j {
                    assert_eq!(penalty[[i, j]], 0.0);
                }
            }
        }

        // Check frequency weighting
        assert_eq!(penalty[[0, 0]], 1.0); // freq 1
        assert_eq!(penalty[[1, 1]], 1.0); // freq 1
        assert_eq!(penalty[[2, 2]], 4.0); // freq 2
        assert_eq!(penalty[[3, 3]], 4.0); // freq 2
    }

    #[test]
    fn test_difference_matrix() {
        // Test first order differences
        let d1 = create_difference_matrix(4, 1);
        assert_eq!(d1.shape(), &[3, 4]);
        assert_eq!(d1[[0, 0]], -1.0);
        assert_eq!(d1[[0, 1]], 1.0);

        // Test second order differences
        let d2 = create_difference_matrix(4, 2);
        assert_eq!(d2.shape(), &[2, 4]);
    }

    #[test]
    fn test_auto_penalty() {
        let hrf = fir_hrf(5, Some(10.0));
        let penalty = auto_penalty_matrix(hrf.as_ref()).unwrap();

        assert_eq!(penalty.shape(), &[5, 5]);
        assert!(is_positive_semidefinite(&penalty));
    }

    #[test]
    fn test_custom_shrink_factor() {
        let hrf = spm_hrf_with_derivatives();
        let opts = PenaltyOptions { shrink_deriv: 5.0 };
        let penalty = penalty_matrix(hrf.as_ref(), 2, Some(opts)).unwrap();

        assert_eq!(penalty[[1, 1]], 5.0);
        assert_eq!(penalty[[2, 2]], 5.0);
    }
}
