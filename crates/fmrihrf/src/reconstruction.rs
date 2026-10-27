//! Reconstruction matrix for HRF basis.

use crate::error::Result;
use crate::hrf::core::Hrf;
use crate::sampling::SamplingFrame;
use crate::types::{HrfValues, TimeArray};
use ndarray::Array1;

/// Generate reconstruction matrix for HRF basis.
///
/// Returns a matrix Phi that converts basis coefficients into a
/// sampled HRF shape. Each column represents one basis function
/// evaluated at the sampling times.
///
/// # Arguments
/// * `hrf` - HRF object with basis functions
/// * `times` - Time points at which to evaluate the basis
///
/// # Returns
/// Matrix with shape (n_samples, nbasis) where each column
/// is a basis function evaluated at the sample times
///
/// # Examples
/// ```
/// use fmrihrf::{hrf::spm_canonical, reconstruction::reconstruction_matrix};
/// use ndarray::Array1;
///
/// let hrf = spm_canonical(None, None, None, None, None, None);
/// let times = Array1::linspace(0.0, 20.0, 101);
/// let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();
/// assert_eq!(r_matrix.shape(), &[101, 1]);
/// ```
pub fn reconstruction_matrix(hrf: &dyn Hrf, times: &TimeArray) -> Result<HrfValues> {
    if times.iter().any(|v| !v.is_finite()) {
        return Err(crate::HrfError::InvalidInput(
            "nonfinite reconstruction time".into(),
        ));
    }
    let values = hrf.evaluate(times)?;
    crate::regressor::convolution::check_shape(&values, times.len(), hrf.nbasis())?;
    Ok(values)
}

/// Generate reconstruction matrix from a SamplingFrame.
///
/// This is a convenience function that extracts time points from
/// a SamplingFrame and generates the reconstruction matrix.
///
/// # Arguments
/// * `hrf` - HRF object with basis functions
/// * `sframe` - SamplingFrame specifying the time grid
///
/// # Returns
/// Matrix with shape (n_samples, nbasis) where each column
/// is a basis function evaluated at the sample times
pub fn reconstruction_matrix_from_sampling(
    hrf: &dyn Hrf,
    sframe: &SamplingFrame,
) -> Result<HrfValues> {
    let times = Array1::from_vec(sframe.samples());
    reconstruction_matrix(hrf, &times)
}

/// Reconstruct HRF from basis coefficients.
///
/// Given a reconstruction matrix and coefficients, compute the
/// reconstructed HRF shape.
///
/// # Arguments
/// * `r_matrix` - Reconstruction matrix from `reconstruction_matrix`
/// * `coefficients` - Coefficients for each basis function
///
/// # Returns
/// Reconstructed HRF values
///
/// # Errors
/// Returns error if coefficient length doesn't match number of basis functions
pub fn reconstruct_hrf(r_matrix: &HrfValues, coefficients: &Array1<f64>) -> Result<Array1<f64>> {
    if r_matrix.ncols() != coefficients.len() {
        return Err(crate::error::HrfError::DimensionMismatch {
            expected: format!("{} coefficients", r_matrix.ncols()),
            actual: format!("{} coefficients", coefficients.len()),
        });
    }

    if r_matrix
        .iter()
        .chain(coefficients.iter())
        .any(|v| !v.is_finite())
    {
        return Err(crate::HrfError::InvalidInput(
            "nonfinite reconstruction input".into(),
        ));
    }
    let result = r_matrix.dot(coefficients);
    if result.iter().any(|v| !v.is_finite()) {
        return Err(crate::HrfError::NumericalError(
            "reconstruction overflow".into(),
        ));
    }
    Ok(result)
}

/// Generate reconstruction matrix for blocked HRF.
///
/// This generates a reconstruction matrix for an HRF that has been
/// convolved with a block of given duration.
///
/// # Arguments
/// * `hrf` - HRF object with basis functions
/// * `times` - Time points at which to evaluate the basis
/// * `block_duration` - Duration of the block in seconds
/// * `precision` - Temporal precision for convolution
///
/// # Returns
/// Matrix with shape (n_samples, nbasis) for blocked HRF
pub fn reconstruction_matrix_blocked(
    hrf: &dyn Hrf,
    times: &TimeArray,
    block_duration: f64,
    precision: f64,
) -> Result<HrfValues> {
    use crate::hrf::HrfExt;

    // Use the HRF's evaluate_block method
    hrf.evaluate_block(times, block_duration, precision, true, false)
}

/// Create design matrix column from stimulus times and HRF.
///
/// This function creates a single column of a design matrix by
/// convolving stimulus onset times with an HRF.
///
/// # Arguments
/// * `hrf` - HRF object
/// * `onsets` - Stimulus onset times
/// * `durations` - Stimulus durations (or None for impulse)
/// * `sframe` - SamplingFrame for output time grid
/// * `basis_index` - Which basis function to use (0-indexed)
///
/// # Returns
/// Column vector for design matrix
pub fn create_regressor(
    hrf: &dyn Hrf,
    onsets: &[f64],
    durations: Option<&[f64]>,
    sframe: &SamplingFrame,
    basis_index: usize,
) -> Result<Array1<f64>> {
    if basis_index >= hrf.nbasis() {
        return Err(crate::error::HrfError::InvalidInput(format!(
            "Basis index {} out of range for HRF with {} basis functions",
            basis_index,
            hrf.nbasis()
        )));
    }

    let times = Array1::from_vec(sframe.samples());
    let mut regressor = Array1::zeros(times.len());

    // Get reconstruction matrix
    let r_matrix = reconstruction_matrix(hrf, &times)?;

    // For each onset, add the HRF response
    for (i, &onset) in onsets.iter().enumerate() {
        let duration = durations.map(|d| d[i]).unwrap_or(0.0);

        if duration > 0.0 {
            // Blocked response
            let shifted_times = &times - onset;
            let blocked_matrix = reconstruction_matrix_blocked(hrf, &shifted_times, duration, 0.1)?;

            // Add this response to the regressor
            for (j, &t) in times.iter().enumerate() {
                if t >= onset && j < blocked_matrix.nrows() {
                    regressor[j] += blocked_matrix[[j, basis_index]];
                }
            }
        } else {
            // Impulse response
            for (j, &t) in times.iter().enumerate() {
                if t >= onset {
                    let hrf_time = t - onset;
                    if hrf_time <= hrf.span() {
                        // Find closest time point in reconstruction matrix
                        let idx = ((hrf_time / times[1]) as usize).min(r_matrix.nrows() - 1);
                        regressor[j] += r_matrix[[idx, basis_index]];
                    }
                }
            }
        }
    }

    Ok(regressor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::functions::{bspline_hrf, spm_canonical, spm_hrf_with_derivatives};
    use crate::sampling::SamplingFrameBuilder;
    use approx::assert_relative_eq;

    #[test]
    fn test_reconstruction_matrix_basic() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 20.0, 101);

        let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();

        assert_eq!(r_matrix.shape(), &[101, 1]);

        // Check peak is around 5-6 seconds
        let peak_idx = r_matrix
            .column(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        let peak_time = times[peak_idx];
        assert!(peak_time > 4.0 && peak_time < 7.0);
    }

    #[test]
    fn test_reconstruction_matrix_multi_basis() {
        let hrf = spm_hrf_with_derivatives();
        let times = Array1::linspace(0.0, 20.0, 101);

        let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();

        assert_eq!(r_matrix.shape(), &[101, 3]);

        // First column should be canonical
        let canonical_peak_idx = r_matrix
            .column(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(times[canonical_peak_idx] > 4.0 && times[canonical_peak_idx] < 7.0);
    }

    #[test]
    fn test_reconstruction_from_sampling() {
        let hrf = bspline_hrf(4, Some(3), Some(10.0));
        let sframe = SamplingFrameBuilder::default()
            .blocklens(vec![50])
            .tr(vec![2.0])
            .build()
            .unwrap();

        let r_matrix = reconstruction_matrix_from_sampling(hrf.as_ref(), &sframe).unwrap();

        assert_eq!(r_matrix.shape(), &[50, 4]);
    }

    #[test]
    fn test_reconstruct_hrf() {
        let hrf = spm_hrf_with_derivatives();
        let times = Array1::linspace(0.0, 20.0, 101);
        let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();

        // Use canonical only
        let coefficients = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let reconstructed = reconstruct_hrf(&r_matrix, &coefficients).unwrap();

        assert_eq!(reconstructed.len(), 101);

        // Should match first column of reconstruction matrix
        for i in 0..101 {
            assert_relative_eq!(reconstructed[i], r_matrix[[i, 0]], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reconstruct_hrf_linear_combo() {
        let hrf = spm_hrf_with_derivatives();
        let times = Array1::linspace(0.0, 20.0, 101);
        let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();

        // Linear combination
        let coefficients = Array1::from_vec(vec![1.0, 0.5, -0.2]);
        let reconstructed = reconstruct_hrf(&r_matrix, &coefficients).unwrap();

        // Manual computation
        for i in 0..101 {
            let expected = r_matrix[[i, 0]] + 0.5 * r_matrix[[i, 1]] - 0.2 * r_matrix[[i, 2]];
            assert_relative_eq!(reconstructed[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_reconstruction_matrix_blocked() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 30.0, 151);

        let r_matrix_impulse = reconstruction_matrix(hrf.as_ref(), &times).unwrap();
        let r_matrix_blocked =
            reconstruction_matrix_blocked(hrf.as_ref(), &times, 10.0, 0.1).unwrap();

        assert_eq!(r_matrix_blocked.shape(), &[151, 1]);

        // Blocked response should be sustained longer
        let sum_impulse: f64 = r_matrix_impulse.sum();
        let sum_blocked: f64 = r_matrix_blocked.sum();
        assert!(sum_blocked > sum_impulse);
    }

    #[test]
    fn test_create_regressor_impulse() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let sframe = SamplingFrameBuilder::default()
            .blocklens(vec![100])
            .tr(vec![1.0])
            .build()
            .unwrap();

        let onsets = vec![10.0, 30.0, 50.0];
        let regressor = create_regressor(hrf.as_ref(), &onsets, None, &sframe, 0).unwrap();

        assert_eq!(regressor.len(), 100);

        // Should have peaks after each onset
        assert!(regressor[15] > regressor[10]); // Peak after first onset
        assert!(regressor[35] > regressor[30]); // Peak after second onset
        assert!(regressor[55] > regressor[50]); // Peak after third onset
    }

    #[test]
    fn test_dimension_mismatch() {
        let hrf = spm_hrf_with_derivatives();
        let times = Array1::linspace(0.0, 20.0, 101);
        let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();

        // Wrong number of coefficients
        let coefficients = Array1::from_vec(vec![1.0, 0.5]); // Only 2, need 3
        let result = reconstruct_hrf(&r_matrix, &coefficients);

        assert!(result.is_err());
    }
}
