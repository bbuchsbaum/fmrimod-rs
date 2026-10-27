//! Design matrix convolution functionality.

use crate::convolve::get_hrf_array;
use crate::design::DesignMatrix;
use crate::types::ValidationError;
use fmri_hrf::{hrf as hrf_lib, Hrf};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Options for design matrix convolution
#[derive(Debug, Clone)]
pub struct ConvolveDesignOptions {
    /// Sampling rate in seconds (TR)
    pub sampling_rate: Option<f64>,
    /// Time points for the design matrix rows
    pub time_points: Option<Array1<f64>>,
    /// Column names for the result
    pub column_names: Option<Vec<String>>,
    /// HRF duration for evaluation (default 30.0 seconds)
    pub hrf_duration: f64,
    /// Whether to normalize HRF (default true)
    pub normalize_hrf: bool,
}

impl Default for ConvolveDesignOptions {
    fn default() -> Self {
        Self {
            sampling_rate: None,
            time_points: None,
            column_names: None,
            hrf_duration: 30.0,
            normalize_hrf: true,
        }
    }
}

/// Convolve a design matrix with a hemodynamic response function.
///
/// This function takes an existing design matrix and convolves each column
/// with the specified HRF. Useful for applying HRF convolution to matrices
/// that were constructed without convolution.
///
/// # Arguments
/// * `design_matrix` - The design matrix to convolve (n_timepoints x n_regressors)
/// * `hrf` - Hemodynamic response function
/// * `options` - Convolution options
///
/// # Returns
/// Convolved design matrix with same dimensions
pub fn convolve_design_matrix(
    design_matrix: &Array2<f64>,
    hrf: &dyn Hrf,
    options: &ConvolveDesignOptions,
) -> Result<Array2<f64>, ValidationError> {
    let n_timepoints = design_matrix.nrows();
    let n_cols = design_matrix.ncols();

    if n_timepoints == 0 || n_cols == 0 {
        return Err(ValidationError::EmptyArray);
    }

    if design_matrix.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::NonFiniteValues);
    }
    let grid_step = if let Some(times) = &options.time_points {
        if times.len() != n_timepoints || times.len() < 2 || times.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::InvalidParameter(
                "sampled convolution needs one finite time per row and at least two times".into(),
            ));
        }
        let step = times[1] - times[0];
        if step <= 0.0
            || !step.is_finite()
            || times
                .windows(2)
                .into_iter()
                .any(|w| (w[1] - w[0] - step).abs() > 1e-10 * step)
        {
            return Err(ValidationError::InvalidParameter("sampled convolution requires a uniform increasing grid; use event convolution for arbitrary times".into()));
        }
        Some(step)
    } else {
        None
    };
    let sampling_rate = options.sampling_rate.or(grid_step).ok_or_else(|| {
        ValidationError::InvalidParameter("sampling_rate or time_points is required".into())
    })?;
    if !sampling_rate.is_finite()
        || sampling_rate <= 0.0
        || grid_step.is_some_and(|dt| (dt - sampling_rate).abs() > 1e-10 * dt)
    {
        return Err(ValidationError::InvalidParameter(
            "sampling_rate must be a positive finite TR consistent with the time grid".into(),
        ));
    }

    // Get HRF array
    let hrf_sr = 1.0 / sampling_rate; // Convert TR to Hz
    let hrf_array = get_hrf_array(hrf, hrf_sr, options.hrf_duration)?;

    // Normalize HRF if requested
    let hrf_normalized = if options.normalize_hrf {
        let hrf_sum: f64 = hrf_array.sum();
        if hrf_sum.abs() > 1e-10 {
            &hrf_array / hrf_sum
        } else {
            // Fall back to max normalization
            let hrf_max = hrf_array.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
            if hrf_max > 0.0 {
                &hrf_array / hrf_max
            } else {
                hrf_array.clone()
            }
        }
    } else {
        hrf_array.clone()
    };

    // Convolve each column
    let mut result = Array2::zeros((n_timepoints, n_cols));

    for i in 0..n_cols {
        let col = design_matrix.column(i);

        // Convolve with HRF using 'full' mode for causal convolution
        let convolved_full = convolve_full(&col.to_owned(), &hrf_normalized);

        // Trim to original length (keep only first n_timepoints)
        // This ensures causality - HRF response follows stimulus
        let convolved: Array1<f64> = convolved_full.slice(ndarray::s![..n_timepoints]).to_owned();

        result.column_mut(i).assign(&convolved);
    }

    Ok(result)
}

/// Full convolution (not same mode)
fn convolve_full(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();
    let result_len = signal_len + kernel_len - 1;

    let mut result = Array1::zeros(result_len);

    // Perform convolution
    for i in 0..signal_len {
        for j in 0..kernel_len {
            result[i + j] += signal[i] * kernel[j];
        }
    }

    result
}

/// Convolve a DesignMatrix with HRF
pub fn convolve_design(
    design: &DesignMatrix,
    hrf: &dyn Hrf,
    options: Option<ConvolveDesignOptions>,
) -> Result<DesignMatrix, ValidationError> {
    let options = options.unwrap_or_default();

    // Set up options with DesignMatrix metadata
    let mut opts = options;
    if opts.time_points.is_none() {
        opts.time_points = Some(design.row_times.clone());
    }
    if opts.column_names.is_none() {
        opts.column_names = Some(design.column_names.clone());
    }

    // Convolve the data
    let convolved_data = convolve_design_matrix(&design.data, hrf, &opts)?;

    // Create new DesignMatrix with convolved data
    Ok(DesignMatrix {
        data: convolved_data,
        column_names: design.column_names.clone(),
        row_times: design.row_times.clone(),
    })
}

/// Convolve a dictionary of regressors with an HRF
pub fn convolve_regressors(
    regressors: &HashMap<String, Array1<f64>>,
    hrf: &dyn Hrf,
    sampling_rate: f64,
) -> Result<HashMap<String, Array1<f64>>, ValidationError> {
    let mut convolved = HashMap::new();

    let options = ConvolveDesignOptions {
        sampling_rate: Some(sampling_rate),
        ..Default::default()
    };

    for (name, regressor) in regressors {
        // Convert to 2D for convolution
        let reg_2d = regressor.clone().insert_axis(Axis(1));

        // Convolve
        let conv_2d = convolve_design_matrix(&reg_2d, hrf, &options)?;

        // Extract as 1D
        let conv_1d = conv_2d.column(0).to_owned();

        convolved.insert(name.clone(), conv_1d);
    }

    Ok(convolved)
}

/// Create a default SPM canonical HRF for convolution
pub fn default_design_hrf() -> Box<dyn Hrf> {
    hrf_lib::spm_canonical(None, None, None, None, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_convolve_full() {
        let signal = array![1.0, 0.0, 0.0];
        let kernel = array![0.5, 1.0];
        let result = convolve_full(&signal, &kernel);

        // Expected: [0.5, 1.0, 0.0, 0.0]
        assert_eq!(result.len(), 4);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.0).abs() < 1e-10);
        assert!((result[3] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_design_matrix() {
        // Create simple design matrix with impulses
        let mut dm = Array2::zeros((10, 2));
        dm[[0, 0]] = 1.0; // Impulse at t=0 in column 0
        dm[[5, 1]] = 1.0; // Impulse at t=5 in column 1

        let hrf = default_design_hrf();
        let options = ConvolveDesignOptions {
            sampling_rate: Some(1.0),
            ..Default::default()
        };

        let result = convolve_design_matrix(&dm, &*hrf, &options).unwrap();

        // Check shape preserved
        assert_eq!(result.shape(), dm.shape());

        // Check that convolution produced non-zero values after impulses
        assert!(result[[1, 0]].abs() > 1e-10); // Response after first impulse
        assert!(result[[6, 1]].abs() > 1e-10); // Response after second impulse
    }

    #[test]
    fn test_convolve_regressors() {
        let mut regressors = HashMap::new();

        let mut reg1 = Array1::zeros(10);
        reg1[0] = 1.0;
        regressors.insert("stim1".to_string(), reg1);

        let mut reg2 = Array1::zeros(10);
        reg2[5] = 1.0;
        regressors.insert("stim2".to_string(), reg2);

        let hrf = default_design_hrf();
        let result = convolve_regressors(&regressors, &*hrf, 1.0).unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.contains_key("stim1"));
        assert!(result.contains_key("stim2"));

        // Check convolution effects
        assert!(result["stim1"][1].abs() > 1e-10);
        assert!(result["stim2"][6].abs() > 1e-10);
    }
}
