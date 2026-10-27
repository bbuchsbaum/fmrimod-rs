//! Direct convolution functions for events with HRF.

use crate::types::{Array, EventSpec, ValidationError};
use fmri_hrf::{hrf as hrf_lib, Hrf};
use ndarray::{Array1, Array2};

/// Options for convolution
#[derive(Debug, Clone)]
pub struct ConvolveOptions {
    /// Sampling rate in Hz (default 1.0)
    pub sampling_rate: f64,
    /// Total duration in seconds (if None, computed from events)
    pub total_duration: Option<f64>,
    /// HRF duration for truncation (default 32.0)
    pub hrf_duration: f64,
    /// Temporal precision for convolution (default 0.1)
    pub precision: f64,
    /// Convolution method ('conv', 'fft', 'direct')
    pub method: String,
}

impl Default for ConvolveOptions {
    fn default() -> Self {
        Self {
            sampling_rate: 1.0,
            total_duration: None,
            hrf_duration: 32.0,
            precision: 0.1,
            method: "conv".to_string(),
        }
    }
}

/// Trait for objects that can be convolved with an HRF
pub trait Convolvable {
    /// Convolve with HRF
    fn convolve(&self, hrf: &dyn Hrf, options: &ConvolveOptions) -> Result<Array, ValidationError>;
}

/// Get HRF values for convolution
pub fn get_hrf_array(
    hrf: &dyn Hrf,
    sampling_rate: f64,
    duration: f64,
) -> Result<Array1<f64>, ValidationError> {
    if !sampling_rate.is_finite()
        || sampling_rate <= 0.0
        || !duration.is_finite()
        || duration <= 0.0
    {
        return Err(ValidationError::InvalidParameter(
            "sampling rate and duration must be positive and finite".into(),
        ));
    }
    if hrf.nbasis() != 1 {
        return Err(ValidationError::InvalidParameter("a single-basis HRF is required by this scalar-kernel API; use event convolution for multiple bases".into()));
    }
    let count = (duration * sampling_rate).floor();
    if !count.is_finite() || count >= (isize::MAX as usize / std::mem::size_of::<f64>()) as f64 {
        return Err(ValidationError::InvalidParameter(
            "HRF sample count is too large".into(),
        ));
    }
    let count = count as usize + 1;
    let t = Array1::from_shape_fn(count, |i| i as f64 / sampling_rate);
    let hrf_values = hrf
        .evaluate(&t)
        .map_err(|e| ValidationError::ComputationError(format!("HRF evaluation failed: {}", e)))?;

    if hrf_values.dim() != (count, 1) || hrf_values.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::ComputationError(
            "HRF values must be finite and match the advertised shape".into(),
        ));
    }
    Ok(hrf_values.column(0).to_owned())
}

/// Discrete convolution (same mode)
pub fn discrete_convolve(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let signal_len = signal.len();
    let kernel_len = kernel.len();

    if signal_len == 0 || kernel_len == 0 {
        return Array1::zeros(signal_len);
    }

    // Standard discrete convolution
    let mut result = Array1::zeros(signal_len);

    for i in 0..signal_len {
        let mut sum = 0.0;
        for j in 0..kernel_len.min(i + 1) {
            if i >= j {
                sum += signal[i - j] * kernel[j];
            }
        }
        result[i] = sum;
    }

    result
}

impl<T: EventSpec + ?Sized> Convolvable for T {
    fn convolve(&self, hrf: &dyn Hrf, options: &ConvolveOptions) -> Result<Array, ValidationError> {
        if !options.sampling_rate.is_finite() || options.sampling_rate <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "sampling rate must be positive and finite".into(),
            ));
        }
        let last = self
            .onsets()
            .iter()
            .zip(self.durations())
            .map(|(&o, &d)| o + d)
            .fold(0.0_f64, f64::max);
        let duration = options
            .total_duration
            .unwrap_or(last + options.hrf_duration);
        let count = (duration * options.sampling_rate).floor();
        if !duration.is_finite()
            || duration < 0.0
            || !count.is_finite()
            || count > (isize::MAX as usize / 8) as f64
        {
            return Err(ValidationError::InvalidParameter(
                "invalid total duration or sample count".into(),
            ));
        }
        let grid = Array1::from_shape_fn(count as usize, |i| i as f64 / options.sampling_rate);
        convolve_at_times(
            self,
            hrf,
            &grid,
            options.hrf_duration,
            options.precision,
            &options.method,
        )
    }
}

/// Convolve any event type implementing [`EventSpec::event_amplitudes`].
pub fn convolve(
    event: &dyn EventSpec,
    hrf: &dyn Hrf,
    options: &ConvolveOptions,
) -> Result<Array, ValidationError> {
    event.convolve(hrf, options)
}

/// Evaluate an event at exact requested times, retaining every HRF basis.
/// Columns are basis-major, matching [`crate::naming::add_basis_suffix`].
pub fn convolve_at_times<T: EventSpec + ?Sized>(
    event: &T,
    hrf: &dyn Hrf,
    grid: &Array1<f64>,
    span: f64,
    precision: f64,
    method: &str,
) -> Result<Array, ValidationError> {
    let amplitudes = event.event_amplitudes()?;
    if amplitudes.nrows() != event.onsets().len() {
        return Err(ValidationError::DimensionError {
            expected: event.onsets().len(),
            actual: amplitudes.nrows(),
        });
    }
    let columns = amplitudes
        .ncols()
        .checked_mul(hrf.nbasis())
        .ok_or_else(|| ValidationError::InvalidParameter("too many convolution columns".into()))?;
    let mut output = Array2::zeros((grid.len(), columns));
    let onsets = event.onsets().to_vec();
    let durations = event.durations().to_vec();
    for c in 0..amplitudes.ncols() {
        let values = fmri_hrf::regressor::convolve_hrf(
            grid,
            &onsets,
            &durations,
            &amplitudes.column(c).to_vec(),
            hrf,
            span,
            precision,
            method,
            true,
        )
        .map_err(|e| ValidationError::ComputationError(e.to_string()))?;
        for b in 0..hrf.nbasis() {
            output
                .column_mut(b * amplitudes.ncols() + c)
                .assign(&values.column(b));
        }
    }
    Ok(output)
}

/// Create a default SPM canonical HRF
pub fn default_hrf() -> Box<dyn Hrf> {
    hrf_lib::spm_canonical(None, None, None, None, None, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventVariable;
    use ndarray::array;

    #[test]
    fn test_discrete_convolve() {
        let signal = array![1.0, 0.0, 0.0, 0.0, 0.0];
        let kernel = array![0.5, 1.0, 0.5];
        let result = discrete_convolve(&signal, &kernel);

        // The convolution of [1.0, 0.0, 0.0, 0.0, 0.0] with [0.5, 1.0, 0.5]
        // Result: [0.5, 1.0, 0.5, 0.0, 0.0]
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
        assert!((result[3] - 0.0).abs() < 1e-10);
        assert!((result[4] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_event_variable() {
        let event = EventVariable::new(
            "test",
            vec![1.0, 5.0, 10.0],
            vec![1.0, 2.0, 3.0],
            vec![2.0, 2.0, 2.0],
        )
        .unwrap();

        let hrf = default_hrf();
        let options = ConvolveOptions::default();

        let result = event.convolve(&*hrf, &options).unwrap();

        // Check shape
        assert_eq!(result.ndim(), 2);
        assert_eq!(result.ncols(), 1); // Single column for continuous event

        // Check that convolution produces non-zero values
        assert!(result.iter().any(|&x| x.abs() > 1e-10));
    }
}
