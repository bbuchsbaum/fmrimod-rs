//! Event convolution with exact sub-bin placement and duration-scaled quadrature.

use crate::error::{HrfError, Result};
use crate::hrf::{Hrf, HrfExt};
use crate::types::{HrfValues, TimeArray};
use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};
use std::sync::Arc;

/// Numerical engine used to evaluate event responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvolutionMethod {
    /// Select direct or FFT using projected event density and kernel length.
    #[default]
    Auto,
    /// Sparse-aware discrete convolution with linear hat projection of events.
    Direct,
    /// FFT convolution using the same projected events and sampled HRF.
    Fft,
    /// Reference evaluation at exact event lags with trapezoid block integration.
    Loop,
}

impl TryFrom<&str> for ConvolutionMethod {
    type Error = HrfError;
    fn try_from(value: &str) -> Result<Self> {
        match value {
            "conv" | "auto" => Ok(Self::Auto),
            "direct" | "Rconv" => Ok(Self::Direct),
            "fft" => Ok(Self::Fft),
            "loop" => Ok(Self::Loop),
            _ => Err(HrfError::InvalidInput(format!(
                "unknown convolution method: {value}"
            ))),
        }
    }
}

/// Evaluate event responses on a finite grid, preserving its input order.
///
/// Impulses contribute `amplitude * h(t - onset)`. Blocks contribute
/// `amplitude * integral_0^duration h(t - onset - u) du`. When `summate`
/// is false, each block response is divided by its duration; overlapping
/// events are always added. Positive durations smaller than `precision`
/// remain blocks. The result has one column per HRF basis function.
///
/// `conv`/`direct` and `fft` share linear hat projection onto a microtime
/// grid. `loop` evaluates the HRF at exact lags and is a reference engine.
/// Unknown method names and invalid dimensions return errors.
#[allow(clippy::too_many_arguments)]
pub fn convolve_hrf(
    grid: &Array1<f64>,
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    hrf: &dyn Hrf,
    span: f64,
    precision: f64,
    method: &str,
    summate: bool,
) -> Result<HrfValues> {
    convolve_hrf_with_method(
        grid,
        onsets,
        durations,
        amplitudes,
        hrf,
        span,
        precision,
        method.try_into()?,
        summate,
    )
}

/// Typed alternative to [`convolve_hrf`].
#[allow(clippy::too_many_arguments)]
pub fn convolve_hrf_with_method(
    grid: &Array1<f64>,
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    hrf: &dyn Hrf,
    span: f64,
    precision: f64,
    method: ConvolutionMethod,
    summate: bool,
) -> Result<HrfValues> {
    validate_inputs(grid, onsets, durations, amplitudes, span, precision)?;
    let mut result = Array2::zeros((grid.len(), hrf.nbasis()));
    if hrf.nbasis() == 0 {
        return Err(HrfError::InvalidInput(
            "HRF must have at least one basis".into(),
        ));
    }
    if grid.is_empty() || onsets.is_empty() {
        return Ok(result);
    }
    if method == ConvolutionMethod::Loop {
        for ((&onset, &duration), &amplitude) in onsets.iter().zip(durations).zip(amplitudes) {
            if amplitude == 0.0 {
                continue;
            }
            let lags = grid.mapv(|t| t - onset);
            let bounded = FiniteKernel {
                inner: hrf,
                cutoff: span,
            };
            let response = bounded.evaluate_block(&lags, duration, precision, summate, false)?;
            check_shape(&response, grid.len(), hrf.nbasis())?;
            for (i, &lag) in lags.iter().enumerate() {
                if lag >= 0.0 && lag <= span + duration {
                    for b in 0..hrf.nbasis() {
                        result[[i, b]] += amplitude * response[[i, b]];
                    }
                }
            }
        }
        return Ok(result);
    }

    let first = grid.iter().copied().fold(f64::INFINITY, f64::min) - span;
    let last = grid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    // Two guard samples keep clipping of future blocks outside interpolation.
    let n_fine = grid_length(last - first, precision, 3)?;
    let time = Array1::from_shape_fn(n_fine, |i| first + i as f64 * precision);
    let neural = build_neural_input(&time, onsets, durations, amplitudes, precision, summate);
    let n_kernel = grid_length(span, precision, 1)?;
    // Only HRFs with an explicit immutable identity participate in caching.
    // Unknown trait implementations are evaluated directly.
    let kernel = crate::utils::cached_hrf_eval(hrf, span, precision)?;
    check_shape(&kernel, n_kernel, hrf.nbasis())?;
    let method = if method == ConvolutionMethod::Auto {
        let full_len = n_fine
            .checked_add(n_kernel)
            .and_then(|n| n.checked_sub(1))
            .and_then(usize::checked_next_power_of_two)
            .ok_or_else(|| HrfError::InvalidInput("convolution length overflow".into()))?;
        let direct_work = neural.iter().filter(|&&v| v != 0.0).count() as f64 * n_kernel as f64;
        let fft_work = 2.0 * full_len as f64 * (full_len as f64).log2();
        if direct_work > fft_work {
            ConvolutionMethod::Fft
        } else {
            ConvolutionMethod::Direct
        }
    } else {
        method
    };
    for b in 0..kernel.ncols() {
        let column = kernel.column(b).to_owned();
        let convolved = match method {
            ConvolutionMethod::Direct => convolve_prefix(&neural, &column, n_fine),
            ConvolutionMethod::Fft => convolve_fft(&neural, &column)?,
            ConvolutionMethod::Loop | ConvolutionMethod::Auto => unreachable!(),
        };
        for (i, &t) in grid.iter().enumerate() {
            let position = (t - first) / precision;
            let j = (position.floor() as usize).min(n_fine - 2);
            let fraction = position - j as f64;
            result[[i, b]] = convolved[j] + fraction * (convolved[j + 1] - convolved[j]);
        }
    }
    Ok(result)
}

pub(crate) fn grid_length(span: f64, dt: f64, extra: usize) -> Result<usize> {
    let count = (span / dt).floor();
    if !count.is_finite() || count < 0.0 || count >= (isize::MAX as usize / 8 - extra) as f64 {
        return Err(HrfError::InvalidInput(
            "requested temporal grid is too large".into(),
        ));
    }
    Ok(count as usize + extra)
}

fn validate_inputs(
    grid: &Array1<f64>,
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    span: f64,
    precision: f64,
) -> Result<()> {
    if !span.is_finite() || span <= 0.0 || !precision.is_finite() || precision <= 0.0 {
        return Err(HrfError::InvalidInput(
            "span and precision must be positive and finite".into(),
        ));
    }
    if onsets.len() != durations.len() || onsets.len() != amplitudes.len() {
        return Err(HrfError::InvalidInput(
            "onsets, durations and amplitudes must have equal lengths".into(),
        ));
    }
    if grid
        .iter()
        .chain(onsets)
        .chain(durations)
        .chain(amplitudes)
        .any(|v| !v.is_finite())
        || durations.iter().any(|&d| d < 0.0)
        || onsets
            .iter()
            .zip(durations)
            .any(|(&o, &d)| !(o + d).is_finite())
    {
        return Err(HrfError::InvalidInput(
            "times and amplitudes must be finite; durations must be nonnegative".into(),
        ));
    }
    Ok(())
}

pub(crate) fn check_shape(values: &HrfValues, rows: usize, bases: usize) -> Result<()> {
    if values.dim() != (rows, bases) {
        return Err(HrfError::DimensionMismatch {
            expected: format!("{rows} rows and {bases} basis columns"),
            actual: format!("{} rows and {} columns", values.nrows(), values.ncols()),
        });
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::InvalidInput(
            "HRF evaluation returned nonfinite values".into(),
        ));
    }
    Ok(())
}

/// Evaluate and add trial-specific HRFs, using exact-lag reference integration.
#[allow(clippy::too_many_arguments)]
pub fn convolve_hrf_trial_varying(
    grid: &Array1<f64>,
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    hrfs: &[Arc<Box<dyn Hrf>>],
    span: f64,
    precision: f64,
    summate: bool,
) -> Result<HrfValues> {
    validate_inputs(grid, onsets, durations, amplitudes, span, precision)?;
    let first = hrfs
        .first()
        .ok_or_else(|| HrfError::InvalidInput("hrfs must contain at least one HRF".into()))?;
    if hrfs.len() != onsets.len() || hrfs.iter().any(|h| h.nbasis() != first.nbasis()) {
        return Err(HrfError::InvalidInput(
            "one HRF per event with equal basis counts is required".into(),
        ));
    }
    let mut result = Array2::zeros((grid.len(), first.nbasis()));
    for (i, hrf) in hrfs.iter().enumerate() {
        result += &convolve_hrf_with_method(
            grid,
            &onsets[i..=i],
            &durations[i..=i],
            &amplitudes[i..=i],
            hrf.as_ref().as_ref(),
            span,
            precision,
            ConvolutionMethod::Loop,
            summate,
        )?;
    }
    Ok(result)
}

// Running integral of a linear hat basis, accumulated into a difference array.
fn add_hat_step(diff: &mut [f64], position: f64, coefficient: f64, dt: f64, n: usize) {
    if position <= 0.0 {
        return;
    }
    let position = position.min((n - 1) as f64);
    let a = position.floor() as usize;
    let f = position - a as f64;
    let v = coefficient * dt;
    if a > 0 {
        diff[0] += v;
        diff[a] -= v;
    }
    let w0 = v * (0.5 + f - 0.5 * f * f);
    let w1 = v * 0.5 * f * f;
    diff[a] += w0;
    diff[a + 1] += w1 - w0;
    diff[a + 2] -= w1;
}

fn build_neural_input(
    time: &Array1<f64>,
    onsets: &[f64],
    durations: &[f64],
    amplitudes: &[f64],
    dt: f64,
    summate: bool,
) -> Array1<f64> {
    let n = time.len();
    let mut diff = vec![0.0; n + 2];
    for ((&onset, &duration), &amplitude) in onsets.iter().zip(durations).zip(amplitudes) {
        let start = (onset - time[0]) / dt;
        if start > (n - 1) as f64 || amplitude == 0.0 {
            continue;
        }
        if duration == 0.0 {
            if start < 0.0 {
                continue;
            }
            let a = start.floor() as usize;
            let f = start - a as f64;
            let w0 = amplitude * (1.0 - f);
            let w1 = amplitude * f;
            diff[a] += w0;
            diff[a + 1] += w1 - w0;
            diff[a + 2] -= w1;
        } else {
            let end = (onset + duration - time[0]) / dt;
            let scale = if summate {
                amplitude
            } else {
                amplitude / duration
            };
            add_hat_step(&mut diff, end, scale, dt, n);
            add_hat_step(&mut diff, start, -scale, dt, n);
        }
    }
    let mut sum = 0.0;
    Array1::from_iter(diff.into_iter().take(n).map(|change| {
        sum += change;
        sum
    }))
}

fn convolve_prefix(signal: &Array1<f64>, kernel: &Array1<f64>, len: usize) -> Array1<f64> {
    let mut result = vec![0.0; len];
    for (i, &amplitude) in signal.iter().take(len).enumerate() {
        if amplitude == 0.0 {
            continue;
        }
        for (out, &h) in result[i..].iter_mut().zip(kernel) {
            *out += amplitude * h;
        }
    }
    Array1::from_vec(result)
}

/// Direct convolution using simple algorithm.
#[cfg(test)]
fn convolve_direct(signal: &Array1<f64>, kernel: &Array1<f64>) -> Array1<f64> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_output = n_signal + n_kernel - 1;

    let mut output = Array1::zeros(n_output);

    // Perform convolution
    for i in 0..n_signal {
        for j in 0..n_kernel {
            output[i + j] += signal[i] * kernel[j];
        }
    }

    output
}

/// FFT-based convolution for efficiency with long signals.
fn convolve_fft(signal: &Array1<f64>, kernel: &Array1<f64>) -> Result<Array1<f64>> {
    let n_signal = signal.len();
    let n_kernel = kernel.len();
    let n_output = n_signal + n_kernel - 1;

    // Determine optimal FFT size (next power of 2)
    let n_fft = next_power_of_2(n_output);

    // Convert to complex and pad
    let mut signal_complex: Vec<Complex<f64>> =
        signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    signal_complex.resize(n_fft, Complex::new(0.0, 0.0));

    let mut kernel_complex: Vec<Complex<f64>> =
        kernel.iter().map(|&x| Complex::new(x, 0.0)).collect();
    kernel_complex.resize(n_fft, Complex::new(0.0, 0.0));

    // Create FFT planner
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    let ifft = planner.plan_fft_inverse(n_fft);

    // Forward FFT
    fft.process(&mut signal_complex);
    fft.process(&mut kernel_complex);

    // Multiply in frequency domain
    for i in 0..n_fft {
        signal_complex[i] *= kernel_complex[i];
    }

    // Inverse FFT
    ifft.process(&mut signal_complex);

    // Normalize and extract real part
    let norm = 1.0 / n_fft as f64;
    let result: Vec<f64> = signal_complex
        .iter()
        .take(n_output)
        .map(|c| c.re * norm)
        .collect();

    Ok(Array1::from_vec(result))
}

/// Find the next power of 2 greater than or equal to n.
fn next_power_of_2(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    let mut power = 1;
    while power < n {
        power *= 2;
    }
    power
}

/// Linear interpolation.
#[cfg(test)]
fn interp_linear(x: &Array1<f64>, y: &[f64], xi: f64) -> f64 {
    if xi <= x[0] {
        return y[0];
    }

    let n = x.len();
    if xi >= x[n - 1] {
        return y[n - 1];
    }

    // Binary search for the interval
    let mut left = 0;
    let mut right = n - 1;

    while right - left > 1 {
        let mid = (left + right) / 2;
        if x[mid] <= xi {
            left = mid;
        } else {
            right = mid;
        }
    }

    // Linear interpolation
    let t = (xi - x[left]) / (x[right] - x[left]);
    y[left] * (1.0 - t) + y[right] * t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::functions::spm_canonical;
    use approx::assert_relative_eq;

    #[test]
    fn test_convolve_impulse() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let grid = Array1::linspace(0.0, 30.0, 301);

        // Single impulse at t=5
        let onsets = vec![5.0];
        let durations = vec![0.0];
        let amplitudes = vec![1.0];

        let result = convolve_hrf(
            &grid,
            &onsets,
            &durations,
            &amplitudes,
            hrf.as_ref(),
            20.0,
            0.1,
            "direct",
            true,
        )
        .unwrap();

        // Peak should be around t=10 (5 + 5 for SPM peak)
        // Since result is 2D, get the first column
        let result_col = result.column(0);
        let peak_idx = result_col
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;

        let peak_time = grid[peak_idx];
        assert!(peak_time > 9.0 && peak_time < 12.0);
    }

    #[test]
    fn test_convolve_block() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let grid = Array1::linspace(0.0, 40.0, 401);

        // Block from t=10 to t=15
        let onsets = vec![10.0];
        let durations = vec![5.0];
        let amplitudes = vec![1.0];

        let result = convolve_hrf(
            &grid,
            &onsets,
            &durations,
            &amplitudes,
            hrf.as_ref(),
            20.0,
            0.1,
            "direct",
            true,
        )
        .unwrap();

        // Response should be sustained longer than impulse
        let result_col = result.column(0);
        let above_half_max: Vec<usize> = result_col
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.5)
            .map(|(idx, _)| idx)
            .collect();

        // Should have sustained response
        assert!(above_half_max.len() > 50); // More than 5 seconds at 0.1s resolution
    }

    #[test]
    fn test_fft_convolution() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let kernel = Array1::from_vec(vec![0.5, 0.5]);

        let result_direct = convolve_direct(&signal, &kernel);
        let result_fft = convolve_fft(&signal, &kernel).unwrap();

        // Results should match
        for i in 0..result_direct.len() {
            assert_relative_eq!(result_direct[i], result_fft[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(2), 2);
        assert_eq!(next_power_of_2(3), 4);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(1000), 1024);
    }

    #[test]
    fn test_build_neural_input_edge_cases() {
        // Test 1: Events at exact bin boundaries
        let time = Array1::linspace(0.0, 10.0, 101);
        let onsets = vec![0.0, 5.0, 10.0];
        let durations = vec![0.0, 0.0, 0.0];
        let amplitudes = vec![1.0, 2.0, 3.0];

        let neural = build_neural_input(&time, &onsets, &durations, &amplitudes, 0.1, true);

        assert_eq!(neural.len(), 101);
        assert_eq!(neural[0], 1.0); // First onset
        assert_eq!(neural[50], 2.0); // Second onset at t=5
                                     // Last onset might be at boundary

        // Test 2: Block events with overlapping durations
        let onsets2 = vec![2.0, 4.0];
        let durations2 = vec![3.0, 3.0]; // Overlapping from 4-5
        let amplitudes2 = vec![1.0, 1.0];

        let neural2 = build_neural_input(&time, &onsets2, &durations2, &amplitudes2, 0.1, true);

        // During overlap (t=4 to t=5), should sum to 2.0
        let overlap_start = 40;
        let overlap_end = 50;
        for i in overlap_start + 1..overlap_end {
            assert_relative_eq!(neural2[i], 0.2, epsilon = 1e-6);
        }

        // Test 3: Events before grid start
        let onsets3 = vec![-5.0, -2.0, 1.0];
        let durations3 = vec![10.0, 2.0, 1.0]; // First extends into grid
        let amplitudes3 = vec![0.5, 1.0, 2.0];

        let neural3 = build_neural_input(&time, &onsets3, &durations3, &amplitudes3, 0.1, true);

        // First event should affect early time points
        // Both first (0.5) and second (1.0) events are active at t=0
        assert_relative_eq!(neural3[0], 0.05, epsilon = 1e-6);
        // At t=1, first (0.5) and third (2.0) events are active
        assert_relative_eq!(neural3[10], 0.15, epsilon = 1e-6);
    }

    #[test]
    fn test_interp_linear_edge_cases() {
        let x = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0]);
        let y = vec![0.0, 10.0, 20.0, 15.0, 5.0];

        // Test exact points
        assert_eq!(interp_linear(&x, &y, 0.0), 0.0);
        assert_eq!(interp_linear(&x, &y, 2.0), 20.0);

        // Test interpolation
        assert_relative_eq!(interp_linear(&x, &y, 1.5), 15.0, epsilon = 1e-6);
        assert_relative_eq!(interp_linear(&x, &y, 2.5), 17.5, epsilon = 1e-6);

        // Test extrapolation (clamps to endpoints)
        assert_eq!(interp_linear(&x, &y, -1.0), 0.0);
        assert_eq!(interp_linear(&x, &y, 5.0), 5.0);

        // Test with single point
        let x_single = Array1::from_vec(vec![5.0]);
        let y_single = vec![42.0];
        assert_eq!(interp_linear(&x_single, &y_single, 0.0), 42.0);
        assert_eq!(interp_linear(&x_single, &y_single, 10.0), 42.0);
    }

    #[test]
    fn test_convolve_precision_error() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let grid = Array1::linspace(0.0, 30.0, 100);

        // Test with invalid precision
        let result = convolve_hrf(
            &grid,
            &[10.0],
            &[0.0],
            &[1.0],
            hrf.as_ref(),
            20.0,
            0.0, // Invalid precision
            "conv",
            true,
        );

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), HrfError::InvalidInput(_)));

        // Test with negative precision
        let result2 = convolve_hrf(
            &grid,
            &[10.0],
            &[0.0],
            &[1.0],
            hrf.as_ref(),
            20.0,
            -0.1, // Negative precision
            "conv",
            true,
        );

        assert!(result2.is_err());
    }

    #[test]
    fn test_convolve_method_variations() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let grid = Array1::linspace(0.0, 50.0, 200);
        let onsets = vec![10.0, 25.0, 40.0];
        let durations = vec![2.0, 0.0, 3.0];
        let amplitudes = vec![1.0, 0.5, 1.5];

        // Test different methods produce similar results
        let methods = vec!["conv", "direct", "fft", "Rconv"];
        let mut results = Vec::new();

        for method in methods {
            let result = convolve_hrf(
                &grid,
                &onsets,
                &durations,
                &amplitudes,
                hrf.as_ref(),
                20.0,
                0.1,
                method,
                true,
            )
            .unwrap();
            results.push(result);
        }

        // conv, direct, and unknown should be identical
        for i in 0..grid.len() {
            assert_relative_eq!(results[0][[i, 0]], results[1][[i, 0]], epsilon = 1e-10);
            assert_relative_eq!(results[0][[i, 0]], results[3][[i, 0]], epsilon = 1e-10);
        }

        // FFT should be very close to direct convolution
        for i in 0..grid.len() {
            assert_relative_eq!(results[0][[i, 0]], results[2][[i, 0]], epsilon = 1e-6);
        }
    }
}

/// Support masking precedes block quadrature, so unsupported kernel tails do
/// not leak into a block that straddles the cutoff.
#[derive(Debug)]
struct FiniteKernel<'a> {
    inner: &'a dyn Hrf,
    cutoff: f64,
}
impl Hrf for FiniteKernel<'_> {
    fn evaluate(&self, times: &TimeArray) -> Result<HrfValues> {
        let mut values = self.inner.evaluate(times)?;
        check_shape(&values, times.len(), self.nbasis())?;
        for (i, &t) in times.iter().enumerate() {
            if t < 0.0 || t > self.cutoff {
                values.row_mut(i).fill(0.0);
            }
        }
        Ok(values)
    }
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }
    fn span(&self) -> f64 {
        self.cutoff
    }
    fn params(&self) -> Vec<(&str, f64)> {
        self.inner.params()
    }
}
