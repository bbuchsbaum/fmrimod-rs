//! Tests for convolution functionality

use approx::assert_relative_eq;
use fmridesign::convolve::{discrete_convolve, ConvolveOptions};
use fmridesign::events::EventVariable;
use fmridesign::types::EventSpec;
use ndarray::{array, Array1};

#[test]
fn test_convolve_array_basic() {
    // Simple impulse response
    let signal = array![1.0, 0.0, 0.0, 0.0, 0.0];
    let kernel = array![1.0, 0.5, 0.25];

    let result = discrete_convolve(&signal, &kernel);

    // Expected: [1.0, 0.5, 0.25, 0.0, 0.0]
    assert_eq!(result.len(), 5);
    assert_relative_eq!(result[0], 1.0);
    assert_relative_eq!(result[1], 0.5);
    assert_relative_eq!(result[2], 0.25);
    assert_relative_eq!(result[3], 0.0);
    assert_relative_eq!(result[4], 0.0);
}

#[test]
fn test_convolve_array_multiple_impulses() {
    let signal = array![1.0, 0.0, 1.0, 0.0, 0.0];
    let kernel = array![1.0, 0.5];

    let result = discrete_convolve(&signal, &kernel);

    // Expected: [1.0, 0.5, 1.0, 0.5, 0.0]
    assert_eq!(result.len(), 5);
    assert_relative_eq!(result[0], 1.0);
    assert_relative_eq!(result[1], 0.5);
    assert_relative_eq!(result[2], 1.0);
    assert_relative_eq!(result[3], 0.5);
    assert_relative_eq!(result[4], 0.0);
}

#[test]
fn test_convolve_array_edge_cases() {
    // Empty signal
    let signal = Array1::<f64>::zeros(0);
    let kernel = array![1.0, 0.5];
    let result = discrete_convolve(&signal, &kernel);
    assert_eq!(result.len(), 0);

    // Empty kernel
    let signal = array![1.0, 2.0, 3.0];
    let kernel = Array1::<f64>::zeros(0);
    let result = discrete_convolve(&signal, &kernel);
    assert_eq!(result.len(), 3);
    for val in result.iter() {
        assert_relative_eq!(*val, 0.0);
    }
}

#[test]
fn test_convolve_options_default() {
    let options = ConvolveOptions::default();
    assert_eq!(options.sampling_rate, 1.0);
    assert_eq!(options.hrf_duration, 32.0);
    assert_eq!(options.precision, 0.1);
    assert_eq!(options.method, "conv");
}

#[test]
fn test_full_convolution_workflow() {
    // Create events
    let onsets = vec![1.0, 5.0, 10.0];
    let values = vec![1.0, 1.0, 1.0];
    let durations = vec![0.0, 0.0, 0.0];
    let event = EventVariable::new("impulse", onsets, values, durations).unwrap();

    // Create HRF kernel (simplified)
    let t = Array1::linspace(0.0, 15.0, 31);
    let hrf_kernel = t.mapv(|x: f64| {
        if x < 1.0 {
            0.0
        } else {
            (x - 1.0).exp() * (-0.5 * (x - 1.0)).exp()
        }
    });

    // TODO: Complete convolution test once full convolution API is implemented
    // This would involve:
    // 1. Creating a sampling frame
    // 2. Converting events to indicator function
    // 3. Convolving with HRF
    // 4. Downsampling to scan times
}

#[test]
fn test_convolve_array_scaling() {
    // Test that convolution preserves scaling
    let signal = array![2.0, 0.0, 0.0];
    let kernel = array![1.0, 0.5];

    let result = discrete_convolve(&signal, &kernel);

    assert_relative_eq!(result[0], 2.0);
    assert_relative_eq!(result[1], 1.0);
    assert_relative_eq!(result[2], 0.0);
}
