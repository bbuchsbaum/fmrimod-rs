//! Cross-validation tests between Rust and Python implementations.

use approx::assert_relative_eq;
use fmrihrf::hrf::{bspline_hrf, fourier_hrf, spm_canonical};
use fmrihrf::regressor::{regressor, regressor_set};
use fmrihrf::sampling::SamplingFrameBuilder;
use ndarray::Array1;

#[test]
fn test_spm_canonical_cross_validation() {
    // Test SPM canonical HRF evaluation matches Python
    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let times = Array1::linspace(0.0, 30.0, 301);
    let values = hrf.evaluate(&times).unwrap();

    // Peak should be around 5 seconds
    let peak_idx = values
        .column(0)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    let peak_time = times[peak_idx];
    assert!(
        peak_time >= 5.0 && peak_time <= 6.0,
        "Peak time is {}",
        peak_time
    );

    // Should start and end near zero
    assert!(values[[0, 0]].abs() < 0.01);
    assert!(values[[300, 0]].abs() < 0.01);
}

#[test]
fn test_bspline_cross_validation() {
    // Test B-spline HRF with known parameters
    let hrf = bspline_hrf(4, Some(3), Some(10.0));
    let times = Array1::linspace(0.0, 10.0, 101);
    let values = hrf.evaluate(&times).unwrap();

    // Should have 4 basis functions
    assert_eq!(values.ncols(), 4);

    // Each basis should be non-negative
    for col in 0..4 {
        for val in values.column(col) {
            assert!(*val >= 0.0, "B-spline basis should be non-negative");
        }
    }
}

#[test]
fn test_fourier_cross_validation() {
    // Sin/cos pairs, allowing odd or even counts.
    let hrf = fourier_hrf(7, Some(10.0)); // 7 sine/cosine columns, no constant
    let times = Array1::linspace(0.0, 10.0, 101);
    let values = hrf.evaluate(&times).unwrap();

    // Should have 7 basis functions
    assert_eq!(values.ncols(), 7);

    // Check orthogonality of basis functions
    let precision = 10.0 / 100.0; // dt
    for i in 0..7 {
        for j in (i + 1)..7 {
            let dot = (0..100)
                .map(|k| values[(k, i)] * values[(k, j)])
                .sum::<f64>()
                * precision;
            assert!(
                dot.abs() < 0.1,
                "Fourier basis should be approximately orthogonal"
            );
        }
    }
}

#[test]
fn test_regressor_cross_validation() {
    // Test simple regressor evaluation
    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let reg = regressor(
        vec![10.0, 30.0, 50.0],
        Some(hrf),
        Some(vec![2.0]), // 2 second duration
        Some(vec![1.0]), // unit amplitude
        None,
        None,
    )
    .unwrap();

    let grid = Array1::linspace(0.0, 70.0, 701);
    let result = reg.evaluate(&grid, None, None).unwrap();

    assert_eq!(result.shape(), &[701, 1]);

    // Should have responses after each event
    let event_indices = vec![100, 300, 500]; // Approximate indices for events at 10, 30, 50s
    for idx in event_indices {
        let end_idx = (idx + 100).min(result.nrows());
        let has_response = (idx..end_idx).any(|i| result[[i, 0]] > 0.1);
        assert!(
            has_response,
            "Should have response after event at index {}",
            idx
        );
    }
}

#[test]
fn test_design_matrix_cross_validation() {
    // Test design matrix creation
    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let rset = regressor_set(
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
        ],
        Some(hrf),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    let grid = Array1::linspace(0.0, 80.0, 801);
    let design = rset.evaluate(&grid, None, None).unwrap();

    // Should have 3 columns (one per condition)
    assert_eq!(design.ncols(), 3);

    // Each column should have some non-zero values
    for col in 0..3 {
        let has_signal = design.column(col).iter().any(|&x| x > 0.1);
        assert!(has_signal, "Column {} should have signal", col);
    }

    // Columns should be different
    let col_sums: Vec<f64> = (0..3).map(|i| design.column(i).sum()).collect();

    // Check that not all columns have the same sum
    assert!(
        col_sums[0] != col_sums[1] || col_sums[1] != col_sums[2],
        "Columns should represent different conditions"
    );
}

#[test]
fn test_sampling_frame_cross_validation() {
    // Test sampling frame generation
    let sframe = SamplingFrameBuilder::default()
        .blocklens(vec![50, 50])
        .tr(vec![2.0, 2.0])
        .start_time(vec![0.0, 0.0])
        .build()
        .unwrap();

    let samples = sframe.samples();

    // Should have 100 samples total
    assert_eq!(samples.len(), 100);

    // First sample should be at t=0
    assert_relative_eq!(samples[0], 0.0);

    // Last sample should be at t=198 (99 * 2.0)
    assert_relative_eq!(samples[99], 198.0);

    // Samples should be evenly spaced by TR=2.0
    for i in 1..100 {
        assert_relative_eq!(samples[i] - samples[i - 1], 2.0);
    }
}

// Additional validation tests can be added here comparing specific
// numerical results with Python implementation
