//! Integration tests for fmrihrf library.

use fmrihrf::{hrf::spm_canonical, sampling::SamplingFrame, HrfExt};
use ndarray::Array1;

#[test]
fn test_spm_canonical_basic() {
    // Create SPM canonical HRF
    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    // Create time points
    let t = Array1::linspace(0.0, 20.0, 21);

    // Evaluate HRF
    let values = hrf.evaluate(&t).unwrap();

    // Check dimensions
    assert_eq!(values.shape(), &[21, 1]);

    // Check that it's normalized to peak of 1
    let max_val = values.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!((max_val - 1.0).abs() < 1e-6);

    // Check that values at t=0 and t=20 are small
    assert!(values[[0, 0]].abs() < 0.01);
    assert!(values[[20, 0]].abs() < 0.1); // SPM HRF has a slow decay
}

#[test]
fn test_sampling_frame_basic() {
    // Create a simple sampling frame
    let sf = SamplingFrame::uniform(100, 2.0, 0.1).unwrap();

    assert_eq!(sf.n_scans(), 100);
    assert_eq!(sf.n_blocks(), 1);
    assert_eq!(sf.tr[0], 2.0);
    assert_eq!(sf.start_time[0], 1.0); // TR/2

    // Check sample times
    let samples = sf.samples();
    assert_eq!(samples.len(), 100);
    assert_eq!(samples[0], 1.0); // First scan at TR/2
    assert_eq!(samples[1], 3.0); // Second scan at TR/2 + TR
}

#[test]
fn test_hrf_with_block_duration() {
    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    // Evaluate with 3-second block duration
    let t = Array1::linspace(0.0, 30.0, 31);
    let block_response = hrf.evaluate_block(&t, 3.0, 0.1, true, false).unwrap();

    assert_eq!(block_response.shape(), &[31, 1]);

    // Block response should be extended compared to impulse response
    let impulse_response = hrf.evaluate(&t).unwrap();

    // Find time to peak for both
    let (block_peak_idx, _) = block_response
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let (impulse_peak_idx, _) = impulse_response
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    // Block response should peak later
    assert!(block_peak_idx > impulse_peak_idx);
}

#[test]
fn test_multi_block_sampling_frame() {
    // Create multi-block sampling frame
    let sf = SamplingFrame::new(
        vec![50, 50, 50],
        vec![2.0, 2.5, 2.0],
        Some(vec![1.0, 1.25, 1.0]),
        0.1,
    )
    .unwrap();

    assert_eq!(sf.n_blocks(), 3);
    assert_eq!(sf.n_scans(), 150);

    let blockids = sf.blockids();
    assert_eq!(blockids[0], 0);
    assert_eq!(blockids[49], 0);
    assert_eq!(blockids[50], 1);
    assert_eq!(blockids[99], 1);
    assert_eq!(blockids[100], 2);
    assert_eq!(blockids[149], 2);
}
