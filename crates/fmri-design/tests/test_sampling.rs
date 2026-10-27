//! Tests for sampling module

use fmri_design::sampling::{multi_block_sampling_frame, uniform_sampling_frame};

#[test]
fn test_uniform_sampling_frame() {
    let frame = uniform_sampling_frame(100, 2.0, None).unwrap();

    assert_eq!(frame.n_scans(), 100);
    assert_eq!(frame.n_blocks(), 1);

    // Check timing
    let times = frame.samples();
    assert!(times.len() > 0);
}

#[test]
fn test_uniform_sampling_frame_with_start() {
    let frame = uniform_sampling_frame(50, 2.0, Some(0.0)).unwrap();

    assert_eq!(frame.n_scans(), 50);
    assert_eq!(frame.n_blocks(), 1);
}

#[test]
fn test_multi_block_sampling_frame() {
    let block_lengths = vec![50, 75, 100];
    let trs = vec![2.0]; // Single TR for all blocks

    let frame = multi_block_sampling_frame(&block_lengths, &trs, None).unwrap();

    assert_eq!(frame.n_blocks(), 3);
    assert_eq!(frame.n_scans(), 225); // 50 + 75 + 100
}

#[test]
fn test_multi_block_different_trs() {
    let block_lengths = vec![30, 40];
    let trs = vec![2.0, 3.0]; // Different TR per block

    let frame = multi_block_sampling_frame(&block_lengths, &trs, Some(0.0)).unwrap();

    assert_eq!(frame.n_blocks(), 2);
    assert_eq!(frame.n_scans(), 70); // 30 + 40
}

#[test]
fn test_multi_block_invalid_trs() {
    let block_lengths = vec![30, 40, 50];
    let trs = vec![2.0, 3.0]; // Too few TRs

    let result = multi_block_sampling_frame(&block_lengths, &trs, None);
    assert!(result.is_err());
}

#[test]
fn test_sampling_frame_timevec() {
    let frame = uniform_sampling_frame(10, 1.0, Some(0.0)).unwrap();
    let times = frame.samples();

    // Should have times for 10 scans
    assert!(times.len() >= 10);
}

#[test]
fn test_sampling_frame_block_info() {
    let block_lengths = vec![20, 30];
    let trs = vec![2.0];

    let frame = multi_block_sampling_frame(&block_lengths, &trs, None).unwrap();

    // Test block boundaries
    let block_ids = frame.blockids();
    assert_eq!(block_ids.len(), 50); // Total scans

    // First 20 should be block 0
    for i in 0..20 {
        assert_eq!(block_ids[i], 0);
    }

    // Next 30 should be block 1
    for i in 20..50 {
        assert_eq!(block_ids[i], 1);
    }
}
