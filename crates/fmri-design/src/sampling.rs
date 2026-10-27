//! Sampling frame re-exports from fmri_hrf.
//!
//! This module re-exports SamplingFrame from fmri_hrf to maintain consistency
//! with the R fmri_design package.

// Re-export key types from fmri_hrf
pub use fmri_hrf::{SamplingFrame, SamplingFrameBuilder};

// Extension trait to add helpful methods to SamplingFrame
pub trait SamplingFrameExt {
    fn from_duration(duration: f64, tr: f64) -> Self;
}

impl SamplingFrameExt for SamplingFrame {
    /// Create a sampling frame from duration and TR.
    ///
    /// # Arguments
    /// * `duration` - Total duration in seconds
    /// * `tr` - Repetition time in seconds
    fn from_duration(duration: f64, tr: f64) -> Self {
        let n_scans = (duration / tr).ceil() as usize;
        uniform_sampling_frame(n_scans, tr, None).unwrap()
    }
}

/// Create a uniform sampling frame with simplified parameters.
///
/// This is a convenience function that creates a sampling frame with
/// uniform TR and a single block.
///
/// # Arguments
/// * `n_scans` - Total number of scans
/// * `tr` - Repetition time in seconds
/// * `start_time` - Start time offset in seconds (default: 0.0)
///
/// # Returns
/// Result containing the configured SamplingFrame
///
/// # Example
/// ```ignore
/// use fmri_design::sampling::uniform_sampling_frame;
///
/// let sf = uniform_sampling_frame(100, 2.0, Some(0.0))?;
/// ```
pub fn uniform_sampling_frame(
    n_scans: usize,
    tr: f64,
    start_time: Option<f64>,
) -> Result<SamplingFrame, fmri_hrf::HrfError> {
    let start = start_time.unwrap_or(tr / 2.0);

    SamplingFrame::builder()
        .blocklens(vec![n_scans])
        .tr(vec![tr])
        .start_time(vec![start])
        .precision(0.1)
        .build()
}

/// Create a multi-block sampling frame.
///
/// This creates a sampling frame with multiple blocks, each potentially
/// having different lengths and TRs.
///
/// # Arguments
/// * `block_lengths` - Number of scans in each block
/// * `trs` - TR for each block (single value or one per block)
/// * `start_time` - Start time offset in seconds
///
/// # Returns
/// Result containing the configured SamplingFrame
pub fn multi_block_sampling_frame(
    block_lengths: &[usize],
    trs: &[f64],
    start_time: Option<f64>,
) -> Result<SamplingFrame, fmri_hrf::HrfError> {
    // Handle single TR vs per-block TRs
    let trs_vec: Vec<f64> = if trs.len() == 1 {
        vec![trs[0]; block_lengths.len()]
    } else if trs.len() == block_lengths.len() {
        trs.to_vec()
    } else {
        return Err(fmri_hrf::HrfError::InvalidInput(
            "Number of TRs must be 1 or match number of blocks".to_string(),
        ));
    };

    // Create start times for each block
    let start_times: Vec<f64> = if let Some(start) = start_time {
        vec![start; block_lengths.len()]
    } else {
        // Default to TR/2 for each block
        trs_vec.iter().map(|tr| tr / 2.0).collect()
    };

    SamplingFrame::builder()
        .blocklens(block_lengths.to_vec())
        .tr(trs_vec)
        .start_time(start_times)
        .precision(0.1)
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_sampling_frame() {
        let sf = uniform_sampling_frame(100, 2.0, None).unwrap();
        assert_eq!(sf.n_scans(), 100);
        assert_eq!(sf.n_blocks(), 1);
    }

    #[test]
    fn test_multi_block_sampling_frame() {
        let block_lengths = vec![50, 75, 100];
        let trs = vec![2.0]; // Single TR for all blocks

        let sf = multi_block_sampling_frame(&block_lengths, &trs, Some(1.0)).unwrap();
        assert_eq!(sf.n_scans(), 225);
        assert_eq!(sf.n_blocks(), 3);
    }

    #[test]
    fn test_multi_block_different_trs() {
        let block_lengths = vec![50, 75];
        let trs = vec![2.0, 1.5];

        let sf = multi_block_sampling_frame(&block_lengths, &trs, None).unwrap();
        assert_eq!(sf.n_scans(), 125);
        assert_eq!(sf.n_blocks(), 2);
    }
}
