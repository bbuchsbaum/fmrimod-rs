//! Sampling frame for fMRI acquisition timing specification.

use crate::error::{HrfError, Result};
use std::fmt;

/// fMRI acquisition timing specification.
///
/// This struct represents the temporal sampling structure of an fMRI experiment,
/// including the timing of acquisitions, block structure, and precision settings.
///
/// # Terminology
/// - **Samples**: The time points at which fMRI volumes are acquired (regularly spaced by TR)
/// - **TR (Repetition Time)**: The time between successive volume acquisitions
/// - **Block**: A continuous run of scans (e.g., one scanning session)
/// - **Start time**: Offset from t=0 for the first scan in a block (typically TR/2)
///
/// Note: This is distinct from stimulus/event onsets, which are the times when
/// experimental conditions occur and are handled separately in design matrix construction.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplingFrame {
    /// Number of scans per block.
    pub blocklens: Vec<usize>,
    /// Repetition time(s) in seconds.
    pub tr: Vec<f64>,
    /// Start time offset(s) in seconds.
    pub start_time: Vec<f64>,
    /// Temporal precision for convolution in seconds.
    pub precision: f64,
}

impl SamplingFrame {
    /// Create a new SamplingFrame for fMRI acquisition timing.
    ///
    /// A SamplingFrame defines when fMRI volumes are acquired during an experiment.
    /// It supports multi-run experiments where each run (block) can have different
    /// parameters. This is essential for proper alignment between stimulus timing
    /// and fMRI acquisition.
    ///
    /// # Arguments
    /// * `blocklens` - Number of scans per block/run (single value or vector)
    /// * `tr` - Repetition time(s) in seconds (single value or vector)
    /// * `start_time` - Start time offset(s) in seconds (defaults to TR/2 for slice timing)
    /// * `precision` - Temporal precision for convolution in seconds (typically 0.1)
    ///
    /// # Examples
    ///
    /// ## Single Run Experiment
    /// ```
    /// use fmri_hrf::sampling::SamplingFrame;
    ///
    /// // 200 volumes, TR=2s, default start time (1s)
    /// let frame = SamplingFrame::new(
    ///     vec![200],
    ///     vec![2.0],
    ///     None::<Vec<f64>>,
    ///     0.1
    /// ).unwrap();
    ///
    /// // Get acquisition times
    /// let scan_times = frame.samples();
    /// assert_eq!(scan_times.len(), 200);
    /// assert_eq!(scan_times[0], 1.0);  // First scan at TR/2
    /// assert_eq!(scan_times[1], 3.0);  // Second scan at TR/2 + TR
    /// ```
    ///
    /// ## Multi-Run Experiment
    /// ```
    /// use fmri_hrf::sampling::SamplingFrame;
    ///
    /// // Three runs: 100, 150, and 100 scans
    /// // Different TRs: 2s, 1.5s, 2s
    /// let frame = SamplingFrame::new(
    ///     vec![100, 150, 100],
    ///     vec![2.0, 1.5, 2.0],
    ///     None::<Vec<f64>>,
    ///     0.1
    /// ).unwrap();
    ///
    /// assert_eq!(frame.n_blocks(), 3);
    /// assert_eq!(frame.n_scans(), 350);
    /// ```
    ///
    /// ## Custom Start Times
    /// ```
    /// use fmri_hrf::sampling::SamplingFrame;
    ///
    /// // Experiment starting at t=0 (no slice timing correction)
    /// let frame = SamplingFrame::new(
    ///     vec![100],
    ///     vec![2.0],
    ///     Some(vec![0.0]),  // Start at t=0
    ///     0.1
    /// ).unwrap();
    ///
    /// let times = frame.samples();
    /// assert_eq!(times[0], 0.0);  // First scan at t=0
    /// ```
    ///
    /// # Errors
    /// Returns error if:
    /// - blocklens contains zero or negative values
    /// - TR contains zero or negative values
    /// - precision is zero or negative
    /// - Length mismatches between vectors (unless single value provided)
    pub fn new(
        blocklens: impl Into<Vec<usize>>,
        tr: impl Into<Vec<f64>>,
        start_time: Option<Vec<f64>>,
        precision: f64,
    ) -> Result<Self> {
        let blocklens = blocklens.into();
        let mut tr = tr.into();

        // Validate inputs
        if blocklens.is_empty() {
            return Err(HrfError::InvalidInput("blocklens cannot be empty".into()));
        }
        if blocklens.contains(&0) {
            return Err(HrfError::InvalidInput("blocklens must be positive".into()));
        }
        if tr.is_empty() {
            return Err(HrfError::InvalidInput("TR cannot be empty".into()));
        }
        if tr.iter().any(|&t| !t.is_finite() || t <= 0.0) {
            return Err(HrfError::InvalidInput("TR must be positive".into()));
        }
        if !precision.is_finite() || precision <= 0.0 {
            return Err(HrfError::InvalidInput("precision must be positive".into()));
        }

        let n_blocks = blocklens.len();

        // Expand TR if single value provided for multiple blocks
        if tr.len() == 1 && n_blocks > 1 {
            tr = vec![tr[0]; n_blocks];
        } else if tr.len() != n_blocks {
            return Err(HrfError::InvalidInput(format!(
                "Length of TR ({}) must be 1 or match number of blocks ({})",
                tr.len(),
                n_blocks
            )));
        }

        // Handle start_time (default to TR/2 for R compatibility)
        let start_time = match start_time {
            Some(st) => {
                if st.len() == 1 && n_blocks > 1 {
                    vec![st[0]; n_blocks]
                } else if st.len() != n_blocks {
                    return Err(HrfError::InvalidInput(format!(
                        "Length of start_time ({}) must be 1 or match number of blocks ({})",
                        st.len(),
                        n_blocks
                    )));
                } else {
                    st
                }
            }
            None => tr.iter().map(|&t| t / 2.0).collect(),
        };

        if start_time.iter().any(|&t| !t.is_finite() || t < 0.0) {
            return Err(HrfError::InvalidInput(
                "start times must be nonnegative and finite".into(),
            ));
        }
        if blocklens
            .iter()
            .try_fold(0_usize, |a, &b| a.checked_add(b))
            .is_none()
            || blocklens
                .iter()
                .zip(&tr)
                .any(|(&n, &dt)| !(n as f64 * dt).is_finite())
        {
            return Err(HrfError::InvalidInput("sampling frame is too large".into()));
        }
        Ok(SamplingFrame {
            blocklens,
            tr,
            start_time,
            precision,
        })
    }

    /// Create a simple SamplingFrame with uniform parameters.
    pub fn uniform(n_scans: usize, tr: f64, precision: f64) -> Result<Self> {
        Self::new(vec![n_scans], vec![tr], None::<Vec<f64>>, precision)
    }

    /// Number of blocks.
    pub fn n_blocks(&self) -> usize {
        self.blocklens.len()
    }

    /// Total number of scans across all blocks.
    pub fn n_scans(&self) -> usize {
        self.blocklens.iter().sum()
    }

    /// Get block ID for each scan.
    pub fn blockids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.n_scans());
        for (block_id, &block_len) in self.blocklens.iter().enumerate() {
            ids.extend(vec![block_id; block_len]);
        }
        ids
    }

    /// Get sample time points for each scan (acquisition times).
    ///
    /// Returns the time point at which each fMRI volume was acquired,
    /// taking into account TR and any start time offsets.
    pub fn samples(&self) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.n_scans());
        let mut cumulative_time = 0.0;

        for (block_idx, &block_len) in self.blocklens.iter().enumerate() {
            let block_start = self.start_time[block_idx];
            let block_tr = self.tr[block_idx];

            for scan_idx in 0..block_len {
                times.push(cumulative_time + block_start + scan_idx as f64 * block_tr);
            }

            cumulative_time += block_len as f64 * block_tr;
        }

        times
    }

    /// Get onset time for each scan.
    ///
    /// Deprecated: Use `samples()` instead. This method name is confusing
    /// as "onsets" typically refers to stimulus/event onsets in fMRI.
    #[deprecated(since = "0.2.0", note = "Use samples() instead")]
    pub fn onsets(&self) -> Vec<f64> {
        self.samples()
    }

    /// Get offset time for each scan (sample time + TR).
    pub fn offsets(&self) -> Vec<f64> {
        let samples = self.samples();
        let blockids = self.blockids();

        samples
            .iter()
            .zip(blockids.iter())
            .map(|(&onset, &block_id)| onset + self.tr[block_id])
            .collect()
    }

    /// Get real time for a fractional scan position.
    pub fn real_time(&self, frac_scan: f64) -> f64 {
        if frac_scan <= 0.0 {
            return 0.0;
        }

        let scan_idx = frac_scan.floor() as usize;
        let frac_part = frac_scan.fract();

        if scan_idx >= self.n_scans() {
            // Beyond last scan
            let last_offset = self.offsets().last().copied().unwrap_or(0.0);
            let last_block = self.blocklens.len() - 1;
            let extra_scans = frac_scan - self.n_scans() as f64;
            return last_offset + extra_scans * self.tr[last_block];
        }

        let onsets = self.samples();
        let blockids = self.blockids();
        let block_id = blockids[scan_idx];

        onsets[scan_idx] + frac_part * self.tr[block_id]
    }

    /// Get global time points for all scans.
    pub fn global_time(&self) -> Vec<f64> {
        self.samples()
    }

    /// Get the total duration of the experiment.
    pub fn total_time(&self) -> f64 {
        self.offsets().last().copied().unwrap_or(0.0)
    }

    /// Convert to dictionary representation (as a HashMap).
    pub fn to_dict(&self) -> std::collections::HashMap<String, serde_json::Value> {
        use serde_json::json;
        let mut dict = std::collections::HashMap::new();

        dict.insert("blocklens".to_string(), json!(self.blocklens));
        dict.insert("tr".to_string(), json!(self.tr));
        dict.insert("start_time".to_string(), json!(self.start_time));
        dict.insert("precision".to_string(), json!(self.precision));
        dict.insert("n_blocks".to_string(), json!(self.n_blocks()));
        dict.insert("n_scans".to_string(), json!(self.n_scans()));

        dict
    }

    /// Create SamplingFrame from dictionary representation.
    pub fn from_dict(data: &std::collections::HashMap<String, serde_json::Value>) -> Result<Self> {
        // Extract blocklens
        let blocklens = data
            .get("blocklens")
            .and_then(|v| v.as_array())
            .ok_or_else(|| HrfError::InvalidInput("Missing or invalid 'blocklens' in dict".into()))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .ok_or_else(|| HrfError::InvalidInput("Invalid blocklens value".into()))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>();

        // Extract TR
        let tr = data
            .get("tr")
            .and_then(|v| v.as_array())
            .ok_or_else(|| HrfError::InvalidInput("Missing or invalid 'tr' in dict".into()))?
            .iter()
            .map(|v| {
                v.as_f64()
                    .ok_or_else(|| HrfError::InvalidInput("Invalid tr value".into()))
            })
            .collect::<Result<Vec<_>>>()?;

        // Extract start_time (optional)
        let start_time = data
            .get("start_time")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .map(|v| {
                        v.as_f64().ok_or_else(|| {
                            HrfError::InvalidInput("Invalid start_time value".into())
                        })
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;

        // Extract precision (optional, default to 0.1)
        let precision = data
            .get("precision")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1);

        Self::new(blocklens, tr, start_time, precision)
    }

    /// Concatenate two sampling frames.
    ///
    /// The second frame's start times are adjusted to begin after the first frame ends.
    ///
    /// # Arguments
    /// * `other` - Another SamplingFrame to concatenate
    ///
    /// # Returns
    /// New SamplingFrame with combined blocks
    ///
    /// # Errors
    /// Returns error if the frames have different precision values
    pub fn concatenate(&self, other: &SamplingFrame) -> Result<Self> {
        if (self.precision - other.precision).abs() > f64::EPSILON {
            return Err(HrfError::InvalidInput(format!(
                "Cannot concatenate SamplingFrames with different precision ({} vs {})",
                self.precision, other.precision
            )));
        }

        // Calculate the end time of the last scan in this frame
        let last_end_time = self.offsets().last().copied().unwrap_or(0.0);

        // Adjust start times for the second frame
        let adjusted_start_times: Vec<f64> = other
            .start_time
            .iter()
            .map(|&t| t + last_end_time)
            .collect();

        // Combine all vectors
        let mut combined_blocklens = self.blocklens.clone();
        combined_blocklens.extend(&other.blocklens);

        let mut combined_tr = self.tr.clone();
        combined_tr.extend(&other.tr);

        let mut combined_start_time = self.start_time.clone();
        combined_start_time.extend(adjusted_start_times);

        Self::new(
            combined_blocklens,
            combined_tr,
            Some(combined_start_time),
            self.precision,
        )
    }
}

// Builder pattern for SamplingFrame
impl SamplingFrame {
    /// Create a builder for SamplingFrame.
    pub fn builder() -> SamplingFrameBuilder {
        SamplingFrameBuilder::default()
    }
}

/// Builder for SamplingFrame.
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SamplingFrameBuilder {
    blocklens: Option<Vec<usize>>,
    tr: Option<Vec<f64>>,
    start_time: Option<Vec<f64>>,
    precision: Option<f64>,
}

impl SamplingFrameBuilder {
    /// Set block lengths.
    pub fn blocklens(mut self, blocklens: impl Into<Vec<usize>>) -> Self {
        self.blocklens = Some(blocklens.into());
        self
    }

    /// Set repetition time(s).
    pub fn tr(mut self, tr: impl Into<Vec<f64>>) -> Self {
        self.tr = Some(tr.into());
        self
    }

    /// Set start time offset(s).
    pub fn start_time(mut self, start_time: impl Into<Vec<f64>>) -> Self {
        self.start_time = Some(start_time.into());
        self
    }

    /// Set temporal precision.
    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Build the SamplingFrame.
    pub fn build(self) -> Result<SamplingFrame> {
        let blocklens = self
            .blocklens
            .ok_or_else(|| HrfError::InvalidInput("blocklens must be specified".into()))?;
        let tr = self
            .tr
            .ok_or_else(|| HrfError::InvalidInput("TR must be specified".into()))?;
        let precision = self.precision.unwrap_or(0.1);

        SamplingFrame::new(blocklens, tr, self.start_time, precision)
    }
}

impl fmt::Display for SamplingFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "SamplingFrame with {} block(s) and {} total scans:",
            self.n_blocks(),
            self.n_scans()
        )?;

        for i in 0..self.blocklens.len() {
            writeln!(
                f,
                "  Block {}: {} scans, TR={:.3}s, start={:.3}s",
                i, self.blocklens[i], self.tr[i], self.start_time[i]
            )?;
        }

        write!(f, "  Precision: {:.3}s", self.precision)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_uniform_sampling_frame() {
        let sf = SamplingFrame::uniform(100, 2.0, 0.1).unwrap();
        assert_eq!(sf.n_blocks(), 1);
        assert_eq!(sf.n_scans(), 100);
        assert_eq!(sf.tr[0], 2.0);
        assert_eq!(sf.start_time[0], 1.0); // TR/2
    }

    #[test]
    fn test_multi_block_sampling_frame() {
        let sf =
            SamplingFrame::new(vec![50, 50], vec![2.0, 2.0], Some(vec![1.0, 1.0]), 0.1).unwrap();

        assert_eq!(sf.n_blocks(), 2);
        assert_eq!(sf.n_scans(), 100);

        let blockids = sf.blockids();
        assert_eq!(blockids[0], 0);
        assert_eq!(blockids[49], 0);
        assert_eq!(blockids[50], 1);
        assert_eq!(blockids[99], 1);
    }

    #[test]
    fn test_samples_and_offsets() {
        let sf = SamplingFrame::uniform(5, 2.0, 0.1).unwrap();
        let samples = sf.samples();
        let offsets = sf.offsets();

        // First scan starts at TR/2 = 1.0
        assert_relative_eq!(samples[0], 1.0);
        assert_relative_eq!(offsets[0], 3.0); // sample + TR

        // Check spacing
        for i in 1..5 {
            assert_relative_eq!(samples[i] - samples[i - 1], 2.0);
        }
    }

    #[test]
    fn test_builder() {
        let sf = SamplingFrame::builder()
            .blocklens(vec![100])
            .tr(vec![2.0])
            .precision(0.05)
            .build()
            .unwrap();

        assert_eq!(sf.n_scans(), 100);
        assert_eq!(sf.precision, 0.05);
    }

    #[test]
    fn test_to_dict_from_dict() {
        let sf1 =
            SamplingFrame::new(vec![50, 50], vec![2.0, 2.5], Some(vec![1.0, 1.5]), 0.1).unwrap();

        // Convert to dict
        let dict = sf1.to_dict();

        // Verify dict contents
        assert_eq!(dict.get("n_blocks").and_then(|v| v.as_u64()), Some(2));
        assert_eq!(dict.get("n_scans").and_then(|v| v.as_u64()), Some(100));
        assert_eq!(dict.get("precision").and_then(|v| v.as_f64()), Some(0.1));

        // Convert back from dict
        let sf2 = SamplingFrame::from_dict(&dict).unwrap();

        // Verify round-trip
        assert_eq!(sf2.blocklens, sf1.blocklens);
        assert_eq!(sf2.tr, sf1.tr);
        assert_eq!(sf2.start_time, sf1.start_time);
        assert_eq!(sf2.precision, sf1.precision);
    }

    #[test]
    fn test_concatenate() {
        let sf1 = SamplingFrame::uniform(50, 2.0, 0.1).unwrap();
        let sf2 = SamplingFrame::uniform(30, 2.0, 0.1).unwrap();

        let combined = sf1.concatenate(&sf2).unwrap();

        assert_eq!(combined.n_blocks(), 2);
        assert_eq!(combined.n_scans(), 80);
        assert_eq!(combined.blocklens, vec![50, 30]);

        // Check that second frame's start time is adjusted
        let samples = combined.samples();
        // First frame ends at time 100.0 (last offset)
        // Second frame should start after that
        assert!(samples[50] > 100.0);
    }

    #[test]
    fn test_concatenate_different_precision() {
        let sf1 = SamplingFrame::uniform(50, 2.0, 0.1).unwrap();
        let sf2 = SamplingFrame::uniform(30, 2.0, 0.2).unwrap();

        // Should fail with different precision
        assert!(sf1.concatenate(&sf2).is_err());
    }
}
