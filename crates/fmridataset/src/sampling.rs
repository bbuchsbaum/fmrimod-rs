use crate::errors::{FmriDatasetError, Result};

#[derive(Debug, Clone)]
pub struct SamplingFrame {
    block_lengths: Vec<usize>,
    tr: Vec<f64>,
    start_time: Vec<f64>,
    precision: f64,
}

impl SamplingFrame {
    pub fn uniform(n_scans: usize, tr: f64, precision: f64) -> Result<Self> {
        SamplingFrame::builder()
            .run_lengths(vec![n_scans])
            .tr(vec![tr])
            .precision(precision)
            .build()
    }
    pub fn builder() -> SamplingFrameBuilder {
        SamplingFrameBuilder::default()
    }

    pub fn n_runs(&self) -> usize {
        self.block_lengths.len()
    }

    pub fn total_timepoints(&self) -> usize {
        self.block_lengths.iter().sum()
    }

    pub fn tr_for_run(&self, run: usize) -> Result<f64> {
        self.tr
            .get(run)
            .copied()
            .ok_or_else(|| FmriDatasetError::InvalidIndex(format!("unknown run {run}")))
    }

    pub fn block_lengths(&self) -> &[usize] {
        &self.block_lengths
    }

    pub fn samples(&self) -> Vec<f64> {
        let mut times = Vec::with_capacity(self.total_timepoints());
        let mut cumulative = 0.0;
        for (idx, &len) in self.block_lengths.iter().enumerate() {
            let start = self.start_time[idx];
            let tr = self.tr[idx];
            for scan in 0..len {
                times.push(cumulative + start + scan as f64 * tr);
            }
            cumulative += len as f64 * tr;
        }
        times
    }

    pub fn block_ids(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.total_timepoints());
        for (idx, &len) in self.block_lengths.iter().enumerate() {
            ids.extend(std::iter::repeat_n(idx, len));
        }
        ids
    }

    pub fn precision(&self) -> f64 {
        self.precision
    }
}

#[derive(Default)]
pub struct SamplingFrameBuilder {
    block_lengths: Option<Vec<usize>>,
    tr: Option<Vec<f64>>,
    start_time: Option<Vec<f64>>,
    precision: Option<f64>,
}

impl SamplingFrameBuilder {
    pub fn run_lengths(mut self, runs: Vec<usize>) -> Self {
        self.block_lengths = Some(runs);
        self
    }

    pub fn tr(mut self, tr: Vec<f64>) -> Self {
        self.tr = Some(tr);
        self
    }

    pub fn start_time(mut self, start: Vec<f64>) -> Self {
        self.start_time = Some(start);
        self
    }

    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = Some(precision);
        self
    }

    pub fn build(self) -> Result<SamplingFrame> {
        let block_lengths = self
            .block_lengths
            .ok_or_else(|| FmriDatasetError::InvalidIndex("run lengths missing".into()))?;
        if block_lengths.is_empty() || block_lengths.contains(&0) {
            return Err(FmriDatasetError::InvalidIndex(
                "run lengths must be positive".into(),
            ));
        }
        block_lengths.iter().try_fold(0usize, |sum, &n| {
            sum.checked_add(n).ok_or(FmriDatasetError::SizeOverflow)
        })?;
        let mut tr = self
            .tr
            .ok_or_else(|| FmriDatasetError::InvalidIndex("TR missing".into()))?;
        if tr.len() == 1 {
            tr = vec![tr[0]; block_lengths.len()];
        }
        if tr.len() != block_lengths.len() {
            return Err(FmriDatasetError::InvalidIndex(
                "TR length must match run count".into(),
            ));
        }
        if tr.iter().any(|t| !t.is_finite() || *t <= 0.0) {
            return Err(crate::errors::invalid("TR", "must be finite and positive"));
        }
        let precision = self.precision.unwrap_or(0.1);
        if !precision.is_finite() || precision <= 0.0 {
            return Err(FmriDatasetError::InvalidIndex(
                "precision must be positive".into(),
            ));
        }
        let mut start_time = self
            .start_time
            .unwrap_or_else(|| tr.iter().map(|t| t / 2.0).collect());
        if start_time.len() == 1 && block_lengths.len() > 1 {
            start_time = vec![start_time[0]; block_lengths.len()];
        }
        if start_time.len() != block_lengths.len() {
            return Err(FmriDatasetError::InvalidIndex(
                "start_time length must match runs".into(),
            ));
        }
        if start_time.iter().any(|t| !t.is_finite()) {
            return Err(crate::errors::invalid("start time", "must be finite"));
        }
        let mut elapsed = 0.0;
        for ((&length, &interval), &start) in block_lengths.iter().zip(&tr).zip(&start_time) {
            if !(elapsed + start + (length - 1) as f64 * interval).is_finite() {
                return Err(crate::errors::invalid("sampling times", "overflow"));
            }
            elapsed += length as f64 * interval;
            if !elapsed.is_finite() {
                return Err(crate::errors::invalid("sampling duration", "overflow"));
            }
        }
        Ok(SamplingFrame {
            block_lengths,
            tr,
            start_time,
            precision,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_frame_samples() {
        let frame = SamplingFrame::builder()
            .run_lengths(vec![2, 2])
            .tr(vec![2.0])
            .precision(0.1)
            .build()
            .unwrap();
        assert_eq!(frame.total_timepoints(), 4);
        let samples = frame.samples();
        assert_eq!(samples.len(), 4);
        assert!(samples[1] > samples[0]);
    }
}
