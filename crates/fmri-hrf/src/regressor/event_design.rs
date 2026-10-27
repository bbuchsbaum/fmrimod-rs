//! Run-local event tables and condition designs.

use super::{convolve_hrf_with_method, ConvolutionMethod, DesignMatrix};
use crate::{Hrf, HrfError, Result, SamplingFrame};
use ndarray::{Array1, Array2, ArrayView2};

/// Borrowed event amplitudes, one row per event and one column per predictor.
///
/// Onsets are in seconds from the start of their run. Run indices are zero
/// based. Durations may contain one value to broadcast or one per event.
/// Evaluation preserves empty runs and never carries an HRF tail across runs.
#[derive(Debug, Clone, Copy)]
pub struct EventTable<'a> {
    /// Event onsets relative to the start of their run, in seconds.
    pub onsets: &'a [f64],
    /// Nonnegative durations, scalar or one per event.
    pub durations: &'a [f64],
    /// Signed predictor amplitudes with one row per event.
    pub amplitudes: ArrayView2<'a, f64>,
    /// Zero-based run index for each event.
    pub run_ids: &'a [usize],
}

impl EventTable<'_> {
    /// Evaluate all predictors in basis-major order: all predictors for basis
    /// one, then all predictors for basis two, matching R's column ordering.
    pub fn evaluate(
        &self,
        frame: &SamplingFrame,
        hrf: &dyn Hrf,
        method: ConvolutionMethod,
    ) -> Result<Array2<f64>> {
        let n = self.onsets.len();
        // Validate public SamplingFrame fields too, since callers can mutate them.
        let frame = SamplingFrame::new(
            frame.blocklens.clone(),
            frame.tr.clone(),
            Some(frame.start_time.clone()),
            frame.precision,
        )?;
        if self.run_ids.len() != n
            || self.amplitudes.nrows() != n
            || !(self.durations.len() == 1 || self.durations.len() == n)
            || self.run_ids.iter().any(|&run| run >= frame.n_blocks())
        {
            return Err(HrfError::InvalidInput(
                "event lengths must agree and run indices must belong to the sampling frame".into(),
            ));
        }
        if self.onsets.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.durations.iter().any(|v| !v.is_finite() || *v < 0.0)
            || self.amplitudes.iter().any(|v| !v.is_finite())
        {
            return Err(HrfError::InvalidInput(
                "events must be finite with nonnegative onsets and durations".into(),
            ));
        }
        let predictors = self.amplitudes.ncols();
        let columns = predictors
            .checked_mul(hrf.nbasis())
            .ok_or_else(|| HrfError::InvalidInput("too many design columns".into()))?;
        let mut output = Array2::zeros((frame.n_scans(), columns));
        let mut row_start = 0;
        for (run, &scans) in frame.blocklens.iter().enumerate() {
            let indices: Vec<_> = self
                .run_ids
                .iter()
                .enumerate()
                .filter_map(|(i, &r)| (r == run).then_some(i))
                .collect();
            if !indices.is_empty() {
                let grid = Array1::from_shape_fn(scans, |i| {
                    frame.start_time[run] + i as f64 * frame.tr[run]
                });
                let onsets: Vec<_> = indices.iter().map(|&i| self.onsets[i]).collect();
                let durations: Vec<_> = indices
                    .iter()
                    .map(|&i| self.durations[if self.durations.len() == 1 { 0 } else { i }])
                    .collect();
                for c in 0..predictors {
                    let amplitudes: Vec<_> =
                        indices.iter().map(|&i| self.amplitudes[[i, c]]).collect();
                    let values = convolve_hrf_with_method(
                        &grid,
                        &onsets,
                        &durations,
                        &amplitudes,
                        hrf,
                        hrf.span(),
                        frame.precision,
                        method,
                        true,
                    )?;
                    for b in 0..hrf.nbasis() {
                        output
                            .slice_mut(ndarray::s![
                                row_start..row_start + scans,
                                b * predictors + c
                            ])
                            .assign(&values.column(b));
                    }
                }
            }
            row_start += scans;
        }
        Ok(output)
    }
}

/// Condition labels and timing for a run-aware event design.
#[derive(Debug, Clone, Copy)]
pub struct ConditionEvents<'a> {
    onsets: &'a [f64],
    conditions: &'a [String],
    run_ids: &'a [usize],
    durations: &'a [f64],
    amplitudes: &'a [f64],
}

impl<'a> ConditionEvents<'a> {
    /// Create unit-amplitude impulses. Run indices are zero based.
    pub fn new(onsets: &'a [f64], conditions: &'a [String], run_ids: &'a [usize]) -> Self {
        Self {
            onsets,
            conditions,
            run_ids,
            durations: &[0.0],
            amplitudes: &[1.0],
        }
    }
    /// Set scalar or per-event durations, in seconds.
    pub fn with_durations(mut self, durations: &'a [f64]) -> Self {
        self.durations = durations;
        self
    }
    /// Set scalar or per-event signed amplitudes.
    pub fn with_amplitudes(mut self, amplitudes: &'a [f64]) -> Self {
        self.amplitudes = amplitudes;
        self
    }
    /// Build a design with lexically sorted conditions and basis-major columns.
    pub fn evaluate(&self, frame: &SamplingFrame, hrf: &dyn Hrf) -> Result<DesignMatrix> {
        let n = self.onsets.len();
        if self.conditions.len() != n || !(self.amplitudes.len() == 1 || self.amplitudes.len() == n)
        {
            return Err(HrfError::InvalidInput(
                "conditions and amplitudes must match events".into(),
            ));
        }
        let mut levels = self.conditions.to_vec();
        levels.sort();
        levels.dedup();
        let mut amplitudes = Array2::zeros((n, levels.len()));
        for (i, condition) in self.conditions.iter().enumerate() {
            let c = levels
                .binary_search(condition)
                .map_err(|_| HrfError::InvalidInput("missing condition".into()))?;
            amplitudes[[i, c]] = self.amplitudes[if self.amplitudes.len() == 1 { 0 } else { i }];
        }
        let data = EventTable {
            onsets: self.onsets,
            durations: self.durations,
            amplitudes: amplitudes.view(),
            run_ids: self.run_ids,
        }
        .evaluate(frame, hrf, ConvolutionMethod::Auto)?;
        let names = (0..hrf.nbasis())
            .flat_map(|b| {
                levels.iter().map(move |level| {
                    if hrf.nbasis() == 1 {
                        level.clone()
                    } else {
                        format!("{level}_b{:02}", b + 1)
                    }
                })
            })
            .collect();
        DesignMatrix::with_names(data, Array1::from(frame.samples()), names)
    }
}
