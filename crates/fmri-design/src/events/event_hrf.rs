//! Event wrapper that applies an HRF to another event.
//!
//! This type is used by the formula/event_model pipeline to represent
//! `hrf(variable)` terms. It delegates all metadata (onsets, durations,
//! levels, etc.) to the inner event, but its `design_matrix` is the
//! HRF-convolved version of the inner event's design matrix.

use std::fmt::Debug;

use ndarray::Array1;

use crate::convolve::convolve_at_times;
use crate::hrf::{create_hrf, HrfSpec as HrfEnumSpec};
use crate::types::{
    Array, DataFrame, DurationArray, EventSpec, EventType, OnsetArray, TimeArray, ValidationError,
};
use fmri_hrf::{
    regressor::{ConvolutionMethod, EventTable},
    SamplingFrame,
};

/// Event specification wrapped with an HRF.
#[derive(Debug)]
pub struct HrfEvent {
    /// Underlying (unconvolved) event specification.
    inner: Box<dyn EventSpec>,
    /// HRF specification to apply.
    hrf_spec: HrfEnumSpec,
    precision: f64,
    runs: Option<(SamplingFrame, Vec<usize>)>,
}

impl HrfEvent {
    /// Create a new HRF-wrapped event.
    pub fn new(inner: Box<dyn EventSpec>, hrf_spec: HrfEnumSpec) -> Self {
        Self {
            inner,
            hrf_spec,
            precision: 0.1,
            runs: None,
        }
    }

    /// Get a reference to the inner event.
    pub fn inner(&self) -> &dyn EventSpec {
        self.inner.as_ref()
    }

    /// Use a specified microtime precision in seconds.
    pub fn with_precision(mut self, precision: f64) -> Result<Self, ValidationError> {
        if !precision.is_finite() || precision <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "precision must be positive and finite".into(),
            ));
        }
        self.precision = precision;
        Ok(self)
    }

    /// Assign each event to a zero-based run. Onsets are local to that run.
    /// The sampling frame retains empty runs and prevents cross-run HRF tails.
    pub fn with_runs(
        mut self,
        frame: SamplingFrame,
        run_ids: Vec<usize>,
    ) -> Result<Self, ValidationError> {
        if run_ids.len() != self.inner.onsets().len()
            || run_ids.iter().any(|&r| r >= frame.n_blocks())
        {
            return Err(ValidationError::InvalidParameter(
                "run indices must match events and sampling frame".into(),
            ));
        }
        self.precision = frame.precision;
        self.runs = Some((frame, run_ids));
        Ok(self)
    }
}

impl EventSpec for HrfEvent {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn event_type(&self) -> EventType {
        self.inner.event_type()
    }

    fn onsets(&self) -> &OnsetArray {
        self.inner.onsets()
    }

    fn durations(&self) -> &DurationArray {
        self.inner.durations()
    }

    fn design_matrix(&self, sampling_points: &TimeArray) -> Result<Array, ValidationError> {
        let hrf = create_hrf(&self.hrf_spec)?;
        if let Some((frame, run_ids)) = &self.runs {
            if sampling_points != Array1::from(frame.samples()) {
                return Err(ValidationError::InvalidParameter(
                    "run-aware HRF must be evaluated on its sampling frame".into(),
                ));
            }
            let amplitudes = self.inner.event_amplitudes()?;
            let onsets = self.onsets().to_vec();
            let durations = self.durations().to_vec();
            EventTable {
                onsets: &onsets,
                durations: &durations,
                amplitudes: amplitudes.view(),
                run_ids,
            }
            .evaluate(frame, hrf.as_ref(), ConvolutionMethod::Auto)
            .map_err(|e| ValidationError::ComputationError(e.to_string()))
        } else {
            convolve_at_times(
                self.inner.as_ref(),
                hrf.as_ref(),
                sampling_points,
                hrf.span(),
                self.precision,
                "conv",
            )
        }
    }

    fn levels(&self) -> Option<&[String]> {
        self.inner.levels()
    }

    fn values(&self) -> Option<&[String]> {
        self.inner.values()
    }

    fn n_columns(&self) -> Option<usize> {
        Some(self.column_names().len())
    }

    fn column_names(&self) -> Vec<String> {
        let bases = create_hrf(&self.hrf_spec).map(|h| h.nbasis()).unwrap_or(1);
        crate::naming::add_basis_suffix(&self.inner.column_names(), bases)
    }

    fn to_dataframe(&self) -> DataFrame {
        // HRF wrapping doesn’t change the underlying event table.
        self.inner.to_dataframe()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EventSpec> {
        Box::new(Self {
            inner: self.inner.clone_box(),
            hrf_spec: self.hrf_spec.clone(),
            precision: self.precision,
            runs: self.runs.clone(),
        })
    }

    fn filter(&self, mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError> {
        let filtered_inner = self.inner.filter(mask)?;
        Ok(Box::new(Self {
            inner: filtered_inner,
            hrf_spec: self.hrf_spec.clone(),
            precision: self.precision,
            runs: self.runs.as_ref().map(|(frame, ids)| {
                (
                    frame.clone(),
                    ids.iter()
                        .zip(mask)
                        .filter_map(|(&id, &keep)| keep.then_some(id))
                        .collect(),
                )
            }),
        }))
    }
}
