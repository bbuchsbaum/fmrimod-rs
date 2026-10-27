//! Ordered cohort execution under a single explicit numeric-memory budget.
//! Each active subject uses one regression worker. Backend threads remain a
//! process-level caller setting (for Accelerate set VECLIB_MAXIMUM_THREADS=1).
use crate::session::{SessionError, SubjectOutput, SubjectSession};
use fmrireg::output::OutputSpec;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug)]
pub struct CohortOptions {
    pub workers: usize,
    pub block_features: usize,
    /// Resident data/model estimates plus bounded active numerical work. Does
    /// not include allocator/backend scratch or output copies retained by sinks.
    pub max_numeric_bytes: usize,
}
#[derive(Debug)]
pub struct CohortReceipt {
    pub subjects: usize,
    pub active_workers: usize,
    pub resident_numeric_bytes: usize,
    pub reserved_numeric_bytes: usize,
}

/// Run one resolved request per subject. Results arrive in input order. A fixed
/// batch holds at most `workers` owned results, including completed-but-not-yet
/// delivered ones; no unbounded reorder queue is used. On error active jobs are
/// joined and no further batch is dispatched. Delivered results are not rolled
/// back. The sink runs on the caller thread and can borrow results without copies.
pub fn run_cohort(
    subjects: &[SubjectSession],
    requests: &[OutputSpec],
    options: CohortOptions,
    mut sink: impl FnMut(usize, &SubjectOutput) -> Result<(), SessionError>,
) -> Result<CohortReceipt, SessionError> {
    if subjects.is_empty()
        || subjects.len() != requests.len()
        || options.workers == 0
        || options.block_features == 0
    {
        return Err(SessionError::Invalid(
            "cohort dimensions or execution options",
        ));
    }
    let mut ids = HashSet::new();
    let mut resident = 0usize;
    let mut per_job = Vec::with_capacity(subjects.len());
    for (subject, request) in subjects.iter().zip(requests) {
        if !ids.insert(subject.subject_id()) {
            return Err(SessionError::Invalid("duplicate cohort subject ID"));
        }
        resident = resident
            .checked_add(subject.resident_numeric_bytes()?)
            .ok_or(SessionError::Overflow)?;
        per_job.push(subject.run_required_bytes(request, options.block_features, true)?);
    }
    let workers = options.workers.min(subjects.len());
    let peak_jobs = per_job
        .chunks(workers)
        .map(|chunk| {
            chunk.iter().try_fold(0usize, |sum, v| {
                sum.checked_add(*v).ok_or(SessionError::Overflow)
            })
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap();
    let total = resident
        .checked_add(peak_jobs)
        .ok_or(SessionError::Overflow)?;
    if total > options.max_numeric_bytes {
        return Err(SessionError::Budget {
            required: total,
            budget: options.max_numeric_bytes,
        });
    }
    for start in (0..subjects.len()).step_by(workers) {
        let end = (start + workers).min(subjects.len());
        std::thread::scope(|scope| -> Result<(), SessionError> {
            let mut jobs = Vec::new();
            for index in start..end {
                let subject = &subjects[index];
                let request = &requests[index];
                let budget = per_job[index];
                jobs.push(
                    scope.spawn(move || subject.run_owned(request, options.block_features, budget)),
                );
            }
            // Scoped joins also finish all remaining jobs when a result/sink
            // fails. Each handle retains its result inside the fixed batch.
            for (offset, job) in jobs.into_iter().enumerate() {
                let output = job
                    .join()
                    .map_err(|_| SessionError::Invalid("subject worker panicked"))??;
                sink(start + offset, &output)?;
            }
            Ok(())
        })?;
    }
    Ok(CohortReceipt {
        subjects: subjects.len(),
        active_workers: workers,
        resident_numeric_bytes: resident,
        reserved_numeric_bytes: total,
    })
}

/// Streaming scalar bridge to the existing independent-subject group provider.
/// Allocate once and copy only the selected effect (and optional variance) from
/// each completed subject. Matching IDs is not evidence of scientific
/// equivalence: the caller supplies the estimand definition and admission.
pub struct ScalarCohortBuilder {
    subjects: Vec<crate::group::Subject>,
    estimand: crate::group::Estimand,
    features: crate::dataset::FeatureAxis,
    admission: crate::group::Admission,
    effects: Vec<f64>,
    variance: Option<Vec<f64>>,
    origins: Vec<crate::group::VarianceOrigin>,
    fits: Vec<crate::group::FitReceipt>,
    next: usize,
}
impl ScalarCohortBuilder {
    pub fn new(
        subjects: Vec<crate::group::Subject>,
        estimand: crate::group::Estimand,
        features: crate::dataset::FeatureAxis,
        admission: crate::group::Admission,
        with_variance: bool,
        budget: usize,
    ) -> crate::group::Result<Self> {
        let count = subjects
            .len()
            .checked_mul(features.axis().len())
            .ok_or(crate::group::Error::Budget)?;
        let bytes = count
            .checked_mul(if with_variance { 16 } else { 8 })
            .ok_or(crate::group::Error::Budget)?;
        if count == 0 || estimand.id.trim().is_empty() {
            return Err(crate::group::Error::Invalid("empty scalar cohort"));
        }
        if bytes > budget {
            return Err(crate::group::Error::Budget);
        }
        Ok(Self {
            subjects,
            estimand,
            features,
            admission,
            effects: vec![0.; count],
            variance: with_variance.then(|| vec![0.; count]),
            origins: Vec::new(),
            fits: Vec::new(),
            next: 0,
        })
    }
    /// Excludes semantic metadata and provider-private scratch. Reserve this in
    /// addition to the cohort executor's own budget while the collector lives.
    pub fn numeric_bytes(&self) -> usize {
        8 * (self.effects.len() + self.variance.as_ref().map_or(0, Vec::len))
    }
    pub fn push(&mut self, index: usize, output: &SubjectOutput) -> crate::group::Result<()> {
        use crate::group::{DegreesOfFreedom, FitReceipt, VarianceOrigin};
        use fmrireg::output::SummaryView;
        if index != self.next
            || index >= self.subjects.len()
            || self.subjects[index].observation.as_str() != output.subject_id()
        {
            return Err(crate::group::Error::Invalid(
                "scalar subject order/identity",
            ));
        }
        self.features
            .space()
            .assert_compatible(output.features().space())?;
        let nf = self.features.axis().len();
        let mut next_feature = 0;
        for (range, summary) in output.blocks() {
            if range.start != next_feature
                || range.end > nf
                || summary.estimates().ncols() != range.len()
            {
                return Err(crate::group::Error::Invalid("scalar summary feature order"));
            }
            let row = summary
                .estimand_ids()
                .iter()
                .position(|id| id == &self.estimand.id)
                .ok_or(crate::group::Error::Invalid(
                    "requested scalar estimand absent",
                ))?;
            if summary.residual_df() != output.receipt().residual_df {
                return Err(crate::group::Error::Invalid("summary df receipt mismatch"));
            }
            for (j, feature) in range.clone().enumerate() {
                let effect = summary.estimates()[(row, j)];
                if !effect.is_finite() {
                    return Err(crate::group::Error::Invalid("nonfinite first-level effect"));
                }
                self.effects[index * nf + feature] = effect;
                if let Some(variance) = self.variance.as_mut() {
                    let v = match summary.as_view() {
                        SummaryView::Marginal(v) => v.variance(row, j)?,
                        SummaryView::Joint(v) => v.variance(row, j)?,
                        SummaryView::Estimates(_) => {
                            return Err(crate::group::Error::Invalid("variance was not requested"))
                        }
                    };
                    variance[index * nf + feature] = v;
                }
            }
            next_feature = range.end;
        }
        if next_feature != nf {
            return Err(crate::group::Error::Invalid("incomplete subject summary"));
        }
        if self.variance.is_some() {
            self.origins.push(VarianceOrigin::Estimated {
                df: DegreesOfFreedom::Scalar(output.receipt().residual_df as f64),
                method: "first-level explicit residual scale".into(),
                approximate: output.receipt().estimated_noise,
            });
        }
        self.fits.push(FitReceipt {
            model: format!(
                "session:{}:generation:{}",
                output.receipt().source_version,
                output.receipt().model_generation
            ),
            noise: format!(
                "estimated={}; coefficients={:?}",
                output.receipt().estimated_noise,
                output.receipt().noise_coefficients
            ),
            selection: "all declared loaded observations, in exact order".into(),
        });
        self.next += 1;
        Ok(())
    }
    pub fn finish(self) -> crate::group::Result<crate::group::IndependentCohort> {
        use crate::dataset::{
            Assay, AssayAnnotations, Axis, FmriFrame, Metadata, NumericBlock, SourceSpec, Values,
        };
        if self.next != self.subjects.len() {
            return Err(crate::group::Error::Invalid("incomplete cohort"));
        }
        let ids = self
            .subjects
            .iter()
            .map(|s| s.observation.clone())
            .collect::<Vec<_>>();
        let shape = [self.subjects.len(), self.features.axis().len()];
        let assay = |name: &str, values: Vec<f64>| {
            Assay::new(
                name,
                SourceSpec::memory(NumericBlock::new(shape, Values::F64(values))?),
                ids.clone(),
                self.features.axis().ids().to_vec(),
                AssayAnnotations::default(),
            )
        };
        let mut assays = vec![assay("effect", self.effects)?];
        let uncertainty = if let Some(v) = self.variance {
            assays.push(assay("variance", v)?);
            Some(("variance".into(), self.origins))
        } else {
            None
        };
        let frame = FmriFrame::new(
            Axis::new(ids, Vec::new())?,
            self.features,
            assays,
            "effect",
            Metadata::new(),
        )?;
        crate::group::IndependentCohort::new(
            frame,
            self.subjects,
            self.estimand,
            "effect".into(),
            uncertainty,
            self.fits,
            self.admission,
        )
    }
}
