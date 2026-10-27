//! Inspectable orchestration of immutable sources, named consumers and fit state.
//! Planning reads no voxel values. Numerical reservations exclude request/source
//! metadata, allocator excess capacity, backend scratch and unreserved sink copies.
use crate::{
    cohort::{run_cohort, CohortOptions},
    ingress::{file_preflight, load_subjects, FileSession, LoadOptions, NiftiSubject},
    outputs::{compile_requirements, CompiledOutput, ConsumerRequirement, OutputRequirementError},
    session::{
        DetachedSession, ModelSpec, NoisePolicy, SessionError, SubjectOutput, SubjectSession,
    },
};
use fmrireg::output::{OutputSpec, Uncertainty};
use std::{error::Error, fmt};

#[derive(Debug)]
pub enum AnalysisError {
    Invalid(&'static str),
    Session(SessionError),
    Requirements(OutputRequirementError),
}
impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) => f.write_str(s),
            Self::Session(e) => e.fmt(f),
            Self::Requirements(e) => e.fmt(f),
        }
    }
}
impl Error for AnalysisError {}
impl From<SessionError> for AnalysisError {
    fn from(e: SessionError) -> Self {
        Self::Session(e)
    }
}
impl From<OutputRequirementError> for AnalysisError {
    fn from(e: OutputRequirementError) -> Self {
        Self::Requirements(e)
    }
}
pub type Result<T> = std::result::Result<T, AnalysisError>;
fn add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or(SessionError::Overflow.into())
}
fn mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b).ok_or(SessionError::Overflow.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpectedUse {
    OneShot,
    ContrastExploration,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoragePolicy {
    Auto,
    Resident,
    FileBatches,
    /// Keep fitted coefficients/scales; release time series after each fit.
    RetainedFits,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetentionPolicy {
    None,
    CoefficientsAndScale,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageMode {
    Resident,
    FileBatches,
    /// Keep fitted coefficients/scales; release time series after each fit.
    RetainedFits,
}
#[derive(Clone, Copy, Debug)]
pub struct ExecutionPolicy {
    pub expected_use: ExpectedUse,
    pub storage: StoragePolicy,
    pub retention: RetentionPolicy,
    pub workers: usize,
    pub block_features: usize,
    /// Known numerical allocations, not an RSS limit.
    pub max_numeric_bytes: usize,
    /// Storage held by the caller throughout execution, e.g. a group collector.
    pub external_reserved_bytes: usize,
}
impl ExecutionPolicy {
    pub fn new(max_numeric_bytes: usize) -> Self {
        Self {
            expected_use: ExpectedUse::OneShot,
            storage: StoragePolicy::Auto,
            retention: RetentionPolicy::None,
            workers: 4,
            block_features: 2048,
            max_numeric_bytes,
            external_reserved_bytes: 0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct SubjectRequest {
    pub model_id: String,
    pub model: ModelSpec,
    pub consumers: Vec<ConsumerRequirement>,
}
pub struct AnalysisSubject {
    pub source: NiftiSubject,
    pub request: SubjectRequest,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataAction {
    Load,
    ReuseResident,
    Reload,
    NotNeeded,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelAction {
    Prepare,
    EstimateNoise,
    Reuse,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitAction {
    ScanData,
    BuildRetainedFit,
    UseRetainedFit,
}
#[derive(Clone, Debug)]
pub struct SubjectDecision {
    pub subject_id: String,
    pub source_version: String,
    pub model_id: String,
    pub data: DataAction,
    pub model: ModelAction,
    pub fit: FitAction,
    pub noise: NoisePolicy,
    pub rank_tolerance: f64,
    pub uncertainty: Uncertainty,
    pub estimand_ids: Vec<String>,
}
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    pub storage: StorageMode,
    pub workers: usize,
    pub retention: RetentionPolicy,
    pub reserved_numeric_bytes: usize,
    pub external_reserved_bytes: usize,
    pub resident_data_bytes: usize,
    pub retained_fit_bytes: usize,
    pub reason: &'static str,
    pub subjects: Vec<SubjectDecision>,
}
#[derive(Debug)]
pub struct ExecutionReceipt {
    pub plan: ExecutionPlan,
    pub subjects: usize,
    pub source_reads: usize,
    pub model_preparations: usize,
    pub model_reuses: usize,
    pub retained_fit_builds: usize,
}
enum State {
    Empty,
    Resident(Vec<SubjectSession>),
    Files(Vec<FileSession>),
    Retained {
        sessions: Vec<Option<DetachedSession>>,
        reads: usize,
    },
}

/// Callback checkpoints occur before work and between load/preparation/output batches.
/// Returning an error cancels future dispatch; already active jobs finish and join.
#[derive(Clone, Copy, Debug)]
pub struct AnalysisProgress {
    pub phase: &'static str,
    pub completed_subjects: usize,
    pub total_subjects: usize,
}

/// One entry point for planning and repeated execution. Sources are immutable for
/// its lifetime. Updates replace model/output requests transactionally, while
/// actual model/whitening equality (not model labels) controls reuse.
pub struct CohortAnalysis {
    sources: Vec<NiftiSubject>,
    models: Vec<ModelSpec>,
    model_ids: Vec<String>,
    outputs: Vec<CompiledOutput>,
    policy: ExecutionPolicy,
    state: State,
}
impl CohortAnalysis {
    pub fn new(subjects: Vec<AnalysisSubject>, policy: ExecutionPolicy) -> Result<Self> {
        if subjects.is_empty() || policy.workers == 0 || policy.block_features == 0 {
            return Err(AnalysisError::Invalid(
                "empty analysis or zero execution dimension",
            ));
        }
        if policy.external_reserved_bytes >= policy.max_numeric_bytes {
            return Err(AnalysisError::Invalid(
                "external reservation exhausts budget",
            ));
        }
        if policy.storage == StoragePolicy::FileBatches && policy.retention != RetentionPolicy::None
        {
            return Err(AnalysisError::Invalid(
                "file batches cannot retain fits; select RetainedFits storage",
            ));
        }
        if policy.storage == StoragePolicy::RetainedFits
            && policy.retention == RetentionPolicy::None
        {
            return Err(AnalysisError::Invalid(
                "RetainedFits storage requires explicit coefficient retention",
            ));
        }
        let (sources, requests): (Vec<_>, Vec<_>) =
            subjects.into_iter().map(|s| (s.source, s.request)).unzip();
        let mut ids = std::collections::HashSet::new();
        if sources.iter().any(|s| !ids.insert(s.subject_id())) {
            return Err(AnalysisError::Invalid("duplicate subject ID"));
        }
        let (models, model_ids, outputs) = compile(&sources, requests)?;
        let analysis = Self {
            sources,
            models,
            model_ids,
            outputs,
            policy,
            state: State::Empty,
        };
        analysis.inspect()?;
        Ok(analysis)
    }
    pub fn sources(&self) -> &[NiftiSubject] {
        &self.sources
    }
    pub fn compiled_outputs(&self) -> &[CompiledOutput] {
        &self.outputs
    }
    pub fn policy(&self) -> ExecutionPolicy {
        self.policy
    }
    /// No voxel reads or state mutation. Consumer maps preserve requested order.
    pub fn inspect(&self) -> Result<ExecutionPlan> {
        self.plan_for(&self.models, &self.model_ids, &self.outputs)
    }
    /// Validate/plan before replacing anything. On failure previous requests and
    /// fit state remain usable. Construct a new analysis to change source data.
    pub fn update(&mut self, requests: Vec<SubjectRequest>) -> Result<()> {
        let (models, ids, outputs) = compile(&self.sources, requests)?;
        self.plan_for(&models, &ids, &outputs)?;
        self.models = models;
        self.model_ids = ids;
        self.outputs = outputs;
        Ok(())
    }
    fn model_reused(&self, i: usize, model: &ModelSpec, mode: StorageMode) -> bool {
        match (&self.state, mode) {
            (State::Resident(s), StorageMode::Resident) => s[i].matches_model(model),
            (State::Files(s), StorageMode::FileBatches) => s[i].matches_model(model),
            (State::Retained { sessions, .. }, StorageMode::RetainedFits) => {
                sessions[i].as_ref().is_some_and(|s| s.matches_model(model))
            }
            _ => false,
        }
    }
    fn retained_reused(&self, i: usize, model: &ModelSpec) -> bool {
        match &self.state {
            State::Resident(s) => s[i].matches_model(model) && s[i].retained_numeric_bytes() > 0,
            State::Retained { sessions, .. } => {
                sessions[i].as_ref().is_some_and(|s| s.matches_model(model))
            }
            _ => false,
        }
    }
    fn plan_for(
        &self,
        models: &[ModelSpec],
        ids: &[String],
        outputs: &[CompiledOutput],
    ) -> Result<ExecutionPlan> {
        let limit = self.policy.max_numeric_bytes - self.policy.external_reserved_bytes;
        let requests = outputs
            .iter()
            .map(|o| o.output().clone())
            .collect::<Vec<_>>();
        let wants_resident = self.policy.storage == StoragePolicy::Resident
            || self.policy.retention != RetentionPolicy::None
            || (self.policy.storage == StoragePolicy::Auto
                && self.policy.expected_use == ExpectedUse::ContrastExploration);
        let resident = if wants_resident && self.policy.storage != StoragePolicy::RetainedFits {
            self.resident_plan(models, &requests, limit)?
        } else {
            None
        };
        let (mode, workers, required, reason) = if let Some((w, bytes)) = resident {
            (
                StorageMode::Resident,
                w,
                bytes,
                "resident data supports repeated requests; retention is explicit",
            )
        } else if self.policy.storage == StoragePolicy::RetainedFits
            || (self.policy.storage == StoragePolicy::Auto
                && self.policy.retention != RetentionPolicy::None)
        {
            let (workers, required, _) = self.detached_plan(models, &requests, limit)?;
            (StorageMode::RetainedFits, workers, required,
             "retained fits answer contrasts without time series; changed models reload bounded subject batches")
        } else {
            if self.policy.storage == StoragePolicy::Resident
                || self.policy.retention != RetentionPolicy::None
            {
                return Err(AnalysisError::Invalid(
                    "resident execution/required retention exceeds numeric budget",
                ));
            }
            let sources = self.sources.iter().collect::<Vec<_>>();
            let cached = match &self.state {
                State::Files(s) => s
                    .iter()
                    .map(FileSession::resident_numeric_bytes)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
                _ => vec![0; sources.len()],
            };
            let reused = models
                .iter()
                .enumerate()
                .map(|(i, m)| self.model_reused(i, m, StorageMode::FileBatches))
                .collect::<Vec<_>>();
            let plan = file_preflight(
                &sources,
                &cached,
                &reused,
                models,
                &requests,
                CohortOptions {
                    workers: self.policy.workers,
                    block_features: self.policy.block_features,
                    max_numeric_bytes: limit,
                },
            )?;
            (
                StorageMode::FileBatches,
                plan.workers,
                plan.total,
                if wants_resident {
                    "resident forecast exceeds budget; file batches preserve the requested model"
                } else {
                    "one-shot or explicitly selected file batches release time series"
                },
            )
        };
        let mut decisions = Vec::new();
        let mut data = 0;
        let mut retained = 0;
        for (i, ((source, model), output)) in
            self.sources.iter().zip(models).zip(outputs).enumerate()
        {
            if mode == StorageMode::Resident {
                data = add(data, source.data_bytes())?;
            }
            if self.policy.retention != RetentionPolicy::None {
                retained = add(
                    retained,
                    mul(mul(model.design.ncols() + 1, source.shape()[1])?, 8)?,
                )?;
            }
            let reused = self.model_reused(i, model, mode);
            let data_action = match (&self.state, mode) {
                (State::Resident(_), StorageMode::Resident) => DataAction::ReuseResident,
                (State::Retained { .. }, StorageMode::RetainedFits) if reused => {
                    DataAction::NotNeeded
                }
                (State::Retained { sessions, .. }, StorageMode::RetainedFits)
                    if sessions[i].is_some() =>
                {
                    DataAction::Reload
                }
                (State::Files(s), StorageMode::FileBatches) if s[i].read_count() > 0 => {
                    DataAction::Reload
                }
                _ => DataAction::Load,
            };
            decisions.push(SubjectDecision {
                subject_id: source.subject_id().into(),
                source_version: source.source_version().into(),
                model_id: ids[i].clone(),
                data: data_action,
                model: if reused {
                    ModelAction::Reuse
                } else if matches!(model.noise, NoisePolicy::EstimateAr { .. }) {
                    ModelAction::EstimateNoise
                } else {
                    ModelAction::Prepare
                },
                fit: if self.policy.retention == RetentionPolicy::None {
                    FitAction::ScanData
                } else if self.retained_reused(i, model)
                    && (mode == StorageMode::Resident || mode == StorageMode::RetainedFits)
                {
                    FitAction::UseRetainedFit
                } else {
                    FitAction::BuildRetainedFit
                },
                noise: model.noise.clone(),
                rank_tolerance: model.tolerance,
                uncertainty: output.output().uncertainty(),
                estimand_ids: output.output().estimands().ids().to_vec(),
            });
        }
        Ok(ExecutionPlan {
            storage: mode,
            workers,
            retention: self.policy.retention,
            reserved_numeric_bytes: add(required, self.policy.external_reserved_bytes)?,
            external_reserved_bytes: self.policy.external_reserved_bytes,
            resident_data_bytes: data,
            retained_fit_bytes: retained,
            reason,
            subjects: decisions,
        })
    }
    fn resident_plan(
        &self,
        models: &[ModelSpec],
        requests: &[OutputSpec],
        limit: usize,
    ) -> Result<Option<(usize, usize)>> {
        let data = self
            .sources
            .iter()
            .try_fold(0usize, |n, s| add(n, s.data_bytes()))?;
        let mut base = data;
        let mut preparation = 0;
        let mut retention_work = 0;
        let mut jobs = Vec::new();
        for (i, (source, model)) in self.sources.iter().zip(models).enumerate() {
            let old = match &self.state {
                State::Resident(s) => s[i].resident_numeric_bytes()? - s[i].data_bytes(),
                _ => 0,
            };
            base = add(base, old)?;
            if !self.model_reused(i, model, StorageMode::Resident) {
                base = add(base, model.resident_bytes()?)?;
            }
            preparation = preparation.max(model.preparation_bytes()?);
            if self.policy.retention != RetentionPolicy::None && !self.retained_reused(i, model) {
                base = add(
                    base,
                    mul(mul(model.design.ncols() + 1, source.shape()[1])?, 8)?,
                )?;
                let rows = add(
                    add(
                        mul(
                            model.design.nrows(),
                            if matches!(model.noise, NoisePolicy::Ols) {
                                1
                            } else {
                                2
                            },
                        )?,
                        model.design.ncols(),
                    )?,
                    1,
                )?;
                retention_work = retention_work.max(mul(
                    mul(rows, self.policy.block_features.min(source.shape()[1]))?,
                    8,
                )?);
            }
            // Conservative even with a retained fit: includes the scan workspace.
            jobs.push(model.execution_bytes(
                source.shape()[1],
                &requests[i],
                self.policy.block_features,
            )?);
        }
        for workers in (1..=self.policy.workers.min(models.len())).rev() {
            let mut peak_jobs = 0;
            let mut peak_load = 0;
            let mut loaded = 0;
            for batch in jobs.chunks(workers) {
                peak_jobs = peak_jobs.max(batch.iter().try_fold(0usize, |a, b| add(a, *b))?);
            }
            for batch in self.sources.chunks(workers) {
                peak_load = peak_load.max(
                    batch
                        .iter()
                        .try_fold(loaded, |a, s| add(a, s.load_required_bytes()))?,
                );
                loaded = batch
                    .iter()
                    .try_fold(loaded, |a, s| add(a, s.data_bytes()))?;
            }
            let peak = add(base, preparation.max(retention_work).max(peak_jobs))?.max(peak_load);
            if peak <= limit {
                return Ok(Some((workers, peak)));
            }
        }
        Ok(None)
    }
    fn detached_plan(
        &self,
        models: &[ModelSpec],
        requests: &[OutputSpec],
        limit: usize,
    ) -> Result<(usize, usize, Vec<usize>)> {
        let mut base = 0;
        let mut jobs = Vec::new();
        for (i, (source, model)) in self.sources.iter().zip(models).enumerate() {
            let old = match &self.state {
                State::Retained { sessions, .. } => sessions[i].as_ref(),
                _ => None,
            };
            base = add(
                base,
                old.map_or(Ok(0), DetachedSession::resident_numeric_bytes)?,
            )?;
            if old.is_some_and(|s| s.matches_model(model)) {
                jobs.push(old.unwrap().run_required_bytes(
                    &requests[i],
                    self.policy.block_features,
                    true,
                )?);
            } else {
                let n = model.design.nrows();
                let p = model.design.ncols();
                let v = source.shape()[1];
                let retained = mul(mul(add(p, 1)?, v)?, 8)?;
                // Reserve all old caches and every replacement, even though replacements
                // are committed individually. Decoder/data, prepare, fit and output work
                // are conservatively summed; active jobs cannot exceed this forecast.
                base = add(base, add(model.resident_bytes()?, retained)?)?;
                let rows = add(
                    add(
                        mul(
                            n,
                            if matches!(model.noise, NoisePolicy::Ols) {
                                1
                            } else {
                                2
                            },
                        )?,
                        p,
                    )?,
                    1,
                )?;
                let fit_work = mul(mul(rows, self.policy.block_features.min(v))?, 8)?;
                jobs.push(add(
                    add(source.load_required_bytes(), model.preparation_bytes()?)?,
                    add(
                        fit_work,
                        model.execution_bytes(v, &requests[i], self.policy.block_features)?,
                    )?,
                )?);
            }
        }
        let reservation = |workers: usize| -> Result<usize> {
            let mut peak = 0;
            for batch in jobs.chunks(workers) {
                peak = peak.max(batch.iter().try_fold(0usize, |a, b| add(a, *b))?);
            }
            add(base, peak)
        };
        let mut workers = self.policy.workers.min(models.len());
        while workers > 1 && reservation(workers)? > limit {
            workers -= 1;
        }
        let required = reservation(workers)?;
        if required > limit {
            return Err(SessionError::Budget {
                required,
                budget: limit,
            }
            .into());
        }
        Ok((workers, required, jobs))
    }
    fn counts(&self) -> (usize, usize, usize) {
        match &self.state {
            State::Empty => (0, 0, 0),
            State::Retained { sessions, reads } => (
                *reads,
                sessions
                    .iter()
                    .flatten()
                    .map(|s| s.preparation_counts().prepared)
                    .sum(),
                sessions
                    .iter()
                    .flatten()
                    .map(|s| s.preparation_counts().reused)
                    .sum(),
            ),
            State::Resident(s) => (
                s.iter().map(SubjectSession::read_count).sum(),
                s.iter().map(|s| s.preparation_counts().prepared).sum(),
                s.iter().map(|s| s.preparation_counts().reused).sum(),
            ),
            State::Files(s) => (
                s.iter().map(FileSession::read_count).sum(),
                s.iter().map(|s| s.preparation_counts().prepared).sum(),
                s.iter().map(|s| s.preparation_counts().reused).sum(),
            ),
        }
    }
    /// Execute the inspected policy. Sinks receive the compiled consumer mapping
    /// alongside identified output; already delivered results are not rolled back.
    /// No process-wide BLAS settings are changed.
    pub fn run(
        &mut self,
        sink: impl FnMut(
            usize,
            &SubjectOutput,
            &CompiledOutput,
        ) -> std::result::Result<(), SessionError>,
    ) -> Result<ExecutionReceipt> {
        self.run_with_progress(|_| Ok(()), sink)
    }
    /// Fallible progress callbacks provide cooperative cancellation. Callbacks run
    /// on the caller thread; an active subject batch always finishes before return.
    pub fn run_with_progress(
        &mut self,
        mut progress: impl FnMut(AnalysisProgress) -> std::result::Result<(), SessionError>,
        mut sink: impl FnMut(
            usize,
            &SubjectOutput,
            &CompiledOutput,
        ) -> std::result::Result<(), SessionError>,
    ) -> Result<ExecutionReceipt> {
        let total_subjects = self.sources.len();
        progress(AnalysisProgress {
            phase: "starting",
            completed_subjects: 0,
            total_subjects,
        })?;
        let plan = self.inspect()?;
        if !matches!(
            (&self.state, plan.storage),
            (State::Empty, _)
                | (State::Resident(_), StorageMode::Resident)
                | (State::Files(_), StorageMode::FileBatches)
                | (State::Retained { .. }, StorageMode::RetainedFits)
        ) {
            self.state = State::Empty;
        }
        let before = self.counts();
        let mut builds = 0;
        let limit = self.policy.max_numeric_bytes - self.policy.external_reserved_bytes;
        if matches!(self.state, State::Empty) {
            self.state = match plan.storage {
                StorageMode::Resident => {
                    let mut sessions = Vec::new();
                    let mut loaded = 0;
                    for batch in self.sources.chunks(plan.workers) {
                        progress(AnalysisProgress {
                            phase: "loading",
                            completed_subjects: sessions.len(),
                            total_subjects,
                        })?;
                        let (mut next, _) = load_subjects(
                            batch,
                            LoadOptions {
                                workers: plan.workers,
                                max_numeric_bytes: limit - loaded,
                            },
                        )?;
                        loaded = add(
                            loaded,
                            batch
                                .iter()
                                .try_fold(0usize, |a, s| add(a, s.data_bytes()))?,
                        )?;
                        sessions.append(&mut next);
                    }
                    State::Resident(sessions)
                }
                StorageMode::RetainedFits => State::Retained {
                    sessions: (0..total_subjects).map(|_| None).collect(),
                    reads: 0,
                },
                StorageMode::FileBatches => {
                    State::Files(self.sources.iter().cloned().map(FileSession::new).collect())
                }
            };
        }
        let requests = self
            .outputs
            .iter()
            .map(|o| o.output().clone())
            .collect::<Vec<_>>();
        let options = CohortOptions {
            workers: plan.workers,
            block_features: self.policy.block_features,
            max_numeric_bytes: limit,
        };
        let detached_jobs = if plan.storage == StorageMode::RetainedFits {
            self.detached_plan(&self.models, &requests, limit)?.2
        } else {
            Vec::new()
        };
        match &mut self.state {
            State::Resident(sessions) => {
                for i in 0..sessions.len() {
                    progress(AnalysisProgress {
                        phase: "preparing",
                        completed_subjects: i,
                        total_subjects,
                    })?;
                    let total = sessions
                        .iter()
                        .try_fold(0usize, |n, s| add(n, s.resident_numeric_bytes()?))?;
                    let other = total - sessions[i].resident_numeric_bytes()?;
                    let available = limit
                        .checked_sub(add(other, sessions[i].data_bytes())?)
                        .ok_or(AnalysisError::Invalid("preparation reservation"))?;
                    sessions[i].prepare(&self.models[i], available)?;
                    if self.policy.retention == RetentionPolicy::CoefficientsAndScale {
                        let total = sessions
                            .iter()
                            .try_fold(0usize, |n, s| add(n, s.resident_numeric_bytes()?))?;
                        let fresh = sessions[i].retained_numeric_bytes() == 0;
                        sessions[i].retain_for_edits(
                            self.policy.block_features,
                            limit
                                .checked_sub(total)
                                .ok_or(AnalysisError::Invalid("retention reservation"))?,
                        )?;
                        builds += usize::from(fresh);
                    }
                }
                progress(AnalysisProgress {
                    phase: "fitting",
                    completed_subjects: 0,
                    total_subjects,
                })?;
                run_cohort(sessions, &requests, options, |i, out| {
                    progress(AnalysisProgress {
                        phase: "fitting",
                        completed_subjects: i + 1,
                        total_subjects,
                    })?;
                    sink(i, out, &self.outputs[i])
                })?;
            }
            State::Files(sessions) => {
                crate::ingress::run_file_cohort(
                    sessions,
                    &self.models,
                    &requests,
                    options,
                    |i, out| {
                        progress(AnalysisProgress {
                            phase: "fitting",
                            completed_subjects: i + 1,
                            total_subjects,
                        })?;
                        sink(i, out, &self.outputs[i])
                    },
                )?;
            }
            State::Retained { sessions, reads } => {
                for (batch_index, batch) in sessions.chunks_mut(plan.workers).enumerate() {
                    let start = batch_index * plan.workers;
                    progress(AnalysisProgress {
                        phase: "fitting",
                        completed_subjects: start,
                        total_subjects,
                    })?;
                    std::thread::scope(|scope| -> Result<()> {
                        let mut handles = Vec::new();
                        for (offset, slot) in batch.iter_mut().enumerate() {
                            let i = start + offset;
                            let source = &self.sources[i];
                            let model = &self.models[i];
                            let output = &requests[i];
                            let block = self.policy.block_features;
                            let job = detached_jobs[i];
                            // Count attempted reads even when a later load/preparation fails.
                            let fresh = !slot.as_ref().is_some_and(|s| s.matches_model(model));
                            *reads += usize::from(fresh);
                            handles.push(scope.spawn(move || -> std::result::Result<(SubjectOutput, bool), SessionError> {
                                if fresh {
                                    let loaded = source.load(source.load_required_bytes())?;
                                    let mut candidate = match slot.as_ref() {
                                        Some(old) => old.replacement(loaded)?,
                                        None => SubjectSession::new(loaded),
                                    };
                                    candidate.prepare(model, model.preparation_bytes()?)?;
                                    let retention = candidate.retention_required_bytes(block)?;
                                    candidate.retain_for_edits(block, retention)?;
                                    *slot = Some(candidate.detach());
                                } else { slot.as_mut().unwrap().record_reuse(); }
                                let output = slot.as_ref().unwrap().run_owned(output, block, job)?;
                                Ok((output, fresh))
                            }));
                        }
                        for (offset, handle) in handles.into_iter().enumerate() {
                            let (out, fresh) = handle.join().map_err(|_| {
                                SessionError::Invalid("retained subject worker panicked")
                            })??;
                            builds += usize::from(fresh);
                            let i = start + offset;
                            progress(AnalysisProgress {
                                phase: "fitting",
                                completed_subjects: i + 1,
                                total_subjects,
                            })?;
                            sink(i, &out, &self.outputs[i])?;
                        }
                        Ok(())
                    })?;
                }
            }
            State::Empty => unreachable!(),
        }
        let after = self.counts();
        Ok(ExecutionReceipt {
            plan,
            subjects: self.sources.len(),
            source_reads: after.0 - before.0,
            model_preparations: after.1 - before.1,
            model_reuses: after.2 - before.2,
            retained_fit_builds: builds,
        })
    }
}
type CompiledRequests = (Vec<ModelSpec>, Vec<String>, Vec<CompiledOutput>);
fn compile(sources: &[NiftiSubject], requests: Vec<SubjectRequest>) -> Result<CompiledRequests> {
    if sources.len() != requests.len() {
        return Err(AnalysisError::Invalid("one request required per source"));
    }
    let mut models = Vec::new();
    let mut ids = Vec::new();
    let mut outputs = Vec::new();
    for (source, request) in sources.iter().zip(requests) {
        request.model.validate_metadata(source.observation_ids())?;
        if request.model_id.trim().is_empty()
            || request
                .consumers
                .iter()
                .any(|r| r.subject_id() != source.subject_id() || r.model_id() != request.model_id)
        {
            return Err(AnalysisError::Invalid(
                "consumer subject/model binding mismatch",
            ));
        }
        validate_noise(&request.model, source.shape())?;
        outputs.push(compile_requirements(
            &request.model.columns,
            &request.consumers,
        )?);
        models.push(request.model);
        ids.push(request.model_id);
    }
    Ok((models, ids, outputs))
}
fn validate_noise(model: &ModelSpec, shape: [usize; 2]) -> Result<()> {
    let runs = match &model.noise {
        NoisePolicy::Ols => return Ok(()),
        NoisePolicy::FixedAr { run_lengths, .. } | NoisePolicy::EstimateAr { run_lengths, .. } => {
            run_lengths
        }
    };
    if runs.contains(&0) || runs.iter().try_fold(0usize, |a, b| add(a, *b))? != shape[0] {
        return Err(AnalysisError::Invalid(
            "noise runs must partition observations",
        ));
    }
    match &model.noise {
        NoisePolicy::FixedAr {
            coefficients,
            resets,
            ..
        } => {
            if coefficients.len() != runs.len()
                || coefficients
                    .iter()
                    .any(|c| c.iter().any(|v| !v.is_finite()))
                || resets.iter().any(|r| *r >= shape[0])
            {
                return Err(AnalysisError::Invalid("fixed AR coefficients or resets"));
            }
        }
        NoisePolicy::EstimateAr {
            noise_valid,
            feature_indices,
            order,
            correction_max_lag,
            ..
        } => {
            let mut ids = std::collections::HashSet::new();
            if model.tolerance.to_bits() != 1e-12f64.to_bits()
                || *order == 0
                || *correction_max_lag < *order
                || noise_valid.as_ref().is_some_and(|v| v.len() != shape[0])
                || feature_indices.is_empty()
                || feature_indices
                    .iter()
                    .any(|i| *i >= shape[1] || !ids.insert(i))
            {
                return Err(AnalysisError::Invalid(
                    "estimated AR policy, tolerance or feature sample",
                ));
            }
        }
        NoisePolicy::Ols => {}
    }
    Ok(())
}
