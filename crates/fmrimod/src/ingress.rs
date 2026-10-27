//! Bounded file admission. File versions are caller assertions, not content hashes.
use crate::{
    dataset::{ArrayReader, FeatureAxis, FeatureSpace, NiftiSource, ObservationId, ReadSelection},
    session::{LoadedSubject, SessionError, SubjectSession},
};

/// Immutable source identity and axes for an independently loadable subject.
#[derive(Clone, Debug)]
pub struct NiftiSubject {
    source: NiftiSource,
    rows: Vec<ObservationId>,
    features: FeatureAxis,
    id: String,
    version: String,
    output_bytes: usize,
    peak_bytes: usize,
}
impl NiftiSubject {
    pub fn new(
        source: NiftiSource,
        rows: Vec<ObservationId>,
        id: impl Into<String>,
        version: impl Into<String>,
    ) -> Result<Self, SessionError> {
        let id = id.into();
        let version = version.into();
        if id.trim().is_empty()
            || version.trim().is_empty()
            || rows.len() != source.shape()[0]
            || rows.is_empty()
        {
            return Err(SessionError::Invalid(
                "file subject identity or observations",
            ));
        }
        let mut seen = std::collections::HashSet::new();
        if rows.iter().any(|r| !seen.insert(r)) {
            return Err(SessionError::Invalid("duplicate observation ID"));
        }
        let reader = source.open_reader()?;
        let cost = reader.estimate(&ReadSelection::all(source.shape())?)?;
        let features = FeatureAxis::from_space(FeatureSpace::Volume(source.space().clone()))?;
        Ok(Self {
            source,
            rows,
            features,
            id,
            version,
            output_bytes: cost.output_bytes,
            peak_bytes: cost.peak_bytes,
        })
    }
    pub fn subject_id(&self) -> &str {
        &self.id
    }
    pub fn source_version(&self) -> &str {
        &self.version
    }
    pub fn observation_ids(&self) -> &[ObservationId] {
        &self.rows
    }
    pub fn features(&self) -> &FeatureAxis {
        &self.features
    }
    pub fn shape(&self) -> [usize; 2] {
        self.source.shape()
    }
    pub fn data_bytes(&self) -> usize {
        self.output_bytes
    }
    pub fn load_required_bytes(&self) -> usize {
        self.peak_bytes
    }
    pub fn load(&self, budget: usize) -> Result<LoadedSubject, SessionError> {
        if self.peak_bytes > budget {
            return Err(SessionError::Budget {
                required: self.peak_bytes,
                budget,
            });
        }
        let mut reader = self.source.open_reader()?;
        LoadedSubject::from_reader(
            &mut reader,
            self.rows.clone(),
            self.features.clone(),
            &self.id,
            &self.version,
            budget,
        )
    }
}
#[derive(Clone, Copy, Debug)]
pub struct LoadOptions {
    pub workers: usize,
    pub max_numeric_bytes: usize,
}
#[derive(Clone, Copy, Debug)]
pub struct LoadReceipt {
    pub active_workers: usize,
    pub resident_numeric_bytes: usize,
    pub reserved_numeric_bytes: usize,
}

/// Preflight all fixed batches before loading voxel values. Includes previously
/// loaded data plus each active reader's peak. Metadata/decompressor scratch is
/// outside the numeric bound. No partial cohort escapes on failure.
pub fn load_subjects(
    sources: &[NiftiSubject],
    options: LoadOptions,
) -> Result<(Vec<SubjectSession>, LoadReceipt), SessionError> {
    if sources.is_empty() || options.workers == 0 {
        return Err(SessionError::Invalid("load dimensions or workers"));
    }
    let mut ids = std::collections::HashSet::new();
    if sources.iter().any(|s| !ids.insert(s.subject_id())) {
        return Err(SessionError::Invalid("duplicate cohort subject ID"));
    }
    let workers = options.workers.min(sources.len());
    let mut resident = 0usize;
    let mut peak = 0usize;
    for batch in sources.chunks(workers) {
        let active = batch.iter().try_fold(resident, |a, s| {
            a.checked_add(s.peak_bytes).ok_or(SessionError::Overflow)
        })?;
        peak = peak.max(active);
        resident = batch.iter().try_fold(resident, |a, s| {
            a.checked_add(s.output_bytes).ok_or(SessionError::Overflow)
        })?;
    }
    if peak > options.max_numeric_bytes {
        return Err(SessionError::Budget {
            required: peak,
            budget: options.max_numeric_bytes,
        });
    }
    let mut loaded = Vec::with_capacity(sources.len());
    for batch in sources.chunks(workers) {
        std::thread::scope(|scope| -> Result<(), SessionError> {
            let handles = batch
                .iter()
                .map(|s| scope.spawn(move || s.load(s.peak_bytes)))
                .collect::<Vec<_>>();
            for h in handles {
                loaded
                    .push(SubjectSession::new(h.join().map_err(|_| {
                        SessionError::Invalid("load worker panicked")
                    })??));
            }
            Ok(())
        })?;
    }
    Ok((
        loaded,
        LoadReceipt {
            active_workers: workers,
            resident_numeric_bytes: resident,
            reserved_numeric_bytes: peak,
        },
    ))
}

/// On-demand data with prepared model state retained between requests. The
/// source/version must remain immutable; construct a new FileSession on change.
pub struct FileSession {
    source: NiftiSubject,
    prepared: Option<crate::session::DetachedSession>,
    reads: usize,
}
impl FileSession {
    pub(crate) fn matches_model(&self, spec: &crate::session::ModelSpec) -> bool {
        self.prepared
            .as_ref()
            .is_some_and(|s| s.matches_model(spec))
    }

    pub fn new(source: NiftiSubject) -> Self {
        Self {
            source,
            prepared: None,
            reads: 0,
        }
    }
    pub fn source(&self) -> &NiftiSubject {
        &self.source
    }
    pub fn read_count(&self) -> usize {
        self.reads
    }
    pub fn preparation_counts(&self) -> crate::session::PreparationCounts {
        self.prepared.as_ref().map_or(
            crate::session::PreparationCounts {
                prepared: 0,
                reused: 0,
            },
            |s| s.preparation_counts(),
        )
    }
    pub fn resident_numeric_bytes(&self) -> Result<usize, SessionError> {
        self.prepared
            .as_ref()
            .map_or(Ok(0), |s| s.resident_numeric_bytes())
    }
    fn run(
        &mut self,
        spec: &crate::session::ModelSpec,
        request: &fmri_reg::output::OutputSpec,
        block: usize,
        work: usize,
    ) -> Result<crate::session::SubjectOutput, SessionError> {
        let data = self.source.load(self.source.peak_bytes)?;
        self.reads += 1;
        let mut session = match self.prepared.take() {
            Some(s) => s.attach(data)?,
            None => SubjectSession::new(data),
        };
        let result = session
            .prepare(spec, work)
            .and_then(|_| session.run_owned(request, block, work));
        self.prepared = Some(session.detach());
        result
    }
}
/// Load, prepare and fit bounded subject batches, then release their data.
/// Prepared models survive across calls. Worker count is reduced to fit the
/// aggregate numeric budget, with preflight before any voxel read. Outputs are
/// delivered in order; completed work is not rolled back if a later job fails.
/// Arbitrary sink retention, source metadata and backend scratch are excluded.
pub fn run_file_cohort(
    sessions: &mut [FileSession],
    specs: &[crate::session::ModelSpec],
    requests: &[fmri_reg::output::OutputSpec],
    options: crate::cohort::CohortOptions,
    mut sink: impl FnMut(usize, &crate::session::SubjectOutput) -> Result<(), SessionError>,
) -> Result<crate::cohort::CohortReceipt, SessionError> {
    let sources = sessions.iter().map(|s| &s.source).collect::<Vec<_>>();
    let cached = sessions
        .iter()
        .map(FileSession::resident_numeric_bytes)
        .collect::<Result<Vec<_>, _>>()?;
    let reused = sessions
        .iter()
        .zip(specs)
        .map(|(s, m)| s.matches_model(m))
        .collect::<Vec<_>>();
    let FilePreflight {
        workers,
        total,
        jobs,
    } = file_preflight(&sources, &cached, &reused, specs, requests, options)?;
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(SessionError::Overflow);
    for (batch_index, batch) in sessions.chunks_mut(workers).enumerate() {
        let start = batch_index * workers;
        std::thread::scope(|scope| -> Result<(), SessionError> {
            let handles = batch
                .iter_mut()
                .enumerate()
                .map(|(offset, s)| {
                    let i = start + offset;
                    let spec = &specs[i];
                    let request = &requests[i];
                    // Existing model is already counted in aggregate resident bytes.
                    let work = jobs[i] - s.source.peak_bytes + s.resident_numeric_bytes()?;
                    Ok(scope.spawn(move || s.run(spec, request, options.block_features, work)))
                })
                .collect::<Result<Vec<_>, SessionError>>()?;
            for (offset, h) in handles.into_iter().enumerate() {
                let out = h
                    .join()
                    .map_err(|_| SessionError::Invalid("file subject worker panicked"))??;
                sink(start + offset, &out)?;
            }
            Ok(())
        })?;
    }
    Ok(crate::cohort::CohortReceipt {
        subjects: sessions.len(),
        active_workers: workers,
        resident_numeric_bytes: sessions
            .iter()
            .try_fold(0usize, |a, s| add(a, s.resident_numeric_bytes()?))?,
        reserved_numeric_bytes: total,
    })
}

pub(crate) struct FilePreflight {
    pub workers: usize,
    pub total: usize,
    jobs: Vec<usize>,
}
pub(crate) fn file_preflight(
    sources: &[&NiftiSubject],
    cached_bytes: &[usize],
    reused: &[bool],
    specs: &[crate::session::ModelSpec],
    requests: &[fmri_reg::output::OutputSpec],
    options: crate::cohort::CohortOptions,
) -> Result<FilePreflight, SessionError> {
    let add = |a: usize, b: usize| a.checked_add(b).ok_or(SessionError::Overflow);
    if sources.is_empty()
        || cached_bytes.len() != sources.len()
        || reused.len() != sources.len()
        || specs.len() != sources.len()
        || requests.len() != sources.len()
        || options.workers == 0
        || options.block_features == 0
    {
        return Err(SessionError::Invalid("file cohort dimensions or options"));
    }
    let mut ids = std::collections::HashSet::new();
    let mut resident = 0usize;
    let mut jobs = Vec::new();
    for (index, ((s, spec), request)) in sources.iter().zip(specs).zip(requests).enumerate() {
        if !ids.insert(s.subject_id()) {
            return Err(SessionError::Invalid("duplicate cohort subject ID"));
        }
        if spec.observation_ids != s.rows || spec.design.nrows() != s.shape()[0] {
            return Err(SessionError::Invalid("file subject/design observations"));
        }
        // Preserve old state plus replacement storage only when preparing a new
        // model. Reused models need no replacement allocation.
        resident = add(
            resident,
            add(
                cached_bytes[index],
                if reused[index] {
                    0
                } else {
                    spec.resident_bytes()?
                },
            )?,
        )?;
        let transient = add(
            spec.preparation_bytes()?,
            spec.execution_bytes(s.shape()[1], request, options.block_features)?,
        )?;
        jobs.push(add(s.peak_bytes, transient)?);
    }
    let reservation = |workers: usize| -> Result<usize, SessionError> {
        let mut peak = 0;
        for batch in jobs.chunks(workers) {
            peak = peak.max(batch.iter().try_fold(0usize, |a, b| add(a, *b))?);
        }
        add(resident, peak)
    };
    let mut workers = options.workers.min(sources.len());
    while workers > 1 && reservation(workers)? > options.max_numeric_bytes {
        workers -= 1;
    }
    let total = reservation(workers)?;
    if total > options.max_numeric_bytes {
        return Err(SessionError::Budget {
            required: total,
            budget: options.max_numeric_bytes,
        });
    }
    Ok(FilePreflight {
        workers,
        total,
        jobs,
    })
}
