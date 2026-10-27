//! Resident single-subject first-level sessions.
//!
//! Budgets in this module bound documented numeric allocations only.  They do
//! not account for allocator overhead or scratch allocated inside a matrix
//! backend / decomposition implementation.

use std::{error::Error, fmt, ops::Range, sync::Arc};

use ndarray::{Array2, Axis};

use crate::{
    dataset::{
        ArrayReader, ExecutionContext, FeatureAxis, MemoryBudget, ObservationId, ReadSelection,
        Values,
    },
    outputs::DesignColumns,
};
use fmri_noise::{ArDiagnostics, ArPlan, ArStart, Pooling};
use fmrireg::{
    output::{OutputSpec, Summary, SummaryView, Uncertainty},
    ArOptions, FitError, GlsModel, LinearModel, MatrixBackend,
};

const ESTIMATED_AR_TOLERANCE: f64 = 1e-12;

pub type Result<T> = std::result::Result<T, SessionError>;

#[derive(Debug)]
pub enum SessionError {
    Invalid(&'static str),
    Budget { required: usize, budget: usize },
    Overflow,
    Dataset(fmridataset::FmriDatasetError),
    Noise(fmri_noise::NoiseError),
    Fit(FitError),
}
impl fmt::Display for SessionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid session input: {message}"),
            Self::Budget { required, budget } => write!(
                f,
                "numeric allocation budget exceeded: required {required}, budget {budget}"
            ),
            Self::Overflow => write!(f, "numeric allocation size overflow"),
            Self::Dataset(error) => error.fmt(f),
            Self::Noise(error) => error.fmt(f),
            Self::Fit(error) => error.fmt(f),
        }
    }
}
impl Error for SessionError {}
impl From<fmridataset::FmriDatasetError> for SessionError {
    fn from(x: fmridataset::FmriDatasetError) -> Self {
        Self::Dataset(x)
    }
}
impl From<fmri_noise::NoiseError> for SessionError {
    fn from(x: fmri_noise::NoiseError) -> Self {
        Self::Noise(x)
    }
}
impl From<FitError> for SessionError {
    fn from(x: FitError) -> Self {
        Self::Fit(x)
    }
}

fn bytes(values: usize) -> Result<usize> {
    values
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(SessionError::Overflow)
}
fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right).ok_or(SessionError::Overflow)
}
fn mul(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or(SessionError::Overflow)
}

/// Immutable, f64 resident data and alignment metadata for one subject.
#[derive(Debug)]
pub struct LoadedSubject {
    data: Array2<f64>,
    observation_ids: Vec<ObservationId>,
    features: Arc<FeatureAxis>,
    subject_id: String,
    source_version: String,
    reads: usize,
}
impl LoadedSubject {
    pub fn from_reader(
        reader: &mut dyn ArrayReader,
        observation_ids: Vec<ObservationId>,
        features: FeatureAxis,
        subject_id: impl Into<String>,
        source_version: impl Into<String>,
        working_budget: usize,
    ) -> Result<Self> {
        let subject_id = subject_id.into();
        let source_version = source_version.into();
        if subject_id.is_empty() || source_version.is_empty() {
            return Err(SessionError::Invalid(
                "subject ID and source version must not be empty",
            ));
        }
        if reader.dtype() != fmridataset::DType::F64 {
            return Err(SessionError::Invalid(
                "f32 input requires explicit external conversion to f64",
            ));
        }
        let shape = reader.shape();
        if shape[0] == 0
            || shape[1] == 0
            || observation_ids.len() != shape[0]
            || features.axis().ids().len() != shape[1]
        {
            return Err(SessionError::Invalid(
                "resident data and axes have incompatible dimensions",
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(observation_ids.len());
        if observation_ids.iter().any(|id| !seen.insert(id.as_str())) {
            return Err(SessionError::Invalid("duplicate observation ID"));
        }
        let context = ExecutionContext::new(MemoryBudget::new(working_budget));
        let block = context.read(reader, &ReadSelection::all(shape)?)?;
        let shape = block.shape();
        let values = match block.into_values() {
            Values::F64(values) => values,
            Values::F32(_) => {
                return Err(SessionError::Invalid("reader changed dtype during read"))
            }
        };
        let data = Array2::from_shape_vec((shape[0], shape[1]), values)
            .map_err(|_| SessionError::Invalid("resident data shape"))?;
        Self::from_parts(
            data,
            observation_ids,
            features,
            subject_id,
            source_version,
            1,
        )
    }
    pub fn from_f64(
        data: Array2<f64>,
        observation_ids: Vec<ObservationId>,
        features: FeatureAxis,
        subject_id: impl Into<String>,
        source_version: impl Into<String>,
    ) -> Result<Self> {
        // Caller-owned arrays may retain an oversized backing allocation after
        // slicing. Canonicalize only this constructor; the reader path already
        // moves a compact row-major Vec and must not double its load budget.
        let data = data.as_standard_layout().into_owned();
        Self::from_parts(
            data,
            observation_ids,
            features,
            subject_id.into(),
            source_version.into(),
            0,
        )
    }
    fn from_parts(
        data: Array2<f64>,
        observation_ids: Vec<ObservationId>,
        features: FeatureAxis,
        subject_id: String,
        source_version: String,
        reads: usize,
    ) -> Result<Self> {
        if subject_id.is_empty()
            || source_version.is_empty()
            || data.nrows() == 0
            || data.ncols() == 0
            || observation_ids.len() != data.nrows()
            || features.axis().ids().len() != data.ncols()
        {
            return Err(SessionError::Invalid(
                "resident data and axes have incompatible dimensions",
            ));
        }
        let mut seen = std::collections::HashSet::with_capacity(observation_ids.len());
        if observation_ids.iter().any(|id| !seen.insert(id.as_str())) {
            return Err(SessionError::Invalid("duplicate observation ID"));
        }
        Ok(Self {
            data,
            observation_ids,
            features: Arc::new(features),
            subject_id,
            source_version,
            reads,
        })
    }
    pub fn data(&self) -> ndarray::ArrayView2<'_, f64> {
        self.data.view()
    }
    pub fn observation_ids(&self) -> &[ObservationId] {
        &self.observation_ids
    }
    pub fn features(&self) -> &FeatureAxis {
        &self.features
    }
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }
    pub fn source_version(&self) -> &str {
        &self.source_version
    }
    pub fn data_bytes(&self) -> usize {
        self.data.len() * 8
    }
    pub fn read_count(&self) -> usize {
        self.reads
    }
}

#[derive(Clone, Debug)]
pub enum NoisePolicy {
    Ols,
    FixedAr {
        run_lengths: Vec<usize>,
        coefficients: Vec<Vec<f64>>,
        startup: ArStart,
        resets: Vec<usize>,
    },
    EstimateAr {
        run_lengths: Vec<usize>,
        noise_valid: Option<Vec<bool>>,
        order: usize,
        correction_max_lag: usize,
        pooling: Pooling,
        startup: ArStart,
        feature_indices: Vec<usize>,
    },
}
impl NoisePolicy {
    fn same_as(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Ols, Self::Ols) => true,
            (
                Self::FixedAr {
                    run_lengths: a,
                    coefficients: b,
                    startup: c,
                    resets: d,
                },
                Self::FixedAr {
                    run_lengths: e,
                    coefficients: f,
                    startup: g,
                    resets: h,
                },
            ) => {
                a == e && b == f && std::mem::discriminant(c) == std::mem::discriminant(g) && d == h
            }
            (
                Self::EstimateAr {
                    run_lengths: a,
                    noise_valid: b,
                    order: c,
                    correction_max_lag: d,
                    pooling: e,
                    startup: f,
                    feature_indices: g,
                },
                Self::EstimateAr {
                    run_lengths: h,
                    noise_valid: i,
                    order: j,
                    correction_max_lag: k,
                    pooling: l,
                    startup: m,
                    feature_indices: n,
                },
            ) => {
                a == h
                    && b == i
                    && c == j
                    && d == k
                    && std::mem::discriminant(e) == std::mem::discriminant(l)
                    && std::mem::discriminant(f) == std::mem::discriminant(m)
                    && g == n
            }
            _ => false,
        }
    }
}

#[derive(Debug)]
pub struct ModelSpec {
    pub design: Array2<f64>,
    pub observation_ids: Vec<ObservationId>,
    pub columns: DesignColumns,
    pub noise: NoisePolicy,
    pub tolerance: f64,
    pub backend: MatrixBackend,
}
impl Clone for ModelSpec {
    fn clone(&self) -> Self {
        Self {
            design: self.design.to_owned(),
            observation_ids: self.observation_ids.clone(),
            columns: self.columns.clone(),
            noise: self.noise.clone(),
            tolerance: self.tolerance,
            backend: self.backend,
        }
    }
}
impl ModelSpec {
    fn same_as(&self, other: &Self) -> bool {
        self.design == other.design
            && self.observation_ids == other.observation_ids
            && self.columns.columns() == other.columns.columns()
            && self.noise.same_as(&other.noise)
            && self.tolerance.to_bits() == other.tolerance.to_bits()
            && self.backend == other.backend
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PreparationCounts {
    pub prepared: usize,
    pub reused: usize,
}
#[derive(Debug, Clone)]
pub struct PreparationReceipt {
    pub source_version: String,
    pub model_generation: usize,
    pub estimated_noise: bool,
    pub conditional_noise: bool,
    pub rank: usize,
    pub residual_df: usize,
    pub noise_coefficients: Option<Vec<Vec<f64>>>,
    pub noise_diagnostics: Option<Vec<ArDiagnostics>>,
}
#[derive(Debug, Clone)]
pub enum PrepareStatus {
    Prepared(PreparationReceipt),
    Reused(PreparationReceipt),
}
impl PrepareStatus {
    pub fn receipt(&self) -> &PreparationReceipt {
        match self {
            Self::Prepared(x) | Self::Reused(x) => x,
        }
    }
}

enum PreparedModel {
    Ols(LinearModel),
    Ar(GlsModel<ArPlan>),
}

/// One resident subject plus at most one prepared model.  A failed preparation
/// leaves the previous model intact.
pub struct SubjectSession {
    subject: LoadedSubject,
    cached: Option<(ModelSpec, PreparedModel, bool, PreparationReceipt)>,
    counts: PreparationCounts,
    generation: usize,
}
impl SubjectSession {
    pub fn new(subject: LoadedSubject) -> Self {
        Self {
            subject,
            cached: None,
            counts: PreparationCounts {
                prepared: 0,
                reused: 0,
            },
            generation: 0,
        }
    }
    pub fn subject(&self) -> &LoadedSubject {
        &self.subject
    }
    pub fn data_bytes(&self) -> usize {
        self.subject.data_bytes()
    }
    pub fn preparation_counts(&self) -> PreparationCounts {
        self.counts
    }
    pub fn subject_id(&self) -> &str {
        self.subject.subject_id()
    }
    pub fn features(&self) -> &FeatureAxis {
        self.subject.features()
    }
    pub fn read_count(&self) -> usize {
        self.subject.read_count()
    }
    pub fn resident_numeric_bytes(&self) -> Result<usize> {
        add(
            self.data_bytes(),
            self.cached
                .as_ref()
                .map_or(Ok(0), |(s, _, _, _)| resident_model_bytes(s))?,
        )
    }
    pub fn prepare(&mut self, spec: &ModelSpec, working_budget: usize) -> Result<PrepareStatus> {
        validate_spec(&self.subject, spec)?;
        if let Some((old, _, _, receipt)) = &self.cached {
            if old.same_as(spec) {
                let required = receipt_numeric_bytes(receipt)?;
                if required > working_budget {
                    return Err(SessionError::Budget {
                        required,
                        budget: working_budget,
                    });
                }
                self.counts.reused += 1;
                return Ok(PrepareStatus::Reused(receipt.clone()));
            }
        }
        let old = self.cached.as_ref().map_or(0, |(s, _, _, _)| {
            resident_model_bytes(s).unwrap_or(usize::MAX)
        });
        let required = add(old, preparation_bytes(spec)?)?;
        if required > working_budget {
            return Err(SessionError::Budget {
                required,
                budget: working_budget,
            });
        }
        let (model, estimated) = build_model(&self.subject, spec)?;
        self.generation += 1;
        let receipt = model_receipt(
            &model,
            spec,
            estimated,
            self.subject.source_version(),
            self.generation,
        )?;
        self.cached = Some((spec.clone(), model, estimated, receipt.clone()));
        self.counts.prepared += 1;
        Ok(PrepareStatus::Prepared(receipt))
    }
    pub fn run_required_bytes(
        &self,
        output: &OutputSpec,
        block_features: usize,
        retain: bool,
    ) -> Result<usize> {
        if block_features == 0 {
            return Err(SessionError::Invalid("zero block feature count"));
        }
        let (spec, model, _, _) = self
            .cached
            .as_ref()
            .ok_or(SessionError::Invalid("model has not been prepared"))?;
        let n = self.subject.data.nrows();
        let p = spec.design.ncols();
        if output.estimands().weights().ncols() != p {
            return Err(SessionError::Invalid(
                "output weights do not match prepared design",
            ));
        }
        let q = output.estimands().weights().nrows();
        let plan = output_preparation_bytes(n, p, q, output.uncertainty())?;
        let workspace = output_workspace_bytes(
            n,
            p,
            q,
            block_features,
            output.uncertainty(),
            matches!(model, PreparedModel::Ar(_)),
        )?;
        let retained = if retain {
            add(
                output_retained_bytes(q, self.subject.data.ncols(), output.uncertainty())?,
                receipt_numeric_bytes(&self.cached.as_ref().unwrap().3)?,
            )?
        } else {
            0
        };
        add(add(plan, workspace)?, retained)
    }
    pub fn run_blocks(
        &self,
        output: &OutputSpec,
        block_features: usize,
        working_budget: usize,
        mut sink: impl for<'a, 'b> FnMut(Range<usize>, &SummaryView<'a, 'b>) -> Result<()>,
    ) -> Result<()> {
        let required = self.run_required_bytes(output, block_features, false)?;
        if required > working_budget {
            return Err(SessionError::Budget {
                required,
                budget: working_budget,
            });
        }
        let (_, model, _, _) = self
            .cached
            .as_ref()
            .ok_or(SessionError::Invalid("model has not been prepared"))?;
        match model {
            PreparedModel::Ols(model) => run_ols(
                model,
                self.subject.data.view(),
                output,
                block_features,
                &mut sink,
            ),
            PreparedModel::Ar(model) => run_ar(
                model,
                self.subject.data.view(),
                output,
                block_features,
                &mut sink,
            ),
        }
    }
    pub fn run_owned(
        &self,
        output: &OutputSpec,
        block_features: usize,
        working_budget: usize,
    ) -> Result<SubjectOutput> {
        let required = self.run_required_bytes(output, block_features, true)?;
        if required > working_budget {
            return Err(SessionError::Budget {
                required,
                budget: working_budget,
            });
        }
        let receipt = self
            .cached
            .as_ref()
            .ok_or(SessionError::Invalid("model has not been prepared"))?
            .3
            .clone();
        let mut blocks = Vec::new();
        self.run_blocks(output, block_features, working_budget, |range, view| {
            blocks.push((range, view.to_owned()));
            Ok(())
        })?;
        Ok(SubjectOutput {
            subject_id: self.subject.subject_id.clone(),
            features: Arc::clone(&self.subject.features),
            blocks,
            receipt,
        })
    }
}

fn model_receipt(
    model: &PreparedModel,
    _spec: &ModelSpec,
    estimated_noise: bool,
    source_version: &str,
    model_generation: usize,
) -> Result<PreparationReceipt> {
    match model {
        PreparedModel::Ols(x) => Ok(PreparationReceipt {
            source_version: source_version.into(),
            model_generation,
            estimated_noise,
            conditional_noise: false,
            rank: x.rank(),
            residual_df: x.residual_df(),
            noise_coefficients: None,
            noise_diagnostics: None,
        }),
        PreparedModel::Ar(x) => {
            let plan = x.noise_plan();
            Ok(PreparationReceipt {
                source_version: source_version.into(),
                model_generation,
                estimated_noise,
                conditional_noise: true,
                rank: x.rank(),
                residual_df: x.residual_df(),
                noise_coefficients: Some(plan.coefficients().to_vec()),
                noise_diagnostics: plan.diagnostics().map(|d| d.to_vec()),
            })
        }
    }
}

fn validate_spec(subject: &LoadedSubject, spec: &ModelSpec) -> Result<()> {
    if spec.design.nrows() != subject.data.nrows()
        || spec.observation_ids != subject.observation_ids
        || spec.columns.len() != spec.design.ncols()
    {
        return Err(SessionError::Invalid(
            "design, columns, or observation IDs do not match subject",
        ));
    }
    if spec.design.ncols() == 0 || spec.design.iter().any(|x| !x.is_finite()) {
        return Err(SessionError::Invalid("empty or nonfinite design"));
    }
    if !spec.tolerance.is_finite() || spec.tolerance <= 0. || spec.tolerance >= 1. {
        return Err(SessionError::Invalid("invalid rank tolerance"));
    }
    Ok(())
}
fn preparation_bytes(spec: &ModelSpec) -> Result<usize> {
    let (n, p) = spec.design.dim();
    let np = mul(n, p)?;
    let p2 = mul(p, p)?;
    // Conservative sum of explicitly visible arrays across preparation phases.
    // Exposed SVD outputs are included; decomposition-private work arrays,
    // allocator excess capacity, and semantic/index metadata are excluded.
    let mut values = add(add(mul(2, np)?, p2)?, p)?;
    match &spec.noise {
        NoisePolicy::Ols => {}
        NoisePolicy::FixedAr { coefficients, .. } => {
            let ar = coefficients
                .iter()
                .try_fold(0, |sum, phi| add(sum, ar_temporary_values(phi.len())?))?;
            values = add(values, add(np, ar)?)?;
        }
        NoisePolicy::EstimateAr {
            run_lengths,
            order,
            correction_max_lag,
            feature_indices,
            ..
        } => {
            let k = feature_indices.len();
            let r = run_lengths.len();
            let l = add(*correction_max_lag, 1)?;
            let pp = add(p, 1)?;
            let parts = [
                np, // whitened design
                mul(r, ar_temporary_values(*order)?)?,
                add(mul(2, np)?, mul(2, p2)?)?, // live initial OLS model
                add(add(mul(3, mul(n, k)?)?, mul(p, k)?)?, k)?, // noise/fitted/residual
                add(mul(2, np)?, p)?,           // correction SVD input and exposed outputs
                np,                             // correction basis
                add(mul(4, mul(n, pp)?)?, mul(pp, pp)?)?, // bias_matrix temporaries
                add(mul(add(mul(2, r)?, 5)?, mul(l, l)?)?, mul(2, l)?)?,
                add(add(mul(p, k)?, k)?, mul(add(r, 4)?, l)?)?,
            ];
            for part in parts {
                values = add(values, part)?;
            }
        }
    }
    add(
        add(resident_model_bytes(spec)?, bytes(values)?)?,
        policy_receipt_bytes(&spec.noise)?,
    )
}
fn ar_temporary_values(order: usize) -> Result<usize> {
    let a = add(order, 1)?;
    add(mul(8, mul(a, a)?)?, mul(16, a)?)
}
fn policy_receipt_bytes(noise: &NoisePolicy) -> Result<usize> {
    match noise {
        NoisePolicy::Ols => Ok(0),
        NoisePolicy::FixedAr { coefficients, .. } => bytes(
            coefficients
                .iter()
                .try_fold(0, |sum, p| add(sum, p.len()))?,
        ),
        NoisePolicy::EstimateAr {
            run_lengths, order, ..
        } => add(
            bytes(mul(run_lengths.len(), *order)?)?,
            mul(run_lengths.len(), std::mem::size_of::<ArDiagnostics>())?,
        ),
    }
}
fn ar_state_values(run_lengths: &[usize], coefficients: &[Vec<f64>]) -> Result<usize> {
    if run_lengths.len() != coefficients.len() {
        return Err(SessionError::Invalid("AR runs and coefficients"));
    }
    coefficients.iter().try_fold(0usize, |total, phi| {
        add(total, add(mul(phi.len(), phi.len())?, mul(3, phi.len())?)?)
    })
}
fn resident_model_values(spec: &ModelSpec) -> Result<usize> {
    let (n, p) = spec.design.dim();
    // Cached ModelSpec design plus LinearModel design/inverse/covariance/row-space.
    let base = add(mul(3, mul(n, p)?)?, mul(2, mul(p, p)?)?)?;
    let ar = match &spec.noise {
        NoisePolicy::FixedAr {
            run_lengths,
            coefficients,
            ..
        } => ar_state_values(run_lengths, coefficients)?,
        // The estimated plan has this requested order for every run; retained
        // coefficients/startup matrices are bounded identically.
        NoisePolicy::EstimateAr {
            run_lengths, order, ..
        } => {
            let per_run = add(mul(*order, *order)?, mul(3, *order)?)?;
            mul(run_lengths.len(), per_run)?
        }
        NoisePolicy::Ols => 0,
    };
    add(base, ar)
}
fn resident_model_bytes(spec: &ModelSpec) -> Result<usize> {
    let diagnostic_bytes = match &spec.noise {
        NoisePolicy::EstimateAr { run_lengths, .. } => mul(
            mul(2, run_lengths.len())?,
            std::mem::size_of::<ArDiagnostics>(),
        )?,
        _ => 0,
    };
    add(bytes(resident_model_values(spec)?)?, diagnostic_bytes)
}
fn build_model(subject: &LoadedSubject, spec: &ModelSpec) -> Result<(PreparedModel, bool)> {
    match &spec.noise {
        NoisePolicy::Ols => Ok((
            PreparedModel::Ols(
                LinearModel::new(spec.design.view(), spec.tolerance)?.with_backend(spec.backend),
            ),
            false,
        )),
        NoisePolicy::FixedAr {
            run_lengths,
            coefficients,
            startup,
            resets,
        } => {
            let plan = ArPlan::new(run_lengths.clone(), coefficients.clone(), *startup)?
                .with_resets(resets)?;
            Ok((
                PreparedModel::Ar(
                    GlsModel::new(spec.design.view(), plan, spec.tolerance)?
                        .with_backend(spec.backend),
                ),
                false,
            ))
        }
        NoisePolicy::EstimateAr {
            run_lengths,
            noise_valid,
            order,
            correction_max_lag,
            pooling,
            startup,
            feature_indices,
        } => {
            if spec.tolerance.to_bits() != ESTIMATED_AR_TOLERANCE.to_bits() {
                return Err(SessionError::Invalid(
                    "estimated AR requires tolerance exactly 1e-12",
                ));
            }
            if feature_indices.is_empty()
                || feature_indices.iter().any(|&i| i >= subject.data.ncols())
            {
                return Err(SessionError::Invalid(
                    "estimated AR feature subset is empty or out of bounds",
                ));
            }
            let mut seen = std::collections::HashSet::new();
            if feature_indices.iter().any(|i| !seen.insert(*i)) {
                return Err(SessionError::Invalid(
                    "estimated AR feature subset contains duplicates",
                ));
            }
            let noise_data = subject.data.select(Axis(1), feature_indices);
            let options = ArOptions {
                run_lengths: run_lengths.clone(),
                noise_valid: noise_valid.clone(),
                order: *order,
                correction_max_lag: *correction_max_lag,
                pooling: *pooling,
                startup: *startup,
            };
            Ok((
                PreparedModel::Ar(
                    GlsModel::estimate_ar(
                        spec.design.view(),
                        noise_data.view(),
                        options,
                        ESTIMATED_AR_TOLERANCE,
                    )?
                    .with_backend(spec.backend),
                ),
                true,
            ))
        }
    }
}
fn output_preparation_bytes(
    n: usize,
    p: usize,
    q: usize,
    uncertainty: Uncertainty,
) -> Result<usize> {
    let weights = mul(q, p)?;
    let geometry = match uncertainty {
        Uncertainty::None => 0,
        Uncertainty::Marginal => q,
        Uncertainty::Joint => add(mul(q, n)?, mul(q, q)?)?,
    };
    bytes(add(add(weights, geometry)?, mul(4, p)?)?)
}
fn output_workspace_bytes(
    n: usize,
    p: usize,
    q: usize,
    capacity: usize,
    uncertainty: Uncertainty,
    transformed: bool,
) -> Result<usize> {
    if n == 0 || p == 0 || q == 0 || capacity == 0 {
        return Err(SessionError::Invalid("zero output workspace dimension"));
    }
    let mut rows = add(p, q)?;
    if uncertainty != Uncertainty::None {
        rows = add(rows, add(n, 1)?)?;
    }
    if transformed {
        rows = add(rows, n)?;
    }
    bytes(mul(rows, capacity)?)
}
fn output_retained_bytes(q: usize, features: usize, uncertainty: Uncertainty) -> Result<usize> {
    bytes(mul(
        q + usize::from(uncertainty != Uncertainty::None),
        features,
    )?)
}
fn receipt_numeric_bytes(receipt: &PreparationReceipt) -> Result<usize> {
    let coefficient_values = receipt
        .noise_coefficients
        .as_ref()
        .map_or(0usize, |runs| runs.iter().map(Vec::len).sum());
    let diagnostic_bytes = receipt
        .noise_diagnostics
        .as_ref()
        .map_or(0usize, |x| x.len() * std::mem::size_of::<ArDiagnostics>());
    add(bytes(coefficient_values)?, diagnostic_bytes)
}
fn run_ols(
    model: &LinearModel,
    data: ndarray::ArrayView2<'_, f64>,
    output: &OutputSpec,
    block: usize,
    sink: &mut impl for<'a, 'b> FnMut(Range<usize>, &SummaryView<'a, 'b>) -> Result<()>,
) -> Result<()> {
    let plan = model.prepare_outputs(output)?;
    let mut workspace = plan.workspace(block)?;
    for start in (0..data.ncols()).step_by(block) {
        let end = (start + block).min(data.ncols());
        let view = plan.run_into(data.slice(ndarray::s![.., start..end]), &mut workspace)?;
        sink(start..end, &view)?;
    }
    Ok(())
}
fn run_ar(
    model: &GlsModel<ArPlan>,
    data: ndarray::ArrayView2<'_, f64>,
    output: &OutputSpec,
    block: usize,
    sink: &mut impl for<'a, 'b> FnMut(Range<usize>, &SummaryView<'a, 'b>) -> Result<()>,
) -> Result<()> {
    let plan = model.prepare_outputs(output)?;
    let mut workspace = plan.workspace(block)?;
    for start in (0..data.ncols()).step_by(block) {
        let end = (start + block).min(data.ncols());
        let view = plan.run_into(data.slice(ndarray::s![.., start..end]), &mut workspace)?;
        sink(start..end, &view)?;
    }
    Ok(())
}

#[derive(Debug)]
pub struct SubjectOutput {
    subject_id: String,
    features: Arc<FeatureAxis>,
    blocks: Vec<(Range<usize>, Summary)>,
    receipt: PreparationReceipt,
}
impl SubjectOutput {
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }
    pub fn features(&self) -> &FeatureAxis {
        &self.features
    }
    pub fn blocks(&self) -> &[(Range<usize>, Summary)] {
        &self.blocks
    }
    pub fn receipt(&self) -> &PreparationReceipt {
        &self.receipt
    }
}
