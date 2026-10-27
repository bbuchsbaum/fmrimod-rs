//! Compile named consumer requirements into one first-level output request.
//!
//! This module intentionally performs only metadata and contrast validation.  It
//! does not read data, choose a run aggregation, or infer equivalence from a
//! display label.  Callers bind `model_id` to the actual design matrix and its
//! numeric rank policy.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use ndarray::Array2;

use fmri_reg::output::{EstimandSet, OutputSpec, Uncertainty};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScientificSemantics {
    id: String,
    label: String,
    unit: String,
    coding: String,
    normalization: String,
    run_aggregation: String,
    semantic_version: String,
}

impl ScientificSemantics {
    pub fn new(
        id: impl Into<String>,
        label: impl Into<String>,
        unit: impl Into<String>,
        coding: impl Into<String>,
        normalization: impl Into<String>,
        run_aggregation: impl Into<String>,
        semantic_version: impl Into<String>,
    ) -> Result<Self, OutputRequirementError> {
        let value = Self {
            id: id.into(),
            label: label.into(),
            unit: unit.into(),
            coding: coding.into(),
            normalization: normalization.into(),
            run_aggregation: run_aggregation.into(),
            semantic_version: semantic_version.into(),
        };
        value.validate()?;
        Ok(value)
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn label(&self) -> &str {
        &self.label
    }
    pub fn unit(&self) -> &str {
        &self.unit
    }
    pub fn coding(&self) -> &str {
        &self.coding
    }
    pub fn normalization(&self) -> &str {
        &self.normalization
    }
    pub fn run_aggregation(&self) -> &str {
        &self.run_aggregation
    }
    pub fn semantic_version(&self) -> &str {
        &self.semantic_version
    }

    fn validate(&self) -> Result<(), OutputRequirementError> {
        for (field, value) in [
            ("estimand id", &self.id),
            ("estimand label", &self.label),
            ("unit", &self.unit),
            ("coding", &self.coding),
            ("normalization", &self.normalization),
            ("run aggregation", &self.run_aggregation),
            ("semantic version", &self.semantic_version),
        ] {
            require_nonempty(field, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DesignColumn {
    id: String,
    unit: String,
    coding: String,
}

impl DesignColumn {
    pub fn new(
        id: impl Into<String>,
        unit: impl Into<String>,
        coding: impl Into<String>,
    ) -> Result<Self, OutputRequirementError> {
        let value = Self {
            id: id.into(),
            unit: unit.into(),
            coding: coding.into(),
        };
        require_nonempty("design column id", &value.id)?;
        require_nonempty("design column unit", &value.unit)?;
        require_nonempty("design column coding", &value.coding)?;
        Ok(value)
    }

    pub fn id(&self) -> &str {
        &self.id
    }
    pub fn unit(&self) -> &str {
        &self.unit
    }
    pub fn coding(&self) -> &str {
        &self.coding
    }
}

#[derive(Clone, Debug)]
pub struct DesignColumns {
    columns: Vec<DesignColumn>,
}

impl DesignColumns {
    pub fn new(columns: Vec<DesignColumn>) -> Result<Self, OutputRequirementError> {
        let mut ids = BTreeSet::new();
        for column in &columns {
            if !ids.insert(column.id.clone()) {
                return Err(OutputRequirementError::DuplicateDesignColumn(
                    column.id.clone(),
                ));
            }
        }
        Ok(Self { columns })
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
    pub fn columns(&self) -> &[DesignColumn] {
        &self.columns
    }
    fn index_of(&self, id: &str) -> Option<usize> {
        self.columns.iter().position(|column| column.id == id)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NamedWeight {
    column_id: String,
    weight: f64,
}

impl NamedWeight {
    pub fn new(column_id: impl Into<String>, weight: f64) -> Result<Self, OutputRequirementError> {
        let value = Self {
            column_id: column_id.into(),
            weight,
        };
        require_nonempty("weight column id", &value.column_id)?;
        if !value.weight.is_finite() {
            return Err(OutputRequirementError::NonFiniteWeight(value.column_id));
        }
        Ok(value)
    }
    pub fn column_id(&self) -> &str {
        &self.column_id
    }
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct NamedLinearEstimand {
    semantics: ScientificSemantics,
    weights: Vec<NamedWeight>,
}

impl NamedLinearEstimand {
    pub fn new(
        semantics: ScientificSemantics,
        weights: Vec<NamedWeight>,
    ) -> Result<Self, OutputRequirementError> {
        let mut names = BTreeSet::new();
        for weight in &weights {
            if !names.insert(weight.column_id.clone()) {
                return Err(OutputRequirementError::DuplicateWeightColumn(
                    weight.column_id.clone(),
                ));
            }
        }
        Ok(Self { semantics, weights })
    }
    pub fn semantics(&self) -> &ScientificSemantics {
        &self.semantics
    }
    pub fn weights(&self) -> &[NamedWeight] {
        &self.weights
    }
}

#[derive(Clone, Debug)]
pub struct ConsumerRequirement {
    consumer_id: String,
    subject_id: String,
    model_id: String,
    estimands: Vec<NamedLinearEstimand>,
    uncertainty: Uncertainty,
}

impl ConsumerRequirement {
    pub fn new(
        consumer_id: impl Into<String>,
        subject_id: impl Into<String>,
        model_id: impl Into<String>,
        estimands: Vec<NamedLinearEstimand>,
        uncertainty: Uncertainty,
    ) -> Result<Self, OutputRequirementError> {
        let value = Self {
            consumer_id: consumer_id.into(),
            subject_id: subject_id.into(),
            model_id: model_id.into(),
            estimands,
            uncertainty,
        };
        require_nonempty("consumer id", &value.consumer_id)?;
        require_nonempty("subject id", &value.subject_id)?;
        require_nonempty("model id", &value.model_id)?;
        let mut ids = BTreeSet::new();
        for estimand in &value.estimands {
            if !ids.insert(estimand.semantics.id.clone()) {
                return Err(OutputRequirementError::DuplicateEstimandInConsumer {
                    consumer_id: value.consumer_id.clone(),
                    estimand_id: estimand.semantics.id.clone(),
                });
            }
        }
        Ok(value)
    }
    pub fn consumer_id(&self) -> &str {
        &self.consumer_id
    }
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }
    pub fn model_id(&self) -> &str {
        &self.model_id
    }
    pub fn estimands(&self) -> &[NamedLinearEstimand] {
        &self.estimands
    }
    pub fn uncertainty(&self) -> Uncertainty {
        self.uncertainty
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedEstimand {
    semantics: ScientificSemantics,
    weights: Vec<f64>,
}

impl ResolvedEstimand {
    /// Constructs a binding for a previously resolved design, validating that
    /// the supplied weights are numeric.  Normal compilation uses named
    /// weights and does not need this constructor.
    pub fn for_reuse(
        semantics: ScientificSemantics,
        weights: Vec<f64>,
    ) -> Result<Self, OutputRequirementError> {
        if weights.iter().any(|weight| !weight.is_finite()) {
            return Err(OutputRequirementError::NonFiniteResolvedWeight);
        }
        Ok(Self { semantics, weights })
    }
    pub fn semantics(&self) -> &ScientificSemantics {
        &self.semantics
    }
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerOutputMap {
    consumer_id: String,
    output_indices: Vec<usize>,
}

impl ConsumerOutputMap {
    pub fn consumer_id(&self) -> &str {
        &self.consumer_id
    }
    /// Indices are in the exact order requested by this consumer.
    pub fn output_indices(&self) -> &[usize] {
        &self.output_indices
    }
}

#[derive(Clone, Debug)]
pub struct CompiledOutput {
    output: OutputSpec,
    resolved_estimands: Vec<ResolvedEstimand>,
    consumer_maps: Vec<ConsumerOutputMap>,
}

impl CompiledOutput {
    pub fn output(&self) -> &OutputSpec {
        &self.output
    }
    pub fn resolved_estimands(&self) -> &[ResolvedEstimand] {
        &self.resolved_estimands
    }
    pub fn consumer_maps(&self) -> &[ConsumerOutputMap] {
        &self.consumer_maps
    }
}

/// Resolves names before any data read and compiles requirements for exactly one
/// subject/model.  A label alone is never a join key.
pub fn compile_requirements(
    design: &DesignColumns,
    requirements: &[ConsumerRequirement],
) -> Result<CompiledOutput, OutputRequirementError> {
    let Some(first) = requirements.first() else {
        return Err(OutputRequirementError::EmptyRequirements);
    };
    let mut consumer_ids = BTreeSet::new();
    let mut unique = Vec::<ResolvedEstimand>::new();
    let mut by_id = BTreeMap::<String, usize>::new();
    let mut consumer_maps = Vec::with_capacity(requirements.len());
    let mut uncertainty = first.uncertainty;

    for requirement in requirements {
        if !consumer_ids.insert(requirement.consumer_id.clone()) {
            return Err(OutputRequirementError::DuplicateConsumer(
                requirement.consumer_id.clone(),
            ));
        }
        if requirement.subject_id != first.subject_id {
            return Err(OutputRequirementError::SubjectMismatch {
                expected: first.subject_id.clone(),
                found: requirement.subject_id.clone(),
            });
        }
        if requirement.model_id != first.model_id {
            return Err(OutputRequirementError::ModelMismatch {
                expected: first.model_id.clone(),
                found: requirement.model_id.clone(),
            });
        }
        uncertainty = max_uncertainty(uncertainty, requirement.uncertainty);
        let mut requested = Vec::with_capacity(requirement.estimands.len());
        for named in &requirement.estimands {
            let resolved = resolve(design, named)?;
            let id = resolved.semantics.id.clone();
            let index = if let Some(&index) = by_id.get(&id) {
                if unique[index] != resolved {
                    return Err(OutputRequirementError::ConflictingEstimand(id));
                }
                index
            } else {
                let index = unique.len();
                unique.push(resolved);
                by_id.insert(id, index);
                index
            };
            requested.push(index);
        }
        consumer_maps.push(ConsumerOutputMap {
            consumer_id: requirement.consumer_id.clone(),
            output_indices: requested,
        });
    }

    let ids = unique
        .iter()
        .map(|estimand| estimand.semantics.id.clone())
        .collect();
    let ncols = design.len();
    let values = unique
        .iter()
        .flat_map(|estimand| estimand.weights.iter().copied())
        .collect();
    let matrix = Array2::from_shape_vec((unique.len(), ncols), values)
        .map_err(|_| OutputRequirementError::InternalMatrixShape)?;
    let estimands = EstimandSet::contrasts(ids, matrix)
        .map_err(|error| OutputRequirementError::Core(error.to_string()))?;
    Ok(CompiledOutput {
        output: OutputSpec::new(estimands).with_uncertainty(uncertainty),
        resolved_estimands: unique,
        consumer_maps,
    })
}

fn resolve(
    design: &DesignColumns,
    named: &NamedLinearEstimand,
) -> Result<ResolvedEstimand, OutputRequirementError> {
    let mut weights = vec![0.0; design.len()];
    for weight in &named.weights {
        let index = design
            .index_of(&weight.column_id)
            .ok_or_else(|| OutputRequirementError::UnknownDesignColumn(weight.column_id.clone()))?;
        weights[index] = weight.weight;
    }
    Ok(ResolvedEstimand {
        semantics: named.semantics.clone(),
        weights,
    })
}

fn max_uncertainty(left: Uncertainty, right: Uncertainty) -> Uncertainty {
    if uncertainty_rank(left) >= uncertainty_rank(right) {
        left
    } else {
        right
    }
}
fn uncertainty_rank(value: Uncertainty) -> u8 {
    match value {
        Uncertainty::None => 0,
        Uncertainty::Marginal => 1,
        Uncertainty::Joint => 2,
    }
}

/// Immutable caller assertions used to decide whether an owned result can be
/// reused.  These are identifiers, not inferred fingerprints: paths and Rust
/// types are deliberately insufficient evidence of equal source content.
#[derive(Clone, Debug, PartialEq)]
pub struct ReuseIdentity {
    source_version: String,
    data_version: String,
    original_row_order: String,
    selected_row_order: String,
    feature_spatial_identity: String,
    design_spec_and_rank_policy: String,
    whitening: WhiteningIdentity,
    backend_numeric_provenance: String,
    output_binding: OutputBinding,
}

/// Exact requested estimand semantics and resolved design-column weights.
/// This is deliberately structural rather than a caller-invented label.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputBinding {
    estimands: Vec<ResolvedEstimand>,
}

impl OutputBinding {
    pub fn new(estimands: Vec<ResolvedEstimand>) -> Result<Self, OutputRequirementError> {
        let mut ids = BTreeSet::new();
        for estimand in &estimands {
            if !ids.insert(estimand.semantics.id.clone()) {
                return Err(OutputRequirementError::DuplicateBoundEstimand(
                    estimand.semantics.id.clone(),
                ));
            }
        }
        Ok(Self { estimands })
    }

    pub fn estimands(&self) -> &[ResolvedEstimand] {
        &self.estimands
    }
}

impl ReuseIdentity {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        source_version: impl Into<String>,
        data_version: impl Into<String>,
        original_row_order: impl Into<String>,
        selected_row_order: impl Into<String>,
        feature_spatial_identity: impl Into<String>,
        design_spec_and_rank_policy: impl Into<String>,
        whitening: WhiteningIdentity,
        backend_numeric_provenance: impl Into<String>,
        output_binding: OutputBinding,
    ) -> Result<Self, OutputRequirementError> {
        let value = Self {
            source_version: source_version.into(),
            data_version: data_version.into(),
            original_row_order: original_row_order.into(),
            selected_row_order: selected_row_order.into(),
            feature_spatial_identity: feature_spatial_identity.into(),
            design_spec_and_rank_policy: design_spec_and_rank_policy.into(),
            whitening,
            backend_numeric_provenance: backend_numeric_provenance.into(),
            output_binding,
        };
        for (field, text) in [
            ("source version", &value.source_version),
            ("data version", &value.data_version),
            ("original row order", &value.original_row_order),
            ("selected row order", &value.selected_row_order),
            ("feature/spatial identity", &value.feature_spatial_identity),
            (
                "design spec/rank policy",
                &value.design_spec_and_rank_policy,
            ),
            (
                "backend numeric provenance",
                &value.backend_numeric_provenance,
            ),
        ] {
            require_nonempty(field, text)?;
        }
        value.whitening.validate()?;
        Ok(value)
    }
    pub fn cache_eligibility(&self) -> CacheEligibility {
        self.whitening.cache_eligibility()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum WhiteningIdentity {
    BuiltIn { version: String },
    CustomVersioned { deterministic_version: String },
    SessionOnly,
}
impl WhiteningIdentity {
    fn validate(&self) -> Result<(), OutputRequirementError> {
        match self {
            Self::BuiltIn { version } => require_nonempty("built-in whitening version", version),
            Self::CustomVersioned {
                deterministic_version,
            } => require_nonempty(
                "custom whitening deterministic version",
                deterministic_version,
            ),
            Self::SessionOnly => Ok(()),
        }
    }
    pub fn cache_eligibility(&self) -> CacheEligibility {
        match self {
            Self::BuiltIn { .. } => CacheEligibility::BuiltIn,
            Self::CustomVersioned { .. } => CacheEligibility::CustomVersioned,
            Self::SessionOnly => CacheEligibility::SessionOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheEligibility {
    BuiltIn,
    CustomVersioned,
    SessionOnly,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutputCapability {
    identity: ReuseIdentity,
    uncertainty: Uncertainty,
}
impl OutputCapability {
    pub fn new(identity: ReuseIdentity, uncertainty: Uncertainty) -> Self {
        Self {
            identity,
            uncertainty,
        }
    }
    /// Capability matching is pure: it neither opens a store nor derives a new
    /// standard error.  A marginal result cannot satisfy a joint request.
    /// Session-only whitening has no supplied session token, so it is never
    /// eligible for reuse across this comparison boundary.
    pub fn can_satisfy(&self, identity: &ReuseIdentity, uncertainty: Uncertainty) -> bool {
        self.identity.cache_eligibility() != CacheEligibility::SessionOnly
            && identity.cache_eligibility() != CacheEligibility::SessionOnly
            && self.identity == *identity
            && uncertainty_rank(self.uncertainty) >= uncertainty_rank(uncertainty)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OutputRequirementError {
    EmptyField(&'static str),
    DuplicateDesignColumn(String),
    DuplicateWeightColumn(String),
    NonFiniteWeight(String),
    DuplicateConsumer(String),
    DuplicateEstimandInConsumer {
        consumer_id: String,
        estimand_id: String,
    },
    UnknownDesignColumn(String),
    SubjectMismatch {
        expected: String,
        found: String,
    },
    ModelMismatch {
        expected: String,
        found: String,
    },
    ConflictingEstimand(String),
    EmptyRequirements,
    DuplicateBoundEstimand(String),
    NonFiniteResolvedWeight,
    InternalMatrixShape,
    Core(String),
}
impl fmt::Display for OutputRequirementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for OutputRequirementError {}

fn require_nonempty(field: &'static str, value: &str) -> Result<(), OutputRequirementError> {
    if value.trim().is_empty() {
        Err(OutputRequirementError::EmptyField(field))
    } else {
        Ok(())
    }
}
