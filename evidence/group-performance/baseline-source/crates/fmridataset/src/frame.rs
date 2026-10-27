use serde::Serialize;

use crate::axis::{validate_names, Axis, FeatureId, Metadata, ObservationId};
use crate::errors::{invalid, FmriDatasetError, Result};
use crate::{
    AxisSelection, ExecutionContext, FeatureAxis, NumericBlock, ReadSelection, SourceSpec,
};

#[derive(Debug, Clone, Default, Serialize)]
pub struct AssayAnnotations {
    pub role: Option<String>,
    pub units: Option<String>,
    pub metadata: Metadata,
}

/// Unbound assay input. Ordered axis IDs make attachment explicit even for a
/// source whose storage format has no axis labels.
#[derive(Debug, Clone, Serialize)]
pub struct Assay {
    name: String,
    source: SourceSpec,
    observation_ids: Vec<ObservationId>,
    feature_ids: Vec<FeatureId>,
    annotations: AssayAnnotations,
}

impl Assay {
    pub fn new(
        name: impl Into<String>,
        source: SourceSpec,
        observation_ids: Vec<ObservationId>,
        feature_ids: Vec<FeatureId>,
        annotations: AssayAnnotations,
    ) -> Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(invalid("assay name", "must not be empty"));
        }
        Axis::new(observation_ids.clone(), Vec::new())?;
        Axis::new(feature_ids.clone(), Vec::new())?;
        let expected = [observation_ids.len(), feature_ids.len()];
        if source.shape() != expected {
            return Err(FmriDatasetError::ShapeMismatch {
                expected,
                found: source.shape(),
            });
        }
        Ok(Self {
            name,
            source,
            observation_ids,
            feature_ids,
            annotations,
        })
    }
    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn source(&self) -> &SourceSpec {
        &self.source
    }
    pub fn observation_ids(&self) -> &[ObservationId] {
        &self.observation_ids
    }
    pub fn feature_ids(&self) -> &[FeatureId] {
        &self.feature_ids
    }
    pub fn annotations(&self) -> &AssayAnnotations {
        &self.annotations
    }
}

/// Immutable, validated observation-by-feature semantics. Clone shares assay
/// storage. Derives Serialize only: deserialization must pass future validators.
#[derive(Debug, Clone, Serialize)]
pub struct FmriFrame {
    observations: Axis<ObservationId>,
    features: FeatureAxis,
    assays: Vec<Assay>,
    active_assay: String,
    metadata: Metadata,
}

impl FmriFrame {
    pub fn new(
        observations: Axis<ObservationId>,
        features: FeatureAxis,
        assays: Vec<Assay>,
        active_assay: impl Into<String>,
        metadata: Metadata,
    ) -> Result<Self> {
        if assays.is_empty() {
            return Err(invalid("assays", "at least one assay is required"));
        }
        validate_names(assays.iter().map(|a| a.name.as_str()))?;
        for assay in &assays {
            if assay.observation_ids != observations.ids()
                || assay.feature_ids != features.axis().ids()
            {
                return Err(invalid(
                    "assay alignment",
                    format!("assay '{}' has different IDs or order", assay.name),
                ));
            }
        }
        let active_assay = active_assay.into();
        if !assays.iter().any(|a| a.name == active_assay) {
            return Err(FmriDatasetError::UnknownId {
                kind: "assay",
                id: active_assay,
            });
        }
        Ok(Self {
            observations,
            features,
            assays,
            active_assay,
            metadata,
        })
    }

    /// Attach a single unlabeled source by explicitly declaring its axes.
    pub fn from_source(
        name: impl Into<String>,
        source: SourceSpec,
        observations: Axis<ObservationId>,
        features: FeatureAxis,
    ) -> Result<Self> {
        let name = name.into();
        let assay = Assay::new(
            name.clone(),
            source,
            observations.ids().to_vec(),
            features.axis().ids().to_vec(),
            AssayAnnotations::default(),
        )?;
        Self::new(observations, features, vec![assay], name, Metadata::new())
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.observations.len(), self.features.axis().len()]
    }
    pub fn observations(&self) -> &Axis<ObservationId> {
        &self.observations
    }
    pub fn features(&self) -> &FeatureAxis {
        &self.features
    }
    pub fn assays(&self) -> &[Assay] {
        &self.assays
    }
    pub fn active_assay(&self) -> &str {
        &self.active_assay
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn assay(&self, name: &str) -> Result<&Assay> {
        self.assays
            .iter()
            .find(|a| a.name == name)
            .ok_or_else(|| FmriDatasetError::UnknownId {
                kind: "assay",
                id: name.into(),
            })
    }
    pub fn collect(&self, name: &str, context: &ExecutionContext) -> Result<NumericBlock> {
        context.collect(self.assay(name)?.source())
    }

    /// A view is itself a valid frame with selected source descriptors. Axis
    /// annotations are subset eagerly; assay and axis-block values stay lazy.
    pub fn view(&self, rows: &AxisSelection, columns: &AxisSelection) -> Result<Self> {
        let observations = self.observations.subset(rows)?;
        let features = self.features.subset(columns)?;
        let selection = ReadSelection::new(
            self.shape(),
            rows.selection.clone(),
            columns.selection.clone(),
        )?;
        let assays = self
            .assays
            .iter()
            .map(|assay| {
                Assay::new(
                    assay.name.clone(),
                    assay.source.view(selection.clone())?,
                    observations.ids().to_vec(),
                    features.axis().ids().to_vec(),
                    assay.annotations.clone(),
                )
            })
            .collect::<Result<_>>()?;
        Self::new(
            observations,
            features,
            assays,
            self.active_assay.clone(),
            self.metadata.clone(),
        )
    }
    pub fn view_ids(&self, observations: &[ObservationId], features: &[FeatureId]) -> Result<Self> {
        self.view(
            &self.observations.select_ids(observations)?,
            &self.features.axis().select_ids(features)?,
        )
    }
}
