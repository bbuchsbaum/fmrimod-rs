use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use serde::Serialize;

use crate::array::SourceSpec;
use crate::errors::{invalid, FmriDatasetError, Result};
use crate::{AxisSelection, ReadSelection, Selection};

pub type Metadata = BTreeMap<String, serde_json::Value>;

/// Marker shared by distinct stable ID types. Construction validates contents.
pub trait AxisId: Clone + Eq + Hash + Debug {
    fn as_str(&self) -> &str;
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
        #[serde(transparent)]
        pub struct $name(String);
        impl $name {
            pub fn new(value: impl Into<String>) -> Result<Self> {
                let value = value.into();
                if value.is_empty() {
                    return Err(invalid("ID", "must not be empty"));
                }
                Ok(Self(value))
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }
        impl AxisId for $name {
            fn as_str(&self) -> &str {
                &self.0
            }
        }
    };
}

id_type!(ObservationId);
id_type!(FeatureId);
id_type!(ComponentId);
id_type!(EntityId);

/// Nullable scalar annotations. Categories retain unused levels and ordering.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum Column {
    Bool(Vec<Option<bool>>),
    Int(Vec<Option<i64>>),
    #[serde(rename = "float64_bits")]
    Float(#[serde(serialize_with = "serialize_nullable_f64")] Vec<Option<f64>>),
    Text(Vec<Option<String>>),
    Categorical {
        codes: Vec<Option<u32>>,
        levels: Vec<String>,
        ordered: bool,
    },
}

fn serialize_nullable_f64<S: serde::Serializer>(
    values: &[Option<f64>],
    serializer: S,
) -> std::result::Result<S::Ok, S::Error> {
    serializer.collect_seq(values.iter().map(|v| v.map(f64::to_bits)))
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Bool(x) => x.len(),
            Self::Int(x) => x.len(),
            Self::Float(x) => x.len(),
            Self::Text(x) => x.len(),
            Self::Categorical { codes, .. } => codes.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn validate(&self, length: usize) -> Result<()> {
        if self.len() != length {
            return Err(FmriDatasetError::LengthMismatch {
                expected: length,
                found: self.len(),
            });
        }
        if let Self::Categorical { codes, levels, .. } = self {
            let mut seen = HashSet::new();
            if !levels.iter().all(|l| seen.insert(l))
                || codes.iter().flatten().any(|&c| c as usize >= levels.len())
            {
                return Err(invalid(
                    "categorical column",
                    "duplicate levels or out-of-range codes",
                ));
            }
        }
        Ok(())
    }
    fn subset(&self, selection: &AxisSelection, n: usize) -> Self {
        let s = &selection.selection;
        match self {
            Self::Bool(x) => Self::Bool((0..n).map(|i| x[s.at(i)]).collect()),
            Self::Int(x) => Self::Int((0..n).map(|i| x[s.at(i)]).collect()),
            Self::Float(x) => Self::Float((0..n).map(|i| x[s.at(i)]).collect()),
            Self::Text(x) => Self::Text((0..n).map(|i| x[s.at(i)].clone()).collect()),
            Self::Categorical {
                codes,
                levels,
                ordered,
            } => Self::Categorical {
                codes: (0..n).map(|i| codes[s.at(i)]).collect(),
                levels: levels.clone(),
                ordered: *ordered,
            },
        }
    }
}

/// Axis-by-component multivariate annotations, read lazily like assays.
#[derive(Debug, Clone, Serialize)]
pub struct AxisBlock {
    source: SourceSpec,
    components: Axis<ComponentId>,
    role: String,
    units: Option<String>,
    metadata: Metadata,
}

impl AxisBlock {
    pub fn new(
        source: SourceSpec,
        components: Axis<ComponentId>,
        role: impl Into<String>,
        units: Option<String>,
        metadata: Metadata,
    ) -> Result<Self> {
        let role = role.into();
        if role.is_empty() {
            return Err(invalid("block role", "must not be empty"));
        }
        if source.shape()[1] != components.len() {
            return Err(FmriDatasetError::LengthMismatch {
                expected: components.len(),
                found: source.shape()[1],
            });
        }
        Ok(Self {
            source,
            components,
            role,
            units,
            metadata,
        })
    }
    pub fn source(&self) -> &SourceSpec {
        &self.source
    }
    pub fn components(&self) -> &Axis<ComponentId> {
        &self.components
    }
    pub fn role(&self) -> &str {
        &self.role
    }
    pub fn units(&self) -> Option<&str> {
        self.units.as_deref()
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Axis<I: AxisId> {
    ids: Vec<I>,
    columns: Vec<(String, Column)>,
    blocks: Vec<(String, AxisBlock)>,
    metadata: Metadata,
}

impl<I: AxisId> Axis<I> {
    pub fn new(ids: Vec<I>, columns: Vec<(String, Column)>) -> Result<Self> {
        let mut seen = HashSet::new();
        if !ids
            .iter()
            .all(|id| !id.as_str().is_empty() && seen.insert(id))
        {
            return Err(invalid("axis IDs", "must be unique and nonempty"));
        }
        validate_names(columns.iter().map(|(name, _)| name.as_str()))?;
        for (_, column) in &columns {
            column.validate(ids.len())?;
        }
        Ok(Self {
            ids,
            columns,
            blocks: Vec::new(),
            metadata: Metadata::new(),
        })
    }
    pub fn with_blocks(mut self, blocks: Vec<(String, AxisBlock)>) -> Result<Self> {
        validate_names(blocks.iter().map(|(name, _)| name.as_str()))?;
        for (_, block) in &blocks {
            if block.source.shape()[0] != self.len() {
                return Err(FmriDatasetError::LengthMismatch {
                    expected: self.len(),
                    found: block.source.shape()[0],
                });
            }
        }
        self.blocks = blocks;
        Ok(self)
    }
    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }
    pub fn len(&self) -> usize {
        self.ids.len()
    }
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
    pub fn ids(&self) -> &[I] {
        &self.ids
    }
    pub fn columns(&self) -> &[(String, Column)] {
        &self.columns
    }
    pub fn blocks(&self) -> &[(String, AxisBlock)] {
        &self.blocks
    }
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }
    pub fn select_ids(&self, ids: &[I]) -> Result<AxisSelection> {
        let positions: std::collections::HashMap<_, _> =
            self.ids.iter().enumerate().map(|(i, id)| (id, i)).collect();
        let indices = ids
            .iter()
            .map(|id| {
                positions
                    .get(id)
                    .copied()
                    .ok_or_else(|| FmriDatasetError::UnknownId {
                        kind: "axis ID",
                        id: id.as_str().into(),
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        AxisSelection::new(Selection::Indices(indices), self.len())
    }
    pub fn subset(&self, selection: &AxisSelection) -> Result<Self> {
        let n = selection.validate_for(self.len())?;
        let ids = (0..n)
            .map(|i| self.ids[selection.selection.at(i)].clone())
            .collect();
        let columns = self
            .columns
            .iter()
            .map(|(name, col)| (name.clone(), col.subset(selection, n)))
            .collect();
        let blocks = self
            .blocks
            .iter()
            .map(|(name, block)| {
                let mut block = block.clone();
                block.source = block.source.view(ReadSelection::new(
                    block.source.shape(),
                    selection.selection.clone(),
                    Selection::All,
                )?)?;
                Ok((name.clone(), block))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            ids,
            columns,
            blocks,
            metadata: self.metadata.clone(),
        })
    }
}

pub(crate) fn validate_names<'a>(names: impl Iterator<Item = &'a str>) -> Result<()> {
    let mut seen = HashSet::new();
    for name in names {
        if name.is_empty() || !seen.insert(name) {
            return Err(invalid("names", "must be unique and nonempty"));
        }
    }
    Ok(())
}
