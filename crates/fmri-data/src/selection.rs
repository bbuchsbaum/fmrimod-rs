use std::collections::HashSet;
use std::ops::Range;

use serde::Serialize;

use crate::errors::{invalid, FmriDatasetError, Result};

/// Zero-based source gather. Repetitions are legal; order is authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum Selection {
    All,
    Range(Range<usize>),
    Indices(Vec<usize>),
}

impl Selection {
    pub fn len(&self, total: usize) -> Result<usize> {
        match self {
            Self::All => Ok(total),
            Self::Range(r) if r.start <= r.end && r.end <= total => Ok(r.end - r.start),
            Self::Indices(ix) if ix.iter().all(|&i| i < total) => Ok(ix.len()),
            _ => Err(FmriDatasetError::InvalidIndex(format!(
                "selection outside 0..{total}"
            ))),
        }
    }

    pub(crate) fn at(&self, index: usize) -> usize {
        match self {
            Self::All => index,
            Self::Range(r) => r.start + index,
            Self::Indices(ix) => ix[index],
        }
    }
}

/// A gather validated against a particular pair of source dimensions.
#[derive(Debug, Clone, Serialize)]
pub struct ReadSelection {
    base_shape: [usize; 2],
    shape: [usize; 2],
    rows: Selection,
    columns: Selection,
}

impl ReadSelection {
    pub fn new(base_shape: [usize; 2], rows: Selection, columns: Selection) -> Result<Self> {
        let shape = [rows.len(base_shape[0])?, columns.len(base_shape[1])?];
        shape[0]
            .checked_mul(shape[1])
            .ok_or(FmriDatasetError::SizeOverflow)?;
        Ok(Self {
            base_shape,
            shape,
            rows,
            columns,
        })
    }

    pub fn all(shape: [usize; 2]) -> Result<Self> {
        Self::new(shape, Selection::All, Selection::All)
    }

    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn rows(&self) -> &Selection {
        &self.rows
    }
    pub fn columns(&self) -> &Selection {
        &self.columns
    }

    pub fn validate_for(&self, shape: [usize; 2]) -> Result<()> {
        if self.base_shape != shape {
            return Err(FmriDatasetError::ShapeMismatch {
                expected: shape,
                found: self.base_shape,
            });
        }
        Ok(())
    }
}

/// Frame subsets preserve axis uniqueness. Source gathers need not.
#[derive(Debug, Clone)]
pub struct AxisSelection {
    pub(crate) selection: Selection,
    extent: usize,
}

impl AxisSelection {
    pub fn new(selection: Selection, extent: usize) -> Result<Self> {
        selection.len(extent)?;
        if let Selection::Indices(ix) = &selection {
            let mut seen = HashSet::new();
            if !ix.iter().all(|i| seen.insert(*i)) {
                return Err(invalid("axis selection", "duplicate positions"));
            }
        }
        Ok(Self { selection, extent })
    }

    pub fn all(extent: usize) -> Self {
        Self {
            selection: Selection::All,
            extent,
        }
    }

    pub(crate) fn validate_for(&self, extent: usize) -> Result<usize> {
        if self.extent != extent {
            return Err(invalid("axis selection", "axis length changed"));
        }
        self.selection.len(extent)
    }
}
