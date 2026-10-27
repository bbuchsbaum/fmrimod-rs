use std::collections::HashSet;

use serde::Serialize;

use crate::array::{reserved, MemoryBudget};
use crate::axis::{Axis, FeatureId};
use crate::errors::{invalid, FmriDatasetError, Result};
use crate::AxisSelection;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct IndexSpace {
    namespace: String,
    ids: Vec<FeatureId>,
}

impl IndexSpace {
    pub fn new(namespace: impl Into<String>, ids: Vec<FeatureId>) -> Result<Self> {
        let namespace = namespace.into();
        if namespace.is_empty() {
            return Err(invalid("index namespace", "must not be empty"));
        }
        Axis::new(ids.clone(), Vec::new())?;
        Ok(Self { namespace, ids })
    }
    pub fn namespace(&self) -> &str {
        &self.namespace
    }
}

/// Native volume geometry. Support uses zero-based x-fastest voxel positions:
/// x + nx * (y + ny * z). Affine maps zero-based voxel coordinates to world.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct VolumeSpace {
    dimensions: [usize; 3],
    affine: [[f64; 4]; 4],
    support: Vec<usize>,
    ids: Vec<FeatureId>,
    template: Option<String>,
}

impl VolumeSpace {
    pub fn new(
        dimensions: [usize; 3],
        affine: [[f64; 4]; 4],
        support: Option<Vec<usize>>,
        template: Option<String>,
    ) -> Result<Self> {
        let count = voxel_count(dimensions)?;
        if affine.iter().flatten().any(|x| !x.is_finite()) || affine[3] != [0.0, 0.0, 0.0, 1.0] {
            return Err(invalid(
                "affine",
                "requires finite homogeneous 4x4 transform",
            ));
        }
        // Scale each spatial row before the determinant check. Finite valid
        // transforms must not be rejected merely because their unit scale
        // makes the unscaled determinant overflow or underflow.
        let mut a = [[0.; 3]; 3];
        for (row, values) in a.iter_mut().enumerate() {
            let scale = affine[row][..3].iter().map(|v| v.abs()).fold(0., f64::max);
            if scale == 0. {
                return Err(invalid("affine", "spatial transform must be nonsingular"));
            }
            for (col, value) in values.iter_mut().enumerate() {
                *value = affine[row][col] / scale;
            }
        }
        let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
        if !det.is_finite() || det == 0.0 {
            return Err(invalid("affine", "spatial transform must be nonsingular"));
        }
        if template.as_ref().is_some_and(|t| t.is_empty()) {
            return Err(invalid("template", "must not be empty"));
        }
        let support = match support {
            Some(s) => s,
            None => {
                let mut s = reserved(count)?;
                s.extend(0..count);
                s
            }
        };
        let mut seen = HashSet::new();
        if !support.iter().all(|&v| v < count && seen.insert(v)) {
            return Err(invalid(
                "volume support",
                "duplicate or out-of-bounds voxel",
            ));
        }
        // IDs match R's one-based stable labels; numerical positions remain zero-based.
        let ids = support
            .iter()
            .map(|v| FeatureId::new(format!("voxel-{}", v + 1)))
            .collect::<Result<_>>()?;
        Ok(Self {
            dimensions,
            affine,
            support,
            ids,
            template,
        })
    }
    pub fn dimensions(&self) -> [usize; 3] {
        self.dimensions
    }
    pub fn affine(&self) -> &[[f64; 4]; 4] {
        &self.affine
    }
    pub fn support(&self) -> &[usize] {
        &self.support
    }
    pub fn template(&self) -> Option<&str> {
        self.template.as_deref()
    }

    /// Scatter a feature vector into x-fastest native storage. Unselected
    /// voxels receive the explicit fill value, never invented spatial geometry.
    pub fn reconstruct(&self, values: &[f64], fill: f64, budget: MemoryBudget) -> Result<Vec<f64>> {
        if values.len() != self.support.len() {
            return Err(FmriDatasetError::LengthMismatch {
                expected: self.support.len(),
                found: values.len(),
            });
        }
        let n = voxel_count(self.dimensions)?;
        budget.check(n.checked_mul(8).ok_or(FmriDatasetError::SizeOverflow)?)?;
        let mut out = reserved(n)?;
        out.resize(n, fill);
        for (&position, &value) in self.support.iter().zip(values) {
            out[position] = value;
        }
        Ok(out)
    }
}

fn voxel_count(dim: [usize; 3]) -> Result<usize> {
    if dim.contains(&0) {
        return Err(invalid("volume dimensions", "must be positive"));
    }
    dim.into_iter().try_fold(1usize, |n, d| {
        n.checked_mul(d).ok_or(FmriDatasetError::SizeOverflow)
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum FeatureSpace {
    Index(IndexSpace),
    Volume(VolumeSpace),
}

impl FeatureSpace {
    pub fn ids(&self) -> &[FeatureId] {
        match self {
            Self::Index(s) => &s.ids,
            Self::Volume(s) => &s.ids,
        }
    }
    /// Exact semantic comparison; no shape-only or implicit approximate alignment.
    /// Portable digest compatibility is deferred until canonical bytes are specified.
    pub fn compatible_with(&self, other: &Self) -> bool {
        self == other
    }
    pub fn assert_compatible(&self, other: &Self) -> Result<()> {
        if self.compatible_with(other) {
            Ok(())
        } else {
            Err(FmriDatasetError::SpaceMismatch)
        }
    }
    pub fn restrict(&self, selection: &AxisSelection) -> Result<Self> {
        let n = selection.validate_for(self.ids().len())?;
        let ids = (0..n)
            .map(|i| self.ids()[selection.selection.at(i)].clone())
            .collect();
        Ok(match self {
            Self::Index(s) => Self::Index(IndexSpace {
                namespace: s.namespace.clone(),
                ids,
            }),
            Self::Volume(s) => Self::Volume(VolumeSpace {
                dimensions: s.dimensions,
                affine: s.affine,
                template: s.template.clone(),
                ids,
                support: (0..n)
                    .map(|i| s.support[selection.selection.at(i)])
                    .collect(),
            }),
        })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct FeatureAxis {
    axis: Axis<FeatureId>,
    space: FeatureSpace,
}

impl FeatureAxis {
    pub fn new(axis: Axis<FeatureId>, space: FeatureSpace) -> Result<Self> {
        if axis.ids() != space.ids() {
            return Err(invalid("feature axis", "IDs and order must match space"));
        }
        Ok(Self { axis, space })
    }
    pub fn from_space(space: FeatureSpace) -> Result<Self> {
        Self::new(Axis::new(space.ids().to_vec(), Vec::new())?, space)
    }
    pub fn axis(&self) -> &Axis<FeatureId> {
        &self.axis
    }
    pub fn space(&self) -> &FeatureSpace {
        &self.space
    }
    pub fn subset(&self, selection: &AxisSelection) -> Result<Self> {
        Self::new(
            self.axis.subset(selection)?,
            self.space.restrict(selection)?,
        )
    }
}
