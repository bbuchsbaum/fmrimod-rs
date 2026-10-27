//! Spatially typed, annotated observation-by-feature data.
//!
//! Frames own validated semantic axes and source descriptions. Views preserve
//! alignment without reading assay values; [`ExecutionContext`] performs bounded
//! reads. The current foundation supports f32/f64 memory sources and index/volume
//! spaces. Its serde representation is internal and is not the FDS wire format.

pub mod array;
pub mod axis;
pub mod errors;
pub mod frame;
#[cfg(feature = "nifti")]
pub mod nifti;
pub mod sampling;
pub mod selection;
pub mod space;

pub use array::{
    ArrayReader, DType, ExecutionContext, MemoryBudget, NumericBlock, ReadCost, SourceSpec, Values,
};
pub use axis::{
    Axis, AxisBlock, Column, ComponentId, EntityId, FeatureId, Metadata, ObservationId,
};
pub use errors::{FmriDatasetError, Result};
pub use frame::{Assay, AssayAnnotations, FmriFrame};
#[cfg(feature = "nifti")]
pub use nifti::{NiftiReader, NiftiSource};
pub use selection::{AxisSelection, ReadSelection, Selection};
pub use space::{FeatureAxis, FeatureSpace, IndexSpace, VolumeSpace};
