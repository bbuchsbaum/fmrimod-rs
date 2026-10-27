//! Experimental design matrices, regressors, contrasts and diagnostics.
//!
//! [`events`] provides event construction; [`sampled_features`] builds sampled
//! regressors. The dataframe/formula frontend is available through [`formula`].
//! Enable the `formula` feature of `fmrimod` to use this component in a pipeline.

// TODO: Re-enable these lints once documentation coverage improves
// #![warn(missing_docs)]
// #![warn(missing_debug_implementations)]

pub mod afni;
pub mod baseline;
pub mod basis;
pub mod cache;
pub mod condition_basis_list;
pub mod contrast;
pub mod convolve;
pub mod convolve_cached;
pub mod convolve_design;
pub mod core;
pub mod covariate;
pub mod crossval;
pub mod design;
pub mod design_colmap;
pub mod error;
pub mod events;
pub mod formula;
pub mod hrf;
pub mod io;
pub mod naming;
pub mod nuisance;
pub mod regressors;
pub mod residualize;
pub mod sampled_features;
pub mod sampling;
pub mod trialwise;
pub mod types;
pub mod utils;
pub mod validation;

// DataFrame implementations
pub mod dataframe_polars;

// Re-export DataFrame (primary export)
pub use dataframe_polars::{DataFrame, DataFrameBuilder};

// Visualization module with conditional features
pub mod visualization {
    //! Visualization functions for fMRI design matrices.

    mod simple;
    pub use simple::{
        correlation_matrix_to_csv, correlation_summary, design_matrix_to_csv, design_summary,
        model_summary_to_txt,
    };

    // Conditionally compile the complex visualization modules
    #[cfg(feature = "plotters")]
    mod correlation_map;
    #[cfg(feature = "plotters")]
    mod design_map;

    // Export complex visualizations when available
    #[cfg(feature = "plotters")]
    pub use correlation_map::correlation_map;
    #[cfg(feature = "plotters")]
    pub use design_map::plot_model_summary;
    #[cfg(feature = "plotters")]
    pub use design_map::{design_map, plot_design_matrix, ColorMap, DesignMapConfig};
}

// Re-export commonly used items
pub mod prelude {
    pub use crate::basis::{IdentityBasis, PolynomialBasis, SplineBasis};
    pub use crate::contrast::{pairwise_contrasts, Contrast, ContrastBuilder};
    pub use crate::core::EventModel;
    pub use crate::design::DesignMatrix;
    pub use crate::events::{EventBasis, EventFactor, EventMatrix, EventTerm, EventVariable};
    pub use crate::formula::{Formula, FormulaBuilder};
    pub use crate::sampled_features::sampled_feature_design;
    pub use crate::sampling::uniform_sampling_frame;
    pub use crate::types::*;
    pub use crate::utils::validation::{uniform_durations, validate_durations, validate_onsets};

    // Re-export DataFrame
    pub use crate::{DataFrame, DataFrameBuilder};
}

pub use crate::core::EventModel;
pub use crate::types::{ContrastType, EventType, ValidationError};

// Secondary re-export handled by primary export above

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
