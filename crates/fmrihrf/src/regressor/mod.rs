//! Regressor functionality for fMRI event modeling.

pub mod convolution;
pub mod core;
pub mod design;
pub mod neural_input;

pub use self::convolution::{
    convolve_hrf, convolve_hrf_trial_varying, convolve_hrf_with_method, ConvolutionMethod,
};
pub use self::core::{null_regressor, regressor, regressor_set, Regressor, RegressorSet};
#[cfg(feature = "serde")]
pub use self::core::{RegressorData, RegressorSetData};
pub use self::design::{
    design_from_regressor, design_from_regressor_set, design_from_regressors,
    design_matrix_from_set, regressor_design, DesignMatrix, DesignOptions, DesignOptionsBuilder,
};
pub use self::neural_input::{neural_input, neural_input_core};
pub mod event_design;
pub use event_design::{ConditionEvents, EventTable};

pub mod sampled;
pub use sampled::{
    FeatureGrid, FeatureOptions, FeatureRun, FeatureScale, FeatureStatistics, SampledFeatures,
};
