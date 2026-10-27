//! Composable fMRI modelling. The default path accepts borrowed events and
//! matrices; enable `formula` for the dataframe/formula frontend.
pub use fmri_data as dataset;
#[cfg(feature = "formula")]
pub use fmri_design as design;
pub use fmri_group as group;
pub use fmri_hrf as hrf;
pub use fmri_noise as noise;
pub use fmri_reg as regression;

pub mod outputs;

pub mod cohort;
pub mod session;

#[cfg(feature = "nifti")]
pub mod ingress;

#[cfg(feature = "nifti")]
pub mod analysis;

pub mod group_requirements;
