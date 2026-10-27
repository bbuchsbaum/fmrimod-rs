//! Composable fMRI modelling. The default path accepts borrowed events and
//! matrices; enable `formula` for the dataframe/formula frontend.
pub use fmri_group as group;
pub use fmri_noise as noise;
pub use fmridataset as dataset;
#[cfg(feature = "formula")]
pub use fmridesign as design;
pub use fmrihrf as hrf;
pub use fmrireg as regression;

pub mod outputs;

pub mod cohort;
pub mod session;

#[cfg(feature = "nifti")]
pub mod ingress;
