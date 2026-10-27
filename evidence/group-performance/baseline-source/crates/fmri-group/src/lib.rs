//! Identified independent-subject group analysis over bounded feature blocks.
//!
//! One scalar first-level estimand per cohort. OLS estimates residual scale;
//! common-effect and DL meta-analysis treat supplied sampling variances as fixed
//! for inference, retaining their original provenance in the run receipt.
//! Repeated participants, joint contrasts, spatial resampling and mixed models
//! require separate APIs and are not silently approximated here.
mod cohort;
mod fit;
mod multiplicity;
pub use cohort::*;
pub use fit::*;
pub use multiplicity::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid group input: {0}")]
    Invalid(&'static str),
    #[error("group working buffer budget exceeded")]
    Budget,
    #[error(transparent)]
    Dataset(#[from] fmridataset::FmriDatasetError),
    #[error(transparent)]
    Fit(#[from] fmrireg::FitError),
}
pub type Result<T> = std::result::Result<T, Error>;
fn nonblank(s: &str) -> bool {
    !s.trim().is_empty()
}
