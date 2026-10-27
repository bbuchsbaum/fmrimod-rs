//! Identified independent-subject group analysis over bounded feature blocks.
//!
//! One scalar first-level estimand per cohort. OLS estimates residual scale;
//! common-effect, DL and PM meta-regression treat supplied sampling variances as
//! fixed for inference, retaining their original provenance in the run receipt.
//! Weighted regression supports named group contrasts and their joint covariance.
//! Repeated participants, multiple correlated first-level estimands per subject,
//! spatial resampling and mixed models require separate APIs.
mod cohort;
mod fit;
mod meta_regression;
mod multiplicity;
mod portable;
pub use cohort::*;
pub use fit::*;
pub use meta_regression::*;
pub use multiplicity::*;
pub use portable::*;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid group input: {0}")]
    Invalid(&'static str),
    #[error("group working buffer budget exceeded")]
    Budget,
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Dataset(#[from] fmridataset::FmriDatasetError),
    #[error(transparent)]
    Fit(#[from] fmrireg::FitError),
}
pub type Result<T> = std::result::Result<T, Error>;
fn nonblank(s: &str) -> bool {
    !s.trim().is_empty()
}
