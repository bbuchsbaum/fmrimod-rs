//! fmri_hrf - Rust implementation of fMRI Hemodynamic Response Function (HRF) and Regressor Tools
//!
//! This crate provides tools for working with hemodynamic response functions (HRFs) in
//! functional magnetic resonance imaging (fMRI) data analysis. It includes implementations
//! of various HRF models, convolution operations, and design matrix generation.
//!
//! # Example
//!
//! ```
//! use fmri_hrf::{hrf::spm_canonical, sampling::SamplingFrame, Hrf};
//! use ndarray::Array1;
//!
//! // Create a sampling frame for 100 scans with TR=2.0s
//! let sf = SamplingFrame::uniform(100, 2.0, 0.1).unwrap();
//!
//! // Create SPM canonical HRF
//! let hrf = spm_canonical(None, None, None, None, None, None);
//!
//! // Evaluate HRF at time points
//! let t = Array1::linspace(0.0, 20.0, 21);
//! let values = hrf.evaluate(&t).unwrap();
//! ```

#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod hrf;
pub mod penalty;
pub mod reconstruction;
pub mod regressor;
pub mod sampling;
pub mod types;
pub mod utils;

#[cfg(feature = "plotting")]
pub mod plotting;

// Re-export commonly used items
pub use error::{HrfError, Result};
pub use hrf::{Hrf, HrfExt};
pub use sampling::{SamplingFrame, SamplingFrameBuilder};
pub use types::{HrfValues, Matrix, TimeArray, Vector};

// Re-export splines functionality for B-spline basis functions
pub use fmri_splines::{BSpline, Spline, SplineError, SplineResult};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
