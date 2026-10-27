//! # Splines
//!
//! A fast, safe Rust implementation of the R splines package functionality.
//!
//! This crate provides functions and data structures for creating and working with:
//! - B-splines (`BSpline`)
//! - Natural cubic splines (`NaturalSpline`)
//! - Interpolation splines (`InterpolationSpline`)
//!
//! ## Examples
//!
//! ```rust
//! use fmri_splines::{BSpline, Spline};
//!
//! // Create a B-spline basis
//! let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let spline = BSpline::new(&x, None, None, 3, false, None)?;
//!
//! // Predict at new points
//! let new_x = vec![1.5, 2.5, 3.5];
//! let predictions = spline.predict(&new_x)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![doc = include_str!("../README.md")]
#![deny(missing_docs)]
#![deny(unsafe_code)]

// Public API exports
pub use crate::core::{BSpline, InterpolationSpline, NaturalSpline};
pub use crate::errors::{SplineError, SplineResult};
pub use crate::types::{Spline, XYVector};
pub use crate::utils::{
    as_vector, as_vector_interpolation, as_vector_natural, spline_des, spline_knots, spline_order,
};

// Internal modules
pub mod basis;
pub mod core;
pub mod errors;
pub mod types;
pub mod utils;

// Re-export common traits and types
pub use ndarray::{Array1, Array2};
pub use num_traits::{Float, Num};

// Include the comparison test modules
#[cfg(test)]
mod test_rust_bs_comparison;

#[cfg(test)]
#[path = "tests/test_final_bs_verification.rs"]
mod test_final_bs_verification;

#[cfg(test)]
#[path = "tests/test_utility_equivalence.rs"]
mod test_utility_equivalence;
