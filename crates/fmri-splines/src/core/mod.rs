//! Core spline implementations

pub mod bspline;
pub mod interpolation;
pub mod naturalspline;
// TODO: Implement additional spline types
// pub mod periodic;
// pub mod polynomial;

pub use bspline::*;
pub use interpolation::*;
pub use naturalspline::*;
