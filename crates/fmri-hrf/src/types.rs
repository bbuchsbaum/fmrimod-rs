//! Common types used throughout the library.

use ndarray::{Array1, Array2};

/// Type alias for 1D array of f64 values.
pub type Vector = Array1<f64>;

/// Type alias for 2D array of f64 values.
pub type Matrix = Array2<f64>;

/// Type alias for time array.
pub type TimeArray = Array1<f64>;

/// Type alias for HRF values (can be 1D or 2D for multi-basis).
pub type HrfValues = Array2<f64>;
