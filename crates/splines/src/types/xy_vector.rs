//! XYVector type - equivalent to R's xyVector class

use crate::errors::{SplineError, SplineResult};
use ndarray::Array1;
use num_traits::Float;
use serde::{Deserialize, Serialize};
use std::fmt;

/// A vector that pairs x and y coordinates together
///
/// This is equivalent to R's xyVector class and is used to represent
/// responses that carry their corresponding inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct XYVector<T: Float> {
    /// X coordinates (inputs)
    pub x: Array1<T>,
    /// Y coordinates (outputs/responses)  
    pub y: Array1<T>,
}

impl<T: Float> XYVector<T> {
    /// Create a new XYVector from x and y coordinates
    pub fn new(x: Vec<T>, y: Vec<T>) -> SplineResult<Self> {
        if x.len() != y.len() {
            return Err(SplineError::dimension_mismatch(x.len(), y.len()));
        }

        Ok(XYVector {
            x: Array1::from(x),
            y: Array1::from(y),
        })
    }

    /// Create from arrays
    pub fn from_arrays(x: Array1<T>, y: Array1<T>) -> SplineResult<Self> {
        if x.len() != y.len() {
            return Err(SplineError::dimension_mismatch(x.len(), y.len()));
        }

        Ok(XYVector { x, y })
    }

    /// Get the length of the vector
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Check if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Get x coordinates as a slice
    pub fn x_values(&self) -> &[T] {
        self.x.as_slice().unwrap()
    }

    /// Get y coordinates as a slice  
    pub fn y_values(&self) -> &[T] {
        self.y.as_slice().unwrap()
    }

    /// Get a reference to the x array
    pub fn x_array(&self) -> &Array1<T> {
        &self.x
    }

    /// Get a reference to the y array
    pub fn y_array(&self) -> &Array1<T> {
        &self.y
    }

    /// Extract just the y values (equivalent to R's asVector method)
    pub fn as_vector(&self) -> Array1<T> {
        self.y.clone()
    }

    /// Get a specific (x, y) pair by index
    pub fn get(&self, index: usize) -> Option<(T, T)> {
        if index < self.len() {
            Some((self.x[index], self.y[index]))
        } else {
            None
        }
    }

    /// Iterate over (x, y) pairs
    pub fn iter(&self) -> impl Iterator<Item = (T, T)> + '_ {
        self.x.iter().zip(self.y.iter()).map(|(&x, &y)| (x, y))
    }

    /// Create a data frame-like representation
    /// Returns (x_vec, y_vec) for further processing
    pub fn to_data_frame(&self) -> (Vec<T>, Vec<T>) {
        (self.x.to_vec(), self.y.to_vec())
    }
}

impl<T: Float + fmt::Display> fmt::Display for XYVector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "XYVector with {} points:", self.len())?;
        for (i, (x, y)) in self.iter().enumerate() {
            if i < 10 {
                // Show first 10 points
                writeln!(f, "  [{:3}] x: {}, y: {}", i, x, y)?;
            } else if i == 10 {
                writeln!(f, "  ... ({} more points)", self.len() - 10)?;
                break;
            }
        }
        Ok(())
    }
}

// Conversion from tuples
impl<T: Float> From<(Vec<T>, Vec<T>)> for XYVector<T> {
    fn from((x, y): (Vec<T>, Vec<T>)) -> Self {
        XYVector::new(x, y).expect("Length mismatch in XYVector::from")
    }
}

impl<T: Float> From<(Array1<T>, Array1<T>)> for XYVector<T> {
    fn from((x, y): (Array1<T>, Array1<T>)) -> Self {
        XYVector::from_arrays(x, y).expect("Length mismatch in XYVector::from")
    }
}
