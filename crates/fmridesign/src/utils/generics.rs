//! Generic utility functions for extracting information from model objects.
//!
//! These functions use trait-based dispatch to provide generic access to
//! various properties of model objects.

use crate::types::{EventSpec, EventType};
use ndarray::{Array1, Array2};
use std::fmt::Debug;

/// Trait for objects that have block identifiers
pub trait HasBlockIds {
    /// Extract block IDs from the object
    fn blockids(&self) -> &Array1<i32>;
}

/// Trait for objects that have block lengths
pub trait HasBlockLengths {
    /// Extract block lengths from the object
    fn blocklens(&self) -> &Array1<usize>;
}

/// Trait for objects that have term names
pub trait HasTermNames {
    /// Extract term names from the object
    fn term_names(&self) -> &[String];
}

/// Trait for objects that have long names
pub trait HasLongNames {
    /// Extract long (descriptive) names from the object
    fn longnames(&self) -> &[String];
}

/// Trait for objects that have short names
pub trait HasShortNames {
    /// Extract short (abbreviated) names from the object
    fn shortnames(&self) -> &[String];
}

/// Trait for objects that have cells (factor combinations)
pub trait HasCells {
    /// Extract cell information (factor level combinations)
    fn cells(&self) -> Vec<Vec<String>>;
}

/// Trait for objects that have conditions
pub trait HasConditions {
    /// Extract condition names or labels
    fn conditions(&self) -> &[String];
}

/// Trait for objects that have onsets
pub trait HasOnsets {
    /// Extract onset times
    fn onsets(&self) -> &Array1<f64>;
}

/// Trait for objects that have durations
pub trait HasDurations {
    /// Extract event durations
    fn durations(&self) -> &Array1<f64>;
}

/// Trait for objects that have elements (matrix elements)
pub trait HasElements {
    /// Extract matrix elements
    fn elements(&self) -> &Array2<f64>;
}

/// Trait for objects that have labels
pub trait HasLabels {
    /// Extract labels
    fn labels(&self) -> &[String];
}

/// Trait for objects that have levels (factor levels)
pub trait HasLevels {
    /// Extract factor levels
    fn levels(&self) -> &[String];
}

/// Trait for objects that have columns
pub trait HasColumns {
    /// Extract column indices or names
    fn columns(&self) -> Vec<usize>;
}

/// Trait for objects that have basis functions
pub trait HasNBasis {
    /// Get number of basis functions
    fn nbasis(&self) -> usize;
}

/// Trait for categorical events
pub trait IsCategorical {
    /// Check if the event is categorical
    fn is_categorical(&self) -> bool;
}

/// Trait for continuous events
pub trait IsContinuous {
    /// Check if the event is continuous
    fn is_continuous(&self) -> bool;
}

/// Trait for objects that have event terms
pub trait HasEventTerms {
    /// Extract event terms
    fn event_terms(&self) -> Vec<Box<dyn EventSpec>>;
}

/// Default implementations for EventSpec types
impl<T: EventSpec> IsCategorical for T {
    fn is_categorical(&self) -> bool {
        matches!(self.event_type(), EventType::Categorical)
    }
}

impl<T: EventSpec> IsContinuous for T {
    fn is_continuous(&self) -> bool {
        matches!(self.event_type(), EventType::Continuous)
    }
}

/// Builder trait for constructing objects from data
pub trait Construct: Sized {
    /// Type of error that can occur during construction
    type Error: Debug;

    /// Construct an object from generic data
    fn construct(data: ConstructData) -> Result<Self, Self::Error>;
}

/// Data structure for generic construction
#[derive(Debug, Clone)]
pub struct ConstructData {
    /// Name or identifier
    pub name: Option<String>,
    /// Numeric data
    pub data: Option<Array2<f64>>,
    /// String labels
    pub labels: Option<Vec<String>>,
    /// Additional parameters
    pub params: std::collections::HashMap<String, ConstructParam>,
}

/// Parameter types for construction
#[derive(Debug, Clone)]
pub enum ConstructParam {
    /// Scalar float value
    Float(f64),
    /// Scalar integer value
    Integer(i64),
    /// String value
    String(String),
    /// Boolean value
    Boolean(bool),
    /// Array of floats
    FloatArray(Vec<f64>),
    /// Array of strings
    StringArray(Vec<String>),
}

impl ConstructData {
    /// Create new empty construction data
    pub fn new() -> Self {
        Self {
            name: None,
            data: None,
            labels: None,
            params: std::collections::HashMap::new(),
        }
    }

    /// Set the name
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the data
    pub fn with_data(mut self, data: Array2<f64>) -> Self {
        self.data = Some(data);
        self
    }

    /// Set the labels
    pub fn with_labels(mut self, labels: Vec<String>) -> Self {
        self.labels = Some(labels);
        self
    }

    /// Add a parameter
    pub fn with_param<S: Into<String>>(mut self, key: S, value: ConstructParam) -> Self {
        self.params.insert(key.into(), value);
        self
    }
}

impl Default for ConstructData {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility function to check if all elements in a slice are equal
pub fn all_equal<T: PartialEq>(slice: &[T]) -> bool {
    if slice.is_empty() {
        return true;
    }
    let first = &slice[0];
    slice.iter().all(|x| x == first)
}

/// Utility function to get unique elements from a slice while preserving order
pub fn unique_ordered<T: Clone + PartialEq>(slice: &[T]) -> Vec<T> {
    let mut result = Vec::new();
    for item in slice {
        if !result.contains(item) {
            result.push(item.clone());
        }
    }
    result
}

/// Utility function to repeat a value n times
pub fn rep<T: Clone>(value: T, n: usize) -> Vec<T> {
    vec![value; n]
}

/// Utility function to create a sequence from start to end (inclusive)
pub fn seq(start: i32, end: i32) -> Vec<i32> {
    if start <= end {
        (start..=end).collect()
    } else {
        (end..=start).rev().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_equal() {
        assert!(all_equal(&[1, 1, 1]));
        assert!(!all_equal(&[1, 2, 1]));
        assert!(all_equal::<i32>(&[]));
    }

    #[test]
    fn test_unique_ordered() {
        assert_eq!(unique_ordered(&[1, 2, 3, 2, 1]), vec![1, 2, 3]);
        assert_eq!(unique_ordered(&["a", "b", "a", "c"]), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_rep() {
        assert_eq!(rep(5, 3), vec![5, 5, 5]);
        assert_eq!(rep("hello", 2), vec!["hello", "hello"]);
    }

    #[test]
    fn test_seq() {
        assert_eq!(seq(1, 5), vec![1, 2, 3, 4, 5]);
        assert_eq!(seq(5, 1), vec![5, 4, 3, 2, 1]);
        assert_eq!(seq(0, 0), vec![0]);
    }
}
