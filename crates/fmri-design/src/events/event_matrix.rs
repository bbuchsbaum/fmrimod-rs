//! Multi-column event implementation.

use crate::types::{
    validate_durations, validate_onsets, Array, DataFrame, DurationArray, EventSpec, EventType,
    OnsetArray, ValidationError,
};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Multi-column event representation.
///
/// This represents events with multiple simultaneous values, such as
/// multi-dimensional parametric modulators or pre-computed design columns.
#[derive(Debug, Clone)]
pub struct EventMatrix {
    name: String,
    onsets: OnsetArray,
    durations: DurationArray,
    values: Array2<f64>,
    column_names: Vec<String>,
}

impl EventMatrix {
    /// Create a new multi-column event.
    ///
    /// # Arguments
    /// * `name` - Name of the event
    /// * `onsets` - Event onset times
    /// * `values` - Matrix of values, shape (n_events, n_columns)
    /// * `durations` - Event durations
    ///
    /// # Example
    /// ```
    /// use fmri_design::events::EventMatrix;
    /// use ndarray::array;
    ///
    /// let values = array![[0.1, 0.2, 0.3],
    ///                     [0.2, 0.1, 0.4],
    ///                     [0.3, 0.3, 0.2],
    ///                     [0.1, 0.4, 0.3]];
    ///
    /// let event = EventMatrix::new(
    ///     "motion",
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     values,
    ///     vec![1.0, 1.0, 1.0, 1.0],
    /// ).unwrap();
    /// ```
    pub fn new<S: Into<String>>(
        name: S,
        onsets: Vec<f64>,
        values: Array2<f64>,
        durations: Vec<f64>,
    ) -> Result<Self, ValidationError> {
        let name = name.into();
        let onsets = validate_onsets(onsets)?;
        let durations = validate_durations(durations, onsets.len())?;

        // Validate dimensions
        if values.nrows() != onsets.len() {
            return Err(ValidationError::LengthMismatch {
                expected: onsets.len(),
                actual: values.nrows(),
            });
        }

        // Check for finite values
        if values.iter().any(|&x| !x.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }

        // Generate default column names
        let n_columns = values.ncols();
        let column_names: Vec<String> = (0..n_columns)
            .map(|i| format!("{}_{}", name, i + 1))
            .collect();

        Ok(Self {
            name,
            onsets,
            durations,
            values,
            column_names,
        })
    }

    /// Set custom column names.
    pub fn with_column_names(mut self, names: Vec<String>) -> Result<Self, ValidationError> {
        if names.len() != self.n_columns() {
            return Err(ValidationError::LengthMismatch {
                expected: self.n_columns(),
                actual: names.len(),
            });
        }
        self.column_names = names;
        Ok(self)
    }

    /// Number of columns.
    pub fn n_columns(&self) -> usize {
        self.values.ncols()
    }

    /// Get a specific column.
    pub fn get_column(&self, index: usize) -> Option<Array1<f64>> {
        if index < self.n_columns() {
            Some(self.values.column(index).to_owned())
        } else {
            None
        }
    }

    /// Get a column by name.
    pub fn get_column_by_name(&self, name: &str) -> Option<Array1<f64>> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.get_column(idx))
    }

    /// Split into separate EventVariable objects by column.
    pub fn split_columns(&self) -> HashMap<String, crate::events::EventVariable> {
        use crate::events::EventVariable;

        let mut result = HashMap::new();

        for (i, col_name) in self.column_names.iter().enumerate() {
            let values: Vec<f64> = self.values.column(i).to_vec();

            if let Ok(event) = EventVariable::new(
                col_name.clone(),
                self.onsets.to_vec(),
                values,
                self.durations.to_vec(),
            ) {
                // Don't center or scale - already processed
                let event = event.with_center(false).with_scale(false);
                result.insert(col_name.clone(), event);
            }
        }

        result
    }

    /// Apply transformation matrix to values.
    pub fn apply_transform(&self, transform: &Array2<f64>) -> Result<Self, ValidationError> {
        // Check dimensions
        if transform.nrows() != self.n_columns() {
            return Err(ValidationError::InvalidParameter(format!(
                "Transform matrix rows ({}) must match number of columns ({})",
                transform.nrows(),
                self.n_columns()
            )));
        }

        // Matrix multiply: values @ transform
        let new_values = self.values.dot(transform);

        // Create new column names
        let n_new_cols = transform.ncols();
        let column_names: Vec<String> = (0..n_new_cols)
            .map(|i| format!("{}_transformed_{}", self.name, i + 1))
            .collect();

        Ok(Self {
            name: format!("{}_transformed", self.name),
            onsets: self.onsets.clone(),
            durations: self.durations.clone(),
            values: new_values,
            column_names,
        })
    }
}

impl EventSpec for EventMatrix {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Matrix
    }

    fn onsets(&self) -> &OnsetArray {
        &self.onsets
    }

    fn durations(&self) -> &DurationArray {
        &self.durations
    }

    fn column_names(&self) -> Vec<String> {
        self.column_names.clone()
    }

    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        Ok(self.values.clone())
    }

    fn design_matrix(
        &self,
        sampling_points: &crate::types::TimeArray,
    ) -> Result<Array, ValidationError> {
        let n_points = sampling_points.len();
        let n_columns = self.n_columns();

        // Initialize design matrix
        let mut design_matrix = Array2::zeros((n_points, n_columns));

        // Fill in values at event times
        for (i, (onset, duration)) in self.onsets.iter().zip(self.durations.iter()).enumerate() {
            let values = self.values.row(i);

            if *duration == 0.0 {
                // Impulse event - find nearest sampling point
                let nearest_idx = sampling_points
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &t)| ((t - onset).abs() * 1000.0) as i64)
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                for j in 0..n_columns {
                    design_matrix[[nearest_idx, j]] = values[j];
                }
            } else {
                // Extended event - fill duration
                for (t_idx, &t) in sampling_points.iter().enumerate() {
                    if t >= *onset && t < onset + duration {
                        for j in 0..n_columns {
                            design_matrix[[t_idx, j]] = values[j];
                        }
                    }
                }
            }
        }

        Ok(design_matrix)
    }

    fn n_columns(&self) -> Option<usize> {
        Some(self.n_columns())
    }

    fn to_dataframe(&self) -> DataFrame {
        use crate::dataframe_polars::DataFrameBuilder;

        let mut builder = DataFrameBuilder::new()
            .with_column("onset", self.onsets.to_vec())
            .expect("Failed to add onset column")
            .with_column("duration", self.durations.to_vec())
            .expect("Failed to add duration column");

        // Add each matrix column
        for (i, col_name) in self.column_names.iter().enumerate() {
            let col_values: Vec<f64> = (0..self.onsets.len())
                .map(|row| self.values[[row, i]])
                .collect();
            builder = builder
                .with_column(col_name, col_values)
                .expect("Failed to add matrix column");
        }

        builder.build()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EventSpec> {
        Box::new(self.clone())
    }

    fn filter(&self, mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError> {
        if mask.len() != self.n_events() {
            return Err(ValidationError::LengthMismatch {
                expected: self.n_events(),
                actual: mask.len(),
            });
        }

        let mut filtered_onsets = Vec::new();
        let mut filtered_durations = Vec::new();
        let mut filtered_rows = Vec::new();

        for (i, &include) in mask.iter().enumerate() {
            if include {
                filtered_onsets.push(self.onsets[i]);
                filtered_durations.push(self.durations[i]);
                filtered_rows.push(i);
            }
        }

        if filtered_onsets.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Filter resulted in empty event".to_string(),
            ));
        }

        // Create new matrix with filtered rows
        let n_cols = self.values.ncols();
        let mut filtered_values = Array2::zeros((filtered_rows.len(), n_cols));

        for (new_i, &old_i) in filtered_rows.iter().enumerate() {
            filtered_values
                .row_mut(new_i)
                .assign(&self.values.row(old_i));
        }

        let mut event = EventMatrix::new(
            self.name.clone(),
            filtered_onsets,
            filtered_values,
            filtered_durations,
        )?;

        // Set column names if present
        if !self.column_names.is_empty() {
            event = event.with_column_names(self.column_names.clone())?;
        }

        Ok(Box::new(event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_event_matrix_creation() {
        let values = array![
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.2],
            [0.1, 0.4, 0.3]
        ];

        let event = EventMatrix::new(
            "motion",
            vec![1.0, 2.0, 3.0, 4.0],
            values,
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        assert_eq!(event.name(), "motion");
        assert_eq!(event.event_type(), EventType::Matrix);
        assert_eq!(event.n_columns(), 3);
        assert_eq!(event.column_names, vec!["motion_1", "motion_2", "motion_3"]);
    }

    #[test]
    fn test_column_names() {
        let values = array![[0.1, 0.2, 0.3], [0.2, 0.1, 0.4]];

        let event = EventMatrix::new("motion", vec![1.0, 2.0], values, vec![1.0, 1.0])
            .unwrap()
            .with_column_names(vec!["x".to_string(), "y".to_string(), "z".to_string()])
            .unwrap();

        assert_eq!(event.column_names, vec!["x", "y", "z"]);
    }

    #[test]
    fn test_get_column() {
        let values = array![
            [0.1, 0.2, 0.3],
            [0.2, 0.1, 0.4],
            [0.3, 0.3, 0.2],
            [0.1, 0.4, 0.3]
        ];

        let event = EventMatrix::new(
            "motion",
            vec![1.0, 2.0, 3.0, 4.0],
            values,
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap()
        .with_column_names(vec!["x".to_string(), "y".to_string(), "z".to_string()])
        .unwrap();

        let col = event.get_column(0).unwrap();
        assert_eq!(col, array![0.1, 0.2, 0.3, 0.1]);

        let col_by_name = event.get_column_by_name("y").unwrap();
        assert_eq!(col_by_name, array![0.2, 0.1, 0.3, 0.4]);
    }

    #[test]
    fn test_design_matrix() {
        let values = array![[1.0, 2.0], [3.0, 4.0]];

        let event = EventMatrix::new("test", vec![0.0, 2.0], values, vec![1.0, 1.0]).unwrap();

        let sampling_points = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[7, 2]);

        // First event at t=0
        assert_eq!(dm[[0, 0]], 1.0);
        assert_eq!(dm[[0, 1]], 2.0);
        assert_eq!(dm[[1, 0]], 1.0);
        assert_eq!(dm[[1, 1]], 2.0);

        // Second event at t=2
        assert_eq!(dm[[4, 0]], 3.0);
        assert_eq!(dm[[4, 1]], 4.0);
        assert_eq!(dm[[5, 0]], 3.0);
        assert_eq!(dm[[5, 1]], 4.0);
    }

    #[test]
    fn test_apply_transform() {
        let values = array![[1.0, 0.0], [0.0, 1.0]];

        let event = EventMatrix::new("test", vec![0.0, 1.0], values, vec![1.0, 1.0]).unwrap();

        // Rotation matrix
        let transform = array![[0.0, -1.0], [1.0, 0.0]];

        let transformed = event.apply_transform(&transform).unwrap();

        assert_eq!(transformed.values, array![[0.0, -1.0], [1.0, 0.0]]);
    }
}
