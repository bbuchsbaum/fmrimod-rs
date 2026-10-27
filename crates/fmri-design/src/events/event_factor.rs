//! Categorical event implementation.

use crate::types::{
    validate_durations, validate_onsets, Array, DataFrame, DurationArray, EventSpec, EventType,
    OnsetArray, ValidationError,
};
use ndarray::Array2;
use std::collections::HashMap;

/// Categorical event with discrete levels.
///
/// This represents events that take on discrete values (conditions,
/// categories, etc.) in an fMRI experiment.
#[derive(Debug, Clone)]
pub struct EventFactor {
    /// Event name
    pub name: String,
    /// Event onset times
    pub onsets: OnsetArray,
    /// Event durations
    pub durations: DurationArray,
    /// Values for each event
    pub values: Vec<String>,
    /// Unique levels
    pub levels: Vec<String>,
    /// Optional contrasts
    pub contrasts: Option<HashMap<String, Vec<f64>>>,
}

impl EventFactor {
    /// Create a new categorical event.
    ///
    /// # Arguments
    /// * `name` - Name of the event
    /// * `onsets` - Event onset times
    /// * `values` - Categorical values for each event
    /// * `durations` - Event durations (scalar or array)
    ///
    /// # Example
    /// ```
    /// use fmri_design::events::EventFactor;
    ///
    /// let event = EventFactor::new(
    ///     "condition",
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec!["A", "B", "A", "B"],
    ///     vec![1.0, 1.0, 1.0, 1.0],
    /// ).unwrap();
    /// ```
    pub fn new<S: Into<String>>(
        name: S,
        onsets: Vec<f64>,
        values: Vec<&str>,
        durations: Vec<f64>,
    ) -> Result<Self, ValidationError> {
        let name = name.into();
        let onsets = validate_onsets(onsets)?;
        let durations = validate_durations(durations, onsets.len())?;

        // Convert values to owned strings
        let values: Vec<String> = values.iter().map(|&s| s.to_string()).collect();

        // Validate lengths match
        if values.len() != onsets.len() {
            return Err(ValidationError::LengthMismatch {
                expected: onsets.len(),
                actual: values.len(),
            });
        }

        // Extract unique levels
        let mut levels: Vec<String> = values.clone();
        levels.sort();
        levels.dedup();

        Ok(Self {
            name,
            onsets,
            durations,
            values,
            levels,
            contrasts: None,
        })
    }

    /// Create with explicit level ordering.
    pub fn with_levels(mut self, levels: Vec<String>) -> Result<Self, ValidationError> {
        // Validate that all values are in levels
        for value in &self.values {
            if !levels.contains(value) {
                return Err(ValidationError::InvalidValue(format!(
                    "Value '{}' not in provided levels",
                    value
                )));
            }
        }
        self.levels = levels;
        Ok(self)
    }

    /// Set contrasts for the levels.
    pub fn with_contrasts(mut self, contrasts: HashMap<String, Vec<f64>>) -> Self {
        self.contrasts = Some(contrasts);
        self
    }

    /// Number of unique levels.
    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get indices where events have a specific level.
    pub fn get_level_indices(&self, level: &str) -> Vec<bool> {
        self.values.iter().map(|v| v == level).collect()
    }

    /// Get onset times for a specific level.
    pub fn get_level_onsets(&self, level: &str) -> Vec<f64> {
        self.values
            .iter()
            .zip(self.onsets.iter())
            .filter(|(v, _)| *v == level)
            .map(|(_, &onset)| onset)
            .collect()
    }

    /// Get durations for a specific level.
    pub fn get_level_durations(&self, level: &str) -> Vec<f64> {
        self.values
            .iter()
            .zip(self.durations.iter())
            .filter(|(v, _)| *v == level)
            .map(|(_, &dur)| dur)
            .collect()
    }

    /// Split into separate EventFactor objects by level.
    pub fn split_by_level(&self) -> HashMap<String, EventFactor> {
        let mut result = HashMap::new();

        for level in &self.levels {
            let indices: Vec<usize> = self
                .values
                .iter()
                .enumerate()
                .filter(|(_, v)| *v == level)
                .map(|(i, _)| i)
                .collect();

            if !indices.is_empty() {
                let onsets: Vec<f64> = indices.iter().map(|&i| self.onsets[i]).collect();
                let durations: Vec<f64> = indices.iter().map(|&i| self.durations[i]).collect();
                let values: Vec<&str> = vec![level.as_str(); indices.len()];

                if let Ok(event) = EventFactor::new(
                    format!("{}_{}", self.name, level),
                    onsets,
                    values,
                    durations,
                ) {
                    result.insert(level.clone(), event);
                }
            }
        }

        result
    }
}

impl EventSpec for EventFactor {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Categorical
    }

    fn onsets(&self) -> &OnsetArray {
        &self.onsets
    }

    fn durations(&self) -> &DurationArray {
        &self.durations
    }

    fn n_columns(&self) -> Option<usize> {
        Some(self.levels.len())
    }

    fn column_names(&self) -> Vec<String> {
        self.levels
            .iter()
            .map(|level| crate::naming::level_token(&self.name, level))
            .collect()
    }

    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        let mut values = Array2::zeros((self.onsets.len(), self.levels.len()));
        for (i, value) in self.values.iter().enumerate() {
            let j = self
                .levels
                .iter()
                .position(|level| level == value)
                .ok_or_else(|| ValidationError::InvalidValue(format!("unknown level: {value}")))?;
            values[[i, j]] = 1.0;
        }
        Ok(values)
    }

    fn design_matrix(
        &self,
        sampling_points: &crate::types::TimeArray,
    ) -> Result<Array, ValidationError> {
        let n_points = sampling_points.len();
        let n_levels = self.n_levels();

        // Initialize design matrix
        let mut design_matrix = Array2::zeros((n_points, n_levels));

        // For each level, create indicator column
        for (level_idx, level) in self.levels.iter().enumerate() {
            let level_onsets = self.get_level_onsets(level);
            let level_durations = self.get_level_durations(level);

            // Create boxcar function for this level
            for (onset, duration) in level_onsets.iter().zip(level_durations.iter()) {
                if *duration == 0.0 {
                    // Zero duration - create impulse at nearest sampling point
                    let nearest_idx = sampling_points
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, &t)| ((t - onset).abs() * 1000.0) as i64)
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    design_matrix[[nearest_idx, level_idx]] = 1.0;
                } else {
                    // Non-zero duration - create boxcar
                    for (t_idx, &t) in sampling_points.iter().enumerate() {
                        // Check if this sampling point overlaps with the event
                        // We need to consider that the sampling point represents an interval
                        // centered around t, so we check if the event overlaps with this interval

                        // Get the TR to determine the sampling interval
                        let tr = if t_idx + 1 < sampling_points.len() {
                            sampling_points[t_idx + 1] - t
                        } else if t_idx > 0 {
                            t - sampling_points[t_idx - 1]
                        } else {
                            2.0 // Default TR if we can't determine it
                        };

                        // The sampling interval is [t - tr/2, t + tr/2)
                        let sample_start = t - tr / 2.0;
                        let sample_end = t + tr / 2.0;

                        // Check if the event [onset, onset+duration) overlaps with [sample_start, sample_end)
                        if sample_end > *onset && sample_start < onset + duration {
                            design_matrix[[t_idx, level_idx]] = 1.0;
                        }
                    }
                }
            }
        }

        Ok(design_matrix)
    }

    fn levels(&self) -> Option<&[String]> {
        Some(&self.levels)
    }

    fn values(&self) -> Option<&[String]> {
        Some(&self.values)
    }

    fn to_dataframe(&self) -> DataFrame {
        use crate::dataframe_polars::DataFrameBuilder;

        DataFrameBuilder::new()
            .with_column("onset", self.onsets.to_vec())
            .expect("Failed to add onset column")
            .with_column("duration", self.durations.to_vec())
            .expect("Failed to add duration column")
            .with_str_column(&self.name, self.values.iter().map(|s| s.as_str()).collect())
            .expect("Failed to add values column")
            .build()
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
        let mut filtered_values = Vec::new();
        let mut filtered_durations = Vec::new();

        for (i, &include) in mask.iter().enumerate() {
            if include {
                filtered_onsets.push(self.onsets[i]);
                filtered_values.push(self.values[i].as_str());
                filtered_durations.push(self.durations[i]);
            }
        }

        if filtered_onsets.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Filter resulted in empty event".to_string(),
            ));
        }

        Ok(Box::new(EventFactor::new(
            self.name.clone(),
            filtered_onsets,
            filtered_values,
            filtered_durations,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_event_factor_creation() {
        let event = EventFactor::new(
            "condition",
            vec![1.0, 2.0, 3.0, 4.0],
            vec!["A", "B", "A", "B"],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        assert_eq!(event.name(), "condition");
        assert_eq!(event.n_levels(), 2);
        assert_eq!(event.levels, vec!["A", "B"]);
    }

    #[test]
    fn test_level_methods() {
        let event = EventFactor::new(
            "condition",
            vec![1.0, 2.0, 3.0, 4.0],
            vec!["A", "B", "A", "B"],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        let a_onsets = event.get_level_onsets("A");
        assert_eq!(a_onsets, vec![1.0, 3.0]);

        let b_onsets = event.get_level_onsets("B");
        assert_eq!(b_onsets, vec![2.0, 4.0]);
    }

    #[test]
    fn test_design_matrix() {
        let event =
            EventFactor::new("condition", vec![0.0, 2.0], vec!["A", "B"], vec![1.0, 1.0]).unwrap();

        let sampling_points = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[7, 2]);

        // Check A column (first event)
        assert_eq!(dm[[0, 0]], 1.0); // t=0.0
        assert_eq!(dm[[1, 0]], 1.0); // t=0.5
        assert_eq!(dm[[2, 0]], 1.0); // t=1.0 (overlaps with [0.75, 1.25] interval)

        // Check B column (second event)
        assert_eq!(dm[[4, 1]], 1.0); // t=2.0
        assert_eq!(dm[[5, 1]], 1.0); // t=2.5
        assert_eq!(dm[[6, 1]], 1.0); // t=3.0 (overlaps with [2.75, 3.25] interval)
    }
}
