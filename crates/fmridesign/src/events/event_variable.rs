//! Continuous event implementation.

use crate::types::{
    validate_durations, validate_onsets, Array, DataFrame, DurationArray, EventSpec, EventType,
    OnsetArray, ValidationError,
};
use ndarray::{Array1, Array2};

/// Continuous event with numeric values.
///
/// This represents events with continuous values (ratings, reaction times,
/// parametric modulators, etc.) in an fMRI experiment.
#[derive(Debug, Clone)]
pub struct EventVariable {
    /// Event name
    pub name: String,
    /// Event onset times
    pub onsets: OnsetArray,
    /// Event durations
    pub durations: DurationArray,
    /// Transformed values (after centering/scaling)
    pub values: Array1<f64>,
    /// Raw values (before transformation)
    pub raw_values: Array1<f64>,
    /// Whether values are centered
    pub center: bool,
    /// Whether values are scaled
    pub scale: bool,
}

impl EventVariable {
    /// Create a new continuous event.
    ///
    /// # Arguments
    /// * `name` - Name of the event
    /// * `onsets` - Event onset times
    /// * `values` - Continuous values for each event
    /// * `durations` - Event durations
    ///
    /// # Example
    /// ```
    /// use fmridesign::events::EventVariable;
    ///
    /// let event = EventVariable::new(
    ///     "rating",
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![7.5, 3.2, 8.1, 5.5],
    ///     vec![1.0, 1.0, 1.0, 1.0],
    /// ).unwrap();
    /// ```
    pub fn new<S: Into<String>>(
        name: S,
        onsets: Vec<f64>,
        values: Vec<f64>,
        durations: Vec<f64>,
    ) -> Result<Self, ValidationError> {
        let name = name.into();
        let onsets = validate_onsets(onsets)?;
        let durations = validate_durations(durations, onsets.len())?;

        // Validate values
        if values.len() != onsets.len() {
            return Err(ValidationError::LengthMismatch {
                expected: onsets.len(),
                actual: values.len(),
            });
        }

        let raw_values = Array1::from_vec(values);

        // Check for finite values
        if raw_values.iter().any(|&x| !x.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }

        // Preserve supplied amplitudes by default, matching R event_variable.
        let transformed_values = Self::transform_values(&raw_values, false, false);

        Ok(Self {
            name,
            onsets,
            durations,
            values: transformed_values,
            raw_values,
            center: false,
            scale: false,
        })
    }

    /// Set whether to center values (subtract mean).
    pub fn with_center(mut self, center: bool) -> Self {
        self.center = center;
        self.values = Self::transform_values(&self.raw_values, center, self.scale);
        self
    }

    /// Set whether to scale values (divide by std).
    pub fn with_scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self.values = Self::transform_values(&self.raw_values, self.center, scale);
        self
    }

    /// Apply centering and/or scaling to values.
    fn transform_values(values: &Array1<f64>, center: bool, scale: bool) -> Array1<f64> {
        let mut result = values.clone();

        if center {
            let mean = values.mean().unwrap_or(0.0);
            result = result - mean;
        }

        if scale {
            let std = result.std(0.0);
            if std > 0.0 {
                result = result / std;
            } else {
                // All values are the same
                result.fill(0.0);
            }
        }

        result
    }

    /// Mean of raw values.
    pub fn mean(&self) -> f64 {
        self.raw_values.mean().unwrap_or(0.0)
    }

    /// Standard deviation of raw values.
    pub fn std(&self) -> f64 {
        self.raw_values.std(0.0)
    }

    /// Minimum of raw values.
    pub fn min(&self) -> f64 {
        self.raw_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    }

    /// Maximum of raw values.
    pub fn max(&self) -> f64 {
        self.raw_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Bin continuous values into discrete categories.
    ///
    /// # Arguments
    /// * `n_bins` - Number of bins to create
    /// * `method` - Binning method: "equal_width", "equal_frequency", or "custom"
    /// * `custom_bins` - Custom bin edges (only used if method is "custom")
    ///
    /// # Returns
    /// A new EventFactor with binned values
    ///
    /// # Example
    /// ```ignore
    /// // Equal width binning
    /// let binned = event.bin_values(3, "equal_width", None)?;
    ///
    /// // Custom bin edges
    /// let binned = event.bin_values(0, "custom", Some(vec![0.0, 2.5, 5.0, 7.5, 10.0]))?;
    /// ```
    pub fn bin_values(
        &self,
        n_bins: usize,
        method: &str,
        custom_bins: Option<Vec<f64>>,
    ) -> Result<crate::events::EventFactor, ValidationError> {
        use crate::events::EventFactor;

        let bin_edges = match method {
            "equal_width" => {
                if n_bins == 0 {
                    return Err(ValidationError::InvalidParameter(
                        "n_bins must be > 0 for equal_width binning".to_string(),
                    ));
                }

                let min_val = self.min();
                let max_val = self.max();
                let width = (max_val - min_val) / n_bins as f64;

                (0..=n_bins)
                    .map(|i| min_val + i as f64 * width)
                    .collect::<Vec<_>>()
            }
            "equal_frequency" | "quantile" => {
                if n_bins == 0 {
                    return Err(ValidationError::InvalidParameter(
                        "n_bins must be > 0 for quantile binning".to_string(),
                    ));
                }

                // Sort values
                let mut sorted_values = self.raw_values.to_vec();
                sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Calculate quantile positions
                let mut edges = vec![sorted_values[0]];
                for i in 1..n_bins {
                    let pos = (i as f64 / n_bins as f64 * sorted_values.len() as f64) as usize;
                    let pos = pos.min(sorted_values.len() - 1);
                    edges.push(sorted_values[pos]);
                }
                edges.push(sorted_values[sorted_values.len() - 1]);

                // Remove duplicates and sort
                edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
                edges.dedup();
                edges
            }
            "custom" => custom_bins.ok_or_else(|| {
                ValidationError::InvalidParameter(
                    "custom_bins required for custom binning method".to_string(),
                )
            })?,
            _ => {
                return Err(ValidationError::InvalidParameter(format!(
                    "Unknown binning method: {}",
                    method
                )));
            }
        };

        // Assign values to bins
        let bin_labels: Vec<String> = self
            .raw_values
            .iter()
            .map(|&val| {
                // Find which bin this value belongs to
                let bin_idx = bin_edges
                    .windows(2)
                    .position(|w| val >= w[0] && val < w[1])
                    .or_else(|| {
                        // Handle the last bin (inclusive on both ends)
                        if val == bin_edges[bin_edges.len() - 1] {
                            Some(bin_edges.len() - 2)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                // Create bin label
                if bin_edges.len() > 1 && bin_idx < bin_edges.len() - 1 {
                    format!(
                        "bin{}_[{:.2},{:.2})",
                        bin_idx + 1,
                        bin_edges[bin_idx],
                        bin_edges[bin_idx + 1]
                    )
                } else {
                    format!("bin{}", bin_idx + 1)
                }
            })
            .collect();

        // Create EventFactor with binned values
        let values_as_refs: Vec<&str> = bin_labels.iter().map(|s| s.as_str()).collect();
        EventFactor::new(
            format!("{}_binned", self.name),
            self.onsets.to_vec(),
            values_as_refs,
            self.durations.to_vec(),
        )
    }
}

impl EventSpec for EventVariable {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Continuous
    }

    fn onsets(&self) -> &OnsetArray {
        &self.onsets
    }

    fn durations(&self) -> &DurationArray {
        &self.durations
    }

    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        Ok(self.values.clone().insert_axis(ndarray::Axis(1)))
    }

    fn design_matrix(
        &self,
        sampling_points: &crate::types::TimeArray,
    ) -> Result<Array, ValidationError> {
        let n_points = sampling_points.len();

        // Initialize design matrix column
        let mut design_matrix = Array2::zeros((n_points, 1));

        // Fill in values at event times
        for (i, (onset, duration)) in self.onsets.iter().zip(self.durations.iter()).enumerate() {
            let value = self.values[i];

            if *duration == 0.0 {
                // Impulse event - find nearest sampling point
                let nearest_idx = sampling_points
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &t)| ((t - onset).abs() * 1000.0) as i64)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                design_matrix[[nearest_idx, 0]] = value;
            } else {
                // Extended event - check overlap with sampling intervals
                for (t_idx, &t) in sampling_points.iter().enumerate() {
                    // Get the TR to determine the sampling interval
                    let tr = if t_idx + 1 < sampling_points.len() {
                        sampling_points[t_idx + 1] - t
                    } else if t_idx > 0 {
                        t - sampling_points[t_idx - 1]
                    } else {
                        1.0 // Default TR if we can't determine it
                    };

                    // The sampling interval is [t - tr/2, t + tr/2)
                    let sample_start = t - tr / 2.0;
                    let sample_end = t + tr / 2.0;

                    // Check if the event [onset, onset+duration) overlaps with [sample_start, sample_end)
                    if sample_end > *onset && sample_start < onset + duration {
                        design_matrix[[t_idx, 0]] = value;
                    }
                }
            }
        }

        Ok(design_matrix)
    }

    fn to_dataframe(&self) -> DataFrame {
        use crate::dataframe_polars::DataFrameBuilder;

        let mut builder = DataFrameBuilder::new()
            .with_column("onset", self.onsets.to_vec())
            .expect("Failed to add onset column")
            .with_column("duration", self.durations.to_vec())
            .expect("Failed to add duration column")
            .with_column(&self.name, self.values.to_vec())
            .expect("Failed to add values column");

        // Add raw values if they differ from transformed values
        if self.values != self.raw_values {
            builder = builder
                .with_column(format!("{}_raw", self.name), self.raw_values.to_vec())
                .expect("Failed to add raw values column");
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
        let mut filtered_values = Vec::new();
        let mut filtered_durations = Vec::new();
        let mut filtered_raw_values = Vec::new();

        for (i, &include) in mask.iter().enumerate() {
            if include {
                filtered_onsets.push(self.onsets[i]);
                filtered_values.push(self.values[i]);
                filtered_durations.push(self.durations[i]);
                filtered_raw_values.push(self.raw_values[i]);
            }
        }

        if filtered_onsets.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Filter resulted in empty event".to_string(),
            ));
        }

        // Create new event with filtered data
        let mut event = EventVariable::new(
            self.name.clone(),
            filtered_onsets,
            filtered_raw_values,
            filtered_durations,
        )?;

        // Apply same transformations
        event = event.with_center(self.center).with_scale(self.scale);

        Ok(Box::new(event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_event_variable_creation() {
        let event = EventVariable::new(
            "rating",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![7.5, 3.2, 8.1, 5.5],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        assert_eq!(event.name(), "rating");
        assert_eq!(event.event_type(), EventType::Continuous);

        // Summary statistics retain the original values.
        let mean = (7.5 + 3.2 + 8.1 + 5.5) / 4.0;
        assert!((event.mean() - mean).abs() < 1e-10);
    }

    #[test]
    fn test_transform_values() {
        let event = EventVariable::new(
            "rating",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap()
        .with_center(true)
        .with_scale(true);

        // After centering and scaling, values should have mean 0 and std 1
        assert!((event.values.mean().unwrap_or(0.0)).abs() < 1e-10);
        assert!((event.values.std(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_design_matrix() {
        let event = EventVariable::new("rating", vec![0.0, 2.0], vec![5.0, 10.0], vec![1.0, 1.0])
            .unwrap()
            .with_center(false); // Don't center for easier testing

        let sampling_points = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[7, 1]);

        // Check values
        assert_eq!(dm[[0, 0]], 5.0); // t=0.0
        assert_eq!(dm[[1, 0]], 5.0); // t=0.5
        assert_eq!(dm[[2, 0]], 5.0); // t=1.0 (overlaps with [0.75, 1.25] interval)
        assert_eq!(dm[[4, 0]], 10.0); // t=2.0
        assert_eq!(dm[[5, 0]], 10.0); // t=2.5
        assert_eq!(dm[[6, 0]], 10.0); // t=3.0 (overlaps with [2.75, 3.25] interval)
    }
}
