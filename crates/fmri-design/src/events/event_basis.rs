//! Basis function event implementation.

use crate::types::{
    validate_durations, validate_onsets, Array, BasisFunction, DataFrame, DurationArray, EventSpec,
    EventType, OnsetArray, ValidationError,
};
use ndarray::{Array1, Array2};

/// Event with basis function expansion.
///
/// This represents continuous events that are expanded using basis functions
/// (polynomials, splines, etc.) to capture non-linear relationships.
#[derive(Debug)]
pub struct EventBasis {
    name: String,
    onsets: OnsetArray,
    durations: DurationArray,
    values: Array1<f64>,
    #[allow(dead_code)] // retained for future basis-aware operations
    basis: Box<dyn BasisFunction>,
    expanded_values: Array2<f64>,
}

impl EventBasis {
    /// Create a new event with basis function expansion.
    ///
    /// # Arguments
    /// * `name` - Name of the event
    /// * `onsets` - Event onset times
    /// * `values` - Values to be expanded by basis functions
    /// * `basis` - Basis function to apply
    /// * `durations` - Event durations
    ///
    /// # Example
    /// ```
    /// use fmri_design::events::EventBasis;
    /// use fmri_design::basis::PolynomialBasis;
    ///
    /// let basis = PolynomialBasis::new(2);
    /// let event = EventBasis::new(
    ///     "rating",
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     vec![1.0, 2.0, 3.0, 4.0],
    ///     Box::new(basis),
    ///     vec![1.0, 1.0, 1.0, 1.0],
    /// ).unwrap();
    /// ```
    pub fn new<S: Into<String>>(
        name: S,
        onsets: Vec<f64>,
        values: Vec<f64>,
        basis: Box<dyn BasisFunction>,
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

        let values = Array1::from_vec(values);

        // Check for finite values
        if values.iter().any(|&x| !x.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }

        // Expand values using basis functions
        let expanded_values = basis.evaluate(&values.view())?;

        Ok(Self {
            name,
            onsets,
            durations,
            values,
            basis,
            expanded_values,
        })
    }

    /// Number of basis functions.
    pub fn n_basis(&self) -> usize {
        self.expanded_values.ncols()
    }

    /// Get basis function names.
    pub fn basis_names(&self) -> Vec<String> {
        (0..self.n_basis())
            .map(|i| format!("{}_basis{}", self.name, i + 1))
            .collect()
    }
}

impl EventSpec for EventBasis {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Basis
    }

    fn onsets(&self) -> &OnsetArray {
        &self.onsets
    }

    fn durations(&self) -> &DurationArray {
        &self.durations
    }

    fn n_columns(&self) -> Option<usize> {
        Some(self.expanded_values.ncols())
    }
    fn column_names(&self) -> Vec<String> {
        self.basis_names()
    }

    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        Ok(self.expanded_values.clone())
    }

    fn design_matrix(
        &self,
        sampling_points: &crate::types::TimeArray,
    ) -> Result<Array, ValidationError> {
        let n_points = sampling_points.len();
        let n_basis = self.n_basis();

        // Initialize design matrix
        let mut design_matrix = Array2::zeros((n_points, n_basis));

        // Fill in expanded values at event times
        for (i, (onset, duration)) in self.onsets.iter().zip(self.durations.iter()).enumerate() {
            let exp_values = self.expanded_values.row(i);

            if *duration == 0.0 {
                // Impulse event - find nearest sampling point
                let nearest_idx = sampling_points
                    .iter()
                    .enumerate()
                    .min_by_key(|(_, &t)| ((t - onset).abs() * 1000.0) as i64)
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                for j in 0..n_basis {
                    design_matrix[[nearest_idx, j]] = exp_values[j];
                }
            } else {
                // Extended event - fill duration
                for (t_idx, &t) in sampling_points.iter().enumerate() {
                    if t >= *onset && t < onset + duration {
                        for j in 0..n_basis {
                            design_matrix[[t_idx, j]] = exp_values[j];
                        }
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
            .with_column(format!("{}_value", self.name), self.values.to_vec())
            .expect("Failed to add values column");

        // Add expanded basis columns
        let basis_names = self.basis_names();
        for (i, basis_name) in basis_names.iter().enumerate() {
            let col_values: Vec<f64> = (0..self.onsets.len())
                .map(|row| self.expanded_values[[row, i]])
                .collect();
            builder = builder
                .with_column(basis_name, col_values)
                .expect("Failed to add basis column");
        }

        builder.build()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EventSpec> {
        // Since BasisFunction doesn't implement Clone, we can't easily clone EventBasis
        // For now, return an error wrapped in a dummy event
        Box::new(
            crate::events::EventVariable::new(
                format!("{}_cloned", self.name),
                self.onsets.to_vec(),
                self.values.to_vec(),
                self.durations.to_vec(),
            )
            .unwrap_or_else(|_| {
                crate::events::EventVariable::new("error", vec![0.0], vec![0.0], vec![0.0]).unwrap()
            }),
        )
    }

    fn filter(&self, mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError> {
        if mask.len() != self.n_events() {
            return Err(ValidationError::LengthMismatch {
                expected: self.n_events(),
                actual: mask.len(),
            });
        }

        // Since BasisFunction doesn't implement Clone, convert to EventVariable
        let mut filtered_onsets = Vec::new();
        let mut filtered_values = Vec::new();
        let mut filtered_durations = Vec::new();

        for (i, &include) in mask.iter().enumerate() {
            if include {
                filtered_onsets.push(self.onsets[i]);
                filtered_values.push(self.values[i]);
                filtered_durations.push(self.durations[i]);
            }
        }

        if filtered_onsets.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Filter resulted in empty event".to_string(),
            ));
        }

        // Return as EventVariable since we can't clone the basis function
        Ok(Box::new(crate::events::EventVariable::new(
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
    use crate::basis::PolynomialBasis;
    use ndarray::array;

    #[test]
    fn test_event_basis_creation() {
        let basis = PolynomialBasis::new(2);
        let event = EventBasis::new(
            "rating",
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1.0, 2.0, 3.0, 4.0],
            Box::new(basis),
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .unwrap();

        assert_eq!(event.name(), "rating");
        assert_eq!(event.event_type(), EventType::Basis);
        assert_eq!(event.n_basis(), 2); // degree 2 with no intercept = 2 basis functions
    }

    #[test]
    fn test_design_matrix() {
        let basis = PolynomialBasis::new(1).with_raw(true).with_intercept(true); // Linear basis with intercept
        let event = EventBasis::new(
            "rating",
            vec![0.0, 2.0],
            vec![1.0, 2.0],
            Box::new(basis),
            vec![1.0, 1.0],
        )
        .unwrap();

        let sampling_points = array![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[7, 2]); // 2 basis functions (intercept + linear)

        // First event at t=0 with value=1
        assert_eq!(dm[[0, 0]], 1.0); // intercept
        assert_eq!(dm[[0, 1]], 1.0); // value
        assert_eq!(dm[[1, 0]], 1.0); // intercept
        assert_eq!(dm[[1, 1]], 1.0); // value

        // Second event at t=2 with value=2
        assert_eq!(dm[[4, 0]], 1.0); // intercept
        assert_eq!(dm[[4, 1]], 2.0); // value
        assert_eq!(dm[[5, 0]], 1.0); // intercept
        assert_eq!(dm[[5, 1]], 2.0); // value
    }
}
