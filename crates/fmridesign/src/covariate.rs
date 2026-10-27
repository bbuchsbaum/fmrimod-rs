//! Covariate support for fMRI models.
//!
//! Covariates are terms that are added directly to the design matrix without
//! HRF convolution. They are useful for including nuisance variables, motion
//! parameters, physiological measurements, or other regressors.

use crate::formula::Term;
use crate::types::{
    Array, DataFrame, DurationArray, EventSpec, EventType, OnsetArray, ValidationError,
};
use ndarray::Array1;

/// Event representation for covariate data.
///
/// This is a special event type that holds covariate data
/// for direct inclusion in the design matrix without HRF convolution.
#[derive(Debug, Clone)]
pub struct CovariateEvent {
    /// Covariate name
    pub name: String,
    /// Covariate values (columns are different covariates)
    pub values: Vec<Vec<f64>>,
    /// Names for each covariate column
    pub covariate_names: Vec<String>,
    /// Time points corresponding to values (optional)
    pub sampling_points: Option<Array1<f64>>,
    /// Empty arrays for EventSpec trait
    empty_onsets: OnsetArray,
    empty_durations: DurationArray,
}

impl CovariateEvent {
    /// Create a new covariate event.
    ///
    /// # Arguments
    /// * `name` - Covariate name
    /// * `values` - Covariate values (must match number of timepoints)
    /// * `sampling_points` - Time points corresponding to values (optional)
    ///
    /// # Example
    /// ```
    /// use fmridesign::covariate::CovariateEvent;
    /// use ndarray::array;
    ///
    /// // Motion parameter
    /// let motion_x = CovariateEvent::new(
    ///     "motion_x",
    ///     array![0.1, 0.2, -0.1, 0.0, 0.3],
    ///     None,
    /// ).unwrap();
    /// ```
    pub fn new(
        name: impl Into<String>,
        values: Array1<f64>,
        sampling_points: Option<Array1<f64>>,
    ) -> Result<Self, ValidationError> {
        let name = name.into();

        // Validate values are finite
        if !values.iter().all(|v| v.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }

        // If sampling points provided, validate length matches
        if let Some(ref sp) = sampling_points {
            if sp.len() != values.len() {
                return Err(ValidationError::LengthMismatch {
                    expected: values.len(),
                    actual: sp.len(),
                });
            }
        }

        Ok(Self {
            name: name.clone(),
            values: vec![values.to_vec()],
            covariate_names: vec![name],
            sampling_points,
            empty_onsets: OnsetArray::zeros(0),
            empty_durations: DurationArray::zeros(0),
        })
    }

    /// Create a multi-covariate event.
    pub fn new_multi(
        name: impl Into<String>,
        values: Vec<Vec<f64>>,
        covariate_names: Vec<String>,
        sampling_points: Option<Array1<f64>>,
    ) -> Result<Self, ValidationError> {
        let name = name.into();

        if values.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "No covariate values provided".to_string(),
            ));
        }

        if values.len() != covariate_names.len() {
            return Err(ValidationError::LengthMismatch {
                expected: covariate_names.len(),
                actual: values.len(),
            });
        }

        // All covariates must have same length
        let n_timepoints = values[0].len();
        for (i, vals) in values.iter().enumerate() {
            if vals.len() != n_timepoints {
                return Err(ValidationError::InvalidParameter(format!(
                    "Covariate {} has {} values, expected {}",
                    covariate_names[i],
                    vals.len(),
                    n_timepoints
                )));
            }

            // Validate values are finite
            if !vals.iter().all(|v| v.is_finite()) {
                return Err(ValidationError::NonFiniteValues);
            }
        }

        // If sampling points provided, validate length matches
        if let Some(ref sp) = sampling_points {
            if sp.len() != n_timepoints {
                return Err(ValidationError::LengthMismatch {
                    expected: n_timepoints,
                    actual: sp.len(),
                });
            }
        }

        Ok(Self {
            name,
            values,
            covariate_names,
            sampling_points,
            empty_onsets: OnsetArray::zeros(0),
            empty_durations: DurationArray::zeros(0),
        })
    }

    /// Number of timepoints.
    pub fn n_timepoints(&self) -> usize {
        if self.values.is_empty() {
            0
        } else {
            self.values[0].len()
        }
    }

    /// Number of covariates.
    pub fn n_covariates(&self) -> usize {
        self.values.len()
    }

    /// Interpolate values to match requested sampling points if needed.
    fn interpolate_to_sampling_points(
        &self,
        sampling_points: &Array1<f64>,
    ) -> Result<ndarray::Array2<f64>, ValidationError> {
        let n_samples = sampling_points.len();
        let n_covariates = self.values.len();
        let mut result = ndarray::Array2::zeros((n_samples, n_covariates));

        if let Some(ref sp) = self.sampling_points {
            if sp == sampling_points {
                // Exact match - no interpolation needed
                for (col, values) in self.values.iter().enumerate() {
                    for (row, &val) in values.iter().enumerate() {
                        result[[row, col]] = val;
                    }
                }
            } else {
                // Linear interpolation with boundary clamping
                let sp_slice = sp.as_slice().expect("contiguous sampling points");
                let n_src = sp_slice.len();

                for (col, values) in self.values.iter().enumerate() {
                    if n_src == 1 {
                        // Single source point: constant value everywhere
                        for i in 0..n_samples {
                            result[[i, col]] = values[0];
                        }
                        continue;
                    }

                    for (i, &target_time) in sampling_points.iter().enumerate() {
                        // Clamp: below range
                        if target_time <= sp_slice[0] {
                            result[[i, col]] = values[0];
                            continue;
                        }
                        // Clamp: above range
                        if target_time >= sp_slice[n_src - 1] {
                            result[[i, col]] = values[n_src - 1];
                            continue;
                        }

                        // Binary search for the interval containing target_time.
                        // partition_point returns the first index where sp >= target_time
                        // is false, i.e. the first index where sp > target_time... we
                        // actually want the upper bracket.
                        let upper = sp_slice.partition_point(|&t| t < target_time);
                        // upper is in 1..n_src because we already handled boundary cases
                        let lower = upper - 1;

                        let t_low = sp_slice[lower];
                        let t_high = sp_slice[upper];

                        if (t_high - t_low).abs() < f64::EPSILON {
                            // Degenerate: two identical time points
                            result[[i, col]] = values[lower];
                        } else {
                            let frac = (target_time - t_low) / (t_high - t_low);
                            result[[i, col]] =
                                values[lower] + (values[upper] - values[lower]) * frac;
                        }
                    }
                }
            }
        } else {
            // No explicit sampling points - assume values match requested points
            if self.n_timepoints() != n_samples {
                Err(ValidationError::LengthMismatch {
                    expected: n_samples,
                    actual: self.n_timepoints(),
                })?
            } else {
                for (col, values) in self.values.iter().enumerate() {
                    for (row, &val) in values.iter().enumerate() {
                        result[[row, col]] = val;
                    }
                }
            }
        }

        Ok(result)
    }
}

impl EventSpec for CovariateEvent {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Continuous // Covariates are treated as continuous
    }

    fn onsets(&self) -> &OnsetArray {
        // Covariates don't have traditional onsets
        &self.empty_onsets
    }

    fn durations(&self) -> &DurationArray {
        // Covariates don't have traditional durations
        &self.empty_durations
    }

    fn design_matrix(&self, sampling_points: &Array1<f64>) -> Result<Array, ValidationError> {
        // Get interpolated values
        self.interpolate_to_sampling_points(sampling_points)
    }

    fn column_names(&self) -> Vec<String> {
        self.covariate_names.clone()
    }

    fn to_dataframe(&self) -> DataFrame {
        use crate::dataframe_polars::DataFrameBuilder;

        // Build dataframe with all covariate columns
        let mut builder = DataFrameBuilder::new();

        for (i, (name, values)) in self.covariate_names.iter().zip(&self.values).enumerate() {
            builder = builder
                .with_column(name, values.clone())
                .expect(&format!("Failed to add covariate column {}", i));
        }

        builder.build()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EventSpec> {
        Box::new(self.clone())
    }

    fn filter(&self, _mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError> {
        // Covariates don't get filtered by time - they apply to the whole time series
        Ok(Box::new(self.clone()))
    }
}

/// Configuration for covariate terms.
#[derive(Debug, Clone)]
pub struct CovariateConfig {
    /// Covariate variable names
    pub covariates: Vec<String>,
    /// Optional prefix for column names
    pub prefix: Option<String>,
}

impl Default for CovariateConfig {
    fn default() -> Self {
        Self {
            covariates: Vec::new(),
            prefix: None,
        }
    }
}

/// Create a covariate term for non-convolved regressors.
///
/// Covariates are added directly to the design matrix without HRF convolution.
/// This is useful for nuisance regressors, motion parameters, physiological
/// signals, or other variables that should not be convolved.
///
/// # Examples
/// ```ignore
/// // Motion parameters
/// let motion_term = covariate(vec!["motion_x", "motion_y", "motion_z"])
///     .with_prefix("motion");
///
/// // In event model formula
/// let model = event_model(
///     "hrf(stimulus) + covariate(motion_x, motion_y)",
///     &data,
///     tr,
///     n_scans
/// )?;
/// ```
pub fn covariate(variables: Vec<String>) -> Term {
    covariate_with_config(CovariateConfig {
        covariates: variables,
        prefix: None,
    })
}

/// Create a covariate term with custom configuration.
pub fn covariate_with_config(config: CovariateConfig) -> Term {
    // Create a special term that will be expanded during model construction
    let mut term = Term::new("_covariate_placeholder_");

    // Store configuration as parameters
    term = term.set_param("_is_covariate", 1.0);

    // Store covariate names as a comma-separated string for now
    // (since we can't store string lists in TermParam directly)
    term = term.set_param("_covariate_names", config.covariates.join(","));

    // Generate name
    let name = if let Some(ref prefix) = &config.prefix {
        format!("{}_{}", prefix, config.covariates.join("_"))
    } else {
        config.covariates.join("_")
    };

    if let Some(prefix) = config.prefix {
        term = term.set_param("_covariate_prefix", prefix);
    }

    term.with_name(name)
}

/// Check if a term is a covariate term.
pub fn is_covariate_term(term: &Term) -> bool {
    term.get_param("_is_covariate")
        .map(|v| v > 0.5)
        .unwrap_or(false)
}

/// Get covariate configuration from a term.
pub fn get_covariate_config(term: &Term) -> Option<CovariateConfig> {
    if !is_covariate_term(term) {
        return None;
    }

    // Extract covariate names from the parameter
    let covariates = term
        .params
        .get("_covariate_names")
        .and_then(|p| {
            if let crate::formula::TermParam::String(s) = p {
                Some(s.split(',').map(|s| s.to_string()).collect())
            } else {
                None
            }
        })
        .unwrap_or_default();

    let prefix = term.params.get("_covariate_prefix").and_then(|p| {
        if let crate::formula::TermParam::String(s) = p {
            Some(s.clone())
        } else {
            None
        }
    });

    Some(CovariateConfig { covariates, prefix })
}

/// Create covariate events from data columns.
/// Create covariate events from columns in a DataFrame.
///
/// This function mirrors the behavior of R's `construct.covariatespec` by
/// validating that each covariate column has exactly `expected_n_timepoints`
/// rows, which should match the total number of scans implied by the
/// `SamplingFrame` in the surrounding model.
pub fn create_covariate_events(
    data: &crate::core::DataFrame,
    covariate_names: &[String],
    expected_n_timepoints: usize,
    _prefix: Option<&str>,
) -> Result<Vec<CovariateEvent>, ValidationError> {
    let mut events = Vec::new();

    for cov_name in covariate_names {
        // Get values from data
        let values = data.get_column(cov_name).ok_or_else(|| {
            ValidationError::InvalidParameter(format!("Covariate '{}' not found in data", cov_name))
        })?;

        // Validate length against the sampling frame expectation
        if values.len() != expected_n_timepoints {
            return Err(ValidationError::LengthMismatch {
                expected: expected_n_timepoints,
                actual: values.len(),
            });
        }

        // Use the covariate name directly - prefix will be handled by column naming
        let event_name = cov_name.clone();

        // Create covariate event
        let event = CovariateEvent::new(
            event_name,
            Array1::from(values.clone()),
            None, // No explicit sampling points
        )?;

        events.push(event);
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_covariate_event() {
        let values = array![0.1, 0.2, -0.1, 0.0, 0.3];
        let event = CovariateEvent::new("motion_x", values.clone(), None).unwrap();

        assert_eq!(event.name(), "motion_x");
        assert_eq!(event.n_timepoints(), 5);
        assert_eq!(event.n_covariates(), 1);
        assert_eq!(event.event_type(), EventType::Continuous);

        // Test design matrix generation
        let sampling_points = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[5, 1]);
        assert_eq!(dm.column(0), values.view());
    }

    #[test]
    fn test_multi_covariate_event() {
        let values = vec![
            vec![0.1, 0.2, -0.1, 0.0, 0.3],
            vec![0.5, -0.2, 0.1, 0.4, -0.3],
            vec![0.0, 0.1, 0.2, 0.3, 0.4],
        ];
        let names = vec![
            "motion_x".to_string(),
            "motion_y".to_string(),
            "motion_z".to_string(),
        ];

        let event = CovariateEvent::new_multi("motion", values.clone(), names, None).unwrap();

        assert_eq!(event.name(), "motion");
        assert_eq!(event.n_timepoints(), 5);
        assert_eq!(event.n_covariates(), 3);
        assert_eq!(
            event.column_names(),
            vec!["motion_x", "motion_y", "motion_z"]
        );

        // Test design matrix generation
        let sampling_points = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let dm = event.design_matrix(&sampling_points).unwrap();

        assert_eq!(dm.shape(), &[5, 3]);
        assert_eq!(dm[[0, 0]], 0.1);
        assert_eq!(dm[[0, 1]], 0.5);
        assert_eq!(dm[[0, 2]], 0.0);
    }

    #[test]
    fn test_covariate_validation() {
        // Non-finite values should fail
        let values = array![0.1, f64::NAN, 0.3];
        let result = CovariateEvent::new("bad", values, None);
        assert!(result.is_err());

        // Mismatched sampling points should fail
        let values = array![0.1, 0.2, 0.3];
        let sp = array![0.0, 1.0]; // Wrong length
        let result = CovariateEvent::new("bad", values, Some(sp));
        assert!(result.is_err());
    }

    #[test]
    fn test_covariate_term() {
        let term = covariate(vec!["motion_x".to_string(), "motion_y".to_string()]);

        assert!(is_covariate_term(&term));

        let config = get_covariate_config(&term).unwrap();
        assert_eq!(config.covariates, vec!["motion_x", "motion_y"]);
        assert_eq!(config.prefix, None);
    }

    #[test]
    fn test_covariate_with_prefix() {
        let mut config = CovariateConfig::default();
        config.covariates = vec!["x".to_string(), "y".to_string(), "z".to_string()];
        config.prefix = Some("motion".to_string());

        let term = covariate_with_config(config);

        assert!(is_covariate_term(&term));
        assert_eq!(term.name, Some("motion_x_y_z".to_string()));

        let retrieved_config = get_covariate_config(&term).unwrap();
        assert_eq!(retrieved_config.prefix, Some("motion".to_string()));
    }

    // --- Linear interpolation tests ---

    #[test]
    fn test_interpolation_exact_match() {
        // When target sampling points exactly equal source, no interpolation needed
        let sp = array![0.0, 1.0, 2.0, 3.0];
        let values = array![10.0, 20.0, 30.0, 40.0];
        let event = CovariateEvent::new("v", values.clone(), Some(sp.clone())).unwrap();

        let dm = event.interpolate_to_sampling_points(&sp).unwrap();
        assert_eq!(dm.shape(), &[4, 1]);
        for i in 0..4 {
            assert_eq!(dm[[i, 0]], values[i]);
        }
    }

    #[test]
    fn test_interpolation_simple_linear() {
        // Two source points: 0.0 -> 0.0, 2.0 -> 10.0
        // Interpolate at 0.5, 1.0, 1.5
        let sp = array![0.0, 2.0];
        let values = array![0.0, 10.0];
        let event = CovariateEvent::new("v", values, Some(sp)).unwrap();

        let target = array![0.5, 1.0, 1.5];
        let dm = event.interpolate_to_sampling_points(&target).unwrap();

        assert!((dm[[0, 0]] - 2.5).abs() < 1e-12);
        assert!((dm[[1, 0]] - 5.0).abs() < 1e-12);
        assert!((dm[[2, 0]] - 7.5).abs() < 1e-12);
    }

    #[test]
    fn test_interpolation_extrapolation_clamping() {
        // Source: 1.0 -> 100.0, 3.0 -> 200.0
        // Targets outside range should clamp
        let sp = array![1.0, 3.0];
        let values = array![100.0, 200.0];
        let event = CovariateEvent::new("v", values, Some(sp)).unwrap();

        let target = array![0.0, 1.0, 2.0, 3.0, 5.0];
        let dm = event.interpolate_to_sampling_points(&target).unwrap();

        assert_eq!(dm[[0, 0]], 100.0); // below range -> clamp to first
        assert_eq!(dm[[1, 0]], 100.0); // exact match on lower bound
        assert!((dm[[2, 0]] - 150.0).abs() < 1e-12); // midpoint
        assert_eq!(dm[[3, 0]], 200.0); // exact match on upper bound
        assert_eq!(dm[[4, 0]], 200.0); // above range -> clamp to last
    }

    #[test]
    fn test_interpolation_multi_covariate() {
        // Two covariates, three source points
        let sp = array![0.0, 1.0, 2.0];
        let values = vec![
            vec![0.0, 10.0, 20.0],  // covariate A: linear 0..20
            vec![100.0, 50.0, 0.0], // covariate B: linear 100..0
        ];
        let names = vec!["A".to_string(), "B".to_string()];
        let event = CovariateEvent::new_multi("cov", values, names, Some(sp)).unwrap();

        let target = array![0.5, 1.5];
        let dm = event.interpolate_to_sampling_points(&target).unwrap();

        assert_eq!(dm.shape(), &[2, 2]);
        assert!((dm[[0, 0]] - 5.0).abs() < 1e-12); // A at 0.5
        assert!((dm[[1, 0]] - 15.0).abs() < 1e-12); // A at 1.5
        assert!((dm[[0, 1]] - 75.0).abs() < 1e-12); // B at 0.5
        assert!((dm[[1, 1]] - 25.0).abs() < 1e-12); // B at 1.5
    }

    #[test]
    fn test_interpolation_single_source_point() {
        // Single source point should produce constant output
        let sp = array![5.0];
        let values = array![42.0];
        let event = CovariateEvent::new("v", values, Some(sp)).unwrap();

        let target = array![0.0, 5.0, 10.0];
        let dm = event.interpolate_to_sampling_points(&target).unwrap();

        assert_eq!(dm[[0, 0]], 42.0);
        assert_eq!(dm[[1, 0]], 42.0);
        assert_eq!(dm[[2, 0]], 42.0);
    }
}
