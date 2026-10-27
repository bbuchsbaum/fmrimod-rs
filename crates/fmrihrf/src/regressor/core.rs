//! Core Regressor classes for fMRI event modeling.

use crate::error::{HrfError, Result};
use crate::hrf::Hrf;
use crate::regressor::convolution::{convolve_hrf, convolve_hrf_trial_varying};
use crate::types::HrfValues;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// Represents event-related regressors for fMRI modeling.
///
/// This struct defines event onsets and associates them with a hemodynamic
/// response function (HRF) to generate predicted time courses.
#[derive(Debug, Clone)]
pub struct Regressor {
    /// Event onset times in seconds
    pub onsets: Vec<f64>,
    /// Hemodynamic response function
    pub hrf: Arc<Box<dyn Hrf>>,
    /// Event durations in seconds
    pub durations: Vec<f64>,
    /// Event amplitudes/scaling factors
    pub amplitudes: Vec<f64>,
    /// HRF temporal span in seconds
    pub span: f64,
    /// Whether block responses are integrated (true) or duration-averaged (false)
    pub summate: bool,
    /// Whether all events were filtered out (zero/NA amplitudes)
    pub filtered_all: bool,
    /// Optional per-event HRFs for trial-varying models.
    /// When Some, each event uses its own HRF instead of `self.hrf`.
    /// All HRFs must have the same nbasis.
    pub per_event_hrfs: Option<Vec<Arc<Box<dyn Hrf>>>>,
}

/// Serializable representation of a Regressor.
///
/// This struct is used for serializing/deserializing Regressor instances,
/// storing the HRF name instead of the trait object.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegressorData {
    /// Explicit portable HRF recipe. Missing legacy/name-only data cannot restore safely.
    #[serde(default)]
    pub hrf_recipe: Option<crate::hrf::HrfRecipe>,
    /// True when event-specific HRFs cannot be represented by the common recipe.
    #[serde(default)]
    pub heterogeneous: bool,
    /// Event onset times in seconds
    pub onsets: Vec<f64>,
    /// HRF name for registry lookup
    pub hrf_name: String,
    /// Event durations in seconds
    pub durations: Vec<f64>,
    /// Event amplitudes/scaling factors
    pub amplitudes: Vec<f64>,
    /// HRF temporal span in seconds
    pub span: f64,
    /// Whether block responses are integrated (true) or duration-averaged (false)
    pub summate: bool,
    /// Whether all events were filtered out
    pub filtered_all: bool,
}

impl Regressor {
    /// Create a new Regressor.
    pub fn new(
        onsets: Vec<f64>,
        hrf: Box<dyn Hrf>,
        durations: Option<Vec<f64>>,
        amplitudes: Option<Vec<f64>>,
        span: Option<f64>,
        summate: Option<bool>,
    ) -> Result<Self> {
        let effective_span = span.unwrap_or_else(|| hrf.span());
        let mut regressor = Self {
            onsets,
            hrf: Arc::new(hrf),
            durations: durations.unwrap_or_else(|| vec![0.0]),
            amplitudes: amplitudes.unwrap_or_else(|| vec![1.0]),
            span: effective_span,
            summate: summate.unwrap_or(true),
            filtered_all: false,
            per_event_hrfs: None,
        };

        regressor.validate_and_process()?;
        Ok(regressor)
    }

    /// Validate and process inputs.
    fn validate_and_process(&mut self) -> Result<()> {
        // Handle single NA onset (represents zero events)
        if self.onsets.len() == 1 && self.onsets[0].is_nan() {
            self.onsets.clear();
        }

        // Validate span
        if !self.span.is_finite() || self.span <= 0.0 {
            return Err(HrfError::InvalidInput(
                "span must be a positive, finite number".into(),
            ));
        }

        let n_onsets = self.onsets.len();

        // Recycle duration and amplitude
        if n_onsets > 0 {
            self.durations = recycle_or_error(&self.durations, n_onsets, "duration")?;
            self.amplitudes = recycle_or_error(&self.amplitudes, n_onsets, "amplitude")?;
        }

        // Filter events based on non-zero and non-NA amplitude
        if n_onsets > 0 {
            let mut keep_indices = Vec::new();
            for (i, &amp) in self.amplitudes.iter().enumerate() {
                if amp != 0.0 && !amp.is_nan() {
                    keep_indices.push(i);
                }
            }

            if self.amplitudes.iter().any(|a| a.is_infinite()) {
                return Err(HrfError::InvalidInput(
                    "amplitudes must be finite or NaN (filtered)".into(),
                ));
            }
            self.filtered_all = keep_indices.is_empty();

            if keep_indices.len() < n_onsets {
                // Filter vectors
                self.onsets = keep_indices.iter().map(|&i| self.onsets[i]).collect();
                self.durations = keep_indices.iter().map(|&i| self.durations[i]).collect();
                self.amplitudes = keep_indices.iter().map(|&i| self.amplitudes[i]).collect();
                // Filter per-event HRFs if present
                if let Some(ref hrfs) = self.per_event_hrfs {
                    self.per_event_hrfs =
                        Some(keep_indices.iter().map(|&i| Arc::clone(&hrfs[i])).collect());
                }
            }
        } else {
            self.filtered_all = true;
        }

        // Validate after filtering
        if !self.onsets.is_empty() {
            for &onset in &self.onsets {
                if onset.is_nan() || !onset.is_finite() {
                    return Err(HrfError::InvalidInput(
                        "onsets must contain finite numeric values".into(),
                    ));
                }
                if onset < 0.0 {
                    return Err(HrfError::InvalidInput("onsets must be non-negative".into()));
                }
            }

            for &duration in &self.durations {
                if duration.is_nan() || !duration.is_finite() {
                    return Err(HrfError::InvalidInput(
                        "duration must contain finite numeric values".into(),
                    ));
                }
                if duration < 0.0 {
                    return Err(HrfError::InvalidInput("duration cannot be negative".into()));
                }
            }
        }

        Ok(())
    }

    /// Create a regressor with per-event (trial-varying) HRFs.
    ///
    /// Each event can have a different HRF. All HRFs must have the same `nbasis()`.
    /// The first HRF is used for metadata (name, span) and as the default `self.hrf`.
    ///
    /// # Arguments
    /// * `onsets` - Event onset times in seconds
    /// * `hrfs` - One HRF per event, or a single HRF to broadcast to all events
    /// * `durations` - Event durations (scalar or per-event; default 0)
    /// * `amplitudes` - Event amplitudes (scalar or per-event; default 1)
    /// * `span` - HRF temporal span in seconds (default: from first HRF)
    /// * `summate` - Whether overlapping responses sum (default: true)
    pub fn new_trial_varying(
        onsets: Vec<f64>,
        hrfs: Vec<Arc<Box<dyn Hrf>>>,
        durations: Option<Vec<f64>>,
        amplitudes: Option<Vec<f64>>,
        span: Option<f64>,
        summate: Option<bool>,
    ) -> Result<Self> {
        if hrfs.is_empty() {
            return Err(HrfError::InvalidInput(
                "hrfs must contain at least one HRF".into(),
            ));
        }

        let metadata_hrf = Arc::clone(&hrfs[0]);
        let effective_span = span.unwrap_or_else(|| metadata_hrf.span());
        let n_onsets = onsets.len();

        // Broadcast or validate length
        let per_event_hrfs = if hrfs.len() == 1 {
            vec![Arc::clone(&hrfs[0]); n_onsets]
        } else if hrfs.len() == n_onsets {
            hrfs
        } else {
            return Err(HrfError::InvalidInput(format!(
                "Length of hrfs ({}) must be 1 or match number of onsets ({})",
                hrfs.len(),
                n_onsets
            )));
        };

        // Validate all HRFs have the same nbasis
        let expected_nbasis = metadata_hrf.nbasis();
        for (i, hrf) in per_event_hrfs.iter().enumerate() {
            if hrf.nbasis() != expected_nbasis {
                return Err(HrfError::InvalidInput(format!(
                    "All trial-varying HRFs must have the same nbasis. \
                     HRF 0 has nbasis={}, but HRF {} has nbasis={}",
                    expected_nbasis,
                    i,
                    hrf.nbasis()
                )));
            }
        }

        // Use the first HRF as the default/metadata HRF
        let mut regressor = Self {
            onsets,
            hrf: metadata_hrf,
            durations: durations.unwrap_or_else(|| vec![0.0]),
            amplitudes: amplitudes.unwrap_or_else(|| vec![1.0]),
            span: effective_span,
            summate: summate.unwrap_or(true),
            filtered_all: false,
            per_event_hrfs: Some(per_event_hrfs),
        };

        regressor.validate_and_process()?;
        Ok(regressor)
    }

    /// Evaluate the regressor at specified time points.
    ///
    /// # Arguments
    /// * `grid` - Time points at which to evaluate (seconds)
    /// * `precision` - Temporal precision for convolution (seconds)
    /// * `method` - Evaluation method ('conv', 'fft', 'direct')
    ///
    /// # Returns
    /// Evaluated regressor values at grid points
    pub fn evaluate(
        &self,
        grid: &Array1<f64>,
        precision: Option<f64>,
        method: Option<&str>,
    ) -> Result<HrfValues> {
        let precision = precision.unwrap_or(0.33);
        let method = method.unwrap_or("conv");

        use crate::regressor::convolution::ConvolutionMethod;
        use std::borrow::Cow;
        let _: ConvolutionMethod = method.try_into()?;
        if grid.is_empty() || grid.iter().any(|x| !x.is_finite()) {
            return Err(HrfError::InvalidInput(
                "grid must be nonempty and finite".into(),
            ));
        }
        if !precision.is_finite() || precision <= 0.0 || !self.span.is_finite() || self.span <= 0.0
        {
            return Err(HrfError::InvalidInput(
                "precision and span must be positive and finite".into(),
            ));
        }
        let sorted_grid = if grid.iter().zip(grid.iter().skip(1)).all(|(a, b)| a <= b) {
            Cow::Borrowed(grid)
        } else {
            let mut sorted = grid.to_vec();
            sorted.sort_by(f64::total_cmp);
            Cow::Owned(Array1::from_vec(sorted))
        };
        if self.onsets.is_empty() {
            return Ok(Array2::zeros((grid.len(), self.hrf.nbasis())));
        }
        if let Some(hrfs) = &self.per_event_hrfs {
            convolve_hrf_trial_varying(
                &sorted_grid,
                &self.onsets,
                &self.durations,
                &self.amplitudes,
                hrfs,
                self.span,
                precision,
                self.summate,
            )
        } else {
            convolve_hrf(
                &sorted_grid,
                &self.onsets,
                &self.durations,
                &self.amplitudes,
                self.hrf.as_ref().as_ref(),
                self.span,
                precision,
                method,
                self.summate,
            )
        }
    }

    /// Shift all event onsets by a temporal offset.
    ///
    /// # Arguments
    /// * `shift_amount` - Amount to shift in seconds (positive = forward)
    ///
    /// # Returns
    /// New Regressor with shifted onsets
    pub fn shift(&self, shift_amount: f64) -> Self {
        Self {
            onsets: self.onsets.iter().map(|&t| t + shift_amount).collect(),
            hrf: Arc::clone(&self.hrf),
            durations: self.durations.clone(),
            amplitudes: self.amplitudes.clone(),
            span: self.span,
            summate: self.summate,
            filtered_all: self.filtered_all,
            per_event_hrfs: self.per_event_hrfs.clone(),
        }
    }

    /// Generate neural input time series from events.
    ///
    /// Creates a boxcar time series representing neural activity,
    /// where each event contributes its amplitude during its duration.
    /// This is the neural input signal before HRF convolution.
    ///
    /// # Arguments
    /// * `start` - Start time in seconds (default: 0.0)
    /// * `end` - End time in seconds (if None, auto-determined from events)
    /// * `resolution` - Time resolution in seconds (default: 0.33)
    ///
    /// # Returns
    /// HashMap with 'time' and 'neural_input' arrays
    ///
    /// # Example
    /// ```
    /// use fmrihrf::regressor::regressor;
    /// use fmrihrf::hrf::spm_canonical;
    ///
    /// let hrf = spm_canonical(None, None, None, None, None, Some(true));
    /// let reg = regressor(vec![1.0, 3.0, 5.0], Some(hrf), Some(vec![0.5]), None, None, None).unwrap();
    ///
    /// // Generate neural input signal
    /// let neural = reg.neural_input(None, None, None).unwrap();
    /// let time = &neural["time"];
    /// let signal = &neural["neural_input"];
    /// ```
    pub fn neural_input(
        &self,
        start: Option<f64>,
        end: Option<f64>,
        resolution: Option<f64>,
    ) -> Result<std::collections::HashMap<String, Array1<f64>>> {
        crate::regressor::neural_input::neural_input(self, start, end, resolution)
    }

    /// Construct portable data, rejecting unsupported/custom or event-specific HRFs.
    #[cfg(feature = "serde")]
    pub fn try_to_data(&self) -> Result<RegressorData> {
        let data = self.to_data();
        if data.hrf_recipe.is_none() || data.heterogeneous {
            return Err(HrfError::InvalidInput(
                "HRF has no supported portable recipe".into(),
            ));
        }
        Ok(data)
    }

    /// Convert to data; unsupported HRFs carry no recipe and cannot be restored.
    /// Prefer `try_to_data` to detect this before writing the artifact.
    #[cfg(feature = "serde")]
    pub fn to_data(&self) -> RegressorData {
        RegressorData {
            hrf_recipe: self.hrf.recipe(),
            heterogeneous: self.per_event_hrfs.is_some(),
            onsets: self.onsets.clone(),
            hrf_name: self.hrf.name().to_string(),
            durations: self.durations.clone(),
            amplitudes: self.amplitudes.clone(),
            span: self.span,
            summate: self.summate,
            filtered_all: self.filtered_all,
        }
    }

    /// Create from serializable data representation.
    #[cfg(feature = "serde")]
    pub fn from_data(data: RegressorData) -> Result<Self> {
        if data.heterogeneous {
            return Err(HrfError::InvalidInput(
                "event-specific HRF serialization is not supported".into(),
            ));
        }
        let hrf = data
            .hrf_recipe
            .ok_or_else(|| {
                HrfError::InvalidInput(
                    "missing portable HRF recipe; name-only restoration is unsafe".into(),
                )
            })?
            .build()?;
        let mut reg = Self::new(
            data.onsets,
            hrf,
            Some(data.durations),
            Some(data.amplitudes),
            Some(data.span),
            Some(data.summate),
        )?;
        reg.filtered_all = data.filtered_all;
        Ok(reg)
    }
}

/// Create an empty regressor with no events.
///
/// This is useful as a placeholder when a condition has no events in a block.
/// Matches R's `null_regressor()`.
///
/// # Arguments
/// * `hrf` - HRF to associate with the regressor
/// * `span` - HRF temporal span in seconds
///
/// # Examples
/// ```
/// use fmrihrf::hrf::spm_canonical;
/// use fmrihrf::regressor::null_regressor;
/// use ndarray::Array1;
///
/// let hrf = spm_canonical(None, None, None, None, None, Some(true));
/// let reg = null_regressor(hrf, 24.0).unwrap();
/// assert_eq!(reg.onsets.len(), 0);
///
/// // Evaluating returns all zeros
/// let grid = Array1::linspace(0.0, 10.0, 11);
/// let result = reg.evaluate(&grid, None, None).unwrap();
/// assert!(result.iter().all(|&v| v.abs() < 1e-10));
/// ```
pub fn null_regressor(hrf: Box<dyn Hrf>, span: f64) -> Result<Regressor> {
    Regressor::new(
        vec![],
        hrf,
        Some(vec![0.0]),
        Some(vec![0.0]),
        Some(span),
        None,
    )
}

/// Create a Regressor object for modeling fMRI events.
///
/// A regressor represents the expected BOLD response to a series of experimental
/// events, created by convolving event timings with a hemodynamic response function.
/// This is the main building block for fMRI design matrices.
///
/// # Arguments
/// * `onsets` - Event onset times in seconds
/// * `hrf` - HRF object (defaults to SPM canonical if None)
/// * `duration` - Event duration(s) in seconds (scalar or per-event)
/// * `amplitude` - Event amplitude(s) (scalar or per-event)
/// * `span` - HRF temporal span in seconds (default: 40.0)
/// * `summate` - Whether overlapping responses sum (default: true)
///
/// # Examples
///
/// ## Basic Event-Related Design
/// ```
/// use fmrihrf::regressor::regressor;
/// use fmrihrf::hrf::spm_canonical;
/// use ndarray::Array1;
///
/// // Create a regressor for visual stimuli presented at specific times
/// let stimulus_onsets = vec![10.0, 30.0, 50.0, 70.0];
/// let hrf = spm_canonical(None, None, None, None, None, Some(true));
///
/// let reg = regressor(
///     stimulus_onsets,
///     Some(hrf),
///     None,  // Default duration (0 = impulse)
///     None,  // Default amplitude (1.0)
///     None,  // Default span
///     None,  // Default summation
/// ).unwrap();
///
/// // Evaluate at TR=2s for 100 scans
/// let times = Array1::linspace(0.0, 198.0, 100);
/// let predicted_signal = reg.evaluate(&times, Some(0.1), None).unwrap();
/// ```
///
/// ## Block Design with Variable Durations
/// ```
/// use fmrihrf::regressor::regressor;
///
/// // 20-second blocks of different conditions
/// let block_onsets = vec![20.0, 60.0, 100.0, 140.0];
/// let block_durations = vec![20.0, 20.0, 20.0, 20.0];
///
/// let reg = regressor(
///     block_onsets,
///     None,  // Use default SPM canonical HRF
///     Some(block_durations),
///     None,
///     None,
///     None,
/// ).unwrap();
/// ```
///
/// ## Parametric Modulation
/// ```
/// use fmrihrf::regressor::regressor;
///
/// // Events with varying intensities
/// let onsets = vec![10.0, 20.0, 30.0, 40.0];
/// let intensities = vec![0.5, 1.0, 1.5, 2.0];  // Parametric weights
///
/// let reg = regressor(
///     onsets,
///     None,
///     None,
///     Some(intensities),  // Modulate amplitude by intensity
///     None,
///     None,
/// ).unwrap();
/// ```
pub fn regressor(
    onsets: Vec<f64>,
    hrf: Option<Box<dyn Hrf>>,
    duration: Option<Vec<f64>>,
    amplitude: Option<Vec<f64>>,
    span: Option<f64>,
    summate: Option<bool>,
) -> Result<Regressor> {
    let hrf = hrf.unwrap_or_else(|| {
        crate::hrf::functions::spm_canonical(None, None, None, None, None, None)
    });
    Regressor::new(onsets, hrf, duration, amplitude, span, summate)
}

/// A set of regressors for multiple conditions.
///
/// Each condition shares the same HRF but has distinct onsets,
/// durations, and amplitudes.
#[derive(Debug, Clone)]
pub struct RegressorSet {
    /// List of Regressor objects, one per condition
    pub regressors: Vec<Regressor>,
    /// Condition names/labels
    pub levels: Vec<String>,
}

/// Serializable representation of a RegressorSet.
#[cfg(feature = "serde")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegressorSetData {
    /// List of serializable regressor data
    pub regressors: Vec<RegressorData>,
    /// Condition names/labels
    pub levels: Vec<String>,
}

impl RegressorSet {
    /// Evaluate all regressors to create design matrix.
    ///
    /// # Arguments
    /// * `grid` - Time points at which to evaluate
    /// * `precision` - Temporal precision
    /// * `method` - Evaluation method
    ///
    /// # Returns
    /// Design matrix with columns for each regressor
    pub fn evaluate(
        &self,
        grid: &Array1<f64>,
        precision: Option<f64>,
        method: Option<&str>,
    ) -> Result<Array2<f64>> {
        let mut results = Vec::new();

        // Evaluate each regressor
        for reg in &self.regressors {
            let result = reg.evaluate(grid, precision, method)?;

            // Result should already be 2D from evaluate method
            let result_2d = result;

            results.push(result_2d);
        }

        // Combine results
        if results.is_empty() {
            Ok(Array2::zeros((grid.len(), 0)))
        } else {
            // Concatenate along columns
            let views: Vec<_> = results.iter().map(|arr| arr.view()).collect();
            Ok(ndarray::concatenate(ndarray::Axis(1), &views).unwrap())
        }
    }

    /// Construct portable data, rejecting unsupported/custom or event-specific HRFs.
    #[cfg(feature = "serde")]
    pub fn try_to_data(&self) -> Result<RegressorSetData> {
        Ok(RegressorSetData {
            regressors: self
                .regressors
                .iter()
                .map(Regressor::try_to_data)
                .collect::<Result<Vec<_>>>()?,
            levels: self.levels.clone(),
        })
    }

    /// Convert to data; unsupported HRFs carry no recipe and cannot be restored.
    /// Prefer `try_to_data` to detect this before writing the artifact.
    #[cfg(feature = "serde")]
    pub fn to_data(&self) -> RegressorSetData {
        RegressorSetData {
            regressors: self.regressors.iter().map(|r| r.to_data()).collect(),
            levels: self.levels.clone(),
        }
    }

    /// Create from serializable data representation.
    #[cfg(feature = "serde")]
    pub fn from_data(data: RegressorSetData) -> Result<Self> {
        let regressors = data
            .regressors
            .into_iter()
            .map(Regressor::from_data)
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            regressors,
            levels: data.levels,
        })
    }
}

/// Create a set of regressors for multiple conditions.
///
/// This function creates a collection of regressors, one for each unique condition
/// in a factorial design. All conditions share the same HRF but have different
/// event timings. This is the standard way to model multiple experimental conditions
/// in fMRI.
///
/// # Arguments
/// * `onsets` - Event onset times for all conditions combined
/// * `fac` - Condition label for each onset (must match length of onsets)
/// * `hrf` - HRF to use for all conditions (defaults to SPM canonical)
/// * `duration` - Event duration(s) - can be scalar or per-event
/// * `amplitude` - Event amplitude(s) - can be scalar or per-event
/// * `span` - HRF temporal span
/// * `summate` - Whether overlapping responses sum
///
/// # Examples
///
/// ## Simple Two-Condition Design
/// ```
/// use fmrihrf::regressor::regressor_set;
/// use ndarray::Array1;
///
/// // Face and house stimuli presented in random order
/// let all_onsets = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0];
/// let conditions = vec![
///     "face".to_string(), "house".to_string(), "face".to_string(),
///     "house".to_string(), "face".to_string(), "house".to_string()
/// ];
///
/// let rset = regressor_set(
///     all_onsets,
///     conditions,
///     None,  // Default HRF
///     None,  // Default durations
///     None,  // Default amplitudes
///     None,  // Default span
///     None,  // Default summation
/// ).unwrap();
///
/// // Creates two regressors: one for "face", one for "house"
/// assert_eq!(rset.levels, vec!["face", "house"]);
/// assert_eq!(rset.regressors.len(), 2);
/// ```
///
/// ## Multi-Condition with Block Design
/// ```
/// use fmrihrf::regressor::regressor_set;
///
/// // 3x3 factorial design: emotion (happy/sad/neutral) x face type (human/animal/object)
/// let onsets = vec![
///     10.0, 30.0, 50.0, 70.0, 90.0, 110.0, 130.0, 150.0, 170.0
/// ];
/// let conditions = vec![
///     "happy_human", "sad_human", "neutral_human",
///     "happy_animal", "sad_animal", "neutral_animal",
///     "happy_object", "sad_object", "neutral_object"
/// ].iter().map(|s| s.to_string()).collect();
///
/// // 10-second blocks
/// let durations = vec![10.0; 9];
///
/// let rset = regressor_set(
///     onsets,
///     conditions,
///     None,
///     Some(durations),
///     None,
///     None,
///     None,
/// ).unwrap();
///
/// // Creates 9 regressors, one for each condition combination
/// assert_eq!(rset.regressors.len(), 9);
/// ```
pub fn regressor_set(
    onsets: Vec<f64>,
    fac: Vec<String>,
    hrf: Option<Box<dyn Hrf>>,
    duration: Option<Vec<f64>>,
    amplitude: Option<Vec<f64>>,
    span: Option<f64>,
    summate: Option<bool>,
) -> Result<RegressorSet> {
    if onsets.len() != fac.len() {
        return Err(HrfError::InvalidInput(
            "onsets and fac must have the same length".into(),
        ));
    }

    // Get unique levels
    let mut levels: Vec<String> = fac.to_vec();
    levels.sort();
    levels.dedup();

    // Recycle duration and amplitude
    let durations = if let Some(d) = duration {
        recycle_or_error(&d, onsets.len(), "duration")?
    } else {
        vec![0.0; onsets.len()]
    };

    let amplitudes = if let Some(a) = amplitude {
        recycle_or_error(&a, onsets.len(), "amplitude")?
    } else {
        vec![1.0; onsets.len()]
    };

    // Share the actual supplied HRF, including its parameters and custom evaluator.
    let hrf = Arc::new(hrf.unwrap_or_else(|| {
        crate::hrf::functions::spm_canonical(None, None, None, None, None, None)
    }));
    let effective_span = span.unwrap_or_else(|| hrf.span());
    let mut regressors = Vec::with_capacity(levels.len());
    for level in &levels {
        let mut level_onsets = Vec::new();
        let mut level_durations = Vec::new();
        let mut level_amplitudes = Vec::new();

        for (i, f) in fac.iter().enumerate() {
            if f == level {
                level_onsets.push(onsets[i]);
                level_durations.push(durations[i]);
                level_amplitudes.push(amplitudes[i]);
            }
        }

        let mut reg = Regressor {
            onsets: level_onsets,
            hrf: Arc::clone(&hrf),
            durations: level_durations,
            amplitudes: level_amplitudes,
            span: effective_span,
            summate: summate.unwrap_or(true),
            filtered_all: false,
            per_event_hrfs: None,
        };
        reg.validate_and_process()?;

        regressors.push(reg);
    }

    Ok(RegressorSet { regressors, levels })
}

/// Recycle scalar to array or validate array length.
fn recycle_or_error(values: &[f64], target_length: usize, name: &str) -> Result<Vec<f64>> {
    if values.is_empty() {
        return Err(HrfError::InvalidInput(format!("{} cannot be empty", name)));
    }

    if values.len() == 1 {
        // Recycle scalar
        Ok(vec![values[0]; target_length])
    } else if values.len() == target_length {
        Ok(values.to_vec())
    } else {
        Err(HrfError::InvalidInput(format!(
            "Length of {} ({}) must match number of onsets ({})",
            name,
            values.len(),
            target_length
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::functions::spm_canonical;

    #[test]
    fn test_regressor_basic() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![10.0, 30.0, 50.0], Some(hrf), None, None, None, None).unwrap();

        assert_eq!(reg.onsets.len(), 3);
        assert_eq!(reg.durations.len(), 3);
        assert_eq!(reg.amplitudes.len(), 3);
    }

    #[test]
    fn test_regressor_filtering() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(
            vec![10.0, 30.0, 50.0],
            Some(hrf),
            None,
            Some(vec![1.0, 0.0, 1.0]), // Middle event has zero amplitude
            None,
            None,
        )
        .unwrap();

        assert_eq!(reg.onsets.len(), 2); // One event filtered out
        assert_eq!(reg.onsets[0], 10.0);
        assert_eq!(reg.onsets[1], 50.0);
    }

    #[test]
    fn test_regressor_evaluate() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![5.0, 15.0], Some(hrf), None, None, None, None).unwrap();

        let grid = Array1::linspace(0.0, 30.0, 301);
        let result = reg.evaluate(&grid, None, None).unwrap();

        assert_eq!(result.nrows(), 301);
        assert_eq!(result.ncols(), 1); // Single basis function
    }

    #[test]
    fn test_regressor_shift() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![10.0, 30.0, 50.0], Some(hrf), None, None, None, None).unwrap();

        let shifted = reg.shift(2.0);
        assert_eq!(shifted.onsets[0], 12.0);
        assert_eq!(shifted.onsets[1], 32.0);
        assert_eq!(shifted.onsets[2], 52.0);
    }

    #[test]
    fn test_regressor_set() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let rset = regressor_set(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
            Some(hrf),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(rset.levels.len(), 3);
        assert_eq!(rset.regressors.len(), 3);

        // Check that each regressor has 2 events
        for reg in &rset.regressors {
            assert_eq!(reg.onsets.len(), 2);
        }
    }

    #[test]
    fn test_regressor_neural_input() {
        use approx::assert_relative_eq;

        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(
            vec![1.0, 3.0], // Two events
            Some(hrf),
            Some(vec![1.0, 1.0]), // Each lasting 1 second
            Some(vec![1.0, 2.0]), // Different amplitudes
            None,
            None,
        )
        .unwrap();

        // Generate neural input
        let result = reg.neural_input(Some(0.0), Some(6.0), Some(0.1)).unwrap();

        assert!(result.contains_key("time"));
        assert!(result.contains_key("neural_input"));

        let time = &result["time"];
        let neural = &result["neural_input"];

        assert_eq!(time.len(), neural.len());

        // Check amplitude during first event (t=1 to t=2, amplitude=1)
        let idx_1_5 = ((1.5 / 0.1) as usize).min(neural.len() - 1);
        assert_relative_eq!(neural[idx_1_5], 1.0, epsilon = 0.1);

        // Check amplitude during second event (t=3 to t=4, amplitude=2)
        let idx_3_5 = ((3.5 / 0.1) as usize).min(neural.len() - 1);
        assert_relative_eq!(neural[idx_3_5], 2.0, epsilon = 0.1);

        // Check zero outside events
        assert_eq!(neural[0], 0.0);
        assert_eq!(neural[neural.len() - 1], 0.0);
    }

    #[test]
    fn test_null_regressor() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = null_regressor(hrf, 24.0).unwrap();
        assert_eq!(reg.onsets.len(), 0);
        assert_eq!(reg.durations.len(), 1); // default single value kept
        assert!(reg.filtered_all);

        // Evaluating should return zeros
        let grid = Array1::linspace(0.0, 10.0, 11);
        let result = reg.evaluate(&grid, None, None).unwrap();
        assert!(result.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_null_regressor_in_design() {
        // Ensure null regressor integrates with design matrix generation
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = null_regressor(hrf, 24.0).unwrap();

        let grid = Array1::linspace(0.0, 30.0, 31);
        let result = reg.evaluate(&grid, Some(0.33), Some("conv")).unwrap();
        assert_eq!(result.nrows(), 31);
        assert_eq!(result.ncols(), 1);
        assert!(result.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn test_regressor_neural_input_impulse() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(
            vec![1.0, 2.0, 3.0], // Three impulse events
            Some(hrf),
            None, // Zero duration (impulses)
            None, // Unit amplitude
            None,
            None,
        )
        .unwrap();

        // Generate neural input with default parameters
        let result = reg.neural_input(None, None, None).unwrap();

        let neural = &result["neural_input"];

        // Count non-zero values (should have spikes at event times)
        let spike_count = neural.iter().filter(|&&val| val > 0.0).count();
        assert!(
            spike_count >= 3,
            "Should have at least 3 spikes for impulse events"
        );
    }

    #[test]
    fn test_trial_varying_basic() {
        use crate::hrf::functions::gaussian_hrf;

        // Two events with different Gaussian HRFs (different peak times)
        let hrf1: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(6.0, 2.0, None));
        let hrf2: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(10.0, 2.0, None));

        let reg =
            Regressor::new_trial_varying(vec![0.0, 20.0], vec![hrf1, hrf2], None, None, None, None)
                .unwrap();

        assert!(reg.per_event_hrfs.is_some());
        assert_eq!(reg.per_event_hrfs.as_ref().unwrap().len(), 2);

        let grid = Array1::linspace(0.0, 40.0, 401);
        let result = reg.evaluate(&grid, None, None).unwrap();
        assert_eq!(result.shape()[0], 401);
        assert_eq!(result.shape()[1], 1);

        // First event peaks around t=6, second around t=30 (onset=20 + peak=10)
        let first_half_max_idx = result
            .slice(ndarray::s![0..200, 0])
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let second_half_max_idx = result
            .slice(ndarray::s![200..401, 0])
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
            + 200;

        // First peak around t=6 (index ~60)
        assert!(
            first_half_max_idx > 30 && first_half_max_idx < 100,
            "First peak at index {} (t={}), expected near 60",
            first_half_max_idx,
            first_half_max_idx as f64 * 0.1
        );
        // Second peak around t=30 (index ~300)
        assert!(
            second_half_max_idx > 250 && second_half_max_idx < 350,
            "Second peak at index {} (t={}), expected near 300",
            second_half_max_idx,
            second_half_max_idx as f64 * 0.1
        );
    }

    #[test]
    fn test_trial_varying_validation_nbasis_mismatch() {
        use crate::hrf::functions::{bspline_hrf, gaussian_hrf};

        let hrf1: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(6.0, 2.0, None));
        // bspline_hrf with 5 basis functions has nbasis=5, different from gaussian's 1
        let hrf2: Arc<Box<dyn Hrf>> = Arc::new(bspline_hrf(5, None, None));

        let result =
            Regressor::new_trial_varying(vec![0.0, 20.0], vec![hrf1, hrf2], None, None, None, None);
        assert!(
            result.is_err(),
            "Should fail when HRFs have different nbasis"
        );
    }

    #[test]
    fn test_trial_varying_validation_length_mismatch() {
        use crate::hrf::functions::gaussian_hrf;

        let hrf1: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(6.0, 2.0, None));
        let hrf2: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(10.0, 2.0, None));

        // 3 onsets but only 2 HRFs (not 1, not 3)
        let result = Regressor::new_trial_varying(
            vec![0.0, 10.0, 20.0],
            vec![hrf1, hrf2],
            None,
            None,
            None,
            None,
        );
        assert!(
            result.is_err(),
            "Should fail when hrfs.len() != 1 and != onsets.len()"
        );
    }

    #[test]
    fn test_trial_varying_broadcast() {
        use crate::hrf::functions::gaussian_hrf;

        let hrf: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(6.0, 2.0, None));

        // Single HRF broadcast to all events
        let reg = Regressor::new_trial_varying(
            vec![0.0, 20.0, 40.0],
            vec![hrf], // length 1, should broadcast
            None,
            None,
            None,
            None,
        )
        .unwrap();

        assert_eq!(reg.per_event_hrfs.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn test_trial_varying_matches_single_hrf() {
        use approx::assert_relative_eq;

        // When all per-event HRFs are the same, the result should match
        // a standard single-HRF regressor.
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let hrf_arc: Arc<Box<dyn Hrf>> =
            Arc::new(spm_canonical(None, None, None, None, None, Some(true)));

        let onsets = vec![5.0, 15.0, 25.0];

        let reg_single = regressor(onsets.clone(), Some(hrf), None, None, None, None).unwrap();

        let reg_tv = Regressor::new_trial_varying(
            onsets,
            vec![Arc::clone(&hrf_arc); 3],
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let grid = Array1::linspace(0.0, 50.0, 501);
        let result_single = reg_single.evaluate(&grid, None, Some("loop")).unwrap();
        let result_tv = reg_tv.evaluate(&grid, None, None).unwrap();

        for i in 0..501 {
            assert_relative_eq!(result_single[[i, 0]], result_tv[[i, 0]], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_trial_varying_with_amplitude_filtering() {
        use crate::hrf::functions::gaussian_hrf;

        let hrf1: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(6.0, 2.0, None));
        let hrf2: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(8.0, 2.0, None));
        let hrf3: Arc<Box<dyn Hrf>> = Arc::new(gaussian_hrf(10.0, 2.0, None));

        // Middle event has zero amplitude -> should be filtered
        let reg = Regressor::new_trial_varying(
            vec![0.0, 10.0, 20.0],
            vec![hrf1, hrf2, hrf3],
            None,
            Some(vec![1.0, 0.0, 1.0]),
            None,
            None,
        )
        .unwrap();

        // After filtering, should have 2 events and 2 per-event HRFs
        assert_eq!(reg.onsets.len(), 2);
        assert_eq!(reg.per_event_hrfs.as_ref().unwrap().len(), 2);
    }
}
