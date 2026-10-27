//! Design matrix creation for fMRI GLM analysis.

use crate::error::Result;
use crate::regressor::core::{Regressor, RegressorSet};
use ndarray::{concatenate, Array1, Array2, Axis};

/// Options for design matrix creation.
///
/// This struct controls various aspects of design matrix generation,
/// including the numerical precision of convolution, the algorithm used,
/// and whether to include an intercept term for GLM analysis.
///
/// # Examples
///
/// ## Using Default Options
/// ```
/// use fmrihrf::regressor::DesignOptions;
///
/// let options = DesignOptions::default();
/// assert_eq!(options.precision, 0.33);
/// assert_eq!(options.method, "conv");
/// assert!(!options.include_intercept);
/// ```
///
/// ## Using the Builder Pattern
/// ```
/// use fmrihrf::regressor::DesignOptions;
///
/// let options = DesignOptions::builder()
///     .precision(0.1)          // 100ms precision
///     .method("fft")           // Use FFT convolution
///     .include_intercept(true) // Add intercept for GLM
///     .build();
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DesignOptions {
    /// Temporal precision for convolution in seconds (default: 0.33).
    /// Smaller values give more accurate convolution at the cost of computation time.
    pub precision: f64,
    /// Evaluation method for convolution (default: "conv").
    /// Options: "conv" (standard convolution), "fft" (FFT-based), "direct" (direct evaluation)
    pub method: String,
    /// Whether to add an intercept column (default: false).
    /// Set to true for standard GLM analysis.
    pub include_intercept: bool,
}

impl Default for DesignOptions {
    fn default() -> Self {
        Self {
            precision: 0.33,
            method: "conv".to_string(),
            include_intercept: false,
        }
    }
}

impl DesignOptions {
    /// Create a new builder for DesignOptions.
    ///
    /// The builder pattern allows for flexible configuration of design options
    /// with a fluent interface.
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignOptions;
    ///
    /// // High precision design matrix with intercept
    /// let options = DesignOptions::builder()
    ///     .precision(0.01)  // 10ms precision
    ///     .include_intercept(true)
    ///     .build();
    ///
    /// // FFT-based convolution
    /// let fft_options = DesignOptions::builder()
    ///     .method("fft")
    ///     .build();
    /// ```
    pub fn builder() -> DesignOptionsBuilder {
        DesignOptionsBuilder::default()
    }
}

/// Builder for DesignOptions with fluent interface.
///
/// This builder allows step-by-step configuration of design matrix options
/// with sensible defaults for any unspecified values.
///
/// # Examples
/// ```
/// use fmrihrf::regressor::DesignOptions;
///
/// // Only set what you need, rest uses defaults
/// let options = DesignOptions::builder()
///     .include_intercept(true)
///     .build();
///
/// // All options specified
/// let custom = DesignOptions::builder()
///     .precision(0.05)
///     .method("direct")
///     .include_intercept(true)
///     .build();
/// ```
#[derive(Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DesignOptionsBuilder {
    precision: Option<f64>,
    method: Option<String>,
    include_intercept: Option<bool>,
}

impl DesignOptionsBuilder {
    /// Set temporal precision for convolution.
    ///
    /// # Arguments
    /// * `precision` - Temporal precision in seconds (must be positive)
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignOptions;
    ///
    /// let options = DesignOptions::builder()
    ///     .precision(0.001)  // 1ms precision for high accuracy
    ///     .build();
    /// ```
    pub fn precision(mut self, precision: f64) -> Self {
        self.precision = Some(precision);
        self
    }

    /// Set evaluation method for convolution.
    ///
    /// # Arguments
    /// * `method` - Convolution method: "conv", "fft", or "direct"
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignOptions;
    ///
    /// // Use FFT for potentially faster convolution
    /// let options = DesignOptions::builder()
    ///     .method("fft")
    ///     .build();
    /// ```
    pub fn method(mut self, method: impl Into<String>) -> Self {
        self.method = Some(method.into());
        self
    }

    /// Set whether to include an intercept column.
    ///
    /// # Arguments
    /// * `include` - True to add intercept column, false otherwise
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignOptions;
    ///
    /// // Standard GLM design with intercept
    /// let options = DesignOptions::builder()
    ///     .include_intercept(true)
    ///     .build();
    /// ```
    pub fn include_intercept(mut self, include: bool) -> Self {
        self.include_intercept = Some(include);
        self
    }

    /// Build the DesignOptions with specified or default values.
    ///
    /// # Returns
    /// DesignOptions with the configured values
    ///
    /// # Default Values
    /// - precision: 0.33 seconds
    /// - method: "conv"
    /// - include_intercept: false
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignOptions;
    ///
    /// // Build with mix of specified and default values
    /// let options = DesignOptions::builder()
    ///     .precision(0.1)  // Specified
    ///     .build();        // method="conv", include_intercept=false (defaults)
    /// ```
    pub fn build(self) -> DesignOptions {
        DesignOptions {
            precision: self.precision.unwrap_or(0.33),
            method: self.method.unwrap_or_else(|| "conv".to_string()),
            include_intercept: self.include_intercept.unwrap_or(false),
        }
    }
}

/// Create a design matrix from a single regressor.
///
/// This function evaluates a regressor at specified time points to create
/// a design matrix suitable for GLM analysis. The resulting matrix has one
/// column per HRF basis function, plus an optional intercept column.
///
/// # Arguments
/// * `regressor` - Single regressor containing event timings and HRF
/// * `grid` - Time points at which to evaluate (typically scan times)
/// * `options` - Design matrix options (precision, method, intercept)
///
/// # Returns
/// Design matrix with shape (n_time_points, n_basis + intercept)
///
/// # Examples
///
/// ## Basic Design Matrix
/// ```
/// use fmrihrf::regressor::{regressor, design_from_regressor};
/// use fmrihrf::regressor::DesignOptions;
/// use ndarray::Array1;
///
/// // Create regressor for task events
/// let events = vec![10.0, 30.0, 50.0, 70.0];
/// let reg = regressor(events, None, None, None, None, None).unwrap();
///
/// // Evaluate at scan times (TR=2s, 100 scans)
/// let scan_times = Array1::linspace(0.0, 198.0, 100);
/// let design = design_from_regressor(&reg, &scan_times, None).unwrap();
///
/// assert_eq!(design.shape(), &[100, 1]); // 100 time points, 1 basis function
/// ```
///
/// ## With Intercept Column
/// ```
/// use fmrihrf::regressor::{regressor, design_from_regressor, DesignOptions};
/// use ndarray::Array1;
///
/// let events = vec![10.0, 30.0, 50.0];
/// let reg = regressor(events, None, None, None, None, None).unwrap();
///
/// // Create design matrix with intercept
/// let options = DesignOptions {
///     include_intercept: true,
///     ..Default::default()
/// };
///
/// let times = Array1::linspace(0.0, 100.0, 50);
/// let design = design_from_regressor(&reg, &times, Some(options)).unwrap();
///
/// assert_eq!(design.shape(), &[50, 2]); // 50 time points, 1 HRF + 1 intercept
/// assert_eq!(design.column(0).sum(), 50.0); // First column is all ones
/// ```
pub fn design_from_regressor(
    regressor: &Regressor,
    grid: &Array1<f64>,
    options: Option<DesignOptions>,
) -> Result<Array2<f64>> {
    let options = options.unwrap_or_default();

    // Evaluate regressor
    let mut design = regressor.evaluate(grid, Some(options.precision), Some(&options.method))?;

    // Add intercept if requested
    if options.include_intercept {
        let intercept = Array2::ones((grid.len(), 1));
        let views = vec![intercept.view(), design.view()];
        design = concatenate(Axis(1), &views).unwrap();
    }

    Ok(design)
}

/// Create a design matrix from a regressor set.
///
/// This function evaluates all regressors in a set to create a multi-condition
/// design matrix. Each condition's regressor contributes one or more columns
/// (depending on the number of HRF basis functions). This is the standard way
/// to create design matrices for multi-condition fMRI experiments.
///
/// # Arguments
/// * `regressor_set` - Set of regressors representing different conditions
/// * `grid` - Time points at which to evaluate (typically scan times)
/// * `options` - Design matrix options (precision, method, intercept)
///
/// # Returns
/// Design matrix with columns for each condition × basis function
///
/// # Examples
///
/// ## Multi-Condition Experiment
/// ```
/// use fmrihrf::regressor::{regressor_set, design_from_regressor_set};
/// use fmrihrf::regressor::DesignOptions;
/// use ndarray::Array1;
///
/// // Create regressor set for 3 conditions
/// let all_onsets = vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0];
/// let conditions = vec![
///     "visual".to_string(), "auditory".to_string(), "motor".to_string(),
///     "visual".to_string(), "auditory".to_string(), "motor".to_string()
/// ];
///
/// let rset = regressor_set(all_onsets, conditions, None, None, None, None, None).unwrap();
///
/// // Create design matrix at scan times
/// let scan_times = Array1::linspace(0.0, 100.0, 50);
/// let design = design_from_regressor_set(&rset, &scan_times, None).unwrap();
///
/// // 3 conditions × 1 basis = 3 columns
/// assert_eq!(design.shape(), &[50, 3]);
/// ```
///
/// ## With Multiple Basis Functions
/// ```
/// use fmrihrf::regressor::{regressor_set, design_from_regressor_set};
/// use fmrihrf::hrf::spm_hrf_with_derivatives;
/// use ndarray::Array1;
///
/// // Use HRF with temporal and dispersion derivatives (3 basis functions)
/// let hrf = spm_hrf_with_derivatives();
///
/// let onsets = vec![10.0, 30.0, 50.0, 70.0];
/// let conditions = vec!["A", "B", "A", "B"].iter().map(|s| s.to_string()).collect();
///
/// let rset = regressor_set(onsets, conditions, Some(hrf), None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 100.0, 50);
/// let design = design_from_regressor_set(&rset, &times, None).unwrap();
///
/// // 2 conditions × 3 basis = 6 columns
/// assert_eq!(design.shape(), &[50, 6]);
/// ```
pub fn design_from_regressor_set(
    regressor_set: &RegressorSet,
    grid: &Array1<f64>,
    options: Option<DesignOptions>,
) -> Result<Array2<f64>> {
    let options = options.unwrap_or_default();

    // Evaluate all regressors
    let mut design =
        regressor_set.evaluate(grid, Some(options.precision), Some(&options.method))?;

    // Add intercept if requested
    if options.include_intercept {
        let intercept = Array2::ones((grid.len(), 1));
        let views = vec![intercept.view(), design.view()];
        design = concatenate(Axis(1), &views).unwrap();
    }

    Ok(design)
}

/// Create a design matrix from multiple regressors.
///
/// This function evaluates a list of regressors to create a design matrix.
/// Unlike `design_from_regressor_set`, this works with arbitrary regressors
/// that may have different HRFs. Each regressor contributes columns based on
/// its number of basis functions.
///
/// # Arguments
/// * `regressors` - List of regressors (can have different HRFs)
/// * `grid` - Time points at which to evaluate (typically scan times)
/// * `options` - Design matrix options (precision, method, intercept)
///
/// # Returns
/// Design matrix with columns for each regressor × its basis functions
///
/// # Examples
///
/// ## Mixed HRF Types
/// ```
/// use fmrihrf::regressor::{regressor, design_from_regressors};
/// use fmrihrf::hrf::{spm_canonical, gamma_hrf};
/// use ndarray::Array1;
///
/// // Task regressor with canonical HRF
/// let task_hrf = spm_canonical(None, None, None, None, None, Some(true));
/// let task_reg = regressor(
///     vec![10.0, 30.0, 50.0],
///     Some(task_hrf),
///     None, None, None, None
/// ).unwrap();
///
/// // Nuisance regressor with different HRF
/// let nuisance_hrf = gamma_hrf(4.0, 1.0, Some(true));
/// let nuisance_reg = regressor(
///     vec![5.0, 25.0, 45.0, 65.0],
///     Some(nuisance_hrf),
///     None, None, None, None
/// ).unwrap();
///
/// // Combine into design matrix
/// let times = Array1::linspace(0.0, 80.0, 40);
/// let design = design_from_regressors(
///     &[task_reg, nuisance_reg],
///     &times,
///     None
/// ).unwrap();
///
/// assert_eq!(design.shape(), &[40, 2]); // 2 regressors × 1 basis each
/// ```
///
/// ## Empty Regressor List
/// ```
/// use fmrihrf::regressor::design_from_regressors;
/// use fmrihrf::regressor::DesignOptions;
/// use ndarray::Array1;
///
/// // Handle empty regressor list gracefully
/// let times = Array1::linspace(0.0, 100.0, 50);
/// let options = DesignOptions {
///     include_intercept: true,
///     ..Default::default()
/// };
///
/// let design = design_from_regressors(&[], &times, Some(options)).unwrap();
///
/// // Only intercept column
/// assert_eq!(design.shape(), &[50, 1]);
/// assert!(design.column(0).iter().all(|&x| x == 1.0));
/// ```
pub fn design_from_regressors(
    regressors: &[Regressor],
    grid: &Array1<f64>,
    options: Option<DesignOptions>,
) -> Result<Array2<f64>> {
    let options = options.unwrap_or_default();

    let mut results = Vec::new();

    // Evaluate each regressor
    for reg in regressors {
        let result = reg.evaluate(grid, Some(options.precision), Some(&options.method))?;
        results.push(result);
    }

    // Concatenate results
    let mut design = if results.is_empty() {
        Array2::zeros((grid.len(), 0))
    } else {
        let views: Vec<_> = results.iter().map(|arr| arr.view()).collect();
        concatenate(Axis(1), &views).unwrap()
    };

    // Add intercept if requested
    if options.include_intercept {
        let intercept = Array2::ones((grid.len(), 1));
        let views = vec![intercept.view(), design.view()];
        design = concatenate(Axis(1), &views).unwrap();
    }

    Ok(design)
}

/// Create a design matrix from regressors (simplified interface).
///
/// This function provides a simplified interface for creating design matrices
/// from a single regressor, with direct parameter specification instead of
/// using a DesignOptions struct. Useful for quick design matrix generation.
///
/// # Arguments
/// * `regressor` - Single regressor to evaluate
/// * `grid` - Time points at which to evaluate (seconds)
/// * `precision` - Temporal precision for convolution (default: 0.33)
/// * `method` - Evaluation method: 'conv', 'fft', or 'direct' (default: 'conv')
/// * `include_intercept` - Whether to add an intercept column (default: false)
///
/// # Returns
/// Design matrix as 2D array
///
/// # Examples
///
/// ## Basic Usage
/// ```
/// use fmrihrf::regressor::{regressor, regressor_design};
/// use ndarray::Array1;
///
/// // Create a simple event-related design
/// let reg = regressor(vec![10.0, 30.0, 50.0], None, None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 80.0, 40);
/// let design = regressor_design(&reg, &times, None, None, None).unwrap();
/// assert_eq!(design.shape(), &[40, 1]);
/// ```
///
/// ## High Temporal Precision
/// ```
/// use fmrihrf::regressor::{regressor, regressor_design};
/// use ndarray::Array1;
///
/// // Use higher precision for more accurate convolution
/// let reg = regressor(vec![5.5, 10.3, 15.7], None, None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 20.0, 100);
/// let design = regressor_design(
///     &reg,
///     &times,
///     Some(0.01),  // 10ms precision
///     Some("conv"),
///     None
/// ).unwrap();
/// ```
///
/// ## With Intercept
/// ```
/// use fmrihrf::regressor::{regressor, regressor_design};
/// use ndarray::Array1;
///
/// // Add intercept for GLM analysis
/// let reg = regressor(vec![10.0, 30.0], None, None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 40.0, 20);
/// let design = regressor_design(
///     &reg,
///     &times,
///     None,
///     None,
///     Some(true)  // Include intercept
/// ).unwrap();
///
/// assert_eq!(design.shape(), &[20, 2]); // HRF + intercept
/// assert!(design.column(0).iter().all(|&x| x == 1.0)); // First col is intercept
/// ```
pub fn regressor_design(
    regressor: &Regressor,
    grid: &Array1<f64>,
    precision: Option<f64>,
    method: Option<&str>,
    include_intercept: Option<bool>,
) -> Result<Array2<f64>> {
    let options = DesignOptions {
        precision: precision.unwrap_or(0.33),
        method: method.unwrap_or("conv").to_string(),
        include_intercept: include_intercept.unwrap_or(false),
    };

    design_from_regressor(regressor, grid, Some(options))
}

/// Design matrix with metadata for fMRI GLM analysis.
///
/// A DesignMatrix combines the numerical design matrix data with metadata
/// including column names and time points. This provides a complete representation
/// of the experimental design for GLM analysis, with convenient methods for
/// accessing columns by name or index.
///
/// # Fields
/// * `data` - The design matrix as a 2D array (time × regressors)
/// * `column_names` - Descriptive names for each column
/// * `time` - Time points corresponding to each row
///
/// # Examples
///
/// ## Creating a Design Matrix
/// ```
/// use fmrihrf::regressor::DesignMatrix;
/// use ndarray::{Array1, Array2};
///
/// // Create a simple 2-condition design matrix
/// let data = Array2::from_shape_vec((10, 2), vec![
///     0.0, 0.0,
///     0.0, 0.1,
///     0.1, 0.0,
///     0.0, 0.3,
///     0.3, 0.0,
///     0.0, 0.5,
///     0.5, 0.0,
///     0.0, 0.3,
///     0.3, 0.0,
///     0.1, 0.0,
/// ]).unwrap();
///
/// let time = Array1::linspace(0.0, 18.0, 10);
/// let dm = DesignMatrix::new(data, time);
///
/// assert_eq!(dm.nrows(), 10);
/// assert_eq!(dm.ncols(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct DesignMatrix {
    /// The design matrix data
    pub data: Array2<f64>,
    /// Column names (if available)
    pub column_names: Vec<String>,
    /// Time points (grid)
    pub time: Array1<f64>,
}

impl DesignMatrix {
    /// Create a new DesignMatrix with auto-generated column names.
    ///
    /// Column names are automatically generated as "col0", "col1", etc.
    /// Use `with_names` to specify custom column names.
    ///
    /// # Arguments
    /// * `data` - Design matrix data (time × regressors)
    /// * `time` - Time points for each row
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignMatrix;
    /// use ndarray::{Array1, Array2};
    ///
    /// let data = Array2::zeros((50, 3));
    /// let time = Array1::linspace(0.0, 98.0, 50);
    /// let dm = DesignMatrix::new(data, time);
    ///
    /// assert_eq!(dm.column_names, vec!["col0", "col1", "col2"]);
    /// ```
    pub fn new(data: Array2<f64>, time: Array1<f64>) -> Self {
        let n_cols = data.ncols();
        let column_names = (0..n_cols).map(|i| format!("col{}", i)).collect();

        Self {
            data,
            column_names,
            time,
        }
    }

    /// Create a DesignMatrix with custom column names.
    ///
    /// This constructor allows you to specify meaningful names for each column,
    /// which is useful for identifying regressors in GLM results.
    ///
    /// # Arguments
    /// * `data` - Design matrix data (time × regressors)
    /// * `time` - Time points for each row
    /// * `column_names` - Names for each column (must match number of columns)
    ///
    /// # Returns
    /// Result containing the DesignMatrix or an error if column count mismatches
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignMatrix;
    /// use ndarray::{Array1, Array2};
    ///
    /// let data = Array2::zeros((100, 4));
    /// let time = Array1::linspace(0.0, 198.0, 100);
    /// let names = vec![
    ///     "intercept".to_string(),
    ///     "task_A".to_string(),
    ///     "task_B".to_string(),
    ///     "motion".to_string(),
    /// ];
    ///
    /// let dm = DesignMatrix::with_names(data, time, names).unwrap();
    /// assert_eq!(dm.column_by_name("task_A").unwrap().len(), 100);
    /// ```
    pub fn with_names(
        data: Array2<f64>,
        time: Array1<f64>,
        column_names: Vec<String>,
    ) -> Result<Self> {
        if column_names.len() != data.ncols() {
            return Err(crate::error::HrfError::InvalidInput(format!(
                "Number of column names ({}) doesn't match number of columns ({})",
                column_names.len(),
                data.ncols()
            )));
        }

        Ok(Self {
            data,
            column_names,
            time,
        })
    }

    /// Get number of rows (time points).
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns (regressors).
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Get a column by index.
    ///
    /// Returns None if the index is out of bounds.
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::DesignMatrix;
    /// use ndarray::{Array1, Array2};
    ///
    /// let data = Array2::from_shape_vec((5, 2),
    ///     vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    /// ).unwrap();
    /// let time = Array1::linspace(0.0, 8.0, 5);
    /// let dm = DesignMatrix::new(data, time);
    ///
    /// let col0 = dm.column(0).unwrap();
    /// assert_eq!(col0[0], 1.0);
    /// assert_eq!(col0[4], 9.0);
    ///
    /// assert!(dm.column(2).is_none()); // Out of bounds
    /// ```
    pub fn column(&self, idx: usize) -> Option<ndarray::ArrayView1<'_, f64>> {
        if idx < self.ncols() {
            Some(self.data.column(idx))
        } else {
            None
        }
    }

    /// Get a column by name.
    ///
    /// Searches for a column with the specified name and returns its data.
    /// Returns None if no column with that name exists.
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::{regressor_set, design_matrix_from_set};
    /// use ndarray::Array1;
    ///
    /// // Create a design matrix with named conditions
    /// let onsets = vec![5.0, 15.0, 25.0, 35.0];
    /// let conditions = vec!["visual", "motor", "visual", "motor"]
    ///     .iter().map(|s| s.to_string()).collect();
    ///
    /// let rset = regressor_set(onsets, conditions, None, None, None, None, None).unwrap();
    /// let times = Array1::linspace(0.0, 50.0, 25);
    /// let dm = design_matrix_from_set(&rset, &times, None).unwrap();
    ///
    /// // Access columns by name
    /// let visual_col = dm.column_by_name("visual").unwrap();
    /// let motor_col = dm.column_by_name("motor").unwrap();
    /// assert_eq!(visual_col.len(), 25);
    /// assert_eq!(motor_col.len(), 25);
    ///
    /// // Non-existent column returns None
    /// assert!(dm.column_by_name("auditory").is_none());
    /// ```
    pub fn column_by_name(&self, name: &str) -> Option<ndarray::ArrayView1<'_, f64>> {
        self.column_names
            .iter()
            .position(|n| n == name)
            .and_then(|idx| self.column(idx))
    }

    /// Add column names based on a RegressorSet's condition labels.
    ///
    /// This method automatically generates descriptive column names based on
    /// the condition labels in the RegressorSet. For multi-basis HRFs, it
    /// appends "_basisN" to the condition name. If an intercept column is
    /// detected (first column named "col0" with extra columns), it's renamed
    /// to "intercept".
    ///
    /// # Arguments
    /// * `regressor_set` - The RegressorSet used to create this design matrix
    ///
    /// # Returns
    /// Result indicating success or error if column counts don't match
    ///
    /// # Examples
    /// ```
    /// use fmrihrf::regressor::{regressor_set, design_from_regressor_set, DesignMatrix};
    /// use fmrihrf::regressor::DesignOptions;
    /// use fmrihrf::hrf::spm_hrf_with_derivatives;
    /// use ndarray::Array1;
    ///
    /// // Create regressor set with HRF that has derivatives
    /// let hrf = spm_hrf_with_derivatives();
    /// let onsets = vec![10.0, 30.0];
    /// let conditions = vec!["task".to_string(); 2];
    /// let rset = regressor_set(onsets, conditions, Some(hrf), None, None, None, None).unwrap();
    ///
    /// // Create design matrix with intercept
    /// let times = Array1::linspace(0.0, 50.0, 25);
    /// let options = DesignOptions {
    ///     include_intercept: true,
    ///     ..Default::default()
    /// };
    /// let data = design_from_regressor_set(&rset, &times, Some(options)).unwrap();
    ///
    /// // Create DesignMatrix and set names
    /// let mut dm = DesignMatrix::new(data, times);
    /// dm.set_condition_names(&rset).unwrap();
    ///
    /// // Check generated names
    /// assert_eq!(dm.column_names[0], "intercept");
    /// assert_eq!(dm.column_names[1], "task_basis0"); // Main HRF
    /// assert_eq!(dm.column_names[2], "task_basis1"); // Temporal derivative
    /// ```
    pub fn set_condition_names(&mut self, regressor_set: &RegressorSet) -> Result<()> {
        let mut names = Vec::new();

        // If intercept is first column
        if self.column_names[0] == "col0" && self.ncols() > regressor_set.levels.len() {
            names.push("intercept".to_string());
        }

        // Add condition names
        for (i, level) in regressor_set.levels.iter().enumerate() {
            let reg = &regressor_set.regressors[i];
            let n_basis = reg.hrf.nbasis();

            if n_basis == 1 {
                names.push(level.clone());
            } else {
                for b in 0..n_basis {
                    names.push(format!("{}_basis{}", level, b));
                }
            }
        }

        if names.len() != self.ncols() {
            return Err(crate::error::HrfError::InvalidInput(
                "Column count mismatch when setting names".into(),
            ));
        }

        self.column_names = names;
        Ok(())
    }
}

/// Create a DesignMatrix from a RegressorSet with automatic column naming.
///
/// This function creates a design matrix with meaningful column names based on
/// the condition labels in the regressor set. For multi-basis HRFs, column names
/// include the basis index. An intercept column is named "intercept" if included.
///
/// # Arguments
/// * `regressor_set` - Set of regressors with condition labels
/// * `grid` - Time points at which to evaluate
/// * `options` - Design matrix options
///
/// # Returns
/// DesignMatrix with data, time points, and descriptive column names
///
/// # Examples
///
/// ## Named Conditions
/// ```
/// use fmrihrf::regressor::{regressor_set, design_matrix_from_set};
/// use ndarray::Array1;
///
/// // Create experimental design
/// let onsets = vec![5.0, 15.0, 25.0, 35.0];
/// let conditions = vec!["left", "right", "left", "right"]
///     .iter().map(|s| s.to_string()).collect();
///
/// let rset = regressor_set(onsets, conditions, None, None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 50.0, 25);
///
/// let dm = design_matrix_from_set(&rset, &times, None).unwrap();
///
/// // Column names match condition labels
/// assert_eq!(dm.column_names, vec!["left", "right"]);
/// assert!(dm.column_by_name("left").is_some());
/// assert!(dm.column_by_name("right").is_some());
/// ```
///
/// ## Multi-Basis Functions
/// ```
/// use fmrihrf::regressor::{regressor_set, design_matrix_from_set};
/// use fmrihrf::regressor::DesignOptions;
/// use fmrihrf::hrf::spm_hrf_with_derivatives;
/// use ndarray::Array1;
///
/// // HRF with temporal derivative (2 basis functions)
/// let hrf = spm_hrf_with_derivatives();
///
/// let onsets = vec![10.0, 30.0];
/// let conditions = vec!["stim".to_string(); 2];
///
/// let rset = regressor_set(onsets, conditions, Some(hrf), None, None, None, None).unwrap();
/// let times = Array1::linspace(0.0, 50.0, 25);
///
/// // Include intercept
/// let options = DesignOptions {
///     include_intercept: true,
///     ..Default::default()
/// };
///
/// let dm = design_matrix_from_set(&rset, &times, Some(options)).unwrap();
///
/// // Column names include basis indices
/// assert_eq!(dm.column_names[0], "intercept");
/// assert_eq!(dm.column_names[1], "stim_basis0"); // Main HRF
/// assert_eq!(dm.column_names[2], "stim_basis1"); // Derivative
/// ```
pub fn design_matrix_from_set(
    regressor_set: &RegressorSet,
    grid: &Array1<f64>,
    options: Option<DesignOptions>,
) -> Result<DesignMatrix> {
    let data = design_from_regressor_set(regressor_set, grid, options)?;
    let mut dm = DesignMatrix::new(data, grid.clone());
    dm.set_condition_names(regressor_set)?;
    Ok(dm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::functions::spm_canonical;
    use crate::regressor::core::{regressor, regressor_set};

    #[test]
    fn test_design_single_regressor() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![10.0, 30.0, 50.0], Some(hrf), None, None, None, None).unwrap();

        let grid = Array1::linspace(0.0, 70.0, 701);
        let design = design_from_regressor(&reg, &grid, None).unwrap();

        assert_eq!(design.shape(), &[701, 1]);
    }

    #[test]
    fn test_design_with_intercept() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(vec![10.0, 30.0, 50.0], Some(hrf), None, None, None, None).unwrap();

        let grid = Array1::linspace(0.0, 70.0, 701);
        let options = DesignOptions {
            include_intercept: true,
            ..Default::default()
        };

        let design = design_from_regressor(&reg, &grid, Some(options)).unwrap();

        assert_eq!(design.shape(), &[701, 2]);
        // First column should be all ones
        assert!(design.column(0).iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_design_regressor_set() {
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

        let grid = Array1::linspace(0.0, 80.0, 801);
        let design = design_from_regressor_set(&rset, &grid, None).unwrap();

        assert_eq!(design.shape(), &[801, 3]); // 3 conditions
    }

    #[test]
    fn test_design_multiple_regressors() {
        let hrf1 = spm_canonical(None, None, None, None, None, Some(true));
        let hrf2 = spm_canonical(None, None, None, None, None, Some(true));

        let reg1 = regressor(vec![10.0, 30.0], Some(hrf1), None, None, None, None).unwrap();
        let reg2 = regressor(vec![20.0, 40.0], Some(hrf2), None, None, None, None).unwrap();

        let grid = Array1::linspace(0.0, 60.0, 601);
        let design = design_from_regressors(&[reg1, reg2], &grid, None).unwrap();

        assert_eq!(design.shape(), &[601, 2]);
    }

    #[test]
    fn test_design_matrix_with_names() {
        let data = Array2::zeros((10, 3));
        let time = Array1::linspace(0.0, 9.0, 10);
        let names = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let dm = DesignMatrix::with_names(data, time, names).unwrap();

        assert_eq!(dm.ncols(), 3);
        assert_eq!(dm.column_names[0], "A");
        assert!(dm.column_by_name("B").is_some());
    }

    #[test]
    fn test_design_options_builder() {
        let options = DesignOptions::builder()
            .precision(0.1)
            .method("fft")
            .include_intercept(true)
            .build();

        assert_eq!(options.precision, 0.1);
        assert_eq!(options.method, "fft");
        assert!(options.include_intercept);
    }

    #[test]
    fn test_design_matrix_from_set() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let rset = regressor_set(
            vec![10.0, 20.0, 30.0],
            vec![
                "cond1".to_string(),
                "cond2".to_string(),
                "cond1".to_string(),
            ],
            Some(hrf),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let grid = Array1::linspace(0.0, 50.0, 501);
        let dm = design_matrix_from_set(&rset, &grid, None).unwrap();

        assert_eq!(dm.column_names.len(), 2);
        assert!(dm.column_names.contains(&"cond1".to_string()));
        assert!(dm.column_names.contains(&"cond2".to_string()));
    }
}
