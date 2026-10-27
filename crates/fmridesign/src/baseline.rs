//! Baseline model specifications.
//!
//! This module implements baseline models for fMRI time series that account for:
//! - Low-frequency drift using various basis functions
//! - Block-wise intercepts
//! - Nuisance regressors (e.g., motion parameters)

use crate::basis::{BSplineBasis, NaturalSplineBasis, PolynomialBasis};
use crate::sampling::SamplingFrame;
use crate::types::BasisFunction;
use crate::types::ValidationError;
use ndarray::{concatenate, Array1, Array2, Axis};
use std::collections::HashMap;

/// Wrapper around SamplingFrame to provide additional methods
#[derive(Debug, Clone)]
pub struct SamplingInfo {
    frame: SamplingFrame,
    block_lens: Vec<usize>,
}

impl SamplingInfo {
    /// Create from a SamplingFrame
    pub fn from_frame(frame: SamplingFrame) -> Self {
        // Extract block lengths directly from the fmrihrf SamplingFrame.
        // This mirrors the R implementation which relies on fmrihrf::blocklens().
        let block_lens = frame.blocklens.clone();
        Self { frame, block_lens }
    }

    /// Create from a SamplingFrame with explicit block lengths
    pub fn from_frame_with_blocks(frame: SamplingFrame, block_lens: Vec<usize>) -> Self {
        Self { frame, block_lens }
    }

    /// Get block lengths
    pub fn block_lengths(&self) -> &[usize] {
        &self.block_lens
    }

    /// Get total number of scans
    pub fn n_scans(&self) -> usize {
        self.frame.n_scans()
    }

    /// Get number of blocks
    pub fn n_blocks(&self) -> usize {
        self.block_lens.len()
    }
}

/// Type of basis function for drift modeling
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BasisType {
    /// Constant (intercept only)
    Constant,
    /// Polynomial basis
    Polynomial,
    /// B-spline basis
    BSpline,
    /// Natural cubic spline basis with linear boundary constraints
    NaturalSpline,
}

/// Type of intercept to include
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterceptType {
    /// Separate intercept per run/block
    Runwise,
    /// Single intercept for all runs
    Global,
    /// No intercept
    None,
}

/// Specification for baseline drift modeling
#[derive(Debug, Clone)]
pub struct BaselineSpec {
    /// Number of basis terms per block
    pub degree: usize,
    /// Type of basis function
    pub basis: BasisType,
    /// Type of intercept
    pub intercept: InterceptType,
    /// Name for the term
    pub name: String,
}

impl BaselineSpec {
    /// Create a new baseline specification
    pub fn new(degree: usize, basis: BasisType, intercept: InterceptType) -> Self {
        let name = format!(
            "baseline_{}_{}",
            match basis {
                BasisType::Constant => "constant",
                BasisType::Polynomial => "poly",
                BasisType::BSpline => "bs",
                BasisType::NaturalSpline => "ns",
            },
            degree
        );

        Self {
            degree: if basis == BasisType::Constant {
                1
            } else {
                degree
            },
            basis,
            intercept,
            name,
        }
    }
}

/// A baseline term in the model
#[derive(Debug, Clone)]
pub struct BaselineTerm {
    /// Variable name for the term
    pub varname: String,
    /// Design matrix for this term
    pub design_matrix: Array2<f64>,
    /// Column indices for each block
    pub col_indices: Vec<Vec<usize>>,
    /// Row indices for each block
    pub row_indices: Vec<Vec<usize>>,
    /// Column names
    pub column_names: Vec<String>,
}

impl BaselineTerm {
    /// Get design matrix for specific blocks
    pub fn get_block_matrix(&self, block_ids: Option<&[usize]>, all_rows: bool) -> Array2<f64> {
        if block_ids.is_none() {
            return self.design_matrix.clone();
        }

        let block_ids = block_ids.unwrap();

        // Get column indices for requested blocks
        let mut col_idx = Vec::new();
        for &bid in block_ids {
            if bid < self.col_indices.len() {
                col_idx.extend(&self.col_indices[bid]);
            }
        }

        if col_idx.is_empty() {
            return Array2::zeros((0, 0));
        }

        if all_rows {
            // Return all rows, selected columns
            self.design_matrix.select(Axis(1), &col_idx)
        } else {
            // Get row indices for requested blocks
            let mut row_idx = Vec::new();
            for &bid in block_ids {
                if bid < self.row_indices.len() {
                    row_idx.extend(&self.row_indices[bid]);
                }
            }

            if row_idx.is_empty() {
                return Array2::zeros((0, 0));
            }

            // Return selected rows and columns
            let selected_rows = self.design_matrix.select(Axis(0), &row_idx);
            selected_rows.select(Axis(1), &col_idx)
        }
    }
}

/// Baseline model for fMRI time series
#[derive(Debug, Clone)]
pub struct BaselineModel {
    /// Dictionary of baseline terms
    pub terms: HashMap<String, Option<BaselineTerm>>,
    /// Drift specification
    pub drift_spec: BaselineSpec,
    /// Sampling frame information
    pub sampling_info: SamplingInfo,
}

impl BaselineModel {
    /// Create a new baseline model
    pub fn new(
        terms: HashMap<String, Option<BaselineTerm>>,
        drift_spec: BaselineSpec,
        sampling_info: SamplingInfo,
    ) -> Self {
        Self {
            terms,
            drift_spec,
            sampling_info,
        }
    }

    /// Get the full design matrix by combining all terms
    pub fn design_matrix(&self) -> Result<Array2<f64>, ValidationError> {
        let mut matrices = Vec::new();
        let mut all_col_names = Vec::new();

        // Collect matrices in order: drift, block, nuisance
        for term_name in &["drift", "block", "nuisance"] {
            if let Some(Some(term)) = self.terms.get(*term_name) {
                matrices.push(term.design_matrix.view());
                all_col_names.extend(term.column_names.clone());
            }
        }

        if matrices.is_empty() {
            // Return empty matrix with correct number of rows
            let n_rows = self.sampling_info.n_scans();
            return Ok(Array2::zeros((n_rows, 0)));
        }

        // Concatenate along columns
        let combined = concatenate(Axis(1), &matrices).map_err(|e| {
            ValidationError::InvalidParameter(format!(
                "Failed to concatenate baseline matrices: {}",
                e
            ))
        })?;

        Ok(combined)
    }

    /// Get column names for the full design matrix
    pub fn column_names(&self) -> Vec<String> {
        let mut names = Vec::new();

        for term_name in &["drift", "block", "nuisance"] {
            if let Some(Some(term)) = self.terms.get(*term_name) {
                names.extend(term.column_names.clone());
            }
        }

        names
    }
}

/// Construct a baseline model from a specification
pub fn baseline_model_from_spec(
    sampling_info: &SamplingInfo,
    spec: &BaselineSpec,
    nuisance_list: Option<Vec<Array2<f64>>>,
) -> Result<BaselineModel, ValidationError> {
    baseline_model(
        spec.basis,
        spec.degree,
        sampling_info,
        spec.intercept,
        nuisance_list,
    )
}

/// Construct a baseline model for fMRI time series
pub fn baseline_model(
    basis: BasisType,
    degree: usize,
    sampling_info: &SamplingInfo,
    intercept: InterceptType,
    nuisance_list: Option<Vec<Array2<f64>>>,
) -> Result<BaselineModel, ValidationError> {
    // Validate basis
    if matches!(basis, BasisType::BSpline | BasisType::NaturalSpline) && degree < 3 {
        return Err(ValidationError::InvalidParameter(format!(
            "{:?} baseline basis must have degree/df >= 3, matching R",
            basis
        )));
    }

    // Create drift specification
    let drift_spec = BaselineSpec::new(degree, basis, intercept);

    // Construct terms
    let mut terms = HashMap::new();

    // Always create drift term
    terms.insert(
        "drift".to_string(),
        Some(construct_drift_term(&drift_spec, sampling_info)?),
    );

    // Create block term if needed
    if intercept != InterceptType::None && basis != BasisType::Constant {
        terms.insert(
            "block".to_string(),
            Some(construct_block_term(sampling_info, intercept)?),
        );
    } else {
        terms.insert("block".to_string(), None);
    }

    // Create nuisance term if provided
    if let Some(nuisance) = nuisance_list {
        terms.insert(
            "nuisance".to_string(),
            Some(make_nuisance_term(nuisance, sampling_info)?),
        );
    } else {
        terms.insert("nuisance".to_string(), None);
    }

    Ok(BaselineModel::new(terms, drift_spec, sampling_info.clone()))
}

/// Construct drift term from specification
fn construct_drift_term(
    spec: &BaselineSpec,
    sinfo: &SamplingInfo,
) -> Result<BaselineTerm, ValidationError> {
    let block_lens = sinfo.block_lengths();

    let mut matrices = Vec::new();
    let mut col_indices = Vec::new();
    let mut row_indices = Vec::new();
    let mut column_names = Vec::new();

    let mut current_col = 0;
    let mut current_row = 0;

    for (i, &block_len) in block_lens.iter().enumerate() {
        // Generate time points for this block
        let t: Array1<f64> = Array1::range(1.0, block_len as f64 + 1.0, 1.0);

        // Apply basis function
        let block_matrix = match spec.basis {
            BasisType::Constant => {
                // Create constant basis (all ones)
                Array2::ones((t.len(), 1))
            }
            BasisType::Polynomial => {
                // Create polynomial basis
                let basis = PolynomialBasis::new(spec.degree);
                basis.evaluate(&t.view())?
            }
            BasisType::NaturalSpline => NaturalSplineBasis::new(spec.degree).evaluate(&t.view())?,
            BasisType::BSpline => {
                // Create B-spline basis
                let basis = BSplineBasis::new(spec.degree).with_degree(spec.degree);
                basis.evaluate(&t.view())?
            }
        };

        let n_cols = block_matrix.ncols();
        matrices.push(block_matrix);

        // Track indices
        col_indices.push((current_col..current_col + n_cols).collect());
        row_indices.push((current_row..current_row + block_len).collect());

        // Generate column names
        for j in 0..n_cols {
            if spec.basis == BasisType::Constant {
                column_names.push(format!(
                    "base_{}_run{}",
                    match spec.basis {
                        BasisType::Constant => "constant",
                        BasisType::Polynomial => "poly",
                        BasisType::BSpline => "bs",
                        BasisType::NaturalSpline => "ns",
                    },
                    i + 1
                ));
            } else {
                column_names.push(format!(
                    "base_{}_run{}_b{}",
                    match spec.basis {
                        BasisType::Constant => "constant",
                        BasisType::Polynomial => "poly",
                        BasisType::BSpline => "bs",
                        BasisType::NaturalSpline => "ns",
                    },
                    i + 1,
                    j + 1
                ));
            }
        }

        current_col += n_cols;
        current_row += block_len;
    }

    // Handle global constant intercept
    let full_matrix =
        if spec.basis == BasisType::Constant && spec.intercept == InterceptType::Global {
            // Combine into single column
            let total_rows = sinfo.n_scans();
            let combined = Array2::ones((total_rows, 1));
            column_names = vec!["base_constant".to_string()];
            col_indices = vec![vec![0]; block_lens.len()];
            combined
        } else {
            // Block diagonal structure
            block_diagonal(&matrices)?
        };

    Ok(BaselineTerm {
        varname: spec.name.clone(),
        design_matrix: full_matrix,
        col_indices,
        row_indices,
        column_names,
    })
}

/// Construct block intercept term
fn construct_block_term(
    sinfo: &SamplingInfo,
    intercept: InterceptType,
) -> Result<BaselineTerm, ValidationError> {
    let block_lens = sinfo.block_lengths();
    let n_blocks = block_lens.len();
    let total_rows = sinfo.n_scans();

    let (matrix, column_names, col_indices) = if intercept == InterceptType::Global {
        // Single intercept column
        let matrix = Array2::ones((total_rows, 1));
        let names = vec!["intercept".to_string()];
        let indices = vec![vec![0]; n_blocks];
        (matrix, names, indices)
    } else {
        // Runwise intercepts
        let matrices: Vec<Array2<f64>> = block_lens
            .iter()
            .map(|&len| Array2::ones((len, 1)))
            .collect();

        let matrix = block_diagonal(&matrices)?;
        let names: Vec<String> = (0..n_blocks)
            .map(|i| format!("intercept_run{}", i + 1))
            .collect();
        let indices: Vec<Vec<usize>> = (0..n_blocks).map(|i| vec![i]).collect();

        (matrix, names, indices)
    };

    // Create row indices
    let mut row_indices = Vec::new();
    let mut current_row = 0;
    for &block_len in block_lens {
        row_indices.push((current_row..current_row + block_len).collect());
        current_row += block_len;
    }

    Ok(BaselineTerm {
        varname: "block".to_string(),
        design_matrix: matrix,
        col_indices,
        row_indices,
        column_names,
    })
}

/// Create nuisance term from list of matrices
fn make_nuisance_term(
    nuisance_list: Vec<Array2<f64>>,
    sinfo: &SamplingInfo,
) -> Result<BaselineTerm, ValidationError> {
    let block_lens = sinfo.block_lengths();

    // Validate input
    if nuisance_list.len() != block_lens.len() {
        return Err(ValidationError::InvalidParameter(format!(
            "Length of nuisance_list ({}) must equal number of blocks ({})",
            nuisance_list.len(),
            block_lens.len()
        )));
    }

    // Validate sizes and collect info
    let mut matrices = Vec::new();
    let mut _total_cols = 0;

    for (i, (nuisance, &block_len)) in nuisance_list.iter().zip(block_lens).enumerate() {
        if nuisance.nrows() != block_len {
            return Err(ValidationError::InvalidParameter(format!(
                "Nuisance matrix {} has {} rows but block has {} timepoints",
                i,
                nuisance.nrows(),
                block_len
            )));
        }

        matrices.push(nuisance.clone());
        _total_cols += nuisance.ncols();
    }

    // Create block diagonal matrix
    let full_matrix = block_diagonal(&matrices)?;

    // Generate column names and indices
    let mut column_names = Vec::new();
    let mut col_indices = Vec::new();
    let mut current_col = 0;

    for (i, mat) in matrices.iter().enumerate() {
        let n_cols = mat.ncols();
        let mut block_cols = Vec::new();

        for j in 0..n_cols {
            column_names.push(format!("nuis_run{}_c{}", i + 1, j + 1));
            block_cols.push(current_col);
            current_col += 1;
        }

        col_indices.push(block_cols);
    }

    // Generate row indices
    let mut row_indices = Vec::new();
    let mut current_row = 0;

    for &block_len in block_lens {
        row_indices.push((current_row..current_row + block_len).collect());
        current_row += block_len;
    }

    Ok(BaselineTerm {
        varname: "nuisance".to_string(),
        design_matrix: full_matrix,
        col_indices,
        row_indices,
        column_names,
    })
}

/// Create block diagonal matrix from list of matrices
fn block_diagonal(matrices: &[Array2<f64>]) -> Result<Array2<f64>, ValidationError> {
    if matrices.is_empty() {
        return Ok(Array2::zeros((0, 0)));
    }

    // Get dimensions
    let n_rows: usize = matrices.iter().map(|m| m.nrows()).sum();
    let n_cols: usize = matrices.iter().map(|m| m.ncols()).sum();

    // Create output matrix
    let mut result = Array2::zeros((n_rows, n_cols));

    // Fill blocks
    let mut row_offset = 0;
    let mut col_offset = 0;

    for mat in matrices {
        let n_r = mat.nrows();
        let n_c = mat.ncols();

        result
            .slice_mut(ndarray::s![
                row_offset..row_offset + n_r,
                col_offset..col_offset + n_c
            ])
            .assign(mat);

        row_offset += n_r;
        col_offset += n_c;
    }

    Ok(result)
}

/// Create a baseline drift specification.
///
/// This is a convenience function for creating baseline drift models.
///
/// # Arguments
///
/// * `degree` - Number of basis functions (or polynomial degree)
/// * `basis` - Type of basis function: "poly", "bs", "ns", or "constant"
/// * `name` - Optional name for the baseline term
///
/// # Examples
///
/// ```
/// use fmridesign::baseline::baseline;
///
/// // Polynomial drift with degree 3
/// let drift = baseline(3, "poly", None);
///
/// // B-spline with 5 basis functions
/// let drift = baseline(5, "bs", Some("my_drift"));
/// ```
pub fn baseline(
    degree: usize,
    basis: &str,
    name: Option<&str>,
) -> Result<BaselineSpec, ValidationError> {
    let basis_type = match basis.to_lowercase().as_str() {
        "constant" => BasisType::Constant,
        "poly" | "polynomial" => BasisType::Polynomial,
        "bs" | "bspline" => BasisType::BSpline,
        "ns" | "natural_spline" => BasisType::NaturalSpline,
        _ => {
            return Err(ValidationError::InvalidParameter(format!(
                "Unknown basis type: '{}'. Use 'poly', 'bs', 'ns', or 'constant'",
                basis
            )))
        }
    };

    let mut spec = BaselineSpec::new(degree, basis_type, InterceptType::Runwise);
    if let Some(n) = name {
        spec.name = n.to_string();
    }

    Ok(spec)
}

/// Specification for nuisance regressors.
#[derive(Debug, Clone)]
pub struct NuisanceSpec {
    /// Name of the nuisance term
    pub name: String,
    /// Column names for the regressors
    pub columns: Option<Vec<String>>,
}

impl NuisanceSpec {
    /// Create a new nuisance specification.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            columns: None,
        }
    }

    /// Set column names.
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }
}

/// Create a nuisance regressor specification.
///
/// # Arguments
///
/// * `name` - Name for the nuisance term
/// * `columns` - Optional column names
///
/// # Examples
///
/// ```
/// use fmridesign::baseline::nuisance;
///
/// // Basic nuisance term
/// let motion = nuisance("motion", None);
///
/// // With specific columns
/// let motion = nuisance("motion", Some(vec!["x", "y", "z", "roll", "pitch", "yaw"]));
/// ```
pub fn nuisance(name: &str, columns: Option<Vec<&str>>) -> NuisanceSpec {
    let mut spec = NuisanceSpec::new(name);
    if let Some(cols) = columns {
        spec = spec.with_columns(cols.into_iter().map(|s| s.to_string()).collect());
    }
    spec
}

/// Specification for block structure.
#[derive(Debug, Clone)]
pub struct BlockSpec {
    /// Block lengths
    pub block_lens: Vec<usize>,
    /// Block names
    pub names: Option<Vec<String>>,
}

impl BlockSpec {
    /// Create a new block specification.
    pub fn new(block_lens: Vec<usize>) -> Self {
        Self {
            block_lens,
            names: None,
        }
    }

    /// Set block names.
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.names = Some(names);
        self
    }
}

/// Create a block structure specification.
///
/// # Arguments
///
/// * `block_lens` - Length of each block in scans
/// * `names` - Optional names for each block
///
/// # Examples
///
/// ```
/// use fmridesign::baseline::block;
///
/// // Three runs of 100 scans each
/// let blocks = block(vec![100, 100, 100], None);
///
/// // With names
/// let blocks = block(vec![100, 100], Some(vec!["run1", "run2"]));
/// ```
pub fn block(block_lens: Vec<usize>, names: Option<Vec<&str>>) -> BlockSpec {
    let mut spec = BlockSpec::new(block_lens);
    if let Some(n) = names {
        spec = spec.with_names(n.into_iter().map(|s| s.to_string()).collect());
    }
    spec
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::uniform_sampling_frame;
    use ndarray::array;

    #[test]
    fn test_baseline_convenience() {
        // Test polynomial baseline
        let poly_spec = baseline(3, "poly", None).unwrap();
        assert_eq!(poly_spec.degree, 3);
        assert_eq!(poly_spec.basis, BasisType::Polynomial);
        assert_eq!(poly_spec.intercept, InterceptType::Runwise);

        // Test B-spline baseline
        let bs_spec = baseline(5, "bs", Some("my_drift")).unwrap();
        assert_eq!(bs_spec.degree, 5);
        assert_eq!(bs_spec.basis, BasisType::BSpline);
        assert_eq!(bs_spec.name, "my_drift");

        // Test invalid basis type
        let err = baseline(3, "invalid", None);
        assert!(err.is_err());
    }

    #[test]
    fn test_nuisance_spec() {
        // Basic nuisance
        let spec = nuisance("motion", None);
        assert_eq!(spec.name, "motion");
        assert!(spec.columns.is_none());

        // With columns
        let spec = nuisance("motion", Some(vec!["x", "y", "z"]));
        assert_eq!(spec.columns.unwrap().len(), 3);
    }

    #[test]
    fn test_block_spec() {
        // Basic blocks
        let spec = block(vec![100, 100, 100], None);
        assert_eq!(spec.block_lens, vec![100, 100, 100]);
        assert!(spec.names.is_none());

        // With names
        let spec = block(vec![100, 100], Some(vec!["run1", "run2"]));
        assert_eq!(spec.names.unwrap(), vec!["run1", "run2"]);
    }

    #[test]
    fn test_baseline_model_creation() {
        let frame = uniform_sampling_frame(100, 2.0, None).unwrap();
        let sampling = SamplingInfo::from_frame(frame);
        let spec = BaselineSpec::new(3, BasisType::Polynomial, InterceptType::Runwise);

        let model = baseline_model_from_spec(&sampling, &spec, None).unwrap();
        // The model contains drift, block, and nuisance terms
        assert!(model.terms.contains_key("drift"));
        assert!(model.terms.get("drift").unwrap().is_some());
    }

    #[test]
    fn test_block_diagonal() {
        let m1 = array![[1.0, 2.0], [3.0, 4.0]];
        let m2 = array![[5.0, 6.0], [7.0, 8.0]];

        let result = block_diagonal(&[m1, m2]).unwrap();
        assert_eq!(result.shape(), &[4, 4]);

        // Check block structure
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[2, 2]], 5.0);
        assert_eq!(result[[2, 3]], 6.0);

        // Check zeros
        assert_eq!(result[[0, 2]], 0.0);
        assert_eq!(result[[2, 0]], 0.0);
    }
}
