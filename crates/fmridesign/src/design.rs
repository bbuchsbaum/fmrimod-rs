//! Design matrix construction and extraction.

use crate::types::{Array, DesignMatrixProvider, ValidationError};
use ndarray::{Array1, Array2, Axis};
use std::fmt;

/// Design matrix with column names and time information.
#[derive(Debug, Clone)]
pub struct DesignMatrix {
    /// The design matrix data
    pub data: Array,
    /// Column names
    pub column_names: Vec<String>,
    /// Time points for each row (optional)
    pub row_times: ndarray::Array1<f64>,
}

impl DesignMatrix {
    /// Create a new design matrix.
    pub fn new(data: Array, column_names: Vec<String>) -> Self {
        assert_eq!(
            data.ncols(),
            column_names.len(),
            "Number of columns must match number of column names"
        );
        let n_rows = data.nrows();
        let row_times = ndarray::Array1::linspace(0.0, n_rows as f64 - 1.0, n_rows);
        Self {
            data,
            column_names,
            row_times,
        }
    }

    /// Create a new design matrix with time points.
    pub fn with_times(
        data: Array,
        column_names: Vec<String>,
        row_times: ndarray::Array1<f64>,
    ) -> Self {
        assert_eq!(
            data.ncols(),
            column_names.len(),
            "Number of columns must match number of column names"
        );
        assert_eq!(
            data.nrows(),
            row_times.len(),
            "Number of rows must match number of time points"
        );
        Self {
            data,
            column_names,
            row_times,
        }
    }

    /// Get number of rows (observations).
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    /// Get number of columns (predictors).
    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }
}

impl DesignMatrixProvider for DesignMatrix {
    fn design_matrix(&self) -> &Array {
        &self.data
    }

    fn column_names(&self) -> Option<&[String]> {
        Some(&self.column_names)
    }
}

impl fmt::Display for DesignMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DesignMatrix({} × {})", self.nrows(), self.ncols())?;
        if !self.column_names.is_empty() {
            write!(f, " columns: [")?;
            for (i, name) in self.column_names.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", name)?;
                if i >= 4 && self.column_names.len() > 6 {
                    write!(f, ", ... ({} more)", self.column_names.len() - 5)?;
                    break;
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

/// Extract design matrix from various types.
///
/// This trait provides a generic way to extract design matrices from
/// different model types in the library.
pub trait ExtractDesignMatrix {
    /// Extract the design matrix.
    fn extract_design_matrix(&self) -> Result<DesignMatrix, ValidationError>;

    /// Extract design matrix for specific blocks.
    fn extract_design_matrix_for_blocks(
        &self,
        _block_ids: &[usize],
    ) -> Result<DesignMatrix, ValidationError> {
        // Default implementation returns an error
        Err(ValidationError::InvalidParameter(
            "Block extraction not implemented for this type".to_string(),
        ))
    }
}

/// Builder for constructing design matrices.
#[derive(Debug)]
pub struct DesignMatrixBuilder {
    columns: Vec<Array1<f64>>,
    column_names: Vec<String>,
    row_times: Option<Array1<f64>>,
}

impl DesignMatrixBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            columns: Vec::new(),
            column_names: Vec::new(),
            row_times: None,
        }
    }

    /// Add a column to the design matrix.
    pub fn add_column(mut self, name: impl Into<String>, column: Array1<f64>) -> Self {
        self.column_names.push(name.into());
        self.columns.push(column);
        self
    }

    /// Add multiple columns.
    pub fn add_columns(
        mut self,
        names: Vec<String>,
        columns: Array2<f64>,
    ) -> Result<Self, ValidationError> {
        if names.len() != columns.ncols() {
            return Err(ValidationError::LengthMismatch {
                expected: names.len(),
                actual: columns.ncols(),
            });
        }

        for (i, name) in names.into_iter().enumerate() {
            self.column_names.push(name);
            self.columns.push(columns.column(i).to_owned());
        }

        Ok(self)
    }

    /// Set row times.
    pub fn with_row_times(mut self, times: Array1<f64>) -> Self {
        self.row_times = Some(times);
        self
    }

    /// Build the design matrix.
    pub fn build(self) -> Result<DesignMatrix, ValidationError> {
        if self.columns.is_empty() {
            return Err(ValidationError::EmptyArray);
        }

        // Check all columns have same length
        let n_rows = self.columns[0].len();
        for col in &self.columns {
            if col.len() != n_rows {
                return Err(ValidationError::LengthMismatch {
                    expected: n_rows,
                    actual: col.len(),
                });
            }
        }

        // Stack columns into matrix
        let mut data = Array2::zeros((n_rows, self.columns.len()));
        for (i, col) in self.columns.into_iter().enumerate() {
            data.column_mut(i).assign(&col);
        }

        // Use provided times or default
        let row_times = self
            .row_times
            .unwrap_or_else(|| Array1::linspace(0.0, n_rows as f64 - 1.0, n_rows));

        Ok(DesignMatrix::with_times(data, self.column_names, row_times))
    }
}

impl Default for DesignMatrixBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Concatenate design matrices horizontally (column-wise).
pub fn hstack_design_matrices(matrices: &[DesignMatrix]) -> Result<DesignMatrix, ValidationError> {
    if matrices.is_empty() {
        return Err(ValidationError::EmptyArray);
    }

    // Check all have same number of rows
    let n_rows = matrices[0].nrows();
    let reference_times = &matrices[0].row_times;
    for dm in matrices {
        if dm.nrows() != n_rows {
            return Err(ValidationError::DimensionError {
                expected: n_rows,
                actual: dm.nrows(),
            });
        }
        if dm.row_times.iter().any(|time| !time.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }
        if dm.row_times != *reference_times {
            return Err(ValidationError::InvalidParameter(
                "Cannot hstack design matrices with different row times".to_string(),
            ));
        }
    }

    // Collect all columns and names
    let mut all_columns = Vec::new();
    let mut all_names = Vec::new();

    for dm in matrices {
        for i in 0..dm.ncols() {
            all_columns.push(dm.data.column(i).to_owned());
            all_names.push(dm.column_names[i].clone());
        }
    }

    // Build combined matrix
    let mut builder = DesignMatrixBuilder::new();
    for (name, col) in all_names.into_iter().zip(all_columns) {
        builder = builder.add_column(name, col);
    }

    // Use times from first matrix
    if !matrices[0].row_times.is_empty() {
        builder = builder.with_row_times(matrices[0].row_times.clone());
    }

    builder.build()
}

/// Subset a design matrix by rows.
pub fn subset_rows(
    design_matrix: &DesignMatrix,
    indices: &[usize],
) -> Result<DesignMatrix, ValidationError> {
    // Validate indices
    let n_rows = design_matrix.nrows();
    for &idx in indices {
        if idx >= n_rows {
            return Err(ValidationError::InvalidParameter(format!(
                "Index {} out of bounds for matrix with {} rows",
                idx, n_rows
            )));
        }
    }

    // Create subset
    let subset_data = design_matrix.data.select(Axis(0), indices);
    let subset_times = indices
        .iter()
        .map(|&i| design_matrix.row_times[i])
        .collect();

    Ok(DesignMatrix::with_times(
        subset_data,
        design_matrix.column_names.clone(),
        subset_times,
    ))
}

/// Subset a design matrix by columns.
pub fn subset_columns(
    design_matrix: &DesignMatrix,
    indices: &[usize],
) -> Result<DesignMatrix, ValidationError> {
    // Validate indices
    let n_cols = design_matrix.ncols();
    for &idx in indices {
        if idx >= n_cols {
            return Err(ValidationError::InvalidParameter(format!(
                "Index {} out of bounds for matrix with {} columns",
                idx, n_cols
            )));
        }
    }

    // Create subset
    let subset_data = design_matrix.data.select(Axis(1), indices);
    let subset_names = indices
        .iter()
        .map(|&i| design_matrix.column_names[i].clone())
        .collect();

    Ok(DesignMatrix::with_times(
        subset_data,
        subset_names,
        design_matrix.row_times.clone(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_design_matrix_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let names = vec!["col1".to_string(), "col2".to_string()];

        let dm = DesignMatrix::new(data.clone(), names.clone());

        assert_eq!(dm.nrows(), 3);
        assert_eq!(dm.ncols(), 2);
        assert_eq!(dm.column_names, names);
        assert_eq!(dm.data, data);
    }

    #[test]
    fn test_design_matrix_with_times() {
        let data = array![[1.0, 2.0], [3.0, 4.0]];
        let names = vec!["A".to_string(), "B".to_string()];
        let times = array![0.0, 2.0];

        let dm = DesignMatrix::with_times(data.clone(), names.clone(), times.clone());

        assert_eq!(dm.row_times, times);
    }

    #[test]
    fn test_design_matrix_builder() {
        let col1 = array![1.0, 2.0, 3.0];
        let col2 = array![4.0, 5.0, 6.0];

        let dm = DesignMatrixBuilder::new()
            .add_column("x1", col1.clone())
            .add_column("x2", col2.clone())
            .build()
            .unwrap();

        assert_eq!(dm.nrows(), 3);
        assert_eq!(dm.ncols(), 2);
        assert_eq!(dm.column_names, vec!["x1", "x2"]);
        assert_eq!(dm.data.column(0), col1);
        assert_eq!(dm.data.column(1), col2);
    }

    #[test]
    fn test_builder_with_multiple_columns() {
        let cols = array![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
        let names = vec!["a".to_string(), "b".to_string()];

        let dm = DesignMatrixBuilder::new()
            .add_columns(names.clone(), cols.clone())
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(dm.column_names, names);
        assert_eq!(dm.data, cols);
    }

    #[test]
    fn test_hstack_design_matrices() {
        let dm1 = DesignMatrix::new(array![[1.0], [2.0], [3.0]], vec!["a".to_string()]);

        let dm2 = DesignMatrix::new(
            array![[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]],
            vec!["b".to_string(), "c".to_string()],
        );

        let combined = hstack_design_matrices(&[dm1, dm2]).unwrap();

        assert_eq!(combined.nrows(), 3);
        assert_eq!(combined.ncols(), 3);
        assert_eq!(combined.column_names, vec!["a", "b", "c"]);
        assert_eq!(combined.data.column(0), array![1.0, 2.0, 3.0]);
        assert_eq!(combined.data.column(1), array![4.0, 6.0, 8.0]);
        assert_eq!(combined.data.column(2), array![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subset_rows() {
        let dm = DesignMatrix::new(
            array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            vec!["x".to_string(), "y".to_string()],
        );

        let subset = subset_rows(&dm, &[0, 2]).unwrap();

        assert_eq!(subset.nrows(), 2);
        assert_eq!(subset.ncols(), 2);
        assert_eq!(subset.data, array![[1.0, 2.0], [5.0, 6.0]]);
    }

    #[test]
    fn test_subset_columns() {
        let dm = DesignMatrix::new(
            array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            vec!["a".to_string(), "b".to_string(), "c".to_string()],
        );

        let subset = subset_columns(&dm, &[0, 2]).unwrap();

        assert_eq!(subset.nrows(), 2);
        assert_eq!(subset.ncols(), 2);
        assert_eq!(subset.column_names, vec!["a", "c"]);
        assert_eq!(subset.data, array![[1.0, 3.0], [4.0, 6.0]]);
    }

    #[test]
    fn test_display() {
        let dm = DesignMatrix::new(
            array![[1.0, 2.0]],
            vec!["col1".to_string(), "col2".to_string()],
        );

        let display = format!("{}", dm);
        assert!(display.contains("DesignMatrix(1 × 2)"));
        assert!(display.contains("col1"));
        assert!(display.contains("col2"));
    }
}
