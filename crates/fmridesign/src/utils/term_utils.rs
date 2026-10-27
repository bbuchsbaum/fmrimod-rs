//! Utility functions for working with model terms and design matrices.

use crate::types::Array;
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

/// Column indices for each term in a model.
#[derive(Debug, Clone)]
pub struct TermIndices {
    /// Map from term names to column indices
    pub indices: HashMap<String, Vec<usize>>,
}

impl TermIndices {
    /// Create new empty term indices
    pub fn new() -> Self {
        Self {
            indices: HashMap::new(),
        }
    }

    /// Add indices for a term
    pub fn add_term(&mut self, name: String, indices: Vec<usize>) {
        self.indices.insert(name, indices);
    }

    /// Get indices for a term
    pub fn get(&self, name: &str) -> Option<&Vec<usize>> {
        self.indices.get(name)
    }

    /// Check if a term exists
    pub fn contains(&self, name: &str) -> bool {
        self.indices.contains_key(name)
    }

    /// Get all term names
    pub fn term_names(&self) -> Vec<&str> {
        self.indices.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for TermIndices {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract column indices for each term in a model.
///
/// Maps each term name to the column indices it occupies in the
/// design matrix. This is useful for extracting term-specific
/// sub-matrices or computing term-wise statistics.
pub fn extract_term_indices(term_names: &[String], column_names: &[String]) -> TermIndices {
    let mut term_indices = TermIndices::new();
    let mut assigned: Vec<bool> = vec![false; column_names.len()];

    // Process each term
    for term_name in term_names {
        let mut indices = Vec::new();

        // Find columns belonging to this term
        for (i, col_name) in column_names.iter().enumerate() {
            if assigned[i] {
                continue;
            }

            // Check if column belongs to this term
            if col_name.starts_with(&format!("{}.", term_name))  // factor levels
                || col_name.starts_with(&format!("{}:", term_name))  // interactions
                || col_name == term_name  // exact match
                || col_name.starts_with(&format!("{}_", term_name))
            // basis functions
            {
                indices.push(i);
                assigned[i] = true;
            }
        }

        if !indices.is_empty() {
            term_indices.add_term(term_name.clone(), indices);
        }
    }

    // Handle unassigned columns
    let unassigned: Vec<usize> = assigned
        .iter()
        .enumerate()
        .filter_map(|(i, &assigned)| if !assigned { Some(i) } else { None })
        .collect();

    if !unassigned.is_empty() {
        term_indices.add_term("baseline".to_string(), unassigned);
    }

    term_indices
}

/// Extract term-specific sub-matrices from the design matrix.
///
/// Returns a map from term names to their corresponding
/// columns in the design matrix.
pub fn extract_term_matrices(
    design_matrix: &Array,
    term_indices: &TermIndices,
    term_names: Option<&[String]>,
) -> HashMap<String, Array2<f64>> {
    let mut matrices = HashMap::new();

    // Determine which terms to extract
    let terms_to_extract: Vec<&str> = if let Some(names) = term_names {
        names.iter().map(|s| s.as_str()).collect()
    } else {
        term_indices.term_names()
    };

    // Extract sub-matrices
    for term in terms_to_extract {
        if let Some(indices) = term_indices.get(term) {
            // Create view of selected columns
            let cols: Vec<usize> = indices.clone();
            let submatrix = design_matrix.select(Axis(1), &cols);
            matrices.insert(term.to_string(), submatrix);
        }
    }

    matrices
}

/// Extract baseline/nuisance terms from a model.
///
/// Returns the columns of the design matrix corresponding to
/// baseline terms (drift, motion regressors, etc.).
pub fn extract_baseline_terms(
    design_matrix: &Array,
    term_indices: &TermIndices,
) -> Option<Array2<f64>> {
    if let Some(baseline_indices) = term_indices.get("baseline") {
        let cols: Vec<usize> = baseline_indices.clone();
        Some(design_matrix.select(Axis(1), &cols))
    } else {
        None
    }
}

/// Split design matrix by terms, preserving order.
///
/// Returns a vector of (term_name, sub_matrix) tuples in the
/// order terms appear.
pub fn split_by_term(
    design_matrix: &Array,
    term_indices: &TermIndices,
    term_order: &[String],
) -> Vec<(String, Array2<f64>)> {
    let mut result = Vec::new();

    // Process terms in order
    for term_name in term_order {
        if let Some(indices) = term_indices.get(term_name) {
            let cols: Vec<usize> = indices.clone();
            let submatrix = design_matrix.select(Axis(1), &cols);
            result.push((term_name.clone(), submatrix));
        }
    }

    // Add baseline if present and not already included
    if !term_order.contains(&"baseline".to_string()) {
        if let Some(indices) = term_indices.get("baseline") {
            let cols: Vec<usize> = indices.clone();
            let submatrix = design_matrix.select(Axis(1), &cols);
            result.push(("baseline".to_string(), submatrix));
        }
    }

    result
}

/// Calculate variance explained by each term.
///
/// For each term, computes the variance of the fitted values
/// when using only that term's columns.
pub fn variance_explained(
    y: &Array1<f64>,
    design_matrix: &Array,
    term_indices: &TermIndices,
) -> HashMap<String, f64> {
    use nalgebra::DMatrix;

    let mut var_explained = HashMap::new();

    for (term_name, indices) in &term_indices.indices {
        // Extract term columns
        let cols: Vec<usize> = indices.clone();
        let x_term = design_matrix.select(Axis(1), &cols);

        let nrows = x_term.nrows();
        let ncols = x_term.ncols();

        // Convert to nalgebra for SVD least squares
        let na_x = DMatrix::from_fn(nrows, ncols, |i, j| x_term[[i, j]]);
        let na_y = nalgebra::DVector::from_fn(nrows, |i, _| y[i]);

        let svd = na_x.svd(true, true);
        match svd.solve(&na_y, 1e-14) {
            Ok(solution) => {
                let na_fitted = DMatrix::from_fn(nrows, ncols, |i, j| x_term[[i, j]]) * &solution;
                let fitted_var = {
                    let mean: f64 = na_fitted.iter().sum::<f64>() / nrows as f64;
                    na_fitted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nrows as f64
                };
                var_explained.insert(term_name.clone(), fitted_var);
            }
            Err(_) => {
                var_explained.insert(term_name.clone(), 0.0);
            }
        }
    }

    var_explained
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_extract_term_indices() {
        let term_names = vec!["condition".to_string(), "rating".to_string()];
        let column_names = vec![
            "condition.A".to_string(),
            "condition.B".to_string(),
            "rating".to_string(),
            "intercept".to_string(),
        ];

        let indices = extract_term_indices(&term_names, &column_names);

        assert_eq!(indices.get("condition"), Some(&vec![0, 1]));
        assert_eq!(indices.get("rating"), Some(&vec![2]));
        assert_eq!(indices.get("baseline"), Some(&vec![3]));
    }

    #[test]
    fn test_split_by_term() {
        let design_matrix = array![
            [1.0, 0.0, 0.5, 1.0],
            [0.0, 1.0, 0.3, 1.0],
            [1.0, 0.0, 0.7, 1.0],
        ];

        let mut term_indices = TermIndices::new();
        term_indices.add_term("condition".to_string(), vec![0, 1]);
        term_indices.add_term("rating".to_string(), vec![2]);
        term_indices.add_term("baseline".to_string(), vec![3]);

        let term_order = vec!["condition".to_string(), "rating".to_string()];
        let split = split_by_term(&design_matrix, &term_indices, &term_order);

        assert_eq!(split.len(), 3);
        assert_eq!(split[0].0, "condition");
        assert_eq!(split[0].1.shape(), &[3, 2]);
        assert_eq!(split[1].0, "rating");
        assert_eq!(split[1].1.shape(), &[3, 1]);
        assert_eq!(split[2].0, "baseline");
        assert_eq!(split[2].1.shape(), &[3, 1]);
    }
}
