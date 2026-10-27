//! Split convolved design matrices by condition.
//!
//! Matches R's `condition_basis_list()` from `fmri_design/R/condition_basis_list.R`.
//! Takes a convolved design matrix and splits it into per-condition basis matrices.

use crate::naming;
use ndarray::Array2;
use std::collections::HashMap;

/// Split a convolved design matrix into per-condition basis matrices.
///
/// Given a design matrix with columns named according to the fmri_design
/// naming convention (`term_tag_condition_tag[_b##]`), this function
/// groups columns by base condition and returns a HashMap mapping each
/// condition tag to its T x d sub-matrix (where d = number of basis functions).
///
/// # Arguments
/// * `design_matrix` - The convolved design matrix (T x C)
/// * `column_names` - Column names matching the design matrix columns
/// * `condition_tags` - Base condition tags (without basis suffix)
/// * `term_tag` - Optional term tag prefix
/// * `nbasis` - Number of basis functions per condition
///
/// # Returns
/// HashMap mapping condition tag to its basis matrix (T x nbasis).
pub fn condition_basis_list(
    design_matrix: &Array2<f64>,
    column_names: &[String],
    condition_tags: &[String],
    term_tag: Option<&str>,
    nbasis: usize,
) -> HashMap<String, Array2<f64>> {
    let mut result = HashMap::new();
    let n_rows = design_matrix.nrows();

    for cond_tag in condition_tags {
        // Generate expected column names for this condition
        let expected_names = naming::make_column_names(term_tag, &[cond_tag.clone()], nbasis);

        // Find matching column indices
        let col_indices: Vec<usize> = expected_names
            .iter()
            .filter_map(|name| column_names.iter().position(|cn| cn == name))
            .collect();

        // Skip if not all expected columns found
        if col_indices.len() != expected_names.len() {
            continue;
        }

        // Extract sub-matrix
        let mut sub_matrix = Array2::zeros((n_rows, col_indices.len()));
        for (out_j, &src_j) in col_indices.iter().enumerate() {
            sub_matrix
                .column_mut(out_j)
                .assign(&design_matrix.column(src_j));
        }

        result.insert(cond_tag.clone(), sub_matrix);
    }

    result
}

/// Split a design matrix by condition using column name pattern matching.
///
/// A simpler version that uses the basis suffix pattern `_b##` to detect
/// multi-basis columns, without requiring explicit condition tags.
pub fn split_by_condition(
    design_matrix: &Array2<f64>,
    column_names: &[String],
) -> HashMap<String, Array2<f64>> {
    let basis_re = regex::Regex::new(r"_b\d+$").unwrap();
    let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

    for (i, name) in column_names.iter().enumerate() {
        let base = basis_re.replace(name, "").to_string();
        groups.entry(base).or_default().push(i);
    }

    let n_rows = design_matrix.nrows();
    let mut result = HashMap::new();

    for (cond, indices) in groups {
        let mut sub = Array2::zeros((n_rows, indices.len()));
        for (out_j, &src_j) in indices.iter().enumerate() {
            sub.column_mut(out_j).assign(&design_matrix.column(src_j));
        }
        result.insert(cond, sub);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_condition_basis_list_single_basis() {
        let dm = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let names = vec![
            "cond_A".to_string(),
            "cond_B".to_string(),
            "cond_C".to_string(),
        ];
        let tags = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = condition_basis_list(&dm, &names, &tags, Some("cond"), 1);

        assert_eq!(result.len(), 3);
        assert_eq!(result["A"].dim(), (3, 1));
        assert_eq!(result["A"][[0, 0]], 1.0);
        assert_eq!(result["B"][[0, 0]], 2.0);
        assert_eq!(result["C"][[0, 0]], 3.0);
    }

    #[test]
    fn test_condition_basis_list_multi_basis() {
        // 2 conditions, 2 basis functions each = 4 columns
        let dm = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let names = vec![
            "cond_A_b01".to_string(),
            "cond_A_b02".to_string(),
            "cond_B_b01".to_string(),
            "cond_B_b02".to_string(),
        ];
        let tags = vec!["A".to_string(), "B".to_string()];

        let result = condition_basis_list(&dm, &names, &tags, Some("cond"), 2);

        assert_eq!(result.len(), 2);
        assert_eq!(result["A"].dim(), (2, 2));
        assert_eq!(result["A"][[0, 0]], 1.0);
        assert_eq!(result["A"][[0, 1]], 2.0);
        assert_eq!(result["B"][[0, 0]], 3.0);
        assert_eq!(result["B"][[0, 1]], 4.0);
    }

    #[test]
    fn test_condition_basis_list_missing_condition() {
        let dm = array![[1.0, 2.0], [3.0, 4.0]];
        let names = vec!["cond_A".to_string(), "cond_B".to_string()];
        let tags = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        let result = condition_basis_list(&dm, &names, &tags, Some("cond"), 1);

        // C is missing from design matrix, should be skipped
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("A"));
        assert!(result.contains_key("B"));
        assert!(!result.contains_key("C"));
    }

    #[test]
    fn test_split_by_condition() {
        let dm = array![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let names = vec![
            "cond_A_b01".to_string(),
            "cond_A_b02".to_string(),
            "cond_B_b01".to_string(),
            "cond_B_b02".to_string(),
        ];

        let result = split_by_condition(&dm, &names);

        assert_eq!(result.len(), 2);
        assert_eq!(result["cond_A"].dim(), (2, 2));
        assert_eq!(result["cond_B"].dim(), (2, 2));
    }

    #[test]
    fn test_split_by_condition_no_basis() {
        let dm = array![[1.0, 2.0], [3.0, 4.0]];
        let names = vec!["face".to_string(), "house".to_string()];

        let result = split_by_condition(&dm, &names);

        assert_eq!(result.len(), 2);
        assert_eq!(result["face"].dim(), (2, 1));
        assert_eq!(result["house"].dim(), (2, 1));
    }
}
