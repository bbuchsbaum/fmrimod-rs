//! Regressor extraction functionality.
//!
//! This module provides functions for extracting regressors from event models
//! and design matrices.

use ndarray::Array2;
use regex::Regex;
use std::collections::HashMap;

use crate::core::{DataFrame, EventModel};
use crate::types::{Array, ValidationError};
// EventTerm used when regressor extraction is fully implemented

/// Options for regressor extraction
#[derive(Debug, Clone)]
pub struct RegressorOptions {
    /// Include baseline/nuisance regressors
    pub include_baseline: bool,
    /// Return as dictionary/map format
    pub as_dict: bool,
}

impl Default for RegressorOptions {
    fn default() -> Self {
        Self {
            include_baseline: true,
            as_dict: false,
        }
    }
}

/// Pattern for matching regressors
#[derive(Debug, Clone)]
pub enum RegressorPattern {
    /// Match exact term name
    Exact(String),
    /// Match using regex pattern
    Regex(Regex),
    /// Match multiple terms
    Multiple(Vec<String>),
}

impl From<&str> for RegressorPattern {
    fn from(s: &str) -> Self {
        RegressorPattern::Exact(s.to_string())
    }
}

impl From<String> for RegressorPattern {
    fn from(s: String) -> Self {
        RegressorPattern::Exact(s)
    }
}

impl From<Regex> for RegressorPattern {
    fn from(r: Regex) -> Self {
        RegressorPattern::Regex(r)
    }
}

impl From<Vec<String>> for RegressorPattern {
    fn from(v: Vec<String>) -> Self {
        RegressorPattern::Multiple(v)
    }
}

impl From<Vec<&str>> for RegressorPattern {
    fn from(v: Vec<&str>) -> Self {
        RegressorPattern::Multiple(v.into_iter().map(|s| s.to_string()).collect())
    }
}

/// Result of regressor extraction
#[derive(Debug, Clone)]
pub enum RegressorResult {
    /// Regressors as a 2D array with column names
    Array { data: Array, columns: Vec<String> },
    /// Regressors as a map from column names to column vectors
    Dict(HashMap<String, Vec<f64>>),
    /// Regressors as a DataFrame
    DataFrame(DataFrame),
}

/// Extract regressors from an event model.
///
/// This function extracts specific regressors or groups of regressors
/// from an EventModel's design matrix. Useful for extracting subsets
/// of the design matrix for specific analyses or visualization.
///
/// # Arguments
///
/// * `model` - The event model containing the design matrix
/// * `pattern` - Optional pattern to match columns (None returns all)
/// * `options` - Extraction options
///
/// # Returns
///
/// Extracted regressors in the requested format
///
/// # Examples
///
/// ```ignore
/// // Get all regressors
/// let all_regs = regressors(&model, None, &Default::default())?;
///
/// // Get specific term's regressors
/// let visual_regs = regressors(&model, Some("visual".into()), &Default::default())?;
///
/// // Use regex pattern
/// let pattern = Regex::new(r"cond_.*")?;
/// let cond_regs = regressors(&model, Some(pattern.into()), &Default::default())?;
///
/// // Get as dictionary
/// let options = RegressorOptions { as_dict: true, ..Default::default() };
/// let reg_dict = regressors(&model, None, &options)?;
/// ```
pub fn regressors(
    model: &EventModel,
    pattern: Option<RegressorPattern>,
    options: &RegressorOptions,
) -> Result<RegressorResult, ValidationError> {
    // Get design matrix and column names
    let design_matrix = model.generate_design_matrix()?;
    let col_names = model.column_names();
    let n_cols = col_names.len();

    // Determine which columns to include
    let col_indices = match pattern {
        None => {
            // Include all columns
            if options.include_baseline {
                (0..n_cols).collect::<Vec<_>>()
            } else {
                // Exclude baseline columns
                filter_baseline_indices(&col_names)
            }
        }
        Some(pattern) => {
            let mut indices = match pattern {
                RegressorPattern::Exact(term) => find_term_indices(&col_names, &term),
                RegressorPattern::Regex(regex) => find_regex_indices(&col_names, &regex),
                RegressorPattern::Multiple(terms) => {
                    let mut all_indices = Vec::new();
                    for term in &terms {
                        all_indices.extend(find_term_indices(&col_names, term));
                    }
                    // Remove duplicates while preserving order
                    let mut seen = std::collections::HashSet::new();
                    all_indices
                        .into_iter()
                        .filter(|&i| seen.insert(i))
                        .collect()
                }
            };

            // Apply baseline filter if needed
            if !options.include_baseline {
                indices = indices
                    .into_iter()
                    .filter(|&i| !is_baseline_column(&col_names[i]))
                    .collect();
            }

            indices
        }
    };

    // Check if any columns were found
    if col_indices.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "No columns found matching the specified pattern".to_string(),
        ));
    }

    // Extract selected columns
    let selected_cols: Vec<String> = col_indices.iter().map(|&i| col_names[i].clone()).collect();

    // Build result based on requested format
    if options.as_dict {
        let mut dict = HashMap::new();
        for (idx, &col_idx) in col_indices.iter().enumerate() {
            let col_data = design_matrix.column(col_idx).to_vec();
            dict.insert(selected_cols[idx].clone(), col_data);
        }
        Ok(RegressorResult::Dict(dict))
    } else {
        // Create sub-matrix with selected columns
        let n_rows = design_matrix.nrows();
        let n_selected = col_indices.len();
        let mut sub_matrix = Array2::zeros((n_rows, n_selected));

        for (new_idx, &old_idx) in col_indices.iter().enumerate() {
            sub_matrix
                .column_mut(new_idx)
                .assign(&design_matrix.column(old_idx));
        }

        Ok(RegressorResult::Array {
            data: sub_matrix,
            columns: selected_cols,
        })
    }
}

/// Get names of regressors matching criteria.
///
/// Convenience function to get just the names without extracting
/// the actual regressor values.
///
/// # Arguments
///
/// * `model` - The event model
/// * `pattern` - Optional pattern to match columns
/// * `include_baseline` - Whether to include baseline/nuisance regressors
///
/// # Returns
///
/// Names of matching regressors
pub fn regressor_names(
    model: &EventModel,
    pattern: Option<RegressorPattern>,
    include_baseline: bool,
) -> Result<Vec<String>, ValidationError> {
    let options = RegressorOptions {
        include_baseline,
        as_dict: false,
    };

    match regressors(model, pattern, &options)? {
        RegressorResult::Array { columns, .. } => Ok(columns),
        RegressorResult::DataFrame(df) => Ok(df.column_names().to_vec()),
        _ => unreachable!("Non-dict result expected"),
    }
}

/// Extract regressors grouped by term.
///
/// This function groups regressors by their originating term,
/// making it easy to work with all regressors from each term
/// separately.
///
/// # Arguments
///
/// * `model` - The event model
/// * `as_arrays` - If true, return arrays; if false, return column names only
///
/// # Returns
///
/// Dictionary mapping term names to their regressors
pub fn term_regressors(
    model: &EventModel,
    as_arrays: bool,
) -> Result<HashMap<String, RegressorResult>, ValidationError> {
    let mut result = HashMap::new();

    // Get all column names to identify unique terms
    let col_names = model.column_names();
    let mut term_names = std::collections::HashSet::new();

    // Extract term names from column names
    for col_name in &col_names {
        // Extract term name using various patterns
        let term_name = if let Some(pos) = col_name.find('[') {
            // Handle "term[level]" pattern
            col_name[..pos].to_string()
        } else if let Some(pos) = col_name.find('.') {
            // Handle "term.level" or "term.1" pattern
            col_name[..pos].to_string()
        } else if let Some(pos) = col_name.find('_') {
            // Handle "term_basis" pattern
            // But be careful not to split actual term names with underscores
            let candidate = &col_name[..pos];
            // Check if this looks like a basis suffix
            if col_name[pos + 1..]
                .chars()
                .next()
                .map_or(false, |c| c.is_digit(10))
            {
                candidate.to_string()
            } else {
                col_name.clone()
            }
        } else {
            // Use full name
            col_name.clone()
        };

        // Skip baseline-related terms
        if !is_baseline_term(&term_name) {
            term_names.insert(term_name);
        }
    }

    // Extract regressors for each term
    for term_name in term_names {
        let pattern = RegressorPattern::Exact(term_name.clone());
        let options = RegressorOptions {
            include_baseline: false,
            as_dict: false,
        };

        match regressors(model, Some(pattern), &options) {
            Ok(reg_result) => {
                if as_arrays {
                    result.insert(term_name, reg_result);
                } else {
                    // Just store column names
                    match reg_result {
                        RegressorResult::Array { columns, .. } => {
                            result.insert(
                                term_name,
                                RegressorResult::Dict(
                                    columns.into_iter().map(|c| (c, Vec::new())).collect(),
                                ),
                            );
                        }
                        _ => {}
                    }
                }
            }
            Err(_) => {
                // Skip terms with no regressors
                continue;
            }
        }
    }

    // Add baseline regressors if present
    let baseline_indices = (0..col_names.len())
        .filter(|&i| is_baseline_column(&col_names[i]))
        .collect::<Vec<_>>();

    if !baseline_indices.is_empty() {
        let baseline_cols: Vec<String> = baseline_indices
            .iter()
            .map(|&i| col_names[i].clone())
            .collect();

        if as_arrays {
            let design_matrix = model.generate_design_matrix()?;
            let n_rows = design_matrix.nrows();
            let n_baseline = baseline_indices.len();
            let mut baseline_matrix = Array2::zeros((n_rows, n_baseline));

            for (new_idx, &old_idx) in baseline_indices.iter().enumerate() {
                baseline_matrix
                    .column_mut(new_idx)
                    .assign(&design_matrix.column(old_idx));
            }

            result.insert(
                "_baseline".to_string(),
                RegressorResult::Array {
                    data: baseline_matrix,
                    columns: baseline_cols,
                },
            );
        } else {
            result.insert(
                "_baseline".to_string(),
                RegressorResult::Dict(baseline_cols.into_iter().map(|c| (c, Vec::new())).collect()),
            );
        }
    }

    Ok(result)
}

/// Extract baseline/nuisance terms from a model.
///
/// Returns the columns of the design matrix corresponding to
/// baseline terms (drift, motion regressors, etc.).
///
/// # Arguments
///
/// * `model` - Model potentially containing baseline terms
///
/// # Returns
///
/// Baseline term matrix or None if no baseline
pub fn baseline_terms(model: &EventModel) -> Result<Option<Array>, ValidationError> {
    let col_names = model.column_names();
    let baseline_indices: Vec<usize> = (0..col_names.len())
        .filter(|&i| is_baseline_column(&col_names[i]))
        .collect();

    if baseline_indices.is_empty() {
        return Ok(None);
    }

    let design_matrix = model.generate_design_matrix()?;
    let n_rows = design_matrix.nrows();
    let n_baseline = baseline_indices.len();
    let mut baseline_matrix = Array2::zeros((n_rows, n_baseline));

    for (new_idx, &old_idx) in baseline_indices.iter().enumerate() {
        baseline_matrix
            .column_mut(new_idx)
            .assign(&design_matrix.column(old_idx));
    }

    Ok(Some(baseline_matrix))
}

/// Split design matrix by terms, preserving order.
///
/// Returns a vector of (term_name, sub_matrix) tuples in the
/// order terms appear in the model.
pub fn split_by_term(model: &EventModel) -> Result<Vec<(String, Array)>, ValidationError> {
    let mut result = Vec::new();

    // Get column indices for each term
    let term_indices = model.column_indices()?;

    // Get design matrix
    let design_matrix = model.generate_design_matrix()?;

    // Process each term in order
    for (term_name, indices) in &term_indices {
        if indices.is_empty() {
            continue;
        }

        // Extract columns for this term
        let n_rows = design_matrix.nrows();
        let n_cols = indices.len();
        let mut term_matrix = Array2::zeros((n_rows, n_cols));

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            term_matrix
                .column_mut(new_idx)
                .assign(&design_matrix.column(old_idx));
        }

        result.push((term_name.clone(), term_matrix));
    }

    Ok(result)
}

// Helper functions

fn find_term_indices(col_names: &[String], term: &str) -> Vec<usize> {
    let mut indices = Vec::new();

    // Create pattern to match term names
    // This handles "term", "term[level]", "term.level", "term_1", etc.
    for (i, col_name) in col_names.iter().enumerate() {
        if col_name == term {
            // Exact match
            indices.push(i);
        } else if col_name.starts_with(&format!("{}[", term))
            || col_name.starts_with(&format!("{}.", term))
            || col_name.starts_with(&format!("{}_", term))
            || col_name.starts_with(&format!("{}:", term))
        {
            // Prefix match with delimiter
            indices.push(i);
        }
    }

    indices
}

fn find_regex_indices(col_names: &[String], regex: &Regex) -> Vec<usize> {
    col_names
        .iter()
        .enumerate()
        .filter(|(_, name)| regex.is_match(name))
        .map(|(i, _)| i)
        .collect()
}

fn filter_baseline_indices(col_names: &[String]) -> Vec<usize> {
    (0..col_names.len())
        .filter(|&i| !is_baseline_column(&col_names[i]))
        .collect()
}

fn is_baseline_column(name: &str) -> bool {
    let lower = name.to_lowercase();
    ["baseline", "drift", "intercept", "block", "nuisance"]
        .iter()
        .any(|&pattern| lower.contains(pattern))
}

fn is_baseline_term(name: &str) -> bool {
    let lower = name.to_lowercase();
    [
        "baseline",
        "drift",
        "intercept",
        "block",
        "nuisance",
        "_baseline",
    ]
    .iter()
    .any(|&pattern| lower == pattern)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventFactor, EventTerm, EventVariable};
    use crate::sampling::uniform_sampling_frame;

    #[test]
    fn test_regressor_extraction() {
        // Create a simple model
        let event1 = EventFactor::new(
            "condition",
            vec![0.0, 10.0, 20.0],
            vec!["A", "B", "A"],
            vec![1.0; 3],
        )
        .unwrap();

        let event2 = EventVariable::new(
            "rating",
            vec![0.0, 10.0, 20.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0; 3],
        )
        .unwrap();

        let term1 = EventTerm::new(vec![Box::new(event1)], false).unwrap();
        let term2 = EventTerm::new(vec![Box::new(event2)], false).unwrap();

        let sampling = uniform_sampling_frame(15, 2.0, Some(0.0)).unwrap();
        let model = EventModel::new(vec![Box::new(term1), Box::new(term2)], sampling).unwrap();

        // Test extracting all regressors
        let all_regs = regressors(&model, None, &Default::default()).unwrap();
        match all_regs {
            RegressorResult::Array { columns, .. } => {
                println!("Actual column names: {:?}", columns);
                // Column names now follow R's varname.level convention
                assert!(
                    columns.len() >= 3,
                    "Expected at least 3 columns, got {}",
                    columns.len()
                );
                assert!(
                    columns.iter().any(|c| c == "condition.A"),
                    "Expected condition.A, got {:?}",
                    columns
                );
                assert!(
                    columns.iter().any(|c| c == "condition.B"),
                    "Expected condition.B, got {:?}",
                    columns
                );
                assert!(
                    columns.iter().any(|c| c == "rating"),
                    "Expected rating, got {:?}",
                    columns
                );
            }
            _ => panic!("Expected Array result"),
        }

        // Test extracting specific term by prefix
        let cond_regs = regressors(&model, Some("condition".into()), &Default::default()).unwrap();

        match cond_regs {
            RegressorResult::Array { columns, data } => {
                assert_eq!(columns.len(), 2);
                assert!(columns.contains(&"condition.A".to_string()));
                assert!(columns.contains(&"condition.B".to_string()));
                assert_eq!(data.ncols(), 2);
            }
            _ => panic!("Expected Array result"),
        }
    }

    #[test]
    fn test_regressor_names() {
        let event = EventFactor::new(
            "task",
            vec![0.0, 5.0, 10.0],
            vec!["easy", "hard", "easy"],
            vec![1.0; 3],
        )
        .unwrap();

        let term = EventTerm::new(vec![Box::new(event)], false).unwrap();
        let sampling = uniform_sampling_frame(15, 1.0, Some(0.0)).unwrap();
        let mut model = EventModel::new(vec![Box::new(term)], sampling).unwrap();
        model.generate_design_matrix().unwrap();

        let names = regressor_names(&model, None, true).unwrap();
        println!("Task regressor names: {:?}", names);
        assert_eq!(names.len(), 2);
        // Names might be like "task[easy]" or similar
        assert!(names.iter().any(|n| n.contains("easy")));
        assert!(names.iter().any(|n| n.contains("hard")));
    }

    #[test]
    fn test_regex_pattern() {
        let event1 = EventFactor::new(
            "cond_visual",
            vec![0.0, 10.0],
            vec!["vis_on", "vis_off"],
            vec![1.0; 2],
        )
        .unwrap();

        let event2 = EventFactor::new(
            "cond_audio",
            vec![5.0, 15.0],
            vec!["aud_on", "aud_off"],
            vec![1.0; 2],
        )
        .unwrap();

        let term1 = EventTerm::new(vec![Box::new(event1)], false).unwrap();
        let term2 = EventTerm::new(vec![Box::new(event2)], false).unwrap();

        let sampling = uniform_sampling_frame(20, 1.0, Some(0.0)).unwrap();
        let model = EventModel::new(vec![Box::new(term1), Box::new(term2)], sampling).unwrap();

        // Get all column names to verify structure
        let all_regs = regressors(&model, None, &Default::default()).unwrap();
        match all_regs {
            RegressorResult::Array { columns, .. } => {
                println!("Regex test column names: {:?}", columns);
                assert_eq!(columns.len(), 4); // 2 levels x 2 terms
                                              // Use regex to match visual columns
                let regex = Regex::new(r"vis_").unwrap();
                let vis_regs = regressors(&model, Some(regex.into()), &Default::default()).unwrap();
                match vis_regs {
                    RegressorResult::Array {
                        columns: vis_cols, ..
                    } => {
                        assert_eq!(vis_cols.len(), 2);
                    }
                    _ => panic!("Expected Array result"),
                }
            }
            _ => panic!("Expected Array result"),
        }
    }
}
