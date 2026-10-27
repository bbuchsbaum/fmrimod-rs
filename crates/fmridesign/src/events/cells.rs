//! Cell and condition extraction for event terms.

use super::EventTerm;
use crate::naming::{continuous_token, level_token, make_cond_tag};
use crate::types::{EventType, ValidationError};
use itertools::Itertools;
use std::collections::HashMap;

/// Represents cells (factor combinations) for an event term.
#[derive(Debug, Clone)]
pub struct Cells {
    /// Column names (factor names)
    pub columns: Vec<String>,
    /// Rows of cell values
    pub rows: Vec<Vec<String>>,
    /// Count of observations for each cell
    pub counts: Vec<usize>,
}

impl Cells {
    /// Create a new Cells structure.
    pub fn new(columns: Vec<String>, rows: Vec<Vec<String>>, counts: Vec<usize>) -> Self {
        assert_eq!(rows.len(), counts.len());
        if !rows.is_empty() {
            assert_eq!(rows[0].len(), columns.len());
        }
        Self {
            columns,
            rows,
            counts,
        }
    }

    /// Check if this represents continuous-only cells.
    pub fn is_continuous_only(&self) -> bool {
        self.rows.len() == 1 && self.rows[0].iter().all(|v| v.is_empty())
    }

    /// Filter to non-empty cells.
    pub fn drop_empty(&self) -> Self {
        let non_empty: Vec<(usize, &Vec<String>)> = self
            .rows
            .iter()
            .enumerate()
            .filter(|(i, _)| self.counts[*i] > 0)
            .map(|(i, row)| (i, row))
            .collect();

        let rows = non_empty.iter().map(|(_, row)| (*row).clone()).collect();
        let counts = non_empty.iter().map(|(i, _)| self.counts[*i]).collect();

        Self::new(self.columns.clone(), rows, counts)
    }

    /// Convert to Vec<Vec<String>> for easy manipulation
    pub fn into_cells(self) -> Vec<Vec<String>> {
        self.rows
    }
}

impl EventTerm {
    /// Internal helper to compute full cells table (including empty cells).
    fn compute_cells(&self) -> Result<Cells, ValidationError> {
        // Separate categorical and continuous events
        let (categorical_events, continuous_events): (Vec<_>, Vec<_>) = self
            .events()
            .iter()
            .partition(|e| e.event_type() == EventType::Categorical);

        // Handle continuous-only case
        if categorical_events.is_empty() {
            let mut columns = Vec::new();

            for event in continuous_events.iter() {
                match event.event_type() {
                    EventType::Matrix => {
                        // Multiple columns - would need column names from EventMatrix
                        // For now, use generic names
                        if let Some(n_cols) = event.n_columns() {
                            for i in 0..n_cols {
                                columns.push(format!("{}_col{}", event.name(), i + 1));
                            }
                        } else {
                            columns.push(event.name().to_string());
                        }
                    }
                    _ => {
                        // Single column
                        columns.push(event.name().to_string());
                    }
                }
            }

            // Create single row with empty values
            let row = vec!["".to_string(); columns.len()];
            let count = if !self.events().is_empty() {
                self.events()[0].onsets().len()
            } else {
                0
            };

            return Ok(Cells::new(columns, vec![row], vec![count]));
        }

        // Handle categorical case
        // Get levels for each categorical event
        let mut all_levels = Vec::new();
        let mut columns = Vec::new();

        for event in &categorical_events {
            columns.push(event.name().to_string());
            if let Some(levels) = event.levels() {
                all_levels.push(levels.to_vec());
            } else {
                return Err(ValidationError::InvalidParameter(format!(
                    "Categorical event {} has no levels",
                    event.name()
                )));
            }
        }

        // Generate all combinations using itertools
        let combinations: Vec<Vec<String>> = all_levels
            .iter()
            .map(|levels| levels.iter().map(|s| s.to_string()))
            .multi_cartesian_product()
            .collect();

        // Count occurrences
        // First, get the observed combinations
        let n_obs = categorical_events[0].onsets().len();
        let mut observed_counts: HashMap<Vec<String>, usize> = HashMap::new();

        for i in 0..n_obs {
            let mut obs_combo = Vec::new();
            for event in &categorical_events {
                // Get the level for this observation
                if let Some(values) = event.values() {
                    if let Some(value) = values.get(i) {
                        obs_combo.push(value.clone());
                    } else {
                        return Err(ValidationError::InvalidParameter(format!(
                            "Event {} missing value at index {}",
                            event.name(),
                            i
                        )));
                    }
                } else {
                    return Err(ValidationError::InvalidParameter(format!(
                        "Event {} has no values",
                        event.name()
                    )));
                }
            }
            *observed_counts.entry(obs_combo).or_insert(0) += 1;
        }

        // Create counts for all combinations
        let counts: Vec<usize> = combinations
            .iter()
            .map(|combo| observed_counts.get(combo).copied().unwrap_or(0))
            .collect();

        Ok(Cells::new(columns, combinations, counts))
    }

    /// Extract cells (factor combinations) from this event term.
    ///
    /// For categorical events, returns all possible combinations of factor levels
    /// with their counts. For continuous events, returns a single row.
    ///
    /// Results are cached for reuse; `drop_empty` is applied on top of the
    /// cached full table.
    pub fn cells(&self, drop_empty: bool) -> Result<Cells, ValidationError> {
        // Compute or reuse cached full cells (with empty levels included)
        let full = self.cells_cache.get_or_try_init(|| self.compute_cells())?;

        if drop_empty {
            Ok(full.drop_empty())
        } else {
            Ok(full.clone())
        }
    }

    /// Extract condition names from this event term.
    ///
    /// Generates descriptive names for all conditions represented by the term.
    /// For categorical events, creates names for each level. For continuous events,
    /// uses column names.
    pub fn conditions(
        &self,
        _drop_empty: bool,
        expand_basis: bool,
    ) -> Result<Vec<String>, ValidationError> {
        // Shortcut for single continuous event with one column
        if self.events().len() == 1 && self.events()[0].event_type() != EventType::Categorical {
            let event = &self.events()[0];

            let base_cond_tags = match event.event_type() {
                EventType::Continuous => vec![continuous_token(event.name())],
                EventType::Matrix => {
                    // Need column names from EventMatrix
                    if let Some(n_cols) = event.n_columns() {
                        (0..n_cols)
                            .map(|i| continuous_token(&format!("{}_col{}", event.name(), i + 1)))
                            .collect()
                    } else {
                        vec![continuous_token(event.name())]
                    }
                }
                EventType::Basis => vec![continuous_token(event.name())],
                _ => vec![continuous_token(event.name())],
            };

            // Handle basis expansion
            let final_cond_tags = if expand_basis && self.nbasis() > 1 {
                let mut expanded = Vec::new();
                for tag in &base_cond_tags {
                    for b in 1..=self.nbasis() {
                        expanded.push(format!("{}_b{}", tag, b));
                    }
                }
                expanded
            } else {
                base_cond_tags
            };

            return Ok(final_cond_tags);
        }

        // Generate tokens for each component
        let mut comp_tokens_list = Vec::new();

        for event in self.events() {
            let tokens = match event.event_type() {
                EventType::Categorical => {
                    // Level tokens for categorical
                    if let Some(levels) = event.levels() {
                        levels
                            .iter()
                            .map(|level| level_token(event.name(), level))
                            .collect()
                    } else {
                        vec![]
                    }
                }
                EventType::Matrix => {
                    // Column names for matrix
                    if let Some(n_cols) = event.n_columns() {
                        (0..n_cols)
                            .map(|i| continuous_token(&format!("{}_col{}", event.name(), i + 1)))
                            .collect()
                    } else {
                        vec![continuous_token(event.name())]
                    }
                }
                EventType::Basis => {
                    // Single token for basis (expansion handled later)
                    vec![continuous_token(event.name())]
                }
                _ => {
                    // Single continuous variable
                    vec![continuous_token(event.name())]
                }
            };

            if !tokens.is_empty() {
                comp_tokens_list.push(tokens);
            }
        }

        // Handle empty case
        if comp_tokens_list.is_empty() {
            return Ok(vec![]);
        }

        // Combine tokens using product
        let base_cond_tags = if comp_tokens_list.len() == 1 {
            // Single component - just use its tokens
            comp_tokens_list[0].clone()
        } else {
            // Multiple components - create all combinations
            comp_tokens_list
                .into_iter()
                .multi_cartesian_product()
                .map(|combo| make_cond_tag(&combo))
                .collect()
        };

        // Handle basis expansion
        let final_cond_tags = if expand_basis && self.nbasis() > 1 {
            let mut expanded = Vec::new();
            for tag in &base_cond_tags {
                for b in 1..=self.nbasis() {
                    expanded.push(format!("{}_b{}", tag, b));
                }
            }
            expanded
        } else {
            base_cond_tags
        };

        Ok(final_cond_tags)
    }

    /// Get the number of basis functions for this term.
    fn nbasis(&self) -> usize {
        // This would need to be implemented based on how basis functions are stored
        // For now, return 1 as default
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventFactor, EventVariable};
    use ndarray::array;

    #[test]
    fn test_cells_categorical() {
        let onsets = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let values = vec!["A", "B", "A", "B", "A", "B"];
        let durations = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let event =
            EventFactor::new("condition", onsets.to_vec(), values, durations.to_vec()).unwrap();
        let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

        let cells = term.cells(false).unwrap();
        assert_eq!(cells.columns, vec!["condition"]);
        assert_eq!(cells.rows.len(), 2); // A and B
        assert_eq!(cells.counts, vec![3, 3]); // 3 of each
    }

    #[test]
    fn test_cells_continuous() {
        let onsets = array![0.0, 1.0, 2.0];
        let values = array![1.0, 2.0, 3.0];
        let durations = array![1.0, 1.0, 1.0];

        let event = EventVariable::new(
            "rating",
            onsets.to_vec(),
            values.to_vec(),
            durations.to_vec(),
        )
        .unwrap();
        let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

        let cells = term.cells(false).unwrap();
        assert_eq!(cells.columns, vec!["rating"]);
        assert_eq!(cells.rows.len(), 1);
        assert_eq!(cells.rows[0], vec![""]);
        assert_eq!(cells.counts, vec![3]);
    }

    #[test]
    fn test_conditions_categorical() {
        let onsets = array![0.0, 1.0, 2.0];
        let values = vec!["face", "house", "face"];
        let durations = array![1.0, 1.0, 1.0];

        let event =
            EventFactor::new("stimulus", onsets.to_vec(), values, durations.to_vec()).unwrap();
        let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

        let conditions = term.conditions(false, false).unwrap();
        assert_eq!(conditions.len(), 2);
        assert!(conditions.contains(&"stimulus.face".to_string()));
        assert!(conditions.contains(&"stimulus.house".to_string()));
    }

    #[test]
    fn test_conditions_continuous() {
        let onsets = array![0.0, 1.0, 2.0];
        let values = array![1.0, 2.0, 3.0];
        let durations = array![1.0, 1.0, 1.0];

        let event = EventVariable::new(
            "rating",
            onsets.to_vec(),
            values.to_vec(),
            durations.to_vec(),
        )
        .unwrap();
        let term = EventTerm::new(vec![Box::new(event)], false).unwrap();

        let conditions = term.conditions(false, false).unwrap();
        assert_eq!(conditions, vec!["rating"]);
    }

    #[test]
    fn test_cells_interaction() {
        let onsets = array![0.0, 1.0, 2.0, 3.0];
        let cond_values = vec!["A", "A", "B", "B"];
        let block_values = vec!["1", "2", "1", "2"];
        let durations = array![1.0, 1.0, 1.0, 1.0];

        let cond = EventFactor::new(
            "condition",
            onsets.to_vec(),
            cond_values,
            durations.to_vec(),
        )
        .unwrap();
        let block =
            EventFactor::new("block", onsets.to_vec(), block_values, durations.to_vec()).unwrap();

        let term = EventTerm::new(vec![Box::new(cond), Box::new(block)], true).unwrap();

        let cells = term.cells(false).unwrap();
        assert_eq!(cells.columns, vec!["condition", "block"]);
        assert_eq!(cells.rows.len(), 4); // All combinations
        assert_eq!(cells.counts, vec![1, 1, 1, 1]); // One of each
    }
}
