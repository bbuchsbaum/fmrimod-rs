//! Event term implementation for interactions and complex events.

use crate::contrast::weights::TermInfo;
use crate::types::{Array, DataFrame, EventSpec, EventType, ValidationError};
use ndarray::{Array1, Array2};
use once_cell::sync::OnceCell;

/// Represents a term combining one or more events.
///
/// This handles interactions between events and creates the
/// appropriate design matrix columns.
#[derive(Debug)]
pub struct EventTerm {
    events: Vec<Box<dyn EventSpec>>,
    name: String,
    interaction: bool,
    /// Cached full cells table (without dropping empty cells).
    pub(crate) cells_cache: OnceCell<crate::events::Cells>,
}

impl Clone for EventTerm {
    fn clone(&self) -> Self {
        Self {
            events: self.events.iter().map(|e| e.clone_box()).collect(),
            name: self.name.clone(),
            interaction: self.interaction,
            cells_cache: OnceCell::new(),
        }
    }
}

impl EventTerm {
    /// Create a new event term.
    ///
    /// # Arguments
    /// * `events` - Events to combine
    /// * `name` - Optional name for the term
    /// * `interaction` - Whether this is an interaction term
    ///
    /// # Example
    /// ```
    /// use fmri_design::events::{EventTerm, EventFactor};
    ///
    /// let condition = EventFactor::new(
    ///     "condition",
    ///     vec![1.0, 2.0],
    ///     vec!["A", "B"],
    ///     vec![1.0, 1.0],
    /// ).unwrap();
    ///
    /// // Single event term
    /// let term = EventTerm::new(vec![Box::new(condition.clone())], false).unwrap();
    ///
    /// // Interaction term
    /// let block = EventFactor::new(
    ///     "block",
    ///     vec![0.0, 1.0],
    ///     vec!["1", "2"],
    ///     vec![1.0, 1.0],
    /// ).unwrap();
    ///
    /// let interaction = EventTerm::new(
    ///     vec![Box::new(condition), Box::new(block)],
    ///     true,
    /// ).unwrap();
    /// ```
    pub fn new(
        events: Vec<Box<dyn EventSpec>>,
        interaction: bool,
    ) -> Result<Self, ValidationError> {
        if events.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "EventTerm requires at least one event".to_string(),
            ));
        }

        // Generate name based on interaction flag
        let name = if interaction && events.len() > 1 {
            events
                .iter()
                .map(|e| e.name())
                .collect::<Vec<_>>()
                .join(":")
        } else {
            events[0].name().to_string()
        };

        Ok(Self {
            events,
            name,
            interaction,
            cells_cache: OnceCell::new(),
        })
    }

    /// Set interaction flag.
    pub fn with_interaction(mut self, interaction: bool) -> Self {
        self.interaction = interaction;
        self
    }

    /// Get the events in this term.
    pub fn events(&self) -> &[Box<dyn EventSpec>] {
        &self.events
    }

    /// Set custom name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }

    /// Number of component events.
    pub fn n_events(&self) -> usize {
        self.events.len()
    }

    /// Names of component events.
    pub fn event_names(&self) -> Vec<&str> {
        self.events.iter().map(|e| e.name()).collect()
    }

    /// Whether all events are categorical.
    pub fn is_categorical(&self) -> bool {
        self.events
            .iter()
            .all(|e| e.event_type() == EventType::Categorical)
    }

    /// Whether all events are continuous.
    pub fn is_continuous(&self) -> bool {
        self.events
            .iter()
            .all(|e| matches!(e.event_type(), EventType::Continuous | EventType::Basis))
    }

    /// Whether term contains both categorical and continuous events.
    pub fn is_mixed(&self) -> bool {
        let has_categorical = self
            .events
            .iter()
            .any(|e| e.event_type() == EventType::Categorical);
        let has_continuous = self.events.iter().any(|e| {
            matches!(
                e.event_type(),
                EventType::Continuous | EventType::Basis | EventType::Matrix
            )
        });
        has_categorical && has_continuous
    }

    /// Get number of columns in design matrix.
    #[allow(dead_code)] // utility for future design matrix pre-allocation
    fn get_n_columns(&self, sampling_points: &Array1<f64>) -> Result<usize, ValidationError> {
        if self.events.len() == 1 {
            // Single event - get its column count
            let dm = self.events[0].design_matrix(sampling_points)?;
            Ok(dm.ncols())
        } else if self.is_categorical() {
            // Categorical interaction - multiply level counts
            let mut n_cols = 1;
            for event in &self.events {
                let dm = event.design_matrix(sampling_points)?;
                n_cols *= dm.ncols();
            }
            Ok(n_cols)
        } else {
            // For other cases, we need to compute the actual design matrix
            let dm = self.compute_design_matrix(sampling_points)?;
            Ok(dm.ncols())
        }
    }

    /// Get column names for this term.
    pub fn get_column_names(&self) -> Vec<String> {
        if self.events.len() == 1 {
            return self.events[0].column_names();
        }

        use crate::naming::{continuous_token, level_token, make_cond_tag};

        if self.is_categorical() {
            // For categorical terms, use R's naming: varname.level
            // For interactions: var1.level1_var2.level2
            let levels = self.get_levels();
            if self.interaction && self.events.len() > 1 {
                // Interaction: combine tokens from each factor
                levels
                    .iter()
                    .map(|level_combo| {
                        let tokens: Vec<String> = self
                            .events
                            .iter()
                            .zip(level_combo.iter())
                            .map(|(ev, lev)| level_token(ev.name(), lev))
                            .collect();
                        make_cond_tag(&tokens)
                    })
                    .collect()
            } else {
                // Single factor: varname.level for each level
                let event = &self.events[0];
                levels
                    .iter()
                    .map(|level_combo| level_token(event.name(), &level_combo[0]))
                    .collect()
            }
        } else if self.is_mixed() {
            // Mixed interaction: categorical x continuous
            // Produces: var.level_contvar for each level
            use crate::naming::level_token;
            let mut col_names = Vec::new();

            let mut cat_event = None;
            let mut cont_names = Vec::new();

            for event in &self.events {
                if event.event_type() == EventType::Categorical {
                    cat_event = Some(event);
                } else {
                    cont_names.push(continuous_token(event.name()));
                }
            }

            if let Some(cat) = cat_event {
                if let Some(levels) = cat.levels() {
                    let cont_token = cont_names.join("_");
                    for level in levels {
                        let cat_tok = level_token(cat.name(), level);
                        col_names.push(format!("{}_{}", cat_tok, cont_token));
                    }
                }
            }

            col_names
        } else if self.events.len() == 1 {
            // Single continuous event - use variable name
            vec![continuous_token(self.events[0].name())]
        } else {
            // Multiple continuous events
            vec![continuous_token(&self.name)]
        }
    }

    /// Get levels for categorical interactions.
    fn get_levels(&self) -> Vec<Vec<String>> {
        if !self.is_categorical() {
            return vec![];
        }

        // Get levels for each categorical event
        let mut all_levels: Vec<Vec<String>> = Vec::new();
        for event in &self.events {
            if let Some(levels) = event.levels() {
                all_levels.push(levels.iter().map(|s| s.to_string()).collect());
            }
        }

        // Generate all combinations
        if all_levels.is_empty() {
            return vec![];
        }

        if all_levels.len() == 1 {
            // Single categorical event
            all_levels[0].iter().map(|l| vec![l.clone()]).collect()
        } else {
            // Multiple categorical events - generate cartesian product
            let mut result = vec![vec![]];

            for levels in all_levels {
                let mut new_result = Vec::new();
                for combo in &result {
                    for level in &levels {
                        let mut new_combo = combo.clone();
                        new_combo.push(level.clone());
                        new_result.push(new_combo);
                    }
                }
                result = new_result;
            }

            result
        }
    }

    /// Compute design matrix for this term.
    fn compute_design_matrix(
        &self,
        sampling_points: &Array1<f64>,
    ) -> Result<Array2<f64>, ValidationError> {
        if self.events.len() == 1 {
            // Single event - just use its design matrix
            self.events[0].design_matrix(sampling_points)
        } else {
            // Multiple events - compute interaction
            let matrices: Result<Vec<_>, _> = self
                .events
                .iter()
                .map(|e| e.design_matrix(sampling_points))
                .collect();
            let matrices = matrices?;

            // Combine matrices based on event types
            if self.is_categorical() {
                self.categorical_interaction(&matrices)
            } else if self.is_continuous() {
                self.continuous_interaction(&matrices)
            } else {
                self.mixed_interaction(&matrices)
            }
        }
    }

    /// Compute interaction for categorical events.
    fn categorical_interaction(
        &self,
        matrices: &[Array2<f64>],
    ) -> Result<Array2<f64>, ValidationError> {
        let n_points = matrices[0].nrows();

        // Calculate total columns (product of individual column counts)
        let n_cols: usize = matrices.iter().map(|m| m.ncols()).product();
        let mut result = Array2::zeros((n_points, n_cols));

        // Generate all column index combinations
        let col_counts: Vec<usize> = matrices.iter().map(|m| m.ncols()).collect();

        for col_idx in 0..n_cols {
            // Decompose column index into individual matrix column indices
            let mut indices = vec![0; matrices.len()];
            let mut remaining = col_idx;

            // Calculate indices from right to left
            for i in (0..matrices.len()).rev() {
                indices[i] = remaining % col_counts[i];
                remaining /= col_counts[i];
            }

            // Ensure indices are within bounds
            for (idx, &count) in indices.iter().zip(col_counts.iter()) {
                if *idx >= count {
                    return Err(ValidationError::InvalidParameter(format!(
                        "Column index {} out of bounds for matrix with {} columns",
                        idx, count
                    )));
                }
            }

            // Multiply indicators for this combination
            let mut col_values = Array1::ones(n_points);
            for (mat_idx, &col_in_mat) in indices.iter().enumerate() {
                let mat_col = matrices[mat_idx].column(col_in_mat);
                col_values = col_values * &mat_col;
            }

            result.column_mut(col_idx).assign(&col_values);
        }

        Ok(result)
    }

    /// Compute interaction for continuous events.
    fn continuous_interaction(
        &self,
        matrices: &[Array2<f64>],
    ) -> Result<Array2<f64>, ValidationError> {
        let mut result = matrices[0].clone();

        for mat in &matrices[1..] {
            // Handle different numbers of columns
            if result.ncols() == 1 && mat.ncols() > 1 {
                // Broadcast single column
                let col = result.column(0).to_owned();
                result = Array2::zeros((result.nrows(), mat.ncols()));
                for j in 0..mat.ncols() {
                    result.column_mut(j).assign(&(&col * &mat.column(j)));
                }
            } else if result.ncols() > 1 && mat.ncols() == 1 {
                // Multiply each column by single column
                let col = mat.column(0);
                for j in 0..result.ncols() {
                    let mut res_col = result.column_mut(j);
                    res_col *= &col;
                }
            } else if result.ncols() == mat.ncols() {
                // Element-wise multiplication
                result = result * mat;
            } else {
                // Outer product for different column counts
                let n_points = result.nrows();
                let n_cols = result.ncols() * mat.ncols();
                let mut new_result = Array2::zeros((n_points, n_cols));

                let mut col = 0;
                for i in 0..result.ncols() {
                    for j in 0..mat.ncols() {
                        let prod = &result.column(i) * &mat.column(j);
                        new_result.column_mut(col).assign(&prod);
                        col += 1;
                    }
                }

                result = new_result;
            }
        }

        Ok(result)
    }

    /// Compute interaction for mixed event types.
    fn mixed_interaction(&self, matrices: &[Array2<f64>]) -> Result<Array2<f64>, ValidationError> {
        // Separate categorical and continuous matrices
        let mut cat_indices = Vec::new();
        let mut cont_indices = Vec::new();

        for (i, event) in self.events.iter().enumerate() {
            if event.event_type() == EventType::Categorical {
                cat_indices.push(i);
            } else {
                cont_indices.push(i);
            }
        }

        // Process each type
        let cat_part = if !cat_indices.is_empty() {
            let cat_matrices: Vec<_> = cat_indices.iter().map(|&i| matrices[i].clone()).collect();
            Some(self.categorical_interaction(&cat_matrices)?)
        } else {
            None
        };

        let cont_part = if !cont_indices.is_empty() {
            let cont_matrices: Vec<_> = cont_indices.iter().map(|&i| matrices[i].clone()).collect();
            Some(self.continuous_interaction(&cont_matrices)?)
        } else {
            None
        };

        // Combine categorical and continuous parts
        match (cat_part, cont_part) {
            (Some(cat), Some(cont)) => {
                // Multiply each categorical column by continuous
                let n_points = cat.nrows();
                let n_cols = cat.ncols() * cont.ncols();
                let mut result = Array2::zeros((n_points, n_cols));

                let mut col = 0;
                for i in 0..cat.ncols() {
                    for j in 0..cont.ncols() {
                        let prod = &cat.column(i) * &cont.column(j);
                        result.column_mut(col).assign(&prod);
                        col += 1;
                    }
                }

                Ok(result)
            }
            (Some(cat), None) => Ok(cat),
            (None, Some(cont)) => Ok(cont),
            (None, None) => Err(ValidationError::InvalidParameter(
                "No valid matrices for interaction".to_string(),
            )),
        }
    }
}

impl EventSpec for EventTerm {
    fn name(&self) -> &str {
        &self.name
    }

    fn event_type(&self) -> EventType {
        EventType::Term
    }

    fn onsets(&self) -> &crate::types::OnsetArray {
        // For a term, return the first event's onsets
        // In practice, terms are used for design matrix generation
        self.events[0].onsets()
    }

    fn durations(&self) -> &crate::types::DurationArray {
        // For a term, return the first event's durations
        self.events[0].durations()
    }

    fn n_columns(&self) -> Option<usize> {
        Some(self.column_names().len())
    }

    fn event_amplitudes(&self) -> Result<Array, ValidationError> {
        let first = &self.events[0];
        if self
            .events
            .iter()
            .any(|e| e.onsets() != first.onsets() || e.durations() != first.durations())
        {
            return Err(ValidationError::InvalidParameter(
                "interaction events must have identical timing".into(),
            ));
        }
        let matrices: Vec<_> = self
            .events
            .iter()
            .map(|e| e.event_amplitudes())
            .collect::<Result<_, _>>()?;
        if matrices.len() == 1 {
            return Ok(matrices.into_iter().next().unwrap());
        }
        if self.is_categorical() {
            self.categorical_interaction(&matrices)
        } else if self.is_continuous() {
            self.continuous_interaction(&matrices)
        } else {
            self.mixed_interaction(&matrices)
        }
    }

    fn design_matrix(
        &self,
        sampling_points: &crate::types::TimeArray,
    ) -> Result<Array, ValidationError> {
        self.compute_design_matrix(sampling_points)
    }

    fn levels(&self) -> Option<&[String]> {
        // For mixed interactions, we don't have simple levels
        // This could be improved to generate level combinations
        if self.is_categorical() && self.events.len() == 1 {
            self.events[0].levels()
        } else {
            None
        }
    }

    fn column_names(&self) -> Vec<String> {
        self.get_column_names()
    }

    fn to_dataframe(&self) -> DataFrame {
        // For EventTerm, we return the DataFrame of the first event
        // This is a simplification - in practice, EventTerm might need
        // a more sophisticated approach for interactions
        if !self.events.is_empty() {
            self.events[0].to_dataframe()
        } else {
            use crate::dataframe_polars::DataFrameBuilder;

            DataFrameBuilder::new().build()
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn EventSpec> {
        Box::new(self.clone())
    }

    fn filter(&self, mask: &[bool]) -> Result<Box<dyn EventSpec>, ValidationError> {
        if mask.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Empty mask provided".to_string(),
            ));
        }

        // Filter all component events
        let mut filtered_events = Vec::new();

        for event in &self.events {
            let filtered = event.filter(mask)?;
            filtered_events.push(filtered);
        }

        // Create new EventTerm with filtered events
        let mut term = EventTerm::new(filtered_events, self.interaction)?;
        term.name = self.name.clone();

        Ok(Box::new(term))
    }
}

/// Create an interaction term from events.
pub fn create_interaction(events: Vec<Box<dyn EventSpec>>, name: Option<String>) -> EventTerm {
    let interaction_name = name.unwrap_or_else(|| {
        events
            .iter()
            .map(|e| e.name())
            .collect::<Vec<_>>()
            .join(":")
    });
    EventTerm::new(events, true)
        .unwrap()
        .with_name(interaction_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{EventFactor, EventVariable};
    use ndarray::array;

    #[test]
    fn test_single_event_term() {
        let condition =
            EventFactor::new("condition", vec![1.0, 2.0], vec!["A", "B"], vec![1.0, 1.0]).unwrap();

        let term = EventTerm::new(vec![Box::new(condition)], false).unwrap();

        assert_eq!(term.name(), "condition");
        assert_eq!(term.n_events(), 1);
        assert!(!term.interaction);
    }

    #[test]
    fn test_categorical_interaction() {
        let condition =
            EventFactor::new("condition", vec![0.0, 1.0], vec!["A", "B"], vec![0.5, 0.5]).unwrap();

        let block =
            EventFactor::new("block", vec![0.0, 1.0], vec!["1", "2"], vec![1.0, 1.0]).unwrap();

        let sampling_points = array![0.0, 0.5, 1.0, 1.5, 2.0];

        // Test individual matrices first
        let cond_dm = condition.design_matrix(&sampling_points).unwrap();
        let block_dm = block.design_matrix(&sampling_points).unwrap();

        // For debugging - both should be 5x2
        assert_eq!(cond_dm.shape(), &[5, 2]);
        assert_eq!(block_dm.shape(), &[5, 2]);

        let term = create_interaction(vec![Box::new(condition), Box::new(block)], None);

        assert_eq!(term.name(), "condition:block");
        assert!(term.is_categorical());

        let dm = term.design_matrix(&sampling_points).unwrap();

        // Should have 2x2 = 4 columns
        assert_eq!(dm.shape(), &[5, 4]);
    }

    #[test]
    fn test_continuous_interaction() {
        let rating1 = EventVariable::new("rating1", vec![0.0, 1.0], vec![1.0, 2.0], vec![0.5, 0.5])
            .unwrap()
            .with_center(false);

        let rating2 = EventVariable::new("rating2", vec![0.0, 1.0], vec![3.0, 4.0], vec![0.5, 0.5])
            .unwrap()
            .with_center(false);

        let term = create_interaction(vec![Box::new(rating1), Box::new(rating2)], None);

        assert!(term.is_continuous());

        let sampling_points = array![0.0, 0.5, 1.0, 1.5];
        let dm = term.design_matrix(&sampling_points).unwrap();

        // Check that values are multiplied
        assert_eq!(dm[[0, 0]], 3.0); // 1.0 * 3.0
        assert_eq!(dm[[2, 0]], 8.0); // 2.0 * 4.0
    }

    #[test]
    fn test_mixed_interaction() {
        let condition =
            EventFactor::new("condition", vec![0.0, 1.0], vec!["A", "B"], vec![0.5, 0.5]).unwrap();

        let rating = EventVariable::new("rating", vec![0.0, 1.0], vec![1.0, 2.0], vec![0.5, 0.5])
            .unwrap()
            .with_center(false);

        let term = create_interaction(vec![Box::new(condition), Box::new(rating)], None);

        assert!(term.is_mixed());

        let sampling_points = array![0.0, 0.5, 1.0, 1.5];
        let dm = term.design_matrix(&sampling_points).unwrap();

        // Should have 2 columns (2 levels * 1 continuous)
        assert_eq!(dm.shape(), &[4, 2]);
    }
}

impl TermInfo for EventTerm {
    fn conditions(&self, _expand_basis: bool) -> Vec<String> {
        // Delegate to the richer EventTerm::conditions implementation in
        // the cells module, which mirrors the R naming logic. We ignore the
        // `expand_basis` flag for now and always use the default behavior.
        //
        // Errors here indicate inconsistent event state; for contrast
        // computation we fall back to an empty condition list.
        self.conditions(false, false).unwrap_or_else(|_| Vec::new())
    }

    fn nbasis(&self) -> usize {
        // For a term, nbasis is the product of nbasis for each component
        1 // Simplified for now
    }

    fn term_name(&self) -> &str {
        &self.name
    }
}

impl EventTerm {
    /// Generate cartesian product of level sets as strings.
    #[allow(dead_code)] // utility for future interaction term expansion
    fn cartesian_product_levels(&self, level_sets: &[Vec<String>]) -> Vec<String> {
        if level_sets.is_empty() {
            return vec![];
        }

        if level_sets.len() == 1 {
            return level_sets[0].clone();
        }

        // Start with the first set
        let mut result = level_sets[0]
            .iter()
            .map(|s| vec![s.clone()])
            .collect::<Vec<_>>();

        // For each subsequent set, extend all existing combinations
        for set in &level_sets[1..] {
            let mut new_result = Vec::new();
            for combo in result {
                for level in set {
                    let mut new_combo = combo.clone();
                    new_combo.push(level.clone());
                    new_result.push(new_combo);
                }
            }
            result = new_result;
        }

        // Join with ":" to create condition names
        result.into_iter().map(|combo| combo.join(":")).collect()
    }
}
