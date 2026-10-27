//! Column metadata for design matrices.
//!
//! This module provides the [`ColumnMap`] struct, which contains structured
//! per-column metadata for a design matrix. It mirrors R's `design_colmap()`
//! generic, giving a queryable view of every column's role, term, basis
//! information, and modulation type.
//!
//! # Overview
//!
//! Each row of a `ColumnMap` describes one column of a design matrix:
//! - **Identity**: `col` (0-based index), `name` (column name)
//! - **Term membership**: `term_tag`, `term_index`
//! - **Condition**: the event condition label (without basis suffix)
//! - **Run/block**: `run` (if block-specific)
//! - **Role**: `"task"`, `"drift"`, `"intercept"`, `"nuisance"`
//! - **Source**: `"event"` or `"baseline"`
//! - **Basis info**: `basis_name`, `basis_ix`, `basis_total`, `basis_label`
//! - **Display**: `pretty_name`, `is_block_diagonal`
//! - **Modulation**: `modulation_type`, `modulation_id`

use regex::Regex;
use serde::{Deserialize, Serialize};

/// Role of a column within the full design matrix.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColumnRole {
    /// Task-related regressor (from event model).
    Task,
    /// Drift / detrending regressor.
    Drift,
    /// Block or global intercept.
    Intercept,
    /// Nuisance / confound regressor.
    Nuisance,
    /// Generic baseline component.
    Baseline,
}

impl ColumnRole {
    /// Return the canonical string tag used in serialised output.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Task => "task",
            Self::Drift => "drift",
            Self::Intercept => "intercept",
            Self::Nuisance => "nuisance",
            Self::Baseline => "baseline",
        }
    }
}

impl std::fmt::Display for ColumnRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Source model that produced a column.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelSource {
    /// Produced by the event model.
    Event,
    /// Produced by the baseline model.
    Baseline,
}

impl ModelSource {
    /// Return the canonical string tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Event => "event",
            Self::Baseline => "baseline",
        }
    }
}

impl std::fmt::Display for ModelSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Type of modulation for a column.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulationType {
    /// Amplitude modulation (standard categorical regressor).
    Amplitude,
    /// Parametric modulation (e.g., polynomial, spline expansion of a continuous variable).
    Parametric,
    /// External covariate (e.g., pre-convolved regressor).
    Covariate,
}

impl ModulationType {
    /// Return the canonical string tag.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Amplitude => "amplitude",
            Self::Parametric => "parametric",
            Self::Covariate => "covariate",
        }
    }
}

impl std::fmt::Display for ModulationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Metadata for a single design-matrix column.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnEntry {
    /// 0-based column index within the design matrix.
    pub col: usize,
    /// Column name (as it appears in the design matrix).
    pub name: String,
    /// Term identifier tag (unique among terms).
    pub term_tag: Option<String>,
    /// 0-based position of the term in the model's term list.
    pub term_index: Option<usize>,
    /// Condition / event label without basis suffix.
    pub condition: Option<String>,
    /// Block / run id when the column is run-specific; `None` if pooled.
    pub run: Option<usize>,
    /// Role of this column.
    pub role: ColumnRole,
    /// Which model produced this column.
    pub model_source: ModelSource,
    /// Name of the HRF or baseline basis (e.g., `"SPMG1"`, `"polynomial"`).
    pub basis_name: Option<String>,
    /// Within-condition basis index (1-based).
    pub basis_ix: Option<usize>,
    /// Total number of basis components for this column's term.
    pub basis_total: Option<usize>,
    /// Human-readable label for the basis component (e.g., `"canonical"`, `"derivative"`).
    pub basis_label: Option<String>,
    /// Concise, human-friendly display name.
    pub pretty_name: String,
    /// `true` when the regressor is per-run/block.
    pub is_block_diagonal: bool,
    /// Modulation type.
    pub modulation_type: Option<ModulationType>,
    /// Modulator identifier (e.g., `"RT"`).
    pub modulation_id: Option<String>,
}

/// A collection of per-column metadata for a design matrix.
///
/// This is the Rust equivalent of R's `design_colmap()` output -- a structured
/// table with one entry per column. Use the builder or the `from_*` constructors
/// to create one.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMap {
    entries: Vec<ColumnEntry>,
}

impl ColumnMap {
    // -- Constructors --------------------------------------------------------

    /// Create an empty `ColumnMap`.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Create a `ColumnMap` from a vector of entries.
    pub fn from_entries(entries: Vec<ColumnEntry>) -> Self {
        Self { entries }
    }

    /// Build a `ColumnMap` for event-model columns.
    ///
    /// # Arguments
    /// * `column_names` - The column names from the design matrix.
    /// * `term_tags` - Per-term tag strings (one per term, order matches terms list).
    /// * `col_indices` - Per-term slice of column indices (0-based) that belong to each term.
    /// * `basis_names` - Per-term basis name (e.g., `"SPMG1"`).
    /// * `basis_totals` - Per-term total number of basis functions.
    /// * `modulation_types` - Per-term modulation classification.
    /// * `modulation_ids` - Per-term modulator identifier.
    pub fn from_event_model(
        column_names: &[String],
        term_tags: &[String],
        col_indices: &[Vec<usize>],
        basis_names: &[Option<String>],
        basis_totals: &[Option<usize>],
        modulation_types: &[Option<ModulationType>],
        modulation_ids: &[Option<String>],
    ) -> Self {
        let n_cols = column_names.len();
        let mut entries = Vec::with_capacity(n_cols);

        // Build reverse mapping: column -> term index
        let mut col_to_term: Vec<Option<usize>> = vec![None; n_cols];
        for (ti, indices) in col_indices.iter().enumerate() {
            for &ci in indices {
                if ci < n_cols {
                    col_to_term[ci] = Some(ti);
                }
            }
        }

        // Regex for parsing basis suffix like `_b01`
        let basis_re = Regex::new(r"_b(\d+)$").expect("invalid regex");

        for ci in 0..n_cols {
            let name = &column_names[ci];
            let ti = col_to_term[ci];

            let term_tag = ti.and_then(|i| term_tags.get(i).cloned());
            let b_name = ti.and_then(|i| basis_names.get(i).cloned().flatten());
            let b_total = ti.and_then(|i| basis_totals.get(i).cloned().flatten());
            let mod_type = ti.and_then(|i| modulation_types.get(i).cloned().flatten());
            let mod_id = ti.and_then(|i| modulation_ids.get(i).cloned().flatten());

            // Parse basis_ix from column name
            let basis_ix = basis_re
                .captures(name)
                .and_then(|caps| caps.get(1))
                .and_then(|m| m.as_str().parse::<usize>().ok());

            // Derive condition: strip term_tag prefix and basis suffix
            let base_no_basis = basis_re.replace(name, "").to_string();
            let condition = if let Some(ref tag) = term_tag {
                let prefix = format!("{}_", tag);
                if base_no_basis.starts_with(&prefix) {
                    Some(base_no_basis[prefix.len()..].to_string())
                } else {
                    Some(base_no_basis.clone())
                }
            } else {
                Some(base_no_basis.clone())
            };

            // Derive basis_label from basis_name + basis_ix
            let basis_label = basis_ix.map(|bix| derive_basis_label(b_name.as_deref(), bix));

            // Derive pretty_name
            let pretty_name = derive_pretty_name(
                name,
                mod_type.as_ref(),
                mod_id.as_deref(),
                basis_ix,
                basis_label.as_deref(),
            );

            entries.push(ColumnEntry {
                col: ci,
                name: name.clone(),
                term_tag,
                term_index: ti,
                condition,
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: b_name,
                basis_ix,
                basis_total: b_total,
                basis_label,
                pretty_name,
                is_block_diagonal: false,
                modulation_type: mod_type,
                modulation_id: mod_id,
            });
        }

        Self { entries }
    }

    /// Build a `ColumnMap` for baseline-model columns.
    ///
    /// This parses column names to infer role, run number, basis component index,
    /// and other metadata from the naming conventions used by the baseline system.
    ///
    /// # Arguments
    /// * `column_names` - Column names from the baseline design matrix.
    /// * `term_names` - Per-term name (e.g., `"drift"`, `"block"`, `"nuisance"`).
    /// * `col_ranges` - Per-term `(start, count)` of 0-based column indices.
    /// * `drift_basis_name` - Optional name of the drift basis (e.g., `"polynomial"`).
    pub fn from_baseline_model(
        column_names: &[String],
        term_names: &[String],
        col_ranges: &[(usize, usize)],
        drift_basis_name: Option<&str>,
    ) -> Self {
        let n_cols = column_names.len();
        let mut entries = Vec::with_capacity(n_cols);

        // Precompile regexes for baseline column patterns
        let block_re = Regex::new(r"_block_(\d+)$").expect("invalid regex");
        let run_suffix_re = Regex::new(r"_(\d+)$").expect("invalid regex");
        let global_re = Regex::new(r"_global$").expect("invalid regex");
        let nuisance_re = Regex::new(r"#(\d+)_(\d+)$").expect("invalid regex");
        let drift_k_re = Regex::new(r"(\d+)_block_").expect("invalid regex");

        for (ti, &(start, count)) in col_ranges.iter().enumerate() {
            let tname = term_names.get(ti).map(|s| s.as_str()).unwrap_or("unknown");
            for offset in 0..count {
                let ci = start + offset;
                if ci >= n_cols {
                    break;
                }
                let name = &column_names[ci];

                let (role, b_name, basis_ix, basis_label, run, is_block_diag) =
                    if tname == "nuisance" || name.contains('#') {
                        // Nuisance column: parse #<block>_<col>
                        // Must be checked before intercept since nuisance names also match _(\d+)$
                        let (r, k) = nuisance_re
                            .captures(name)
                            .map(|c| {
                                let r = c.get(1).and_then(|m| m.as_str().parse::<usize>().ok());
                                let k = c.get(2).and_then(|m| m.as_str().parse::<usize>().ok());
                                (r, k)
                            })
                            .unwrap_or((None, None));
                        let label = k.map(|ki| format!("component_{:02}", ki));
                        (
                            ColumnRole::Nuisance,
                            Some("nuisance".to_string()),
                            k,
                            label,
                            r,
                            true,
                        )
                    } else if tname == "drift" || name.starts_with("base_") {
                        // Drift column: parse component index k and run from pattern base_<basis><k>_block_<r>
                        let k = drift_k_re
                            .captures(name)
                            .and_then(|c| c.get(1))
                            .and_then(|m| m.as_str().parse::<usize>().ok());
                        let r = block_re
                            .captures(name)
                            .and_then(|c| c.get(1))
                            .and_then(|m| m.as_str().parse::<usize>().ok());
                        let label = k.map(|ki| format!("component_{:02}", ki));
                        let bname = drift_basis_name
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| "baseline".to_string());
                        (ColumnRole::Drift, Some(bname), k, label, r, true)
                    } else if tname == "block"
                        || global_re.is_match(name)
                        || run_suffix_re.is_match(name)
                    {
                        // Intercept column
                        if global_re.is_match(name) {
                            (
                                ColumnRole::Intercept,
                                Some("constant".to_string()),
                                Some(1),
                                Some("intercept".to_string()),
                                None,
                                false,
                            )
                        } else {
                            let r = run_suffix_re
                                .captures(name)
                                .and_then(|c| c.get(1))
                                .and_then(|m| m.as_str().parse::<usize>().ok());
                            (
                                ColumnRole::Intercept,
                                Some("constant".to_string()),
                                None,
                                Some("intercept".to_string()),
                                r,
                                true,
                            )
                        }
                    } else {
                        // Fallback baseline
                        let r = block_re
                            .captures(name)
                            .and_then(|c| c.get(1))
                            .and_then(|m| m.as_str().parse::<usize>().ok());
                        let is_bd = r.is_some();
                        (
                            ColumnRole::Baseline,
                            Some("baseline".to_string()),
                            None,
                            None,
                            r,
                            is_bd,
                        )
                    };

                entries.push(ColumnEntry {
                    col: ci,
                    name: name.clone(),
                    term_tag: Some(tname.to_string()),
                    term_index: Some(ti),
                    condition: None,
                    run,
                    role,
                    model_source: ModelSource::Baseline,
                    basis_name: b_name,
                    basis_ix,
                    basis_total: None, // could be filled in a second pass
                    basis_label,
                    pretty_name: name.clone(),
                    is_block_diagonal: is_block_diag,
                    modulation_type: None,
                    modulation_id: None,
                });
            }
        }

        // Second pass: fill basis_total per term (max basis_ix within the term)
        for ti in 0..term_names.len() {
            let max_bix = entries
                .iter()
                .filter(|e| e.term_index == Some(ti))
                .filter_map(|e| e.basis_ix)
                .max();
            if let Some(mx) = max_bix {
                for e in entries.iter_mut().filter(|e| e.term_index == Some(ti)) {
                    e.basis_total = Some(mx);
                }
            }
        }

        Self { entries }
    }

    // -- Accessors -----------------------------------------------------------

    /// Number of columns described.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the map is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Immutable slice of all column entries.
    pub fn entries(&self) -> &[ColumnEntry] {
        &self.entries
    }

    /// Mutable slice of all column entries.
    pub fn entries_mut(&mut self) -> &mut [ColumnEntry] {
        &mut self.entries
    }

    /// Get a specific entry by column index.
    pub fn get(&self, col: usize) -> Option<&ColumnEntry> {
        self.entries.iter().find(|e| e.col == col)
    }

    /// Get column names as a vector.
    pub fn column_names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    // -- Querying / filtering ------------------------------------------------

    /// Return entries matching a given role.
    pub fn filter_by_role(&self, role: &ColumnRole) -> Vec<&ColumnEntry> {
        self.entries.iter().filter(|e| &e.role == role).collect()
    }

    /// Return entries matching a given model source.
    pub fn filter_by_source(&self, source: &ModelSource) -> Vec<&ColumnEntry> {
        self.entries
            .iter()
            .filter(|e| &e.model_source == source)
            .collect()
    }

    /// Return entries for a specific term tag.
    pub fn filter_by_term_tag(&self, tag: &str) -> Vec<&ColumnEntry> {
        self.entries
            .iter()
            .filter(|e| e.term_tag.as_deref() == Some(tag))
            .collect()
    }

    /// Return entries for a specific condition.
    pub fn filter_by_condition(&self, condition: &str) -> Vec<&ColumnEntry> {
        self.entries
            .iter()
            .filter(|e| e.condition.as_deref() == Some(condition))
            .collect()
    }

    /// Return entries for a specific run.
    pub fn filter_by_run(&self, run: usize) -> Vec<&ColumnEntry> {
        self.entries.iter().filter(|e| e.run == Some(run)).collect()
    }

    /// Return column indices for entries matching a predicate.
    pub fn select_columns<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&ColumnEntry) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(e))
            .map(|e| e.col)
            .collect()
    }

    // -- Merging -------------------------------------------------------------

    /// Append another `ColumnMap`, shifting column indices by the current length.
    ///
    /// This is useful when horizontally concatenating event and baseline design matrices.
    pub fn append(&mut self, other: &ColumnMap) {
        let offset = self.len();
        for entry in &other.entries {
            let mut shifted = entry.clone();
            shifted.col += offset;
            self.entries.push(shifted);
        }
    }

    /// Concatenate event and baseline column maps into a single map.
    pub fn merge_event_baseline(event_map: &ColumnMap, baseline_map: &ColumnMap) -> ColumnMap {
        let mut merged = event_map.clone();
        merged.append(baseline_map);
        merged
    }
}

impl Default for ColumnMap {
    fn default() -> Self {
        Self::new()
    }
}

// -- Helpers -----------------------------------------------------------------

/// Derive a human-readable basis label from basis name and index.
fn derive_basis_label(basis_name: Option<&str>, basis_ix: usize) -> String {
    match basis_name {
        Some(name) if name.contains("SPMG3") => match basis_ix.min(3).max(1) {
            1 => "canonical".to_string(),
            2 => "derivative".to_string(),
            3 => "dispersion".to_string(),
            _ => format!("component_{:02}", basis_ix),
        },
        Some(name) if name.contains("SPMG2") => match basis_ix.min(2).max(1) {
            1 => "canonical".to_string(),
            2 => "derivative".to_string(),
            _ => format!("component_{:02}", basis_ix),
        },
        Some(name) if name.contains("FIR") => {
            format!("lag_{:02}", basis_ix.saturating_sub(1))
        }
        _ => format!("component_{:02}", basis_ix),
    }
}

/// Derive a pretty display name for a column.
fn derive_pretty_name(
    name: &str,
    mod_type: Option<&ModulationType>,
    mod_id: Option<&str>,
    basis_ix: Option<usize>,
    basis_label: Option<&str>,
) -> String {
    match mod_type {
        Some(ModulationType::Parametric) => {
            match (mod_id, basis_ix) {
                // Single-basis parametric: just use the modulator id
                (Some(id), None) => id.to_string(),
                // Multi-basis parametric: include basis label when available
                (Some(id), Some(_bix)) => {
                    if let Some(label) = basis_label {
                        if !label.is_empty() {
                            return format!("{}_{}", id, label);
                        }
                    }
                    // Fallback: use the full column name
                    name.to_string()
                }
                _ => name.to_string(),
            }
        }
        _ => name.to_string(),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_column_role_display() {
        assert_eq!(ColumnRole::Task.as_str(), "task");
        assert_eq!(ColumnRole::Drift.as_str(), "drift");
        assert_eq!(ColumnRole::Intercept.as_str(), "intercept");
        assert_eq!(ColumnRole::Nuisance.as_str(), "nuisance");
        assert_eq!(ColumnRole::Baseline.as_str(), "baseline");
    }

    #[test]
    fn test_model_source_display() {
        assert_eq!(ModelSource::Event.as_str(), "event");
        assert_eq!(ModelSource::Baseline.as_str(), "baseline");
    }

    #[test]
    fn test_modulation_type_display() {
        assert_eq!(ModulationType::Amplitude.as_str(), "amplitude");
        assert_eq!(ModulationType::Parametric.as_str(), "parametric");
        assert_eq!(ModulationType::Covariate.as_str(), "covariate");
    }

    #[test]
    fn test_empty_column_map() {
        let cm = ColumnMap::new();
        assert!(cm.is_empty());
        assert_eq!(cm.len(), 0);
    }

    #[test]
    fn test_from_event_model_basic() {
        // Simulate an event model with one term (cond) having two conditions (A, B)
        // and a single-basis HRF.
        let column_names = vec!["cond_cond.A".to_string(), "cond_cond.B".to_string()];
        let term_tags = vec!["cond".to_string()];
        let col_indices = vec![vec![0, 1]];
        let basis_names = vec![Some("SPMG1".to_string())];
        let basis_totals = vec![Some(1usize)];
        let mod_types = vec![Some(ModulationType::Amplitude)];
        let mod_ids: Vec<Option<String>> = vec![None];

        let cm = ColumnMap::from_event_model(
            &column_names,
            &term_tags,
            &col_indices,
            &basis_names,
            &basis_totals,
            &mod_types,
            &mod_ids,
        );

        assert_eq!(cm.len(), 2);

        let e0 = &cm.entries()[0];
        assert_eq!(e0.col, 0);
        assert_eq!(e0.name, "cond_cond.A");
        assert_eq!(e0.term_tag.as_deref(), Some("cond"));
        assert_eq!(e0.term_index, Some(0));
        assert_eq!(e0.condition.as_deref(), Some("cond.A"));
        assert_eq!(e0.role, ColumnRole::Task);
        assert_eq!(e0.model_source, ModelSource::Event);
        assert!(!e0.is_block_diagonal);

        let e1 = &cm.entries()[1];
        assert_eq!(e1.condition.as_deref(), Some("cond.B"));
    }

    #[test]
    fn test_from_event_model_with_basis() {
        // Two conditions, 2-basis HRF (SPMG2) -> 4 columns
        let column_names = vec![
            "cond_cond.A_b01".to_string(),
            "cond_cond.B_b01".to_string(),
            "cond_cond.A_b02".to_string(),
            "cond_cond.B_b02".to_string(),
        ];
        let term_tags = vec!["cond".to_string()];
        let col_indices = vec![vec![0, 1, 2, 3]];
        let basis_names = vec![Some("SPMG2".to_string())];
        let basis_totals = vec![Some(2usize)];
        let mod_types = vec![Some(ModulationType::Amplitude)];
        let mod_ids: Vec<Option<String>> = vec![None];

        let cm = ColumnMap::from_event_model(
            &column_names,
            &term_tags,
            &col_indices,
            &basis_names,
            &basis_totals,
            &mod_types,
            &mod_ids,
        );

        assert_eq!(cm.len(), 4);

        // First column: basis_ix=1, label=canonical
        assert_eq!(cm.entries()[0].basis_ix, Some(1));
        assert_eq!(cm.entries()[0].basis_label.as_deref(), Some("canonical"));
        assert_eq!(cm.entries()[0].condition.as_deref(), Some("cond.A"));

        // Third column: basis_ix=2, label=derivative
        assert_eq!(cm.entries()[2].basis_ix, Some(2));
        assert_eq!(cm.entries()[2].basis_label.as_deref(), Some("derivative"));
    }

    #[test]
    fn test_from_event_model_parametric() {
        // Parametric modulator: poly_RT with 3 basis columns
        let column_names = vec![
            "poly_RT_01".to_string(),
            "poly_RT_02".to_string(),
            "poly_RT_03".to_string(),
        ];
        let term_tags = vec!["poly_RT".to_string()];
        let col_indices = vec![vec![0, 1, 2]];
        let basis_names: Vec<Option<String>> = vec![None];
        let basis_totals: Vec<Option<usize>> = vec![None];
        let mod_types = vec![Some(ModulationType::Parametric)];
        let mod_ids = vec![Some("RT".to_string())];

        let cm = ColumnMap::from_event_model(
            &column_names,
            &term_tags,
            &col_indices,
            &basis_names,
            &basis_totals,
            &mod_types,
            &mod_ids,
        );

        assert_eq!(cm.len(), 3);
        for e in cm.entries() {
            assert_eq!(e.modulation_type, Some(ModulationType::Parametric));
            assert_eq!(e.modulation_id.as_deref(), Some("RT"));
        }
        // pretty_name for single-basis parametric without basis_ix -> "RT"
        // (these don't have _bNN suffix, so basis_ix is None)
        assert_eq!(cm.entries()[0].pretty_name, "RT");
    }

    #[test]
    fn test_from_baseline_model_drift() {
        let column_names = vec![
            "base_poly1_block_1".to_string(),
            "base_poly2_block_1".to_string(),
            "base_poly1_block_2".to_string(),
            "base_poly2_block_2".to_string(),
        ];
        let term_names = vec!["drift".to_string()];
        let col_ranges = vec![(0, 4)];

        let cm = ColumnMap::from_baseline_model(
            &column_names,
            &term_names,
            &col_ranges,
            Some("polynomial"),
        );

        assert_eq!(cm.len(), 4);
        for e in cm.entries() {
            assert_eq!(e.role, ColumnRole::Drift);
            assert_eq!(e.model_source, ModelSource::Baseline);
            assert!(e.is_block_diagonal);
            assert_eq!(e.basis_name.as_deref(), Some("polynomial"));
        }
        assert_eq!(cm.entries()[0].run, Some(1));
        assert_eq!(cm.entries()[0].basis_ix, Some(1));
        assert_eq!(cm.entries()[2].run, Some(2));
    }

    #[test]
    fn test_from_baseline_model_intercept_global() {
        let column_names = vec!["intercept_global".to_string()];
        let term_names = vec!["block".to_string()];
        let col_ranges = vec![(0, 1)];

        let cm = ColumnMap::from_baseline_model(&column_names, &term_names, &col_ranges, None);

        assert_eq!(cm.len(), 1);
        let e = &cm.entries()[0];
        assert_eq!(e.role, ColumnRole::Intercept);
        assert!(!e.is_block_diagonal);
        assert_eq!(e.run, None);
        assert_eq!(e.basis_label.as_deref(), Some("intercept"));
    }

    #[test]
    fn test_from_baseline_model_intercept_per_run() {
        let column_names = vec!["block_1".to_string(), "block_2".to_string()];
        let term_names = vec!["block".to_string()];
        let col_ranges = vec![(0, 2)];

        let cm = ColumnMap::from_baseline_model(&column_names, &term_names, &col_ranges, None);

        assert_eq!(cm.len(), 2);
        assert_eq!(cm.entries()[0].role, ColumnRole::Intercept);
        assert!(cm.entries()[0].is_block_diagonal);
        assert_eq!(cm.entries()[0].run, Some(1));
        assert_eq!(cm.entries()[1].run, Some(2));
    }

    #[test]
    fn test_from_baseline_model_nuisance() {
        let column_names = vec![
            "motion#1_1".to_string(),
            "motion#1_2".to_string(),
            "motion#2_1".to_string(),
            "motion#2_2".to_string(),
        ];
        let term_names = vec!["nuisance".to_string()];
        let col_ranges = vec![(0, 4)];

        let cm = ColumnMap::from_baseline_model(&column_names, &term_names, &col_ranges, None);

        assert_eq!(cm.len(), 4);
        for e in cm.entries() {
            assert_eq!(e.role, ColumnRole::Nuisance);
            assert!(e.is_block_diagonal);
        }
        assert_eq!(cm.entries()[0].run, Some(1));
        assert_eq!(cm.entries()[0].basis_ix, Some(1));
        assert_eq!(cm.entries()[2].run, Some(2));
    }

    #[test]
    fn test_filter_by_role() {
        let column_names = vec!["task_A".to_string(), "drift_1".to_string()];
        let entries = vec![
            ColumnEntry {
                col: 0,
                name: "task_A".to_string(),
                term_tag: Some("cond".to_string()),
                term_index: Some(0),
                condition: Some("A".to_string()),
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "task_A".to_string(),
                is_block_diagonal: false,
                modulation_type: None,
                modulation_id: None,
            },
            ColumnEntry {
                col: 1,
                name: "drift_1".to_string(),
                term_tag: Some("drift".to_string()),
                term_index: Some(0),
                condition: None,
                run: Some(1),
                role: ColumnRole::Drift,
                model_source: ModelSource::Baseline,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "drift_1".to_string(),
                is_block_diagonal: true,
                modulation_type: None,
                modulation_id: None,
            },
        ];
        let cm = ColumnMap::from_entries(entries);

        let task_cols = cm.filter_by_role(&ColumnRole::Task);
        assert_eq!(task_cols.len(), 1);
        assert_eq!(task_cols[0].name, "task_A");

        let drift_cols = cm.filter_by_role(&ColumnRole::Drift);
        assert_eq!(drift_cols.len(), 1);
        assert_eq!(drift_cols[0].name, "drift_1");
    }

    #[test]
    fn test_select_columns() {
        let entries = vec![
            ColumnEntry {
                col: 0,
                name: "a".to_string(),
                term_tag: None,
                term_index: None,
                condition: Some("A".to_string()),
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "a".to_string(),
                is_block_diagonal: false,
                modulation_type: None,
                modulation_id: None,
            },
            ColumnEntry {
                col: 1,
                name: "b".to_string(),
                term_tag: None,
                term_index: None,
                condition: Some("B".to_string()),
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "b".to_string(),
                is_block_diagonal: false,
                modulation_type: None,
                modulation_id: None,
            },
        ];
        let cm = ColumnMap::from_entries(entries);

        let cols = cm.select_columns(|e| e.condition.as_deref() == Some("A"));
        assert_eq!(cols, vec![0]);
    }

    #[test]
    fn test_merge_event_baseline() {
        let event_entries = vec![ColumnEntry {
            col: 0,
            name: "ev".to_string(),
            term_tag: None,
            term_index: None,
            condition: None,
            run: None,
            role: ColumnRole::Task,
            model_source: ModelSource::Event,
            basis_name: None,
            basis_ix: None,
            basis_total: None,
            basis_label: None,
            pretty_name: "ev".to_string(),
            is_block_diagonal: false,
            modulation_type: None,
            modulation_id: None,
        }];
        let base_entries = vec![ColumnEntry {
            col: 0,
            name: "base".to_string(),
            term_tag: None,
            term_index: None,
            condition: None,
            run: None,
            role: ColumnRole::Drift,
            model_source: ModelSource::Baseline,
            basis_name: None,
            basis_ix: None,
            basis_total: None,
            basis_label: None,
            pretty_name: "base".to_string(),
            is_block_diagonal: true,
            modulation_type: None,
            modulation_id: None,
        }];

        let ev_map = ColumnMap::from_entries(event_entries);
        let base_map = ColumnMap::from_entries(base_entries);
        let merged = ColumnMap::merge_event_baseline(&ev_map, &base_map);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged.entries()[0].col, 0);
        assert_eq!(merged.entries()[0].model_source, ModelSource::Event);
        // Baseline column index shifted by 1
        assert_eq!(merged.entries()[1].col, 1);
        assert_eq!(merged.entries()[1].model_source, ModelSource::Baseline);
    }

    #[test]
    fn test_derive_basis_label_spmg3() {
        assert_eq!(derive_basis_label(Some("SPMG3"), 1), "canonical");
        assert_eq!(derive_basis_label(Some("SPMG3"), 2), "derivative");
        assert_eq!(derive_basis_label(Some("SPMG3"), 3), "dispersion");
    }

    #[test]
    fn test_derive_basis_label_fir() {
        assert_eq!(derive_basis_label(Some("FIR"), 1), "lag_00");
        assert_eq!(derive_basis_label(Some("FIR"), 5), "lag_04");
    }

    #[test]
    fn test_derive_basis_label_generic() {
        assert_eq!(derive_basis_label(Some("BSpline"), 3), "component_03");
        assert_eq!(derive_basis_label(None, 2), "component_02");
    }

    #[test]
    fn test_column_names() {
        let entries = vec![
            ColumnEntry {
                col: 0,
                name: "alpha".to_string(),
                term_tag: None,
                term_index: None,
                condition: None,
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "alpha".to_string(),
                is_block_diagonal: false,
                modulation_type: None,
                modulation_id: None,
            },
            ColumnEntry {
                col: 1,
                name: "beta".to_string(),
                term_tag: None,
                term_index: None,
                condition: None,
                run: None,
                role: ColumnRole::Task,
                model_source: ModelSource::Event,
                basis_name: None,
                basis_ix: None,
                basis_total: None,
                basis_label: None,
                pretty_name: "beta".to_string(),
                is_block_diagonal: false,
                modulation_type: None,
                modulation_id: None,
            },
        ];
        let cm = ColumnMap::from_entries(entries);
        assert_eq!(cm.column_names(), vec!["alpha", "beta"]);
    }
}
