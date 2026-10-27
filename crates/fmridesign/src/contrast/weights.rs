//! Contrast weight computation.
//!
//! This module implements the computation of numerical weights for contrast specifications.

use ndarray::{s, Array1, Array2, ArrayView1};
use regex::Regex;
use std::collections::HashMap;

use super::spec::*;
use crate::types::{Array, ValidationError, Vector};

/// Evaluate a where-clause expression against a list of condition names,
/// returning a boolean mask indicating which conditions match.
///
/// Supported expression forms:
/// - Regex matching: `"A.*"` matches conditions whose name matches the regex
/// - Equality: `"cond == A"` matches conditions containing "A"
/// - Logical OR: `"A | B"` matches conditions containing "A" or "B"
/// - Negation: `"!A"` matches conditions NOT containing "A"
///
/// Returns an error if no conditions match the clause.
pub fn evaluate_where_clause(
    clause: &str,
    conditions: &[String],
) -> Result<Vec<bool>, ValidationError> {
    let trimmed = clause.trim();
    if trimmed.is_empty() {
        return Ok(vec![true; conditions.len()]);
    }

    let mask = evaluate_where_expr(trimmed, conditions)?;

    // Error if the clause matched nothing
    if !mask.iter().any(|&b| b) {
        return Err(ValidationError::InvalidParameter(format!(
            "Where clause '{}' matched no conditions (available: {:?})",
            clause, conditions
        )));
    }

    Ok(mask)
}

/// Inner evaluation that does NOT check for empty results (used by negation).
fn evaluate_where_expr(expr: &str, conditions: &[String]) -> Result<Vec<bool>, ValidationError> {
    let trimmed = expr.trim();

    // --- Logical OR: "A | B" or "A || B" ---
    // Split on `|` but not inside regex character classes.
    // Simple heuristic: split on ` | ` or ` || ` (space-delimited).
    if trimmed.contains(" | ") || trimmed.contains(" || ") {
        // Normalize double-pipe to single-pipe for splitting
        let normalized = trimmed.replace(" || ", " | ");
        let parts: Vec<&str> = normalized.split(" | ").collect();
        let mut combined = vec![false; conditions.len()];
        for part in parts {
            let sub = evaluate_where_expr(part.trim(), conditions)?;
            for (i, &val) in sub.iter().enumerate() {
                if val {
                    combined[i] = true;
                }
            }
        }
        return Ok(combined);
    }

    // --- Negation: "!X" ---
    if let Some(inner) = trimmed.strip_prefix('!') {
        let inner = inner.trim();
        let sub = evaluate_where_expr(inner, conditions)?;
        return Ok(sub.iter().map(|&b| !b).collect());
    }

    // --- Equality: "var == value" (with or without quotes) ---
    if let Some(caps) = Regex::new(r#"^\s*\w+\s*==\s*['"]?([^'"]+?)['"]?\s*$"#)
        .unwrap()
        .captures(trimmed)
    {
        let target = caps.get(1).unwrap().as_str().trim();
        let mask: Vec<bool> = conditions.iter().map(|c| c.contains(target)).collect();
        return Ok(mask);
    }

    // --- Inequality: "var != value" ---
    if let Some(caps) = Regex::new(r#"^\s*\w+\s*!=\s*['"]?([^'"]+?)['"]?\s*$"#)
        .unwrap()
        .captures(trimmed)
    {
        let target = caps.get(1).unwrap().as_str().trim();
        let mask: Vec<bool> = conditions.iter().map(|c| !c.contains(target)).collect();
        return Ok(mask);
    }

    // --- Bare token: try as regex first, fall back to substring match ---
    if let Ok(re) = Regex::new(&format!("^(?:{})$", trimmed)) {
        let mask: Vec<bool> = conditions.iter().map(|c| re.is_match(c)).collect();
        // If regex matched at least one, use it
        if mask.iter().any(|&b| b) {
            return Ok(mask);
        }
    }

    // Fallback: substring containment
    let mask: Vec<bool> = conditions.iter().map(|c| c.contains(trimmed)).collect();
    Ok(mask)
}

/// Helper: apply a where-clause filter to a list of conditions and return
/// (filtered_indices, filtered_conditions). If `where_clause` is `None`,
/// returns all indices and the original conditions unchanged.
fn apply_where_filter(
    where_clause: Option<&Formula>,
    conditions: &[String],
) -> Result<(Vec<usize>, Vec<String>), ValidationError> {
    match where_clause {
        None => {
            let indices: Vec<usize> = (0..conditions.len()).collect();
            Ok((indices, conditions.to_vec()))
        }
        Some(formula) => {
            let mask = evaluate_where_clause(&formula.expr, conditions)?;
            let indices: Vec<usize> = mask
                .iter()
                .enumerate()
                .filter(|(_, &b)| b)
                .map(|(i, _)| i)
                .collect();
            let filtered: Vec<String> = indices.iter().map(|&i| conditions[i].clone()).collect();
            Ok((indices, filtered))
        }
    }
}

/// Tolerance for sum-to-zero checks.
const CONTRAST_TOLERANCE: f64 = 1e-8;

/// Computed contrast with weights.
#[derive(Debug, Clone)]
pub struct Contrast {
    /// The term this contrast applies to.
    pub term_name: String,
    /// Contrast name.
    pub name: String,
    /// Weight matrix (n_conditions x n_contrasts).
    pub weights: Array,
    /// Condition names corresponding to weight rows.
    pub condnames: Vec<String>,
    /// Original specification.
    pub contrast_spec: ContrastSpecEnum,
}

impl Contrast {
    /// Create a new computed contrast.
    pub fn new(
        term_name: String,
        name: String,
        weights: Array,
        condnames: Vec<String>,
        contrast_spec: ContrastSpecEnum,
    ) -> Self {
        Self {
            term_name,
            name,
            weights,
            condnames,
            contrast_spec,
        }
    }

    /// Check if this is an F-contrast (multiple columns).
    pub fn is_fcontrast(&self) -> bool {
        self.weights.ndim() > 1 && self.weights.ncols() > 1
    }

    /// Get the weight vector (for single contrasts).
    pub fn weight_vector(&self) -> Option<ArrayView1<'_, f64>> {
        if self.weights.ndim() == 1 {
            Some(
                self.weights
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap(),
            )
        } else if self.weights.ncols() == 1 {
            Some(self.weights.column(0))
        } else {
            None
        }
    }
}

/// Trait for objects that can provide conditions and cells.
pub trait TermInfo {
    /// Get condition names from the term.
    ///
    /// When `expand_basis` is true and [`Self::nbasis`] exceeds one, names
    /// must be ordered basis-major: every base condition for basis one, then
    /// every base condition for basis two, and so on. This matches design
    /// matrix columns and [`crate::naming::add_basis_suffix`].
    fn conditions(&self, expand_basis: bool) -> Vec<String>;

    /// Get the number of basis functions (1 if not a basis term).
    fn nbasis(&self) -> usize {
        1
    }

    /// Get cells (categorical combinations) as a table.
    /// Returns (data, column_names) where data is n_cells x n_factors.
    fn cells(&self) -> Option<(Vec<Vec<String>>, Vec<String>)> {
        None
    }

    /// Get the term name.
    fn term_name(&self) -> &str;
}

/// Generic contrast weight computation.
pub fn contrast_weights<T: TermInfo>(
    spec: &ContrastSpecEnum,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let contrast = match spec {
        ContrastSpecEnum::Unit(s) => compute_unit_weights(s, term),
        ContrastSpecEnum::Pair(s) => compute_pair_weights(s, term),
        ContrastSpecEnum::Column(s) => compute_column_weights(s, term),
        ContrastSpecEnum::Poly(s) => compute_poly_weights(s, term),
        ContrastSpecEnum::Oneway(s) => compute_oneway_weights(s, term),
        ContrastSpecEnum::Interaction(s) => compute_interaction_weights(s, term),
        ContrastSpecEnum::Formula(s) => compute_formula_weights(s, term),
        ContrastSpecEnum::FContrast(s) => compute_fcontrast_weights(s, term),
    }?;
    align_basis_expansion(contrast, term)
}

/// Align contrast rows to a term's basis-major design columns.
fn align_basis_expansion<T: TermInfo>(
    mut contrast: Contrast,
    term: &T,
) -> Result<Contrast, ValidationError> {
    if term.nbasis() <= 1 {
        return Ok(contrast);
    }

    let base_names = term.conditions(false);
    let expanded_names = term.conditions(true);
    let expected_expanded = base_names.len().checked_mul(term.nbasis()).ok_or_else(|| {
        ValidationError::InvalidParameter("too many basis-expanded conditions".into())
    })?;
    if expanded_names.len() != expected_expanded {
        return Err(ValidationError::InvalidParameter(format!(
            "basis-expanded condition count must be {} ({} conditions × {} bases), got {}",
            expected_expanded,
            base_names.len(),
            term.nbasis(),
            expanded_names.len()
        )));
    }

    if contrast.condnames == base_names && contrast.weights.nrows() == base_names.len() {
        let mut expanded_weights = Array2::zeros((expected_expanded, contrast.weights.ncols()));
        for basis in 0..term.nbasis() {
            for condition in 0..base_names.len() {
                let target = basis * base_names.len() + condition;
                expanded_weights
                    .row_mut(target)
                    .assign(&contrast.weights.row(condition));
            }
        }
        contrast.weights = expanded_weights;
        contrast.condnames = expanded_names;
        return Ok(contrast);
    }

    if contrast.condnames == expanded_names && contrast.weights.nrows() == expected_expanded {
        return Ok(contrast);
    }

    Err(ValidationError::InvalidParameter(
        "contrast rows and condition names do not align with basis-expanded design columns"
            .to_string(),
    ))
}

/// Calculate contrast weights from logical masks.
fn calculate_mask_weights(
    names: &[String],
    a_mask: &[bool],
    b_mask: Option<&[bool]>,
) -> Result<Vector, ValidationError> {
    let n = names.len();
    if a_mask.len() != n {
        return Err(ValidationError::LengthMismatch {
            expected: n,
            actual: a_mask.len(),
        });
    }

    let n_a = a_mask.iter().filter(|&&x| x).count();
    let mut n_b = 0;

    if let Some(b_mask) = b_mask {
        if b_mask.len() != n {
            return Err(ValidationError::LengthMismatch {
                expected: n,
                actual: b_mask.len(),
            });
        }

        // Check for overlap
        for i in 0..n {
            if a_mask[i] && b_mask[i] {
                return Err(ValidationError::InvalidParameter(
                    "Masks for group A and group B overlap".to_string(),
                ));
            }
        }

        n_b = b_mask.iter().filter(|&&x| x).count();
    }

    // Check for empty selection
    if n_a == 0 || (b_mask.is_some() && n_b == 0) {
        return Err(ValidationError::InvalidParameter(
            "Every requested contrast group must select at least one condition".to_string(),
        ));
    }

    // Initialize weights
    let mut weights = Array1::zeros(n);

    // Assign weights
    if n_a > 0 {
        let w_a = 1.0 / n_a as f64;
        for i in 0..n {
            if a_mask[i] {
                weights[i] = w_a;
            }
        }
    }

    if let Some(b_mask) = b_mask {
        if n_b > 0 {
            let w_b = -1.0 / n_b as f64;
            for i in 0..n {
                if b_mask[i] {
                    weights[i] = w_b;
                }
            }
        }
    }

    // Check sum to zero for pairwise contrasts
    if b_mask.is_some() && n_a > 0 && n_b > 0 {
        let sum = weights.sum();
        if sum.abs() > CONTRAST_TOLERANCE {
            eprintln!("Warning: Contrast weights do not sum to zero (sum={})", sum);
        }
    }

    Ok(weights)
}

/// Compute weights for unit contrast.
fn compute_unit_weights<T: TermInfo>(
    spec: &UnitContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_condnames = term.conditions(false);

    if all_condnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, 1)),
            vec![],
            ContrastSpecEnum::Unit(spec.clone()),
        ));
    }

    let n = all_condnames.len();
    let mut weights = Array1::zeros(n);

    // Apply where clause: only matching conditions get non-zero weight
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &all_condnames)?;
    let n_active = active_indices.len();

    if n_active == 0 {
        return Err(ValidationError::InvalidParameter(
            "Where clause matched no conditions for unit contrast".to_string(),
        ));
    }

    let w = 1.0 / n_active as f64;
    for &i in &active_indices {
        weights[i] = w;
    }

    let weights_2d = weights.insert_axis(ndarray::Axis(1));

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights_2d,
        all_condnames,
        ContrastSpecEnum::Unit(spec.clone()),
    ))
}

/// Compute weights for pairwise contrast.
///
/// Pair selectors identify an exact full condition name or an exact final
/// dot-separated level. Use [`super::spec::ColumnContrastSpec`] for regex
/// selection rather than relying on partial condition-name matches.
fn compute_pair_weights<T: TermInfo>(
    spec: &PairContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let base_condnames = term.conditions(false);

    if base_condnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, 1)),
            vec![],
            ContrastSpecEnum::Pair(spec.clone()),
        ));
    }

    // Get cells if available
    let _cells = term.cells();

    // Apply where-clause to restrict the set of eligible conditions
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &base_condnames)?;
    let active_set: std::collections::HashSet<usize> = active_indices.iter().copied().collect();

    // Create masks based on formulas, only among active conditions
    let n = base_condnames.len();
    let mut mask_a = vec![false; n];
    let mut mask_b = vec![false; n];

    for (i, cond) in base_condnames.iter().enumerate() {
        if !active_set.contains(&i) {
            continue;
        }
        if matches_pair_selector(cond, &spec.a.expr) {
            mask_a[i] = true;
        }
        if matches_pair_selector(cond, &spec.b.expr) {
            mask_b[i] = true;
        }
    }

    // Calculate weights
    let weights_vec = calculate_mask_weights(&base_condnames, &mask_a, Some(&mask_b))?;

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights_vec.insert_axis(ndarray::Axis(1)),
        base_condnames,
        ContrastSpecEnum::Pair(spec.clone()),
    ))
}

/// Match a pair selector against a full condition name or its final level.
fn matches_pair_selector(condition: &str, selector: &str) -> bool {
    condition == selector
        || condition
            .rsplit_once('.')
            .is_some_and(|(_, level)| level == selector)
}

/// Compute weights for column contrast using regex patterns.
fn compute_column_weights<T: TermInfo>(
    spec: &ColumnContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_colnames = term.conditions(true);

    if all_colnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, 1)),
            vec![],
            ContrastSpecEnum::Column(spec.clone()),
        ));
    }

    // Apply where-clause to restrict eligible columns
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &all_colnames)?;
    let active_set: std::collections::HashSet<usize> = active_indices.iter().copied().collect();

    // Compile regex patterns
    let regex_a = Regex::new(&spec.pattern_a)
        .map_err(|e| ValidationError::InvalidParameter(format!("Invalid regex: {}", e)))?;

    let regex_b = spec
        .pattern_b
        .as_ref()
        .map(|p| Regex::new(p))
        .transpose()
        .map_err(|e| ValidationError::InvalidParameter(format!("Invalid regex: {}", e)))?;

    // Find matching columns among active set
    let n = all_colnames.len();
    let mut mask_a = vec![false; n];
    let mut mask_b = vec![false; n];

    for (i, name) in all_colnames.iter().enumerate() {
        if !active_set.contains(&i) {
            continue;
        }
        if regex_a.is_match(name) {
            mask_a[i] = true;
        } else if let Some(ref regex_b) = regex_b {
            if regex_b.is_match(name) {
                mask_b[i] = true;
            }
        }
    }

    // Calculate weights
    let weights_vec = if regex_b.is_some() {
        calculate_mask_weights(&all_colnames, &mask_a, Some(&mask_b))?
    } else {
        calculate_mask_weights(&all_colnames, &mask_a, None)?
    };

    let weights = weights_vec.insert_axis(ndarray::Axis(1));

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights,
        all_colnames,
        ContrastSpecEnum::Column(spec.clone()),
    ))
}

/// Compute polynomial contrast weights.
fn compute_poly_weights<T: TermInfo>(
    spec: &PolyContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_condnames = term.conditions(false);

    if all_condnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, spec.degree)),
            vec![],
            ContrastSpecEnum::Poly(spec.clone()),
        ));
    }

    // Apply where-clause to restrict which conditions participate
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &all_condnames)?;
    let n_unique = active_indices.len();

    if n_unique <= spec.degree {
        return Err(ValidationError::InvalidParameter(format!(
            "Polynomial degree ({}) is too high for number of levels ({})",
            spec.degree, n_unique
        )));
    }

    // Initialize weight matrix (one column per polynomial degree)
    let mut weights = Array2::zeros((all_condnames.len(), spec.degree));

    // Generate values for polynomial evaluation
    let vals: Vec<f64> = if let Some(ref value_map) = spec.value_map {
        if value_map.len() != n_unique {
            return Err(ValidationError::LengthMismatch {
                expected: n_unique,
                actual: value_map.len(),
            });
        }
        value_map.clone()
    } else {
        // Default to equally spaced values
        (0..n_unique).map(|i| i as f64).collect()
    };

    // Create polynomial contrast matrix
    // Build matrix of polynomial basis functions
    let mut x_matrix = Array2::zeros((n_unique, spec.degree + 1));

    // Fill with powers of vals
    for i in 0..n_unique {
        let mut power = 1.0;
        for j in 0..=spec.degree {
            x_matrix[[i, j]] = power;
            power *= vals[i];
        }
    }

    // Remove intercept column
    let x_no_intercept = x_matrix.slice(s![.., 1..]).to_owned();

    // Center each column (subtract mean) to ensure orthogonal to intercept
    let mut x_centered = x_no_intercept.clone();
    for j in 0..spec.degree {
        let col_mean = x_centered.column(j).sum() / n_unique as f64;
        for i in 0..n_unique {
            x_centered[[i, j]] -= col_mean;
        }
    }

    // QR decomposition for orthogonalization using Gram-Schmidt
    let mut q = x_centered.clone();

    for j in 0..spec.degree {
        // Orthogonalize column j against previous columns
        for k in 0..j {
            let dot = q.column(j).dot(&q.column(k));
            let norm_sq = q.column(k).dot(&q.column(k));
            if norm_sq > 0.0 {
                let proj_coef = dot / norm_sq;
                for i in 0..n_unique {
                    q[[i, j]] -= proj_coef * q[[i, k]];
                }
            }
        }

        // Normalize
        let norm = q.column(j).dot(&q.column(j)).sqrt();
        if norm > 0.0 {
            for i in 0..n_unique {
                q[[i, j]] /= norm;
            }
        }
    }

    // Assign polynomial weights only to the active condition rows
    for (qi, &row_idx) in active_indices.iter().enumerate() {
        for j in 0..spec.degree {
            weights[[row_idx, j]] = q[[qi, j]];
        }
    }

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights,
        all_condnames,
        ContrastSpecEnum::Poly(spec.clone()),
    ))
}

/// Compute one-way contrast weights (main effect).
fn compute_oneway_weights<T: TermInfo>(
    spec: &OnewayContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_condnames = term.conditions(false);

    if all_condnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, 0)),
            vec![],
            ContrastSpecEnum::Oneway(spec.clone()),
        ));
    }

    // Apply where-clause to restrict which conditions participate
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &all_condnames)?;
    let k = active_indices.len();

    if k < 2 {
        return Err(ValidationError::InvalidParameter(format!(
            "Factor has only {} level(s) after where-clause, cannot create contrast",
            k
        )));
    }

    // Generate contrast matrix (k-1 orthogonal contrasts)
    // Using successive differences coding, placed at active row indices
    let n_total = all_condnames.len();
    let mut weights = Array2::zeros((n_total, k - 1));

    for ci in 0..(k - 1) {
        // Within the active subset, contrast level ci against the average of later levels
        let row_i = active_indices[ci];
        weights[[row_i, ci]] = (k - ci - 1) as f64 / (k - ci) as f64;
        for cj in (ci + 1)..k {
            let row_j = active_indices[cj];
            weights[[row_j, ci]] = -1.0 / (k - ci) as f64;
        }
    }

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights,
        all_condnames,
        ContrastSpecEnum::Oneway(spec.clone()),
    ))
}

/// Compute interaction contrast weights.
fn compute_interaction_weights<T: TermInfo>(
    spec: &InteractionContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_condnames = term.conditions(false);

    if all_condnames.is_empty() {
        return Ok(Contrast::new(
            term.term_name().to_string(),
            spec.name.clone(),
            Array2::zeros((0, 0)),
            vec![],
            ContrastSpecEnum::Interaction(spec.clone()),
        ));
    }

    // Apply where-clause to restrict which conditions participate
    let (active_indices, _) = apply_where_filter(spec.where_clause.as_ref(), &all_condnames)?;

    if active_indices.len() != 4 {
        return Err(ValidationError::InvalidParameter(format!(
            "The current interaction contrast supports exactly four ordered 2x2 cells, found {} after where-clause",
            active_indices.len()
        )));
    }

    // Simple 2x2 interaction contrast placed at active indices
    let mut weights = Array2::zeros((all_condnames.len(), 1));
    weights[[active_indices[0], 0]] = 1.0;
    weights[[active_indices[1], 0]] = -1.0;
    weights[[active_indices[2], 0]] = -1.0;
    weights[[active_indices[3], 0]] = 1.0;

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights,
        all_condnames,
        ContrastSpecEnum::Interaction(spec.clone()),
    ))
}

/// Compute weights for formula-based contrast.
fn compute_formula_weights<T: TermInfo>(
    spec: &ContrastFormulaSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    let all_condnames = term.conditions(false);
    let (active_indices, active_names) =
        apply_where_filter(spec.where_clause.as_ref(), &all_condnames)?;
    let expression = spec.formula.expr.trim().trim_start_matches('~').trim();
    let mut weights = Array1::zeros(all_condnames.len());
    if let Some(inner) = expression
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
    {
        let values: Vec<f64> = if inner.trim().is_empty() {
            Vec::new()
        } else {
            inner
                .split(',')
                .map(|s| {
                    s.trim().parse::<f64>().map_err(|_| {
                        ValidationError::InvalidParameter("invalid explicit contrast weight".into())
                    })
                })
                .collect::<Result<_, _>>()?
        };
        if values.len() != active_indices.len() {
            return Err(ValidationError::LengthMismatch {
                expected: active_indices.len(),
                actual: values.len(),
            });
        }
        for (&index, value) in active_indices.iter().zip(values) {
            weights[index] = value;
        }
    } else {
        // A deliberately bounded linear grammar: signed names, optional finite
        // numeric coefficients, and backticks for names containing operators.
        let token = Regex::new(r"^\s*([+-]?)\s*(?:(\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|\.\d+(?:[eE][+-]?\d+)?)\s*\*\s*)?(`[^`]+`|[A-Za-z_][A-Za-z0-9_.:]*)").unwrap();
        let mut rest = expression;
        let mut first = true;
        while !rest.trim().is_empty() {
            let caps = token.captures(rest).ok_or_else(|| {
                ValidationError::InvalidParameter(format!(
                    "unsupported linear contrast expression near {rest:?}"
                ))
            })?;
            if !first && caps[1].is_empty() {
                return Err(ValidationError::InvalidParameter(
                    "contrast terms require + or -".into(),
                ));
            }
            let sign = if &caps[1] == "-" { -1.0 } else { 1.0 };
            let coefficient = caps
                .get(2)
                .map_or(Ok(1.0), |v| v.as_str().parse::<f64>())
                .map_err(|_| {
                    ValidationError::InvalidParameter("invalid contrast coefficient".into())
                })?;
            let name = caps[3].trim_matches('`');
            let matches: Vec<_> = active_names
                .iter()
                .enumerate()
                .filter(|(_, v)| v.as_str() == name)
                .map(|(i, _)| i)
                .collect();
            if matches.len() != 1 {
                return Err(ValidationError::InvalidParameter(format!(
                    "contrast name {name:?} must identify exactly one active condition"
                )));
            }
            weights[active_indices[matches[0]]] += sign * coefficient;
            rest = &rest[caps.get(0).unwrap().end()..];
            first = false;
        }
        if first {
            return Err(ValidationError::InvalidParameter(
                "empty contrast formula".into(),
            ));
        }
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::InvalidParameter(
            "contrast weights must be finite".into(),
        ));
    }

    let weights_2d = weights.insert_axis(ndarray::Axis(1));

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        weights_2d,
        all_condnames,
        ContrastSpecEnum::Formula(spec.clone()),
    ))
}

/// Compute weights for an F-contrast by stacking sub-contrast weight vectors as columns.
fn compute_fcontrast_weights<T: TermInfo>(
    spec: &FContrastSpec,
    term: &T,
) -> Result<Contrast, ValidationError> {
    if spec.sub_contrasts.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "F-contrast requires at least one sub-contrast".to_string(),
        ));
    }

    let mut weight_columns: Vec<Array1<f64>> = Vec::new();
    let mut condnames: Option<Vec<String>> = None;

    for sub_spec in &spec.sub_contrasts {
        let sub_contrast = contrast_weights(sub_spec, term)?;

        if let Some(ref existing) = condnames {
            if &sub_contrast.condnames != existing {
                return Err(ValidationError::InvalidParameter(
                    "All sub-contrasts in an F-contrast must have the same condition names"
                        .to_string(),
                ));
            }
        } else {
            condnames = Some(sub_contrast.condnames.clone());
        }

        for j in 0..sub_contrast.weights.ncols() {
            weight_columns.push(sub_contrast.weights.column(j).to_owned());
        }
    }

    let condnames = condnames.unwrap();
    let n_conds = condnames.len();
    let n_cols = weight_columns.len();

    let mut f_matrix = Array2::zeros((n_conds, n_cols));
    for (j, col) in weight_columns.iter().enumerate() {
        f_matrix.column_mut(j).assign(col);
    }

    Ok(Contrast::new(
        term.term_name().to_string(),
        spec.name.clone(),
        f_matrix,
        condnames,
        ContrastSpecEnum::FContrast(spec.clone()),
    ))
}

/// Compute weights for a set of contrasts.
pub fn contrast_weights_set<T: TermInfo>(
    specs: &ContrastSet,
    term: &T,
) -> HashMap<String, Result<Contrast, ValidationError>> {
    let mut results = HashMap::new();

    for spec in specs.iter() {
        let result = contrast_weights(spec, term);
        let name = spec.name().to_string();
        results.insert(name, result);
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple TermInfo implementation for testing.
    struct MockTerm {
        name: String,
        conditions: Vec<String>,
    }

    impl MockTerm {
        fn new(name: &str, conditions: Vec<&str>) -> Self {
            Self {
                name: name.to_string(),
                conditions: conditions.into_iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl TermInfo for MockTerm {
        fn conditions(&self, _expand_basis: bool) -> Vec<String> {
            self.conditions.clone()
        }
        fn term_name(&self) -> &str {
            &self.name
        }
    }

    // ---- evaluate_where_clause tests ----

    #[test]
    fn test_where_clause_equality() {
        let conds = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mask = evaluate_where_clause("cond == A", &conds).unwrap();
        assert_eq!(mask, vec![true, false, false]);
    }

    #[test]
    fn test_where_clause_equality_quoted() {
        let conds = vec!["face".to_string(), "house".to_string(), "car".to_string()];
        let mask = evaluate_where_clause("cond == 'house'", &conds).unwrap();
        assert_eq!(mask, vec![false, true, false]);
    }

    #[test]
    fn test_where_clause_inequality() {
        let conds = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mask = evaluate_where_clause("cond != B", &conds).unwrap();
        assert_eq!(mask, vec![true, false, true]);
    }

    #[test]
    fn test_where_clause_regex() {
        let conds = vec![
            "face_happy".to_string(),
            "face_sad".to_string(),
            "house".to_string(),
            "car".to_string(),
        ];
        let mask = evaluate_where_clause("face_.*", &conds).unwrap();
        assert_eq!(mask, vec![true, true, false, false]);
    }

    #[test]
    fn test_where_clause_or() {
        let conds = vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ];
        let mask = evaluate_where_clause("A | C", &conds).unwrap();
        assert_eq!(mask, vec![true, false, true, false]);
    }

    #[test]
    fn test_where_clause_negation() {
        let conds = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let mask = evaluate_where_clause("!B", &conds).unwrap();
        assert_eq!(mask, vec![true, false, true]);
    }

    #[test]
    fn test_where_clause_empty_result_errors() {
        let conds = vec!["A".to_string(), "B".to_string()];
        let result = evaluate_where_clause("cond == Z", &conds);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_clause_empty_string_selects_all() {
        let conds = vec!["A".to_string(), "B".to_string()];
        let mask = evaluate_where_clause("", &conds).unwrap();
        assert_eq!(mask, vec![true, true]);
    }

    // ---- unit contrast with where clause ----

    #[test]
    fn test_unit_contrast_with_where_clause() {
        let term = MockTerm::new("cond", vec!["A", "B", "C", "D"]);
        let spec = UnitContrastSpec {
            name: "selected".to_string(),
            where_clause: Some(Formula::new("A | B")),
        };
        let result = compute_unit_weights(&spec, &term).unwrap();
        let w = result.weights.column(0);
        // Only A and B should have weight 0.5
        assert!((w[0] - 0.5).abs() < 1e-10);
        assert!((w[1] - 0.5).abs() < 1e-10);
        assert!((w[2]).abs() < 1e-10);
        assert!((w[3]).abs() < 1e-10);
    }

    #[test]
    fn test_unit_contrast_no_where_clause() {
        let term = MockTerm::new("cond", vec!["A", "B", "C"]);
        let spec = UnitContrastSpec {
            name: "all".to_string(),
            where_clause: None,
        };
        let result = compute_unit_weights(&spec, &term).unwrap();
        let w = result.weights.column(0);
        let expected = 1.0 / 3.0;
        for i in 0..3 {
            assert!((w[i] - expected).abs() < 1e-10);
        }
    }

    // ---- pair contrast with where clause ----

    #[test]
    fn test_pair_contrast_with_where_clause() {
        // Conditions: run1.A, run1.B, run2.A, run2.B
        // Where clause filters to run1 only, then pair A vs B within that subset
        let term = MockTerm::new("cond", vec!["run1.A", "run1.B", "run2.A", "run2.B"]);
        let spec = PairContrastSpec {
            name: "A_vs_run1.B".to_string(),
            a: Formula::new("A"),
            b: Formula::new("B"),
            where_clause: Some(Formula::new("run1.*")),
        };
        let result = compute_pair_weights(&spec, &term).unwrap();
        let w = result.weights.column(0);
        // run1.A should be +1, run1.B should be -1, run2 conditions should be 0
        assert!((w[0] - 1.0).abs() < 1e-10);
        assert!((w[1] - (-1.0)).abs() < 1e-10);
        assert!((w[2]).abs() < 1e-10);
        assert!((w[3]).abs() < 1e-10);
    }

    // ---- oneway contrast with where clause ----

    #[test]
    fn test_oneway_contrast_with_where_clause() {
        let term = MockTerm::new("cond", vec!["A", "B", "C", "D"]);
        let spec = OnewayContrastSpec {
            name: "subset_main".to_string(),
            a: Formula::new("cond"),
            where_clause: Some(Formula::new("A | B | C")),
        };
        let result = compute_oneway_weights(&spec, &term).unwrap();
        // Should produce a matrix with 4 rows, 2 columns (3-1)
        assert_eq!(result.weights.nrows(), 4);
        assert_eq!(result.weights.ncols(), 2);
        // D (index 3) should have all-zero weights
        assert!((result.weights[[3, 0]]).abs() < 1e-10);
        assert!((result.weights[[3, 1]]).abs() < 1e-10);
    }

    // ---- where clause with regex negation ----

    #[test]
    fn test_where_clause_negation_regex() {
        let conds = vec![
            "face_happy".to_string(),
            "face_sad".to_string(),
            "house".to_string(),
        ];
        let mask = evaluate_where_clause("!house", &conds).unwrap();
        assert_eq!(mask, vec![true, true, false]);
    }

    // ---- where clause double-pipe OR ----

    #[test]
    fn test_where_clause_double_pipe_or() {
        let conds = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mask = evaluate_where_clause("X || Z", &conds).unwrap();
        assert_eq!(mask, vec![true, false, true]);
    }

    // ---- substring matching fallback ----

    #[test]
    fn test_where_clause_substring_match() {
        let conds = vec![
            "cond_A".to_string(),
            "cond_B".to_string(),
            "other".to_string(),
        ];
        let mask = evaluate_where_clause("cond_", &conds).unwrap();
        assert_eq!(mask, vec![true, true, false]);
    }
}
