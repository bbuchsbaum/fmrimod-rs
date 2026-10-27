//! Contrast specifications and weight computation for design matrices.
//!
//! This module provides tools for defining statistical contrasts and computing
//! their weights for hypothesis testing in fMRI analyses.

pub mod fcontrast;
pub mod spec;
pub mod weights;

// Re-export main types
pub use fcontrast::*;
pub use spec::*;
pub use weights::{contrast_weights, Contrast as ComputedContrast, TermInfo};

// Legacy compatibility
use crate::types::{ContrastSpec as OldContrastSpec, ContrastType, ValidationError, Vector};
use ndarray::Array1;
use regex::Regex;

/// Contrast specification.
#[derive(Debug, Clone)]
pub struct Contrast {
    name: String,
    contrast_type: ContrastType,
    weights: Option<Vector>,
    formula: Option<String>,
    patterns: Option<(String, Option<String>)>, // (pattern_A, pattern_B)
    factor: Option<String>,
    polynomial_degree: Option<usize>,
    #[allow(dead_code)] // retained for level-aware contrast computation
    levels: Option<Vec<String>>,
    sub_contrasts: Option<Vec<Contrast>>,
}

impl Contrast {
    /// Create a new contrast.
    pub fn new<S: Into<String>>(name: S, contrast_type: ContrastType) -> Self {
        Self {
            name: name.into(),
            contrast_type,
            weights: None,
            formula: None,
            patterns: None,
            factor: None,
            polynomial_degree: None,
            levels: None,
            sub_contrasts: None,
        }
    }

    /// Create a simple contrast (single condition).
    pub fn simple<S: Into<String>>(name: S, condition: S) -> Self {
        let mut contrast = Self::new(name, ContrastType::Simple);
        contrast.formula = Some(condition.into());
        contrast
    }

    /// Create a pairwise contrast (A vs B).
    pub fn pair<S: Into<String>>(name: S, condition_a: S, condition_b: S) -> Self {
        let mut contrast = Self::new(name, ContrastType::Pair);
        contrast.formula = Some(format!("{} - {}", condition_a.into(), condition_b.into()));
        contrast
    }

    /// Create a column contrast using regex patterns.
    pub fn column<S: Into<String>>(name: S, pattern_a: S, pattern_b: Option<S>) -> Self {
        let mut contrast = Self::new(name, ContrastType::Simple);
        contrast.patterns = Some((pattern_a.into(), pattern_b.map(|s| s.into())));
        contrast
    }

    /// Create a polynomial contrast.
    pub fn polynomial<S: Into<String>>(name: S, factor: S, degree: usize) -> Self {
        let mut contrast = Self::new(name, ContrastType::Polynomial);
        contrast.factor = Some(factor.into());
        contrast.polynomial_degree = Some(degree);
        // For now, store degree in formula
        contrast.formula = Some(format!(
            "poly({}, {})",
            contrast.factor.as_ref().unwrap(),
            degree
        ));
        contrast
    }

    /// Create an F-contrast from multiple contrasts.
    pub fn f_contrast<S: Into<String>>(name: S, contrasts: Vec<Contrast>) -> Self {
        let mut contrast = Self::new(name, ContrastType::F);
        let sub_names: Vec<String> = contrasts.iter().map(|c| c.name.clone()).collect();
        contrast.formula = Some(format!("F[{}]", sub_names.join(", ")));
        contrast.sub_contrasts = Some(contrasts);
        contrast
    }

    /// Get the sub-contrasts (for F-contrasts).
    pub fn sub_contrasts(&self) -> Option<&[Contrast]> {
        self.sub_contrasts.as_deref()
    }

    /// Set explicit weights.
    pub fn with_weights(mut self, weights: Vector) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl OldContrastSpec for Contrast {
    fn name(&self) -> &str {
        &self.name
    }

    fn contrast_type(&self) -> ContrastType {
        self.contrast_type
    }

    fn compute_weights(&self, columns: &[String]) -> Result<Vector, ValidationError> {
        if matches!(
            self.contrast_type,
            ContrastType::F | ContrastType::Interaction
        ) {
            return Err(ValidationError::NotImplemented(format!(
                "{:?} contrast weight computation",
                self.contrast_type
            )));
        }

        // If weights are explicitly set, return them
        if let Some(ref weights) = self.weights {
            if weights.len() != columns.len() {
                return Err(ValidationError::LengthMismatch {
                    expected: columns.len(),
                    actual: weights.len(),
                });
            }
            if weights.iter().any(|weight| !weight.is_finite()) {
                return Err(ValidationError::NonFiniteValues);
            }
            return Ok(weights.clone());
        }

        // Otherwise compute based on type
        match self.contrast_type {
            ContrastType::Simple => {
                if let Some(ref formula) = self.formula {
                    compute_simple_weights(formula, columns)
                } else if let Some((ref pattern_a, ref pattern_b)) = self.patterns {
                    compute_column_weights(pattern_a, pattern_b.as_deref(), columns)
                } else {
                    Err(ValidationError::InvalidParameter(
                        "Simple contrast requires formula or pattern".to_string(),
                    ))
                }
            }
            ContrastType::Pair => {
                if let Some(ref formula) = self.formula {
                    compute_pair_weights(formula, columns)
                } else {
                    Err(ValidationError::InvalidParameter(
                        "Pair contrast requires formula".to_string(),
                    ))
                }
            }
            ContrastType::Polynomial => {
                if let Some(ref factor) = self.factor {
                    compute_polynomial_weights(factor, columns, self.polynomial_degree.unwrap_or(1))
                } else {
                    Err(ValidationError::InvalidParameter(
                        "Polynomial contrast requires factor".to_string(),
                    ))
                }
            }
            unsupported => Err(ValidationError::NotImplemented(format!(
                "{:?} contrast weight computation",
                unsupported
            ))),
        }
    }
}

/// Compute weights for a simple contrast.
fn compute_simple_weights(condition: &str, columns: &[String]) -> Result<Vector, ValidationError> {
    let mut weights = Array1::zeros(columns.len());
    let mut found = false;

    for (i, col) in columns.iter().enumerate() {
        if col == condition {
            weights[i] = 1.0;
            found = true;
        }
    }

    if !found {
        return Err(ValidationError::InvalidParameter(format!(
            "Condition '{}' not found in columns",
            condition
        )));
    }

    // Normalize to sum to 1
    let sum = weights.sum();
    if sum > 0.0 {
        weights /= sum;
    }

    Ok(weights)
}

/// Compute weights for a pairwise contrast.
fn compute_pair_weights(formula: &str, columns: &[String]) -> Result<Vector, ValidationError> {
    // Parse "A - B" formula
    let parts: Vec<&str> = formula.split(" - ").collect();
    if parts.len() != 2 {
        return Err(ValidationError::InvalidParameter(
            "Pair contrast formula must be 'A - B'".to_string(),
        ));
    }

    let condition_a = parts[0].trim();
    let condition_b = parts[1].trim();
    if condition_a.is_empty() || condition_b.is_empty() || condition_a == condition_b {
        return Err(ValidationError::InvalidParameter(
            "Pair contrast requires two distinct non-empty column names".to_string(),
        ));
    }

    let mut weights = Array1::zeros(columns.len());
    let mut found_a = false;
    let mut found_b = false;

    for (i, col) in columns.iter().enumerate() {
        if col == condition_a {
            weights[i] = 1.0;
            found_a = true;
        } else if col == condition_b {
            weights[i] = -1.0;
            found_b = true;
        }
    }

    if !found_a || !found_b {
        return Err(ValidationError::InvalidParameter(format!(
            "Conditions '{}' or '{}' not found in columns",
            condition_a, condition_b
        )));
    }

    // Normalize positive and negative weights separately
    let pos_sum = weights.iter().filter(|&&x| x > 0.0).sum::<f64>();
    let neg_sum = weights.iter().filter(|&&x| x < 0.0).sum::<f64>().abs();

    if pos_sum > 0.0 && neg_sum > 0.0 {
        for w in weights.iter_mut() {
            if *w > 0.0 {
                *w /= pos_sum;
            } else if *w < 0.0 {
                *w /= neg_sum;
            }
        }
    }

    Ok(weights)
}

/// Compute weights using column name patterns.
fn compute_column_weights(
    pattern_a: &str,
    pattern_b: Option<&str>,
    columns: &[String],
) -> Result<Vector, ValidationError> {
    let regex_a = Regex::new(pattern_a)
        .map_err(|e| ValidationError::InvalidParameter(format!("Invalid regex: {}", e)))?;

    let regex_b = pattern_b
        .map(|p| Regex::new(p))
        .transpose()
        .map_err(|e| ValidationError::InvalidParameter(format!("Invalid regex: {}", e)))?;

    let mut weights = Array1::zeros(columns.len());
    let mut found_a = false;
    let mut found_b = false;

    for (i, col) in columns.iter().enumerate() {
        let matches_a = regex_a.is_match(col);
        let matches_b = regex_b.as_ref().is_some_and(|regex| regex.is_match(col));
        if matches_a && matches_b {
            return Err(ValidationError::InvalidParameter(format!(
                "Patterns '{}' and '{}' overlap on column '{}'",
                pattern_a,
                pattern_b.unwrap_or_default(),
                col
            )));
        }
        if matches_a {
            weights[i] = 1.0;
            found_a = true;
        } else if matches_b {
            weights[i] = -1.0;
            found_b = true;
        }
    }

    if !found_a {
        return Err(ValidationError::InvalidParameter(format!(
            "Pattern '{}' matched no columns",
            pattern_a
        )));
    }

    if pattern_b.is_some() && !found_b {
        return Err(ValidationError::InvalidParameter(format!(
            "Pattern '{}' matched no columns",
            pattern_b.unwrap()
        )));
    }

    // Normalize weights
    if pattern_b.is_some() {
        // For pairwise, normalize positive and negative separately
        let pos_sum = weights.iter().filter(|&&x| x > 0.0).sum::<f64>();
        let neg_sum = weights.iter().filter(|&&x| x < 0.0).sum::<f64>().abs();

        if pos_sum > 0.0 && neg_sum > 0.0 {
            for w in weights.iter_mut() {
                if *w > 0.0 {
                    *w /= pos_sum;
                } else if *w < 0.0 {
                    *w /= neg_sum;
                }
            }
        }
    } else {
        // For simple contrast, normalize to sum to 1
        let sum = weights.sum();
        if sum > 0.0 {
            weights /= sum;
        }
    }

    Ok(weights)
}

/// Compute polynomial contrast weights.
fn compute_polynomial_weights(
    factor: &str,
    columns: &[String],
    degree: usize,
) -> Result<Vector, ValidationError> {
    // Find columns related to the factor
    let factor_cols: Vec<(usize, &str)> = columns
        .iter()
        .enumerate()
        .filter(|(_, col)| col.contains(factor))
        .map(|(i, col)| (i, col.as_str()))
        .collect();

    if factor_cols.is_empty() {
        return Err(ValidationError::InvalidParameter(format!(
            "Factor '{}' not found in columns",
            factor
        )));
    }

    let n_levels = factor_cols.len();
    if n_levels < 2 {
        return Err(ValidationError::InvalidParameter(
            "Polynomial contrast requires at least 2 levels".to_string(),
        ));
    }

    // Generate polynomial coefficients
    let mut weights = Array1::zeros(columns.len());

    // For linear contrast (degree 1), use simple linear spacing
    if degree == 1 {
        let step = 2.0 / (n_levels as f64 - 1.0);
        for (j, &(i, _)) in factor_cols.iter().enumerate() {
            weights[i] = -1.0 + j as f64 * step;
        }
    } else {
        return Err(ValidationError::NotImplemented(format!(
            "Polynomial contrast degree {} is not implemented",
            degree
        )));
    }

    // Ensure sum to zero
    let mean = weights.sum() / factor_cols.len() as f64;
    for &(i, _) in &factor_cols {
        weights[i] -= mean;
    }

    Ok(weights)
}

/// Helper to create pairwise contrasts for all pairs of conditions.
pub fn pairwise_contrasts(factor: &str, columns: &[String]) -> Vec<Contrast> {
    // Find unique levels for the factor
    let mut levels = Vec::new();
    for col in columns {
        if col.contains(factor) {
            // Extract level name from column
            if let Some(level) = col.split('.').nth(1) {
                if !levels.contains(&level) {
                    levels.push(level);
                }
            }
        }
    }

    // Create contrasts for all pairs
    let mut contrasts = Vec::new();
    for i in 0..levels.len() {
        for j in (i + 1)..levels.len() {
            let name = format!("{}_{}_vs_{}", factor, levels[i], levels[j]);
            let contrast = Contrast::pair(
                name,
                format!("{}.{}", factor, levels[i]),
                format!("{}.{}", factor, levels[j]),
            );
            contrasts.push(contrast);
        }
    }

    contrasts
}

/// Builder for creating contrasts.
#[derive(Debug)]
pub struct ContrastBuilder {
    name: String,
    contrast_type: ContrastType,
    formula: Option<String>,
    patterns: Option<(String, Option<String>)>,
    weights: Option<Vector>,
}

impl ContrastBuilder {
    /// Create a new contrast builder.
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            contrast_type: ContrastType::Simple,
            formula: None,
            patterns: None,
            weights: None,
        }
    }

    /// Set the contrast type.
    pub fn contrast_type(mut self, ctype: ContrastType) -> Self {
        self.contrast_type = ctype;
        self
    }

    /// Set the formula.
    pub fn formula<S: Into<String>>(mut self, formula: S) -> Self {
        self.formula = Some(formula.into());
        self
    }

    /// Set column patterns.
    pub fn patterns<S: Into<String>>(mut self, pattern_a: S, pattern_b: Option<S>) -> Self {
        self.patterns = Some((pattern_a.into(), pattern_b.map(|s| s.into())));
        self
    }

    /// Set explicit weights.
    pub fn weights(mut self, weights: Vector) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Build the contrast.
    pub fn build(self) -> Contrast {
        let mut contrast = Contrast::new(self.name, self.contrast_type);
        contrast.formula = self.formula;
        contrast.patterns = self.patterns;
        contrast.weights = self.weights;
        contrast
    }
}

/// General contrast creation function with flexible specification.
///
/// This function provides a unified interface for creating contrasts using
/// various specification formats.
///
/// # Examples
///
/// ```ignore
/// // Simple contrast
/// let c1 = contrast("main_effect", "condition_A");
///
/// // Pairwise contrast
/// let c2 = contrast("A_vs_B", "condition_A - condition_B");
///
/// // Polynomial contrast
/// let c3 = contrast("linear_trend", poly("factor", 1));
///
/// // Custom weights
/// let c4 = contrast("custom", vec![1.0, -0.5, -0.5]);
///
/// // F-contrast from multiple contrasts
/// let c5 = contrast("omnibus", vec![c1, c2]);
/// ```
pub fn contrast<S, T>(name: S, spec: T) -> Result<ContrastSpecEnum, ValidationError>
where
    S: Into<String>,
    T: IntoContrastSpec,
{
    spec.into_contrast_spec(name.into())
}

/// Trait for types that can be converted into contrast specifications.
pub trait IntoContrastSpec {
    /// Convert into a contrast specification.
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError>;
}

// Implementation for string formula
impl IntoContrastSpec for &str {
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError> {
        // Parse the string to determine contrast type
        let trimmed = self.trim();

        // Check for subtraction (pairwise contrast)
        if trimmed.contains(" - ") {
            let parts: Vec<&str> = trimmed.split(" - ").collect();
            if parts.len() == 2 {
                return Ok(ContrastSpecEnum::Pair(PairContrastSpec {
                    name,
                    a: Formula::new(parts[0].trim()),
                    b: Formula::new(parts[1].trim()),
                    where_clause: None,
                }));
            }
        }

        // Check for polynomial notation
        if trimmed.starts_with("poly(") && trimmed.ends_with(")") {
            let inner = &trimmed[5..trimmed.len() - 1];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 2 {
                if let Ok(degree) = parts[1].trim().parse::<usize>() {
                    return Ok(ContrastSpecEnum::Poly(PolyContrastSpec {
                        name,
                        a: Formula::new(parts[0].trim()),
                        degree,
                        value_map: None,
                        where_clause: None,
                    }));
                }
            }
        }

        // Default to formula contrast for simple expressions
        Ok(ContrastSpecEnum::Formula(ContrastFormulaSpec {
            name,
            formula: Formula::new(trimmed),
            where_clause: None,
        }))
    }
}

impl IntoContrastSpec for String {
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError> {
        self.as_str().into_contrast_spec(name)
    }
}

// Implementation for numeric weights
impl IntoContrastSpec for Vec<f64> {
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError> {
        // Create a formula contrast with explicit weights
        Ok(ContrastSpecEnum::Formula(ContrastFormulaSpec {
            name,
            formula: Formula::with_weights(self),
            where_clause: None,
        }))
    }
}

impl IntoContrastSpec for Array1<f64> {
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError> {
        self.to_vec().into_contrast_spec(name)
    }
}

// Implementation for multiple contrasts (F-contrast)
impl IntoContrastSpec for Vec<ContrastSpecEnum> {
    fn into_contrast_spec(self, name: String) -> Result<ContrastSpecEnum, ValidationError> {
        if self.is_empty() {
            Err(ValidationError::InvalidParameter(
                "Empty contrast list".to_string(),
            ))
        } else {
            Ok(ContrastSpecEnum::FContrast(FContrastSpec {
                name,
                sub_contrasts: self,
                where_clause: None,
            }))
        }
    }
}

/// Helper function for polynomial contrasts
pub fn poly<S: Into<String>>(factor: S, degree: usize) -> String {
    format!("poly({}, {})", factor.into(), degree)
}

/// Helper function for interaction contrasts
pub fn interaction<S: Into<String>, T: Into<String>>(a: S, b: T) -> String {
    format!("{} * {}", a.into(), b.into())
}

/// Create pairwise contrast specifications between levels of a factor
pub fn pairwise_contrast_specs<S: Into<String>>(
    factor: S,
    levels: &[String],
) -> Vec<ContrastSpecEnum> {
    let factor_name = factor.into();
    let mut contrasts = Vec::new();

    for i in 0..levels.len() {
        for j in (i + 1)..levels.len() {
            let name = format!("{}.{}_vs_{}", factor_name, levels[i], levels[j]);
            let spec = ContrastSpecEnum::Pair(PairContrastSpec {
                name,
                a: Formula::new(&levels[i]),
                b: Formula::new(&levels[j]),
                where_clause: None,
            });
            contrasts.push(spec);
        }
    }

    contrasts
}

/// Create sliding-window contrast specifications between adjacent levels.
pub fn sliding_window_contrast_specs(factor: &str, levels: &[String]) -> Vec<ContrastSpecEnum> {
    spec::sliding_window_contrasts(factor, levels, 1)
}
