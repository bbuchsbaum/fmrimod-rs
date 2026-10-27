//! Contrast specification types and builders.
//!
//! This module provides types for specifying different kinds of statistical contrasts.

// ValidationError used in downstream modules

/// A contrast expression or explicit weight vector.
/// Linear expressions accept exact condition names, +, -, and numeric coefficients.
#[derive(Debug, Clone)]
pub struct Formula {
    pub expr: String,
}

impl Formula {
    /// Create a new formula.
    pub fn new<S: Into<String>>(expr: S) -> Self {
        Self { expr: expr.into() }
    }

    /// Create a formula from explicit weights.
    pub fn with_weights(weights: Vec<f64>) -> Self {
        let weight_str = weights
            .iter()
            .map(|w| format!("{}", w))
            .collect::<Vec<_>>()
            .join(", ");
        Self {
            expr: format!("[{}]", weight_str),
        }
    }
}

impl From<&str> for Formula {
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<String> for Formula {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

/// Base trait for contrast specifications.
pub trait ContrastSpec: Send + Sync {
    /// Get the name of the contrast.
    fn name(&self) -> &str;

    /// Get the contrast type name for display.
    fn spec_type(&self) -> &str;
}

/// Unit contrast specification (weights sum to 1).
#[derive(Debug, Clone)]
pub struct UnitContrastSpec {
    pub name: String,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for UnitContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "unit"
    }
}

/// Pairwise contrast specification (A vs B).
#[derive(Debug, Clone)]
pub struct PairContrastSpec {
    pub name: String,
    pub a: Formula,
    pub b: Formula,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for PairContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "pair"
    }
}

/// Column-based contrast using regex patterns.
#[derive(Debug, Clone)]
pub struct ColumnContrastSpec {
    pub name: String,
    pub pattern_a: String,
    pub pattern_b: Option<String>,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for ColumnContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "column"
    }
}

/// Polynomial contrast specification.
#[derive(Debug, Clone)]
pub struct PolyContrastSpec {
    pub name: String,
    pub a: Formula,
    pub degree: usize,
    pub value_map: Option<Vec<f64>>,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for PolyContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "polynomial"
    }
}

/// One-way ANOVA contrast (main effect).
#[derive(Debug, Clone)]
pub struct OnewayContrastSpec {
    pub name: String,
    pub a: Formula,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for OnewayContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "oneway"
    }
}

/// Interaction contrast specification.
#[derive(Debug, Clone)]
pub struct InteractionContrastSpec {
    pub name: String,
    pub a: Formula,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for InteractionContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "interaction"
    }
}

/// F-contrast specification (multiple sub-contrasts combined for joint testing).
#[derive(Debug, Clone)]
pub struct FContrastSpec {
    pub name: String,
    pub sub_contrasts: Vec<ContrastSpecEnum>,
    pub where_clause: Option<String>,
}

impl ContrastSpec for FContrastSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "f_contrast"
    }
}

/// Formula-based contrast specification.
#[derive(Debug, Clone)]
pub struct ContrastFormulaSpec {
    pub name: String,
    pub formula: Formula,
    pub where_clause: Option<Formula>,
}

impl ContrastSpec for ContrastFormulaSpec {
    fn name(&self) -> &str {
        &self.name
    }

    fn spec_type(&self) -> &str {
        "formula"
    }
}

/// Enum wrapper for contrast specifications to enable pattern matching.
#[derive(Debug, Clone)]
pub enum ContrastSpecEnum {
    Unit(UnitContrastSpec),
    Pair(PairContrastSpec),
    Column(ColumnContrastSpec),
    Poly(PolyContrastSpec),
    Oneway(OnewayContrastSpec),
    Interaction(InteractionContrastSpec),
    Formula(ContrastFormulaSpec),
    FContrast(FContrastSpec),
}

impl ContrastSpec for ContrastSpecEnum {
    fn name(&self) -> &str {
        match self {
            Self::Unit(s) => s.name(),
            Self::Pair(s) => s.name(),
            Self::Column(s) => s.name(),
            Self::Poly(s) => s.name(),
            Self::Oneway(s) => s.name(),
            Self::Interaction(s) => s.name(),
            Self::Formula(s) => s.name(),
            Self::FContrast(s) => s.name(),
        }
    }

    fn spec_type(&self) -> &str {
        match self {
            Self::Unit(s) => s.spec_type(),
            Self::Pair(s) => s.spec_type(),
            Self::Column(s) => s.spec_type(),
            Self::Poly(s) => s.spec_type(),
            Self::Oneway(s) => s.spec_type(),
            Self::Interaction(s) => s.spec_type(),
            Self::Formula(s) => s.spec_type(),
            Self::FContrast(s) => s.spec_type(),
        }
    }
}

impl From<UnitContrastSpec> for ContrastSpecEnum {
    fn from(spec: UnitContrastSpec) -> Self {
        Self::Unit(spec)
    }
}

impl From<PairContrastSpec> for ContrastSpecEnum {
    fn from(spec: PairContrastSpec) -> Self {
        Self::Pair(spec)
    }
}

impl From<ColumnContrastSpec> for ContrastSpecEnum {
    fn from(spec: ColumnContrastSpec) -> Self {
        Self::Column(spec)
    }
}

impl From<PolyContrastSpec> for ContrastSpecEnum {
    fn from(spec: PolyContrastSpec) -> Self {
        Self::Poly(spec)
    }
}

impl From<OnewayContrastSpec> for ContrastSpecEnum {
    fn from(spec: OnewayContrastSpec) -> Self {
        Self::Oneway(spec)
    }
}

impl From<InteractionContrastSpec> for ContrastSpecEnum {
    fn from(spec: InteractionContrastSpec) -> Self {
        Self::Interaction(spec)
    }
}

impl From<ContrastFormulaSpec> for ContrastSpecEnum {
    fn from(spec: ContrastFormulaSpec) -> Self {
        Self::Formula(spec)
    }
}

impl From<FContrastSpec> for ContrastSpecEnum {
    fn from(spec: FContrastSpec) -> Self {
        Self::FContrast(spec)
    }
}

/// Collection of contrast specifications.
#[derive(Debug, Clone)]
pub struct ContrastSet {
    pub contrasts: Vec<ContrastSpecEnum>,
}

impl ContrastSet {
    /// Create a new contrast set.
    pub fn new() -> Self {
        Self {
            contrasts: Vec::new(),
        }
    }

    /// Add a contrast to the set.
    pub fn add<C: Into<ContrastSpecEnum>>(mut self, contrast: C) -> Self {
        self.contrasts.push(contrast.into());
        self
    }

    /// Iterate over contrasts.
    pub fn iter(&self) -> impl Iterator<Item = &ContrastSpecEnum> {
        self.contrasts.iter()
    }
}

// Convenience constructors

/// Create a unit contrast.
pub fn unit_contrast<S: Into<String>>(name: S, where_clause: Option<&str>) -> UnitContrastSpec {
    UnitContrastSpec {
        name: name.into(),
        where_clause: where_clause.map(Formula::from),
    }
}

/// Create a pairwise contrast.
pub fn pair_contrast<S: Into<String>>(
    name: S,
    a: &str,
    b: &str,
    where_clause: Option<&str>,
) -> PairContrastSpec {
    PairContrastSpec {
        name: name.into(),
        a: Formula::from(a),
        b: Formula::from(b),
        where_clause: where_clause.map(Formula::from),
    }
}

/// Create a column contrast.
pub fn column_contrast<S: Into<String>>(
    name: S,
    pattern_a: &str,
    pattern_b: Option<&str>,
) -> ColumnContrastSpec {
    ColumnContrastSpec {
        name: name.into(),
        pattern_a: pattern_a.to_string(),
        pattern_b: pattern_b.map(|s| s.to_string()),
        where_clause: None,
    }
}

/// Create a polynomial contrast.
pub fn poly_contrast<S: Into<String>>(name: S, factor: &str, degree: usize) -> PolyContrastSpec {
    PolyContrastSpec {
        name: name.into(),
        a: Formula::from(factor),
        degree,
        value_map: None,
        where_clause: None,
    }
}

/// Create a one-way contrast.
pub fn oneway_contrast<S: Into<String>>(
    name: S,
    factor: &str,
    where_clause: Option<&str>,
) -> OnewayContrastSpec {
    OnewayContrastSpec {
        name: name.into(),
        a: Formula::from(factor),
        where_clause: where_clause.map(Formula::from),
    }
}

/// Create an F-contrast from multiple sub-contrasts.
pub fn f_contrast_spec<S: Into<String>>(
    name: S,
    sub_contrasts: Vec<ContrastSpecEnum>,
) -> FContrastSpec {
    FContrastSpec {
        name: name.into(),
        sub_contrasts,
        where_clause: None,
    }
}

/// Create an interaction contrast.
pub fn interaction_contrast<S: Into<String>>(
    name: S,
    factors: &str,
    where_clause: Option<&str>,
) -> InteractionContrastSpec {
    InteractionContrastSpec {
        name: name.into(),
        a: Formula::from(factors),
        where_clause: where_clause.map(Formula::from),
    }
}

/// Create a one-against-all contrast.
pub fn one_against_all_contrast<S: Into<String>>(factor: S, levels: &[String]) -> ContrastSet {
    let factor_name = factor.into();
    let mut set = ContrastSet::new();

    for level in levels {
        let name = format!("{}.{}_vs_rest", factor_name, level);
        let a = format!("{} == '{}'", factor_name, level);
        let b = format!("{} != '{}'", factor_name, level);
        set = set.add(pair_contrast(name, &a, &b, None));
    }

    set
}

/// Create pairwise contrasts for all pairs of levels.
pub fn pairwise_contrasts<S: Into<String>>(factor: S, levels: &[String]) -> ContrastSet {
    let factor_name = factor.into();
    let mut set = ContrastSet::new();

    for i in 0..levels.len() {
        for j in (i + 1)..levels.len() {
            let name = format!("{}.{}_vs_{}", factor_name, levels[i], levels[j]);
            let a = format!("{} == '{}'", factor_name, levels[i]);
            let b = format!("{} == '{}'", factor_name, levels[j]);
            set = set.add(pair_contrast(name, &a, &b, None));
        }
    }

    set
}

/// Generate sliding-window contrasts across ordered levels.
pub fn sliding_window_contrasts<S: Into<String>>(
    factor: S,
    levels: &[String],
    window_size: usize,
) -> Vec<ContrastSpecEnum> {
    let factor_name = factor.into();
    let n = levels.len();
    if n < 2 || window_size == 0 || 2 * window_size > n {
        return Vec::new();
    }
    let n_contrasts = n - 2 * window_size + 1;
    let mut contrasts = Vec::with_capacity(n_contrasts);
    for i in 0..n_contrasts {
        let a_levels = &levels[i..i + window_size];
        let b_levels = &levels[i + window_size..i + 2 * window_size];
        let a_expr = if window_size == 1 {
            format!("{} == '{}'", factor_name, a_levels[0])
        } else {
            let quoted: Vec<String> = a_levels.iter().map(|l| format!("\"{}\"", l)).collect();
            format!("{} %in% c({})", factor_name, quoted.join(","))
        };
        let b_expr = if window_size == 1 {
            format!("{} == '{}'", factor_name, b_levels[0])
        } else {
            let quoted: Vec<String> = b_levels.iter().map(|l| format!("\"{}\"", l)).collect();
            format!("{} %in% c({})", factor_name, quoted.join(","))
        };
        let con_name = format!("win_{}_vs_{}", a_levels.join("-"), b_levels.join("-"));
        contrasts.push(ContrastSpecEnum::Pair(PairContrastSpec {
            name: con_name,
            a: Formula::new(a_expr),
            b: Formula::new(b_expr),
            where_clause: None,
        }));
    }
    contrasts
}
