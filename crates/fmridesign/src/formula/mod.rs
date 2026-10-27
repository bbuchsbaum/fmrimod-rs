//! Formula parsing and DSL for model specification.

pub mod builder;
pub mod context;
pub mod dsl;
pub mod event_model;
pub mod parser;
pub mod term;

pub use builder::ExprBuilder;
pub use context::{
    parse_formula_with_context, DataSource, FormulaContext, FormulaContextBuilder, TermDataBinding,
    Variable,
};
pub use dsl::{event, hrf, EventVar, HrfNamespace};
pub use event_model::{event_model, EventModelBuilder};
pub use parser::{parse_formula, FormulaExpr, FormulaParser};
pub use term::{interaction, term, HrfSpec, Term, TermBuilder, TermParam};

use crate::types::ValidationError;

/// Parse a formula string directly.
///
/// This is a convenience function for `Formula::parse(formula)`.
///
/// # Examples
/// ```ignore
/// use fmridesign::formula::formula;
///
/// let f = formula("hrf(stimulus) + covariate(motion_x)").unwrap();
/// ```
pub fn formula(formula_str: &str) -> Result<Formula, ValidationError> {
    Formula::parse(formula_str)
}

/// Formula specification.
#[derive(Debug, Clone)]
pub struct Formula {
    /// Formula string
    pub formula: String,
}

impl Formula {
    /// Parse a formula string.
    pub fn parse(formula: &str) -> Result<Self, ValidationError> {
        // Validate the formula string by attempting to parse it
        let parser = FormulaParser::new();
        let _parsed = parser.parse(formula)?; // This validates the syntax

        Ok(Self {
            formula: formula.to_string(),
        })
    }

    /// Get the formula string.
    pub fn as_str(&self) -> &str {
        &self.formula
    }

    /// Parse and convert to Terms.
    pub fn to_terms(&self) -> Result<Vec<Term>, ValidationError> {
        let parser = FormulaParser::new();
        let parsed = parser.parse(&self.formula)?;
        parser.to_terms(&parsed)
    }

    /// Parse with context and create events.
    pub fn to_events(
        &self,
        context: &FormulaContext,
    ) -> Result<Vec<Box<dyn crate::types::EventSpec>>, ValidationError> {
        parse_formula_with_context(&self.formula, context)
    }

    /// Create a formula from individual terms.
    pub fn from_terms<I, T>(terms: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: AsRef<str>,
    {
        let terms_str: Vec<String> = terms.into_iter().map(|t| t.as_ref().to_string()).collect();

        Self {
            formula: terms_str.join(" + "),
        }
    }

    /// Add a term to this formula.
    pub fn add_term<T: AsRef<str>>(mut self, term: T) -> Self {
        if self.formula.is_empty() {
            self.formula = term.as_ref().to_string();
        } else {
            self.formula.push_str(" + ");
            self.formula.push_str(term.as_ref());
        }
        self
    }

    /// Combine two formulas with +.
    pub fn add(mut self, other: &Formula) -> Self {
        if self.formula.is_empty() {
            self.formula = other.formula.clone();
        } else {
            self.formula.push_str(" + ");
            self.formula.push_str(&other.formula);
        }
        self
    }
}

impl std::fmt::Display for Formula {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.formula)
    }
}

impl From<&str> for Formula {
    fn from(formula_str: &str) -> Self {
        Self {
            formula: formula_str.to_string(),
        }
    }
}

impl From<String> for Formula {
    fn from(formula: String) -> Self {
        Self { formula }
    }
}

/// Builder for creating formulas.
#[derive(Debug)]
pub struct FormulaBuilder {
    terms: Vec<String>,
}

impl FormulaBuilder {
    /// Create a new formula builder.
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a term to the formula.
    pub fn add_term<S: Into<String>>(mut self, term: S) -> Self {
        self.terms.push(term.into());
        self
    }

    /// Build the formula.
    pub fn build(self) -> Formula {
        Formula {
            formula: self.terms.join(" + "),
        }
    }
}

impl Default for FormulaBuilder {
    fn default() -> Self {
        Self::new()
    }
}
