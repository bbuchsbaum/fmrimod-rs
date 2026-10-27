//! Domain-specific language (DSL) for formula specification using operators.

use super::term::{HrfSpec, Term};
use crate::types::{BasisFunction, ValidationError};
use std::ops::{Add, Mul};

/// Represents an event variable in the DSL.
///
/// This enables the use of Rust operators to create terms in a natural syntax.
///
/// # Examples
/// ```
/// use fmridesign::formula::{event, hrf};
///
/// let condition = event("condition");
/// let block = event("block");
///
/// // Single event with HRF
/// let term1 = condition.clone().hrf("spm_canonical");
///
/// // Interaction
/// let term2 = condition * block;
/// ```
#[derive(Debug, Clone)]
pub struct EventVar {
    name: String,
}

impl EventVar {
    /// Create a new event variable.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Apply HRF to this event.
    pub fn hrf(self, hrf_spec: impl Into<HrfSpec>) -> Term {
        Term::new(self.name).with_hrf(hrf_spec)
    }

    /// Apply basis function to this event.
    pub fn basis(self, basis: Box<dyn BasisFunction>) -> Term {
        Term::new(self.name).with_basis(basis)
    }

    /// Convert to a term.
    pub fn into_term(self) -> Term {
        Term::new(self.name)
    }
}

/// Expression combining multiple events (e.g., interactions).
#[derive(Debug, Clone)]
pub struct EventExpr {
    events: Vec<String>,
}

impl EventExpr {
    /// Apply HRF to this expression.
    pub fn hrf(self, hrf_spec: impl Into<HrfSpec>) -> Result<Term, ValidationError> {
        Term::interaction(self.events).map(|t| t.with_hrf(hrf_spec))
    }

    /// Apply basis function to this expression.
    pub fn basis(self, basis: Box<dyn BasisFunction>) -> Result<Term, ValidationError> {
        Term::interaction(self.events).map(|t| t.with_basis(basis))
    }

    /// Convert to a term.
    pub fn into_term(self) -> Result<Term, ValidationError> {
        Term::interaction(self.events)
    }
}

/// Create interaction using multiplication operator.
impl Mul for EventVar {
    type Output = EventExpr;

    fn mul(self, rhs: EventVar) -> Self::Output {
        EventExpr {
            events: vec![self.name, rhs.name],
        }
    }
}

impl Mul<EventVar> for EventExpr {
    type Output = EventExpr;

    fn mul(mut self, rhs: EventVar) -> Self::Output {
        self.events.push(rhs.name);
        self
    }
}

/// Represents a model formula that can be built incrementally.
#[derive(Debug, Clone)]
pub struct FormulaExpr {
    terms: Vec<Term>,
}

impl FormulaExpr {
    /// Create a new empty formula.
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a term to the formula.
    pub fn add_term(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }

    /// Get the terms in this formula.
    pub fn terms(&self) -> &[Term] {
        &self.terms
    }

    /// Convert to a vector of terms.
    pub fn into_terms(self) -> Vec<Term> {
        self.terms
    }
}

impl Default for FormulaExpr {
    fn default() -> Self {
        Self::new()
    }
}

/// Add terms to create a formula.
impl Add<Term> for Term {
    type Output = FormulaExpr;

    fn add(self, rhs: Term) -> Self::Output {
        FormulaExpr::new().add_term(self).add_term(rhs)
    }
}

impl Add<Term> for FormulaExpr {
    type Output = FormulaExpr;

    fn add(self, rhs: Term) -> Self::Output {
        self.add_term(rhs)
    }
}

/// HRF namespace for convenient HRF specifications.
pub struct HrfNamespace;

impl HrfNamespace {
    /// SPM canonical HRF.
    pub fn spm_canonical() -> HrfSpec {
        HrfSpec::Named("spm_canonical".to_string())
    }

    /// Gamma HRF.
    pub fn gamma() -> HrfSpec {
        HrfSpec::Named("gamma".to_string())
    }

    /// Simple HRF.
    pub fn simple() -> HrfSpec {
        HrfSpec::Named("simple".to_string())
    }

    /// Double gamma HRF.
    pub fn double_gamma() -> HrfSpec {
        HrfSpec::Named("double_gamma".to_string())
    }
}

/// Convenient function to create an event variable.
pub fn event(name: impl Into<String>) -> EventVar {
    EventVar::new(name)
}

/// HRF namespace instance.
/// HRF namespace instance.
#[allow(non_upper_case_globals)]
pub const hrf: HrfNamespace = HrfNamespace;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let e = event("condition");
        assert_eq!(e.name, "condition");
    }

    #[test]
    fn test_event_with_hrf() {
        let term = event("stimulus").hrf("gamma");
        assert_eq!(term.events, vec!["stimulus"]);
        match term.hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
            _ => panic!("Expected named HRF"),
        }
    }

    #[test]
    fn test_event_interaction() {
        let cond = event("condition");
        let block = event("block");
        let expr = cond * block;
        assert_eq!(expr.events, vec!["condition", "block"]);
    }

    #[test]
    fn test_formula_building() {
        let cond = event("condition").hrf("spm_canonical");
        let rating = event("rating").into_term();

        let formula = cond + rating;
        assert_eq!(formula.terms.len(), 2);
    }

    #[test]
    fn test_hrf_namespace() {
        let term = event("stimulus").hrf(HrfNamespace::spm_canonical());
        match term.hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "spm_canonical"),
            _ => panic!("Expected named HRF"),
        }
    }
}
