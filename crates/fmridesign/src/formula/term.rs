//! Core Term structure for formula specification.

use crate::types::{BasisFunction, ValidationError};
use std::collections::HashMap;

/// Represents a term in an event model formula.
///
/// A term specifies how one or more events are transformed and combined
/// to create columns in the design matrix.
#[derive(Debug)]
pub struct Term {
    /// Event variable name(s). If multiple, creates an interaction.
    pub events: Vec<String>,
    /// HRF specification (name or parameters)
    pub hrf: Option<HrfSpec>,
    /// Basis function for continuous variables
    pub basis: Option<Box<dyn BasisFunction>>,
    /// Custom name for this term
    pub name: Option<String>,
    /// Additional parameters
    pub params: HashMap<String, TermParam>,
}

impl Clone for Term {
    fn clone(&self) -> Self {
        Self {
            events: self.events.clone(),
            hrf: self.hrf.clone(),
            basis: None, // Cannot clone trait objects, so we skip basis
            name: self.name.clone(),
            params: self.params.clone(),
        }
    }
}

/// HRF specification
#[derive(Debug, Clone)]
pub enum HrfSpec {
    /// Named HRF (e.g., "spm_canonical", "gamma")
    Named(String),
    /// Custom HRF parameters
    Custom(HashMap<String, f64>),
}

impl From<&str> for HrfSpec {
    fn from(s: &str) -> Self {
        HrfSpec::Named(s.to_string())
    }
}

impl From<String> for HrfSpec {
    fn from(s: String) -> Self {
        HrfSpec::Named(s)
    }
}

/// Parameter value for terms
#[derive(Debug, Clone, PartialEq)]
pub enum TermParam {
    /// String value
    String(String),
    /// Numeric value
    Number(f64),
    /// Boolean value
    Bool(bool),
    /// List of strings
    StringList(Vec<String>),
    /// List of numbers
    NumberList(Vec<f64>),
}

impl Term {
    /// Create a new term with a single event.
    pub fn new(event: impl Into<String>) -> Self {
        Self {
            events: vec![event.into()],
            hrf: None,
            basis: None,
            name: None,
            params: HashMap::new(),
        }
    }

    /// Create a new term with multiple events (interaction).
    pub fn interaction(events: Vec<String>) -> Result<Self, ValidationError> {
        if events.len() < 2 {
            return Err(ValidationError::InvalidParameter(
                "Interaction requires at least 2 events".to_string(),
            ));
        }

        Ok(Self {
            events,
            hrf: None,
            basis: None,
            name: None,
            params: HashMap::new(),
        })
    }

    /// Set HRF for this term.
    pub fn with_hrf(mut self, hrf: impl Into<HrfSpec>) -> Self {
        self.hrf = Some(hrf.into());
        self
    }

    /// Set basis function for this term.
    pub fn with_basis(mut self, basis: Box<dyn BasisFunction>) -> Self {
        self.basis = Some(basis);
        self
    }

    /// Set custom name for this term.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set a parameter.
    pub fn set_param(mut self, key: impl Into<String>, value: impl Into<TermParam>) -> Self {
        self.params.insert(key.into(), value.into());
        self
    }

    /// Get the effective name for this term.
    pub fn effective_name(&self) -> String {
        if let Some(ref name) = self.name {
            name.clone()
        } else if self.is_interaction() {
            self.events.join(":")
        } else {
            self.events[0].clone()
        }
    }

    /// Check if this is an interaction term.
    pub fn is_interaction(&self) -> bool {
        self.events.len() > 1
    }

    /// Get a parameter value.
    pub fn get_param(&self, key: &str) -> Option<f64> {
        self.params.get(key).and_then(|p| match p {
            TermParam::Number(n) => Some(*n),
            TermParam::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        })
    }
}

impl From<&str> for TermParam {
    fn from(s: &str) -> Self {
        TermParam::String(s.to_string())
    }
}

impl From<String> for TermParam {
    fn from(s: String) -> Self {
        TermParam::String(s)
    }
}

impl From<f64> for TermParam {
    fn from(n: f64) -> Self {
        TermParam::Number(n)
    }
}

impl From<bool> for TermParam {
    fn from(b: bool) -> Self {
        TermParam::Bool(b)
    }
}

impl From<Vec<String>> for TermParam {
    fn from(v: Vec<String>) -> Self {
        TermParam::StringList(v)
    }
}

impl From<Vec<f64>> for TermParam {
    fn from(v: Vec<f64>) -> Self {
        TermParam::NumberList(v)
    }
}

/// Builder for terms with a fluent interface.
pub struct TermBuilder {
    term: Term,
}

impl TermBuilder {
    /// Create a new term builder.
    pub fn new(event: impl Into<String>) -> Self {
        Self {
            term: Term::new(event),
        }
    }

    /// Create an interaction term builder.
    pub fn interaction(events: Vec<String>) -> Result<Self, ValidationError> {
        Ok(Self {
            term: Term::interaction(events)?,
        })
    }

    /// Set HRF.
    pub fn hrf(mut self, hrf: impl Into<HrfSpec>) -> Self {
        self.term = self.term.with_hrf(hrf);
        self
    }

    /// Set basis function.
    pub fn basis(mut self, basis: Box<dyn BasisFunction>) -> Self {
        self.term = self.term.with_basis(basis);
        self
    }

    /// Set custom name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.term = self.term.with_name(name);
        self
    }

    /// Set a parameter.
    pub fn param(mut self, key: impl Into<String>, value: impl Into<TermParam>) -> Self {
        self.term = self.term.set_param(key, value);
        self
    }

    /// Build the term.
    pub fn build(self) -> Term {
        self.term
    }
}

/// Convenience function for creating a term.
pub fn term(event: impl Into<String>) -> TermBuilder {
    TermBuilder::new(event)
}

/// Convenience function for creating an interaction term.
pub fn interaction(events: Vec<String>) -> Result<TermBuilder, ValidationError> {
    TermBuilder::interaction(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_creation() {
        let t = Term::new("condition");
        assert_eq!(t.events, vec!["condition"]);
        assert!(!t.is_interaction());
        assert_eq!(t.effective_name(), "condition");
    }

    #[test]
    fn test_interaction_term() {
        let t = Term::interaction(vec!["condition".to_string(), "block".to_string()]).unwrap();
        assert!(t.is_interaction());
        assert_eq!(t.effective_name(), "condition:block");
    }

    #[test]
    fn test_term_with_hrf() {
        let t = Term::new("condition").with_hrf("spm_canonical");
        match t.hrf {
            Some(HrfSpec::Named(name)) => assert_eq!(name, "spm_canonical"),
            _ => panic!("Expected named HRF"),
        }
    }

    #[test]
    fn test_term_builder() {
        let t = term("stimulus")
            .hrf("gamma")
            .name("stim")
            .param("duration", 2.0)
            .build();

        assert_eq!(t.events, vec!["stimulus"]);
        assert_eq!(t.name, Some("stim".to_string()));
        match t.params.get("duration") {
            Some(TermParam::Number(n)) => assert_eq!(*n, 2.0),
            _ => panic!("Expected numeric parameter"),
        }
    }
}
