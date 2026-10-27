//! Formula builder implementation.

use crate::formula::parser::FormulaExpr;
use crate::formula::term::Term;

/// Build formula expressions programmatically.
#[derive(Debug)]
pub struct ExprBuilder {
    expr: FormulaExpr,
}

impl ExprBuilder {
    /// Create a new expression builder from a variable.
    pub fn variable<S: Into<String>>(name: S) -> Self {
        Self {
            expr: FormulaExpr::Variable(name.into()),
        }
    }

    /// Add another term to this expression.
    pub fn add(self, other: ExprBuilder) -> Self {
        Self {
            expr: FormulaExpr::Addition(Box::new(self.expr), Box::new(other.expr)),
        }
    }

    /// Create an interaction with another term.
    pub fn interact(self, other: ExprBuilder) -> Self {
        Self {
            expr: FormulaExpr::Interaction(Box::new(self.expr), Box::new(other.expr)),
        }
    }

    /// Get the built expression.
    pub fn build(self) -> FormulaExpr {
        self.expr
    }
}

/// Event model builder using Terms.
#[derive(Debug)]
pub struct EventModelBuilder {
    terms: Vec<Term>,
}

impl EventModelBuilder {
    /// Create a new event model builder.
    pub fn new() -> Self {
        Self { terms: Vec::new() }
    }

    /// Add a term to the model.
    pub fn add_term(mut self, term: Term) -> Self {
        self.terms.push(term);
        self
    }

    /// Add multiple terms to the model.
    pub fn add_terms(mut self, terms: Vec<Term>) -> Self {
        self.terms.extend(terms);
        self
    }

    /// Build the model (returns the terms).
    pub fn build(self) -> Vec<Term> {
        self.terms
    }
}

impl Default for EventModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
