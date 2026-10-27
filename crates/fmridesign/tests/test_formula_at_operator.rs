//! Tests for the @ operator in formula DSL

use fmridesign::formula::{parse_formula, FormulaParser, TermParam};
use fmridesign::types::ValidationError;

#[test]
fn test_at_operator_with_poly() {
    let parser = FormulaParser::new();

    // Basic polynomial expansion
    let formula = parser.parse("y ~ x@poly(2)").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "x");
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("poly".to_string()))
    );
    assert_eq!(
        terms[0].params.get("poly_degree"),
        Some(&TermParam::Number(2.0))
    );
}

#[test]
fn test_at_operator_with_bs() {
    let parser = FormulaParser::new();

    // B-spline expansion
    let formula = parser.parse("y ~ stimulus@bs(4)").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "stimulus");
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("bs".to_string()))
    );
    assert_eq!(terms[0].params.get("bs_df"), Some(&TermParam::Number(4.0)));
}

#[test]
fn test_at_operator_with_kwargs() {
    let parser = FormulaParser::new();

    // B-spline with keyword arguments
    let formula = parser.parse("y ~ rt@bs(df=6)").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "rt");
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("bs".to_string()))
    );
    assert_eq!(terms[0].params.get("bs_df"), Some(&TermParam::Number(6.0)));
}

#[test]
fn test_at_operator_with_simple_basis() {
    let parser = FormulaParser::new();

    // Simple basis name
    let formula = parser.parse("y ~ x@linear").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "x");
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("linear".to_string()))
    );
}

#[test]
fn test_at_operator_in_complex_formula() {
    let parser = FormulaParser::new();

    // Complex formula with @ operator
    let formula = parser
        .parse("y ~ condition + rt@poly(3) + hrf(stimulus)")
        .unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 3);

    // First term: condition
    assert_eq!(terms[0].effective_name(), "condition");

    // Second term: rt with polynomial basis
    assert_eq!(terms[1].effective_name(), "rt");
    assert_eq!(
        terms[1].params.get("basis_type"),
        Some(&TermParam::String("poly".to_string()))
    );
    assert_eq!(
        terms[1].params.get("poly_degree"),
        Some(&TermParam::Number(3.0))
    );

    // Third term: stimulus with HRF
    assert_eq!(terms[2].effective_name(), "stimulus");
    assert!(terms[2].hrf.is_some());
}

#[test]
fn test_at_operator_with_interaction() {
    let parser = FormulaParser::new();

    // Interaction with basis expansion
    let formula = parser.parse("y ~ condition:block@poly(2)").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "condition:block");
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("poly".to_string()))
    );
    assert_eq!(
        terms[0].params.get("poly_degree"),
        Some(&TermParam::Number(2.0))
    );
}

#[test]
fn test_at_operator_precedence() {
    let parser = FormulaParser::new();

    // @ should bind tighter than :
    let formula = parser.parse("y ~ x@poly(2):condition").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    // This should be parsed as (x@poly(2)):condition, creating an interaction
    // between the basis-expanded x and condition
    assert!(terms[0].is_interaction());
}

#[test]
fn test_invalid_at_operator_usage() {
    let parser = FormulaParser::new();

    // Invalid basis function
    let result = parser.parse("y ~ x@unknown(2)");
    assert!(result.is_err() || parser.to_terms(&result.unwrap()).is_err());

    // Invalid right-hand side
    let result = parser.parse("y ~ x@(1 + 2)");
    assert!(result.is_err() || parser.to_terms(&result.unwrap()).is_err());
}

#[test]
fn test_at_operator_with_hrf() {
    let parser = FormulaParser::new();

    // Combining @ operator with HRF
    let formula = parser.parse("y ~ hrf(stimulus@poly(2))").unwrap();
    let terms = parser.to_terms(&formula).unwrap();

    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].effective_name(), "stimulus");
    assert!(terms[0].hrf.is_some());
    assert_eq!(
        terms[0].params.get("basis_type"),
        Some(&TermParam::String("poly".to_string()))
    );
}
