//! Tests for formula parsing

use fmridesign::formula::{parse_formula, FormulaParser, HrfSpec, Term, TermParam};

#[test]
fn test_parse_simple_variable() {
    let parser = FormulaParser::new();
    let formula = parser.parse("condition").unwrap();

    assert!(formula.lhs.is_none());
    // Direct comparison of the expression would require accessing the enum variant

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition"]);
}

#[test]
fn test_parse_formula_with_lhs() {
    let parser = FormulaParser::new();
    let formula = parser.parse("onset ~ condition + rating").unwrap();

    assert_eq!(formula.lhs, Some("onset".to_string()));

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert_eq!(terms[1].events, vec!["rating"]);
}

#[test]
fn test_parse_hrf_function() {
    let parser = FormulaParser::new();
    let formula = parser.parse("hrf(condition)").unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition"]);
}

#[test]
fn test_parse_hrf_with_type() {
    let parser = FormulaParser::new();
    let formula = parser.parse("hrf(condition, \"gamma\")").unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition"]);

    match &terms[0].hrf {
        Some(HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
        _ => panic!("Expected named HRF"),
    }
}

#[test]
fn test_parse_hrf_with_kwargs() {
    let parser = FormulaParser::new();
    let formula = parser
        .parse("hrf(condition, hrf=\"spm_canonical\", duration=2.0)")
        .unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);

    match &terms[0].hrf {
        Some(HrfSpec::Named(name)) => assert_eq!(name, "spm_canonical"),
        _ => panic!("Expected named HRF"),
    }

    match terms[0].params.get("duration") {
        Some(TermParam::Number(n)) => assert_eq!(*n, 2.0),
        _ => panic!("Expected duration parameter"),
    }
}

#[test]
fn test_parse_interaction() {
    let parser = FormulaParser::new();
    let formula = parser.parse("condition:block").unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["condition", "block"]);
    assert!(terms[0].is_interaction());
}

#[test]
fn test_parse_complex_formula() {
    let parser = FormulaParser::new();
    let formula = parser
        .parse("onset ~ hrf(condition) + hrf(rating) + condition:block")
        .unwrap();

    assert_eq!(formula.lhs, Some("onset".to_string()));

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 3);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert_eq!(terms[1].events, vec!["rating"]);
    assert_eq!(terms[2].events, vec!["condition", "block"]);
}

#[test]
fn test_parse_poly_function() {
    let parser = FormulaParser::new();
    let formula = parser.parse("poly(rating, 3)").unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["rating"]);

    match terms[0].params.get("poly_degree") {
        Some(TermParam::Number(n)) => assert_eq!(*n, 3.0),
        _ => panic!("Expected poly_degree parameter"),
    }
}

#[test]
fn test_parse_trialwise() {
    let parser = FormulaParser::new();
    let formula = parser
        .parse("trialwise(basis=\"spmg1\", lag=0.5, nbasis=3)")
        .unwrap();

    let terms = parser.to_terms(&formula).unwrap();
    assert_eq!(terms.len(), 1);
    assert_eq!(terms[0].events, vec!["trialwise"]);

    // Check parameters
    match terms[0].params.get("basis") {
        Some(TermParam::String(s)) => assert_eq!(s, "spmg1"),
        _ => panic!("Expected basis parameter"),
    }

    match terms[0].params.get("lag") {
        Some(TermParam::Number(n)) => assert_eq!(*n, 0.5),
        _ => panic!("Expected lag parameter"),
    }

    match terms[0].params.get("nbasis") {
        Some(TermParam::Number(n)) => assert_eq!(*n, 3.0),
        _ => panic!("Expected nbasis parameter"),
    }
}

#[test]
fn test_parse_invalid_syntax() {
    let parser = FormulaParser::new();

    // Missing closing parenthesis
    assert!(parser.parse("hrf(condition").is_err());

    // Invalid operator
    assert!(parser.parse("condition & rating").is_err());

    // Empty function - This is actually valid syntax
    // hrf() means hrf with no arguments
    let parsed = parser.parse("hrf()").unwrap();
    assert!(parsed.lhs.is_none());
    // The RHS should be a function call expression
}

#[test]
fn test_parse_formula_convenience() {
    // Test the convenience function
    let terms = parse_formula("condition + rating").unwrap();
    assert_eq!(terms.len(), 2);
    assert_eq!(terms[0].events, vec!["condition"]);
    assert_eq!(terms[1].events, vec!["rating"]);
}
