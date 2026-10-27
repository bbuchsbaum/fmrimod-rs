//! Tests for formula DSL

use fmridesign::formula::{event, hrf, interaction, term, HrfSpec, Term};

#[test]
fn test_event_creation() {
    let e = event("condition");
    // EventVar doesn't expose name directly, but we can convert to term
    let term = e.into_term();
    assert_eq!(term.effective_name(), "condition");
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
    let interaction = cond * block;

    // Convert to term to check events
    let term = interaction.into_term().unwrap();
    assert_eq!(term.events, vec!["condition", "block"]);
}

#[test]
fn test_interaction_with_hrf() {
    let cond = event("condition");
    let block = event("block");
    let term = (cond * block).hrf("spm_canonical").unwrap();

    assert_eq!(term.events, vec!["condition", "block"]);
    assert!(term.is_interaction());

    match term.hrf {
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

    match t.hrf {
        Some(HrfSpec::Named(name)) => assert_eq!(name, "gamma"),
        _ => panic!("Expected named HRF"),
    }

    match t.params.get("duration") {
        Some(fmridesign::formula::TermParam::Number(n)) => assert_eq!(*n, 2.0),
        _ => panic!("Expected duration parameter"),
    }
}

#[test]
fn test_interaction_builder() {
    let t = interaction(vec!["condition".to_string(), "block".to_string()])
        .unwrap()
        .hrf("double_gamma")
        .build();

    assert_eq!(t.events, vec!["condition", "block"]);
    assert!(t.is_interaction());
    assert_eq!(t.effective_name(), "condition:block");
}

#[test]
fn test_formula_building() {
    let cond = event("condition").hrf("spm_canonical");
    let rating = event("rating").into_term();

    let formula = cond + rating;
    assert_eq!(formula.terms().len(), 2);
}

#[test]
fn test_hrf_namespace() {
    use fmridesign::formula::HrfNamespace;
    let term = event("stimulus").hrf(HrfNamespace::spm_canonical());

    match term.hrf {
        Some(HrfSpec::Named(name)) => assert_eq!(name, "spm_canonical"),
        _ => panic!("Expected named HRF"),
    }
}

#[test]
fn test_term_with_name() {
    let t = Term::new("condition").with_name("main_effect");
    assert_eq!(t.name, Some("main_effect".to_string()));
    assert_eq!(t.effective_name(), "main_effect");
}

#[test]
fn test_term_without_name() {
    let t = Term::new("condition");
    assert_eq!(t.name, None);
    assert_eq!(t.effective_name(), "condition");
}

#[test]
fn test_term_set_param() {
    let t = Term::new("condition")
        .set_param("lag", 2.0)
        .set_param("type", "event");

    match t.params.get("lag") {
        Some(fmridesign::formula::TermParam::Number(n)) => assert_eq!(*n, 2.0),
        _ => panic!("Expected lag parameter"),
    }

    match t.params.get("type") {
        Some(fmridesign::formula::TermParam::String(s)) => assert_eq!(s, "event"),
        _ => panic!("Expected type parameter"),
    }
}
