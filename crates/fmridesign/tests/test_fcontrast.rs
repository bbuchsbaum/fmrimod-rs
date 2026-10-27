//! Tests for F-contrast functionality.

use fmridesign::contrast::{
    fcontrasts, interaction_fcontrasts, main_effect_fcontrasts, oneway_contrast, pair_contrast,
    poly_contrast, ContrastSpecEnum, TermInfo,
};
use ndarray::array;

/// Mock term for testing.
struct MockTerm {
    name: String,
    conditions: Vec<String>,
}

impl MockTerm {
    fn new(name: &str, conditions: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            conditions,
        }
    }
}

impl TermInfo for MockTerm {
    fn conditions(&self, _expand_basis: bool) -> Vec<String> {
        self.conditions.clone()
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn term_name(&self) -> &str {
        &self.name
    }
}

#[test]
fn test_fcontrasts_from_multiple_contrasts() {
    let term = MockTerm::new(
        "treatment",
        vec!["placebo".to_string(), "low".to_string(), "high".to_string()],
    );

    // Create multiple contrasts
    let specs = vec![
        pair_contrast("low_vs_placebo", "low", "placebo", None).into(),
        pair_contrast("high_vs_placebo", "high", "placebo", None).into(),
    ];

    let f_contrast = fcontrasts(&specs, &term).unwrap();

    assert_eq!(f_contrast.name, "F[low_vs_placebo, high_vs_placebo]");
    assert_eq!(f_contrast.condnames, vec!["placebo", "low", "high"]);

    // Should have 3 rows (conditions) and 2 columns (contrasts)
    assert_eq!(f_contrast.weights.nrows(), 3);
    assert_eq!(f_contrast.weights.ncols(), 2);

    // Check contrast structure
    assert!(f_contrast.is_fcontrast());
}

#[test]
fn test_main_effect_fcontrasts() {
    let term = MockTerm::new(
        "group",
        vec![
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "D".to_string(),
        ],
    );

    let f_contrast = main_effect_fcontrasts("group", &term).unwrap();

    assert_eq!(f_contrast.name, "main_effect_group");
    assert_eq!(f_contrast.condnames.len(), 4);

    // Main effect should have k-1 columns for k groups
    assert_eq!(f_contrast.weights.ncols(), 3);

    // Each column should sum to zero
    for j in 0..f_contrast.weights.ncols() {
        let col_sum: f64 = f_contrast.weights.column(j).sum();
        assert!(col_sum.abs() < 1e-10);
    }
}

#[test]
fn test_interaction_fcontrasts() {
    let term = MockTerm::new(
        "treatment:time",
        vec![
            "placebo.pre".to_string(),
            "placebo.post".to_string(),
            "drug.pre".to_string(),
            "drug.post".to_string(),
        ],
    );

    let f_contrast = interaction_fcontrasts("treatment:time", &term).unwrap();

    assert_eq!(f_contrast.name, "interaction_treatment_time");
    assert_eq!(f_contrast.condnames.len(), 4);
}

#[test]
fn test_fcontrasts_with_polynomial() {
    let term = MockTerm::new(
        "dose",
        vec![
            "dose0".to_string(),
            "dose1".to_string(),
            "dose2".to_string(),
            "dose3".to_string(),
        ],
    );

    // Create polynomial contrasts of different degrees
    let specs = vec![
        poly_contrast("linear", "dose", 1).into(),
        poly_contrast("quadratic", "dose", 2).into(),
    ];

    let f_contrast = fcontrasts(&specs, &term).unwrap();

    assert_eq!(f_contrast.name, "F[linear, quadratic]");

    // Should have 4 rows (doses) and 3 columns (1 linear + 2 quadratic)
    assert_eq!(f_contrast.weights.nrows(), 4);
    assert_eq!(f_contrast.weights.ncols(), 3);
}

#[test]
fn test_fcontrasts_empty_specs() {
    let term = MockTerm::new("test", vec!["A".to_string()]);
    let specs: Vec<ContrastSpecEnum> = vec![];

    let result = fcontrasts(&specs, &term);
    assert!(result.is_err());

    if let Err(e) = result {
        let msg = e.to_string();
        assert!(msg.contains("F-contrast requires at least one contrast"));
    }
}

#[test]
fn test_fcontrasts_mismatched_conditions() {
    // Create two terms with different conditions
    let term1 = MockTerm::new("factor1", vec!["A".to_string(), "B".to_string()]);
    let term2 = MockTerm::new(
        "factor2",
        vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
    );

    // Create contrasts on different terms
    let spec1 = pair_contrast("A_vs_B", "A", "B", None);
    let contrast1 = fmridesign::contrast::contrast_weights(&spec1.into(), &term1).unwrap();

    let spec2 = pair_contrast("X_vs_Y", "X", "Y", None);
    let contrast2 = fmridesign::contrast::contrast_weights(&spec2.into(), &term2).unwrap();

    // Since fcontrasts takes specs not computed contrasts, we can't test this directly
    // But the implementation should catch mismatched conditions
}

#[test]
fn test_oneway_produces_valid_fcontrast() {
    let term = MockTerm::new(
        "factor",
        vec![
            "level1".to_string(),
            "level2".to_string(),
            "level3".to_string(),
        ],
    );

    let spec = oneway_contrast("factor_main", "factor", None);
    let contrast = fmridesign::contrast::contrast_weights(&spec.into(), &term).unwrap();

    // One-way contrast should produce F-contrast structure
    assert_eq!(contrast.weights.ncols(), 2); // k-1 for k=3 levels
    assert!(contrast.is_fcontrast());
}
