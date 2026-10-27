//! Tests for contrast weight computation.

use fmridesign::contrast::{
    column_contrast, contrast_weights, oneway_contrast, pair_contrast, poly_contrast,
    unit_contrast, ContrastSpecEnum, FContrastSpec, TermInfo,
};
use ndarray::array;

/// Mock term for testing.
struct MockTerm {
    name: String,
    conditions: Vec<String>,
    nbasis: usize,
}

impl MockTerm {
    fn new(name: &str, conditions: Vec<String>) -> Self {
        Self {
            name: name.to_string(),
            conditions,
            nbasis: 1,
        }
    }

    fn with_basis(mut self, nbasis: usize) -> Self {
        self.nbasis = nbasis;
        self
    }
}

impl TermInfo for MockTerm {
    fn conditions(&self, expand_basis: bool) -> Vec<String> {
        if expand_basis && self.nbasis > 1 {
            let mut expanded = Vec::new();
            for i in 0..self.nbasis {
                for cond in &self.conditions {
                    expanded.push(format!("{}.{}", cond, i + 1));
                }
            }
            expanded
        } else {
            self.conditions.clone()
        }
    }

    fn nbasis(&self) -> usize {
        self.nbasis
    }

    fn term_name(&self) -> &str {
        &self.name
    }
}

#[test]
fn test_pair_contrast_weights() {
    let term = MockTerm::new(
        "factor",
        vec!["A".to_string(), "B".to_string(), "C".to_string()],
    );

    let spec = pair_contrast("A_vs_B", "A", "B", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    assert_eq!(contrast.name, "A_vs_B");
    assert_eq!(contrast.condnames, vec!["A", "B", "C"]);

    let weights = contrast.weight_vector().unwrap();
    assert_eq!(weights.len(), 3);

    // A should have positive weight
    assert!(weights[0] > 0.0);
    // B should have negative weight
    assert!(weights[1] < 0.0);
    // C should have zero weight
    assert_eq!(weights[2], 0.0);

    // Weights should sum to approximately zero
    let sum: f64 = weights.iter().sum();
    assert!(sum.abs() < 1e-10);
}

#[test]
fn pair_selectors_do_not_match_partial_condition_names() {
    let bare = MockTerm::new("factor", vec!["A".into(), "AB".into(), "C".into()]);
    let namespaced = MockTerm::new(
        "factor",
        vec![
            "condition.A".into(),
            "condition.AB".into(),
            "condition.C".into(),
        ],
    );
    let spec = pair_contrast("a_vs_c", "A", "C", None);

    assert_eq!(
        contrast_weights(&spec.clone().into(), &bare)
            .unwrap()
            .weight_vector()
            .unwrap(),
        array![1.0, 0.0, -1.0]
    );
    assert_eq!(
        contrast_weights(&spec.into(), &namespaced)
            .unwrap()
            .weight_vector()
            .unwrap(),
        array![1.0, 0.0, -1.0]
    );
}

#[test]
fn pair_level_selection_stays_aligned_after_basis_expansion() {
    let term = MockTerm::new(
        "factor",
        vec![
            "condition.A".into(),
            "condition.AB".into(),
            "condition.C".into(),
        ],
    )
    .with_basis(2);
    let spec = pair_contrast("a_vs_c", "A", "C", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    assert_eq!(
        contrast.condnames,
        vec![
            "condition.A.1",
            "condition.AB.1",
            "condition.C.1",
            "condition.A.2",
            "condition.AB.2",
            "condition.C.2",
        ]
    );
    assert_eq!(
        contrast.weight_vector().unwrap(),
        array![1.0, 0.0, -1.0, 1.0, 0.0, -1.0]
    );
}

#[test]
fn test_column_contrast_weights() {
    let term = MockTerm::new(
        "stimulus",
        vec![
            "stim.A.1".to_string(),
            "stim.A.2".to_string(),
            "stim.B.1".to_string(),
            "stim.B.2".to_string(),
        ],
    );

    let spec = column_contrast("A_vs_B", r"\.A\.", Some(r"\.B\."));
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    assert_eq!(contrast.name, "A_vs_B");
    assert_eq!(contrast.condnames.len(), 4);

    let weights = contrast.weight_vector().unwrap();

    // Check that A columns have positive weights
    assert!(weights[0] > 0.0);
    assert!(weights[1] > 0.0);

    // Check that B columns have negative weights
    assert!(weights[2] < 0.0);
    assert!(weights[3] < 0.0);

    // Check normalization
    assert!((weights[0] - 0.5).abs() < 1e-10);
    assert!((weights[2] + 0.5).abs() < 1e-10);
}

#[test]
fn test_polynomial_contrast() {
    let term = MockTerm::new(
        "dose",
        vec![
            "dose0".to_string(),
            "dose1".to_string(),
            "dose2".to_string(),
            "dose3".to_string(),
        ],
    );

    let spec = poly_contrast("linear_trend", "dose", 1);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    assert_eq!(contrast.name, "linear_trend");

    // Should have 1 column for linear trend
    assert_eq!(contrast.weights.ncols(), 1);

    let weights = contrast.weight_vector().unwrap();

    // Linear trend should increase monotonically
    for i in 1..weights.len() {
        assert!(weights[i] > weights[i - 1]);
    }

    // Should sum to zero
    let sum: f64 = weights.iter().sum();
    assert!(sum.abs() < 1e-10);
}

#[test]
fn test_oneway_contrast() {
    let term = MockTerm::new(
        "group",
        vec![
            "control".to_string(),
            "treatment1".to_string(),
            "treatment2".to_string(),
        ],
    );

    let spec = oneway_contrast("group_effect", "group", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    assert_eq!(contrast.name, "group_effect");

    // One-way contrast should produce k-1 columns for k levels
    assert_eq!(contrast.weights.ncols(), 2);

    // Each column should sum to zero
    for j in 0..contrast.weights.ncols() {
        let col_sum: f64 = contrast.weights.column(j).sum();
        assert!(col_sum.abs() < 1e-10);
    }
}

#[test]
fn test_contrast_with_basis_expansion() {
    let term = MockTerm::new("stimulus", vec!["A".to_string(), "B".to_string()]).with_basis(3);

    let spec = pair_contrast("A_vs_B", "A", "B", None);
    let contrast = contrast_weights(&spec.into(), &term).unwrap();

    // With basis expansion, should have 2 conditions * 3 basis = 6 rows
    let expanded_conds = term.conditions(true);
    assert_eq!(expanded_conds.len(), 6);
    assert_eq!(contrast.condnames.len(), 6);

    let weights = contrast.weight_vector().unwrap();

    assert_eq!(
        contrast.condnames,
        vec!["A.1", "B.1", "A.2", "B.2", "A.3", "B.3"]
    );
    assert_eq!(weights, array![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]);

    // Every basis block pairs A's positive weight with B's negative weight.
    assert!(weights[0] > 0.0);
    assert!(weights[1] < 0.0);
    assert!(weights[2] > 0.0);
    assert!(weights[3] < 0.0);
    assert!(weights[4] > 0.0);
    assert!(weights[5] < 0.0);
}

#[test]
fn unit_and_polynomial_contrasts_expand_basis_major() {
    let unit_term = MockTerm::new("stimulus", vec!["A".into(), "B".into()]).with_basis(3);
    let unit = contrast_weights(&unit_contrast("mean", None).into(), &unit_term).unwrap();
    assert_eq!(
        unit.condnames,
        vec!["A.1", "B.1", "A.2", "B.2", "A.3", "B.3"]
    );
    assert_eq!(
        unit.weight_vector().unwrap(),
        array![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    );

    let poly_term =
        MockTerm::new("dose", vec!["low".into(), "mid".into(), "high".into()]).with_basis(2);
    let poly = contrast_weights(&poly_contrast("linear", "dose", 1).into(), &poly_term).unwrap();
    assert_eq!(
        poly.condnames,
        vec!["low.1", "mid.1", "high.1", "low.2", "mid.2", "high.2"]
    );
    assert_eq!(poly.weights.row(0), poly.weights.row(3));
    assert_eq!(poly.weights.row(1), poly.weights.row(4));
    assert_eq!(poly.weights.row(2), poly.weights.row(5));
}

#[test]
fn mixed_fcontrast_preserves_aligned_subcontrast_rows() {
    let term = MockTerm::new("stimulus", vec!["A".into(), "B".into()]).with_basis(2);
    let spec = ContrastSpecEnum::FContrast(FContrastSpec {
        name: "mixed".into(),
        sub_contrasts: vec![
            unit_contrast("mean", None).into(),
            pair_contrast("a_vs_b", "A", "B", None).into(),
        ],
        where_clause: None,
    });
    let contrast = contrast_weights(&spec, &term).unwrap();

    assert_eq!(contrast.condnames, vec!["A.1", "B.1", "A.2", "B.2"]);
    assert_eq!(
        contrast.weights,
        array![[0.5, 1.0], [0.5, -1.0], [0.5, 1.0], [0.5, -1.0]]
    );
}

struct InvalidBasisMetadataTerm;

impl TermInfo for InvalidBasisMetadataTerm {
    fn conditions(&self, expand_basis: bool) -> Vec<String> {
        if expand_basis {
            vec!["A.1".into()]
        } else {
            vec!["A".into(), "B".into()]
        }
    }

    fn nbasis(&self) -> usize {
        2
    }

    fn term_name(&self) -> &str {
        "invalid"
    }
}

#[test]
fn invalid_basis_metadata_is_rejected() {
    let error = contrast_weights(
        &unit_contrast("mean", None).into(),
        &InvalidBasisMetadataTerm,
    )
    .unwrap_err();
    assert!(error.to_string().contains("basis-expanded condition count"));
}

#[test]
fn explicit_and_linear_contrasts_keep_the_requested_estimand() {
    use fmridesign::contrast::{contrast, ContrastFormulaSpec, Formula};
    let term = MockTerm::new("factor", vec!["A".into(), "B".into(), "C".into()]);
    let vector = contrast("custom", vec![0.0, 2.0, -3.0]).unwrap();
    assert_eq!(
        contrast_weights(&vector, &term)
            .unwrap()
            .weight_vector()
            .unwrap(),
        array![0.0, 2.0, -3.0]
    );
    for expr in ["2*B-3*C", "~ -C + 2*B - 2*C", "2e0*`B` - 3*C"] {
        let spec = ContrastFormulaSpec {
            name: "linear".into(),
            formula: Formula::new(expr),
            where_clause: None,
        };
        assert_eq!(
            contrast_weights(&spec.into(), &term)
                .unwrap()
                .weight_vector()
                .unwrap(),
            array![0.0, 2.0, -3.0]
        );
    }
    for expr in ["missing", "A B", "A*B", "A/2", "[1,2]", "[NaN,0,0]", ""] {
        let spec = ContrastFormulaSpec {
            name: "invalid".into(),
            formula: Formula::new(expr),
            where_clause: None,
        };
        assert!(contrast_weights(&spec.into(), &term).is_err(), "{expr}");
    }
}

#[test]
fn a_missing_pair_group_cannot_silently_become_a_unit_contrast() {
    let term = MockTerm::new("condition", vec!["A".into(), "AB".into(), "C".into()]);
    for (a, b) in [("A", "missing"), ("missing", "C")] {
        let spec = pair_contrast("missing", a, b, None);
        assert!(contrast_weights(&spec.into(), &term).is_err());
    }
}
