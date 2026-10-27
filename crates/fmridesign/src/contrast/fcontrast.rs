//! F-contrast support for hypothesis testing.

use super::spec::*;
use super::weights::{contrast_weights, Contrast, TermInfo};
use crate::types::ValidationError;
use ndarray::{s, Array1, Array2};

/// Generate F-contrast matrix from multiple contrasts.
pub fn fcontrasts<T: TermInfo>(
    specs: &[ContrastSpecEnum],
    term: &T,
) -> Result<Contrast, ValidationError> {
    if specs.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "F-contrast requires at least one contrast specification".to_string(),
        ));
    }

    // Compute weights for each contrast
    let mut all_weights = Vec::new();
    let mut condnames = None;

    for spec in specs {
        let contrast = contrast_weights(spec, term)?;

        // Verify all contrasts have same condition names
        if let Some(ref names) = condnames {
            if &contrast.condnames != names {
                return Err(ValidationError::InvalidParameter(
                    "All contrasts must have the same condition names".to_string(),
                ));
            }
        } else {
            condnames = Some(contrast.condnames.clone());
        }

        // Extract weight vectors
        if contrast.weights.ndim() == 1 {
            all_weights.push(
                contrast
                    .weights
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap()
                    .to_owned(),
            );
        } else {
            // For multi-column contrasts, add each column
            for j in 0..contrast.weights.ncols() {
                all_weights.push(contrast.weights.column(j).to_owned());
            }
        }
    }

    let condnames = condnames.unwrap();
    let n_conds = condnames.len();
    let n_contrasts = all_weights.len();

    // Build F-contrast matrix
    let mut f_matrix = Array2::zeros((n_conds, n_contrasts));
    for (j, weights) in all_weights.iter().enumerate() {
        f_matrix.slice_mut(s![.., j]).assign(weights);
    }

    // Create combined name
    let names: Vec<&str> = specs.iter().map(|s| s.name()).collect();
    let combined_name = format!("F[{}]", names.join(", "));

    // Create a proper F-contrast spec
    let f_spec = FContrastSpec {
        name: combined_name.clone(),
        sub_contrasts: specs.to_vec(),
        where_clause: None,
    };

    Ok(Contrast::new(
        term.term_name().to_string(),
        combined_name,
        f_matrix,
        condnames,
        ContrastSpecEnum::FContrast(f_spec),
    ))
}

/// Generate F-contrasts for main effects.
pub fn main_effect_fcontrasts<T: TermInfo>(
    factor_name: &str,
    term: &T,
) -> Result<Contrast, ValidationError> {
    // Create oneway contrast spec
    let spec = oneway_contrast(format!("main_effect_{}", factor_name), factor_name, None);

    // Compute weights
    contrast_weights(&spec.into(), term)
}

/// Generate F-contrasts for interaction effects.
pub fn interaction_fcontrasts<T: TermInfo>(
    factors: &str,
    term: &T,
) -> Result<Contrast, ValidationError> {
    // Create interaction contrast spec
    let spec = interaction_contrast(
        format!("interaction_{}", factors.replace(':', "_")),
        factors,
        None,
    );

    // Compute weights
    contrast_weights(&spec.into(), term)
}

/// Compute Kronecker product of two matrices.
///
/// The Kronecker product A ⊗ B is a block matrix where each element a_ij of A
/// is replaced by the block a_ij * B.
pub fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    let mut result = Array2::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            let a_ij = a[[i, j]];
            let block = b.mapv(|b_val| a_ij * b_val);

            let row_start = i * p;
            let row_end = (i + 1) * p;
            let col_start = j * q;
            let col_end = (j + 1) * q;

            result
                .slice_mut(s![row_start..row_end, col_start..col_end])
                .assign(&block);
        }
    }

    result
}

/// Generate F-contrast for interactions using Kronecker products.
///
/// For an interaction A:B where A has p levels and B has q levels,
/// this generates the contrast matrix for testing the interaction effect.
pub fn interaction_fcontrast_kronecker<T: TermInfo>(
    factor_a: &str,
    factor_b: &str,
    term: &T,
) -> Result<Contrast, ValidationError> {
    // Get the cells (factorial combinations)
    let cells = term.cells().ok_or_else(|| {
        ValidationError::InvalidParameter(
            "Term must provide cells for interaction contrast".to_string(),
        )
    })?;

    let (cell_data, col_names) = cells;

    // Find indices for the two factors
    let factor_a_idx = col_names
        .iter()
        .position(|name| name == factor_a)
        .ok_or_else(|| {
            ValidationError::InvalidParameter(format!("Factor '{}' not found in term", factor_a))
        })?;

    let factor_b_idx = col_names
        .iter()
        .position(|name| name == factor_b)
        .ok_or_else(|| {
            ValidationError::InvalidParameter(format!("Factor '{}' not found in term", factor_b))
        })?;

    // Extract unique levels for each factor
    let mut levels_a: Vec<String> = Vec::new();
    let mut levels_b: Vec<String> = Vec::new();

    for row in &cell_data {
        let level_a = &row[factor_a_idx];
        let level_b = &row[factor_b_idx];

        if !levels_a.contains(level_a) {
            levels_a.push(level_a.clone());
        }
        if !levels_b.contains(level_b) {
            levels_b.push(level_b.clone());
        }
    }

    let n_a = levels_a.len();
    let n_b = levels_b.len();

    // Create contrast matrices for main effects
    // For factor A: centered contrasts (sum to zero)
    let mut c_a = Array2::zeros((n_a, n_a - 1));
    for i in 0..n_a - 1 {
        c_a[[i, i]] = 1.0;
        c_a[[n_a - 1, i]] = -1.0;
    }

    // For factor B: centered contrasts (sum to zero)
    let mut c_b = Array2::zeros((n_b, n_b - 1));
    for i in 0..n_b - 1 {
        c_b[[i, i]] = 1.0;
        c_b[[n_b - 1, i]] = -1.0;
    }

    // Compute Kronecker product for interaction contrast
    let interaction_matrix = kronecker_product(&c_a, &c_b);

    // Map to full design matrix columns
    let conditions = term.conditions(false);
    let n_conds = conditions.len();
    let n_contrasts = (n_a - 1) * (n_b - 1);

    let mut full_matrix = Array2::zeros((n_conds, n_contrasts));

    // Map each cell to its position in the full design
    for (cell_idx, row) in cell_data.iter().enumerate() {
        let level_a = &row[factor_a_idx];
        let level_b = &row[factor_b_idx];

        let a_idx = levels_a.iter().position(|l| l == level_a).unwrap();
        let b_idx = levels_b.iter().position(|l| l == level_b).unwrap();

        // Find corresponding condition
        let cond_idx = conditions
            .iter()
            .position(|cond| cond.contains(level_a) && cond.contains(level_b))
            .unwrap_or(cell_idx);

        // Copy interaction weights
        let interaction_idx = a_idx * n_b + b_idx;
        if interaction_idx < interaction_matrix.nrows() {
            full_matrix
                .slice_mut(s![cond_idx, ..])
                .assign(&interaction_matrix.row(interaction_idx));
        }
    }

    let name = format!("{}:{}_interaction", factor_a, factor_b);

    // Create a dummy spec for the interaction
    let spec = ContrastFormulaSpec {
        name: name.clone(),
        formula: Formula::from(format!("{} * {}", factor_a, factor_b)),
        where_clause: None,
    };

    Ok(Contrast::new(
        term.term_name().to_string(),
        name,
        full_matrix,
        conditions,
        ContrastSpecEnum::Formula(spec),
    ))
}

/// Create orthogonal polynomial contrasts using Kronecker products.
///
/// For a factor with k levels, this creates k-1 orthogonal polynomial contrasts
/// (linear, quadratic, cubic, etc.).
pub fn polynomial_contrasts_kronecker(
    _factor: &str,
    levels: &[String],
    max_degree: Option<usize>,
) -> Result<Array2<f64>, ValidationError> {
    let k = levels.len();
    if k < 2 {
        return Err(ValidationError::InvalidParameter(
            "Need at least 2 levels for polynomial contrasts".to_string(),
        ));
    }

    let max_deg = max_degree.unwrap_or(k - 1).min(k - 1);
    let mut contrast_matrix = Array2::zeros((k, max_deg));

    // Generate orthogonal polynomial values
    let x: Vec<f64> = (0..k).map(|i| i as f64).collect();
    let x_centered: Vec<f64> = x.iter().map(|&xi| xi - (k - 1) as f64 / 2.0).collect();

    // Linear contrast
    if max_deg >= 1 {
        let linear: Array1<f64> = Array1::from_vec(x_centered.clone());
        let norm = (linear.dot(&linear)).sqrt();
        contrast_matrix.column_mut(0).assign(&(linear / norm));
    }

    // Quadratic contrast
    if max_deg >= 2 {
        let quadratic: Array1<f64> = x_centered
            .iter()
            .map(|&xi| xi * xi - (k * k - 1) as f64 / 12.0)
            .collect();
        let norm = (quadratic.dot(&quadratic)).sqrt();
        contrast_matrix.column_mut(1).assign(&(quadratic / norm));
    }

    // Higher order polynomials using Gram-Schmidt orthogonalization
    for degree in 3..=max_deg {
        let mut poly: Array1<f64> = x_centered
            .iter()
            .map(|&xi| xi.powi(degree as i32))
            .collect();

        // Orthogonalize against previous polynomials
        for j in 0..degree - 1 {
            let prev = contrast_matrix.column(j);
            let proj = poly.dot(&prev) / prev.dot(&prev);
            poly = poly - proj * &prev;
        }

        let norm = (poly.dot(&poly)).sqrt();
        if norm > 1e-10 {
            contrast_matrix
                .column_mut(degree - 1)
                .assign(&(poly / norm));
        }
    }

    Ok(contrast_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Simple mock TermInfo for testing.
    struct MockTerm {
        name: String,
        conditions: Vec<String>,
    }

    impl MockTerm {
        fn new(name: &str, conditions: Vec<&str>) -> Self {
            Self {
                name: name.to_string(),
                conditions: conditions.into_iter().map(|s| s.to_string()).collect(),
            }
        }
    }

    impl TermInfo for MockTerm {
        fn conditions(&self, _expand_basis: bool) -> Vec<String> {
            self.conditions.clone()
        }

        fn term_name(&self) -> &str {
            &self.name
        }
    }

    #[test]
    fn test_kronecker_product() {
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[0.0, 5.0], [6.0, 7.0]];

        let result = kronecker_product(&a, &b);

        // Expected result:
        // [0, 5, 0, 10]
        // [6, 7, 12, 14]
        // [0, 15, 0, 20]
        // [18, 21, 24, 28]

        assert_eq!(result.dim(), (4, 4));
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[0, 1]], 5.0);
        assert_eq!(result[[0, 2]], 0.0);
        assert_eq!(result[[0, 3]], 10.0);
        assert_eq!(result[[1, 0]], 6.0);
        assert_eq!(result[[1, 1]], 7.0);
        assert_eq!(result[[3, 3]], 28.0);
    }

    #[test]
    fn test_polynomial_contrasts() {
        let levels = vec!["low".to_string(), "medium".to_string(), "high".to_string()];
        let contrasts = polynomial_contrasts_kronecker("factor", &levels, Some(2)).unwrap();

        assert_eq!(contrasts.dim(), (3, 2));

        // Check orthogonality
        let col0 = contrasts.column(0);
        let col1 = contrasts.column(1);
        let dot_product = col0.dot(&col1);
        assert!(dot_product.abs() < 1e-10);

        // Check normalization
        assert!((col0.dot(&col0) - 1.0).abs() < 1e-10);
        assert!((col1.dot(&col1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_fcontrast_spec_creation() {
        let sub1 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_B".to_string(),
            a: Formula::new("A"),
            b: Formula::new("B"),
            where_clause: None,
        });
        let sub2 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_C".to_string(),
            a: Formula::new("A"),
            b: Formula::new("C"),
            where_clause: None,
        });

        let f_spec = f_contrast_spec("omnibus", vec![sub1, sub2]);
        assert_eq!(f_spec.name, "omnibus");
        assert_eq!(f_spec.sub_contrasts.len(), 2);
        assert!(f_spec.where_clause.is_none());
    }

    #[test]
    fn test_fcontrast_spec_trait() {
        let f_spec = FContrastSpec {
            name: "my_f_test".to_string(),
            sub_contrasts: vec![],
            where_clause: None,
        };
        assert_eq!(ContrastSpec::name(&f_spec), "my_f_test");
        assert_eq!(f_spec.spec_type(), "f_contrast");
    }

    #[test]
    fn test_fcontrast_enum_variant() {
        let sub = ContrastSpecEnum::Unit(UnitContrastSpec {
            name: "mean".to_string(),
            where_clause: None,
        });
        let f_spec = FContrastSpec {
            name: "f_test".to_string(),
            sub_contrasts: vec![sub],
            where_clause: None,
        };
        let enum_val = ContrastSpecEnum::FContrast(f_spec);
        assert_eq!(enum_val.name(), "f_test");
        assert_eq!(enum_val.spec_type(), "f_contrast");
    }

    #[test]
    fn test_fcontrast_from_impl() {
        let f_spec = FContrastSpec {
            name: "test".to_string(),
            sub_contrasts: vec![],
            where_clause: None,
        };
        let enum_val: ContrastSpecEnum = f_spec.into();
        assert!(matches!(enum_val, ContrastSpecEnum::FContrast(_)));
    }

    #[test]
    fn test_fcontrast_weights_from_pair_sub_contrasts() {
        let term = MockTerm::new("condition", vec!["A", "B", "C"]);

        let sub1 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_B".to_string(),
            a: Formula::new("A"),
            b: Formula::new("B"),
            where_clause: None,
        });
        let sub2 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_C".to_string(),
            a: Formula::new("A"),
            b: Formula::new("C"),
            where_clause: None,
        });

        let f_spec = ContrastSpecEnum::FContrast(FContrastSpec {
            name: "omnibus".to_string(),
            sub_contrasts: vec![sub1, sub2],
            where_clause: None,
        });

        let result = contrast_weights(&f_spec, &term).unwrap();

        assert_eq!(result.name, "omnibus");
        assert_eq!(result.condnames.len(), 3);
        // Two sub-contrasts produce a matrix with 2 columns
        assert_eq!(result.weights.ncols(), 2);
        assert_eq!(result.weights.nrows(), 3);
        // This should be an F-contrast
        assert!(result.is_fcontrast());
        assert!(matches!(
            result.contrast_spec,
            ContrastSpecEnum::FContrast(_)
        ));
    }

    #[test]
    fn test_fcontrast_weights_single_sub_contrast() {
        let term = MockTerm::new("condition", vec!["A", "B"]);

        let sub = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_B".to_string(),
            a: Formula::new("A"),
            b: Formula::new("B"),
            where_clause: None,
        });

        let f_spec = ContrastSpecEnum::FContrast(FContrastSpec {
            name: "single_f".to_string(),
            sub_contrasts: vec![sub],
            where_clause: None,
        });

        let result = contrast_weights(&f_spec, &term).unwrap();
        assert_eq!(result.weights.ncols(), 1);
        assert_eq!(result.weights.nrows(), 2);
    }

    #[test]
    fn test_fcontrast_weights_empty_sub_contrasts() {
        let term = MockTerm::new("condition", vec!["A", "B"]);

        let f_spec = ContrastSpecEnum::FContrast(FContrastSpec {
            name: "empty_f".to_string(),
            sub_contrasts: vec![],
            where_clause: None,
        });

        let result = contrast_weights(&f_spec, &term);
        assert!(result.is_err());
    }

    #[test]
    fn test_fcontrasts_helper_uses_proper_spec() {
        let term = MockTerm::new("condition", vec!["A", "B", "C"]);

        let sub1 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_B".to_string(),
            a: Formula::new("A"),
            b: Formula::new("B"),
            where_clause: None,
        });
        let sub2 = ContrastSpecEnum::Pair(PairContrastSpec {
            name: "A_vs_C".to_string(),
            a: Formula::new("A"),
            b: Formula::new("C"),
            where_clause: None,
        });

        let result = fcontrasts(&[sub1, sub2], &term).unwrap();
        // The fcontrasts helper should now produce an FContrast spec
        assert!(matches!(
            result.contrast_spec,
            ContrastSpecEnum::FContrast(_)
        ));
        assert!(result.is_fcontrast());
    }

    #[test]
    fn test_into_contrast_spec_for_vec() {
        use crate::contrast::IntoContrastSpec;

        let sub1 = ContrastSpecEnum::Unit(UnitContrastSpec {
            name: "mean_A".to_string(),
            where_clause: None,
        });
        let sub2 = ContrastSpecEnum::Unit(UnitContrastSpec {
            name: "mean_B".to_string(),
            where_clause: None,
        });

        let specs = vec![sub1, sub2];
        let result = specs.into_contrast_spec("f_test".to_string()).unwrap();

        match result {
            ContrastSpecEnum::FContrast(f) => {
                assert_eq!(f.name, "f_test");
                assert_eq!(f.sub_contrasts.len(), 2);
            }
            _ => panic!("Expected FContrast variant"),
        }
    }

    #[test]
    fn test_into_contrast_spec_for_empty_vec() {
        use crate::contrast::IntoContrastSpec;

        let specs: Vec<ContrastSpecEnum> = vec![];
        let result = specs.into_contrast_spec("empty".to_string());
        assert!(result.is_err());
    }
}
