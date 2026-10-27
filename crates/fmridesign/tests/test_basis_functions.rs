//! Tests for basis functions

use approx::assert_relative_eq;
use fmridesign::basis::{BSplineBasis, IdentityBasis, PolynomialBasis};
use fmridesign::types::BasisFunction;
use ndarray::{array, Array1};

#[test]
fn test_polynomial_basis() {
    // Test raw polynomials (not orthogonal)
    let poly = PolynomialBasis::new(2).with_raw(true);
    let x = array![0.0, 1.0, 2.0, 3.0];

    let result = poly.evaluate(&x.view()).unwrap();

    // Default polynomial basis excludes the intercept.
    assert_eq!(result.shape(), &[4, 2]);

    // Check first column (x^1)
    assert_relative_eq!(result[[0, 0]], 0.0);
    assert_relative_eq!(result[[1, 0]], 1.0);
    assert_relative_eq!(result[[2, 0]], 2.0);
    assert_relative_eq!(result[[3, 0]], 3.0);

    // Check second column (x^2)
    assert_relative_eq!(result[[0, 1]], 0.0);
    assert_relative_eq!(result[[1, 1]], 1.0);
    assert_relative_eq!(result[[2, 1]], 4.0);
    assert_relative_eq!(result[[3, 1]], 9.0);
}

#[test]
fn test_polynomial_with_intercept() {
    let poly = PolynomialBasis::new(2).with_intercept(true).with_raw(true);
    let x = array![0.0, 1.0, 2.0];

    let result = poly.evaluate(&x.view()).unwrap();

    // With degree 2 and intercept, should have 3 columns (intercept, x, x^2)
    assert_eq!(result.shape(), &[3, 3]);

    // First column should be all ones (intercept)
    assert_relative_eq!(result[[0, 0]], 1.0);
    assert_relative_eq!(result[[1, 0]], 1.0);
    assert_relative_eq!(result[[2, 0]], 1.0);
}

#[test]
fn test_polynomial_names() {
    let poly = PolynomialBasis::new(3);
    // Polynomial of degree 3 defaults to x^1 through x^3.
    assert_eq!(poly.nbasis(), 3);
    assert_eq!(poly.name(), "poly3");
}

#[test]
fn test_bspline_basis() {
    let spline = BSplineBasis::new(4).with_intercept(true);
    let x = Array1::linspace(0.0, 1.0, 50);

    let result = spline.evaluate(&x.view()).unwrap();

    // Should have shape (50, 4)
    assert_eq!(result.shape(), &[50, 4]);

    // Check that values are non-negative
    for val in result.iter() {
        assert!(*val >= 0.0);
    }

    // Check that interior rows sum to approximately 1 (partition of unity)
    // B-splines don't necessarily sum to 1 at the boundaries
    for i in 10..40 {
        let row_sum: f64 = (0..4).map(|j| result[[i, j]]).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_bspline_names() {
    let spline = BSplineBasis::new(5);
    assert_eq!(spline.nbasis(), 5);
    assert_eq!(spline.name(), "bs5");
}

#[test]
fn test_identity_basis() {
    let identity = IdentityBasis::new();
    let x = array![1.0, 2.0, 3.0, 4.0];

    let result = identity.evaluate(&x.view()).unwrap();

    // Should return the input as a column matrix
    assert_eq!(result.shape(), &[4, 1]);
    assert_relative_eq!(result[[0, 0]], 1.0);
    assert_relative_eq!(result[[1, 0]], 2.0);
    assert_relative_eq!(result[[2, 0]], 3.0);
    assert_relative_eq!(result[[3, 0]], 4.0);
}

#[test]
fn test_polynomial_larger_degree() {
    let poly = PolynomialBasis::new(4);
    let x = array![0.0, 1.0, 2.0, 3.0, 4.0];

    let result = poly.evaluate(&x.view()).unwrap();
    // Default polynomial basis excludes the intercept.
    assert_eq!(result.shape(), &[5, 4]);
}

#[test]
fn test_zero_degree() {
    // The Rust implementation requires degree >= 1
    // This is different from Python which allows degree 0
    // Test that it returns an error
    let poly = PolynomialBasis::new(0);
    let x = array![1.0, 2.0, 3.0];
    let result = poly.evaluate(&x.view());

    assert!(result.is_err());
    if let Err(e) = result {
        match e {
            fmridesign::types::ValidationError::InvalidParameter(msg) => {
                assert!(msg.contains("degree must be at least 1"));
            }
            _ => panic!("Expected InvalidParameter error"),
        }
    }
}

#[test]
fn test_small_bspline_df() {
    // Test with minimum degrees of freedom
    let spline = BSplineBasis::new(3);
    assert_eq!(spline.nbasis(), 3);

    // Test with larger df
    let spline = BSplineBasis::new(5);
    assert_eq!(spline.nbasis(), 5);
}

#[test]
fn test_bspline_cox_de_boor() {
    // Test that Cox-de Boor implementation gives proper B-spline properties
    let spline = BSplineBasis::new(4).with_intercept(true);
    let x = Array1::linspace(0.0, 1.0, 100);

    let result = spline.evaluate(&x.view()).unwrap();

    // Check non-negativity
    for val in result.iter() {
        assert!(*val >= 0.0, "B-spline values should be non-negative");
    }

    // Check partition of unity in the interior
    for i in 20..80 {
        let row_sum: f64 = (0..4).map(|j| result[[i, j]]).sum();
        assert_relative_eq!(row_sum, 1.0, epsilon = 1e-10);
    }

    // Check smoothness - values should change gradually
    for j in 0..4 {
        for i in 1..99 {
            let diff = (result[[i, j]] - result[[i - 1, j]]).abs();
            assert!(diff < 0.1, "B-spline should be smooth");
        }
    }
}

#[test]
fn test_bspline_with_knots() {
    // Test B-spline with specified knots
    let knots = vec![0.25, 0.5, 0.75];
    let spline = BSplineBasis::with_knots(knots.clone()).with_intercept(true);

    // With 3 interior knots and degree 3 (cubic), we get 3 + 3 + 1 = 7 basis functions
    assert_eq!(spline.nbasis(), 7);

    let x = Array1::linspace(0.0, 1.0, 100);
    let result = spline.evaluate(&x.view()).unwrap();

    assert_eq!(result.shape(), &[100, 7]);
}

#[test]
fn test_bspline_boundary_behavior() {
    // Test behavior at boundaries
    let spline = BSplineBasis::new(4)
        .with_intercept(true)
        .with_boundary_knots(0.0, 1.0);
    let x = array![-0.1, 0.0, 0.5, 1.0, 1.1];

    let result = spline.evaluate(&x.view()).unwrap();

    // R continues the boundary cubic polynomial. With no interior knots,
    // these are the analytic Bernstein polynomials, including extrapolation.
    for (i, &value) in x.iter().enumerate() {
        let expected = [
            (1.0 - value).powi(3),
            3.0 * value * (1.0 - value).powi(2),
            3.0 * value.powi(2) * (1.0 - value),
            value.powi(3),
        ];
        for j in 0..4 {
            assert_relative_eq!(result[[i, j]], expected[j], epsilon = 1e-12);
        }
    }

    // Values at boundaries should be non-zero for at least one basis function
    let row0_sum: f64 = (0..4).map(|j| result[[1, j]]).sum();
    let row3_sum: f64 = (0..4).map(|j| result[[3, j]]).sum();

    // The issue might be that B-splines at the exact boundary might be zero
    // due to the half-open interval definition. Let's check values just inside.
    let x_inner = array![0.001, 0.5, 0.999];
    let result_inner = spline.evaluate(&x_inner.view()).unwrap();
    let row0_inner: f64 = (0..4).map(|j| result_inner[[0, j]]).sum();
    let row2_inner: f64 = (0..4).map(|j| result_inner[[2, j]]).sum();

    assert!(
        row0_inner > 0.0 || row0_sum > 0.0,
        "At least one basis should be non-zero near lower boundary"
    );
    assert!(
        row2_inner > 0.0 || row3_sum > 0.0,
        "At least one basis should be non-zero near upper boundary"
    );
}

#[test]
fn test_natural_spline_basis() {
    use fmridesign::basis::NaturalSplineBasis;

    let ns = NaturalSplineBasis::new(4);
    let x = Array1::linspace(0.0, 1.0, 50);

    let result = ns.evaluate(&x.view()).unwrap();

    // df specifies the number of returned columns after constraints.
    assert_eq!(result.shape(), &[50, 4]);

    // Check non-negativity is not guaranteed for natural splines
    // but values should be reasonable
    for val in result.iter() {
        assert!(
            val.abs() < 10.0,
            "Natural spline values should be reasonable"
        );
    }
}

#[test]
fn test_orthogonal_polynomials() {
    // Test orthogonal polynomials
    let poly = PolynomialBasis::new(3).with_raw(false); // orthogonal
    let x = Array1::linspace(-1.0, 1.0, 100);

    let result = poly.evaluate(&x.view()).unwrap();

    // Check orthogonality (columns should be approximately orthogonal)
    for i in 0..3 {
        for j in (i + 1)..3 {
            let dot_product: f64 = (0..100).map(|k| result[[k, i]] * result[[k, j]]).sum();
            assert_relative_eq!(dot_product, 0.0, epsilon = 1e-10);
        }
    }
}
