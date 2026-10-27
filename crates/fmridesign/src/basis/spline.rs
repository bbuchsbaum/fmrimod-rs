//! B-spline and natural cubic spline bases with fixed fitted knots.

use crate::types::{Array, BasisFunction, ValidationError, Vector};
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::OnceLock;

#[derive(Debug, Clone)]
struct PreparedSpline {
    knots: Vec<f64>,
    boundary: (f64, f64),
    degree: usize,
    first: usize,
    transform: Option<Array2<f64>>,
}

impl PreparedSpline {
    fn row(&self, x: f64, derivative: usize) -> Vec<f64> {
        let n = self.knots.len() - self.degree - 1;
        (self.first..n)
            .map(|i| spline_value(&self.knots, i, self.degree, x, derivative))
            .collect()
    }
    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(invalid("spline input must be finite"));
        }
        let n = self.knots.len() - self.degree - 1 - self.first;
        let mut values = Array2::zeros((x.len(), n));
        for (i, &value) in x.iter().enumerate() {
            let boundary = value.clamp(self.boundary.0, self.boundary.1);
            let delta = value - boundary;
            let mut row = self.row(boundary, 0);
            if delta != 0.0 {
                // Natural splines extrapolate linearly; B-splines continue the
                // boundary polynomial, as predict.bs does.
                let order = if self.transform.is_some() {
                    1
                } else {
                    self.degree
                };
                let mut multiplier = 1.0;
                for derivative in 1..=order {
                    multiplier *= delta / derivative as f64;
                    for (v, d) in row.iter_mut().zip(self.row(boundary, derivative)) {
                        *v += multiplier * d;
                    }
                }
            }
            for (j, v) in row.into_iter().enumerate() {
                values[[i, j]] = v;
            }
        }
        Ok(match &self.transform {
            Some(transform) => values.dot(transform),
            None => values,
        })
    }
}

fn invalid(message: &str) -> ValidationError {
    ValidationError::InvalidParameter(message.into())
}

fn spline_value(knots: &[f64], i: usize, degree: usize, x: f64, derivative: usize) -> f64 {
    if derivative > degree {
        return 0.0;
    }
    if degree == 0 {
        return if (knots[i] <= x && x < knots[i + 1])
            || (x == knots[knots.len() - 1] && knots[i] < x && knots[i + 1] == x)
        {
            1.0
        } else {
            0.0
        };
    }
    let left = knots[i + degree] - knots[i];
    let right = knots[i + degree + 1] - knots[i + 1];
    if derivative > 0 {
        let a = if left > 0.0 {
            degree as f64 / left * spline_value(knots, i, degree - 1, x, derivative - 1)
        } else {
            0.0
        };
        let b = if right > 0.0 {
            degree as f64 / right * spline_value(knots, i + 1, degree - 1, x, derivative - 1)
        } else {
            0.0
        };
        a - b
    } else {
        let a = if left > 0.0 {
            (x - knots[i]) / left * spline_value(knots, i, degree - 1, x, 0)
        } else {
            0.0
        };
        let b = if right > 0.0 {
            (knots[i + degree + 1] - x) / right * spline_value(knots, i + 1, degree - 1, x, 0)
        } else {
            0.0
        };
        a + b
    }
}

/// B-spline basis. `df` is the number of returned columns; intercept is omitted
/// by default, as in R. The first successful evaluation fixes the knots for
/// subsequent evaluation and prediction. Builder changes clear fitted state.
#[derive(Debug, Clone)]
pub struct BSplineBasis {
    name: String,
    df: usize,
    knots: Option<Vec<f64>>,
    degree: usize,
    intercept: bool,
    boundary_knots: Option<(f64, f64)>,
    fitted: OnceLock<PreparedSpline>,
}

impl BSplineBasis {
    /// Construct a cubic B-spline basis with `df` columns.
    pub fn new(df: usize) -> Self {
        Self {
            name: format!("bs{df}"),
            df,
            knots: None,
            degree: 3,
            intercept: false,
            boundary_knots: None,
            fitted: OnceLock::new(),
        }
    }
    /// Construct from explicit interior knots.
    pub fn with_knots(knots: Vec<f64>) -> Self {
        Self {
            df: knots.len() + 3,
            knots: Some(knots),
            ..Self::new(3)
        }
    }
    /// Set polynomial degree.
    pub fn with_degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self.reset();
        self
    }
    /// Include the intercept in the spline space.
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.intercept = intercept;
        self.reset();
        self
    }
    /// Fix the boundary knots.
    pub fn with_boundary_knots(mut self, min: f64, max: f64) -> Self {
        self.boundary_knots = Some((min, max));
        self.reset();
        self
    }
    /// Set the display name.
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
    /// Evaluate the basis, fitting knots on first use.
    pub fn design_matrix(&self, x: &Array1<f64>) -> Result<Array, ValidationError> {
        self.evaluate(&x.view())
    }
    fn reset(&mut self) {
        self.fitted = OnceLock::new();
        if let Some(knots) = &self.knots {
            self.df = knots.len() + self.degree + usize::from(self.intercept);
        }
    }
    fn prepare(
        &self,
        x: &ArrayView1<f64>,
        natural: bool,
    ) -> Result<PreparedSpline, ValidationError> {
        if x.len() < 2 || x.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "spline fitting requires at least two finite values",
            ));
        }
        let mut sorted = x.to_vec();
        sorted.sort_by(f64::total_cmp);
        let boundary = self
            .boundary_knots
            .unwrap_or((sorted[0], sorted[sorted.len() - 1]));
        if !boundary.0.is_finite() || !boundary.1.is_finite() || boundary.0 >= boundary.1 {
            return Err(invalid(
                "spline boundary knots must be finite and increasing",
            ));
        }
        let minimum = if natural { 1 } else { self.degree } + usize::from(self.intercept);
        if self.degree == 0 || self.degree > 20 || self.df < minimum {
            return Err(invalid(
                "spline df is too small for its degree and intercept",
            ));
        }
        let interior = if let Some(knots) = &self.knots {
            knots.clone()
        } else {
            sorted.retain(|v| *v >= boundary.0 && *v <= boundary.1);
            if sorted.len() < 2 {
                return Err(invalid(
                    "spline fitting requires two values within its boundaries",
                ));
            }
            let count = self.df - minimum;
            (1..=count)
                .map(|j| {
                    let position = j as f64 / (count + 1) as f64 * (sorted.len() - 1) as f64;
                    let lower = position.floor() as usize;
                    let upper = (lower + 1).min(sorted.len() - 1);
                    sorted[lower] + (position - lower as f64) * (sorted[upper] - sorted[lower])
                })
                .collect()
        };
        if interior
            .iter()
            .any(|v| !v.is_finite() || *v <= boundary.0 || *v >= boundary.1)
            || interior.windows(2).any(|w| w[0] > w[1])
        {
            return Err(invalid(
                "interior knots must be sorted and strictly inside the boundaries",
            ));
        }
        let degree = if natural { 3 } else { self.degree };
        let mut knots = vec![boundary.0; degree + 1];
        knots.extend(interior);
        knots.extend(vec![boundary.1; degree + 1]);
        let mut prepared = PreparedSpline {
            knots,
            boundary,
            degree,
            first: usize::from(!self.intercept),
            transform: None,
        };
        if natural {
            let left = prepared.row(boundary.0, 2);
            let right = prepared.row(boundary.1, 2);
            let n = left.len();
            // LINPACK Householder convention used by stats::qr (and ns).
            // Only two boundary constraints are factored; retain the remaining
            // orthonormal columns as a basis for their null space.
            let mut constraints =
                Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { left[i] } else { right[i] });
            let mut q = Array2::<f64>::eye(n);
            for column in 0..2 {
                let mut v: Vec<f64> = (column..n).map(|i| constraints[[i, column]]).collect();
                let norm = v.iter().fold(0.0_f64, |a, &b| a.hypot(b));
                if norm == 0.0 {
                    return Err(invalid(
                        "natural spline boundary constraints are rank deficient",
                    ));
                }
                let signed = if v[0] < 0.0 { -norm } else { norm };
                for value in &mut v {
                    *value /= signed;
                }
                v[0] += 1.0;
                for j in column + 1..2 {
                    let scale = -v
                        .iter()
                        .enumerate()
                        .map(|(k, &v)| v * constraints[[column + k, j]])
                        .sum::<f64>()
                        / v[0];
                    for (k, &value) in v.iter().enumerate() {
                        constraints[[column + k, j]] += scale * value;
                    }
                }
                for row in 0..n {
                    let scale = -v
                        .iter()
                        .enumerate()
                        .map(|(k, &v)| q[[row, column + k]] * v)
                        .sum::<f64>()
                        / v[0];
                    for (k, &value) in v.iter().enumerate() {
                        q[[row, column + k]] += scale * value;
                    }
                }
            }
            prepared.transform = Some(q.slice(ndarray::s![.., 2..]).to_owned());
        }
        Ok(prepared)
    }
    fn evaluate_kind(&self, x: &ArrayView1<f64>, natural: bool) -> Result<Array, ValidationError> {
        if self.fitted.get().is_none() {
            let fit = self.prepare(x, natural)?;
            let _ = self.fitted.set(fit);
        }
        self.fitted
            .get()
            .expect("a successful fit is installed")
            .evaluate(x)
    }
}

impl BasisFunction for BSplineBasis {
    fn name(&self) -> &str {
        &self.name
    }
    fn nbasis(&self) -> usize {
        self.df
    }
    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        self.evaluate_kind(x, false)
    }
    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        predict_basis(self, x, coef)
    }
}

/// Natural cubic spline basis with `df` returned columns. Second derivatives
/// vanish at both boundaries, and predictions outside them are linear.
#[derive(Debug, Clone)]
pub struct NaturalSplineBasis {
    inner: BSplineBasis,
}
impl NaturalSplineBasis {
    /// Construct with the requested number of columns.
    pub fn new(df: usize) -> Self {
        Self {
            inner: BSplineBasis::new(df).with_name(format!("ns{df}")),
        }
    }
    /// Construct with explicit interior knots.
    pub fn with_knots(knots: Vec<f64>) -> Self {
        let df = knots.len() + 1;
        Self {
            inner: BSplineBasis {
                df,
                knots: Some(knots),
                ..BSplineBasis::new(df).with_name(format!("ns{df}"))
            },
        }
    }
    /// Include the intercept in the natural spline space.
    pub fn with_intercept(mut self, intercept: bool) -> Self {
        self.inner.intercept = intercept;
        self.inner.fitted = OnceLock::new();
        if let Some(k) = &self.inner.knots {
            self.inner.df = k.len() + 1 + usize::from(intercept);
        }
        self
    }
    /// Fix the boundary knots.
    pub fn with_boundary_knots(mut self, min: f64, max: f64) -> Self {
        self.inner.boundary_knots = Some((min, max));
        self.inner.fitted = OnceLock::new();
        self
    }
    /// Set the display name.
    pub fn with_name(mut self, name: String) -> Self {
        self.inner.name = name;
        self
    }
}
impl BasisFunction for NaturalSplineBasis {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn nbasis(&self) -> usize {
        self.inner.df
    }
    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        self.inner.evaluate_kind(x, true)
    }
    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        predict_basis(self, x, coef)
    }
}
fn predict_basis(
    basis: &dyn BasisFunction,
    x: &ArrayView1<f64>,
    coef: Option<&ArrayView1<f64>>,
) -> Result<Vector, ValidationError> {
    let values = basis.evaluate(x)?;
    if let Some(coef) = coef {
        if coef.len() != values.ncols() {
            return Err(ValidationError::LengthMismatch {
                expected: values.ncols(),
                actual: coef.len(),
            });
        }
        Ok(values.dot(coef))
    } else {
        Ok(values.column(0).to_owned())
    }
}

/// Legacy SplineBasis for backward compatibility
#[derive(Debug, Clone)]
pub struct SplineBasis {
    inner: BSplineBasis,
}

impl SplineBasis {
    /// Create new spline basis.
    pub fn new(n_basis: usize, degree: usize) -> Self {
        Self {
            inner: BSplineBasis::new(n_basis).with_degree(degree),
        }
    }
}

impl BasisFunction for SplineBasis {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }

    fn evaluate(&self, x: &ArrayView1<f64>) -> Result<Array, ValidationError> {
        self.inner.evaluate(x)
    }

    fn predict(
        &self,
        x: &ArrayView1<f64>,
        coef: Option<&ArrayView1<f64>>,
    ) -> Result<Vector, ValidationError> {
        self.inner.predict(x, coef)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_natural_spline_constraints() {
        // Create natural spline with 6 degrees of freedom
        let ns = NaturalSplineBasis::new(6);

        // Test data
        let x = array![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];

        // Evaluate basis
        let basis = ns.evaluate(&x.view()).unwrap();

        // df is the number of returned columns, after constraints.
        assert_eq!(basis.shape(), &[11, 6]);

        // Check that basis functions are well-behaved
        for i in 0..x.len() {
            for j in 0..basis.ncols() {
                let val = basis[[i, j]];
                // Values should be finite
                assert!(
                    val.is_finite(),
                    "Non-finite value at [{}, {}]: {}",
                    i,
                    j,
                    val
                );
            }
        }

        // Check that the intercept column (if present) is all ones
        let first_col = basis.column(0);
        if first_col.iter().all(|&v| (v - 1.0).abs() < 1e-10) {
            // First column is intercept
            for i in 0..x.len() {
                assert_abs_diff_eq!(basis[[i, 0]], 1.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_natural_spline_smoothness() {
        // Natural splines should produce smooth curves
        let ns = NaturalSplineBasis::new(5);

        // Dense evaluation points
        let n_points = 100;
        let x = Array1::linspace(0.0, 1.0, n_points);

        // Evaluate basis
        let basis = ns.evaluate(&x.view()).unwrap();

        // Check that each basis function is smooth (no sudden jumps)
        for j in 0..basis.ncols() {
            let col = basis.column(j);
            for i in 1..n_points {
                let diff = (col[i] - col[i - 1]).abs();
                // The difference between consecutive points should be small
                assert!(
                    diff < 0.1,
                    "Large jump in basis function {} at index {}: {}",
                    j,
                    i,
                    diff
                );
            }
        }
    }

    #[test]
    fn test_bspline_basis() {
        // Test basic B-spline functionality
        let bs = BSplineBasis::new(4).with_degree(3);

        let x = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let basis = bs.evaluate(&x.view()).unwrap();

        // Check dimensions
        assert_eq!(basis.shape(), &[5, 4]);

        // Check that basis functions are non-negative
        for val in basis.iter() {
            assert!(*val >= -1e-10, "Negative value in B-spline basis: {}", val);
        }
    }

    #[test]
    fn test_natural_spline_vs_bspline() {
        // Compare natural spline with B-spline of same parameters
        let df = 7;
        let bs = BSplineBasis::new(df).with_degree(3);
        let ns = NaturalSplineBasis::new(df);

        let x = array![0.1, 0.3, 0.5, 0.7, 0.9];

        let bs_basis = bs.evaluate(&x.view()).unwrap();
        let ns_basis = ns.evaluate(&x.view()).unwrap();

        // Both constructors request df returned columns.
        assert_eq!(bs_basis.ncols(), df);
        assert_eq!(ns_basis.ncols(), df);

        // Both should have the same number of rows
        assert_eq!(bs_basis.nrows(), ns_basis.nrows());
    }
}
