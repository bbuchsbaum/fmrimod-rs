//! Derivative methods for HRF objects.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use crate::types::{HrfValues, TimeArray};
use ndarray::{Array1, Array2};
use std::f64::consts::E;

/// Method for computing derivatives.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DerivativeMethod {
    /// Use analytic if available, otherwise numeric
    Auto,
    /// Force numerical differentiation
    Numeric,
    /// Force analytic (error if not available)
    Analytic,
}

/// Trait for HRFs that support analytic derivatives.
pub trait AnalyticDerivative: Hrf {
    /// Compute the analytic derivative at given time points.
    fn derivative(&self, t: &TimeArray) -> Result<HrfValues>;

    /// Check if this HRF has analytic derivative support.
    fn has_analytic_derivative(&self) -> bool {
        true
    }
}

/// Compute the derivative of an HRF at given time points.
///
/// # Arguments
/// * `hrf` - HRF to differentiate
/// * `t` - Time points at which to evaluate the derivative
/// * `method` - Derivative method
///
/// # Returns
/// Array of derivative values. Shape is (len(t), 1) for single basis,
/// or (len(t), nbasis) for multi-basis HRFs.
pub fn deriv<H: Hrf + ?Sized>(
    hrf: &H,
    t: &TimeArray,
    method: DerivativeMethod,
) -> Result<HrfValues> {
    if method != DerivativeMethod::Numeric {
        if let Some(result) = hrf.analytic_derivative(t) {
            return result;
        }
        if method == DerivativeMethod::Analytic {
            return Err(HrfError::InvalidInput(
                "analytic derivative unavailable for this HRF".into(),
            ));
        }
    }
    numeric_derivative(hrf, t)
}

/// Compute the derivative of an HRF with analytic support.
pub fn deriv_analytic<H: AnalyticDerivative>(hrf: &H, t: &TimeArray) -> Result<HrfValues> {
    hrf.derivative(t)
}

/// Compute derivative using numerical differentiation.
///
/// Uses central finite differences for interior points and
/// forward/backward differences at boundaries.
pub fn numeric_derivative<H: Hrf + ?Sized>(hrf: &H, t: &TimeArray) -> Result<HrfValues> {
    use crate::regressor::convolution::check_shape;
    if t.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::InvalidInput(
            "derivative times must be finite".into(),
        ));
    }
    let dx = 1e-4;
    let lo = t.mapv(|v| if (0.0..dx).contains(&v) { v } else { v - dx });
    let hi = t.mapv(|v| {
        if v <= hrf.span() && v + dx > hrf.span() {
            v
        } else {
            v + dx
        }
    });
    let a = hrf.evaluate(&lo)?;
    let mut b = hrf.evaluate(&hi)?;
    check_shape(&a, t.len(), hrf.nbasis())?;
    check_shape(&b, t.len(), hrf.nbasis())?;
    for i in 0..t.len() {
        for j in 0..hrf.nbasis() {
            b[(i, j)] = (b[(i, j)] - a[(i, j)]) / (hi[i] - lo[i]);
        }
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::NumericalError("nonfinite derivative".into()));
    }
    Ok(b)
}

/// An HRF wrapper that computes the nth numeric derivative of a base HRF.
///
/// Matches R's `makeDeriv(HRF, n)`. Uses central finite differences
/// applied iteratively for higher-order derivatives.
struct DerivativeHrf {
    base: Box<dyn Hrf>,
    n: usize,
    cached_name: String,
}

impl std::fmt::Debug for DerivativeHrf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivativeHrf")
            .field("base", &self.base.name())
            .field("n", &self.n)
            .finish()
    }
}

/// Compute the nth numeric derivative of an HRF at given time points.
///
/// For n=1, uses central finite differences directly on the base HRF.
/// For n>1, recursively applies differentiation by evaluating the (n-1)th
/// derivative at offset time points and computing finite differences.
fn nth_numeric_derivative(hrf: &dyn Hrf, t: &TimeArray, order: usize) -> Result<HrfValues> {
    if order == 0 {
        return hrf.evaluate(t);
    }

    let n_basis = hrf.nbasis();
    let n_times = t.len();
    let mut result = Array2::zeros((n_times, n_basis));

    let dx = 1e-4;

    for i in 0..n_times {
        let ti = t[i];

        let (t_minus, t_plus) = if ti > dx {
            (ti - dx, ti + dx)
        } else {
            (ti, ti + 2.0 * dx)
        };

        let t_vec_minus = Array1::from_elem(1, t_minus);
        let t_vec_plus = Array1::from_elem(1, t_plus);

        // Recursively get the (n-1)th derivative at the offset points
        let y_minus = nth_numeric_derivative(hrf, &t_vec_minus, order - 1)?;
        let y_plus = nth_numeric_derivative(hrf, &t_vec_plus, order - 1)?;

        let h = t_plus - t_minus;
        for j in 0..n_basis {
            result[[i, j]] = (y_plus[[0, j]] - y_minus[[0, j]]) / h;
        }
    }

    Ok(result)
}

impl Hrf for DerivativeHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        nth_numeric_derivative(&*self.base, t, self.n)
    }

    fn name(&self) -> &str {
        &self.cached_name
    }

    fn nbasis(&self) -> usize {
        self.base.nbasis()
    }

    fn span(&self) -> f64 {
        self.base.span()
    }

    fn params(&self) -> Vec<(&str, f64)> {
        let mut p = self.base.params();
        p.push(("deriv_order", self.n as f64));
        p
    }
}

/// Create a new HRF that evaluates as the nth numeric derivative of the base HRF.
///
/// This wraps an existing HRF, applying numeric differentiation when evaluated.
/// Matches R's `makeDeriv(HRF, n)`.
///
/// # Arguments
/// * `base_hrf` - The HRF to differentiate
/// * `n` - Order of differentiation (1 = first derivative, 2 = second, etc.)
///
/// # Examples
/// ```
/// use fmrihrf::hrf::{spm_canonical, derivative_hrf};
/// use ndarray::Array1;
///
/// let base = spm_canonical(None, None, None, None, None, None);
/// let deriv = derivative_hrf(base, 1);
/// assert_eq!(deriv.nbasis(), 1);
///
/// let t = Array1::linspace(0.0, 20.0, 201);
/// let result = deriv.evaluate(&t).unwrap();
/// assert_eq!(result.shape(), &[201, 1]);
/// ```
pub fn derivative_hrf(base_hrf: Box<dyn Hrf>, n: usize) -> Box<dyn Hrf> {
    let cached_name = format!("deriv{}({})", n, base_hrf.name());
    Box::new(DerivativeHrf {
        base: base_hrf,
        n,
        cached_name,
    })
}

/// Analytic derivative of SPM canonical HRF (SPMG1).
///
/// # Arguments
/// * `t` - Time points
/// * `p1` - Shape parameter for positive gamma (default: 6.0)
/// * `p2` - Shape parameter for negative gamma (default: 16.0)
/// * `a1` - Amplitude scaling factor (default: 1/6)
///
/// # Returns
/// Derivative values at time points t
pub fn spmg1_derivative(t: &TimeArray, p1: f64, p2: f64, a1: f64) -> Array1<f64> {
    let c = crate::hrf::functions::SPM_C;
    let mut result = Array1::zeros(t.len());

    for (i, &ti) in t.iter().enumerate() {
        if ti >= 0.0 {
            result[i] = E.powf(-ti)
                * (a1 * ti.powf(p1 - 1.0) * (p1 - ti) - c * ti.powf(p2 - 1.0) * (p2 - ti));
        }
    }

    result
}

/// Analytic second derivative of SPM canonical HRF.
///
/// # Arguments
/// * `t` - Time points
/// * `p1` - Shape parameter for positive gamma (default: 6.0)
/// * `p2` - Shape parameter for negative gamma (default: 16.0)
/// * `a1` - Amplitude scaling factor (default: 1/6)
///
/// # Returns
/// Second derivative values at time points t
pub fn spmg1_second_derivative(t: &TimeArray, p1: f64, p2: f64, a1: f64) -> Array1<f64> {
    let c = crate::hrf::functions::SPM_C;
    let mut result = Array1::zeros(t.len());

    for (i, &ti) in t.iter().enumerate() {
        if ti >= 0.0 {
            // Components and their derivatives
            let d1 = a1 * ti.powf(p1 - 1.0) * (p1 - ti);
            let d2 = c * ti.powf(p2 - 1.0) * (p2 - ti);
            let d1_prime = a1 * ((p1 - 1.0) * ti.powf(p1 - 2.0) * (p1 - ti) - ti.powf(p1 - 1.0));
            let d2_prime = c * ((p2 - 1.0) * ti.powf(p2 - 2.0) * (p2 - ti) - ti.powf(p2 - 1.0));

            result[i] = E.powf(-ti) * (d1_prime - d2_prime - (d1 - d2));
        }
    }

    result
}

/// SPM canonical HRF with analytic derivative support.
#[derive(Debug, Clone)]
pub struct SpmG1Hrf {
    p1: f64,
    p2: f64,
    a1: f64,
}

impl SpmG1Hrf {
    /// Create a new SPM canonical HRF with analytic derivatives.
    pub fn new(p1: f64, p2: f64, a1: f64) -> Self {
        Self { p1, p2, a1 }
    }
}

impl Hrf for SpmG1Hrf {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Spmg {
            columns: 1,
            p1: self.p1,
            p2: self.p2,
            a1: self.a1,
        })
    }
    fn analytic_derivative(&self, t: &TimeArray) -> Option<Result<HrfValues>> {
        Some(AnalyticDerivative::derivative(self, t))
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let canonical = crate::hrf::functions::evaluate_spm_canonical_r_formula(
            t,
            self.p1,
            self.p2,
            self.a1,
            crate::hrf::functions::SPM_C,
            false,
        )?;
        let mut result = Array2::zeros((t.len(), 1));
        result.column_mut(0).assign(&canonical.column(0));
        Ok(result)
    }

    fn name(&self) -> &str {
        "spmg1"
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn span(&self) -> f64 {
        24.0
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("p1", self.p1), ("p2", self.p2), ("a1", self.a1)]
    }
}

impl AnalyticDerivative for SpmG1Hrf {
    fn derivative(&self, t: &TimeArray) -> Result<HrfValues> {
        let deriv = spmg1_derivative(t, self.p1, self.p2, self.a1);
        Ok(deriv.insert_axis(ndarray::Axis(1)))
    }
}

/// SPM HRF with temporal derivative basis.
#[derive(Debug, Clone)]
pub struct SpmG2Hrf {
    p1: f64,
    p2: f64,
    a1: f64,
}

impl SpmG2Hrf {
    /// Create a new SPM HRF with temporal derivative.
    pub fn new(p1: f64, p2: f64, a1: f64) -> Self {
        Self { p1, p2, a1 }
    }
}

impl Hrf for SpmG2Hrf {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Spmg {
            columns: 2,
            p1: self.p1,
            p2: self.p2,
            a1: self.a1,
        })
    }
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::SpmDerivatives
    }
    fn analytic_derivative(&self, t: &TimeArray) -> Option<Result<HrfValues>> {
        Some(AnalyticDerivative::derivative(self, t))
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let canonical = crate::hrf::functions::evaluate_spm_canonical_r_formula(
            t,
            self.p1,
            self.p2,
            self.a1,
            crate::hrf::functions::SPM_C,
            false,
        )?;
        let mut result = Array2::zeros((t.len(), 2));
        result.column_mut(0).assign(&canonical.column(0));
        result
            .column_mut(1)
            .assign(&spmg1_derivative(t, self.p1, self.p2, self.a1));
        Ok(result)
    }

    fn name(&self) -> &str {
        "spmg2"
    }

    fn nbasis(&self) -> usize {
        2
    }

    fn span(&self) -> f64 {
        24.0
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("p1", self.p1), ("p2", self.p2), ("a1", self.a1)]
    }
}

impl AnalyticDerivative for SpmG2Hrf {
    fn derivative(&self, t: &TimeArray) -> Result<HrfValues> {
        // First basis derivative (derivative of canonical)
        let first_deriv = spmg1_derivative(t, self.p1, self.p2, self.a1);

        // Second basis derivative (second derivative of canonical)
        let second_deriv = spmg1_second_derivative(t, self.p1, self.p2, self.a1);

        // Stack as columns
        let mut result = Array2::zeros((t.len(), 2));
        result.slice_mut(ndarray::s![.., 0]).assign(&first_deriv);
        result.slice_mut(ndarray::s![.., 1]).assign(&second_deriv);

        Ok(result)
    }
}

/// SPM-sign finite dispersion difference at fixed positive-gamma mean and mass.
/// The 0.01 dispersion step matches current R fmrihrf; it is not a second time derivative.
pub fn spmg1_dispersion_derivative(t: &TimeArray, p1: f64, a1: f64) -> Array1<f64> {
    use libm::lgamma;
    let a = p1 + 1.0;
    let d = 1.01_f64;
    let mass = a1 * lgamma(a).exp();
    t.mapv(|x| {
        if x <= 0.0 {
            return 0.0;
        }
        let base = ((a - 1.0) * x.ln() - x - lgamma(a)).exp();
        let log_ratio = (a / d - a) * x.ln() + x * (1.0 - 1.0 / d) + lgamma(a)
            - lgamma(a / d)
            - (a / d) * d.ln();
        // expm1 avoids subtracting nearly equal gamma densities.
        -mass * base * log_ratio.exp_m1() / 0.01
    })
}

/// SPM HRF with temporal and dispersion derivatives.
#[derive(Debug, Clone)]
pub struct SpmG3Hrf {
    p1: f64,
    p2: f64,
    a1: f64,
}

impl SpmG3Hrf {
    /// Create a new SPM HRF with temporal and dispersion derivatives.
    pub fn new(p1: f64, p2: f64, a1: f64) -> Self {
        Self { p1, p2, a1 }
    }
}

impl Hrf for SpmG3Hrf {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Spmg {
            columns: 3,
            p1: self.p1,
            p2: self.p2,
            a1: self.a1,
        })
    }
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::SpmDerivatives
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let canonical = crate::hrf::functions::evaluate_spm_canonical_r_formula(
            t,
            self.p1,
            self.p2,
            self.a1,
            crate::hrf::functions::SPM_C,
            false,
        )?;
        let mut result = Array2::zeros((t.len(), 3));
        result.column_mut(0).assign(&canonical.column(0));
        result
            .column_mut(1)
            .assign(&spmg1_derivative(t, self.p1, self.p2, self.a1));
        result
            .column_mut(2)
            .assign(&spmg1_dispersion_derivative(t, self.p1, self.a1));
        Ok(result)
    }

    fn name(&self) -> &str {
        "spmg3"
    }

    fn nbasis(&self) -> usize {
        3
    }

    fn span(&self) -> f64 {
        24.0
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("p1", self.p1), ("p2", self.p2), ("a1", self.a1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_numeric_derivative() {
        // Test with a simple linear function
        #[derive(Debug)]
        struct LinearHrf;
        impl Hrf for LinearHrf {
            fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
                Ok(t.clone().insert_axis(ndarray::Axis(1)))
            }
            fn name(&self) -> &str {
                "linear"
            }
            fn nbasis(&self) -> usize {
                1
            }
            fn span(&self) -> f64 {
                10.0
            }
            fn params(&self) -> Vec<(&str, f64)> {
                vec![]
            }
        }

        let hrf = LinearHrf;
        let t = Array1::linspace(0.0, 10.0, 11);
        let deriv = numeric_derivative(&hrf, &t).unwrap();

        // Derivative of linear function should be constant (1.0)
        for i in 0..t.len() {
            assert_relative_eq!(deriv[[i, 0]], 1.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn test_spmg1_derivative() {
        let t = Array1::linspace(0.0, 30.0, 31);
        let deriv = spmg1_derivative(&t, 6.0, 16.0, 1.0 / 6.0);

        // Should be zero at t=0
        assert_relative_eq!(deriv[0], 0.0, epsilon = 1e-10);

        // Should cross zero (HRF peak)
        let mut found_zero_crossing = false;
        for i in 1..deriv.len() - 1 {
            if deriv[i] > 0.0 && deriv[i + 1] < 0.0 {
                found_zero_crossing = true;
                break;
            }
        }
        assert!(found_zero_crossing);
    }

    #[test]
    fn test_spmg1_hrf_with_derivative() {
        let hrf = SpmG1Hrf::new(6.0, 16.0, 1.0 / 6.0);
        let t = Array1::linspace(0.0, 20.0, 21);

        // Test evaluation
        let values = hrf.evaluate(&t).unwrap();
        assert_eq!(values.shape(), &[21, 1]);

        // Test analytic derivative
        let deriv = hrf.derivative(&t).unwrap();
        assert_eq!(deriv.shape(), &[21, 1]);
    }

    #[test]
    fn test_spmg2_hrf() {
        let hrf = SpmG2Hrf::new(6.0, 16.0, 1.0 / 6.0);
        let t = Array1::linspace(0.0, 20.0, 21);

        // Should have 2 basis functions
        assert_eq!(hrf.nbasis(), 2);

        let values = hrf.evaluate(&t).unwrap();
        assert_eq!(values.shape(), &[21, 2]);

        // Test derivatives
        let deriv = hrf.derivative(&t).unwrap();
        assert_eq!(deriv.shape(), &[21, 2]);
    }

    #[test]
    fn test_spmg3_hrf() {
        let hrf = SpmG3Hrf::new(6.0, 16.0, 1.0 / 6.0);
        let t = Array1::linspace(0.0, 20.0, 21);

        // Should have 3 basis functions
        assert_eq!(hrf.nbasis(), 3);

        let values = hrf.evaluate(&t).unwrap();
        assert_eq!(values.shape(), &[21, 3]);
    }

    #[test]
    fn test_derivative_hrf() {
        let base = crate::hrf::functions::gaussian_hrf(6.0, 2.0, Some(false));
        let deriv = derivative_hrf(base, 1);
        assert_eq!(deriv.name(), "deriv1(gaussian_hrf)");
        assert_eq!(deriv.nbasis(), 1);

        let t = Array1::linspace(0.0, 20.0, 201);
        let result = deriv.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[201, 1]);
        // Derivative should be positive early (rising) and negative later (falling)
        let has_pos = result.iter().any(|&v| v > 0.0);
        let has_neg = result.iter().any(|&v| v < 0.0);
        assert!(has_pos, "Should have positive derivative");
        assert!(has_neg, "Should have negative derivative");
    }

    #[test]
    fn test_derivative_hrf_second_order() {
        let base = crate::hrf::functions::gaussian_hrf(6.0, 2.0, Some(false));
        let deriv2 = derivative_hrf(base, 2);
        assert_eq!(deriv2.name(), "deriv2(gaussian_hrf)");
        let t = Array1::linspace(0.0, 20.0, 201);
        let result = deriv2.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[201, 1]);
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_derivative_hrf_zero_order() {
        // Zero-order derivative should be identical to base
        let base = crate::hrf::functions::gaussian_hrf(6.0, 2.0, Some(false));
        let t = Array1::linspace(0.0, 20.0, 201);
        let base_vals = base.evaluate(&t).unwrap();

        let deriv0 = derivative_hrf(
            crate::hrf::functions::gaussian_hrf(6.0, 2.0, Some(false)),
            0,
        );
        let deriv0_vals = deriv0.evaluate(&t).unwrap();
        for i in 0..201 {
            assert_relative_eq!(base_vals[[i, 0]], deriv0_vals[[i, 0]], epsilon = 1e-10);
        }
    }
}
