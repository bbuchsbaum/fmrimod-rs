//! Core HRF trait and implementations.

use crate::error::{HrfError, Result};
use crate::types::{HrfValues, TimeArray};
use ndarray::{Array1, Array2};
use std::fmt::Debug;

/// Base trait for Hemodynamic Response Functions.
///
/// An HRF represents a hemodynamic response function that can be evaluated
/// at arbitrary time points. HRFs can have one or more basis functions.
pub trait Hrf: Debug + Send + Sync {
    /// Portable construction data, or None for unrepresented custom implementations.
    fn recipe(&self) -> Option<super::HrfRecipe> {
        None
    }
    /// Scientific penalty policy; display names never determine regularization.
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::Identity
    }

    /// Optional analytic derivative available through trait objects.
    fn analytic_derivative(&self, _t: &TimeArray) -> Option<Result<HrfValues>> {
        None
    }

    /// Optional process-unique identity for a time-invariant HRF. The default
    /// disables shared caching: a name and numeric parameters cannot identify
    /// arbitrary implementations or captured empirical data.
    fn cache_identity(&self) -> Option<u64> {
        None
    }

    /// Evaluate the HRF at time points t.
    ///
    /// # Arguments
    /// * `t` - Time points at which to evaluate the HRF
    ///
    /// # Returns
    /// Array of HRF values. Shape is (n_time_points, n_basis) where n_basis >= 1.
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues>;

    /// Get the name of the HRF.
    fn name(&self) -> &str;

    /// Get the number of basis functions.
    fn nbasis(&self) -> usize;

    /// Get the temporal span in seconds.
    fn span(&self) -> f64;

    /// Get a copy of the HRF parameters.
    fn params(&self) -> Vec<(&str, f64)>;
}

/// Extension trait providing additional HRF functionality.
pub trait HrfExt: Hrf {
    /// Evaluate HRF with optional block duration.
    ///
    /// This method extends basic HRF evaluation to handle sustained stimuli
    /// by convolving the HRF with a boxcar function of specified duration.
    ///
    /// # Arguments
    /// * `grid` - Time points at which to evaluate
    /// * `duration` - Duration of block/sustained stimulus in seconds
    /// * `precision` - Temporal precision for convolution in seconds
    /// * `summate` - If true, integrate block responses; if false, divide by duration
    /// * `normalize` - If true, normalize result to unit peak
    fn evaluate_block(
        &self,
        grid: &TimeArray,
        duration: f64,
        precision: f64,
        summate: bool,
        normalize: bool,
    ) -> Result<HrfValues> {
        use crate::regressor::convolution::{check_shape, grid_length};
        if grid.is_empty() || grid.iter().any(|t| !t.is_finite()) {
            return Err(HrfError::InvalidInput(
                "grid must be nonempty and finite".into(),
            ));
        }
        if !duration.is_finite() || duration < 0.0 || !precision.is_finite() || precision <= 0.0 {
            return Err(HrfError::InvalidInput(
                "duration must be nonnegative and finite; precision must be positive and finite"
                    .into(),
            ));
        }
        let mut result = if duration == 0.0 {
            self.evaluate(grid)?
        } else {
            let steps = grid_length(duration, precision, 1)?;
            let mut result = Array2::zeros((grid.len(), self.nbasis()));
            let mut left = 0.0;
            let mut previous = self.evaluate(grid)?;
            check_shape(&previous, grid.len(), self.nbasis())?;
            // Include the exact endpoint, even for blocks shorter than one step.
            for i in 1..=steps {
                let right = (i as f64 * precision).min(duration);
                if right <= left {
                    break;
                }
                let shifted = grid.mapv(|t| t - right);
                let next = self.evaluate(&shifted)?;
                check_shape(&next, grid.len(), self.nbasis())?;
                let weight = (right - left) / 2.0;
                ndarray::Zip::from(&mut result)
                    .and(&previous)
                    .and(&next)
                    .for_each(|out, &a, &b| *out += weight * (a + b));
                previous = next;
                left = right;
            }
            if !summate {
                result /= duration;
            }
            result
        };
        check_shape(&result, grid.len(), self.nbasis())?;
        if normalize {
            for mut column in result.columns_mut() {
                let peak = column.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                if peak > 0.0 {
                    column /= peak;
                }
            }
        }
        Ok(result)
    }

    /// Evaluate the derivative of the HRF at time points t.
    ///
    /// This method computes the derivative (gradient) of the HRF with respect to time.
    /// For multi-basis HRFs, it returns the derivative of each basis function.
    ///
    /// # Arguments
    /// * `t` - Time points at which to evaluate the derivative
    /// * `method` - Derivative method (Auto, Numeric, or Analytic)
    ///
    /// # Returns
    /// Array of derivative values. Shape is (n_time_points, n_basis).
    ///
    /// # Example
    /// ```
    /// use fmri_hrf::hrf::{spm_canonical, HrfExt};
    /// use fmri_hrf::hrf::derivatives::DerivativeMethod;
    /// use ndarray::Array1;
    ///
    /// let hrf = spm_canonical(None, None, None, None, None, Some(true));
    /// let t = Array1::linspace(0.0, 20.0, 201);
    /// let deriv = hrf.evaluate_deriv(&t, DerivativeMethod::Auto).unwrap();
    /// ```
    fn evaluate_deriv(
        &self,
        t: &TimeArray,
        method: crate::hrf::derivatives::DerivativeMethod,
    ) -> Result<HrfValues> {
        crate::hrf::derivatives::deriv(self, t, method)
    }
}

// Blanket implementation of HrfExt for all types implementing Hrf
impl<T: Hrf + ?Sized> HrfExt for T {}

/// Function pointer used by an immutable parameterized HRF.
pub type HrfEvaluator = fn(&TimeArray, &[(String, f64)]) -> Result<HrfValues>;

/// A simple struct-based HRF implementation for basic HRFs.
#[derive(Debug, Clone)]
pub struct SimpleHrf {
    cache_identity: u64,
    name: String,
    nbasis: usize,
    span: f64,
    params: Vec<(String, f64)>,
    eval_fn: HrfEvaluator,
    recipe: Option<super::HrfRecipe>,
}

impl SimpleHrf {
    pub(crate) fn with_recipe(mut self, recipe: super::HrfRecipe) -> Self {
        self.recipe = Some(recipe);
        self
    }

    /// Create a new SimpleHrf.
    pub fn new(
        name: impl Into<String>,
        nbasis: usize,
        span: f64,
        params: Vec<(String, f64)>,
        eval_fn: HrfEvaluator,
    ) -> Self {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self {
            cache_identity: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            name: name.into(),
            nbasis,
            span,
            params,
            eval_fn,
            recipe: None,
        }
    }
}

impl Hrf for SimpleHrf {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        self.recipe.clone()
    }
    fn cache_identity(&self) -> Option<u64> {
        Some(self.cache_identity)
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        (self.eval_fn)(t, &self.params)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        self.nbasis
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        self.params.iter().map(|(k, v)| (k.as_str(), *v)).collect()
    }
}

/// Create a new single-basis HRF as a weighted combination of basis functions.
///
/// This function takes an HRF with `nbasis` basis functions and a vector of
/// coefficients, and returns a new HRF with a single basis function that
/// represents the weighted sum of the original basis functions.
///
/// # Arguments
/// * `hrf` - Base HRF with one or more basis functions
/// * `coefficients` - Weights for each basis function (must have length == nbasis())
///
/// # Returns
/// A new HRF with nbasis=1 that evaluates as:
/// `h[0] * basis_0(t) + h[1] * basis_1(t) + ... + h[n-1] * basis_{n-1}(t)`
///
/// # Errors
/// Returns error if coefficient length doesn't match number of basis functions.
///
/// # Example
/// ```ignore
/// use fmri_hrf::hrf::{spm_with_derivatives, hrf_from_coefficients};
///
/// // Create SPM canonical with derivatives (3 basis functions)
/// let hrf = spm_with_derivatives(None, None, None, None, None, None);
///
/// // Create custom HRF with specific weights for each derivative
/// let coefficients = vec![1.0, 0.5, -0.2];  // canonical + 50% temporal + -20% dispersion
/// let custom_hrf = hrf_from_coefficients(hrf, &coefficients)?;
///
/// // custom_hrf now has nbasis=1 and represents the weighted combination
/// ```
pub fn hrf_from_coefficients(hrf: Box<dyn Hrf>, coefficients: &[f64]) -> Result<Box<dyn Hrf>> {
    // Validate coefficient length
    if coefficients.len() != hrf.nbasis() {
        return Err(HrfError::InvalidInput(format!(
            "coefficient length ({}) must equal nbasis ({})",
            coefficients.len(),
            hrf.nbasis()
        )));
    }

    let name = format!("{}_from_coef", hrf.name());
    let span = hrf.span();

    Ok(Box::new(WeightedHrf {
        base_hrf: hrf,
        coefficients: coefficients.to_vec(),
        name,
        span,
    }))
}

/// An HRF that represents a weighted combination of basis functions from another HRF.
///
/// This struct is created by `hrf_from_coefficients` and evaluates as the weighted
/// sum of the base HRF's basis functions.
#[derive(Debug)]
pub struct WeightedHrf {
    base_hrf: Box<dyn Hrf>,
    coefficients: Vec<f64>,
    name: String,
    span: f64,
}

impl Hrf for WeightedHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        // Evaluate the base HRF to get all basis functions
        let basis_values = self.base_hrf.evaluate(t)?;

        // Check dimensions
        if basis_values.ncols() != self.coefficients.len() {
            return Err(HrfError::InvalidInput(format!(
                "basis function count mismatch: expected {}, got {}",
                self.coefficients.len(),
                basis_values.ncols()
            )));
        }

        // Compute weighted sum: result(t) = sum_i coef_i * basis_i(t)
        let mut result = Array1::<f64>::zeros(t.len());
        for (i, &coef) in self.coefficients.iter().enumerate() {
            let basis_col = basis_values.column(i);
            for (j, &val) in basis_col.iter().enumerate() {
                result[j] += coef * val;
            }
        }

        // Return as 2D array with single basis function
        Ok(result.insert_axis(ndarray::Axis(1)))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1 // Always single basis after weighting
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        // Get base params and add coefficient info
        let mut params = self.base_hrf.params();
        params.push(("n_coefficients", self.coefficients.len() as f64));
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Axis;

    // Test HRF implementation
    #[derive(Debug, Clone)]
    struct TestHrf;

    impl Hrf for TestHrf {
        fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
            // Simple triangular HRF peaking at t=5
            let values = t.mapv(|t| {
                if t <= 5.0 {
                    t / 5.0
                } else if t <= 10.0 {
                    2.0 - t / 5.0
                } else {
                    0.0
                }
            });
            Ok(values.insert_axis(Axis(1)))
        }

        fn name(&self) -> &str {
            "test_hrf"
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

    #[test]
    fn test_basic_evaluation() {
        let hrf = TestHrf;
        let t = Array1::linspace(0.0, 10.0, 11);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[11, 1]);
        assert_relative_eq!(result[[5, 0]], 1.0); // Peak at t=5
        assert_relative_eq!(result[[0, 0]], 0.0); // Start at 0
        assert_relative_eq!(result[[10, 0]], 0.0); // End at 0
    }

    #[test]
    fn test_block_evaluation() {
        let hrf = TestHrf;
        let t = Array1::linspace(0.0, 15.0, 16);
        let result = hrf.evaluate_block(&t, 2.0, 0.5, true, false).unwrap();

        // Block convolution should spread out the response
        assert_eq!(result.shape(), &[16, 1]);
        // Peak should be shifted and spread
        let max_val = result.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val > 0.0);
    }

    // Multi-basis HRF for testing
    #[derive(Debug, Clone)]
    struct MultiBasisHrf;

    impl Hrf for MultiBasisHrf {
        fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
            let n = t.len();
            let mut result = Array2::zeros((n, 3));

            // Three basis functions: main, derivative, dispersion
            for (i, &time) in t.iter().enumerate() {
                if time >= 0.0 && time <= 20.0 {
                    // Main response
                    result[[i, 0]] = (time / 5.0).exp() * (-time / 5.0).exp();
                    // Temporal derivative
                    result[[i, 1]] = if time > 0.0 {
                        result[[i, 0]] * (1.0 - time / 5.0)
                    } else {
                        0.0
                    };
                    // Dispersion derivative
                    result[[i, 2]] = result[[i, 0]] * ((time - 5.0) / 10.0);
                }
            }

            Ok(result)
        }

        fn name(&self) -> &str {
            "multi_basis_hrf"
        }

        fn nbasis(&self) -> usize {
            3
        }

        fn span(&self) -> f64 {
            20.0
        }

        fn params(&self) -> Vec<(&str, f64)> {
            vec![("n_basis", 3.0), ("span", 20.0)]
        }
    }

    #[test]
    fn test_multi_basis_hrf() {
        let hrf = MultiBasisHrf;
        let t = Array1::linspace(0.0, 20.0, 101);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[101, 3]);
        assert_eq!(hrf.nbasis(), 3);
        assert_eq!(hrf.span(), 20.0);
        assert_eq!(hrf.name(), "multi_basis_hrf");

        // Check params
        let params = hrf.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0], ("n_basis", 3.0));
        assert_eq!(params[1], ("span", 20.0));

        // Each basis should have some non-zero values
        for col in 0..3 {
            let col_sum: f64 = result.column(col).sum();
            assert!(
                col_sum.abs() > 0.0,
                "Basis {} should have non-zero values",
                col
            );
        }
    }

    #[test]
    fn test_block_evaluation_multi_basis() {
        let hrf = MultiBasisHrf;
        let t = Array1::linspace(0.0, 30.0, 150);
        let result = hrf.evaluate_block(&t, 5.0, 0.2, true, false).unwrap();

        assert_eq!(result.shape(), &[150, 3]);

        // Block convolution should produce wider response
        let main_basis = result.column(0);
        let above_threshold: Vec<usize> = main_basis
            .iter()
            .enumerate()
            .filter(|(_, &val)| val > 0.1)
            .map(|(idx, _)| idx)
            .collect();

        // Should have extended response due to block
        assert!(above_threshold.len() > 20);
    }

    #[test]
    fn test_simple_hrf_implementation() {
        // Test the SimpleHrf implementation
        let params = vec![("param1".to_string(), 1.0), ("param2".to_string(), 2.0)];

        let eval_fn = |t: &TimeArray, _params: &[(String, f64)]| -> Result<HrfValues> {
            let values = t.mapv(|x| (-x).exp());
            Ok(values.insert_axis(Axis(1)))
        };

        let simple_hrf = SimpleHrf::new("exponential", 1, 10.0, params.clone(), eval_fn);

        assert_eq!(simple_hrf.name(), "exponential");
        assert_eq!(simple_hrf.nbasis(), 1);
        assert_eq!(simple_hrf.span(), 10.0);

        let hrf_params = simple_hrf.params();
        assert_eq!(hrf_params.len(), 2);
        assert_eq!(hrf_params[0], ("param1", 1.0));
        assert_eq!(hrf_params[1], ("param2", 2.0));

        let t = Array1::linspace(0.0, 5.0, 50);
        let result = simple_hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[50, 1]);

        // Check exponential decay
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-6);
        assert!(result[[49, 0]] < 0.1);
    }

    #[test]
    fn test_hrf_edge_cases() {
        let hrf = TestHrf;

        // Test with negative times
        let t = Array1::linspace(-5.0, 5.0, 11);
        let result = hrf.evaluate(&t).unwrap();

        // Should handle negative times gracefully
        assert_eq!(result.shape(), &[11, 1]);

        // Test with single time point
        let t_single = Array1::from_vec(vec![5.0]);
        let result_single = hrf.evaluate(&t_single).unwrap();
        assert_eq!(result_single.shape(), &[1, 1]);
        assert_relative_eq!(result_single[[0, 0]], 1.0);

        // Test with empty array
        let t_empty = Array1::zeros(0);
        let result_empty = hrf.evaluate(&t_empty).unwrap();
        assert_eq!(result_empty.shape(), &[0, 1]);
    }

    #[test]
    fn test_hrf_from_coefficients() {
        // Create a multi-basis HRF
        let multi_hrf = Box::new(MultiBasisHrf) as Box<dyn Hrf>;

        // Test with equal weights (should give average of basis functions)
        let coeffs = vec![1.0, 1.0, 1.0];
        let weighted_hrf = hrf_from_coefficients(multi_hrf, &coeffs).unwrap();

        // Check properties
        assert_eq!(weighted_hrf.nbasis(), 1);
        assert!(weighted_hrf.name().contains("from_coef"));
        assert_eq!(weighted_hrf.span(), 20.0);

        // Evaluate and check shape
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = weighted_hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[21, 1]);

        // Check that result is non-zero
        let max_val = result.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val > 0.0);
    }

    #[test]
    fn test_hrf_from_coefficients_single_basis() {
        // Test with a single-basis HRF
        let test_hrf = Box::new(TestHrf) as Box<dyn Hrf>;

        // Single coefficient should scale the HRF
        let coeffs = vec![2.0];
        let scaled_hrf = hrf_from_coefficients(test_hrf, &coeffs).unwrap();

        assert_eq!(scaled_hrf.nbasis(), 1);

        // Evaluate both
        let t = Array1::linspace(0.0, 10.0, 11);
        let original = TestHrf.evaluate(&t).unwrap();
        let scaled = scaled_hrf.evaluate(&t).unwrap();

        // Scaled should be 2x the original
        for i in 0..t.len() {
            assert_relative_eq!(scaled[[i, 0]], 2.0 * original[[i, 0]], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hrf_from_coefficients_invalid_length() {
        let multi_hrf = Box::new(MultiBasisHrf) as Box<dyn Hrf>;

        // Wrong number of coefficients should fail
        let coeffs = vec![1.0, 2.0]; // Need 3 for MultiBasisHrf
        let result = hrf_from_coefficients(multi_hrf, &coeffs);

        assert!(result.is_err());
        if let Err(HrfError::InvalidInput(msg)) = result {
            assert!(msg.contains("coefficient length"));
        }
    }

    #[test]
    fn test_evaluate_deriv_numeric() {
        // Test derivative evaluation with numeric method
        let hrf = TestHrf;
        let t = Array1::linspace(0.0, 10.0, 11);

        // Use numeric derivative method
        let deriv = hrf
            .evaluate_deriv(&t, crate::hrf::derivatives::DerivativeMethod::Numeric)
            .unwrap();

        // Check shape
        assert_eq!(deriv.shape(), &[11, 1]);

        // TestHrf is triangular: f(t) = t/5 for t<=5, f(t) = 2-t/5 for t>5
        // So derivative should be approximately 0.2 for t<5 and -0.2 for t>5

        // Check derivative in ascending region (t=2, should be ~0.2)
        assert_relative_eq!(deriv[[2, 0]], 0.2, epsilon = 1e-3);

        // Check derivative in descending region (t=8, should be ~-0.2)
        assert_relative_eq!(deriv[[8, 0]], -0.2, epsilon = 1e-3);

        // At the peak (t=5), derivative should be near 0 or have a discontinuity
        assert!(
            deriv[[5, 0]].abs() < 0.3,
            "Derivative at peak should be small"
        );
    }

    #[test]
    fn test_evaluate_deriv_multi_basis() {
        // Test derivative with multi-basis HRF
        let hrf = MultiBasisHrf;
        let t = Array1::linspace(0.0, 20.0, 21);

        let deriv = hrf
            .evaluate_deriv(&t, crate::hrf::derivatives::DerivativeMethod::Auto)
            .unwrap();

        // Should have derivatives for all 3 basis functions
        assert_eq!(deriv.shape(), &[21, 3]);

        // Check that derivatives are non-zero
        for basis_idx in 0..3 {
            let max_abs = deriv
                .column(basis_idx)
                .iter()
                .map(|&x| x.abs())
                .fold(0.0, f64::max);
            assert!(
                max_abs > 0.0,
                "Derivative for basis {} should be non-zero",
                basis_idx
            );
        }
    }

    #[test]
    fn test_evaluate_deriv_trait_object() {
        // Test that evaluate_deriv works through trait objects
        let hrf: Box<dyn Hrf> = Box::new(TestHrf);
        let t = Array1::linspace(0.0, 10.0, 101);

        // Should work through HrfExt trait
        let deriv = hrf
            .evaluate_deriv(&t, crate::hrf::derivatives::DerivativeMethod::Auto)
            .unwrap();

        assert_eq!(deriv.shape(), &[101, 1]);

        // Check that result is reasonable (non-zero for most points)
        let non_zero_count = deriv.iter().filter(|&&x| x.abs() > 1e-6).count();
        assert!(
            non_zero_count > 50,
            "Expected most derivative values to be non-zero"
        );
    }
}
