//! Empirical HRF functions for data-driven HRF estimation.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use crate::types::{HrfValues, TimeArray};
use fmri_splines::Spline;
use ndarray::{Array1, Array2};
use std::fmt::Debug;
use std::sync::Arc;

/// Interpolation method for empirical HRF.
#[derive(Debug, Clone, Copy)]
pub enum InterpolationMethod {
    /// Linear interpolation between points.
    Linear,
    /// Cubic spline interpolation (requires at least 4 points).
    Cubic,
}

/// Create an empirical HRF from measured data points.
///
/// This function creates an HRF by interpolating between provided time points
/// and their corresponding HRF values. This is useful for creating HRFs from
/// measured fMRI data or for importing HRFs from other software.
///
/// # Arguments
/// * `t` - Time points where HRF was measured
/// * `y` - HRF values at the corresponding time points
/// * `name` - Optional name for the HRF (defaults to "empirical_hrf")
/// * `span` - Optional temporal span (defaults to max time point)
///
/// # Returns
/// A boxed HRF trait object that can be used like any other HRF.
///
/// # Examples
/// ```
/// use fmri_hrf::hrf::empirical_hrf;
/// use ndarray::array;
///
/// // Create HRF from measured points
/// let t_points = array![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.];
/// let y_values = array![0., 0.2, 0.8, 1.0, 0.7, 0.3, 0., -0.1, -0.05, 0., 0.];
///
/// let hrf = empirical_hrf(&t_points, &y_values, None, None).unwrap();
///
/// // Evaluate at new time points
/// let new_times = ndarray::Array1::linspace(0., 20., 100);
/// let response = hrf.evaluate(&new_times).unwrap();
/// ```
pub fn empirical_hrf(
    t: &Array1<f64>,
    y: &Array1<f64>,
    name: Option<&str>,
    span: Option<f64>,
) -> Result<Box<dyn Hrf>> {
    empirical_hrf_with_method(t, y, name, span, InterpolationMethod::Linear)
}

/// Construct an empirical HRF with explicit interpolation. Linear matches R and
/// ScalaFIM. Cubic is an opt-in smooth interpolation, never an implicit fallback.
pub fn empirical_hrf_with_method(
    t: &Array1<f64>,
    y: &Array1<f64>,
    name: Option<&str>,
    span: Option<f64>,
    method: InterpolationMethod,
) -> Result<Box<dyn Hrf>> {
    // Validate inputs
    if t.len() != y.len() {
        return Err(HrfError::InvalidInput(
            "t and y must have the same length".into(),
        ));
    }

    if t.len() < 2 {
        return Err(HrfError::InvalidInput(
            "Need at least 2 points to create empirical HRF".into(),
        ));
    }

    // Check for NaN or infinite values
    if t.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(HrfError::InvalidInput(
            "Time points cannot contain NaN or infinite values".into(),
        ));
    }

    if y.iter().any(|&x| x.is_nan() || x.is_infinite()) {
        return Err(HrfError::InvalidInput(
            "HRF values cannot contain NaN or infinite values".into(),
        ));
    }

    // Sort by time
    let mut indices: Vec<usize> = (0..t.len()).collect();
    indices.sort_by(|&i, &j| t[i].partial_cmp(&t[j]).unwrap());

    let t_sorted: Vec<f64> = indices.iter().map(|&i| t[i]).collect();
    let y_sorted: Vec<f64> = indices.iter().map(|&i| y[i]).collect();

    // Determine span
    let span = span.unwrap_or(t_sorted[t_sorted.len() - 1]);

    if t_sorted.windows(2).any(|w| w[0] == w[1]) {
        return Err(HrfError::InvalidInput(
            "empirical times must be distinct".into(),
        ));
    }
    if !span.is_finite() || span <= 0.0 || span < *t_sorted.last().unwrap() {
        return Err(HrfError::InvalidInput(
            "span must be positive, finite and cover samples".into(),
        ));
    }

    // Create interpolator
    let interpolator = Interpolator::new(t_sorted.clone(), y_sorted.clone(), method)?;

    // Create empirical HRF struct
    Ok(Box::new(EmpiricalHrf {
        name: name.unwrap_or("empirical_hrf").to_string(),
        span,
        n_points: t.len(),
        interpolator: Arc::new(interpolator),
    }))
}

/// Alias for empirical_hrf for R compatibility.
pub fn gen_empirical_hrf(
    t: &Array1<f64>,
    y: &Array1<f64>,
    name: Option<&str>,
    span: Option<f64>,
) -> Result<Box<dyn Hrf>> {
    empirical_hrf(t, y, name, span)
}

/// Empirical HRF implementation.
#[derive(Debug, Clone)]
struct EmpiricalHrf {
    name: String,
    span: f64,
    n_points: usize,
    interpolator: Arc<Interpolator>,
}

impl Hrf for EmpiricalHrf {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Empirical {
            times: self.interpolator.t.clone(),
            values: self.interpolator.y.clone(),
            name: self.name.clone(),
            span: self.span,
            cubic: matches!(self.interpolator.method, InterpolationMethod::Cubic),
        })
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        self.interpolator.interpolate(t)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![
            ("n_points", self.n_points as f64),
            (
                "method",
                match self.interpolator.method {
                    InterpolationMethod::Linear => 0.0,
                    InterpolationMethod::Cubic => 1.0,
                },
            ),
        ]
    }
}

/// Internal interpolator for empirical HRF.
#[derive(Debug, Clone)]
struct Interpolator {
    t: Vec<f64>,
    y: Vec<f64>,
    method: InterpolationMethod,
    // For R-compatible splines
    spline: Option<fmri_splines::InterpolationSpline<f64>>,
}

impl Interpolator {
    fn new(t: Vec<f64>, y: Vec<f64>, method: InterpolationMethod) -> Result<Self> {
        let spline = match method {
            InterpolationMethod::Cubic => {
                if t.len() < 4 {
                    return Err(HrfError::InvalidInput(
                        "Cubic interpolation requires at least 4 points".into(),
                    ));
                }

                // Create interpolation spline using the splines library
                // Order 4 = cubic spline (degree 3)
                Some(
                    fmri_splines::InterpolationSpline::new(&t, &y, 4).map_err(|e| {
                        HrfError::NumericalError(format!("Spline creation failed: {}", e))
                    })?,
                )
            }
            InterpolationMethod::Linear => None,
        };

        Ok(Self {
            t,
            y,
            method,
            spline,
        })
    }

    fn interpolate(&self, t_eval: &TimeArray) -> Result<HrfValues> {
        if t_eval.iter().any(|t| !t.is_finite()) {
            return Err(HrfError::InvalidInput(
                "evaluation times must be finite".into(),
            ));
        }
        let mut result = Array2::zeros((t_eval.len(), 1));

        for (i, &t_val) in t_eval.iter().enumerate() {
            result[[i, 0]] = self.interpolate_point(t_val)?;
        }

        Ok(result)
    }

    fn interpolate_point(&self, t_val: f64) -> Result<f64> {
        // Handle extrapolation - return 0 outside bounds
        if t_val < self.t[0] || t_val > self.t[self.t.len() - 1] {
            return Ok(0.0);
        }

        match &self.spline {
            Some(spline) => spline
                .evaluate(t_val)
                .map_err(|e| HrfError::NumericalError(e.to_string())),
            None => {
                let idx = self
                    .t
                    .partition_point(|x| *x <= t_val)
                    .saturating_sub(1)
                    .min(self.t.len() - 2);
                let alpha = (t_val - self.t[idx]) / (self.t[idx + 1] - self.t[idx]);
                Ok(self.y[idx] * (1.0 - alpha) + self.y[idx + 1] * alpha)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_empirical_hrf_basic() {
        let t = Array1::from_vec(vec![0., 5., 10., 15., 20.]);
        let y = Array1::from_vec(vec![0., 1., 0.5, 0.1, 0.]);

        let hrf = empirical_hrf(&t, &y, None, None).unwrap();

        assert_eq!(hrf.name(), "empirical_hrf");
        assert_eq!(hrf.nbasis(), 1);
        assert_eq!(hrf.span(), 20.0);

        // Test evaluation at original points
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[5, 1]);

        // Should approximately match original values
        for i in 0..5 {
            assert_relative_eq!(result[[i, 0]], y[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_empirical_hrf_interpolation() {
        let t = Array1::from_vec(vec![0., 10., 20.]);
        let y = Array1::from_vec(vec![0., 1., 0.]);

        let hrf = empirical_hrf(&t, &y, Some("test_hrf"), Some(25.0)).unwrap();

        assert_eq!(hrf.name(), "test_hrf");
        assert_eq!(hrf.span(), 25.0);

        // Test interpolation at intermediate points
        let t_eval = Array1::from_vec(vec![0., 5., 10., 15., 20.]);
        let result = hrf.evaluate(&t_eval).unwrap();

        // Linear interpolation expected values
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[[1, 0]], 0.5, epsilon = 1e-6);
        assert_relative_eq!(result[[2, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[[3, 0]], 0.5, epsilon = 1e-6);
        assert_relative_eq!(result[[4, 0]], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_empirical_hrf_extrapolation() {
        let t = Array1::from_vec(vec![5., 10., 15.]);
        let y = Array1::from_vec(vec![0.5, 1., 0.5]);

        let hrf = empirical_hrf(&t, &y, None, None).unwrap();

        // Test extrapolation (should return 0)
        let t_eval = Array1::from_vec(vec![0., 2.5, 5., 15., 20., 25.]);
        let result = hrf.evaluate(&t_eval).unwrap();

        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-6); // Before range
        assert_relative_eq!(result[[1, 0]], 0.0, epsilon = 1e-6); // Before range
        assert_relative_eq!(result[[2, 0]], 0.5, epsilon = 1e-6); // First point
        assert_relative_eq!(result[[3, 0]], 0.5, epsilon = 1e-6); // Last point
        assert_relative_eq!(result[[4, 0]], 0.0, epsilon = 1e-6); // After range
        assert_relative_eq!(result[[5, 0]], 0.0, epsilon = 1e-6); // After range
    }

    #[test]
    fn test_empirical_hrf_errors() {
        let t = Array1::from_vec(vec![0., 5., 10.]);
        let y = Array1::from_vec(vec![0., 1.]);

        // Different lengths
        assert!(empirical_hrf(&t, &y, None, None).is_err());

        // Too few points
        let t_short = Array1::from_vec(vec![0.]);
        let y_short = Array1::from_vec(vec![1.]);
        assert!(empirical_hrf(&t_short, &y_short, None, None).is_err());

        // NaN values
        let t_nan = Array1::from_vec(vec![0., f64::NAN, 10.]);
        let y_good = Array1::from_vec(vec![0., 1., 0.]);
        assert!(empirical_hrf(&t_nan, &y_good, None, None).is_err());
    }

    #[test]
    fn test_gen_empirical_hrf_alias() {
        let t = Array1::from_vec(vec![0., 5., 10.]);
        let y = Array1::from_vec(vec![0., 1., 0.]);

        let hrf1 = empirical_hrf(&t, &y, None, None).unwrap();
        let hrf2 = gen_empirical_hrf(&t, &y, None, None).unwrap();

        // Both should produce the same results
        let t_eval = Array1::linspace(0., 10., 11);
        let result1 = hrf1.evaluate(&t_eval).unwrap();
        let result2 = hrf2.evaluate(&t_eval).unwrap();

        for i in 0..11 {
            assert_relative_eq!(result1[[i, 0]], result2[[i, 0]], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_empirical_hrf_cubic_spline() {
        // Test with enough points for cubic spline
        let t = Array1::from_vec(vec![0., 2., 4., 6., 8., 10.]);
        let y = Array1::from_vec(vec![0., 0.3, 0.8, 1.0, 0.6, 0.]);

        let hrf = empirical_hrf(&t, &y, None, None).unwrap();

        // Test smoothness of interpolation
        let t_eval = Array1::linspace(0., 10., 101);
        let result = hrf.evaluate(&t_eval).unwrap();

        // Check that we get original values at knots
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(result[[20, 0]], 0.3, epsilon = 1e-6);
        assert_relative_eq!(result[[40, 0]], 0.8, epsilon = 1e-6);
        assert_relative_eq!(result[[60, 0]], 1.0, epsilon = 1e-6);
        assert_relative_eq!(result[[80, 0]], 0.6, epsilon = 1e-6);
        assert_relative_eq!(result[[100, 0]], 0.0, epsilon = 1e-6);
    }
}
