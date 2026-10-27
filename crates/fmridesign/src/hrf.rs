//! HRF (Hemodynamic Response Function) utilities and integration.
//!
//! This module provides various HRF models and utilities for convolving
//! event designs with hemodynamic response functions.

use crate::types::ValidationError;
use fmrihrf::{hrf as hrf_lib, Hrf, HrfError};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// HRF specification with parameters
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum HrfSpec {
    /// Standard named HRF (e.g., "canonical", "gamma")
    Named {
        name: String,
        params: HashMap<String, f64>,
    },
    /// FIR basis set
    Fir {
        n_basis: usize,
        length: f64,
        tr: f64,
    },
    /// B-spline basis set
    BSpline {
        n_basis: usize,
        length: f64,
        order: usize,
    },
    /// Fourier basis set
    Fourier {
        n_basis: usize,
        length: f64,
        tr: f64,
    },
    /// Custom HRF function
    Custom { values: Array1<f64>, tr: f64 },
}

impl HrfSpec {
    /// Create a canonical HRF spec
    pub fn canonical() -> Self {
        Self::Named {
            name: "canonical".to_string(),
            params: HashMap::new(),
        }
    }

    /// Create a gamma HRF spec
    pub fn gamma(shape: f64, scale: f64) -> Self {
        let mut params = HashMap::new();
        params.insert("shape".to_string(), shape);
        params.insert("scale".to_string(), scale);
        Self::Named {
            name: "gamma".to_string(),
            params,
        }
    }

    /// Create a FIR basis spec
    pub fn fir(n_basis: usize, length: f64, tr: f64) -> Self {
        Self::Fir {
            n_basis,
            length,
            tr,
        }
    }

    /// Create a B-spline basis spec
    pub fn bspline(n_basis: usize, length: f64, order: usize) -> Self {
        Self::BSpline {
            n_basis,
            length,
            order,
        }
    }

    /// Create a Fourier basis spec
    pub fn fourier(n_basis: usize, length: f64, tr: f64) -> Self {
        Self::Fourier {
            n_basis,
            length,
            tr,
        }
    }

    /// Create a custom HRF spec from values
    pub fn custom(values: Array1<f64>, tr: f64) -> Self {
        Self::Custom { values, tr }
    }
}

/// Create an HRF from specification
pub fn create_hrf(spec: &HrfSpec) -> Result<Box<dyn Hrf>, ValidationError> {
    // Validate serialized specifications at the numerical construction boundary.
    match spec {
        HrfSpec::Fir {
            n_basis,
            length,
            tr,
        }
        | HrfSpec::Fourier {
            n_basis,
            length,
            tr,
        } if *n_basis == 0
            || !length.is_finite()
            || *length <= 0.0
            || !tr.is_finite()
            || *tr <= 0.0 =>
        {
            return Err(ValidationError::InvalidParameter(
                "invalid basis count/span/TR".into(),
            ))
        }
        HrfSpec::BSpline {
            n_basis,
            length,
            order,
        } if *n_basis == 0
            || *order == 0
            || *n_basis < order.saturating_sub(1)
            || !length.is_finite()
            || *length <= 0.0 =>
        {
            return Err(ValidationError::InvalidParameter(
                "invalid spline count/span/degree".into(),
            ))
        }
        _ => {}
    }
    match spec {
        HrfSpec::Named { name, params } => create_named_hrf(name, params),
        HrfSpec::Fir {
            n_basis,
            length,
            tr: _,
        } => {
            // fmrihrf's fir_hrf returns Box<dyn Hrf>, not Result
            Ok(hrf_lib::fir_hrf(*n_basis, Some(*length)))
        }
        HrfSpec::BSpline {
            n_basis,
            length,
            order,
        } => {
            // fmrihrf's bspline_hrf returns Box<dyn Hrf>, not Result
            Ok(hrf_lib::bspline_hrf(*n_basis, Some(*order), Some(*length)))
        }
        HrfSpec::Fourier {
            n_basis,
            length,
            tr: _,
        } => {
            // fmrihrf's fourier_hrf returns Box<dyn Hrf>, not Result
            Ok(hrf_lib::fourier_hrf(*n_basis, Some(*length)))
        }
        HrfSpec::Custom { values, tr } => {
            if values.len() < 2
                || !tr.is_finite()
                || *tr <= 0.0
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(ValidationError::InvalidParameter(
                    "custom HRF needs two finite samples and a positive finite TR".into(),
                ));
            }
            // Use fmrihrf's empirical_hrf for custom HRF from values
            let t = Array1::linspace(0.0, (values.len() - 1) as f64 * tr, values.len());
            match hrf_lib::empirical_hrf(&t, values, Some("custom"), None) {
                Ok(hrf) => Ok(hrf),
                Err(e) => Err(ValidationError::InvalidParameter(format!(
                    "Failed to create custom HRF: {}",
                    e
                ))),
            }
        }
    }
}

/// Create a named HRF
fn create_named_hrf(
    name: &str,
    params: &HashMap<String, f64>,
) -> Result<Box<dyn Hrf>, ValidationError> {
    if params.values().any(|v| !v.is_finite()) {
        return Err(ValidationError::InvalidParameter(
            "nonfinite HRF parameter".into(),
        ));
    }
    let spm = matches!(
        name,
        "canonical"
            | "spm_canonical"
            | "spm"
            | "spmg1"
            | "spmg2"
            | "spmg3"
            | "spm_time"
            | "spm_time_derivative"
            | "spm_time_dispersion"
    );
    if spm {
        if params
            .keys()
            .any(|k| !["p1", "p2", "a1"].contains(&k.as_str()))
        {
            return Err(ValidationError::InvalidParameter(
                "SPMG parameters are p1, p2, a1".into(),
            ));
        }
        let p1 = params.get("p1").copied().unwrap_or(5.0);
        let p2 = params.get("p2").copied().unwrap_or(15.0);
        let a1 = params
            .get("a1")
            .copied()
            .unwrap_or(fmrihrf::hrf::functions::SPM_A1);
        if p1 <= 0.0 || p2 <= 0.0 || a1 <= 0.0 {
            return Err(ValidationError::InvalidParameter(
                "positive SPMG parameters required".into(),
            ));
        }
        return Ok(match name {
            "spmg2" | "spm_time" | "spm_time_derivative" => {
                Box::new(hrf_lib::SpmG2Hrf::new(p1, p2, a1))
            }
            "spmg3" | "spm_time_dispersion" => Box::new(hrf_lib::SpmG3Hrf::new(p1, p2, a1)),
            _ => Box::new(hrf_lib::SpmG1Hrf::new(p1, p2, a1)),
        });
    }
    match name {
        "gamma" => {
            let shape = params.get("shape").copied().unwrap_or(6.0);
            let rate = params
                .get("rate")
                .copied()
                .unwrap_or_else(|| 1.0 / params.get("scale").copied().unwrap_or(1.0));
            Ok(hrf_lib::gamma_hrf(shape, rate, Some(false)))
        }
        "gaussian" => {
            let mean = params.get("mean").copied().unwrap_or(6.0);
            let std = params
                .get("sd")
                .or_else(|| params.get("std"))
                .copied()
                .unwrap_or(2.0);
            Ok(hrf_lib::gaussian_hrf(mean, std, None))
        }
        "mexhat" | "mexican_hat" => {
            let sigma = params.get("sigma").copied().unwrap_or(2.0);
            let center = params.get("center").copied().unwrap_or(6.0);
            Ok(hrf_lib::mexhat_hrf(sigma, center))
        }
        "sine" => {
            let n_basis = params.get("n_basis").map(|v| *v as usize).unwrap_or(4);
            let span = params.get("span").copied().unwrap_or(30.0);
            Ok(hrf_lib::sine_basis_hrf(n_basis, span))
        }
        "half_cosine" => {
            let width = params.get("width").copied().unwrap_or(12.0);
            let center = params.get("center").copied().unwrap_or(6.0);
            Ok(hrf_lib::half_cosine_hrf(width, center))
        }
        "inv_logit" | "inverse_logit" => {
            let mu1 = params
                .get("mu1")
                .or(params.get("slope"))
                .copied()
                .unwrap_or(6.0);
            let s1 = params.get("s1").copied().unwrap_or(1.0);
            let mu2 = params
                .get("mu2")
                .or(params.get("shift"))
                .copied()
                .unwrap_or(16.0);
            let s2 = params.get("s2").copied().unwrap_or(1.0);
            let lag = params.get("lag").copied().unwrap_or(0.0);
            Ok(hrf_lib::inv_logit_hrf(mu1, s1, mu2, s2, lag))
        }
        "lwu" => {
            // Use Liu, Wu, and Gao (2017) HRF model
            hrf_lib::lwu_hrf(
                params.get("tau").copied(),
                params.get("sigma").copied(),
                params.get("rho").copied(),
                None,
            )
            .map_err(|e| {
                ValidationError::InvalidParameter(format!("Failed to create LWU HRF: {}", e))
            })
        }
        // Try to get from predefined library
        _ => {
            match hrf_lib::get_predefined_hrf(name) {
                Some(hrf) => Ok(hrf),
                None => {
                    // Try registry as last resort
                    match hrf_lib::get_hrf(name) {
                        Ok(hrf) => Ok(hrf),
                        Err(_) => Err(ValidationError::InvalidParameter(format!(
                            "Unknown HRF: {}",
                            name
                        ))),
                    }
                }
            }
        }
    }
}

/// Get HRF duration in seconds
pub fn get_hrf_duration(spec: &HrfSpec) -> f64 {
    match spec {
        HrfSpec::Named { .. } => 32.0, // Default for named HRFs
        HrfSpec::Fir { length, .. } => *length,
        HrfSpec::BSpline { length, .. } => *length,
        HrfSpec::Fourier { length, .. } => *length,
        HrfSpec::Custom { values, tr } => values.len() as f64 * tr,
    }
}

/// Simple HRF for testing
pub fn simple_hrf() -> Box<dyn Hrf> {
    #[derive(Debug)]
    struct SimpleHrf;

    impl Hrf for SimpleHrf {
        fn name(&self) -> &str {
            "simple"
        }

        fn nbasis(&self) -> usize {
            1
        }

        fn span(&self) -> f64 {
            20.0
        }

        fn params(&self) -> Vec<(&str, f64)> {
            vec![]
        }

        fn evaluate(&self, t: &Array1<f64>) -> Result<Array2<f64>, HrfError> {
            let result = t.mapv(|v| v * (-v / 2.0).exp());
            let max_val = result.fold(0.0f64, |a, &b| a.max(b));
            let normalized = if max_val > 0.0 {
                result / max_val
            } else {
                result
            };
            Ok(normalized.insert_axis(ndarray::Axis(1)))
        }
    }

    Box::new(SimpleHrf)
}

/// List all available HRF names
pub fn list_available_hrfs() -> Vec<&'static str> {
    vec![
        "canonical",
        "spm_canonical",
        "spm",
        "gamma",
        "gaussian",
        "mexhat",
        "mexican_hat",
        "sine",
        "half_cosine",
        "inv_logit",
        "inverse_logit",
        "lwu",
        "spm_time",
        "spm_time_derivative",
        "spm_time_dispersion",
        "spmg1",
        "spmg2",
        "spmg3",
    ]
}

/// Create HRF from string name (convenience function)
pub fn create_hrf_by_name(name: &str) -> Option<Box<dyn Hrf>> {
    let spec = HrfSpec::Named {
        name: name.to_string(),
        params: HashMap::new(),
    };
    match create_hrf(&spec) {
        Ok(hrf) => Some(hrf),
        Err(_) => None,
    }
}

/// Create HRF with generators from fmrihrf
pub use fmrihrf::hrf::{
    bind_hrfs, block_hrf, bspline_generator, fir_generator, fourier_generator, gamma_generator,
    hrf_from_basis, lag_hrf, make_hrf, make_hrf_from_params, normalize_hrf,
};

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_canonical_hrf() {
        let spec = HrfSpec::canonical();
        let hrf = create_hrf(&spec).unwrap();

        let t = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = hrf.evaluate(&t).unwrap();

        // Check shape
        assert_eq!(result.shape(), &[6, 1]);

        // Check peak is around 5-6 seconds
        let peak_idx = result
            .column(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert!(peak_idx >= 4 && peak_idx <= 6);
    }

    #[test]
    fn test_fir_basis() {
        let spec = HrfSpec::fir(5, 10.0, 2.0);
        let hrf = create_hrf(&spec).unwrap();

        let t = array![0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
        let result = hrf.evaluate(&t).unwrap();

        // Check that we get the expected number of basis columns
        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 5);

        // Each column should be non-negative (FIR basis functions are indicator-like)
        for j in 0..5 {
            let col_max = result
                .column(j)
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(
                col_max >= 0.0,
                "FIR basis column {} should have non-negative max",
                j
            );
        }
    }

    #[test]
    fn test_fourier_basis() {
        let spec = HrfSpec::fourier(5, 10.0, 1.0);
        let hrf = create_hrf(&spec).unwrap();

        let t = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let result = hrf.evaluate(&t).unwrap();

        // Check shape: 6 time points x 5 basis functions
        assert_eq!(result.nrows(), 6);
        assert_eq!(result.ncols(), 5);

        // Quarter-period values distinguish sin/cos order and exclude a DC term.
        let quarter = hrf.evaluate(&array![0., 2.5, 5., 7.5]).unwrap();
        for (i, &expected) in [0., 1., 0., -1.].iter().enumerate() {
            assert!((quarter[(i, 0)] - expected).abs() < 1e-12);
        }
        assert!((quarter[(0, 1)] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_custom_hrf() {
        let values = array![0.0, 0.5, 1.0, 0.5, 0.0];
        let spec = HrfSpec::custom(values.clone(), 1.0);
        let hrf = create_hrf(&spec).unwrap();

        let t = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let result = hrf.evaluate(&t).unwrap();

        // Check shape
        assert_eq!(result.shape(), &[5, 1]);

        // Values should match the input at these time points
        assert!((result[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((result[[1, 0]] - 0.5).abs() < 1e-6);
        assert!((result[[2, 0]] - 1.0).abs() < 1e-6);
        assert!((result[[3, 0]] - 0.5).abs() < 1e-6);
        assert!((result[[4, 0]] - 0.0).abs() < 1e-6);
    }
}

/// Get HRF from specification.
pub fn get_hrf_from_spec(spec: &HrfSpec) -> Result<Box<dyn Hrf>, ValidationError> {
    create_hrf(spec)
}

/// Get HRF by name with parameters.
pub fn get_hrf(name: &str, params: &HashMap<String, f64>) -> Result<Box<dyn Hrf>, ValidationError> {
    let spec = HrfSpec::Named {
        name: name.to_string(),
        params: params.clone(),
    };
    create_hrf(&spec)
}
