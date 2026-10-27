//! HRF generator functions for creating and combining HRFs.

use crate::error::{HrfError, Result};
use crate::hrf::core::Hrf;
use crate::hrf::functions::*;
use crate::hrf::library::get_predefined_hrf;
use crate::hrf::registry::get_hrf;
use crate::types::{HrfValues, TimeArray};
use ndarray::{concatenate, Array2, Axis};
use std::collections::HashMap;

/// Generate an HRF with optional lag and block width.
///
/// This is a general-purpose HRF generator that can create HRFs
/// and apply lag and/or block decorators.
#[derive(Debug)]
pub struct HrfGenerator {
    base_hrf: Box<dyn Hrf>,
    lag: f64,
    block_width: f64,
    normalize: bool,
}

impl HrfGenerator {
    /// Create a new HRF generator.
    pub fn new(base_hrf: Box<dyn Hrf>) -> Self {
        Self {
            base_hrf,
            lag: 0.0,
            block_width: 0.0,
            normalize: false,
        }
    }

    /// Set temporal lag in seconds.
    pub fn with_lag(mut self, lag: f64) -> Self {
        self.lag = lag;
        self
    }

    /// Set block width in seconds.
    pub fn with_block(mut self, width: f64) -> Self {
        self.block_width = width;
        self
    }

    /// Enable normalization to unit peak.
    pub fn normalized(mut self) -> Self {
        self.normalize = true;
        self
    }

    /// Build the final HRF with all decorators applied.
    pub fn build(self) -> Box<dyn Hrf> {
        let mut hrf = self.base_hrf;

        if self.lag != 0.0 {
            hrf = Box::new(LaggedHrf {
                inner: hrf,
                lag: self.lag,
            });
        }

        if self.block_width > 0.0 {
            hrf = Box::new(BlockedHrf {
                inner: hrf,
                width: self.block_width,
            });
        }

        if self.normalize {
            hrf = crate::hrf::normalization::normalize_legacy(hrf);
        }

        hrf
    }
}

/// Decorator that applies temporal lag to an HRF.
#[derive(Debug)]
struct LaggedHrf {
    inner: Box<dyn Hrf>,
    lag: f64,
}

impl Hrf for LaggedHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        // Shift time points by lag
        let shifted_t = t.mapv(|time| time - self.lag);
        self.inner.evaluate(&shifted_t)
    }

    fn name(&self) -> &str {
        "lagged_hrf"
    }

    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }

    fn span(&self) -> f64 {
        self.inner.span() + self.lag.max(0.0)
    }

    fn params(&self) -> Vec<(&str, f64)> {
        let mut params = self.inner.params();
        params.push(("lag", self.lag));
        params
    }
}

/// Decorator that applies block convolution to an HRF.
#[derive(Debug)]
struct BlockedHrf {
    inner: Box<dyn Hrf>,
    width: f64,
}

impl Hrf for BlockedHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        // Use the evaluate_block method from HrfExt
        use crate::hrf::HrfExt;
        self.inner.evaluate_block(t, self.width, 0.1, true, false)
    }

    fn name(&self) -> &str {
        "blocked_hrf"
    }

    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }

    fn span(&self) -> f64 {
        self.inner.span() + self.width
    }

    fn params(&self) -> Vec<(&str, f64)> {
        let mut params = self.inner.params();
        params.push(("block_width", self.width));
        params
    }
}

/// Combine multiple HRFs into a single multi-basis HRF.
pub fn bind_hrfs(hrfs: Vec<Box<dyn Hrf>>) -> Box<dyn Hrf> {
    Box::new(CombinedHrf { hrfs })
}

/// Combined multi-basis HRF.
#[derive(Debug)]
struct CombinedHrf {
    hrfs: Vec<Box<dyn Hrf>>,
}

impl Hrf for CombinedHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let mut results = Vec::new();

        for hrf in &self.hrfs {
            results.push(hrf.evaluate(t)?);
        }

        // Concatenate along columns (basis dimension)
        let views: Vec<_> = results.iter().map(|arr| arr.view()).collect();
        Ok(concatenate(Axis(1), &views).unwrap())
    }

    fn name(&self) -> &str {
        "combined_hrf"
    }

    fn nbasis(&self) -> usize {
        self.hrfs.iter().map(|h| h.nbasis()).sum()
    }

    fn span(&self) -> f64 {
        self.hrfs
            .iter()
            .map(|h| h.span())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    fn params(&self) -> Vec<(&str, f64)> {
        let mut all_params = Vec::new();
        for (i, hrf) in self.hrfs.iter().enumerate() {
            for (name, value) in hrf.params() {
                all_params.push((
                    Box::leak(format!("hrf{}_{}", i, name).into_boxed_str()) as &str,
                    value,
                ));
            }
        }
        all_params
    }
}

/// Helper function to create gamma HRF generator.
pub fn gamma_generator(shape: f64, scale: f64) -> HrfGenerator {
    HrfGenerator::new(gamma_hrf(shape, scale, None))
}

/// Helper function to create B-spline HRF generator.
pub fn bspline_generator(n_basis: usize, degree: Option<usize>, span: Option<f64>) -> HrfGenerator {
    HrfGenerator::new(bspline_hrf(n_basis, degree, span))
}

/// Helper function to create FIR HRF generator.
pub fn fir_generator(n_basis: usize, span: Option<f64>) -> HrfGenerator {
    HrfGenerator::new(fir_hrf(n_basis, span))
}

/// Helper function to create Fourier HRF generator.
pub fn fourier_generator(n_basis: usize, span: Option<f64>) -> HrfGenerator {
    HrfGenerator::new(fourier_hrf(n_basis, span))
}

/// Create Daguerre generator for B-spline basis with optimized knot placement.
///
/// The Daguerre basis uses Laguerre-type orthogonal polynomials with
/// exponential decay, matching the R package `fmri_hrf` implementation.
///
/// # Arguments
/// * `n_basis` - Number of basis functions (default: 3)
/// * `span` - Temporal span in seconds (default: 30.0)
/// * `scale` - Time-scaling parameter (default: 4.0)
///
/// # Returns
/// Daguerre HRF generator
pub fn daguerre_generator(
    n_basis: Option<usize>,
    span: Option<f64>,
    scale: Option<f64>,
) -> HrfGenerator {
    let n = n_basis.unwrap_or(3);
    let span = span.unwrap_or(30.0);
    let scale = scale.unwrap_or(4.0);
    HrfGenerator::new(daguerre_hrf(n, span, scale))
}

/// Create Daguerre HRF using Laguerre-type orthogonal polynomials.
///
/// Implements the recurrence relation from the R `daguerre_basis` function:
/// - Basis 0: exp(-x/2)
/// - Basis 1: (1 - x) * exp(-x/2)
/// - Basis k (k >= 2): ((2k-1-x) * basis\[k-1\] - (k-1) * basis\[k-2\]) / k
///
/// where x = t / scale. Each column is normalized by its maximum absolute value.
fn daguerre_hrf(n_basis: usize, span: f64, scale: f64) -> Box<dyn Hrf> {
    #[derive(Debug)]
    struct DaguerreHrf {
        n_basis: usize,
        span: f64,
        scale: f64,
    }

    impl Hrf for DaguerreHrf {
        fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
            crate::penalty::PenaltyKind::DaguerreDecay
        }
        fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
            let n_time = t.len();
            let mut basis = Array2::<f64>::zeros((n_time, self.n_basis));

            // Scale time: x = t / scale
            let x = t.mapv(|ti| ti / self.scale);

            // First basis (n=0): exp(-x/2)
            for i in 0..n_time {
                basis[[i, 0]] = (-x[i] / 2.0).exp();
            }

            // Second basis (n=1): (1 - x) * exp(-x/2)
            if self.n_basis > 1 {
                for i in 0..n_time {
                    basis[[i, 1]] = (1.0 - x[i]) * (-x[i] / 2.0).exp();
                }
            }

            // Higher order via recurrence:
            // basis[n] = ((2k-1-x)*basis[n-1] - (k-1)*basis[n-2]) / k
            // where k = n (0-indexed n corresponds to R's k = n-1 in 1-indexed)
            for n in 2..self.n_basis {
                let k = n as f64;
                for i in 0..n_time {
                    basis[[i, n]] = ((2.0 * k - 1.0 - x[i]) * basis[[i, n - 1]]
                        - (k - 1.0) * basis[[i, n - 2]])
                        / k;
                }
            }

            // Normalize each column by max absolute value
            for col in 0..self.n_basis {
                let max_abs = basis
                    .column(col)
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0_f64, f64::max);
                if max_abs > 1e-10 {
                    for i in 0..n_time {
                        basis[[i, col]] /= max_abs;
                    }
                }
            }

            Ok(basis)
        }

        fn name(&self) -> &str {
            "daguerre"
        }

        fn nbasis(&self) -> usize {
            self.n_basis
        }

        fn span(&self) -> f64 {
            self.span
        }

        fn params(&self) -> Vec<(&str, f64)> {
            vec![
                ("n_basis", self.n_basis as f64),
                ("span", self.span),
                ("scale", self.scale),
            ]
        }
    }

    Box::new(DaguerreHrf {
        n_basis,
        span,
        scale,
    })
}

/// Apply temporal lag to an HRF.
///
/// This decorator shifts the HRF in time by the specified lag amount.
///
/// # Arguments
/// * `hrf` - The HRF to lag
/// * `lag` - Time lag in seconds
///
/// # Returns
/// New HRF with temporal lag applied
pub fn lag_hrf(hrf: Box<dyn Hrf>, lag: f64) -> Box<dyn Hrf> {
    Box::new(LaggedHrf { inner: hrf, lag })
}

/// Create a blocked/sustained version of an HRF.
///
/// This decorator convolves the HRF with a boxcar function to model
/// sustained stimulation.
///
/// # Arguments
/// * `hrf` - The HRF to block
/// * `width` - Duration of the block in seconds
///
/// # Returns
/// New HRF representing blocked response
pub fn block_hrf(hrf: Box<dyn Hrf>, width: f64) -> Box<dyn Hrf> {
    Box::new(BlockedHrf { inner: hrf, width })
}

/// Normalize an HRF to unit peak amplitude.
///
/// This decorator scales the HRF so that its maximum absolute value is 1.
///
/// # Arguments
/// * `hrf` - The HRF to normalize
///
/// # Returns
/// New HRF with unit peak amplitude
pub fn normalize_hrf(hrf: Box<dyn Hrf>) -> Box<dyn Hrf> {
    crate::hrf::normalization::normalize_legacy(hrf)
}

/// Create HRF from coefficient vector using a basis set.
pub fn hrf_from_basis(basis_hrf: Box<dyn Hrf>, coefficients: Vec<f64>) -> Result<Box<dyn Hrf>> {
    if coefficients.len() != basis_hrf.nbasis() {
        return Err(crate::error::HrfError::InvalidInput(format!(
            "Expected {} coefficients, got {}",
            basis_hrf.nbasis(),
            coefficients.len()
        )));
    }

    Ok(Box::new(CoefficientsHrf {
        basis: basis_hrf,
        coefficients,
    }))
}

/// HRF defined by coefficients applied to a basis.
#[derive(Debug)]
struct CoefficientsHrf {
    basis: Box<dyn Hrf>,
    coefficients: Vec<f64>,
}

impl Hrf for CoefficientsHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let basis_values = self.basis.evaluate(t)?;

        // Matrix multiplication: basis_values @ coefficients
        let mut result = Array2::zeros((t.len(), 1));
        for i in 0..t.len() {
            let mut sum = 0.0;
            for j in 0..self.coefficients.len() {
                sum += basis_values[[i, j]] * self.coefficients[j];
            }
            result[[i, 0]] = sum;
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        "coefficients_hrf"
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn span(&self) -> f64 {
        self.basis.span()
    }

    fn params(&self) -> Vec<(&str, f64)> {
        let mut params = vec![];
        for (i, &coef) in self.coefficients.iter().enumerate() {
            params.push((
                Box::leak(format!("coef_{}", i).into_boxed_str()) as &str,
                coef,
            ));
        }
        params
    }
}

/// Generate an HRF from a string specification or registry name.
///
/// This function provides a flexible way to create HRFs from specifications.
/// It supports:
/// - Registry names: "spm", "gamma", etc.
/// - Function calls: "gamma(shape=6.0,rate=1.0)"
/// - Predefined library HRFs
///
/// # Arguments
/// * `spec` - HRF specification string
/// * `lag` - Optional temporal lag in seconds
/// * `block_width` - Optional block width in seconds
/// * `normalize` - Whether to normalize to unit peak
///
/// # Examples
/// ```no_run
/// use fmri_hrf::hrf::generators::gen_hrf;
///
/// // From registry
/// let hrf1 = gen_hrf("spm", 0.0, 0.0, false).unwrap();
///
/// // With parameters
/// let hrf2 = gen_hrf("gamma(shape=8.0,rate=1.0)", 0.0, 0.0, false).unwrap();
///
/// // With decorators
/// let hrf3 = gen_hrf("spm", 2.0, 1.0, true).unwrap();
/// ```
pub fn gen_hrf(spec: &str, lag: f64, block_width: f64, normalize: bool) -> Result<Box<dyn Hrf>> {
    // Parse the specification
    let base_hrf = if spec.contains('(') {
        // Parse function call syntax
        parse_hrf_spec(spec)?
    } else {
        // Try registry first
        if let Ok(hrf) = get_hrf(spec) {
            hrf
        } else if let Some(hrf) = get_predefined_hrf(spec) {
            hrf
        } else {
            return Err(HrfError::InvalidInput(format!(
                "Unknown HRF specification: {}",
                spec
            )));
        }
    };

    // Apply decorators
    let mut hrf = base_hrf;

    if lag != 0.0 {
        hrf = Box::new(LaggedHrf { inner: hrf, lag });
    }

    if block_width > 0.0 {
        hrf = Box::new(BlockedHrf {
            inner: hrf,
            width: block_width,
        });
    }

    if normalize {
        hrf = crate::hrf::normalization::normalize_legacy(hrf);
    }

    Ok(hrf)
}

/// Build a basis by lagging independent instances produced by a factory.
pub fn gen_hrf_lagged(factory: impl Fn() -> Box<dyn Hrf>, lags: &[f64]) -> Box<dyn Hrf> {
    bind_hrfs(lags.iter().map(|&lag| lag_hrf(factory(), lag)).collect())
}

/// Build a basis from several sustained-stimulus durations, without normalization.
pub fn gen_hrf_blocked(factory: impl Fn() -> Box<dyn Hrf>, widths: &[f64]) -> Box<dyn Hrf> {
    bind_hrfs(
        widths
            .iter()
            .map(|&width| block_hrf(factory(), width))
            .collect(),
    )
}

/// Parse HRF specification string with function call syntax.
///
/// Supports syntax like "gamma(shape=6.0,rate=1.0)" or "bspline(N=5,degree=3)".
fn parse_hrf_spec(spec: &str) -> Result<Box<dyn Hrf>> {
    // Find function name and parameters
    let paren_pos = spec
        .find('(')
        .ok_or_else(|| HrfError::InvalidInput("Invalid HRF specification format".into()))?;

    let func_name = &spec[..paren_pos].to_lowercase();
    let params_str = &spec[paren_pos + 1..spec.len() - 1];

    // Parse parameters with enhanced parsing
    let params = parse_params(params_str)?;

    // Create HRF based on function name
    match func_name.as_str() {
        "gamma" => {
            let shape = params.get("shape").and_then(|v| v.as_f64()).unwrap_or(6.0);
            let rate = params.get("rate").and_then(|v| v.as_f64()).unwrap_or(1.0);
            normalize_from_spec(gamma_hrf(shape, rate, Some(false)), params.get("normalize"))
        }
        "gaussian" => {
            let mean = params.get("mean").and_then(|v| v.as_f64()).unwrap_or(6.0);
            let sd = params.get("sd").and_then(|v| v.as_f64()).unwrap_or(2.0);
            normalize_from_spec(gaussian_hrf(mean, sd, Some(false)), params.get("normalize"))
        }
        "bspline" => {
            let n = params.get("n").and_then(|v| v.as_usize()).unwrap_or(5);
            let degree = params.get("degree").and_then(|v| v.as_usize());
            let span = params.get("span").and_then(|v| v.as_f64());
            Ok(bspline_hrf(n, degree, span))
        }
        "fourier" => {
            let n = params.get("n").and_then(|v| v.as_usize()).unwrap_or(5);
            let span = params.get("span").and_then(|v| v.as_f64());
            Ok(fourier_hrf(n, span))
        }
        "fir" => {
            let n = params.get("n").and_then(|v| v.as_usize()).unwrap_or(10);
            let span = params.get("span").and_then(|v| v.as_f64());
            Ok(fir_hrf(n, span))
        }
        "mexhat" => {
            let sigma = params.get("sigma").and_then(|v| v.as_f64()).unwrap_or(2.0);
            let center = params.get("center").and_then(|v| v.as_f64()).unwrap_or(6.0);
            Ok(mexhat_hrf(sigma, center))
        }
        "sine" => {
            let n_basis = params
                .get("n_basis")
                .or_else(|| params.get("N"))
                .and_then(|v| v.as_usize())
                .unwrap_or(5);
            let span = params.get("span").and_then(|v| v.as_f64()).unwrap_or(24.0);
            Ok(sine_basis_hrf(n_basis, span))
        }
        "half_cosine" | "halfcos" => {
            if params.contains_key("width") || params.contains_key("center") {
                let width = params
                    .get("width")
                    .and_then(ParamValue::as_f64)
                    .unwrap_or(10.0);
                let center = params
                    .get("center")
                    .and_then(ParamValue::as_f64)
                    .unwrap_or(5.0);
                Ok(half_cosine_hrf(width, center))
            } else {
                let value = |key: &str, default: f64| {
                    params
                        .get(key)
                        .and_then(ParamValue::as_f64)
                        .unwrap_or(default)
                };
                Ok(half_cosine_hrf_segments(
                    value("h1", 1.0),
                    value("h2", 5.0),
                    value("h3", 7.0),
                    value("h4", 7.0),
                    value("f1", 0.0),
                    value("f2", 0.0),
                ))
            }
        }
        "inv_logit" | "inverse_logit" => {
            let mu1 = params.get("mu1").and_then(|v| v.as_f64()).unwrap_or(6.0);
            let s1 = params.get("s1").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let mu2 = params.get("mu2").and_then(|v| v.as_f64()).unwrap_or(16.0);
            let s2 = params.get("s2").and_then(|v| v.as_f64()).unwrap_or(1.0);
            let lag = params.get("lag").and_then(|v| v.as_f64()).unwrap_or(0.0);
            Ok(inv_logit_hrf(mu1, s1, mu2, s2, lag))
        }
        "lwu" => {
            let tau = params.get("tau").and_then(|v| v.as_f64());
            let sigma = params.get("sigma").and_then(|v| v.as_f64());
            let rho = params.get("rho").and_then(|v| v.as_f64());
            let normalize = params.get("normalize").map(|v| match v {
                ParamValue::String(s) => s.as_str(),
                ParamValue::Boolean(true) => "height",
                _ => "none",
            });
            lwu_hrf(tau, sigma, rho, normalize)
        }
        "daguerre" | "dag" => {
            let n = params.get("n").and_then(|v| v.as_usize());
            let span = params.get("span").and_then(|v| v.as_f64());
            let scale = params.get("scale").and_then(|v| v.as_f64());
            Ok(daguerre_generator(n, span, scale).build())
        }
        _ => Err(HrfError::InvalidInput(format!(
            "Unknown HRF function: {}",
            func_name
        ))),
    }
}

/// Represents a parsed parameter value.
#[derive(Debug, Clone)]
enum ParamValue {
    Float(f64),
    Integer(i64),
    Boolean(bool),
    String(String),
}

impl ParamValue {
    /// Try to get as f64.
    fn as_f64(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            ParamValue::Integer(v) => Some(*v as f64),
            ParamValue::Boolean(v) => Some(if *v { 1.0 } else { 0.0 }),
            ParamValue::String(_) => None,
        }
    }

    /// Try to get as usize.
    fn as_usize(&self) -> Option<usize> {
        match self {
            ParamValue::Integer(v) if *v >= 0 => Some(*v as usize),
            ParamValue::Float(v) if v.fract() == 0.0 && *v >= 0.0 => Some(*v as usize),
            _ => None,
        }
    }

    /// Try to get as bool.
    #[cfg(test)]
    fn as_bool(&self) -> Option<bool> {
        match self {
            ParamValue::Boolean(v) => Some(*v),
            ParamValue::Integer(v) => Some(*v != 0),
            ParamValue::Float(v) => Some(*v != 0.0),
            ParamValue::String(s) => match s.to_lowercase().as_str() {
                "true" | "yes" | "1" => Some(true),
                "false" | "no" | "0" => Some(false),
                _ => None,
            },
        }
    }

    /// Try to get as string.
    #[allow(dead_code)]
    fn as_string(&self) -> String {
        match self {
            ParamValue::String(s) => s.clone(),
            ParamValue::Float(v) => v.to_string(),
            ParamValue::Integer(v) => v.to_string(),
            ParamValue::Boolean(v) => v.to_string(),
        }
    }
}

/// Parse a parameter value from string.
fn parse_param_value(value_str: &str) -> ParamValue {
    let trimmed = value_str.trim();

    // Try boolean
    match trimmed.to_lowercase().as_str() {
        "true" | "yes" => return ParamValue::Boolean(true),
        "false" | "no" => return ParamValue::Boolean(false),
        _ => {}
    }

    // Try integer
    if let Ok(v) = trimmed.parse::<i64>() {
        return ParamValue::Integer(v);
    }

    // Try float
    if let Ok(v) = trimmed.parse::<f64>() {
        return ParamValue::Float(v);
    }

    // Remove quotes if present and return as string
    let unquoted = if (trimmed.starts_with('"') && trimmed.ends_with('"'))
        || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
    {
        &trimmed[1..trimmed.len() - 1]
    } else {
        trimmed
    };

    ParamValue::String(unquoted.to_string())
}

/// Parse parameter string into HashMap.
fn parse_params(params_str: &str) -> Result<HashMap<String, ParamValue>> {
    let mut params = HashMap::new();

    if params_str.is_empty() {
        return Ok(params);
    }

    // Split by commas, but respect quoted strings
    let mut current_param = String::new();
    let mut in_quotes = false;
    let mut quote_char = ' ';

    for ch in params_str.chars() {
        match ch {
            '"' | '\'' if !in_quotes => {
                in_quotes = true;
                quote_char = ch;
                current_param.push(ch);
            }
            ch if ch == quote_char && in_quotes => {
                in_quotes = false;
                current_param.push(ch);
            }
            ',' if !in_quotes => {
                // Process the completed parameter
                if !current_param.trim().is_empty() {
                    let parts: Vec<&str> = current_param.trim().split('=').collect();
                    if parts.len() != 2 {
                        return Err(HrfError::InvalidInput(format!(
                            "Invalid parameter format: {}",
                            current_param
                        )));
                    }

                    let key = parts[0].trim().to_lowercase();
                    let value = parse_param_value(parts[1]);
                    params.insert(key, value);
                }
                current_param.clear();
            }
            _ => {
                current_param.push(ch);
            }
        }
    }

    // Process the last parameter
    if !current_param.trim().is_empty() {
        let parts: Vec<&str> = current_param.trim().split('=').collect();
        if parts.len() != 2 {
            return Err(HrfError::InvalidInput(format!(
                "Invalid parameter format: {}",
                current_param
            )));
        }

        let key = parts[0].trim().to_lowercase();
        let value = parse_param_value(parts[1]);
        params.insert(key, value);
    }

    Ok(params)
}

/// Parse parameter string into f64 HashMap (for backward compatibility).
#[allow(dead_code)]
fn parse_params_numeric(params_str: &str) -> Result<HashMap<String, f64>> {
    let params = parse_params(params_str)?;
    let mut numeric_params = HashMap::new();

    for (key, value) in params {
        if let Some(v) = value.as_f64() {
            numeric_params.insert(key, v);
        }
    }

    Ok(numeric_params)
}

/// Create an HRF from a string specification.
///
/// This is the most flexible way to create HRFs, supporting both predefined
/// HRFs from the registry and parameterized HRF functions. The string format
/// is `function_name(param1=value1, param2=value2, ...)`.
///
/// # String Format
/// - Simple names: `"spm_canonical"`, `"gamma"`, `"spm"`
/// - With parameters: `"gamma(shape=8.0, rate=1.0)"`
/// - Case insensitive: `"GAMMA"` or `"gamma"`
/// - Various parameter types: `"fir(N=10, span=20.0)"`
///
/// # Available Functions
/// - **Predefined**: spm_canonical, spm_with_derivatives, glover, etc.
/// - **gamma**: Gamma function with shape and rate parameters
/// - **bspline**: B-spline basis (N basis functions)
/// - **fir**: Finite impulse response basis
/// - **fourier**: Fourier basis (sine/cosine)
/// - **gaussian**: Gaussian-shaped HRF
/// - **half_cosine**: Half-cosine window
/// - **daguerre**: Laguerre-type orthogonal polynomial basis for HRF modeling
///
/// # Examples
///
/// ## Predefined HRFs
/// ```
/// use fmri_hrf::hrf::make_hrf;
///
/// // SPM canonical HRF
/// let spm = make_hrf("spm_canonical").unwrap();
///
/// // Gamma HRF (default parameters)
/// let gamma = make_hrf("gamma").unwrap();
/// ```
///
/// ## Parameterized HRFs
/// ```
/// use fmri_hrf::hrf::make_hrf;
///
/// // Gamma HRF with custom parameters
/// let gamma = make_hrf("gamma(shape=8.0, rate=1.0)").unwrap();
///
/// // B-spline basis with 7 functions
/// let bspline = make_hrf("bspline(N=7, degree=3, span=30.0)").unwrap();
///
/// // FIR basis for deconvolution
/// let fir = make_hrf("fir(N=20, span=40.0)").unwrap();
/// ```
///
/// ## Complex Specifications
/// ```
/// use fmri_hrf::hrf::make_hrf;
///
/// // Normalized LWU HRF
/// let lwu = make_hrf("lwu(tau=2.0, sigma=0.3, rho=0.8, normalize=height)").unwrap();
///
/// // Boolean parameters
/// let gamma_norm = make_hrf("gamma(shape=6.0, rate=1.0, normalize=true)").unwrap();
///
/// // Multiple parameters
/// let multi = make_hrf("fir(N=15, span=30.0)").unwrap();
/// ```
pub fn make_hrf(spec: &str) -> Result<Box<dyn Hrf>> {
    gen_hrf(spec, 0.0, 0.0, false)
}

/// Create an HRF from a parameter map.
pub fn make_hrf_from_params(params: &HashMap<String, String>) -> Result<Box<dyn Hrf>> {
    let hrf_type = params
        .get("type")
        .ok_or_else(|| HrfError::InvalidInput("Missing 'type' parameter".into()))?;

    // Convert string params to numeric where needed
    let mut numeric_params = HashMap::new();
    for (key, value) in params {
        if key != "type" {
            if let Ok(num) = value.parse::<f64>() {
                numeric_params.insert(key.clone(), num);
            }
        }
    }

    // Build specification string
    let param_strs: Vec<String> = numeric_params
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect();

    let spec = if param_strs.is_empty() {
        hrf_type.clone()
    } else {
        format!("{}({})", hrf_type, param_strs.join(","))
    };

    make_hrf(&spec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    #[test]
    fn test_hrf_generator_basic() {
        let hrf = gamma_generator(6.0, 1.0).build();
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 1]);
    }

    #[test]
    fn test_hrf_with_lag() {
        let hrf = gamma_generator(6.0, 1.0).with_lag(2.0).build();

        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        // Should be zero at t=0 and t=1 due to lag
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 0]], 0.0);
    }

    #[test]
    fn test_hrf_normalized() {
        let hrf = gamma_generator(6.0, 1.0).normalized().build();

        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        let max_val = result.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_combined_hrfs() {
        let hrf1 = gamma_hrf(6.0, 1.0, Some(true));
        let hrf2 = gaussian_hrf(8.0, 2.0, Some(true));

        let combined = bind_hrfs(vec![hrf1, hrf2]);
        assert_eq!(combined.nbasis(), 2);

        let t = Array1::linspace(0.0, 20.0, 21);
        let result = combined.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 2]);
    }

    #[test]
    fn test_hrf_from_basis() {
        let basis = bspline_hrf(3, Some(2), Some(10.0));
        let coefficients = vec![0.5, 1.0, 0.5];

        let hrf = hrf_from_basis(basis, coefficients).unwrap();
        let t = Array1::linspace(0.0, 10.0, 11);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[11, 1]);
    }

    #[test]
    fn test_gen_hrf_from_string() {
        // Test registry lookup
        let hrf1 = gen_hrf("spm", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf1.nbasis(), 1);

        // Test with parameters
        let hrf2 = gen_hrf("gamma(shape=8.0,rate=1.0)", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf2.nbasis(), 1);

        // Test with decorators
        let hrf3 = gen_hrf("spm", 2.0, 0.0, false).unwrap();
        assert_eq!(hrf3.name(), "lagged_hrf");

        // Test with normalization
        let hrf4 = gen_hrf("gamma(shape=6.0,rate=1.0)", 0.0, 0.0, true).unwrap();
        let t = Array1::linspace(0.0, 20.0, 1001);
        let values = hrf4.evaluate(&t).unwrap();
        let max_val = values.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parse_hrf_spec() {
        // Test gamma with parameters
        let hrf = parse_hrf_spec("gamma(shape=8.0,rate=1.5)").unwrap();
        assert_eq!(hrf.nbasis(), 1);

        // Test bspline
        let hrf = parse_hrf_spec("bspline(n=7,degree=3)").unwrap();
        assert_eq!(hrf.nbasis(), 7);

        // Test fourier
        let hrf = parse_hrf_spec("fourier(n=5)").unwrap();
        assert_eq!(hrf.nbasis(), 5);
    }

    #[test]
    fn test_parse_params() {
        // Test numeric parameters
        let params = parse_params("shape=6.0,rate=1.0").unwrap();
        assert_eq!(params.get("shape").and_then(|v| v.as_f64()), Some(6.0));
        assert_eq!(params.get("rate").and_then(|v| v.as_f64()), Some(1.0));

        // Test integer parameters
        let params = parse_params("n=7,degree=3").unwrap();
        assert_eq!(params.get("n").and_then(|v| v.as_usize()), Some(7));
        assert_eq!(params.get("degree").and_then(|v| v.as_usize()), Some(3));

        // Test boolean parameters
        let params = parse_params("normalize=true,summate=false").unwrap();
        assert_eq!(
            params.get("normalize").and_then(|v| v.as_bool()),
            Some(true)
        );
        assert_eq!(params.get("summate").and_then(|v| v.as_bool()), Some(false));

        // Test string parameters
        let params = parse_params("type=gamma,name=\"my hrf\"").unwrap();
        assert!(matches!(params.get("type"), Some(ParamValue::String(s)) if s == "gamma"));
        assert!(matches!(params.get("name"), Some(ParamValue::String(s)) if s == "my hrf"));

        // Test case insensitive keys
        let params = parse_params("N=5,Shape=6.0").unwrap();
        assert_eq!(params.get("n").and_then(|v| v.as_usize()), Some(5));
        assert_eq!(params.get("shape").and_then(|v| v.as_f64()), Some(6.0));

        // Test empty params
        let params = parse_params("").unwrap();
        assert!(params.is_empty());

        // Test invalid format
        assert!(parse_params("invalid").is_err());
    }

    #[test]
    fn test_enhanced_string_parsing() {
        // Test with integer N parameter
        let hrf = gen_hrf("bspline(N=7,degree=3)", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf.nbasis(), 7);

        // Test with boolean normalize
        let hrf = gen_hrf("gamma(shape=8.0,rate=1.0,normalize=true)", 0.0, 0.0, false).unwrap();
        let t = Array1::linspace(0.0, 20.0, 1001);
        let values = hrf.evaluate(&t).unwrap();
        let max_val = values.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);

        // Test case insensitive function names
        let hrf1 = gen_hrf("GAMMA(shape=6.0,rate=1.0)", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf1.nbasis(), 1);

        // Test aliases
        let hrf2 = gen_hrf("halfcos(width=10.0)", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf2.nbasis(), 1);
    }

    #[test]
    fn test_make_hrf() {
        // Simple specification
        let hrf1 = make_hrf("spm").unwrap();
        assert_eq!(hrf1.nbasis(), 1);

        // With parameters
        let hrf2 = make_hrf("gamma(shape=8.0,rate=1.0)").unwrap();
        assert_eq!(hrf2.nbasis(), 1);
    }

    #[test]
    fn test_make_hrf_from_params() {
        let mut params = HashMap::new();
        params.insert("type".to_string(), "gamma".to_string());
        params.insert("shape".to_string(), "8.0".to_string());
        params.insert("rate".to_string(), "1.5".to_string());

        let hrf = make_hrf_from_params(&params).unwrap();
        assert_eq!(hrf.nbasis(), 1);
    }

    #[test]
    fn test_decorators() {
        // Test lag decorator
        let hrf = gamma_hrf(6.0, 1.0, Some(true));
        let lagged = lag_hrf(hrf, 2.0);
        assert_eq!(lagged.name(), "lagged_hrf");

        // Test block decorator
        let hrf = gamma_hrf(6.0, 1.0, Some(true));
        let blocked = block_hrf(hrf, 1.0);
        assert_eq!(blocked.name(), "blocked_hrf");

        // Test normalize decorator
        let hrf = gamma_hrf(6.0, 1.0, Some(false));
        let normalized = normalize_hrf(hrf);
        assert_eq!(normalized.name(), "normalized_hrf");

        // Test normalized values
        let t = Array1::linspace(0.0, 20.0, 1001);
        let values = normalized.evaluate(&t).unwrap();
        let max_val = values.fold(f64::NEG_INFINITY, |acc, &val| acc.max(val.abs()));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_daguerre_generator() {
        // Test default parameters (R defaults: n_basis=3, scale=4)
        let hrf1 = daguerre_generator(None, None, None).build();
        assert_eq!(hrf1.name(), "daguerre");
        assert_eq!(hrf1.nbasis(), 3);
        assert_eq!(hrf1.span(), 30.0);

        // Test custom parameters
        let hrf2 = daguerre_generator(Some(10), Some(40.0), Some(2.0)).build();
        assert_eq!(hrf2.nbasis(), 10);
        assert_eq!(hrf2.span(), 40.0);

        // Test evaluation
        let t = Array1::linspace(0.0, 30.0, 100);
        let values = hrf1.evaluate(&t).unwrap();
        assert_eq!(values.shape(), &[100, 3]);
    }

    #[test]
    fn test_daguerre_from_string() {
        let hrf1 = gen_hrf("daguerre", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf1.nbasis(), 3);

        let hrf2 = gen_hrf("daguerre(n=10,span=40.0)", 0.0, 0.0, false).unwrap();
        assert_eq!(hrf2.nbasis(), 10);
        assert_eq!(hrf2.span(), 40.0);
    }

    #[test]
    fn test_daguerre_laguerre_basis() {
        // scale=1 for simple verification against known Laguerre values
        let hrf = daguerre_hrf(3, 24.0, 1.0);
        let t = Array1::from_vec(vec![0.0, 1.0, 2.0, 4.0, 8.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[5, 3]);

        // At t=0 (x=0): exp(0)=1 is the max for column 0, so normalized value = 1.0
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);

        // Laguerre basis[0] (exp(-x/2)) should be monotonically decreasing
        for i in 1..5 {
            assert!(
                result[[i, 0]] <= result[[i - 1, 0]] + 1e-10,
                "Column 0 should be monotonically decreasing"
            );
        }

        // All values should be finite
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_daguerre_with_scale() {
        // R default scale=4
        let hrf = daguerre_hrf(3, 30.0, 4.0);
        let t = Array1::linspace(0.0, 30.0, 31);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[31, 3]);
        // Should be well-defined, no NaN or Inf
        assert!(result.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_daguerre_single_basis() {
        // Edge case: single basis function (just exp(-x/2))
        let hrf = daguerre_hrf(1, 20.0, 1.0);
        let t = Array1::from_vec(vec![0.0, 2.0, 4.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[3, 1]);
        // At t=0, exp(0)=1 is the max, so normalized = 1.0
        assert!((result[[0, 0]] - 1.0).abs() < 1e-10);
        // At t=2, exp(-1) / 1.0 ~ 0.3679
        assert!((result[[1, 0]] - (-1.0_f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_daguerre_scale_parameter_in_params() {
        let hrf = daguerre_hrf(3, 30.0, 4.0);
        let params = hrf.params();
        assert!(params
            .iter()
            .any(|(name, val)| *name == "scale" && (*val - 4.0).abs() < 1e-10));
    }
}
fn normalize_from_spec(hrf: Box<dyn Hrf>, value: Option<&ParamValue>) -> Result<Box<dyn Hrf>> {
    use crate::hrf::normalization::{normalize_hrf_with, Normalization};
    let mode = match value {
        None | Some(ParamValue::Boolean(false)) => Normalization::None,
        Some(ParamValue::Boolean(true)) => Normalization::UnitPeak,
        Some(ParamValue::String(s)) => match s.as_str() {
            "none" => Normalization::None,
            "height" | "unit_peak" => Normalization::UnitPeak,
            "area" | "unit_integral" => Normalization::UnitIntegral,
            "spm" => Normalization::Spm,
            "unit_peak_per_basis" => Normalization::UnitPeakPerBasis,
            _ => {
                return Err(HrfError::InvalidParameters(format!(
                    "unknown normalization: {s}"
                )))
            }
        },
        _ => {
            return Err(HrfError::InvalidParameters(
                "normalization must be a boolean or mode name".into(),
            ))
        }
    };
    normalize_hrf_with(hrf, mode)
}
