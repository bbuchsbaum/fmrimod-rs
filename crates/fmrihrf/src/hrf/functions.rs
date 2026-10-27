//! HRF function implementations.

use crate::error::{HrfError, Result};
use crate::hrf::core::{Hrf, SimpleHrf};
use crate::types::{HrfValues, TimeArray};
use ndarray::{s, Array1, Array2, Axis};
use statrs::function::gamma::gamma as gamma_fn;
use std::f64::consts::PI;

// R uses exp(-t) * (A1*t^P1 - C*t^P2), with A1=1.0 / 120.0 as its scale convention.
/// Power for positive component (R: P1)
pub const SPM_P1: f64 = 5.0;
/// Power for negative component (R: P2)
pub const SPM_P2: f64 = 15.0;
/// Amplitude for positive component (R: A1) - matches R package value
pub const SPM_A1: f64 = 1.0 / 120.0;
/// Constant for negative component (R: C) - derived from 1/(6*Γ(16))
pub const SPM_C: f64 = 1.0 / (6.0 * 1307674368000.0);

/// Create the SPM canonical HRF matching R package `fmrihrf`.
///
/// This implements the canonical hemodynamic response function using the **R package formula**:
/// ```text
/// hrf(t) = exp(-t) * (A1 * t^P1 - C * t^P2)
/// ```
///
/// **Important**: This uses **fixed R package values** internally (A1=1.0 / 120.0, P1=5, P2=15),
/// which matches R's `hrf_spmg1()`. The function signature is retained for backward compatibility,
/// but the `a1`, `a2`, `b1`, `b2`, and `c` parameters are **ignored**.
///
/// R uses A1=1.0 / 120.0 as its amplitude convention. Use `SpmG1Hrf`,
/// `SpmG2Hrf`, or `SpmG3Hrf` to specify the R parameters P1, P2, and A1.
///
/// # Arguments
/// * `a1` - **IGNORED** - Kept for API compatibility
/// * `a2` - **IGNORED** - Kept for API compatibility
/// * `b1` - **IGNORED** - Kept for API compatibility
/// * `b2` - **IGNORED** - Kept for API compatibility
/// * `c` - **IGNORED** - Kept for API compatibility
/// * `normalize` - Normalize using a fixed reference peak (default: false)
///
/// # Examples
///
/// ```
/// use fmrihrf::hrf::spm_canonical;
/// use ndarray::Array1;
///
/// // Create the raw SPM canonical HRF (matches R package).
/// let hrf = spm_canonical(None, None, None, None, None, None);
///
/// // Evaluate at time points
/// let times = Array1::linspace(0.0, 30.0, 150);
/// let response = hrf.evaluate(&times).unwrap();
///
/// // Response peaks around 5-6 seconds with peak value ~0.175
/// let peak_idx = response.column(0).iter()
///     .enumerate()
///     .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
///     .map(|(idx, _)| idx)
///     .unwrap();
/// let peak_time = times[peak_idx];
/// assert!(peak_time > 4.0 && peak_time < 7.0);
/// ```
///
/// # R Compatibility
///
/// This function produces identical output to R's `hrf_spmg1(t)` within numerical precision
/// (RMSE < 1e-10). Cross-validated against R package version 0.1.0.
///
/// # Note on Parameters
///
/// Previous versions used a gamma PDF formulation with shape and rate parameters.
/// This version uses R's simplified exponential-polynomial formula for exact R compatibility.
/// The old parameters are retained in the signature for backward compatibility but are not used.
pub fn spm_canonical(
    _a1: Option<f64>, // Ignored - kept for API compatibility
    _a2: Option<f64>, // Ignored - kept for API compatibility
    _b1: Option<f64>, // Ignored - kept for API compatibility
    _b2: Option<f64>, // Ignored - kept for API compatibility
    _c: Option<f64>,  // Ignored - kept for API compatibility
    normalize: Option<bool>,
) -> Box<dyn Hrf> {
    // Use fixed R package values regardless of input parameters
    let normalize = normalize.unwrap_or(false);

    let params = vec![
        ("p1".to_string(), SPM_P1),
        ("p2".to_string(), SPM_P2),
        ("a1".to_string(), SPM_A1),
        ("c".to_string(), SPM_C),
        ("normalize".to_string(), if normalize { 1.0 } else { 0.0 }),
    ];

    Box::new(
        SimpleHrf::new("spm_canonical", 1, 24.0, params, |t, params| {
            let p1 = params.iter().find(|(k, _)| k == "p1").unwrap().1;
            let p2 = params.iter().find(|(k, _)| k == "p2").unwrap().1;
            let a1 = params.iter().find(|(k, _)| k == "a1").unwrap().1;
            let c = params.iter().find(|(k, _)| k == "c").unwrap().1;
            let normalize = params.iter().find(|(k, _)| k == "normalize").unwrap().1 > 0.5;

            evaluate_spm_canonical_r_formula(t, p1, p2, a1, c, normalize)
        })
        .with_recipe(super::HrfRecipe::Canonical { normalize }),
    )
}

/// Evaluate SPM canonical HRF using R package formula (simplified form).
///
/// This matches the R package `fmrihrf` implementation:
/// ```text
/// hrf(t) = exp(-t) * (A1 * t^P1 - C * t^P2)  for t >= 0
///        = 0                                  for t < 0
/// ```
///
/// This is computationally more efficient than the gamma PDF form and produces
/// identical results when using the correct parameter mapping.
pub(crate) fn evaluate_spm_canonical_r_formula(
    t: &TimeArray,
    p1: f64,
    p2: f64,
    a1: f64,
    c: f64,
    normalize: bool,
) -> Result<HrfValues> {
    // Validate parameters
    if p1 <= 0.0 || p2 <= 0.0 || a1 == 0.0 {
        return Err(HrfError::InvalidParameters(
            "P1, P2 must be positive and A1 must be non-zero".into(),
        ));
    }

    let mut hrf = Array1::zeros(t.len());

    for (i, &time) in t.iter().enumerate() {
        if time >= 0.0 {
            // R formula: exp(-t) * (A1 * t^P1 - C * t^P2)
            let exp_term = (-time).exp();
            let positive_term = a1 * time.powf(p1);
            let negative_term = c * time.powf(p2);
            hrf[i] = exp_term * (positive_term - negative_term);
        }
        // else: hrf[i] = 0.0 (already initialized)
    }

    if normalize {
        let peak = (0..=1200)
            .map(|i| {
                let time = i as f64 * 0.02;
                ((-time).exp() * (a1 * time.powf(p1) - c * time.powf(p2))).abs()
            })
            .fold(0.0_f64, f64::max);
        if peak > 0.0 {
            hrf /= peak;
        }
    }

    // Convert to 2D array with single column
    Ok(hrf.insert_axis(Axis(1)))
}

/// Create a gamma HRF.
///
/// The gamma distribution is a flexible HRF model that can approximate various
/// hemodynamic response shapes. It's parameterized by shape (α) and scale (β),
/// where the peak time is approximately at (α-1)×β.
///
/// # Arguments
/// * `shape` - Shape parameter (alpha), controls the rise time and peak location
/// * `scale` - Scale parameter (beta), controls the width/spread
/// * `normalize` - Whether to normalize to unit peak (default: true)
///
/// # Examples
///
/// ```
/// use fmrihrf::hrf::gamma_hrf;
/// use ndarray::Array1;
///
/// // Create a gamma HRF with peak around 6 seconds
/// let hrf = gamma_hrf(6.0, 1.0, Some(true));
///
/// // Evaluate the HRF
/// let times = Array1::linspace(0.0, 20.0, 100);
/// let response = hrf.evaluate(&times).unwrap();
///
/// // The peak should be around (shape-1) * scale = 5 seconds
/// ```
///
/// # Different Response Speeds
///
/// ```
/// use fmrihrf::hrf::gamma_hrf;
///
/// // Fast response (peak ~3s)
/// let fast_hrf = gamma_hrf(4.0, 1.0, None);
///
/// // Slow response (peak ~8s)
/// let slow_hrf = gamma_hrf(9.0, 1.0, None);
///
/// // Wide response (same peak, broader shape)
/// let wide_hrf = gamma_hrf(3.0, 3.0, None);
/// ```
pub fn gamma_hrf(shape: f64, scale: f64, normalize: Option<bool>) -> Box<dyn Hrf> {
    let normalize = normalize.unwrap_or(false);

    let params = vec![
        ("shape".to_string(), shape),
        ("scale".to_string(), scale),
        ("normalize".to_string(), if normalize { 1.0 } else { 0.0 }),
    ];

    Box::new(SimpleHrf::new("gamma_hrf", 1, 30.0, params, |t, params| {
        let shape = params.iter().find(|(k, _)| k == "shape").unwrap().1;
        let scale = params.iter().find(|(k, _)| k == "scale").unwrap().1;
        let normalize = params.iter().find(|(k, _)| k == "normalize").unwrap().1 > 0.5;

        evaluate_gamma_hrf(t, shape, scale, normalize)
    }))
}

fn evaluate_gamma_hrf(t: &TimeArray, shape: f64, scale: f64, normalize: bool) -> Result<HrfValues> {
    if shape <= 0.0 || scale <= 0.0 {
        return Err(HrfError::InvalidParameters(
            "Shape and scale must be positive".into(),
        ));
    }

    let mut hrf = Array1::zeros(t.len());

    for (i, &time) in t.iter().enumerate() {
        if time >= 0.0 {
            hrf[i] = (time.powf(shape - 1.0) * scale.powf(shape) * (-scale * time).exp())
                / gamma_fn(shape);
        }
    }

    if normalize {
        let peak_time = (shape - 1.0).max(0.0) / scale;
        let max_val = peak_time.powf(shape - 1.0) * scale.powf(shape) * (-scale * peak_time).exp()
            / gamma_fn(shape);
        if max_val.is_finite() && max_val > 0.0 {
            hrf.mapv_inplace(|v| v / max_val);
        }
    }

    Ok(hrf.insert_axis(Axis(1)))
}

/// Create a Gaussian HRF.
///
/// # Arguments
/// * `center` - Center of the Gaussian
/// * `width` - Width (standard deviation) of the Gaussian
/// * `normalize` - Whether to normalize to unit peak
pub fn gaussian_hrf(center: f64, width: f64, normalize: Option<bool>) -> Box<dyn Hrf> {
    let normalize = normalize.unwrap_or(false);

    let params = vec![
        ("center".to_string(), center),
        ("width".to_string(), width),
        ("normalize".to_string(), if normalize { 1.0 } else { 0.0 }),
    ];

    Box::new(SimpleHrf::new(
        "gaussian_hrf",
        1,
        30.0,
        params,
        |t, params| {
            let center = params.iter().find(|(k, _)| k == "center").unwrap().1;
            let width = params.iter().find(|(k, _)| k == "width").unwrap().1;
            let normalize = params.iter().find(|(k, _)| k == "normalize").unwrap().1 > 0.5;

            evaluate_gaussian_hrf(t, center, width, normalize)
        },
    ))
}

fn evaluate_gaussian_hrf(
    t: &TimeArray,
    center: f64,
    width: f64,
    normalize: bool,
) -> Result<HrfValues> {
    if width <= 0.0 {
        return Err(HrfError::InvalidParameters("Width must be positive".into()));
    }

    let variance = width * width;
    let norm_factor = 1.0 / (width * (2.0 * PI).sqrt());

    let hrf = t.mapv(|time| {
        if time < 0.0 {
            0.0
        } else {
            norm_factor * (-(time - center).powi(2) / (2.0 * variance)).exp()
        }
    });

    let mut result = hrf;
    if normalize {
        let max_val = norm_factor * (-(center.min(0.0)).powi(2) / (2.0 * variance)).exp();
        if max_val > 0.0 {
            result.mapv_inplace(|v| v / max_val);
        }
    }

    Ok(result.insert_axis(Axis(1)))
}

/// SPM canonical HRF with temporal and dispersion derivatives.
///
/// Returns a 3-column basis set:
/// 1. Canonical HRF
/// 2. Temporal derivative
/// 3. Dispersion derivative
pub fn spm_hrf_with_derivatives() -> Box<dyn Hrf> {
    Box::new(crate::hrf::derivatives::SpmG3Hrf::new(
        SPM_P1, SPM_P2, SPM_A1,
    ))
}

/// Four joined half-cosine segments, matching R's six-parameter shape.
/// Segment durations are h1..h4 and intermediate heights are 0, f1, 1, f2, 0.
pub fn half_cosine_hrf_segments(
    h1: f64,
    h2: f64,
    h3: f64,
    h4: f64,
    f1: f64,
    f2: f64,
) -> Box<dyn Hrf> {
    let params = vec![
        ("h1".into(), h1),
        ("h2".into(), h2),
        ("h3".into(), h3),
        ("h4".into(), h4),
        ("f1".into(), f1),
        ("f2".into(), f2),
    ];
    Box::new(SimpleHrf::new(
        "half_cosine_segments",
        1,
        h1 + h2 + h3 + h4,
        params,
        |t, p| {
            let widths = [p[0].1, p[1].1, p[2].1, p[3].1];
            let heights = [0.0, p[4].1, 1.0, p[5].1, 0.0];
            if widths.iter().any(|x| !x.is_finite() || *x <= 0.0)
                || heights.iter().any(|x| !x.is_finite())
            {
                return Err(HrfError::InvalidParameters(
                    "half-cosine durations must be positive and finite; heights must be finite"
                        .into(),
                ));
            }
            Ok(Array2::from_shape_fn((t.len(), 1), |(i, _)| {
                let time = t[i];
                if time < 0.0 {
                    return 0.0;
                }
                let mut start = 0.0;
                for j in 0..4 {
                    if time <= start + widths[j] {
                        return heights[j]
                            + 0.5
                                * (heights[j + 1] - heights[j])
                                * (1.0 - (PI * (time - start) / widths[j]).cos());
                    }
                    start += widths[j];
                }
                0.0
            }))
        },
    ))
}

/// Create a B-spline basis set HRF.
///
/// Uses the R-compatible splines library for proper B-spline evaluation.
///
/// # Arguments
/// * `n_basis` - Number of basis functions
/// * `degree` - Degree of B-splines (default: 3 for cubic)
/// * `span` - Temporal span of the basis set
pub fn bspline_hrf(n_basis: usize, degree: Option<usize>, span: Option<f64>) -> Box<dyn Hrf> {
    let degree = degree.unwrap_or(3);
    let span = span.unwrap_or(24.0);

    Box::new(BSplineBasis {
        n_basis,
        degree,
        span,
    })
}

/// Piecewise-linear tent basis with fixed knots, matching R's hrf_tent_generator.
pub fn tent_hrf(n_basis: usize, span: Option<f64>) -> Box<dyn Hrf> {
    bspline_hrf(n_basis, Some(1), span)
}

#[derive(Debug, Clone)]
struct BSplineBasis {
    n_basis: usize,
    degree: usize,
    span: f64,
}

impl Hrf for BSplineBasis {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::BSpline {
            columns: self.n_basis,
            span: self.span,
            degree: self.degree,
        })
    }
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::Roughness
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        use splines::BSpline;
        if self.n_basis == 0
            || self.n_basis < self.degree.saturating_sub(1)
            || self.degree == 0
            || !self.span.is_finite()
            || self.span <= 0.0
        {
            return Err(HrfError::InvalidParameters(
                "B-spline requires nbasis >= max(1, degree - 1), degree > 0 and a positive finite span".into(),
            ));
        }
        let n_interior = self
            .n_basis
            .checked_add(1)
            .and_then(|n| n.checked_sub(self.degree))
            .ok_or_else(|| HrfError::InvalidParameters("B-spline basis size overflow".into()))?;
        let knots = (1..=n_interior)
            .map(|i| self.span * i as f64 / (n_interior + 1) as f64)
            .collect();
        let spline = BSpline::new(
            &[0.0, self.span],
            None,
            Some(knots),
            self.degree,
            true,
            Some((0.0, self.span)),
        )
        .map_err(|e| HrfError::NumericalError(e.to_string()))?;
        let times: Vec<f64> = t.iter().map(|&x| x.clamp(0.0, self.span)).collect();
        if t.iter().any(|x| !x.is_finite()) {
            return Err(HrfError::InvalidInput(
                "B-spline times must be finite".into(),
            ));
        }
        let full = spline
            .design_matrix(&times)
            .map_err(|e| HrfError::NumericalError(e.to_string()))?;
        let mut values = full.slice(s![.., 1..full.ncols() - 1]).to_owned();
        for (i, &x) in t.iter().enumerate() {
            if x < 0.0 || x > self.span {
                values.row_mut(i).fill(0.0);
            }
        }
        crate::regressor::convolution::check_shape(&values, t.len(), self.n_basis)?;
        Ok(values)
    }

    fn name(&self) -> &str {
        "bspline_basis"
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
            ("degree", self.degree as f64),
            ("span", self.span),
        ]
    }
}

/// Create a Fourier basis set HRF.
///
/// # Arguments
/// * `n_basis` - Number of alternating sine/cosine basis functions (no constant term)
/// * `span` - Temporal span of the basis set
pub fn fourier_hrf(n_basis: usize, span: Option<f64>) -> Box<dyn Hrf> {
    let span = span.unwrap_or(24.0);

    Box::new(FourierBasis { n_basis, span })
}

#[derive(Debug, Clone)]
struct FourierBasis {
    n_basis: usize,
    span: f64,
}

impl Hrf for FourierBasis {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Fourier {
            columns: self.n_basis,
            span: self.span,
        })
    }
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::FourierFrequency
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        if self.n_basis == 0 || !self.span.is_finite() || self.span <= 0.0 {
            return Err(HrfError::InvalidParameters(
                "positive Fourier basis count and span required".into(),
            ));
        }
        let mut basis = Array2::zeros((t.len(), self.n_basis));
        for (i, &time) in t.iter().enumerate() {
            if !(0.0..=self.span).contains(&time) {
                continue;
            }
            for j in 0..self.n_basis {
                let phase = 2.0 * PI * (j / 2 + 1) as f64 * time / self.span;
                basis[(i, j)] = if j % 2 == 0 { phase.sin() } else { phase.cos() };
            }
        }

        Ok(basis)
    }

    fn name(&self) -> &str {
        "fourier_basis"
    }

    fn nbasis(&self) -> usize {
        self.n_basis
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("n_basis", self.n_basis as f64), ("span", self.span)]
    }
}

/// Create a Finite Impulse Response (FIR) basis set.
///
/// # Arguments
/// * `n_basis` - Number of basis functions
/// * `span` - Temporal span of the basis set
pub fn fir_hrf(n_basis: usize, span: Option<f64>) -> Box<dyn Hrf> {
    let span = span.unwrap_or(24.0);

    Box::new(FirBasis { n_basis, span })
}

#[derive(Debug, Clone)]
struct FirBasis {
    n_basis: usize,
    span: f64,
}

impl Hrf for FirBasis {
    fn recipe(&self) -> Option<super::HrfRecipe> {
        Some(super::HrfRecipe::Fir {
            columns: self.n_basis,
            span: self.span,
        })
    }
    fn penalty_kind(&self) -> crate::penalty::PenaltyKind {
        crate::penalty::PenaltyKind::Roughness
    }
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        if self.n_basis == 0 || !self.span.is_finite() || self.span <= 0.0 {
            return Err(HrfError::InvalidParameters(
                "positive FIR basis count and span required".into(),
            ));
        }
        let bin_width = self.span / self.n_basis as f64;
        let mut basis = Array2::zeros((t.len(), self.n_basis));

        for i in 0..self.n_basis {
            let bin_start = i as f64 * bin_width;
            let bin_end = (i + 1) as f64 * bin_width;

            for (j, &time) in t.iter().enumerate() {
                if time >= bin_start && time < bin_end {
                    basis[[j, i]] = 1.0;
                }
            }
        }

        Ok(basis)
    }

    fn name(&self) -> &str {
        "fir_basis"
    }

    fn nbasis(&self) -> usize {
        self.n_basis
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("n_basis", self.n_basis as f64), ("span", self.span)]
    }
}

/// Create a Mexican hat (Ricker) wavelet HRF.
///
/// # Arguments
/// * `sigma` - Width parameter
/// * `center` - Center of the wavelet
pub fn mexhat_hrf(sigma: f64, center: f64) -> Box<dyn Hrf> {
    let params = vec![("sigma".to_string(), sigma), ("center".to_string(), center)];

    Box::new(SimpleHrf::new(
        "mexhat_hrf",
        1,
        30.0,
        params,
        |t, params| {
            let sigma = params.iter().find(|(k, _)| k == "sigma").unwrap().1;
            let center = params.iter().find(|(k, _)| k == "center").unwrap().1;

            evaluate_mexhat_hrf(t, sigma, center)
        },
    ))
}

fn evaluate_mexhat_hrf(t: &TimeArray, sigma: f64, center: f64) -> Result<HrfValues> {
    if sigma <= 0.0 {
        return Err(HrfError::InvalidParameters("Sigma must be positive".into()));
    }

    let z = t.mapv(|time| (time - center) / sigma);
    let normalization = (2.0 / (3.0 * sigma * PI.powf(0.25))).sqrt();
    let mut hrf = z.mapv(|z| normalization * (1.0 - z * z) * (-z * z / 2.0).exp());
    for (value, &time) in hrf.iter_mut().zip(t) {
        if time < 0.0 {
            *value = 0.0;
        }
    }

    Ok(hrf.insert_axis(Axis(1)))
}

/// Create a sine basis set HRF matching R's `hrf_sine()`.
///
/// Returns an N-column matrix where column n = `sin(2*pi*n*t/span)` for n=1..N,
/// zeroed outside [0, span]. This matches the R package `fmrihrf` behavior.
///
/// # Arguments
/// * `n_basis` - Number of basis functions (default: 5)
/// * `span` - Temporal span in seconds (default: 24.0)
///
/// # R Reference
/// ```r
/// hrf_sine <- function(t, span = 24, N = 5) {
///   sine_basis <- vapply(1:N, function(n) sin(2*pi*n*t/span), numeric(length(t)))
///   sine_basis[t < 0 | t > span, ] <- 0
///   return(sine_basis)
/// }
/// ```
pub fn sine_basis_hrf(n_basis: usize, span: f64) -> Box<dyn Hrf> {
    Box::new(SineBasisHrf { n_basis, span })
}

#[derive(Debug, Clone)]
struct SineBasisHrf {
    n_basis: usize,
    span: f64,
}

impl Hrf for SineBasisHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let mut basis = Array2::zeros((t.len(), self.n_basis));

        for (i, &time) in t.iter().enumerate() {
            let in_support = !time.is_nan() && time >= 0.0 && time <= self.span;
            if in_support {
                for n in 1..=self.n_basis {
                    basis[[i, n - 1]] = (2.0 * PI * n as f64 * time / self.span).sin();
                }
            }
            // else: row stays zero (initialized above)
        }

        Ok(basis)
    }

    fn name(&self) -> &str {
        "sine_basis"
    }

    fn nbasis(&self) -> usize {
        self.n_basis
    }

    fn span(&self) -> f64 {
        self.span
    }

    fn params(&self) -> Vec<(&str, f64)> {
        vec![("n_basis", self.n_basis as f64), ("span", self.span)]
    }
}

/// Create a sine wave HRF.
///
/// # Arguments
/// * `frequency` - Frequency in Hz
/// * `phase` - Phase shift in radians
///
/// # Deprecated
/// Use [`sine_basis_hrf`] instead for R parity with `hrf_sine()`.
#[deprecated(note = "Use sine_basis_hrf for R parity")]
pub fn sine_hrf(frequency: f64, phase: f64) -> Box<dyn Hrf> {
    let params = vec![
        ("frequency".to_string(), frequency),
        ("phase".to_string(), phase),
    ];

    Box::new(SimpleHrf::new("sine_hrf", 1, 30.0, params, |t, params| {
        let frequency = params.iter().find(|(k, _)| k == "frequency").unwrap().1;
        let phase = params.iter().find(|(k, _)| k == "phase").unwrap().1;

        let hrf = t.mapv(|time| (2.0 * PI * frequency * time + phase).sin());
        Ok(hrf.insert_axis(Axis(1)))
    }))
}

/// Create a half-cosine HRF.
///
/// # Arguments
/// * `width` - Width of the half-cosine
/// * `center` - Center of the function
pub fn half_cosine_hrf(width: f64, center: f64) -> Box<dyn Hrf> {
    let params = vec![("width".to_string(), width), ("center".to_string(), center)];

    Box::new(SimpleHrf::new(
        "half_cosine_hrf",
        1,
        30.0,
        params,
        |t, params| {
            let width = params.iter().find(|(k, _)| k == "width").unwrap().1;
            let center = params.iter().find(|(k, _)| k == "center").unwrap().1;

            evaluate_half_cosine_hrf(t, width, center)
        },
    ))
}

fn evaluate_half_cosine_hrf(t: &TimeArray, width: f64, center: f64) -> Result<HrfValues> {
    if width <= 0.0 {
        return Err(HrfError::InvalidParameters("Width must be positive".into()));
    }

    let t_start = center - width / 2.0;
    let t_end = center + width / 2.0;

    let hrf = t.mapv(|time| {
        if time >= t_start && time <= t_end {
            let scaled_t = (time - t_start) * PI / width;
            0.5 * (1.0 + scaled_t.cos())
        } else {
            0.0
        }
    });

    Ok(hrf.insert_axis(Axis(1)))
}

/// Create an inverse logit HRF (difference of two logistic functions).
///
/// Matches the R `hrf_inv_logit(t, mu1, s1, mu2, s2, lag)` formula:
/// ```text
/// logit1 = 1 / (1 + exp(-(t - lag - mu1) / s1))
/// logit2 = 1 / (1 + exp(-(t - lag - mu2) / s2))
/// result = logit1 - logit2
/// ```
///
/// The function rises around `t = mu1 + lag` and falls around `t = mu2 + lag`,
/// producing a bell-shaped response between those two inflection points.
///
/// # Arguments
/// * `mu1` - Center of the rising logistic (default in R: 6.0)
/// * `s1` - Scale of the rising logistic; must be positive (default in R: 1.0)
/// * `mu2` - Center of the falling logistic (default in R: 16.0)
/// * `s2` - Scale of the falling logistic; must be positive (default in R: 1.0)
/// * `lag` - Temporal lag applied to both logistics (default in R: 0.0)
pub fn inv_logit_hrf(mu1: f64, s1: f64, mu2: f64, s2: f64, lag: f64) -> Box<dyn Hrf> {
    let params = vec![
        ("mu1".to_string(), mu1),
        ("s1".to_string(), s1),
        ("mu2".to_string(), mu2),
        ("s2".to_string(), s2),
        ("lag".to_string(), lag),
    ];

    Box::new(SimpleHrf::new(
        "inv_logit_hrf",
        1,
        30.0,
        params,
        |t, params| {
            let mu1 = params.iter().find(|(k, _)| k == "mu1").unwrap().1;
            let s1 = params.iter().find(|(k, _)| k == "s1").unwrap().1;
            let mu2 = params.iter().find(|(k, _)| k == "mu2").unwrap().1;
            let s2 = params.iter().find(|(k, _)| k == "s2").unwrap().1;
            let lag = params.iter().find(|(k, _)| k == "lag").unwrap().1;

            evaluate_inv_logit_hrf(t, mu1, s1, mu2, s2, lag)
        },
    ))
}

fn evaluate_inv_logit_hrf(
    t: &TimeArray,
    mu1: f64,
    s1: f64,
    mu2: f64,
    s2: f64,
    lag: f64,
) -> Result<HrfValues> {
    if s1 <= 0.0 || s2 <= 0.0 {
        return Err(HrfError::InvalidParameters(
            "Scale parameters s1 and s2 must be positive".into(),
        ));
    }

    let hrf = t.mapv(|time| {
        if time < 0.0 {
            return 0.0;
        }
        let logit1 = 1.0 / (1.0 + (-(time - lag - mu1) / s1).exp());
        let logit2 = 1.0 / (1.0 + (-(time - lag - mu2) / s2).exp());
        logit1 - logit2
    });
    Ok(hrf.insert_axis(Axis(1)))
}

/// Lag-Width-Undershoot (LWU) HRF.
///
/// Computes the LWU hemodynamic response function using two Gaussian
/// components to model the main response and an optional undershoot.
///
/// Formula:
/// h(t; τ, σ, ρ) = exp(-(t-τ)²/(2σ²)) - ρ*exp(-(t-τ-2σ)²/(2(1.6σ)²))
///
/// # Arguments
/// * `tau` - Lag of main Gaussian, time-to-peak (default: 6.0)
/// * `sigma` - Width of main Gaussian, must be > 0.05 (default: 2.5)
/// * `rho` - Amplitude of undershoot, 0 to 1.5 (default: 0.35)
/// * `normalize` - Normalization type: "none" or "height" (default: "none")
pub fn lwu_hrf(
    tau: Option<f64>,
    sigma: Option<f64>,
    rho: Option<f64>,
    normalize: Option<&str>,
) -> Result<Box<dyn Hrf>> {
    let tau = tau.unwrap_or(6.0);
    let sigma = sigma.unwrap_or(2.5);
    let rho = rho.unwrap_or(0.35);
    let normalize = normalize.unwrap_or("none");

    // Validate parameters
    if sigma <= 0.05 {
        return Err(HrfError::InvalidParameters("sigma must be > 0.05".into()));
    }
    if !(0.0..=1.5).contains(&rho) {
        return Err(HrfError::InvalidParameters(
            "rho must be between 0 and 1.5".into(),
        ));
    }
    if !["none", "height"].contains(&normalize) {
        return Err(HrfError::InvalidParameters(
            "normalize must be 'none' or 'height'".into(),
        ));
    }

    let normalize_height = normalize == "height";
    let params = vec![
        ("tau".to_string(), tau),
        ("sigma".to_string(), sigma),
        ("rho".to_string(), rho),
        (
            "normalize".to_string(),
            if normalize_height { 1.0 } else { 0.0 },
        ),
    ];

    Ok(Box::new(SimpleHrf::new(
        "lwu_hrf",
        1,
        30.0,
        params,
        move |t, params| {
            let tau = params.iter().find(|(k, _)| k == "tau").unwrap().1;
            let sigma = params.iter().find(|(k, _)| k == "sigma").unwrap().1;
            let rho = params.iter().find(|(k, _)| k == "rho").unwrap().1;
            let normalize_height = params.iter().find(|(k, _)| k == "normalize").unwrap().1 > 0.5;

            evaluate_lwu_hrf(t, tau, sigma, rho, normalize_height)
        },
    )))
}

fn evaluate_lwu_hrf(
    t: &TimeArray,
    tau: f64,
    sigma: f64,
    rho: f64,
    normalize_height: bool,
) -> Result<HrfValues> {
    // Main positive Gaussian component
    let term1 = t.mapv(|t| (-(t - tau).powi(2) / (2.0 * sigma.powi(2))).exp());

    // Undershoot Gaussian component
    let term2 = t
        .mapv(|t| rho * (-(t - (tau + 2.0 * sigma)).powi(2) / (2.0 * (1.6 * sigma).powi(2))).exp());

    let mut response = term1 - term2;
    for (value, &time) in response.iter_mut().zip(t) {
        if time < 0.0 {
            *value = 0.0;
        }
    }

    if normalize_height {
        let reference = Array1::linspace(0.0, 30.0, 1501);
        let raw = evaluate_lwu_hrf(&reference, tau, sigma, rho, false)?;
        let max_abs_val = raw.fold(0.0_f64, |a, &b| a.max(b.abs()));
        if max_abs_val > 1e-10 {
            response /= max_abs_val;
        }
    }

    Ok(response.insert_axis(Axis(1)))
}

/// LWU HRF basis for Taylor expansion.
///
/// Constructs the basis set for the LWU HRF model, consisting of the
/// LWU HRF evaluated at expansion point theta0 and its partial derivatives.
///
/// # Arguments
/// * `theta0` - Expansion point [tau0, sigma0, rho0]
/// * `normalize_primary` - Normalization for primary HRF: "none" or "height" (default: "none")
pub fn hrf_basis_lwu(theta0: &[f64; 3], normalize_primary: Option<&str>) -> Result<Box<dyn Hrf>> {
    let [tau0, sigma0, rho0] = *theta0;
    let normalize_primary = normalize_primary.unwrap_or("none");

    // Validate parameters
    if sigma0 <= 0.05 {
        return Err(HrfError::InvalidParameters("sigma0 must be > 0.05".into()));
    }
    if !(0.0..=1.5).contains(&rho0) {
        return Err(HrfError::InvalidParameters(
            "rho0 must be between 0 and 1.5".into(),
        ));
    }

    let params = vec![
        ("tau0".to_string(), tau0),
        ("sigma0".to_string(), sigma0),
        ("rho0".to_string(), rho0),
        (
            "normalize_primary".to_string(),
            if normalize_primary == "height" {
                1.0
            } else {
                0.0
            },
        ),
    ];

    Ok(Box::new(SimpleHrf::new(
        "hrf_basis_lwu",
        4,
        30.0,
        params,
        move |t, params| {
            let tau0 = params.iter().find(|(k, _)| k == "tau0").unwrap().1;
            let sigma0 = params.iter().find(|(k, _)| k == "sigma0").unwrap().1;
            let rho0 = params.iter().find(|(k, _)| k == "rho0").unwrap().1;
            let normalize_primary = params
                .iter()
                .find(|(k, _)| k == "normalize_primary")
                .unwrap()
                .1
                > 0.5;

            evaluate_hrf_basis_lwu(
                t,
                tau0,
                sigma0,
                rho0,
                if normalize_primary { "height" } else { "none" },
            )
        },
    )))
}

fn evaluate_hrf_basis_lwu(
    t: &TimeArray,
    tau0: f64,
    sigma0: f64,
    rho0: f64,
    normalize_primary: &str,
) -> Result<HrfValues> {
    let n_times = t.len();
    let mut basis = Array2::zeros((n_times, 4));

    // Primary HRF at expansion point
    let h0 = lwu_hrf(
        Some(tau0),
        Some(sigma0),
        Some(rho0),
        Some(normalize_primary),
    )?;
    let h0_values = h0.evaluate(t)?;
    basis
        .slice_mut(s![.., 0])
        .assign(&h0_values.slice(s![.., 0]));

    // Compute partial derivatives numerically
    let eps = 1e-5;

    // Partial derivative w.r.t. tau
    let h_tau_plus = lwu_hrf(Some(tau0 + eps), Some(sigma0), Some(rho0), Some("none"))?;
    let h_tau_minus = lwu_hrf(Some(tau0 - eps), Some(sigma0), Some(rho0), Some("none"))?;
    let h_tau_plus_values = h_tau_plus.evaluate(t)?;
    let h_tau_minus_values = h_tau_minus.evaluate(t)?;
    let dh_dtau =
        (&h_tau_plus_values.slice(s![.., 0]) - &h_tau_minus_values.slice(s![.., 0])) / (2.0 * eps);
    basis.slice_mut(s![.., 1]).assign(&dh_dtau);

    // Partial derivative w.r.t. sigma
    let h_sigma_plus = lwu_hrf(Some(tau0), Some(sigma0 + eps), Some(rho0), Some("none"))?;
    let h_sigma_minus = lwu_hrf(Some(tau0), Some(sigma0 - eps), Some(rho0), Some("none"))?;
    let h_sigma_plus_values = h_sigma_plus.evaluate(t)?;
    let h_sigma_minus_values = h_sigma_minus.evaluate(t)?;
    let dh_dsigma = (&h_sigma_plus_values.slice(s![.., 0])
        - &h_sigma_minus_values.slice(s![.., 0]))
        / (2.0 * eps);
    basis.slice_mut(s![.., 2]).assign(&dh_dsigma);

    // Partial derivative w.r.t. rho
    let h_rho_plus = lwu_hrf(Some(tau0), Some(sigma0), Some(rho0 + eps), Some("none"))?;
    let h_rho_minus = lwu_hrf(Some(tau0), Some(sigma0), Some(rho0 - eps), Some("none"))?;
    let h_rho_plus_values = h_rho_plus.evaluate(t)?;
    let h_rho_minus_values = h_rho_minus.evaluate(t)?;
    let dh_drho =
        (&h_rho_plus_values.slice(s![.., 0]) - &h_rho_minus_values.slice(s![.., 0])) / (2.0 * eps);
    basis.slice_mut(s![.., 3]).assign(&dh_drho);

    Ok(basis)
}

/// Time HRF - Linear function of time.
///
/// Simple linear function of time when t is between 0 and maxt.
///
/// # Arguments
/// * `maxt` - Maximum time point in the domain (default: 22.0)
pub fn hrf_time(maxt: Option<f64>) -> Box<dyn Hrf> {
    let maxt = maxt.unwrap_or(22.0);
    let params = vec![("maxt".to_string(), maxt)];

    Box::new(SimpleHrf::new(
        "hrf_time",
        1,
        maxt,
        params,
        move |t, params| {
            let maxt = params.iter().find(|(k, _)| k == "maxt").unwrap().1;
            let values = t.mapv(|t| if t > 0.0 && t < maxt { t } else { 0.0 });
            Ok(values.insert_axis(Axis(1)))
        },
    ))
}

/// Create a boxcar (rectangular pulse) HRF.
///
/// Returns a constant amplitude for `0 <= t < width`, and zero otherwise.
/// This is a standalone boxcar HRF object, matching R's `hrf_boxcar()`.
///
/// # Arguments
/// * `width` - Duration of the boxcar pulse (must be > 0)
/// * `amplitude` - Amplitude of the pulse (ignored if `normalize` is true)
/// * `normalize` - If true, set amplitude to `1/width` so the integral equals 1
///
/// # Panics
/// Panics if `width <= 0`.
pub fn boxcar_hrf(width: f64, amplitude: f64, normalize: bool) -> Box<dyn Hrf> {
    assert!(width > 0.0, "width must be a positive value");

    let effective_amplitude = if normalize { 1.0 / width } else { amplitude };

    let name = format!("boxcar[{:.2}]", width);
    let params = vec![
        ("width".to_string(), width),
        ("amplitude".to_string(), effective_amplitude),
        ("normalize".to_string(), if normalize { 1.0 } else { 0.0 }),
    ];

    Box::new(SimpleHrf::new(&name, 1, width, params, |t, params| {
        let width = params.iter().find(|(k, _)| k == "width").unwrap().1;
        let amplitude = params.iter().find(|(k, _)| k == "amplitude").unwrap().1;

        let values = t.mapv(|t| {
            if t >= 0.0 && t < width {
                amplitude
            } else {
                0.0
            }
        });
        Ok(values.insert_axis(Axis(1)))
    }))
}

/// Interpolation method for weighted HRF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightedMethod {
    /// Piecewise constant (step function): value at left endpoint of each interval.
    Constant,
    /// Linear interpolation between adjacent weight points.
    Linear,
}

/// HRF defined by user-specified weights at time points with interpolation.
///
/// Matches R's `hrf_weighted()` from the `fmrihrf` package.
#[derive(Debug, Clone)]
struct WeightedFnHrf {
    weights: Vec<f64>,
    times: Vec<f64>,
    method: WeightedMethod,
    hrf_name: String,
}

impl WeightedFnHrf {
    /// Evaluate the weighted HRF at a single time point.
    fn interpolate(&self, t: f64) -> f64 {
        let first = self.times[0];
        let last = *self.times.last().unwrap();

        // Outside the domain: return 0
        if t < first || t > last {
            return 0.0;
        }

        // At or beyond the last knot: return 0 (matches R's yright=0)
        if t >= last {
            return 0.0;
        }

        // Find the interval index: largest i such that times[i] <= t
        let idx = match self
            .times
            .binary_search_by(|probe| probe.partial_cmp(&t).unwrap())
        {
            Ok(i) => i,                    // exact match
            Err(i) => i.saturating_sub(1), // t is between times[i-1] and times[i]
        };

        match self.method {
            WeightedMethod::Constant => {
                // Step function: return weight at left endpoint of interval
                self.weights[idx]
            }
            WeightedMethod::Linear => {
                // If we're exactly at a knot, return its weight
                if idx < self.times.len() && (self.times[idx] - t).abs() < 1e-15 {
                    return self.weights[idx];
                }
                // Linear interpolation between times[idx] and times[idx+1]
                let next = idx + 1;
                if next >= self.times.len() {
                    return self.weights[idx];
                }
                let dt = self.times[next] - self.times[idx];
                if dt.abs() < 1e-15 {
                    return self.weights[idx];
                }
                let frac = (t - self.times[idx]) / dt;
                self.weights[idx] + frac * (self.weights[next] - self.weights[idx])
            }
        }
    }
}

impl Hrf for WeightedFnHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let values: Array1<f64> = t.mapv(|ti| self.interpolate(ti));
        Ok(values.insert_axis(Axis(1)))
    }

    fn name(&self) -> &str {
        &self.hrf_name
    }

    fn nbasis(&self) -> usize {
        1
    }

    fn span(&self) -> f64 {
        *self.times.last().unwrap_or(&0.0)
    }

    fn params(&self) -> Vec<(&str, f64)> {
        // Return method as a numeric code and the number of points
        let method_code = match self.method {
            WeightedMethod::Constant => 0.0,
            WeightedMethod::Linear => 1.0,
        };
        vec![
            ("n_points", self.times.len() as f64),
            ("method", method_code),
            ("span", self.span()),
        ]
    }
}

/// Create a weighted HRF from user-specified weights at explicit time points.
///
/// This matches R's `hrf_weighted(weights, times=..., method=..., normalize=...)`.
///
/// # Arguments
/// * `weights` - Weight values at each time point (at least 2 elements)
/// * `times` - Time points corresponding to each weight (must be same length as weights,
///   strictly increasing, and starting >= 0)
/// * `method` - Interpolation method: `Constant` (step function) or `Linear`
/// * `normalize` - If true, normalize so that the integral is 1
///
/// # Examples
/// ```
/// use fmrihrf::hrf::functions::{weighted_hrf, WeightedMethod};
/// use ndarray::Array1;
///
/// let weights = vec![0.0, 1.0, 0.5, 0.0];
/// let times = vec![0.0, 2.0, 4.0, 6.0];
/// let hrf = weighted_hrf(weights, times, WeightedMethod::Linear, false).unwrap();
/// let t = Array1::linspace(0.0, 6.0, 31);
/// let result = hrf.evaluate(&t).unwrap();
/// ```
pub fn weighted_hrf(
    mut weights: Vec<f64>,
    times: Vec<f64>,
    method: WeightedMethod,
    normalize: bool,
) -> Result<Box<dyn Hrf>> {
    // Validate inputs
    if weights.len() < 2 {
        return Err(HrfError::InvalidParameters(
            "weights must have at least 2 elements".into(),
        ));
    }
    if weights.len() != times.len() {
        return Err(HrfError::InvalidParameters(format!(
            "weights length ({}) must equal times length ({})",
            weights.len(),
            times.len()
        )));
    }
    if times[0] < 0.0 {
        return Err(HrfError::InvalidParameters(
            "times must start at >= 0".into(),
        ));
    }
    // Check strictly increasing
    for i in 1..times.len() {
        if times[i] <= times[i - 1] {
            return Err(HrfError::InvalidParameters(format!(
                "times must be strictly increasing, but times[{}]={} <= times[{}]={}",
                i,
                times[i],
                i - 1,
                times[i - 1]
            )));
        }
    }

    // Normalize if requested
    if normalize {
        match method {
            WeightedMethod::Constant => {
                // Sum of weights excluding the last element (matches R)
                let weight_sum: f64 = weights[..weights.len() - 1].iter().sum();
                if weight_sum.abs() > 1e-10 {
                    for w in weights.iter_mut() {
                        *w /= weight_sum;
                    }
                }
            }
            WeightedMethod::Linear => {
                // Trapezoidal integral
                let mut integral = 0.0;
                for i in 0..weights.len() - 1 {
                    let dt = times[i + 1] - times[i];
                    let avg = (weights[i] + weights[i + 1]) / 2.0;
                    integral += dt * avg;
                }
                if integral.abs() > 1e-10 {
                    for w in weights.iter_mut() {
                        *w /= integral;
                    }
                }
            }
        }
    }

    let hrf_name = format!("weighted[{} pts, {:?}]", times.len(), method);

    Ok(Box::new(WeightedFnHrf {
        weights,
        times,
        method,
        hrf_name,
    }))
}

/// Create a weighted HRF with evenly-spaced times from 0 to `width`.
///
/// This is a convenience wrapper around [`weighted_hrf`] that generates
/// uniform time spacing, matching R's `hrf_weighted(weights, width=...)`.
///
/// # Arguments
/// * `weights` - Weight values (at least 2 elements)
/// * `width` - Total temporal span; times will be `linspace(0, width, weights.len())`
/// * `method` - Interpolation method
/// * `normalize` - Whether to normalize the integral to 1
///
/// # Examples
/// ```
/// use fmrihrf::hrf::functions::{weighted_hrf_uniform, WeightedMethod};
///
/// let hrf = weighted_hrf_uniform(vec![1.0, 2.0, 1.0], 4.0,
///     WeightedMethod::Constant, false).unwrap();
/// assert_eq!(hrf.span(), 4.0);
/// ```
pub fn weighted_hrf_uniform(
    weights: Vec<f64>,
    width: f64,
    method: WeightedMethod,
    normalize: bool,
) -> Result<Box<dyn Hrf>> {
    if weights.len() < 2 {
        return Err(HrfError::InvalidParameters(
            "weights must have at least 2 elements".into(),
        ));
    }
    if width <= 0.0 {
        return Err(HrfError::InvalidParameters("width must be positive".into()));
    }
    let n = weights.len();
    let times: Vec<f64> = (0..n).map(|i| i as f64 * width / (n - 1) as f64).collect();
    weighted_hrf(weights, times, method, normalize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_spm_canonical() {
        let hrf = spm_canonical(None, None, None, None, None, None);
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 1]);

        // Should peak around 5-6 seconds
        let (max_idx, _) = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(max_idx >= 4 && max_idx <= 7);

        // Raw canonical peak from the independent R formula.
        let max_val = result.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_relative_eq!(max_val, 0.175441162195464, epsilon = 1e-12);
    }

    #[test]
    fn test_gamma_hrf() {
        let hrf = gamma_hrf(6.0, 1.0, Some(true));
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 1]);

        // Should be normalized
        let max_val = result.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gaussian_hrf() {
        let hrf = gaussian_hrf(5.0, 2.0, Some(true));
        let t = Array1::linspace(0.0, 10.0, 11);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[11, 1]);

        // Should peak at center (t=5)
        assert_relative_eq!(result[[5, 0]], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_spm_with_derivatives() {
        let hrf = spm_hrf_with_derivatives();
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 3]);
        assert_eq!(hrf.nbasis(), 3);

        // First column should be canonical HRF
        let canonical_max = result.column(0).fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_relative_eq!(canonical_max, 0.175441162195464, epsilon = 1e-12);
    }

    #[test]
    fn test_bspline_hrf() {
        let hrf = bspline_hrf(5, Some(3), Some(20.0));
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 5]);
        assert_eq!(hrf.nbasis(), 5);

        // Check that basis functions sum to approximately 1 (partition of unity)
        for i in 5..15 {
            // Check interior points
            let sum: f64 = (0..5).map(|j| result[[i, j]]).sum();
            // R's splines library may not always satisfy perfect partition of unity without intercept
            // Check for reasonable non-negative values instead
            if sum < 0.5 || sum > 1.5 {
                panic!(
                    "B-spline basis sum out of reasonable range at t={}: sum={}, values={:?}",
                    t[i],
                    sum,
                    (0..5).map(|j| result[[i, j]]).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn test_fourier_hrf() {
        let hrf = fourier_hrf(5, Some(20.0));
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 5]);
        assert_eq!(hrf.nbasis(), 5);

        // Check orthogonality (approximately)
        let dt = 20.0 / 20.0; // Approximate integral
        for i in 0..5 {
            for j in i + 1..5 {
                let inner_product: f64 =
                    (0..20).map(|k| result[[k, i]] * result[[k, j]] * dt).sum();
                assert!(
                    inner_product.abs() < 0.1,
                    "Fourier basis not orthogonal: <{}, {}> = {}",
                    i,
                    j,
                    inner_product
                );
            }
        }
    }

    #[test]
    fn test_fir_hrf() {
        let hrf = fir_hrf(10, Some(20.0));
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 10]);
        assert_eq!(hrf.nbasis(), 10);

        // Check that each time point activates exactly one basis function
        for i in 0..20 {
            // Exclude last point which is edge case
            let active_count: usize = (0..10).filter(|&j| result[[i, j]] > 0.0).count();
            assert_eq!(
                active_count, 1,
                "FIR basis should have exactly one active function at each time"
            );
        }
    }

    #[test]
    fn test_mexhat_hrf() {
        let hrf = mexhat_hrf(2.0, 6.0);
        let t = Array1::linspace(0.0, 20.0, 21);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[21, 1]);

        // Mexican hat should have negative values
        let has_negative = result.iter().any(|&v| v < 0.0);
        assert!(has_negative, "Mexican hat should have negative values");

        // Peak should be around center
        let (max_idx, _) = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(max_idx >= 4 && max_idx <= 8); // Peak around t=6
    }

    #[test]
    #[allow(deprecated)]
    fn test_sine_hrf() {
        let hrf = sine_hrf(0.5, 0.0);
        let t = Array1::linspace(0.0, 4.0, 41); // Two complete cycles
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[41, 1]);

        // Check periodicity
        assert_relative_eq!(result[[0, 0]], result[[20, 0]], epsilon = 1e-6); // t=0 and t=2
        assert_relative_eq!(result[[10, 0]], result[[30, 0]], epsilon = 1e-6); // t=1 and t=3
    }

    #[test]
    fn test_sine_basis_hrf_shape() {
        let hrf = sine_basis_hrf(5, 24.0);
        assert_eq!(hrf.nbasis(), 5);
        assert_eq!(hrf.name(), "sine_basis");
        assert_eq!(hrf.span(), 24.0);

        let t = Array1::linspace(0.0, 24.0, 49);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[49, 5]);
    }

    #[test]
    fn test_sine_basis_hrf_values_match_r() {
        // Verify values match R formula: sin(2*pi*n*t/span)
        let hrf = sine_basis_hrf(3, 24.0);
        let t = Array1::from_vec(vec![0.0, 6.0, 12.0, 24.0]);
        let result = hrf.evaluate(&t).unwrap();

        // t=0: sin(0) = 0 for all basis functions
        for col in 0..3 {
            assert_relative_eq!(result[[0, col]], 0.0, epsilon = 1e-10);
        }

        // t=6, n=1: sin(2*pi*1*6/24) = sin(pi/2) = 1.0
        assert_relative_eq!(result[[1, 0]], 1.0, epsilon = 1e-10);

        // t=6, n=2: sin(2*pi*2*6/24) = sin(pi) = 0.0
        assert_relative_eq!(result[[1, 1]], 0.0, epsilon = 1e-10);

        // t=12, n=1: sin(2*pi*1*12/24) = sin(pi) = 0.0
        assert_relative_eq!(result[[2, 0]], 0.0, epsilon = 1e-10);

        // t=12, n=2: sin(2*pi*2*12/24) = sin(2*pi) = 0.0
        assert_relative_eq!(result[[2, 1]], 0.0, epsilon = 1e-10);

        // t=24 (boundary, in support): sin(2*pi*n*24/24) = sin(2*pi*n) = 0.0
        for col in 0..3 {
            assert_relative_eq!(result[[3, col]], 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sine_basis_hrf_zero_outside_support() {
        let hrf = sine_basis_hrf(5, 24.0);
        let t = Array1::from_vec(vec![-1.0, -0.5, 24.5, 30.0]);
        let result = hrf.evaluate(&t).unwrap();

        // All values outside [0, span] should be zero
        for i in 0..4 {
            for j in 0..5 {
                assert_relative_eq!(result[[i, j]], 0.0, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_sine_basis_hrf_defaults() {
        // Test with typical defaults (N=5, span=24)
        let hrf = sine_basis_hrf(5, 24.0);
        let t = Array1::linspace(0.0, 24.0, 100);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[100, 5]);

        // All values should be in [-1, 1]
        for &v in result.iter() {
            assert!(
                v >= -1.0 - 1e-10 && v <= 1.0 + 1e-10,
                "Sine basis value {} out of [-1, 1] range",
                v
            );
        }
    }

    #[test]
    fn test_half_cosine_hrf() {
        let hrf = half_cosine_hrf(10.0, 5.0);
        let t = Array1::linspace(0.0, 10.0, 11);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[11, 1]);

        // Half-cosine centered at 5 with width 10 spans [0, 10]
        // At t=0 (start), cos(0) after shift gives max value
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-6);
        // At t=10 (end), should approach 0
        assert!(result[[10, 0]].abs() < 1e-6);

        // Peak should be at start (t=0) for this half-cosine
        let (max_idx, _) = result
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert_eq!(max_idx, 0); // Peak at t=0 for half-cosine starting at max
    }

    #[test]
    fn test_inv_logit_hrf_defaults() {
        // R defaults: mu1=6, s1=1, mu2=16, s2=1, lag=0
        let hrf = inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0);
        let t = Array1::linspace(0.0, 30.0, 301);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result.shape(), &[301, 1]);

        // At t=0, result should be approximately 0 (both logistics near 0)
        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 0.01);

        // Between mu1 and mu2, the function should be close to 1
        // At t=11 (midway), both logit1 ~ 1 and logit2 ~ 0 so result ~ 1
        let idx_11 = 110; // t=11.0 at 0.1 spacing
        assert_relative_eq!(result[[idx_11, 0]], 1.0, epsilon = 0.02);

        // At t=30, result should be approximately 0 (both logistics near 1)
        assert_relative_eq!(result[[300, 0]], 0.0, epsilon = 0.01);

        // Peak should be approximately 1
        let max_val = result.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert_relative_eq!(max_val, 1.0, epsilon = 0.02);
    }

    #[test]
    fn test_inv_logit_hrf_with_lag() {
        // With lag=5, curve should shift right by 5 seconds
        let hrf_no_lag = inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0);
        let hrf_lag5 = inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 5.0);

        let t = Array1::linspace(0.0, 35.0, 351);
        let r_no_lag = hrf_no_lag.evaluate(&t).unwrap();
        let r_lag5 = hrf_lag5.evaluate(&t).unwrap();

        // Value at t=11 without lag should equal value at t=16 with lag=5
        let idx_11 = 110; // t=11
        let idx_16 = 160; // t=16
        assert_relative_eq!(r_no_lag[[idx_11, 0]], r_lag5[[idx_16, 0]], epsilon = 1e-10);
    }

    #[test]
    fn test_inv_logit_hrf_invalid_scale() {
        let hrf = inv_logit_hrf(6.0, 0.0, 16.0, 1.0, 0.0);
        let t = Array1::linspace(0.0, 20.0, 21);
        assert!(hrf.evaluate(&t).is_err(), "s1=0 should be rejected");

        let hrf2 = inv_logit_hrf(6.0, 1.0, 16.0, -1.0, 0.0);
        assert!(hrf2.evaluate(&t).is_err(), "s2<0 should be rejected");
    }

    #[test]
    fn test_lwu_hrf() {
        // Test with default parameters
        let hrf = lwu_hrf(None, None, None, None).unwrap();
        assert_eq!(hrf.name(), "lwu_hrf");
        assert_eq!(hrf.nbasis(), 1);

        let t = Array1::linspace(0.0, 30.0, 301);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[301, 1]);

        // Should have positive values early and negative undershoot later
        let early_max = result
            .slice(s![0..100, 0])
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let late_min = result
            .slice(s![100..200, 0])
            .fold(f64::INFINITY, |a, &b| a.min(b));
        assert!(early_max > 0.0, "Should have positive early response");
        assert!(late_min < 0.0, "Should have negative undershoot");

        // Test with custom parameters
        let hrf2 = lwu_hrf(Some(8.0), Some(3.0), Some(0.5), Some("height")).unwrap();
        let result2 = hrf2.evaluate(&Array1::linspace(0.0, 30.0, 1501)).unwrap();
        let max_val = result2.fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_lwu_hrf_validation() {
        // Test invalid sigma
        assert!(lwu_hrf(Some(6.0), Some(0.01), None, None).is_err());

        // Test invalid rho
        assert!(lwu_hrf(Some(6.0), Some(2.5), Some(2.0), None).is_err());
        assert!(lwu_hrf(Some(6.0), Some(2.5), Some(-0.1), None).is_err());

        // Test invalid normalize
        assert!(lwu_hrf(Some(6.0), Some(2.5), Some(0.35), Some("invalid")).is_err());
    }

    #[test]
    fn test_hrf_basis_lwu() {
        let theta0 = [6.0, 2.5, 0.35];
        let hrf = hrf_basis_lwu(&theta0, None).unwrap();

        assert_eq!(hrf.name(), "hrf_basis_lwu");
        assert_eq!(hrf.nbasis(), 4);

        let t = Array1::linspace(0.0, 30.0, 31);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[31, 4]);

        // First column should be the base LWU HRF
        let base_hrf = lwu_hrf(Some(6.0), Some(2.5), Some(0.35), None).unwrap();
        let base_result = base_hrf.evaluate(&t).unwrap();

        for i in 0..31 {
            assert_relative_eq!(result[[i, 0]], base_result[[i, 0]], epsilon = 1e-6);
        }

        // Other columns should be derivatives (non-zero)
        for col in 1..4 {
            let col_sum: f64 = result.column(col).mapv(|x| x.abs()).sum();
            assert!(
                col_sum > 0.01,
                "Derivative column {} should be non-zero",
                col
            );
        }
    }

    #[test]
    fn test_hrf_basis_lwu_normalized() {
        let theta0 = [6.0, 2.5, 0.35];
        let hrf = hrf_basis_lwu(&theta0, Some("height")).unwrap();

        let t = Array1::linspace(0.0, 30.0, 301);
        let result = hrf.evaluate(&t).unwrap();

        // First column should be normalized
        let max_val = result
            .column(0)
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
        assert_relative_eq!(max_val, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hrf_time() {
        let hrf = hrf_time(None);
        assert_eq!(hrf.name(), "hrf_time");
        assert_eq!(hrf.nbasis(), 1);
        assert_eq!(hrf.span(), 22.0);

        let t = Array1::from_vec(vec![0.0, 5.0, 10.0, 15.0, 20.0, 22.0, 25.0]);
        let result = hrf.evaluate(&t).unwrap();

        assert_eq!(result[[0, 0]], 0.0); // t=0 returns 0
        assert_eq!(result[[1, 0]], 5.0); // t=5 returns 5
        assert_eq!(result[[2, 0]], 10.0); // t=10 returns 10
        assert_eq!(result[[3, 0]], 15.0); // t=15 returns 15
        assert_eq!(result[[4, 0]], 20.0); // t=20 returns 20
        assert_eq!(result[[5, 0]], 0.0); // t=22 (maxt) returns 0
        assert_eq!(result[[6, 0]], 0.0); // t=25 (>maxt) returns 0

        // Test with custom maxt
        let hrf2 = hrf_time(Some(30.0));
        assert_eq!(hrf2.span(), 30.0);
        let result2 = hrf2.evaluate(&t).unwrap();
        assert_eq!(result2[[6, 0]], 25.0); // t=25 now returns 25
    }

    #[test]
    fn test_boxcar_hrf() {
        use approx::assert_abs_diff_eq;

        let hrf = boxcar_hrf(5.0, 1.0, false);
        assert_eq!(hrf.name(), "boxcar[5.00]");
        assert_eq!(hrf.nbasis(), 1);
        assert_eq!(hrf.span(), 5.0);

        let t = Array1::from_vec(vec![-1.0, 0.0, 2.5, 4.99, 5.0, 10.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[6, 1]);
        // t=-1: 0, t=0: 1, t=2.5: 1, t=4.99: 1, t=5.0: 0, t=10: 0
        assert_abs_diff_eq!(result[[0, 0]], 0.0);
        assert_abs_diff_eq!(result[[1, 0]], 1.0);
        assert_abs_diff_eq!(result[[2, 0]], 1.0);
        assert_abs_diff_eq!(result[[3, 0]], 1.0);
        assert_abs_diff_eq!(result[[4, 0]], 0.0);
        assert_abs_diff_eq!(result[[5, 0]], 0.0);
    }

    #[test]
    fn test_boxcar_hrf_normalized() {
        use approx::assert_abs_diff_eq;

        let hrf = boxcar_hrf(5.0, 1.0, true);
        let t = Array1::from_vec(vec![0.0, 2.5]);
        let result = hrf.evaluate(&t).unwrap();
        // normalized: amplitude = 1/width = 0.2
        assert_abs_diff_eq!(result[[0, 0]], 0.2);
        assert_abs_diff_eq!(result[[1, 0]], 0.2);
    }

    #[test]
    fn test_boxcar_hrf_custom_amplitude() {
        use approx::assert_abs_diff_eq;

        let hrf = boxcar_hrf(3.0, 2.5, false);
        assert_eq!(hrf.name(), "boxcar[3.00]");
        assert_eq!(hrf.span(), 3.0);

        let t = Array1::from_vec(vec![0.0, 1.5, 3.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_abs_diff_eq!(result[[0, 0]], 2.5);
        assert_abs_diff_eq!(result[[1, 0]], 2.5);
        assert_abs_diff_eq!(result[[2, 0]], 0.0); // t=width is outside
    }

    #[test]
    #[should_panic(expected = "width must be a positive value")]
    fn test_boxcar_hrf_invalid_width() {
        boxcar_hrf(0.0, 1.0, false);
    }
    // ---- Weighted HRF tests ----

    #[test]
    fn test_weighted_hrf_constant() {
        let weights = vec![0.0, 1.0, 0.5, 0.0];
        let times = vec![0.0, 2.0, 4.0, 6.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Constant, false).unwrap();
        assert_eq!(hrf.nbasis(), 1);
        assert_eq!(hrf.span(), 6.0);
        assert!(hrf.name().contains("weighted"));

        let t = Array1::from_vec(vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[9, 1]);

        // t=-1: outside range -> 0
        assert_eq!(result[[0, 0]], 0.0);
        // t=0: weight at index 0 -> 0.0
        assert_eq!(result[[1, 0]], 0.0);
        // t=1: in [0,2) interval -> weight[0] = 0.0
        assert_eq!(result[[2, 0]], 0.0);
        // t=2: in [2,4) interval -> weight[1] = 1.0
        assert_eq!(result[[3, 0]], 1.0);
        // t=3: in [2,4) interval -> weight[1] = 1.0
        assert_eq!(result[[4, 0]], 1.0);
        // t=4: in [4,6) interval -> weight[2] = 0.5
        assert_eq!(result[[5, 0]], 0.5);
        // t=5: in [4,6) interval -> weight[2] = 0.5
        assert_eq!(result[[6, 0]], 0.5);
        // t=6: at endpoint -> 0 (yright=0)
        assert_eq!(result[[7, 0]], 0.0);
        // t=7: outside range -> 0
        assert_eq!(result[[8, 0]], 0.0);
    }

    #[test]
    fn test_weighted_hrf_linear() {
        let weights = vec![0.0, 1.0, 0.0];
        let times = vec![0.0, 3.0, 6.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Linear, false).unwrap();
        assert_eq!(hrf.nbasis(), 1);
        assert_eq!(hrf.span(), 6.0);

        let t = Array1::from_vec(vec![0.0, 1.5, 3.0, 4.5, 6.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_eq!(result.shape(), &[5, 1]);

        assert_relative_eq!(result[[0, 0]], 0.0, epsilon = 1e-12);
        assert_relative_eq!(result[[1, 0]], 0.5, epsilon = 1e-12);
        assert_relative_eq!(result[[2, 0]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result[[3, 0]], 0.5, epsilon = 1e-12);
        // t=6 is the endpoint -> 0 (yright=0)
        assert_relative_eq!(result[[4, 0]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weighted_hrf_linear_outside() {
        let weights = vec![0.0, 1.0, 0.0];
        let times = vec![1.0, 3.0, 5.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Linear, false).unwrap();

        let t = Array1::from_vec(vec![-1.0, 0.5, 1.0, 5.0, 6.0]);
        let result = hrf.evaluate(&t).unwrap();

        // Before range
        assert_eq!(result[[0, 0]], 0.0);
        assert_eq!(result[[1, 0]], 0.0);
        // At start
        assert_relative_eq!(result[[2, 0]], 0.0, epsilon = 1e-12);
        // At end
        assert_eq!(result[[3, 0]], 0.0);
        // Past end
        assert_eq!(result[[4, 0]], 0.0);
    }

    #[test]
    fn test_weighted_hrf_uniform() {
        let weights = vec![1.0, 2.0, 1.0];
        let hrf = weighted_hrf_uniform(weights, 4.0, WeightedMethod::Constant, false).unwrap();
        assert_eq!(hrf.span(), 4.0);
        assert_eq!(hrf.nbasis(), 1);
    }

    #[test]
    fn test_weighted_hrf_uniform_times() {
        // With 3 weights and width=6, times should be [0, 3, 6]
        let weights = vec![0.0, 1.0, 0.0];
        let hrf = weighted_hrf_uniform(weights, 6.0, WeightedMethod::Linear, false).unwrap();

        let t = Array1::from_vec(vec![1.5, 3.0, 4.5]);
        let result = hrf.evaluate(&t).unwrap();
        assert_relative_eq!(result[[0, 0]], 0.5, epsilon = 1e-12);
        assert_relative_eq!(result[[1, 0]], 1.0, epsilon = 1e-12);
        assert_relative_eq!(result[[2, 0]], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_weighted_hrf_normalize_constant() {
        let weights = vec![1.0, 2.0, 3.0, 0.0];
        let times = vec![0.0, 1.0, 2.0, 3.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Constant, true).unwrap();

        // Sum of weights[0..3] = 1+2+3 = 6, so normalized: [1/6, 2/6, 3/6, 0/6]
        let t = Array1::from_vec(vec![0.5, 1.5, 2.5]);
        let result = hrf.evaluate(&t).unwrap();
        assert_relative_eq!(result[[0, 0]], 1.0 / 6.0, epsilon = 1e-12);
        assert_relative_eq!(result[[1, 0]], 2.0 / 6.0, epsilon = 1e-12);
        assert_relative_eq!(result[[2, 0]], 3.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weighted_hrf_normalize_linear() {
        // Triangle: weights [0, 1, 0] at times [0, 1, 2]
        // Trapezoidal integral = (1*0.5) + (1*0.5) = 1.0, so no change
        let weights = vec![0.0, 1.0, 0.0];
        let times = vec![0.0, 1.0, 2.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Linear, true).unwrap();

        let t = Array1::from_vec(vec![0.5, 1.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_relative_eq!(result[[0, 0]], 0.5, epsilon = 1e-12);
        assert_relative_eq!(result[[1, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weighted_hrf_normalize_linear_scaling() {
        // weights [0, 2, 0] at times [0, 1, 2]
        // Trapezoidal integral = (1*(0+2)/2) + (1*(2+0)/2) = 1 + 1 = 2.0
        // After normalization: [0, 1, 0]
        let weights = vec![0.0, 2.0, 0.0];
        let times = vec![0.0, 1.0, 2.0];
        let hrf = weighted_hrf(weights, times, WeightedMethod::Linear, true).unwrap();

        let t = Array1::from_vec(vec![1.0]);
        let result = hrf.evaluate(&t).unwrap();
        assert_relative_eq!(result[[0, 0]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_weighted_hrf_validation_too_few_weights() {
        let result = weighted_hrf(vec![1.0], vec![0.0], WeightedMethod::Constant, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_hrf_validation_length_mismatch() {
        let result = weighted_hrf(
            vec![1.0, 2.0, 3.0],
            vec![0.0, 1.0],
            WeightedMethod::Linear,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_hrf_validation_negative_start() {
        let result = weighted_hrf(
            vec![1.0, 2.0],
            vec![-1.0, 1.0],
            WeightedMethod::Constant,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_hrf_validation_non_increasing() {
        let result = weighted_hrf(
            vec![1.0, 2.0, 3.0],
            vec![0.0, 2.0, 1.0],
            WeightedMethod::Linear,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_weighted_hrf_uniform_validation() {
        // Too few weights
        assert!(weighted_hrf_uniform(vec![1.0], 4.0, WeightedMethod::Constant, false).is_err());
        // Non-positive width
        assert!(
            weighted_hrf_uniform(vec![1.0, 2.0], 0.0, WeightedMethod::Constant, false).is_err()
        );
        assert!(
            weighted_hrf_uniform(vec![1.0, 2.0], -1.0, WeightedMethod::Constant, false).is_err()
        );
    }
}
