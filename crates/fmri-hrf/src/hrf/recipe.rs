//! Explicit portable HRF construction data. Custom closures are never guessed from names.
use super::*;
use crate::{HrfError, Result};
use ndarray::Array1;
/// Version-one construction recipes for qualified portable families.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(tag = "family", deny_unknown_fields))]
pub enum HrfRecipe {
    /// Legacy canonical factory with its normalization choice.
    Canonical {
        /// Normalize on the factory reference grid.
        normalize: bool,
    },
    /// Parameterized SPMG basis.
    Spmg {
        /// Number of SPMG basis columns (1–3).
        columns: usize,
        /// Response exponent.
        p1: f64,
        /// Undershoot exponent.
        p2: f64,
        /// Response amplitude.
        a1: f64,
    },
    /// Sampled empirical curve and explicit interpolation choice.
    Empirical {
        /// Distinct finite sample times.
        times: Vec<f64>,
        /// Measured values.
        values: Vec<f64>,
        /// Display name.
        name: String,
        /// Span covering the samples.
        span: f64,
        /// Explicit cubic rather than linear interpolation.
        cubic: bool,
    },
    /// Piecewise-constant basis.
    Fir {
        /// Basis count.
        columns: usize,
        /// Support span in seconds.
        span: f64,
    },
    /// Endpoint-constrained spline basis.
    BSpline {
        /// Basis count.
        columns: usize,
        /// Support span in seconds.
        span: f64,
        /// Spline degree.
        degree: usize,
    },
    /// Alternating sine/cosine basis.
    Fourier {
        /// Basis count.
        columns: usize,
        /// Support span in seconds.
        span: f64,
    },
}
impl HrfRecipe {
    /// Validate and construct without registry lookups or default substitution.
    pub fn build(&self) -> Result<Box<dyn Hrf>> {
        match self {
            Self::Canonical { normalize } => Ok(spm_canonical(
                None,
                None,
                None,
                None,
                None,
                Some(*normalize),
            )),
            Self::Spmg {
                columns,
                p1,
                p2,
                a1,
            } => {
                if [*p1, *p2, *a1].iter().any(|v| !v.is_finite() || *v <= 0.) {
                    return Err(HrfError::InvalidInput("invalid SPMG recipe".into()));
                }
                match columns {
                    1 => Ok(Box::new(SpmG1Hrf::new(*p1, *p2, *a1))),
                    2 => Ok(Box::new(SpmG2Hrf::new(*p1, *p2, *a1))),
                    3 => Ok(Box::new(SpmG3Hrf::new(*p1, *p2, *a1))),
                    _ => Err(HrfError::InvalidInput(
                        "SPMG recipe columns must be 1..3".into(),
                    )),
                }
            }
            Self::Empirical {
                times,
                values,
                name,
                span,
                cubic,
            } => empirical_hrf_with_method(
                &Array1::from_vec(times.clone()),
                &Array1::from_vec(values.clone()),
                Some(name),
                Some(*span),
                if *cubic {
                    InterpolationMethod::Cubic
                } else {
                    InterpolationMethod::Linear
                },
            ),
            Self::Fir { columns, span }
            | Self::BSpline { columns, span, .. }
            | Self::Fourier { columns, span } => {
                if *columns == 0 || !span.is_finite() || *span <= 0. {
                    return Err(HrfError::InvalidInput("invalid basis recipe".into()));
                }
                match self {
                    Self::Fir { .. } => Ok(fir_hrf(*columns, Some(*span))),
                    Self::Fourier { .. } => Ok(fourier_hrf(*columns, Some(*span))),
                    Self::BSpline { degree, .. } => {
                        if *degree == 0 || *columns < degree.saturating_sub(1) {
                            return Err(HrfError::InvalidInput(
                                "invalid spline recipe degree".into(),
                            ));
                        }
                        Ok(bspline_hrf(*columns, Some(*degree), Some(*span)))
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
