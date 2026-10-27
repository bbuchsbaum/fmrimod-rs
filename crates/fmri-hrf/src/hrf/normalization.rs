//! Fixed-reference HRF normalization, independent of the evaluation grid.

use crate::{Hrf, HrfError, HrfValues, Result, TimeArray};
use ndarray::Array1;
use std::sync::OnceLock;

/// Scaling convention matching R's `normalize_hrf()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// Retain the raw HRF and relative basis scales.
    #[default]
    None,
    /// Divide all columns by the canonical sum on 1600 points from 0 to 32 s.
    Spm,
    /// Divide all columns by the canonical peak on a fixed 50 Hz reference grid.
    UnitPeak,
    /// Divide all columns by the canonical trapezoid integral on that grid.
    UnitIntegral,
    /// Independently normalize each basis column on that grid.
    UnitPeakPerBasis,
}

/// Normalize an HRF with factors computed once, preserving derivative scaling
/// except when `UnitPeakPerBasis` is explicitly selected.
pub fn normalize_hrf_with(hrf: Box<dyn Hrf>, mode: Normalization) -> Result<Box<dyn Hrf>> {
    if mode == Normalization::None {
        return Ok(hrf);
    }
    let scales = factors(hrf.as_ref(), mode)?;
    let cell = OnceLock::new();
    let _ = cell.set(Ok(scales));
    Ok(Box::new(FixedNormalizedHrf {
        inner: hrf,
        mode,
        scales: cell,
    }))
}

pub(super) fn normalize_legacy(hrf: Box<dyn Hrf>) -> Box<dyn Hrf> {
    Box::new(FixedNormalizedHrf {
        inner: hrf,
        mode: Normalization::UnitPeakPerBasis,
        scales: OnceLock::new(),
    })
}

#[derive(Debug)]
struct FixedNormalizedHrf {
    inner: Box<dyn Hrf>,
    mode: Normalization,
    scales: OnceLock<std::result::Result<Vec<f64>, String>>,
}

impl Hrf for FixedNormalizedHrf {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues> {
        let scales = self
            .scales
            .get_or_init(|| factors(self.inner.as_ref(), self.mode).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| HrfError::InvalidParameters(e.clone()))?;
        let mut values = self.inner.evaluate(t)?;
        crate::regressor::convolution::check_shape(&values, t.len(), self.nbasis())?;
        for (mut col, &scale) in values.columns_mut().into_iter().zip(scales) {
            col /= scale;
        }
        Ok(values)
    }
    fn name(&self) -> &str {
        "normalized_hrf"
    }
    fn nbasis(&self) -> usize {
        self.inner.nbasis()
    }
    fn span(&self) -> f64 {
        self.inner.span()
    }
    fn params(&self) -> Vec<(&str, f64)> {
        let mut p = self.inner.params();
        p.push(("normalization", self.mode as u8 as f64));
        p
    }
}

fn factors(hrf: &dyn Hrf, mode: Normalization) -> Result<Vec<f64>> {
    let span = hrf.span();
    if !span.is_finite() || span <= 0.0 || hrf.nbasis() == 0 {
        return Err(HrfError::InvalidParameters(
            "normalization requires a positive finite span and at least one basis".into(),
        ));
    }
    let grid = if mode == Normalization::Spm {
        Array1::linspace(0.0, 32.0, 1600)
    } else {
        let intervals = (span * 50.0).round_ties_even();
        if !intervals.is_finite() || intervals >= (isize::MAX as usize / 8) as f64 {
            return Err(HrfError::InvalidParameters(
                "normalization reference grid is too large".into(),
            ));
        }
        let n = intervals as usize + 1;
        Array1::linspace(0.0, span, n.max(2))
    };
    let values = hrf.evaluate(&grid)?;
    crate::regressor::convolution::check_shape(&values, grid.len(), hrf.nbasis())?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::InvalidParameters(
            "HRF reference values must be finite".into(),
        ));
    }
    if mode == Normalization::UnitPeakPerBasis {
        return Ok(values
            .columns()
            .into_iter()
            .map(|col| {
                let peak = col.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
                if peak == 0.0 {
                    1.0
                } else {
                    peak
                }
            })
            .collect());
    }
    let first = values.column(0);
    let factor = match mode {
        Normalization::None => 1.0,
        Normalization::Spm => first.sum(),
        Normalization::UnitPeak => first.iter().fold(0.0_f64, |a, &b| a.max(b.abs())),
        Normalization::UnitIntegral => (1..grid.len())
            .map(|i| (grid[i] - grid[i - 1]) * (first[i] + first[i - 1]) / 2.0)
            .sum(),
        Normalization::UnitPeakPerBasis => unreachable!(),
    };
    if !factor.is_finite() || factor.abs() < f64::MIN_POSITIVE {
        return Err(HrfError::InvalidParameters(
            "normalization factor is zero or nonfinite".into(),
        ));
    }
    Ok(vec![factor; hrf.nbasis()])
}
