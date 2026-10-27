//! Run-aware nuisance columns and explicit observation censoring.
use crate::{
    baseline::InterceptType,
    basis::PolynomialBasis,
    design::DesignMatrix,
    types::{BasisFunction, ValidationError},
};
use fmrihrf::SamplingFrame;
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::collections::HashSet;

/// Drift is built on each run's complete original scan grid, before censoring.
#[derive(Debug, Clone, Copy)]
pub enum DriftBasis {
    None,
    /// Orthonormal polynomial columns, excluding the constant.
    Polynomial {
        degree: usize,
    },
    /// Orthonormal DCT-II components 1..floor(2*N*TR/cutoff_seconds).
    /// The cutoff endpoint is included; cutoff must exceed twice each run's TR.
    Dct {
        cutoff_seconds: f64,
    },
}
fn invalid(s: &str) -> ValidationError {
    ValidationError::InvalidParameter(s.into())
}
fn checked_frame(frame: &SamplingFrame) -> Result<SamplingFrame, ValidationError> {
    let f = SamplingFrame::new(
        frame.blocklens.clone(),
        frame.tr.clone(),
        Some(frame.start_time.clone()),
        frame.precision,
    )
    .map_err(|e| invalid(&e.to_string()))?;
    if f.samples().iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::NonFiniteValues);
    }
    Ok(f)
}
fn zeros(rows: usize, cols: usize) -> Result<Array2<f64>, ValidationError> {
    if rows
        .checked_mul(cols)
        .is_none_or(|n| n > isize::MAX as usize / 8)
    {
        return Err(invalid("design size overflow"));
    }
    Ok(Array2::zeros((rows, cols)))
}
/// Build run-local drift plus explicitly requested intercepts. No implicit intercept,
/// high-pass filtering, censoring, or nuisance-column removal is performed.
pub fn drift_design(
    frame: &SamplingFrame,
    basis: DriftBasis,
    intercept: InterceptType,
) -> Result<DesignMatrix, ValidationError> {
    let frame = checked_frame(frame)?;
    let counts: Vec<usize> = frame
        .blocklens
        .iter()
        .zip(&frame.tr)
        .map(|(&n, &tr)| match basis {
            DriftBasis::None => Ok(0),
            DriftBasis::Polynomial { degree } if degree < n => Ok(degree),
            DriftBasis::Polynomial { .. } => {
                Err(invalid("polynomial degree must be below every run length"))
            }
            DriftBasis::Dct { cutoff_seconds } => {
                if !cutoff_seconds.is_finite() || cutoff_seconds <= 0. {
                    return Err(invalid("DCT cutoff must be positive finite seconds"));
                }
                let extent = (2. * n as f64) * (tr / cutoff_seconds);
                if !extent.is_finite() || extent >= n as f64 {
                    return Err(invalid("DCT cutoff must exceed twice every run TR"));
                }
                Ok(extent.floor() as usize)
            }
        })
        .collect::<Result<_, _>>()?;
    let intercepts = match intercept {
        InterceptType::None => 0,
        InterceptType::Global => 1,
        InterceptType::Runwise => frame.n_blocks(),
    };
    let cols = counts
        .iter()
        .try_fold(intercepts, |acc, &n| acc.checked_add(n))
        .ok_or_else(|| invalid("column count overflow"))?;
    let mut data = zeros(frame.n_scans(), cols)?;
    let mut names = Vec::with_capacity(cols);
    let (mut row, mut col) = (0, 0);
    for (run, (&n, &count)) in frame.blocklens.iter().zip(&counts).enumerate() {
        if count > 0 {
            match basis {
                DriftBasis::Polynomial { degree } => {
                    let times = Array1::from_iter((1..=n).map(|i| i as f64));
                    let values = PolynomialBasis::new(degree).evaluate(&times.view())?;
                    data.slice_mut(s![row..row + n, col..col + count])
                        .assign(&values);
                }
                DriftBasis::Dct { .. } => {
                    let scale = (2. / n as f64).sqrt();
                    for i in 0..n {
                        for k in 1..=count {
                            data[(row + i, col + k - 1)] = scale
                                * (std::f64::consts::PI * (i as f64 + 0.5) * k as f64 / n as f64)
                                    .cos();
                        }
                    }
                }
                DriftBasis::None => unreachable!(),
            }
        }
        let tag = match basis {
            DriftBasis::Polynomial { .. } => "poly",
            _ => "dct",
        };
        names.extend((1..=count).map(|k| format!("drift.run{}.{}.{}", run + 1, tag, k)));
        col += count;
        row += n;
    }
    match intercept {
        InterceptType::None => {}
        InterceptType::Global => {
            data.column_mut(col).fill(1.);
            names.push("intercept".into());
        }
        InterceptType::Runwise => {
            let mut row = 0;
            for (run, &n) in frame.blocklens.iter().enumerate() {
                data.slice_mut(s![row..row + n, col + run]).fill(1.);
                names.push(format!("intercept.run{}", run + 1));
                row += n;
            }
        }
    }
    Ok(DesignMatrix::with_times(
        data,
        names,
        Array1::from_vec(frame.samples()),
    ))
}

/// Borrowed finite covariates for one run. Different runs may have different columns.
pub struct NuisanceRun<'a> {
    pub values: ArrayView2<'a, f64>,
    pub names: &'a [&'a str],
}
/// Assemble block-diagonal covariates with names `nuisance.run1.<name>`, etc.
/// Values remain in their original units. Missing/nonfinite values are errors;
/// constant or collinear columns are preserved for explicit rank assessment.
pub fn nuisance_design(
    frame: &SamplingFrame,
    runs: &[NuisanceRun<'_>],
) -> Result<DesignMatrix, ValidationError> {
    let frame = checked_frame(frame)?;
    if runs.len() != frame.n_blocks() {
        return Err(invalid("one nuisance matrix per run required"));
    }
    let mut cols = 0usize;
    for (run, &n) in runs.iter().zip(&frame.blocklens) {
        let mut names = HashSet::new();
        if run.values.nrows() != n || run.values.ncols() != run.names.len() {
            return Err(invalid("nuisance dimensions do not match run or names"));
        }
        if run.values.iter().any(|x| !x.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }
        if run
            .names
            .iter()
            .any(|s| s.trim().is_empty() || !names.insert(*s))
        {
            return Err(invalid(
                "nuisance names must be nonblank and unique per run",
            ));
        }
        cols = cols
            .checked_add(run.names.len())
            .ok_or_else(|| invalid("column count overflow"))?;
    }
    let mut data = zeros(frame.n_scans(), cols)?;
    let (mut row, mut col) = (0, 0);
    let mut names = Vec::with_capacity(cols);
    for (r, run) in runs.iter().enumerate() {
        let (n, p) = run.values.dim();
        data.slice_mut(s![row..row + n, col..col + p])
            .assign(&run.values);
        names.extend(
            run.names
                .iter()
                .map(|name| format!("nuisance.run{}.{name}", r + 1)),
        );
        row += n;
        col += p;
    }
    Ok(DesignMatrix::with_times(
        data,
        names,
        Array1::from_vec(frame.samples()),
    ))
}

/// Original-grid censor decisions: true means exclude/model with a spike.
/// This is not an AR noise-estimation mask. Selection never creates a compressed
/// SamplingFrame: use retained original indices to preserve run and gap identity.
#[derive(Debug, Clone)]
pub struct CensorPlan {
    frame: SamplingFrame,
    retained: Vec<usize>,
    censored: Vec<usize>,
}
impl CensorPlan {
    pub fn new(frame: &SamplingFrame, censored: &[bool]) -> Result<Self, ValidationError> {
        let frame = checked_frame(frame)?;
        if censored.len() != frame.n_scans() {
            return Err(invalid("censor mask must match the original scan count"));
        }
        let retained = (0..censored.len())
            .filter(|&i| !censored[i])
            .collect::<Vec<_>>();
        if retained.is_empty() {
            return Err(invalid("censoring must retain at least one observation"));
        }
        Ok(Self {
            frame,
            retained,
            censored: (0..censored.len()).filter(|&i| censored[i]).collect(),
        })
    }
    pub fn retained_indices(&self) -> &[usize] {
        &self.retained
    }
    /// One unit column per censored row, on the unchanged original grid.
    pub fn spikes(&self) -> Result<DesignMatrix, ValidationError> {
        let mut data = zeros(self.frame.n_scans(), self.censored.len())?;
        let mut names = Vec::new();
        for (col, &row) in self.censored.iter().enumerate() {
            data[(row, col)] = 1.;
            names.push(format!("censor.scan{}", row + 1));
        }
        Ok(DesignMatrix::with_times(
            data,
            names,
            Array1::from_vec(self.frame.samples()),
        ))
    }
    /// Select the same original rows from a response/covariate matrix. Values on
    /// excluded rows are not inspected; retained values must be finite.
    pub fn select_rows(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>, ValidationError> {
        if values.nrows() != self.frame.n_scans() {
            return Err(invalid("row selection requires the original scan count"));
        }
        let selected = values.select(Axis(0), &self.retained);
        if selected.iter().any(|v| !v.is_finite()) {
            return Err(ValidationError::NonFiniteValues);
        }
        Ok(selected)
    }
    /// Exclude rows from an already constructed design without recomputing drift
    /// or convolution, preserving original acquisition times and column names.
    pub fn select_design(&self, design: &DesignMatrix) -> Result<DesignMatrix, ValidationError> {
        if design.column_names.len() != design.ncols()
            || design.row_times.to_vec() != self.frame.samples()
        {
            return Err(invalid(
                "design must match the original acquisition grid and names",
            ));
        }
        Ok(DesignMatrix::with_times(
            self.select_rows(design.data.view())?,
            design.column_names.clone(),
            design.row_times.select(Axis(0), &self.retained),
        ))
    }
}
