//! Run-aware AR(1) correlation whitening. Innovation scale is estimated by the
//! downstream regression; this transform does not divide by innovation SD.
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};
use thiserror::Error;
mod finite;

#[derive(Debug, Error)]
pub enum NoiseError {
    #[error("invalid noise input: {0}")]
    Invalid(&'static str),
}
pub type Result<T> = std::result::Result<T, NoiseError>;

/// First observation policy at each independent run or explicitly reset segment.
#[derive(Debug, Clone, Copy)]
pub enum FirstSample {
    /// Stationary AR(1) likelihood: multiply the first sample by sqrt(1-phi²).
    Stationary,
    /// Zero prehistory: leave the first sample unchanged.
    ZeroHistory,
}

/// Validated AR(1) transform with one coefficient per run.
/// Explicit segment resets do not remove rows from the regression objective.
#[derive(Debug, Clone)]
pub struct Ar1Plan {
    run_lengths: Vec<usize>,
    phi: Vec<f64>,
    first: FirstSample,
    resets: Vec<bool>,
}
impl Ar1Plan {
    pub fn new(run_lengths: Vec<usize>, phi: Vec<f64>, first: FirstSample) -> Result<Self> {
        if run_lengths.is_empty()
            || run_lengths.contains(&0)
            || phi.len() != run_lengths.len()
            || phi.iter().any(|p| !p.is_finite() || p.abs() >= 1.0)
        {
            return Err(NoiseError::Invalid(
                "positive run lengths and one stationary finite phi per run required",
            ));
        }
        let n = run_lengths
            .iter()
            .try_fold(0usize, |n, r| n.checked_add(*r))
            .ok_or(NoiseError::Invalid("run length overflow"))?;
        let mut resets = vec![false; n];
        let mut start = 0;
        for len in &run_lengths {
            resets[start] = true;
            start += len;
        }
        Ok(Self {
            run_lengths,
            phi,
            first,
            resets,
        })
    }
    /// Add zero-based segment starts (for example, the sample after a censored row).
    /// Rows remain present. Excluding rows from the objective is a separate choice.
    pub fn with_resets(mut self, starts: &[usize]) -> Result<Self> {
        if starts.iter().any(|i| *i >= self.resets.len()) {
            return Err(NoiseError::Invalid("reset outside time axis"));
        }
        for i in starts {
            self.resets[*i] = true;
        }
        Ok(self)
    }
    pub fn coefficients(&self) -> &[f64] {
        &self.phi
    }
    pub fn observations(&self) -> usize {
        self.resets.len()
    }
    pub fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.validate_values(values)?;
        let mut out = Array2::zeros(values.raw_dim());
        self.apply_into_validated(values, out.view_mut())?;
        Ok(out)
    }
    /// Apply this plan into a caller-owned output matrix for repeated reuse.
    pub fn apply_into(
        &self,
        values: ArrayView2<'_, f64>,
        out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        self.validate_values(values)?;
        if out.dim() != values.dim() {
            return Err(NoiseError::Invalid("matrix dimensions or nonfinite values"));
        }
        self.apply_into_validated(values, out)
    }
    fn validate_values(&self, values: ArrayView2<'_, f64>) -> Result<()> {
        if values.nrows() != self.observations()
            || values.ncols() == 0
            || !finite::all_finite(values)
        {
            return Err(NoiseError::Invalid("matrix dimensions or nonfinite values"));
        }
        Ok(())
    }
    fn apply_into_validated(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        let mut start = 0;
        for (&len, &phi) in self.run_lengths.iter().zip(&self.phi) {
            let scale = match self.first {
                FirstSample::Stationary => (1.0 - phi * phi).sqrt(),
                FirstSample::ZeroHistory => 1.0,
            };
            for t in start..start + len {
                if self.resets[t] {
                    Zip::from(out.row_mut(t))
                        .and(values.row(t))
                        .for_each(|output, &value| *output = scale * value);
                } else {
                    Zip::from(out.row_mut(t))
                        .and(values.row(t))
                        .and(values.row(t - 1))
                        .for_each(|output, &current, &previous| *output = current - phi * previous);
                }
            }
            start += len;
        }
        if !finite::all_finite(out.view()) {
            return Err(NoiseError::Invalid("whitening overflow"));
        }
        Ok(())
    }
}

/// Pool run estimates by surviving observations, or retain a coefficient per run.
#[derive(Debug, Clone, Copy)]
pub enum Pooling {
    Global,
    Run,
}

/// Estimate fixed-order AR(1) from observed noise or *uncorrected* residuals.
///
/// Centers each feature within a run over retained observations; lag products
/// never span a run or a censored gap. Uses pair-count autocovariances and bounds
/// phi to [-0.99, 0.99], matching fmriAR's fixed AR(1) convention. This does NOT
/// correct the bias induced by fitting a design. Callers estimating noise from
/// fitted residuals must explicitly accept that limitation. Constant runs and
/// runs without valid lag pairs error instead of silently selecting IID noise.
/// `valid` affects estimation only; the returned plan retains every row and
/// restarts filtering after each excluded observation.
pub fn estimate_ar1_uncorrected(
    noise: ArrayView2<'_, f64>,
    run_lengths: Vec<usize>,
    valid: Option<&[bool]>,
    pooling: Pooling,
    first: FirstSample,
) -> Result<Ar1Plan> {
    let mut plan = Ar1Plan::new(run_lengths.clone(), vec![0.0; run_lengths.len()], first)?;
    if noise.nrows() != plan.observations()
        || noise.ncols() == 0
        || noise.iter().any(|x| !x.is_finite())
        || valid.is_some_and(|v| v.len() != noise.nrows())
    {
        return Err(NoiseError::Invalid("noise matrix or validity mask"));
    }
    let retained = |i: usize| valid.is_none_or(|v| v[i]);
    let mut start = 0;
    let mut counts = Vec::new();
    for (run, &len) in run_lengths.iter().enumerate() {
        let end = start + len;
        let n = (start..end).filter(|i| retained(*i)).count();
        let pairs = (start + 1..end)
            .filter(|i| retained(*i) && retained(*i - 1))
            .count();
        if n < 2 || pairs == 0 {
            return Err(NoiseError::Invalid("run has too few retained lag pairs"));
        }
        let mut g0 = 0.0;
        let mut g1 = 0.0;
        for j in 0..noise.ncols() {
            let mean = (start..end)
                .filter(|i| retained(*i))
                .map(|i| noise[(i, j)])
                .sum::<f64>()
                / n as f64;
            for i in start..end {
                if retained(i) {
                    let x = noise[(i, j)] - mean;
                    g0 += x * x;
                    if i > start && retained(i - 1) {
                        g1 += x * (noise[(i - 1, j)] - mean);
                    }
                }
            }
        }
        if g0 <= 0.0 || !g0.is_finite() || !g1.is_finite() {
            return Err(NoiseError::Invalid("zero or nonfinite noise variance"));
        }
        plan.phi[run] = (g1 / pairs as f64 / (g0 / n as f64)).clamp(-0.99, 0.99);
        counts.push(n);
        start = end;
    }
    if matches!(pooling, Pooling::Global) {
        let total: usize = counts.iter().sum();
        let phi = plan
            .phi
            .iter()
            .zip(&counts)
            .map(|(p, n)| p * (*n as f64 / total as f64))
            .sum();
        plan.phi.fill(phi);
    }
    if let Some(valid) = valid {
        for i in 1..valid.len() {
            if !valid[i - 1] {
                plan.resets[i] = true;
            }
        }
    }
    Ok(plan)
}

mod correction;
pub use correction::{Ar1Estimator, ArEstimator};

mod ar;
pub use ar::{estimate_ar_uncorrected, ArDiagnostics, ArPlan, ArStart};

/// Matched linear transform applied identically to design and data.
pub trait WhiteningPlan {
    fn observations(&self) -> usize;
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>>;
    /// Default compatibility path for custom plans. Built-in plans override this
    /// with specialized kernels; selected-observation AR uses O(p) scratch.
    fn apply_into(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        if out.dim() != values.dim() {
            return Err(NoiseError::Invalid("output matrix dimensions"));
        }
        let transformed = self.apply(values)?;
        if transformed.dim() != values.dim() {
            return Err(NoiseError::Invalid("transformed matrix dimensions"));
        }
        out.assign(&transformed);
        Ok(())
    }
}
impl WhiteningPlan for Ar1Plan {
    fn observations(&self) -> usize {
        self.observations()
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.apply(values)
    }
    fn apply_into(&self, values: ArrayView2<'_, f64>, out: ArrayViewMut2<'_, f64>) -> Result<()> {
        self.apply_into(values, out)
    }
}
impl WhiteningPlan for ArPlan {
    fn observations(&self) -> usize {
        self.observations()
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.apply(values)
    }
    fn apply_into(&self, values: ArrayView2<'_, f64>, out: ArrayViewMut2<'_, f64>) -> Result<()> {
        self.apply_into(values, out)
    }
}

mod selected;
pub use selected::{GapPolicy, ObservationSelection, SelectedArPlan};

pub mod selected_estimation;
