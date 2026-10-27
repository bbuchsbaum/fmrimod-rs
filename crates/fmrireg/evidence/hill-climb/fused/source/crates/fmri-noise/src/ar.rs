//! Fixed-order autoregression with explicit startup and estimation diagnostics.
use crate::{NoiseError, Pooling, Result};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, ArrayView2, ArrayViewMut2, Zip};

/// Treatment of the first observations in every independent segment.
#[derive(Debug, Clone, Copy)]
pub enum ArStart {
    /// Filter with unobserved history set to zero, matching fmriAR's AR(p>1) path.
    ZeroHistory,
    /// Whiten the first p observations using their stationary joint covariance.
    Stationary,
}

/// Estimation diagnostics before global coefficient pooling.
#[derive(Debug, Clone)]
pub struct ArDiagnostics {
    pub retained_observations: usize,
    /// Retained fraction of nonzero autocovariance lags (one means no PD repair).
    pub acvf_shrinkage: f64,
    pub stability_adjusted: bool,
    /// Zero for an explicitly uncorrected fit.
    pub correction_lags: usize,
}

/// Reusable, validated AR(p) transform. Coefficients may differ by run.
/// Empty coefficient vectors explicitly request IID for a known-parameter run.
#[derive(Debug, Clone)]
pub struct ArPlan {
    lengths: Vec<usize>,
    coefficients: Vec<Vec<f64>>,
    resets: Vec<bool>,
    startup: ArStart,
    initial: Vec<DMatrix<f64>>,
    pub(crate) diagnostics: Option<Vec<ArDiagnostics>>,
    pub(crate) pooling_adjusted: bool,
}
impl ArPlan {
    pub fn new(lengths: Vec<usize>, coefficients: Vec<Vec<f64>>, startup: ArStart) -> Result<Self> {
        if lengths.is_empty() || lengths.contains(&0) || coefficients.len() != lengths.len() {
            return Err(NoiseError::Invalid(
                "one coefficient vector per positive run length required",
            ));
        }
        let n = lengths
            .iter()
            .try_fold(0usize, |n, &r| n.checked_add(r))
            .ok_or(NoiseError::Invalid("run length overflow"))?;
        let mut initial = Vec::new();
        for phi in &coefficients {
            stationary(phi)?;
            let p = phi.len();
            let inverse = if matches!(startup, ArStart::Stationary) && p > 0 {
                let gamma = unit_acvf(phi)?;
                let covariance = DMatrix::from_fn(p, p, |i, j| gamma[i.abs_diff(j)]);
                let l = covariance
                    .cholesky()
                    .ok_or(NoiseError::Invalid(
                        "stationary startup covariance is not positive definite",
                    ))?
                    .l();
                l.try_inverse()
                    .ok_or(NoiseError::Invalid("singular stationary startup"))?
            } else {
                DMatrix::zeros(0, 0)
            };
            if inverse.iter().any(|v| !v.is_finite()) {
                return Err(NoiseError::Invalid("startup covariance overflow"));
            }
            initial.push(inverse);
        }
        let mut resets = vec![false; n];
        let mut start = 0;
        for &len in &lengths {
            resets[start] = true;
            start += len;
        }
        Ok(Self {
            lengths,
            coefficients,
            resets,
            startup,
            initial,
            diagnostics: None,
            pooling_adjusted: false,
        })
    }
    /// Segment starts are zero-based; rows are retained, including censored rows.
    pub fn with_resets(mut self, starts: &[usize]) -> Result<Self> {
        if starts.iter().any(|&i| i >= self.observations()) {
            return Err(NoiseError::Invalid("reset outside time axis"));
        }
        for &i in starts {
            self.resets[i] = true;
        }
        Ok(self)
    }
    pub fn observations(&self) -> usize {
        self.resets.len()
    }
    pub fn coefficients(&self) -> &[Vec<f64>] {
        &self.coefficients
    }
    pub fn diagnostics(&self) -> Option<&[ArDiagnostics]> {
        self.diagnostics.as_deref()
    }
    pub fn pooling_adjusted(&self) -> bool {
        self.pooling_adjusted
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
            return Err(NoiseError::Invalid(
                "AR matrix dimensions or nonfinite values",
            ));
        }
        self.apply_into_validated(values, out)
    }
    fn validate_values(&self, values: ArrayView2<'_, f64>) -> Result<()> {
        if values.nrows() != self.observations()
            || values.ncols() == 0
            || !crate::finite::all_finite(values)
        {
            return Err(NoiseError::Invalid(
                "AR matrix dimensions or nonfinite values",
            ));
        }
        Ok(())
    }
    fn apply_into_validated(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        let mut start = 0;
        for (run, &len) in self.lengths.iter().enumerate() {
            let phi = &self.coefficients[run];
            let mut segment = start;
            for t in start..start + len {
                if self.resets[t] {
                    segment = t;
                }
                let offset = t - segment;
                if matches!(self.startup, ArStart::Stationary) && offset < phi.len() {
                    out.row_mut(t).fill(0.0);
                    for k in 0..=offset {
                        let weight = self.initial[run][(offset, k)];
                        Zip::from(out.row_mut(t))
                            .and(values.row(segment + k))
                            .for_each(|output, &value| *output += weight * value);
                    }
                } else if phi.len() <= 4 {
                    // Keep the original per-feature `0.0 + phi[0] * y[t - 1]
                    // + ...` evaluation order, but visit each output feature only
                    // once. Zip handles arbitrary row strides, including negative
                    // strides, without a contiguity assumption.
                    match phi.len().min(offset) {
                        0 => Zip::from(out.row_mut(t))
                            .and(values.row(t))
                            .for_each(|output, &current| *output = current),
                        1 => Zip::from(out.row_mut(t))
                            .and(values.row(t))
                            .and(values.row(t - 1))
                            .for_each(|output, &current, &lag_1| {
                                let mut sum = 0.0;
                                sum += phi[0] * lag_1;
                                *output = current - sum;
                            }),
                        2 => Zip::from(out.row_mut(t))
                            .and(values.row(t))
                            .and(values.row(t - 1))
                            .and(values.row(t - 2))
                            .for_each(|output, &current, &lag_1, &lag_2| {
                                let mut sum = 0.0;
                                sum += phi[0] * lag_1;
                                sum += phi[1] * lag_2;
                                *output = current - sum;
                            }),
                        3 => Zip::from(out.row_mut(t))
                            .and(values.row(t))
                            .and(values.row(t - 1))
                            .and(values.row(t - 2))
                            .and(values.row(t - 3))
                            .for_each(|output, &current, &lag_1, &lag_2, &lag_3| {
                                let mut sum = 0.0;
                                sum += phi[0] * lag_1;
                                sum += phi[1] * lag_2;
                                sum += phi[2] * lag_3;
                                *output = current - sum;
                            }),
                        4 => Zip::from(out.row_mut(t))
                            .and(values.row(t))
                            .and(values.row(t - 1))
                            .and(values.row(t - 2))
                            .and(values.row(t - 3))
                            .and(values.row(t - 4))
                            .for_each(|output, &current, &lag_1, &lag_2, &lag_3, &lag_4| {
                                let mut sum = 0.0;
                                sum += phi[0] * lag_1;
                                sum += phi[1] * lag_2;
                                sum += phi[2] * lag_3;
                                sum += phi[3] * lag_4;
                                *output = current - sum;
                            }),
                        _ => unreachable!("small-order AR path has at most four lags"),
                    }
                } else {
                    // Keep the lag sum separate from the final subtraction: this
                    // preserves the original floating-point evaluation order.
                    out.row_mut(t).fill(0.0);
                    for (k, &coefficient) in phi.iter().take(offset).enumerate() {
                        Zip::from(out.row_mut(t))
                            .and(values.row(t - k - 1))
                            .for_each(|sum, &value| *sum += coefficient * value);
                    }
                    Zip::from(out.row_mut(t))
                        .and(values.row(t))
                        .for_each(|sum, &current| *sum = current - *sum);
                }
            }
            start += len;
        }
        if !crate::finite::all_finite(out.view()) {
            return Err(NoiseError::Invalid("AR whitening overflow"));
        }
        Ok(())
    }
}

// Schur step-down recursion: all reflection coefficients must lie inside (-1,1).
fn stationary(phi: &[f64]) -> Result<()> {
    if phi.iter().any(|x| !x.is_finite()) {
        return Err(NoiseError::Invalid("nonfinite AR coefficients"));
    }
    let mut a = phi.to_vec();
    while let Some(&k) = a.last() {
        if !k.is_finite() || k.abs() >= 1.0 {
            return Err(NoiseError::Invalid("nonstationary AR coefficients"));
        }
        let m = a.len() - 1;
        a = (0..m)
            .map(|j| (a[j] + k * a[m - j - 1]) / (1.0 - k * k))
            .collect();
    }
    Ok(())
}

// Solve stationary Yule-Walker equations with unit innovation variance.
fn unit_acvf(phi: &[f64]) -> Result<Vec<f64>> {
    let p = phi.len();
    let mut a = DMatrix::identity(p + 1, p + 1);
    for h in 0..=p {
        for (k, &v) in phi.iter().enumerate() {
            a[(h, h.abs_diff(k + 1))] -= v;
        }
    }
    let mut b = DVector::zeros(p + 1);
    b[0] = 1.0;
    let g = a
        .lu()
        .solve(&b)
        .ok_or(NoiseError::Invalid("singular AR covariance equations"))?;
    if g[0] <= 0.0 || g.iter().any(|v| !v.is_finite()) {
        return Err(NoiseError::Invalid("invalid AR covariance"));
    }
    Ok(g.as_slice().to_vec())
}

pub(crate) fn stabilize(phi: &[f64]) -> Result<(Vec<f64>, bool)> {
    let mut a = phi.to_vec();
    let mut pacf = vec![0.0; phi.len()];
    for m in (1..=phi.len()).rev() {
        let k = a[m - 1];
        pacf[m - 1] = k.clamp(-0.99, 0.99);
        let den = (1.0 - k * k).max(1e-12);
        a = (0..m - 1)
            .map(|j| (a[j] + k * a[m - j - 2]) / den)
            .collect();
    }
    let mut out = Vec::new();
    for k in pacf {
        let m = out.len();
        let mut next: Vec<f64> = (0..m).map(|j| out[j] - k * out[m - j - 1]).collect();
        next.push(k);
        out = next;
    }
    for _ in 0..60 {
        let p = out.len();
        let companion = DMatrix::from_fn(p, p, |i, j| {
            if i == 0 {
                out[j]
            } else if i == j + 1 {
                1.0
            } else {
                0.0
            }
        });
        let radius = companion
            .complex_eigenvalues()
            .iter()
            .map(|z| z.re.hypot(z.im))
            .fold(0.0_f64, f64::max);
        if radius.is_finite() && radius < 1.0 / (1.0 + 1e-6) && stationary(&out).is_ok() {
            let adjusted = out.iter().zip(phi).any(|(a, b)| (a - b).abs() > 1e-12);
            return Ok((out, adjusted));
        }
        for v in &mut out {
            *v *= 0.9;
        }
    }
    Err(NoiseError::Invalid("unable to stabilize AR coefficients"))
}

pub(crate) fn fit_acvf(gamma: &[f64], order: usize) -> Result<(Vec<f64>, f64, bool)> {
    if order == 0 || gamma.len() <= order || gamma.iter().any(|v| !v.is_finite()) || gamma[0] <= 0.0
    {
        return Err(NoiseError::Invalid(
            "invalid autocovariance or unsupported AR order",
        ));
    }
    let mut g: Vec<_> = gamma[..=order].iter().map(|v| v / gamma[0]).collect();
    let positive = |scale: f64| {
        let cov = DMatrix::from_fn(order + 1, order + 1, |i, j| {
            if i == j {
                1.0
            } else {
                scale * g[i.abs_diff(j)]
            }
        });
        cov.symmetric_eigen()
            .eigenvalues
            .iter()
            .all(|v| v.is_finite() && *v >= 1e-6)
    };
    let mut scale = 1.0;
    if !positive(scale) {
        let mut hi = 1.0;
        let mut lo = 0.0;
        for _ in 0..50 {
            let mid = (lo + hi) / 2.0;
            if positive(mid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        scale = lo;
    }
    for v in &mut g[1..] {
        *v *= scale;
    }
    let cov = DMatrix::from_fn(order, order, |i, j| g[i.abs_diff(j)]);
    let phi = cov
        .cholesky()
        .ok_or(NoiseError::Invalid("invalid Yule-Walker system"))?
        .solve(&DVector::from_column_slice(&g[1..]));
    let (phi, adjusted) = stabilize(phi.as_slice())?;
    Ok((phi, scale, adjusted))
}

pub(crate) struct FitPolicy {
    pub order: usize,
    pub pooling: Pooling,
    pub startup: ArStart,
    pub corrected: bool,
}

pub(crate) fn assemble(
    lengths: Vec<usize>,
    valid: &[bool],
    gammas: &[Vec<f64>],
    counts: &[usize],
    policy: FitPolicy,
) -> Result<ArPlan> {
    let FitPolicy {
        order,
        pooling,
        startup,
        corrected,
    } = policy;
    let mut coefficients = Vec::new();
    let mut diagnostics = Vec::new();
    for (g, &n) in gammas.iter().zip(counts) {
        let (phi, scale, adjusted) = fit_acvf(g, order)?;
        coefficients.push(phi);
        diagnostics.push(ArDiagnostics {
            retained_observations: n,
            acvf_shrinkage: scale,
            stability_adjusted: adjusted,
            correction_lags: if corrected { g.len() - 1 } else { 0 },
        });
    }
    let mut pooling_adjusted = false;
    if matches!(pooling, Pooling::Global) {
        let total = counts.iter().sum::<usize>() as f64;
        let pooled: Vec<_> = (0..order)
            .map(|j| {
                coefficients
                    .iter()
                    .zip(counts)
                    .map(|(c, n)| c[j] * (*n as f64 / total))
                    .sum()
            })
            .collect();
        let (pooled, adjusted) = stabilize(&pooled)?;
        pooling_adjusted = adjusted;
        coefficients.fill(pooled);
    }
    let resets: Vec<_> = (1..valid.len()).filter(|i| !valid[*i - 1]).collect();
    let mut plan = ArPlan::new(lengths, coefficients, startup)?.with_resets(&resets)?;
    plan.diagnostics = Some(diagnostics);
    plan.pooling_adjusted = pooling_adjusted;
    Ok(plan)
}

/// Fixed-order, explicitly uncorrected noise estimation. No run may silently
/// reduce the requested order. Censoring affects lag pairs, not regression rows.
pub fn estimate_ar_uncorrected(
    noise: ArrayView2<'_, f64>,
    lengths: Vec<usize>,
    valid: Option<&[bool]>,
    order: usize,
    pooling: Pooling,
    startup: ArStart,
) -> Result<ArPlan> {
    let shape = ArPlan::new(
        lengths.clone(),
        vec![vec![]; lengths.len()],
        ArStart::ZeroHistory,
    )?;
    if order == 0
        || order >= noise.nrows()
        || noise.nrows() != shape.observations()
        || noise.ncols() == 0
        || noise.iter().any(|v| !v.is_finite())
        || valid.is_some_and(|v| v.len() != noise.nrows())
    {
        return Err(NoiseError::Invalid("AR noise matrix, mask or order"));
    }
    let valid = valid.map_or_else(|| vec![true; noise.nrows()], |v| v.to_vec());
    let mut start = 0;
    let mut gammas = Vec::new();
    let mut counts = Vec::new();
    for &len in &lengths {
        let idx: Vec<_> = (start..start + len).filter(|i| valid[*i]).collect();
        gammas.push(autocovariance(noise, &idx, order)?);
        counts.push(idx.len());
        start += len;
    }
    assemble(
        lengths,
        &valid,
        &gammas,
        &counts,
        FitPolicy {
            order,
            pooling,
            startup,
            corrected: false,
        },
    )
}

pub(crate) fn autocovariance(
    values: ArrayView2<'_, f64>,
    idx: &[usize],
    lag: usize,
) -> Result<Vec<f64>> {
    if idx.len() <= lag {
        return Err(NoiseError::Invalid(
            "insufficient retained samples for requested order",
        ));
    }
    let means: Vec<_> = (0..values.ncols())
        .map(|j| idx.iter().map(|i| values[(*i, j)]).sum::<f64>() / idx.len() as f64)
        .collect();
    let mut gamma = vec![0.0; lag + 1];
    for (h, value) in gamma.iter_mut().enumerate() {
        let mut pairs = 0usize;
        for a in h..idx.len() {
            let b = a - h;
            if idx[a] - idx[b] != h {
                continue;
            }
            pairs += 1;
            for j in 0..values.ncols() {
                *value += (values[(idx[a], j)] - means[j]) * (values[(idx[b], j)] - means[j]);
            }
        }
        if pairs == 0 {
            return Err(NoiseError::Invalid(
                "no retained lag pairs for requested order",
            ));
        }
        *value /= pairs as f64 * values.ncols() as f64;
    }
    Ok(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonpositive_definite_acvf_repair_matches_r() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/arp.json")).unwrap();
        for c in f["repair"].as_array().unwrap() {
            let g: Vec<_> = c["gamma"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            let (phi, scale, _) = fit_acvf(&g, g.len() - 1).unwrap();
            assert!((scale - c["scale"].as_f64().unwrap()).abs() < 2e-10);
            for (a, b) in phi.iter().zip(c["phi"].as_array().unwrap()) {
                assert!((a - b.as_f64().unwrap()).abs() < 2e-8);
            }
        }
    }
}
