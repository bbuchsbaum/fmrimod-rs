//! Explicit extensions for selecting, pooling, and scaling noise models.
//!
//! These APIs deliberately keep estimation, known-parameter whitening, and
//! plug-in scale estimation separate.  In particular, none of them removes a
//! row or substitutes IID noise after a failed fit.
use crate::{ar, NoiseError, Result, WhiteningPlan};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, ArrayView2, ArrayViewMut2};

fn checked_lengths(lengths: &[usize]) -> Result<usize> {
    if lengths.is_empty() || lengths.contains(&0) {
        return Err(NoiseError::Invalid("positive run lengths required"));
    }
    lengths
        .iter()
        .try_fold(0usize, |a, &b| a.checked_add(b))
        .ok_or(NoiseError::Invalid("run length overflow"))
}

fn solve_least_squares(design: &DMatrix<f64>, response: &DVector<f64>) -> Result<DVector<f64>> {
    if design.nrows() <= design.ncols()
        || design.iter().any(|x| !x.is_finite())
        || response.iter().any(|x| !x.is_finite())
    {
        return Err(NoiseError::Invalid(
            "insufficient or nonfinite least-squares regression",
        ));
    }
    // SVD gives a well-defined tall least-squares solve and a scale-relative
    // rank test. nalgebra's QR and column-pivoted QR only solve square systems.
    let (rows, cols) = design.shape();
    let svd = design.clone().svd(true, true);
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let tolerance = f64::EPSILON * rows.max(cols) as f64 * largest;
    if !largest.is_finite()
        || largest == 0.0
        || svd
            .singular_values
            .iter()
            .any(|&s| !s.is_finite() || s <= tolerance)
    {
        return Err(NoiseError::Invalid(
            "rank-deficient least-squares regression",
        ));
    }
    let beta = svd
        .solve(response, tolerance)
        .map_err(|_| NoiseError::Invalid("SVD least-squares solve failed"))?;
    if beta.iter().any(|x| !x.is_finite()) {
        return Err(NoiseError::Invalid("nonfinite least-squares coefficients"));
    }
    Ok(beta)
}

/// Information criterion for `select_ar_order_conditional`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderCriterion {
    Aic,
    Bic,
}

/// One conditional-likelihood AR order candidate.
#[derive(Debug, Clone)]
pub struct ArOrderCandidate {
    pub order: usize,
    /// All candidates use exactly this many trailing observations.
    pub observations: usize,
    pub innovation_variance: f64,
    pub score: f64,
    pub coefficients: Option<Vec<f64>>,
    pub failure: Option<&'static str>,
}

/// Auditable result of AR order selection.  The p=0 candidate is explicit.
#[derive(Debug, Clone)]
pub struct ArOrderSelection {
    pub criterion: OrderCriterion,
    pub common_lag: usize,
    pub selected: usize,
    pub candidates: Vec<ArOrderCandidate>,
}

/// Select an AR order using Gaussian conditional likelihood on a common sample.
///
/// The series is mean-centred once using a common estimated constant. Every
/// order, including p=0, is scored on `n - max_order` observations with
/// `n log(RSS/n) + penalty`; the reported penalty counts innovation variance
/// and AR coefficients, but deliberately omits the common mean parameter.
/// Failed candidates remain visible and are never replaced by a hidden IID fit.
pub fn select_ar_order_conditional(
    series: &[f64],
    max_order: usize,
    criterion: OrderCriterion,
) -> Result<ArOrderSelection> {
    if max_order == 0
        || max_order >= series.len().saturating_sub(1)
        || series.iter().any(|x| !x.is_finite())
    {
        return Err(NoiseError::Invalid(
            "finite series and feasible positive maximum order required",
        ));
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let y: Vec<f64> = series.iter().map(|x| x - mean).collect();
    let n = series.len() - max_order;
    let mut candidates = Vec::with_capacity(
        max_order
            .checked_add(1)
            .ok_or(NoiseError::Invalid("AR order overflow"))?,
    );
    for p in 0..=max_order {
        let (coefficients, residuals) = if p == 0 {
            (Some(Vec::new()), y[max_order..].to_vec())
        } else {
            let design = DMatrix::from_fn(n, p, |r, c| y[max_order + r - c - 1]);
            let response = DVector::from_column_slice(&y[max_order..]);
            match solve_least_squares(&design, &response) {
                Ok(beta) => {
                    let residuals = (&response - design * &beta).as_slice().to_vec();
                    let coeff = beta.as_slice().to_vec();
                    if ar::stationary(&coeff).is_ok() {
                        (Some(coeff), residuals)
                    } else {
                        candidates.push(ArOrderCandidate {
                            order: p,
                            observations: n,
                            innovation_variance: f64::NAN,
                            score: f64::INFINITY,
                            coefficients: None,
                            failure: Some("nonstationary conditional fit"),
                        });
                        continue;
                    }
                }
                Err(_) => {
                    candidates.push(ArOrderCandidate {
                        order: p,
                        observations: n,
                        innovation_variance: f64::NAN,
                        score: f64::INFINITY,
                        coefficients: None,
                        failure: Some("rank-deficient conditional fit"),
                    });
                    continue;
                }
            }
        };
        let variance = residuals.iter().map(|e| e * e).sum::<f64>() / n as f64;
        if !variance.is_finite() || variance <= 0.0 {
            candidates.push(ArOrderCandidate {
                order: p,
                observations: n,
                innovation_variance: variance,
                score: f64::INFINITY,
                coefficients,
                failure: Some("nonpositive innovation variance"),
            });
            continue;
        }
        let parameters = (p + 1) as f64;
        let penalty = match criterion {
            OrderCriterion::Aic => 2.0 * parameters,
            OrderCriterion::Bic => parameters * (n as f64).ln(),
        };
        candidates.push(ArOrderCandidate {
            order: p,
            observations: n,
            innovation_variance: variance,
            score: n as f64 * variance.ln() + penalty,
            coefficients,
            failure: None,
        });
    }
    let selected = candidates
        .iter()
        .filter(|c| c.failure.is_none())
        .min_by(|a, b| a.score.total_cmp(&b.score))
        .map(|c| c.order)
        .ok_or(NoiseError::Invalid("no valid AR order candidate"))?;
    Ok(ArOrderSelection {
        criterion,
        common_lag: max_order,
        selected,
        candidates,
    })
}

fn invertible(theta: &[f64]) -> Result<()> {
    let reflected: Vec<_> = theta.iter().map(|x| -*x).collect();
    ar::stationary(&reflected).map_err(|_| NoiseError::Invalid("noninvertible MA coefficients"))
}

/// Known stable/invertible ARMA coefficients, independently reset per run.
#[derive(Debug, Clone)]
pub struct ArmaPlan {
    lengths: Vec<usize>,
    ar: Vec<Vec<f64>>,
    ma: Vec<Vec<f64>>,
    resets: Vec<bool>,
}
impl ArmaPlan {
    pub fn new(
        lengths: Vec<usize>,
        ar_coefficients: Vec<Vec<f64>>,
        ma_coefficients: Vec<Vec<f64>>,
    ) -> Result<Self> {
        let total = checked_lengths(&lengths)?;
        if ar_coefficients.len() != lengths.len() || ma_coefficients.len() != lengths.len() {
            return Err(NoiseError::Invalid(
                "one ARMA coefficient pair per run required",
            ));
        }
        for (ar, ma) in ar_coefficients.iter().zip(&ma_coefficients) {
            ar::stationary(ar)?;
            invertible(ma)?;
        }
        let mut resets = vec![false; total];
        let mut start = 0;
        for &n in &lengths {
            resets[start] = true;
            start += n;
        }
        Ok(Self {
            lengths,
            ar: ar_coefficients,
            ma: ma_coefficients,
            resets,
        })
    }
    pub fn with_resets(mut self, starts: &[usize]) -> Result<Self> {
        if starts.iter().any(|&i| i >= self.resets.len()) {
            return Err(NoiseError::Invalid("reset outside time axis"));
        }
        for &i in starts {
            self.resets[i] = true;
        }
        Ok(self)
    }
    pub fn run_lengths(&self) -> &[usize] {
        &self.lengths
    }
    pub fn coefficients(&self) -> (&[Vec<f64>], &[Vec<f64>]) {
        (&self.ar, &self.ma)
    }
    pub fn observations(&self) -> usize {
        self.resets.len()
    }
    pub fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut out = Array2::zeros(values.raw_dim());
        self.apply_into(values, out.view_mut())?;
        Ok(out)
    }
    pub fn apply_into(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        if values.nrows() != self.observations()
            || values.ncols() == 0
            || out.dim() != values.dim()
            || values.iter().any(|x| !x.is_finite())
        {
            return Err(NoiseError::Invalid(
                "ARMA matrix dimensions or nonfinite values",
            ));
        }
        let mut begin = 0;
        for run in 0..self.lengths.len() {
            let end = begin + self.lengths[run];
            let mut segment = begin;
            for t in begin..end {
                if self.resets[t] {
                    segment = t;
                }
                for col in 0..values.ncols() {
                    let offset = t - segment;
                    let ar_sum: f64 = self.ar[run]
                        .iter()
                        .take(offset)
                        .enumerate()
                        .map(|(k, a)| a * values[(t - k - 1, col)])
                        .sum();
                    let ma_sum: f64 = self.ma[run]
                        .iter()
                        .take(offset)
                        .enumerate()
                        .map(|(k, a)| a * out[(t - k - 1, col)])
                        .sum();
                    out[(t, col)] = values[(t, col)] - ar_sum - ma_sum;
                }
            }
            begin = end;
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(NoiseError::Invalid("ARMA whitening overflow"));
        }
        Ok(())
    }
    /// Whiten matched design and response matrices with the same run/reset plan.
    pub fn apply_pair(
        &self,
        design: ArrayView2<'_, f64>,
        response: ArrayView2<'_, f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        if design.nrows() != response.nrows() {
            return Err(NoiseError::Invalid(
                "matched design and response rows required",
            ));
        }
        Ok((self.apply(design)?, self.apply(response)?))
    }
}
impl WhiteningPlan for ArmaPlan {
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

/// Explicit Hannan--Rissanen fit for one contiguous series.
#[derive(Debug, Clone)]
pub struct HannanRissanenFit {
    pub ar: Vec<f64>,
    pub ma: Vec<f64>,
    pub innovation_variance: f64,
    pub high_order: usize,
    pub observations: usize,
}
pub fn estimate_hannan_rissanen(
    series: &[f64],
    p: usize,
    q: usize,
    high_order: usize,
    iterations: usize,
) -> Result<HannanRissanenFit> {
    if series.len() < 10
        || high_order == 0
        || high_order >= series.len() / 2
        || p.checked_add(q).is_none()
        || p + q == 0
        || p >= series.len()
        || q >= series.len()
        || series.iter().any(|x| !x.is_finite())
    {
        return Err(NoiseError::Invalid(
            "single finite series and feasible positive ARMA orders required",
        ));
    }
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let y: Vec<_> = series.iter().map(|x| x - mean).collect();
    // HR requires this one declared high-order fit, not selection across every
    // smaller order. Reuse the same stable least-squares kernel directly.
    let initial_design = DMatrix::from_fn(y.len() - high_order, high_order, |r, c| {
        y[high_order + r - c - 1]
    });
    let initial_response = DVector::from_column_slice(&y[high_order..]);
    let init_phi = solve_least_squares(&initial_design, &initial_response)?
        .as_slice()
        .to_vec();
    ar::stationary(&init_phi)?;
    let mut e = vec![0.0; y.len()];
    for t in high_order..y.len() {
        e[t] = y[t]
            - init_phi
                .iter()
                .enumerate()
                .map(|(k, a)| a * y[t - k - 1])
                .sum::<f64>();
    }
    let m = high_order.max(p).max(q);
    let parameters = p
        .checked_add(q)
        .ok_or(NoiseError::Invalid("ARMA order overflow"))?;
    let minimum = m
        .checked_add(parameters)
        .ok_or(NoiseError::Invalid("ARMA order overflow"))?;
    if y.len() <= minimum {
        return Err(NoiseError::Invalid(
            "not enough observations for ARMA QR regression",
        ));
    }
    let mut ar_coeff = vec![0.0; p];
    let mut ma_coeff = vec![0.0; q];
    for _ in 0..=iterations {
        let rows = y.len() - m;
        let design = DMatrix::from_fn(rows, parameters, |r, c| {
            let t = m + r;
            if c < p {
                y[t - c - 1]
            } else {
                e[t - (c - p) - 1]
            }
        });
        let response = DVector::from_column_slice(&y[m..]);
        let beta = solve_least_squares(&design, &response)?;
        if p > 0 {
            ar_coeff.copy_from_slice(&beta.as_slice()[..p]);
        }
        if q > 0 {
            ma_coeff.copy_from_slice(&beta.as_slice()[p..]);
        }
        ar::stationary(&ar_coeff)?;
        invertible(&ma_coeff)?;
        e.fill(0.0);
        for t in 0..y.len() {
            let ar_sum: f64 = ar_coeff
                .iter()
                .take(t)
                .enumerate()
                .map(|(k, a)| a * y[t - k - 1])
                .sum();
            let ma_sum: f64 = ma_coeff
                .iter()
                .take(t)
                .enumerate()
                .map(|(k, a)| a * e[t - k - 1])
                .sum();
            e[t] = y[t] - ar_sum - ma_sum;
        }
    }
    let variance = e[m..].iter().map(|x| x * x).sum::<f64>() / (y.len() - m) as f64;
    if !variance.is_finite() || variance <= 0.0 {
        return Err(NoiseError::Invalid("invalid ARMA innovation variance"));
    }
    Ok(HannanRissanenFit {
        ar: ar_coeff,
        ma: ma_coeff,
        innovation_variance: variance,
        high_order,
        observations: y.len() - m,
    })
}

/// Parcel-specific AR plans retain labels in original column order.
/// Callers must preserve the response feature order when applying the plan.
#[derive(Debug, Clone)]
pub struct ParcelArPlan {
    labels: Vec<u32>,
    ids: Vec<u32>,
    plans: Vec<crate::ArPlan>,
}
impl ParcelArPlan {
    pub fn parcel_ids(&self) -> &[u32] {
        &self.ids
    }
    pub fn labels(&self) -> &[u32] {
        &self.labels
    }
    /// AR coefficients for one parcel, ordered by the original run order.
    pub fn coefficients(&self, parcel_id: u32) -> Option<&[Vec<f64>]> {
        self.ids
            .iter()
            .position(|&id| id == parcel_id)
            .map(|index| self.plans[index].coefficients())
    }
    /// Only this method is provided for parcel models: it returns a separately
    /// whitened design for each parcel alongside columns with that parcel's Y.
    pub fn apply_matched(
        &self,
        design: ArrayView2<'_, f64>,
        response: ArrayView2<'_, f64>,
    ) -> Result<ParcelWhitened> {
        if response.ncols() != self.labels.len() || design.nrows() != response.nrows() {
            return Err(NoiseError::Invalid(
                "per-parcel matched design and response required",
            ));
        }
        let mut data = Array2::zeros(response.raw_dim());
        let mut designs = Vec::with_capacity(self.ids.len());
        for (pid, plan) in self.ids.iter().zip(&self.plans) {
            let cols: Vec<_> = self
                .labels
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| (v == *pid).then_some(i))
                .collect();
            let y = response.select(ndarray::Axis(1), &cols);
            let yw = plan.apply(y.view())?;
            for (k, &col) in cols.iter().enumerate() {
                data.column_mut(col).assign(&yw.column(k));
            }
            designs.push(plan.apply(design)?);
        }
        Ok(ParcelWhitened {
            parcel_ids: self.ids.clone(),
            designs,
            response: data,
        })
    }
}
#[derive(Debug, Clone)]
pub struct ParcelWhitened {
    pub parcel_ids: Vec<u32>,
    pub designs: Vec<Array2<f64>>,
    pub response: Array2<f64>,
}

/// Estimate pooled parcel AR(p), shrinking each parcel ACVF toward the global
/// ACVF by the caller-provided convex weight `global_weight` in [0, 1].
pub fn estimate_parcel_ar(
    residuals: ArrayView2<'_, f64>,
    run_lengths: Vec<usize>,
    labels: Vec<u32>,
    order: usize,
    global_weight: f64,
    startup: crate::ArStart,
) -> Result<ParcelArPlan> {
    let total = checked_lengths(&run_lengths)?;
    if residuals.nrows() != total
        || residuals.ncols() == 0
        || labels.len() != residuals.ncols()
        || order == 0
        || !global_weight.is_finite()
        || !(0.0..=1.0).contains(&global_weight)
        || residuals.iter().any(|x| !x.is_finite())
    {
        return Err(NoiseError::Invalid(
            "parcel residuals, labels, order, or shrinkage",
        ));
    }
    let mut ids = labels.clone();
    ids.sort_unstable();
    ids.dedup();
    // Each ACVF is the average of feature-wise, within-run centred products.
    // It is deliberately not the ACVF of a parcel mean: anti-correlated voxel
    // columns must not cancel their common temporal covariance.
    let mut raw: Vec<(usize, Vec<Vec<f64>>)> = Vec::new();
    for &id in &ids {
        let cols: Vec<_> = labels
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| (v == id).then_some(i))
            .collect();
        raw.push((
            cols.len(),
            pooled_feature_acvf_by_run(residuals, &cols, &run_lengths, order)?,
        ));
    }
    let global: Vec<Vec<f64>> = (0..run_lengths.len())
        .map(|r| {
            (0..=order)
                .map(|lag| {
                    let n_features: usize = raw.iter().map(|(n, _)| *n).sum();
                    raw.iter().map(|(n, p)| *n as f64 * p[r][lag]).sum::<f64>() / n_features as f64
                })
                .collect()
        })
        .collect();
    let mut plans = Vec::new();
    for (_, parcel) in raw {
        let coefficients: Result<Vec<Vec<f64>>> = parcel
            .iter()
            .zip(&global)
            .map(|(local, all)| {
                let gamma: Vec<_> = local
                    .iter()
                    .zip(all)
                    .map(|(a, b)| (1.0 - global_weight) * a + global_weight * b)
                    .collect();
                strict_yule_walker(&gamma, order)
            })
            .collect();
        plans.push(crate::ArPlan::new(
            run_lengths.clone(),
            coefficients?,
            startup,
        )?);
    }
    Ok(ParcelArPlan { labels, ids, plans })
}
fn pooled_feature_acvf_by_run(
    values: ArrayView2<'_, f64>,
    columns: &[usize],
    lengths: &[usize],
    order: usize,
) -> Result<Vec<Vec<f64>>> {
    let mut out = Vec::new();
    let mut start = 0;
    for &n in lengths {
        if n <= order {
            return Err(NoiseError::Invalid("run too short for parcel AR order"));
        }
        let means: Vec<_> = columns
            .iter()
            .map(|&col| {
                (start..start + n)
                    .map(|row| values[(row, col)])
                    .sum::<f64>()
                    / n as f64
            })
            .collect();
        let g: Vec<_> = (0..=order)
            .map(|lag| {
                columns
                    .iter()
                    .zip(&means)
                    .map(|(&col, &mean)| {
                        (start + lag..start + n)
                            .map(|row| {
                                (values[(row, col)] - mean) * (values[(row - lag, col)] - mean)
                            })
                            .sum::<f64>()
                            / (n - lag) as f64
                    })
                    .sum::<f64>()
                    / columns.len() as f64
            })
            .collect();
        if g[0] <= 0.0 {
            return Err(NoiseError::Invalid("zero parcel variance"));
        }
        out.push(g);
        start += n;
    }
    Ok(out)
}

/// Strict AR fit: neither ACVF repair nor coefficient stabilization is applied.
fn strict_yule_walker(gamma: &[f64], order: usize) -> Result<Vec<f64>> {
    if gamma.len() <= order || gamma.iter().any(|x| !x.is_finite()) || gamma[0] <= 0.0 {
        return Err(NoiseError::Invalid("invalid parcel autocovariance"));
    }
    let covariance = DMatrix::from_fn(order, order, |i, j| gamma[i.abs_diff(j)]);
    let min_eigenvalue = covariance
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    if !min_eigenvalue.is_finite() || min_eigenvalue <= gamma[0] * 1e-10 {
        return Err(NoiseError::Invalid(
            "parcel autocovariance is not strictly positive definite",
        ));
    }
    let phi = covariance
        .cholesky()
        .ok_or(NoiseError::Invalid(
            "parcel Yule-Walker factorization failed",
        ))?
        .solve(&DVector::from_column_slice(&gamma[1..=order]));
    ar::stationary(phi.as_slice())?;
    Ok(phi.as_slice().to_vec())
}

/// Known per-run innovation SDs applied after an existing correlation plan.
#[derive(Debug, Clone)]
pub struct InnovationScaled<P> {
    inner: P,
    lengths: Vec<usize>,
    sd: Vec<f64>,
}
impl<P: WhiteningPlan> InnovationScaled<P> {
    pub fn new(inner: P, lengths: Vec<usize>, innovation_sd: Vec<f64>) -> Result<Self> {
        let n = checked_lengths(&lengths)?;
        if n != inner.observations()
            || innovation_sd.len() != lengths.len()
            || innovation_sd.iter().any(|x| !x.is_finite() || *x <= 0.0)
        {
            return Err(NoiseError::Invalid(
                "matching positive innovation SD per run required",
            ));
        }
        Ok(Self {
            inner,
            lengths,
            sd: innovation_sd,
        })
    }
    pub fn innovation_sd(&self) -> &[f64] {
        &self.sd
    }
}
impl<P: WhiteningPlan> WhiteningPlan for InnovationScaled<P> {
    fn observations(&self) -> usize {
        self.inner.observations()
    }
    fn apply(&self, values: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let mut out = self.inner.apply(values)?;
        self.scale_into(out.view_mut())?;
        Ok(out)
    }
    fn apply_into(
        &self,
        values: ArrayView2<'_, f64>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<()> {
        if out.dim() != values.dim() || values.nrows() != self.observations() || values.ncols() == 0
        {
            return Err(NoiseError::Invalid("scaled whitening output dimensions"));
        }
        self.inner.apply_into(values, out.view_mut())?;
        self.scale_into(out)
    }
}
impl<P: WhiteningPlan> InnovationScaled<P> {
    fn scale_into(&self, mut out: ArrayViewMut2<'_, f64>) -> Result<()> {
        if out.nrows() != self.observations()
            || out.ncols() == 0
            || out.iter().any(|x| !x.is_finite())
        {
            return Err(NoiseError::Invalid("nonfinite inner whitening output"));
        }
        let mut start = 0;
        for (&n, &sd) in self.lengths.iter().zip(&self.sd) {
            for mut row in out.slice_mut(ndarray::s![start..start + n, ..]).rows_mut() {
                row.mapv_inplace(|x| x / sd);
            }
            start += n;
        }
        if out.iter().any(|x| !x.is_finite()) {
            return Err(NoiseError::Invalid("innovation scale division overflow"));
        }
        Ok(())
    }
}

/// Residual plug-in innovation scales; this makes no uncertainty claim.
#[derive(Debug, Clone)]
pub struct InnovationScaleDiagnostics {
    pub innovation_sd: Vec<f64>,
    pub observations: Vec<usize>,
    pub features: usize,
    pub method: &'static str,
}
pub fn estimate_residual_innovation_scale<P: WhiteningPlan>(
    plan: &P,
    residuals: ArrayView2<'_, f64>,
    run_lengths: &[usize],
) -> Result<InnovationScaleDiagnostics> {
    let n = checked_lengths(run_lengths)?;
    if n != plan.observations() || residuals.nrows() != n || residuals.ncols() == 0 {
        return Err(NoiseError::Invalid(
            "matching residuals and run lengths required",
        ));
    }
    let e = plan.apply(residuals)?;
    if e.dim() != residuals.dim() || e.iter().any(|v| !v.is_finite()) {
        return Err(NoiseError::Invalid("invalid inner residual transform"));
    }
    let mut start = 0;
    let mut sd = Vec::new();
    for &len in run_lengths {
        let ss = e
            .slice(ndarray::s![start..start + len, ..])
            .iter()
            .map(|x| x * x)
            .sum::<f64>();
        let denom = len as f64 * e.ncols() as f64;
        if ss <= 0.0 || !ss.is_finite() {
            return Err(NoiseError::Invalid("nonpositive residual innovation scale"));
        }
        sd.push((ss / denom).sqrt());
        start += len;
    }
    Ok(InnovationScaleDiagnostics {
        innovation_sd: sd,
        observations: run_lengths.to_vec(),
        features: e.ncols(),
        method: "residual RMS plug-in; no uncertainty estimate",
    })
}
