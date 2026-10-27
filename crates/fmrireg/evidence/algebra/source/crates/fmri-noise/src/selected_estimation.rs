//! Finite-lag design-bias correction for stationary AR(1) on selected rows.
use crate::{
    ArStart, GapPolicy, NoiseError, ObservationSelection, Pooling, Result, SelectedArPlan,
};
use nalgebra::{DMatrix, DVector};
use ndarray::{Array2, ArrayView2};

type Pairs = Vec<(usize, usize)>;
/// Estimation diagnostics. Covariances use residuals divided by residual_scale;
/// ratios and fitted AR coefficients are invariant to this normalization.
#[derive(Debug, Clone)]
pub struct SelectedAr1Diagnostics {
    pub coefficients: Vec<f64>,
    pub corrected_acvf: Vec<Vec<f64>>,
    pub pair_counts: Vec<Vec<usize>>,
    pub residual_scale: f64,
    pub correction_lags: usize,
    pub correction_condition: f64,
    pub design_rank: usize,
}
/// Reusable correction E[raw moments] = A * covariance_lags. For Run pooling,
/// A couples all run covariance vectors through the selected design projector.
/// Covariance beyond max_lag is truncated, an explicit approximation for AR(1).
pub struct SelectedAr1Estimator {
    selection: ObservationSelection,
    gaps: GapPolicy,
    pooling: Pooling,
    lag: usize,
    q: DMatrix<f64>,
    runs: Vec<Vec<usize>>,
    pairs: Vec<Pairs>,
    correction: Array2<f64>,
    inverse: DMatrix<f64>,
    condition: f64,
}
impl SelectedAr1Estimator {
    /// selected_design must have one row per retained original observation.
    /// Every requested moment must have support; the lag budget is never reduced.
    pub fn new(
        selected_design: ArrayView2<'_, f64>,
        selection: ObservationSelection,
        gaps: GapPolicy,
        pooling: Pooling,
        max_lag: usize,
        relative_tolerance: f64,
    ) -> Result<Self> {
        let n = selected_design.nrows();
        let p = selected_design.ncols();
        if n != selection.retained_indices().len()
            || p == 0
            || max_lag == 0
            || max_lag >= selection.original_observations()
            || !relative_tolerance.is_finite()
            || relative_tolerance <= 0.
            || relative_tolerance >= 1.
            || selected_design.iter().any(|v| !v.is_finite())
        {
            return Err(NoiseError::Invalid(
                "selected correction design, tolerance or lag budget",
            ));
        }
        let svd = DMatrix::from_fn(n, p, |i, j| selected_design[(i, j)]).svd(true, false);
        let rank = svd
            .singular_values
            .iter()
            .filter(|&&s| s > svd.singular_values[0] * relative_tolerance)
            .count();
        if rank == 0 || n.saturating_sub(rank) < 3 {
            return Err(NoiseError::Invalid(
                "insufficient selected design rank or residual degrees of freedom",
            ));
        }
        let q = svd
            .u
            .ok_or(NoiseError::Invalid("selected design SVD failed"))?
            .columns(0, rank)
            .into_owned();
        let mut runs = vec![Vec::new(); selection.original_run_lengths().len()];
        let mut run_of = vec![0; n];
        let mut segment = vec![0; n];
        let mut run = 0;
        let mut end = selection.original_run_lengths()[0];
        let mut seg = 0;
        for (i, &row) in selection.retained_indices().iter().enumerate() {
            while row >= end {
                run += 1;
                end += selection.original_run_lengths()[run];
            }
            if i == 0 || run_of[i - 1] != run || row != selection.retained_indices()[i - 1] + 1 {
                seg += 1;
            }
            run_of[i] = run;
            segment[i] = seg;
            runs[run].push(i);
        }
        if matches!(pooling, Pooling::Run) && runs.iter().any(|r| r.len() < 3) {
            return Err(NoiseError::Invalid("runwise selected estimation needs at least three observations in every original run"));
        }
        let groups = if matches!(pooling, Pooling::Global) {
            1
        } else {
            runs.len()
        };
        let dim = groups
            .checked_mul(max_lag + 1)
            .ok_or(NoiseError::Invalid("correction size overflow"))?;
        if dim >= n {
            return Err(NoiseError::Invalid(
                "too many selected covariance moments for retained observations",
            ));
        }
        let mut pairs = vec![Vec::new(); dim];
        let rows = selection.retained_indices();
        for a in 0..n {
            let group = if matches!(pooling, Pooling::Global) {
                0
            } else {
                run_of[a]
            };
            for b in (0..=a).rev() {
                let lag = rows[a] - rows[b];
                if lag > max_lag || run_of[a] != run_of[b] {
                    break;
                }
                if matches!(gaps, GapPolicy::Restart) && segment[a] != segment[b] {
                    continue;
                }
                pairs[group * (max_lag + 1) + lag].push((a, b));
            }
        }
        if pairs.iter().any(Vec::is_empty) {
            return Err(NoiseError::Invalid(
                "unsupported selected lag: reduce explicit lag budget or change selection",
            ));
        }
        // R=C(I-QQ')=I-UV', where C removes each retained run mean.
        // B contains orthonormal run indicators; U=[B,CQ], V=[B,Q].
        let active: Vec<_> = runs.iter().filter(|r| !r.is_empty()).collect();
        let d = active.len() + rank;
        let mut u = DMatrix::zeros(n, d);
        let mut v = DMatrix::zeros(n, d);
        for (r, idx) in active.iter().enumerate() {
            let scale = 1. / (idx.len() as f64).sqrt();
            for &i in idx.iter() {
                u[(i, r)] = scale;
                v[(i, r)] = scale;
            }
            for j in 0..rank {
                let mean = idx.iter().map(|&i| q[(i, j)]).sum::<f64>() / idx.len() as f64;
                for &i in idx.iter() {
                    u[(i, active.len() + j)] = q[(i, j)] - mean;
                    v[(i, active.len() + j)] = q[(i, j)];
                }
            }
        }
        let mut correction = Array2::zeros((dim, dim));
        for k in 0..dim {
            let mut skv = DMatrix::zeros(n, d);
            for &(a, b) in &pairs[k] {
                for j in 0..d {
                    skv[(a, j)] += v[(b, j)];
                    if a != b {
                        skv[(b, j)] += v[(a, j)];
                    }
                }
            }
            let small = v.transpose() * &skv;
            let us = &u * small;
            for h in 0..dim {
                let mut total = 0.;
                for &(a, b) in &pairs[h] {
                    let mut value = if h == k { 1. } else { 0. };
                    for j in 0..d {
                        value -= skv[(a, j)] * u[(b, j)] + u[(a, j)] * skv[(b, j)];
                        value += us[(a, j)] * u[(b, j)];
                    }
                    total += value;
                }
                correction[(h, k)] = total / pairs[h].len() as f64;
            }
        }
        let a = DMatrix::from_fn(dim, dim, |i, j| correction[(i, j)]);
        let s = a.clone().svd(false, false).singular_values;
        let condition = s[0] / s[dim - 1];
        if !condition.is_finite() || condition > 1e6 {
            return Err(NoiseError::Invalid(
                "ill-conditioned selected AR correction; reduce lag budget",
            ));
        }
        let inverse = a
            .lu()
            .try_inverse()
            .ok_or(NoiseError::Invalid("selected AR correction solve failed"))?;
        if inverse.iter().any(|v| !v.is_finite()) {
            return Err(NoiseError::Invalid("selected AR correction overflow"));
        }
        Ok(Self {
            selection,
            gaps,
            pooling,
            lag: max_lag,
            q,
            runs,
            pairs,
            correction,
            inverse,
            condition,
        })
    }
    pub fn bias_matrix(&self) -> &Array2<f64> {
        &self.correction
    }
    /// Residuals must be from OLS on this retained design. Orthogonality is checked;
    /// callers remain responsible for the original row identities and provenance.
    pub fn fit(
        &self,
        residuals: ArrayView2<'_, f64>,
    ) -> Result<(SelectedArPlan, SelectedAr1Diagnostics)> {
        let (n, features) = residuals.dim();
        if n != self.q.nrows() || features == 0 || residuals.iter().any(|v| !v.is_finite()) {
            return Err(NoiseError::Invalid(
                "selected correction residual dimensions or values",
            ));
        }
        let scale = residuals.iter().map(|v| v.abs()).fold(0., f64::max);
        if scale <= 0. {
            return Err(NoiseError::Invalid("zero selected residual variance"));
        }
        let mut raw = vec![0.; self.pairs.len()];
        let mut norm = 0.;
        let mut projected = 0.;
        let mut centered = vec![0.; n];
        for j in 0..features {
            for i in 0..n {
                centered[i] = residuals[(i, j)] / scale;
                norm += centered[i] * centered[i];
            }
            for k in 0..self.q.ncols() {
                let z = (0..n).map(|i| self.q[(i, k)] * centered[i]).sum::<f64>();
                projected += z * z;
            }
            for idx in &self.runs {
                if !idx.is_empty() {
                    let mean = idx.iter().map(|&i| centered[i]).sum::<f64>() / idx.len() as f64;
                    for &i in idx {
                        centered[i] -= mean;
                    }
                }
            }
            for (h, pairs) in self.pairs.iter().enumerate() {
                raw[h] += pairs
                    .iter()
                    .map(|&(a, b)| centered[a] * centered[b])
                    .sum::<f64>()
                    / pairs.len() as f64
                    / features as f64;
            }
        }
        if !norm.is_finite() || !projected.is_finite() || projected > norm * 1e-12 {
            return Err(NoiseError::Invalid(
                "residuals incompatible with selected OLS projection",
            ));
        }
        let corrected = &self.inverse * DVector::from_vec(raw);
        let mut gammas = Vec::new();
        let mut phi = Vec::new();
        for group in 0..corrected.len() / (self.lag + 1) {
            let g = corrected
                .rows(group * (self.lag + 1), self.lag + 1)
                .iter()
                .copied()
                .collect::<Vec<_>>();
            if g.iter().any(|v| !v.is_finite()) || g[0] <= 0. {
                return Err(NoiseError::Invalid(
                    "nonpositive corrected selected variance",
                ));
            }
            let coefficient = g[1] / g[0];
            if !coefficient.is_finite() || coefficient.abs() >= 1. {
                return Err(NoiseError::Invalid(
                    "nonstationary corrected selected AR1 estimate; no clipping or fallback",
                ));
            }
            gammas.push(g);
            phi.push(coefficient);
        }
        let coefficients = if matches!(self.pooling, Pooling::Global) {
            vec![phi[0]; self.runs.len()]
        } else {
            phi
        };
        let plan = SelectedArPlan::new(
            self.selection.clone(),
            coefficients.iter().map(|&p| vec![p]).collect(),
            ArStart::Stationary,
            self.gaps,
        )?;
        let diagnostics = SelectedAr1Diagnostics {
            coefficients,
            corrected_acvf: gammas,
            pair_counts: self
                .pairs
                .chunks(self.lag + 1)
                .map(|chunk| chunk.iter().map(Vec::len).collect())
                .collect(),
            residual_scale: scale,
            correction_lags: self.lag,
            correction_condition: self.condition,
            design_rank: self.q.ncols(),
        };
        Ok((plan, diagnostics))
    }
}
