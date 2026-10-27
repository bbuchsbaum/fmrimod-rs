use crate::{Ar1Plan, FirstSample, NoiseError, Pooling, Result};
use nalgebra::DMatrix;
use ndarray::{Array2, ArrayView2};

/// Prepared design-bias correction for fixed AR(1) fitted from OLS residuals.
///
/// Corrects the finite-lag autocovariance mixing induced by the supplied design
/// and runwise centering. The lag budget is explicit: long-memory correlations
/// beyond it are not corrected. Preparation stores O(n * rank + runs * lag²)
/// values; it never materializes the n-by-n residual projection. It rejects
/// unidentifiable correction systems rather than silently using uncorrected noise.
pub struct Ar1Estimator {
    q: Array2<f64>,
    lengths: Vec<usize>,
    valid: Vec<bool>,
    indices: Vec<Vec<usize>>,
    inverses: Vec<DMatrix<f64>>,
    matrices: Vec<Array2<f64>>,
    pooling: Pooling,
}
impl Ar1Estimator {
    pub fn new(
        design: ArrayView2<'_, f64>,
        lengths: Vec<usize>,
        valid: Option<&[bool]>,
        max_lag: usize,
        pooling: Pooling,
    ) -> Result<Self> {
        let n = design.nrows();
        let p = design.ncols();
        let plan = Ar1Plan::new(
            lengths.clone(),
            vec![0.0; lengths.len()],
            FirstSample::Stationary,
        )?;
        if n != plan.observations()
            || p == 0
            || max_lag == 0
            || design.iter().any(|x| !x.is_finite())
            || valid.is_some_and(|v| v.len() != n)
        {
            return Err(NoiseError::Invalid(
                "correction design, runs, mask or lag budget",
            ));
        }
        let svd = DMatrix::from_fn(n, p, |i, j| design[(i, j)]).svd(true, false);
        let max = svd.singular_values[0];
        let rank = svd
            .singular_values
            .iter()
            .filter(|s| **s > max * 1e-12)
            .count();
        if n.saturating_sub(rank) < 3 {
            return Err(NoiseError::Invalid(
                "insufficient residual degrees of freedom for AR correction",
            ));
        }
        let u = svd.u.ok_or(NoiseError::Invalid("design SVD failed"))?;
        let q = Array2::from_shape_fn((n, rank), |(i, j)| u[(i, j)]);
        let valid = valid.map_or_else(|| vec![true; n], |v| v.to_vec());
        let mut indices = Vec::new();
        let mut matrices = Vec::new();
        let mut inverses = Vec::new();
        let mut start = 0;
        for &len in &lengths {
            let idx: Vec<_> = (start..start + len).filter(|i| valid[*i]).collect();
            if idx.len() < 3 {
                return Err(NoiseError::Invalid(
                    "too few retained samples for correction",
                ));
            }
            let mut longest = 0usize;
            let mut segment = 0usize;
            for &retained in &valid[start..start + len] {
                if retained {
                    segment += 1;
                    longest = longest.max(segment);
                } else {
                    segment = 0;
                }
            }
            let lag = max_lag.min(n - rank - 1).min(longest.saturating_sub(1));
            if lag == 0 {
                return Err(NoiseError::Invalid("no retained lag pairs"));
            }
            let a = bias_matrix(&q, &lengths, &idx, lag);
            let m = DMatrix::from_fn(lag + 1, lag + 1, |i, j| a[(i, j)]);
            let svd = m.clone().svd(false, false);
            let largest = svd.singular_values[0];
            let smallest = svd.singular_values[lag];
            if !largest.is_finite() || smallest <= largest * 1e-10 {
                return Err(NoiseError::Invalid("ill-conditioned AR correction system"));
            }
            let inv = m
                .lu()
                .try_inverse()
                .ok_or(NoiseError::Invalid("AR correction solve"))?;
            let norm = |matrix: &DMatrix<f64>| {
                matrix
                    .column_iter()
                    .map(|c| c.iter().map(|v| v.abs()).sum::<f64>())
                    .fold(0.0_f64, f64::max)
            };
            let original = DMatrix::from_fn(lag + 1, lag + 1, |i, j| a[(i, j)]);
            if 1.0 / (norm(&original) * norm(&inv)) < 1e-6 {
                return Err(NoiseError::Invalid(
                    "AR correction reciprocal condition below 1e-6; reduce lag budget",
                ));
            }
            indices.push(idx);
            matrices.push(a);
            inverses.push(inv);
            start += len;
        }
        Ok(Self {
            q,
            lengths,
            valid,
            indices,
            inverses,
            matrices,
            pooling,
        })
    }
    /// Per-run maps E[raw ACVF] = A * true ACVF, including runwise centering.
    pub fn bias_matrices(&self) -> &[Array2<f64>] {
        &self.matrices
    }

    /// Residuals must come from OLS projection by the preparation design.
    /// Numerical orthogonality is checked; callers remain responsible for provenance.
    fn corrected_acvf(&self, residuals: ArrayView2<'_, f64>) -> Result<Vec<Vec<f64>>> {
        if residuals.nrows() != self.q.nrows()
            || residuals.ncols() == 0
            || residuals.iter().any(|x| !x.is_finite())
        {
            return Err(NoiseError::Invalid("correction residual matrix"));
        }
        let norm = residuals.iter().map(|x| x * x).sum::<f64>();
        let projection = self.q.t().dot(&residuals);
        let projected = projection.iter().map(|x| x * x).sum::<f64>();
        if norm <= 0.0 || !norm.is_finite() || !projected.is_finite() || projected > norm * 1e-12 {
            return Err(NoiseError::Invalid(
                "residuals are not compatible with OLS design projection",
            ));
        }
        let mut gammas = Vec::new();
        for (idx, inverse) in self.indices.iter().zip(&self.inverses) {
            let l = inverse.nrows();
            let mut raw = vec![0.0; l];
            let mut pairs = vec![0usize; l];
            let means: Vec<f64> = (0..residuals.ncols())
                .map(|j| idx.iter().map(|i| residuals[(*i, j)]).sum::<f64>() / idx.len() as f64)
                .collect();
            for h in 0..l {
                for a in h..idx.len() {
                    let b = a - h;
                    if idx[a] - idx[b] != h {
                        continue;
                    }
                    pairs[h] += 1;
                    for j in 0..residuals.ncols() {
                        raw[h] += (residuals[(idx[a], j)] - means[j])
                            * (residuals[(idx[b], j)] - means[j]);
                    }
                }
                raw[h] /= (pairs[h] * residuals.ncols()) as f64;
            }
            let corrected = inverse * nalgebra::DVector::from_vec(raw);
            if corrected.iter().any(|x| !x.is_finite()) || corrected[0] <= 0.0 {
                return Err(NoiseError::Invalid("nonpositive corrected noise variance"));
            }
            gammas.push(corrected.as_slice().to_vec());
        }
        Ok(gammas)
    }
    pub fn fit(&self, residuals: ArrayView2<'_, f64>) -> Result<Ar1Plan> {
        let mut phi: Vec<f64> = self
            .corrected_acvf(residuals)?
            .iter()
            .map(|g| (g[1] / g[0]).clamp(-0.99, 0.99))
            .collect();
        if matches!(self.pooling, Pooling::Global) {
            let total: usize = self.indices.iter().map(Vec::len).sum();
            let pooled = phi
                .iter()
                .zip(&self.indices)
                .map(|(p, i)| p * i.len() as f64 / total as f64)
                .sum();
            phi.fill(pooled);
        }
        let resets: Vec<_> = (1..self.valid.len())
            .filter(|i| !self.valid[*i - 1])
            .collect();
        Ar1Plan::new(self.lengths.clone(), phi, FirstSample::Stationary)?.with_resets(&resets)
    }
}

// R = D - U V', where D selects retained rows; U=[1,Q_i-mean(Q_i)]
// and V=[mean retained-row indicator,Q]. Expand R S_k R' using these
// low-rank terms instead of constructing the dense residual-forming operator.
fn bias_matrix(q: &Array2<f64>, lengths: &[usize], idx: &[usize], lag: usize) -> Array2<f64> {
    let n = q.nrows();
    let d = q.ncols() + 1;
    let nv = idx.len();
    let mut u = Array2::zeros((nv, d));
    let mut v = Array2::zeros((n, d));
    for (a, &i) in idx.iter().enumerate() {
        u[(a, 0)] = 1.0;
        v[(i, 0)] = 1.0 / nv as f64;
    }
    for j in 1..d {
        let mean = idx.iter().map(|i| q[(*i, j - 1)]).sum::<f64>() / nv as f64;
        for (a, &i) in idx.iter().enumerate() {
            u[(a, j)] = q[(i, j - 1)] - mean;
        }
        for i in 0..n {
            v[(i, j)] = q[(i, j - 1)];
        }
    }
    let mut out = Array2::zeros((lag + 1, lag + 1));
    for k in 0..=lag {
        let mut skv = Array2::zeros((n, d));
        let mut start = 0;
        for &len in lengths {
            for i in start..start + len {
                for j in 0..d {
                    if k == 0 {
                        skv[(i, j)] = v[(i, j)];
                    } else {
                        if i >= start + k {
                            skv[(i, j)] += v[(i - k, j)];
                        }
                        if i + k < start + len {
                            skv[(i, j)] += v[(i + k, j)];
                        }
                    }
                }
            }
            start += len;
        }
        let us = u.dot(&v.t().dot(&skv));
        for h in 0..=lag {
            let mut count = 0;
            let mut sum = 0.0;
            for a in h..nv {
                let b = a - h;
                if idx[a] - idx[b] != h {
                    continue;
                }
                count += 1;
                let mut value = if idx[a].abs_diff(idx[b]) == k {
                    1.0
                } else {
                    0.0
                };
                for j in 0..d {
                    value -= skv[(idx[a], j)] * u[(b, j)] + u[(a, j)] * skv[(idx[b], j)];
                    value += us[(a, j)] * u[(b, j)];
                }
                sum += value;
            }
            out[(h, k)] = sum / count as f64;
        }
    }
    out
}

/// Prepared fixed-order AR(p) estimator for design-compatible OLS residuals.
/// Correction is performed at the full supported lag budget before truncating
/// to the AR order. Insufficient lag support errors rather than reducing order.
pub struct ArEstimator {
    correction: Ar1Estimator,
    order: usize,
    startup: crate::ArStart,
}
impl ArEstimator {
    pub fn new(
        design: ArrayView2<'_, f64>,
        lengths: Vec<usize>,
        valid: Option<&[bool]>,
        order: usize,
        max_lag: usize,
        pooling: Pooling,
        startup: crate::ArStart,
    ) -> Result<Self> {
        if order == 0 || max_lag < order {
            return Err(NoiseError::Invalid(
                "positive AR order and correction budget >= order required",
            ));
        }
        let correction = Ar1Estimator::new(design, lengths, valid, max_lag, pooling)?;
        if correction.inverses.iter().any(|a| a.nrows() <= order) {
            return Err(NoiseError::Invalid(
                "requested AR order exceeds supported correction lags",
            ));
        }
        Ok(Self {
            correction,
            order,
            startup,
        })
    }
    pub fn bias_matrices(&self) -> &[Array2<f64>] {
        self.correction.bias_matrices()
    }
    pub fn fit(&self, residuals: ArrayView2<'_, f64>) -> Result<crate::ArPlan> {
        let gammas = self.correction.corrected_acvf(residuals)?;
        let counts: Vec<_> = self.correction.indices.iter().map(Vec::len).collect();
        crate::ar::assemble(
            self.correction.lengths.clone(),
            &self.correction.valid,
            &gammas,
            &counts,
            crate::ar::FitPolicy {
                order: self.order,
                pooling: self.correction.pooling,
                startup: self.startup,
                corrected: true,
            },
        )
    }
}

#[cfg(test)]
mod arp_reference {
    use super::*;
    #[test]
    fn correction_stage_matches_r_with_consecutive_censor_rows() {
        let f: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/arp.json")).unwrap();
        let matrix = |v: &serde_json::Value| {
            let a = v.as_array().unwrap();
            Array2::from_shape_fn((a.len(), a[0].as_array().unwrap().len()), |(i, j)| {
                a[i][j].as_f64().unwrap()
            })
        };
        let x = matrix(&f["X"]);
        let residuals = matrix(&f["residuals"]);
        let mut valid = vec![true; 160];
        for i in f["censor"].as_array().unwrap() {
            valid[i.as_u64().unwrap() as usize - 1] = false;
        }
        let e = Ar1Estimator::new(x.view(), vec![80, 80], Some(&valid), 10, Pooling::Run).unwrap();
        for (r, a) in e.bias_matrices().iter().enumerate() {
            let expected = matrix(&f["bias"][r]);
            let delta = a
                .iter()
                .zip(&expected)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(delta < 1e-12, "bias run {r}: {delta}");
        }
        for (r, g) in e
            .corrected_acvf(residuals.view())
            .unwrap()
            .iter()
            .enumerate()
        {
            for (h, v) in g.iter().enumerate() {
                let expected = f["corrected_gamma"][r][h].as_f64().unwrap();
                assert!(
                    (v - expected).abs() < 1e-11,
                    "ACVF run={r} lag={h}: {v} != {expected}"
                );
            }
        }
    }
}
