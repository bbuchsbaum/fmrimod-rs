//! Declared-family, two-sided max-T inference with an explicit transformation law.
use crate::{Error, GroupContrast, GroupDesign, IndependentCohort, Result};
use fmri_data::FeatureId;
use nalgebra::{DMatrix, DVector};
use std::collections::BTreeMap;

#[derive(Clone, Debug)]
pub enum PermutationMethod {
    /// Independent symmetric errors under the zero-mean null. Intercept-only.
    SignFlip,
    /// Permute reduced-model residuals within the declared row-bound blocks.
    /// This is generally approximate with nuisance covariates, even when all
    /// transformations are enumerated. Block labels do not establish validity.
    FreedmanLane { blocks: Vec<String> },
}
#[derive(Clone, Copy, Debug)]
pub enum PermutationSchedule {
    Exhaustive {
        max_transformations: usize,
    },
    /// Uniform draws with replacement; p=(1+exceedances)/(1+draws).
    MonteCarlo {
        draws: usize,
        seed: u64,
    },
}
#[derive(Clone, Debug)]
pub struct PermutationOptions {
    pub method: PermutationMethod,
    pub schedule: PermutationSchedule,
    /// Caller-supplied justification for symmetry/exchangeability under the null.
    pub justification: String,
    pub block_features: usize,
    /// Numeric workspace, including family statistics and the maximum distribution.
    /// Excludes resident input/design and metadata/string storage, not an RSS limit.
    pub max_working_bytes: usize,
    pub rank_tolerance: f64,
}
#[derive(Clone, Debug)]
pub struct PermutationFeature {
    pub feature: FeatureId,
    pub statistic: f64,
    pub p_uncorrected: f64,
    pub p_max_t: f64,
}
#[derive(Clone, Debug)]
pub struct PermutationResult {
    pub contrast: String,
    pub features: Vec<PermutationFeature>,
    pub max_abs_t: Vec<f64>,
    pub transformations: usize,
    pub residual_df: usize,
    pub numeric_working_bytes: usize,
    pub options: PermutationOptions,
    pub design: GroupDesign,
    pub subjects: Vec<crate::Subject>,
    pub estimand: crate::Estimand,
    pub admission: crate::Admission,
}

/// All features in `cohort` constitute one declared family. Select that family
/// before calling. Missing/nonfinite data and undefined observed t statistics
/// reject the call; execution blocks never define separate correction families.
pub fn permutation_max_t(
    cohort: &IndependentCohort,
    design: &GroupDesign,
    contrast: &GroupContrast,
    options: PermutationOptions,
) -> Result<PermutationResult> {
    let [n, nf] = cohort.frame().shape();
    let p = design.columns.len();
    if design.rows != cohort.frame().observations().ids()
        || contrast.columns != design.columns
        || options.justification.trim().is_empty()
        || options.block_features == 0
        || !options.rank_tolerance.is_finite()
        || options.rank_tolerance <= 0.
        || options.rank_tolerance >= 1.
        || n <= p
    {
        return Err(Error::Invalid(
            "permutation identity, dimensions or options",
        ));
    }
    let groups = match &options.method {
        PermutationMethod::SignFlip => {
            if p != 1
                || design
                    .values
                    .column(0)
                    .iter()
                    .any(|v| *v != design.values[(0, 0)])
            {
                return Err(Error::Invalid(
                    "sign flips require an intercept-only design",
                ));
            }
            Vec::new()
        }
        PermutationMethod::FreedmanLane { blocks } => {
            if blocks.len() != n || blocks.iter().any(|s| s.trim().is_empty()) {
                return Err(Error::Invalid("exchangeability block binding"));
            }
            let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
            for (i, label) in blocks.iter().enumerate() {
                grouped.entry(label).or_default().push(i);
            }
            grouped.into_values().collect::<Vec<_>>()
        }
    };
    let count = match options.schedule {
        PermutationSchedule::Exhaustive {
            max_transformations,
        } => {
            let total = if matches!(options.method, PermutationMethod::SignFlip) {
                u32::try_from(n).ok().and_then(|n| 1usize.checked_shl(n))
            } else {
                groups.iter().try_fold(1usize, |a, g| {
                    factorial(g.len()).and_then(|b| a.checked_mul(b))
                })
            }
            .ok_or(Error::Invalid("exhaustive transformation count overflow"))?;
            if total > max_transformations {
                return Err(Error::Invalid("exhaustive transformation limit"));
            }
            total
        }
        PermutationSchedule::MonteCarlo { draws, .. } => draws,
    };
    if count == 0 || count == usize::MAX {
        return Err(Error::Invalid(
            "positive bounded transformation count required",
        ));
    }
    // No full family data or transformation-by-subject matrix is retained.
    let cells = 12u128 * n as u128 * options.block_features.min(nf) as u128
        + 12u128 * n as u128 * (p + 1) as u128
        + 8u128 * p as u128 * p as u128
        + 8u128 * nf as u128
        + 2u128 * count as u128;
    let bytes = cells
        .checked_mul(8)
        .filter(|v| *v <= options.max_working_bytes as u128)
        .ok_or(Error::Budget)? as usize;
    let mut x = DMatrix::from_fn(n, p, |i, j| design.values[(i, j)]);
    let mut log_contrast = Vec::with_capacity(p);
    for j in 0..p {
        let scale = x.column(j).iter().map(|v| v.abs()).fold(0., f64::max);
        if scale == 0. || !scale.is_finite() {
            return Err(Error::Invalid("permutation design scaling"));
        }
        x.column_mut(j).scale_mut(1. / scale);
        log_contrast.push(if contrast.weights[j] == 0. {
            f64::NEG_INFINITY
        } else {
            contrast.weights[j].abs().ln() - scale.ln()
        });
    }
    let log_max = log_contrast
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let c = DVector::from_iterator(
        p,
        log_contrast
            .iter()
            .enumerate()
            .map(|(j, v)| contrast.weights[j].signum() * (v - log_max).exp()),
    );
    if c.iter()
        .enumerate()
        .any(|(j, v)| !v.is_finite() || (*v == 0. && contrast.weights[j] != 0.))
    {
        return Err(Error::Invalid("unsupported contrast scaling"));
    }
    let svd = x.svd(true, true);
    let limit = options.rank_tolerance * svd.singular_values.max();
    if svd.singular_values.iter().any(|s| *s <= limit) {
        return Err(Error::Invalid(
            "permutation design requires full column rank",
        ));
    }
    let u = svd.u.ok_or(Error::Invalid("SVD U"))?;
    let vt = svd.v_t.ok_or(Error::Invalid("SVD V"))?;
    let mut h = &u * (vt * &c).component_div(&svd.singular_values);
    let max = h.iter().map(|v| v.abs()).fold(0., f64::max);
    if !max.is_finite() || max == 0. {
        return Err(Error::Invalid("permutation contrast geometry"));
    }
    h /= max;
    let hnorm = h.norm();
    h /= hnorm;
    if h.iter().any(|v| !v.is_finite()) {
        return Err(Error::Invalid("permutation contrast geometry"));
    }
    let hnorm = 1.;
    let mut observed = Vec::with_capacity(nf);
    let mut counts = Vec::with_capacity(nf);
    let mut maxima = vec![0.0_f64; count];
    for start in (0..nf).step_by(options.block_features) {
        let end = start + options.block_features.min(nf - start);
        let values = crate::fit::read(cohort, cohort.effect_assay(), start, end, bytes)?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid(
                "permutation family must be complete and finite",
            ));
        }
        let y = DMatrix::from_fn(n, end - start, |i, j| values[(i, j)]);
        // Exact contrast-null projection H0 = UU' - h_normalized h_normalized'.
        let fitted = if p == 1 {
            DMatrix::zeros(n, end - start)
        } else {
            &u * (u.transpose() * &y) - &h * (h.transpose() * &y)
        };
        let residual = &y - &fitted;
        let mut local = vec![0usize; end - start];
        let mut obs = Vec::with_capacity(end - start);
        for j in 0..end - start {
            obs.push(t_stat(
                &y.column(j).into_owned(),
                &u,
                &h,
                hnorm,
                n - p,
                false,
            )?);
        }
        let seed = match options.schedule {
            PermutationSchedule::MonteCarlo { seed, .. } => seed,
            _ => 0,
        };
        let mut rng = SplitMix(seed);
        let mut transformed = DVector::zeros(n);
        let mut mapping: Vec<_> = (0..n).collect();
        let mut signs = vec![1.; n];
        for (b, maximum) in maxima.iter_mut().enumerate() {
            match &options.method {
                PermutationMethod::SignFlip => {
                    for (i, s) in signs.iter_mut().enumerate() {
                        let negative = match options.schedule {
                            PermutationSchedule::Exhaustive { .. } => (b >> i) & 1 != 0,
                            PermutationSchedule::MonteCarlo { .. } => rng.next() & 1 != 0,
                        };
                        *s = if negative { -1. } else { 1. };
                    }
                }
                PermutationMethod::FreedmanLane { .. } => {
                    let mut code = b;
                    for g in &groups {
                        let mut order = g.clone();
                        match options.schedule {
                            PermutationSchedule::MonteCarlo { .. } => {
                                for k in (1..order.len()).rev() {
                                    let ix = rng.index(k + 1);
                                    order.swap(k, ix);
                                }
                            }
                            PermutationSchedule::Exhaustive { .. } => {
                                let base = factorial(g.len())
                                    .ok_or(Error::Invalid("permutation count"))?;
                                let mut rank = code % base;
                                code /= base;
                                let mut pool = g.clone();
                                for (k, value) in order.iter_mut().enumerate() {
                                    let f = factorial(g.len() - k - 1)
                                        .ok_or(Error::Invalid("permutation count"))?;
                                    *value = pool.remove(rank / f);
                                    rank %= f;
                                }
                            }
                        }
                        for (&dst, &src) in g.iter().zip(&order) {
                            mapping[dst] = src;
                        }
                    }
                }
            }
            for j in 0..end - start {
                for i in 0..n {
                    transformed[i] = match options.method {
                        PermutationMethod::SignFlip => signs[i] * y[(i, j)],
                        PermutationMethod::FreedmanLane { .. } => {
                            fitted[(i, j)] + residual[(mapping[i], j)]
                        }
                    };
                }
                let t = t_stat(&transformed, &u, &h, hnorm, n - p, true)?.abs();
                *maximum = maximum.max(t);
                if exceeds(t, obs[j].abs()) {
                    local[j] += 1;
                }
            }
        }
        observed.extend(obs);
        counts.extend(local);
    }
    let add = usize::from(matches!(
        options.schedule,
        PermutationSchedule::MonteCarlo { .. }
    ));
    let features = observed
        .iter()
        .enumerate()
        .map(|(j, &statistic)| {
            let exceed = maxima
                .iter()
                .filter(|&&t| exceeds(t, statistic.abs()))
                .count();
            PermutationFeature {
                feature: cohort.frame().features().axis().ids()[j].clone(),
                statistic,
                p_uncorrected: (counts[j] + add) as f64 / (count + add) as f64,
                p_max_t: (exceed + add) as f64 / (count + add) as f64,
            }
        })
        .collect();
    Ok(PermutationResult {
        contrast: contrast.id.clone(),
        features,
        max_abs_t: maxima,
        transformations: count,
        residual_df: n - p,
        numeric_working_bytes: bytes,
        options,
        design: design.clone(),
        subjects: cohort.subjects().to_vec(),
        estimand: cohort.estimand().clone(),
        admission: cohort.admission().clone(),
    })
}
fn exceeds(value: f64, observed: f64) -> bool {
    value >= observed - 64. * f64::EPSILON * observed.max(1.)
}
fn factorial(n: usize) -> Option<usize> {
    (1..=n).try_fold(1usize, |a, b| a.checked_mul(b))
}
fn t_stat(
    y: &DVector<f64>,
    u: &DMatrix<f64>,
    h: &DVector<f64>,
    hnorm: f64,
    df: usize,
    allow_infinite: bool,
) -> Result<f64> {
    let mut r = y - u * (u.transpose() * y);
    r -= u * (u.transpose() * &r);
    let norm = r.norm();
    let effect = h.dot(y);
    if !norm.is_finite() || !effect.is_finite() {
        return Err(Error::Invalid("nonfinite permutation statistic"));
    }
    if norm <= 32. * f64::EPSILON * y.norm() {
        if allow_infinite && effect != 0. {
            return Ok(effect.signum() * f64::INFINITY);
        }
        return Err(Error::Invalid("undefined permutation statistic"));
    }
    let t = (effect / hnorm) / (norm / (df as f64).sqrt());
    if !t.is_finite() {
        return Err(Error::Invalid("permutation statistic overflow"));
    }
    Ok(t)
}
struct SplitMix(u64);
impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = self.next();
            if x >= threshold {
                return (x % n) as usize;
            }
        }
    }
}
