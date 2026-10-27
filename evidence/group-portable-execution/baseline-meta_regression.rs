//! Independent-subject weighted regression with explicitly requested inference.
use crate::{
    Distribution, Error, FitOptions, GroupContrast, GroupDesign, IndependentCohort, Missingness,
    ModelReceipt, Result, RunReceipt,
};
use fmridataset::FeatureId;
use nalgebra::{linalg::ColPivQR, DMatrix, DVector, Dyn};
use ndarray::Array2;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeterogeneityEstimator {
    CommonEffect,
    DerSimonianLaird,
    PauleMandel,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaInference {
    Normal,
    ModifiedKnappHartung,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaUncertainty {
    None,
    Marginal,
    Joint,
}
/// Effects are always requested. Statistics/p-values and confidence intervals
/// require at least marginal uncertainty; invalid combinations are rejected.
#[derive(Clone, Debug)]
pub struct MetaOutputSpec {
    pub contrasts: Vec<GroupContrast>,
    pub uncertainty: MetaUncertainty,
    pub p_values: bool,
    pub confidence_level: Option<f64>,
}
impl MetaOutputSpec {
    pub fn new(contrasts: Vec<GroupContrast>) -> Self {
        Self {
            contrasts,
            uncertainty: MetaUncertainty::None,
            p_values: false,
            confidence_level: None,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct PauleMandelOptions {
    /// Relative tolerance for Q/(n-p), independent of measurement units.
    pub q_tolerance: f64,
    /// Includes the initial fixed-effect fit and bracketing evaluations.
    pub max_evaluations: usize,
}
impl Default for PauleMandelOptions {
    fn default() -> Self {
        Self {
            q_tolerance: 1e-12,
            max_evaluations: 128,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct MetaRegressionOptions {
    pub execution: FitOptions,
    pub heterogeneity: HeterogeneityEstimator,
    pub inference: MetaInference,
    pub paule_mandel: PauleMandelOptions,
}
impl Default for MetaRegressionOptions {
    fn default() -> Self {
        Self {
            execution: FitOptions::default(),
            heterogeneity: HeterogeneityEstimator::CommonEffect,
            inference: MetaInference::Normal,
            paule_mandel: PauleMandelOptions::default(),
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaStatus {
    Estimated,
    MissingObservation,
    InsufficientObservations,
    RankDeficient,
    NumericalFailure,
    NonConvergence,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TauConvergence {
    Fixed,
    Boundary,
    ClosedForm,
    Converged,
    EvaluationLimit,
    NumericalFailure,
    RankDeficient,
}
#[derive(Clone, Debug)]
pub struct MetaDiagnostics {
    pub q_fixed: f64,
    pub q_final: f64,
    pub tau_squared: f64,
    pub residual_df: usize,
    pub covariance_scale: f64,
    pub convergence: TauConvergence,
    pub evaluations: usize,
    pub newton_steps: usize,
    pub bisection_steps: usize,
    pub bracket: Option<[f64; 2]>,
    /// Smallest/largest absolute diagonal of the column-scaled pivoted R.
    pub qr_diagonal_ratio: f64,
    pub dl_complement_fallback: bool,
}
#[derive(Clone, Debug)]
pub struct MetaEstimates {
    pub effects: Vec<f64>,
    pub standard_errors: Option<Vec<f64>>,
    pub joint_covariance: Option<Array2<f64>>,
    pub statistics: Option<Vec<f64>>,
    pub p_values: Option<Vec<f64>>,
    pub confidence_intervals: Option<Vec<[f64; 2]>>,
    pub distribution: Distribution,
}
#[derive(Clone, Debug)]
pub struct MetaRegressionFeature {
    pub feature: FeatureId,
    pub n: usize,
    pub rank: usize,
    pub status: MetaStatus,
    pub estimates: Option<MetaEstimates>,
    pub diagnostics: Option<MetaDiagnostics>,
}
#[derive(Clone, Debug)]
pub struct MetaRegressionBlock {
    pub start: usize,
    pub contrasts: Vec<String>,
    pub features: Vec<MetaRegressionFeature>,
}
#[derive(Clone, Debug)]
pub struct MetaRegressionReceipt {
    pub run: RunReceipt,
    pub options: MetaRegressionOptions,
    pub outputs: MetaOutputSpec,
    pub weighted_evaluations: usize,
}

/// Prepared metadata and bounded retained-row buffers. Each feature and each
/// candidate tau² receives its own weighted QR; unequal weights never reuse
/// another feature's geometry. The existing first-level solver is unchanged.
#[derive(Debug)]
pub struct PreparedMetaRegression {
    design: GroupDesign,
    outputs: MetaOutputSpec,
    options: MetaRegressionOptions,
    constant: Option<usize>,
    rows: Vec<usize>,
    y: Vec<f64>,
    v: Vec<f64>,
}
impl PreparedMetaRegression {
    pub fn new(
        design: &GroupDesign,
        outputs: MetaOutputSpec,
        options: MetaRegressionOptions,
    ) -> Result<Self> {
        let p = design.columns.len();
        if outputs.contrasts.is_empty()
            || outputs
                .contrasts
                .iter()
                .any(|c| c.columns != design.columns)
            || outputs
                .contrasts
                .iter()
                .map(|c| &c.id)
                .collect::<HashSet<_>>()
                .len()
                != outputs.contrasts.len()
            || (outputs.uncertainty == MetaUncertainty::None
                && (outputs.p_values || outputs.confidence_level.is_some()))
            || outputs
                .confidence_level
                .is_some_and(|l| !l.is_finite() || l <= 0. || l >= 1.)
            || !options.paule_mandel.q_tolerance.is_finite()
            || options.paule_mandel.q_tolerance <= 0.
            || options.paule_mandel.q_tolerance >= 1.
            || options.paule_mandel.max_evaluations == 0
        {
            return Err(Error::Invalid("meta-regression output or solver options"));
        }
        working_bytes(design.rows.len(), p, &outputs, &options.execution, 1)?;
        if design.rows.len() <= p {
            return Err(Error::Invalid("meta-regression requires n > p"));
        }
        let constant = (0..p).find(|&j| {
            design.values[(0, j)] != 0.
                && design
                    .values
                    .column(j)
                    .iter()
                    .all(|&v| v == design.values[(0, j)])
        });
        let rows = (0..design.rows.len()).collect::<Vec<_>>();
        let y = vec![0.; rows.len()];
        let v = vec![1.; rows.len()];
        weighted_fit(
            design,
            &rows,
            &y,
            &v,
            0.,
            constant,
            options.execution.rank_tolerance,
        )
        .map_err(|_| Error::Invalid("meta-regression requires full column rank"))?;
        Ok(Self {
            design: design.clone(),
            outputs,
            options,
            constant,
            rows: Vec::new(),
            y: Vec::new(),
            v: Vec::new(),
        })
    }
    pub fn fit(
        &mut self,
        c: &IndependentCohort,
        mut sink: impl FnMut(MetaRegressionBlock) -> Result<()>,
    ) -> Result<MetaRegressionReceipt> {
        if self.design.rows != c.frame().observations().ids() {
            return Err(Error::Invalid("meta-regression observation binding"));
        }
        let variance = c.variance_assay().ok_or(Error::Invalid(
            "meta-regression requires genuine sampling variance",
        ))?;
        let [n, nf] = c.frame().shape();
        let p = self.design.columns.len();
        let o = self.options.execution;
        let bytes = working_bytes(n, p, &self.outputs, &o, nf)?;
        let mut run = crate::fit::receipt(
            c,
            &o,
            "Weighted meta-regression: conditional on input variances",
            bytes,
        );
        run.model = ModelReceipt::MetaRegression {
            columns: self.design.columns.clone(),
            design: self.design.values.clone(),
            contrasts: self
                .outputs
                .contrasts
                .iter()
                .map(|c| (c.id.clone(), c.weights.to_vec()))
                .collect(),
            rank_tolerance: o.rank_tolerance,
        };
        let mut weighted_evaluations = 0;
        for start in (0..nf).step_by(o.block_features) {
            let end = start + o.block_features.min(nf - start);
            let y = crate::fit::read(c, c.effect_assay(), start, end, bytes)?;
            let v = crate::fit::read(c, variance, start, end, bytes)?;
            let mut features = Vec::with_capacity(end - start);
            for j in 0..end - start {
                self.rows.clear();
                self.y.clear();
                self.v.clear();
                for i in 0..n {
                    if y[(i, j)].is_finite() && v[(i, j)].is_finite() && v[(i, j)] > 0. {
                        self.rows.push(i);
                        self.y.push(y[(i, j)]);
                        self.v.push(v[(i, j)]);
                    }
                }
                let mut result = MetaRegressionFeature {
                    feature: c.frame().features().axis().ids()[start + j].clone(),
                    n: self.rows.len(),
                    rank: 0,
                    status: MetaStatus::InsufficientObservations,
                    estimates: None,
                    diagnostics: None,
                };
                if self.rows.len() != n && o.missingness == Missingness::RejectFeature {
                    result.status = MetaStatus::MissingObservation;
                } else if self.rows.len() >= o.min_observations && self.rows.len() > p {
                    let mut counter = Counter::default();
                    match self.estimate(&mut counter) {
                        Ok((fit, mut diagnostics)) => {
                            result.rank = p;
                            diagnostics.evaluations = counter.evaluations;
                            match requested_outputs(
                                &fit,
                                &self.outputs,
                                self.options.inference,
                                &diagnostics,
                            ) {
                                Ok(estimates) => {
                                    result.estimates = Some(estimates);
                                    result.status = MetaStatus::Estimated;
                                }
                                Err(status) => result.status = status,
                            }
                            result.diagnostics = Some(diagnostics);
                        }
                        Err((status, diagnostics)) => {
                            result.status = status;
                            result.diagnostics = diagnostics;
                        }
                    }
                    run.factorizations += counter.successful;
                    weighted_evaluations += counter.evaluations;
                }
                features.push(result);
            }
            sink(MetaRegressionBlock {
                start,
                contrasts: self
                    .outputs
                    .contrasts
                    .iter()
                    .map(|c| c.id.clone())
                    .collect(),
                features,
            })?;
            run.blocks += 1;
        }
        Ok(MetaRegressionReceipt {
            run,
            options: self.options,
            outputs: self.outputs.clone(),
            weighted_evaluations,
        })
    }
    fn evaluate(
        &self,
        tau: f64,
        count: &mut Counter,
    ) -> std::result::Result<WeightedFit, MetaStatus> {
        count.evaluations += 1;
        let fit = weighted_fit(
            &self.design,
            &self.rows,
            &self.y,
            &self.v,
            tau,
            self.constant,
            self.options.execution.rank_tolerance,
        )?;
        count.successful += 1;
        Ok(fit)
    }
    fn estimate(
        &self,
        count: &mut Counter,
    ) -> std::result::Result<(WeightedFit, MetaDiagnostics), (MetaStatus, Option<MetaDiagnostics>)>
    {
        let mut fit = self.evaluate(0., count).map_err(|s| (s, None))?;
        let df = self.rows.len() - self.design.columns.len();
        let mut diag = MetaDiagnostics {
            q_fixed: fit.q,
            q_final: fit.q,
            tau_squared: 0.,
            residual_df: df,
            covariance_scale: 1.,
            convergence: TauConvergence::Fixed,
            evaluations: 1,
            newton_steps: 0,
            bisection_steps: 0,
            bracket: None,
            qr_diagonal_ratio: fit.diagonal_ratio,
            dl_complement_fallback: false,
        };
        let method = self.options.heterogeneity;
        if method != HeterogeneityEstimator::CommonEffect {
            if fit.q <= df as f64 {
                diag.convergence = TauConvergence::Boundary;
            } else {
                let (c, fallback) = fit.dl_denominator().map_err(|s| (s, Some(diag.clone())))?;
                diag.dl_complement_fallback = fallback;
                // Avoid overflowing 1/c when variance_scale/c is representable.
                let initial = (fit.q - df as f64) * (fit.variance_scale / c);
                if !initial.is_finite() || initial <= 0. {
                    return Err((MetaStatus::NumericalFailure, Some(diag)));
                }
                if method == HeterogeneityEstimator::DerSimonianLaird {
                    fit = self
                        .evaluate(initial, count)
                        .map_err(|s| (s, Some(diag.clone())))?;
                    diag.tau_squared = initial;
                    diag.convergence = TauConvergence::ClosedForm;
                } else {
                    let mut lo = 0.;
                    let mut hi = initial;
                    let mut tau = 0.;
                    loop {
                        diag.evaluations = count.evaluations;
                        diag.q_final = fit.q;
                        diag.tau_squared = tau;
                        diag.bracket = Some([lo, hi]);
                        diag.qr_diagonal_ratio = fit.diagonal_ratio;
                        if tau > 0.
                            && (fit.q - df as f64).abs()
                                <= self.options.paule_mandel.q_tolerance * df as f64
                        {
                            diag.convergence = TauConvergence::Converged;
                            break;
                        }
                        if count.evaluations >= self.options.paule_mandel.max_evaluations {
                            diag.convergence = TauConvergence::EvaluationLimit;
                            return Err((MetaStatus::NonConvergence, Some(diag)));
                        }
                        let next = if tau == 0. {
                            hi
                        } else if fit.q > df as f64 && tau == hi {
                            lo = hi;
                            hi *= 2.;
                            hi
                        } else {
                            if fit.q > df as f64 {
                                lo = tau;
                            } else {
                                hi = tau;
                            }
                            let newton = tau + (fit.q - df as f64) / fit.derivative_magnitude;
                            if newton.is_finite() && newton > lo && newton < hi {
                                diag.newton_steps += 1;
                                newton
                            } else {
                                diag.bisection_steps += 1;
                                lo + (hi - lo) * 0.5
                            }
                        };
                        if !next.is_finite() || next <= 0. || next == tau {
                            diag.convergence = TauConvergence::NumericalFailure;
                            return Err((MetaStatus::NumericalFailure, Some(diag)));
                        }
                        tau = next;
                        fit = self.evaluate(tau, count).map_err(|s| {
                            diag.evaluations = count.evaluations;
                            diag.convergence = if s == MetaStatus::RankDeficient {
                                TauConvergence::RankDeficient
                            } else {
                                TauConvergence::NumericalFailure
                            };
                            (s, Some(diag.clone()))
                        })?;
                    }
                }
            }
        }
        diag.q_final = fit.q;
        diag.evaluations = count.evaluations;
        diag.qr_diagonal_ratio = fit.diagonal_ratio;
        diag.covariance_scale = match self.options.inference {
            MetaInference::Normal => 1.,
            MetaInference::ModifiedKnappHartung => (fit.q / df as f64).max(1.),
        };
        Ok((fit, diag))
    }
}
/// One-shot convenience wrapper; prepared execution binds metadata and reuses row buffers.
pub fn fit_meta_regression(
    c: &IndependentCohort,
    d: &GroupDesign,
    outputs: MetaOutputSpec,
    options: MetaRegressionOptions,
    sink: impl FnMut(MetaRegressionBlock) -> Result<()>,
) -> Result<MetaRegressionReceipt> {
    if d.rows != c.frame().observations().ids() {
        return Err(Error::Invalid("meta-regression observation binding"));
    }
    working_bytes(
        d.rows.len(),
        d.columns.len(),
        &outputs,
        &options.execution,
        c.frame().shape()[1],
    )?;
    PreparedMetaRegression::new(d, outputs, options)?.fit(c, sink)
}
fn working_bytes(
    n: usize,
    p: usize,
    outputs: &MetaOutputSpec,
    o: &FitOptions,
    nf: usize,
) -> Result<usize> {
    let base = crate::fit::budget(n, p, outputs.contrasts.len(), o, nf)? as u128;
    let k = outputs.contrasts.len() as u128;
    let b = o.block_features.min(nf) as u128;
    let joint = if outputs.uncertainty == MetaUncertainty::Joint {
        8 * k * k * b
    } else {
        0
    };
    let bytes =
        base + joint + 8 * (8 * n as u128 * p as u128 + 8 * p as u128 * p as u128 + 16 * n as u128);
    if bytes > o.max_working_bytes as u128 {
        Err(Error::Budget)
    } else {
        Ok(bytes as usize)
    }
}
#[derive(Default)]
struct Counter {
    evaluations: usize,
    successful: usize,
}
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut s = 0_f64;
    let mut correction = 0.;
    for x in values {
        let t = s + x;
        correction += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + correction
}
struct WeightedFit {
    design: DMatrix<f64>,
    qr: ColPivQR<f64, Dyn, Dyn>,
    r: DMatrix<f64>,
    coefficients: Vec<f64>,
    column_scales: Vec<f64>,
    weights: Vec<f64>,
    variance_scale: f64,
    q: f64,
    derivative_magnitude: f64,
    diagonal_ratio: f64,
}
fn weighted_fit(
    d: &GroupDesign,
    rows: &[usize],
    y: &[f64],
    v: &[f64],
    tau: f64,
    constant: Option<usize>,
    tol: f64,
) -> std::result::Result<WeightedFit, MetaStatus> {
    let n = rows.len();
    let p = d.columns.len();
    let variance_scale = v.iter().map(|v| v + tau).fold(f64::INFINITY, f64::min);
    if variance_scale <= 0.
        || !variance_scale.is_finite()
        || v.iter().any(|v| !(v + tau).is_finite())
    {
        return Err(MetaStatus::NumericalFailure);
    }
    let weights = v
        .iter()
        .map(|v| variance_scale / (v + tau))
        .collect::<Vec<_>>();
    if weights.iter().any(|&w| w < f64::MIN_POSITIVE) {
        // Subnormal normalized weights have insufficient relative precision for
        // this QR path's residual/complement calculations. Report unsupported
        // arithmetic instead of silently dropping or flooring an observation.
        return Err(MetaStatus::NumericalFailure);
    }
    let anchor = if constant.is_some() { y[0] } else { 0. };
    let scale = y.iter().map(|y| (y - anchor).abs()).fold(0_f64, f64::max);
    let response_scale = if scale == 0. { 1. } else { scale };
    if !response_scale.is_finite() {
        return Err(MetaStatus::NumericalFailure);
    }
    let mut a = DMatrix::from_fn(n, p, |i, j| weights[i].sqrt() * d.values[(rows[i], j)]);
    let column_scales = (0..p)
        .map(|j| a.column(j).iter().map(|x| x.abs()).fold(0_f64, f64::max))
        .collect::<Vec<_>>();
    if column_scales.iter().any(|s| !s.is_finite() || *s <= 0.) {
        return Err(MetaStatus::RankDeficient);
    }
    for j in 0..p {
        for i in 0..n {
            a[(i, j)] /= column_scales[j];
        }
    }
    let qr = a.clone().col_piv_qr();
    let r = qr.r();
    let largest = r.diagonal().iter().map(|x| x.abs()).fold(0_f64, f64::max);
    let smallest = r
        .diagonal()
        .iter()
        .map(|x| x.abs())
        .fold(f64::INFINITY, f64::min);
    if !largest.is_finite() || smallest <= tol * largest {
        return Err(MetaStatus::RankDeficient);
    }
    let mut rhs = DVector::from_fn(n, |i, _| {
        weights[i].sqrt() * ((y[i] - anchor) / response_scale)
    });
    qr.q_tr_mul(&mut rhs);
    let mut gamma = r
        .solve_upper_triangular(&rhs.rows(0, p).into_owned())
        .ok_or(MetaStatus::RankDeficient)?;
    qr.p().inv_permute_rows(&mut gamma);
    let difficult = weights.iter().any(|&w| w < 1e-12) || smallest / largest < 1e-8;
    if difficult {
        // Two residual-refinement steps correct rounded coefficients using the
        // original weighted design, without changing weights or the estimand.
        for _ in 0..2 {
            let mut residual = DVector::from_fn(n, |i, _| {
                weights[i].sqrt() * ((y[i] - anchor) / response_scale)
                    - sum((0..p).map(|j| a[(i, j)] * gamma[j]))
            });
            qr.q_tr_mul(&mut residual);
            let mut correction = r
                .solve_upper_triangular(&residual.rows(0, p).into_owned())
                .ok_or(MetaStatus::RankDeficient)?;
            qr.p().inv_permute_rows(&mut correction);
            gamma += correction;
        }
    }
    let mut coefficients = (0..p)
        .map(|j| gamma[j] * response_scale / column_scales[j])
        .collect::<Vec<_>>();
    if let Some(j) = constant {
        coefficients[j] += anchor / d.values[(rows[0], j)];
    }
    let multiplier = response_scale / variance_scale.sqrt();
    let mut standardized = Vec::with_capacity(n);
    for i in 0..n {
        let sw = weights[i].sqrt();
        let predicted =
            sum((0..p).map(|j| (sw * d.values[(rows[i], j)] / column_scales[j]) * gamma[j]));
        standardized.push((sw * ((y[i] - anchor) / response_scale) - predicted) * multiplier);
    }
    // Project the explicit residual, rather than the original response, into QR
    // coordinates. This removes fitted-space coefficient-rounding contamination
    // without subtracting squared norms (important with high-leverage rows).
    let mut coordinates = DVector::from_vec(standardized.clone());
    qr.q_tr_mul(&mut coordinates);
    let q = sum(coordinates.iter().skip(p).map(|r| r * r));
    let leakage = sum(coordinates.iter().take(p).map(|r| r * r));
    // Q-norm agreement alone cannot validate the weighted derivative. Reconstruct
    // row residuals at large weight spread, poor conditioning or fitted-space
    // leakage. These select an algorithm; they are not statistical floors or a
    // certificate of arbitrary ill-conditioned forward accuracy.
    let mut derivative_usable = true;
    if difficult || leakage > 1e-16 * q {
        for (i, residual) in standardized.iter_mut().enumerate() {
            let basis = residual_basis(&a, &qr, &r, i)?;
            let energy = sum(basis.coordinates.iter().skip(p).map(|x| x * x));
            derivative_usable &= basis.energy_change <= 1e-10 * energy;
            *residual = sum((p..n).map(|j| basis.coordinates[j] * coordinates[j]));
        }
        let reconstructed = sum(standardized.iter().map(|x| x * x));
        if !reconstructed.is_finite() || (reconstructed - q).abs() > 1e-8 * q.max(f64::MIN_POSITIVE)
        {
            return Err(MetaStatus::NumericalFailure);
        }
    }
    // An uncertain derivative disables Newton; bracketed bisection remains valid.
    let derivative_magnitude = if derivative_usable {
        sum(standardized
            .iter()
            .zip(v)
            .map(|(r, v)| (r / (v + tau).sqrt()).powi(2)))
    } else {
        f64::NAN
    };
    if !q.is_finite() || coefficients.iter().any(|x| !x.is_finite()) {
        return Err(MetaStatus::NumericalFailure);
    }
    Ok(WeightedFit {
        design: a,
        qr,
        r,
        coefficients,
        column_scales,
        weights,
        variance_scale,
        q,
        derivative_magnitude,
        diagonal_ratio: smallest / largest,
    })
}
// Refine residual coordinates of e_i using A as supplied, rather than relying
// on a cancellation-prone tail of Q'e_i for near-unit leverage observations.
struct ResidualBasis {
    coordinates: DVector<f64>,
    energy_change: f64,
}
fn residual_basis(
    a: &DMatrix<f64>,
    qr: &ColPivQR<f64, Dyn, Dyn>,
    r: &DMatrix<f64>,
    i: usize,
) -> std::result::Result<ResidualBasis, MetaStatus> {
    let (n, p) = a.shape();
    let mut rhs = DVector::zeros(n);
    rhs[i] = 1.;
    qr.q_tr_mul(&mut rhs);
    let mut gamma = r
        .solve_upper_triangular(&rhs.rows(0, p).into_owned())
        .ok_or(MetaStatus::RankDeficient)?;
    qr.p().inv_permute_rows(&mut gamma);
    let mut previous_tail = vec![];
    for _ in 0..2 {
        let mut residual = DVector::from_fn(n, |j, _| {
            f64::from(j == i) - sum((0..p).map(|k| a[(j, k)] * gamma[k]))
        });
        qr.q_tr_mul(&mut residual);
        previous_tail = residual.iter().skip(p).copied().collect();
        let mut correction = r
            .solve_upper_triangular(&residual.rows(0, p).into_owned())
            .ok_or(MetaStatus::RankDeficient)?;
        qr.p().inv_permute_rows(&mut correction);
        if correction.iter().any(|x| !x.is_finite()) {
            return Err(MetaStatus::NumericalFailure);
        }
        gamma += correction;
    }
    let mut residual = DVector::from_fn(n, |j, _| {
        f64::from(j == i) - sum((0..p).map(|k| a[(j, k)] * gamma[k]))
    });
    qr.q_tr_mul(&mut residual);
    let energy_change = sum(residual
        .iter()
        .skip(p)
        .zip(previous_tail)
        .map(|(a, b)| (a * a - b * b).abs()));
    Ok(ResidualBasis {
        coordinates: residual,
        energy_change,
    })
}
impl WeightedFit {
    /// Return variance_scale * tr(P). Use Q's complement if 1-h subtraction
    /// would dominate the answer. Applying Q' to basis vectors uses O(n) scratch,
    /// not an n by n projector, and sums nonnegative complement squares.
    fn dl_denominator(&self) -> std::result::Result<(f64, bool), MetaStatus> {
        let q = self.qr.q();
        let sw = sum(self.weights.iter().copied());
        let fast = sum((0..q.nrows())
            .map(|i| self.weights[i] * (1. - sum((0..q.ncols()).map(|j| q[(i, j)].powi(2))))));
        let fallback = !fast.is_finite() || fast <= 1e-8 * sw;
        let c = if fallback {
            let mut terms = Vec::with_capacity(q.nrows());
            let mut changes = Vec::with_capacity(q.nrows());
            for i in 0..q.nrows() {
                let e = residual_basis(&self.design, &self.qr, &self.r, i)?;
                terms.push(
                    self.weights[i] * sum(e.coordinates.iter().skip(q.ncols()).map(|x| x * x)),
                );
                changes.push(self.weights[i] * e.energy_change);
            }
            let value = sum(terms.into_iter());
            if sum(changes.into_iter()) > 1e-10 * value {
                return Err(MetaStatus::NumericalFailure);
            }
            value
        } else {
            fast
        };
        if !c.is_finite() || c <= 0. {
            Err(MetaStatus::NumericalFailure)
        } else {
            Ok((c, fallback))
        }
    }
    fn contrast_geometry(
        &self,
        contrasts: &[GroupContrast],
    ) -> std::result::Result<Vec<Vec<f64>>, MetaStatus> {
        let p = self.r.nrows();
        let mut geometry = Vec::with_capacity(contrasts.len());
        for contrast in contrasts {
            let mut z = DVector::from_fn(p, |j, _| {
                contrast.weights[j] / self.column_scales[j] * self.variance_scale.sqrt()
            });
            self.qr.p().permute_rows(&mut z);
            // R' z = permuted, rescaled contrast; G is the Gram matrix of z.
            for i in 0..p {
                z[i] = (z[i] - sum((0..i).map(|j| self.r[(j, i)] * z[j]))) / self.r[(i, i)];
            }
            if z.iter().any(|v| !v.is_finite()) {
                return Err(MetaStatus::NumericalFailure);
            }
            geometry.push(z.as_slice().to_vec());
        }
        Ok(geometry)
    }
}
fn requested_outputs(
    fit: &WeightedFit,
    spec: &MetaOutputSpec,
    inference: MetaInference,
    diag: &MetaDiagnostics,
) -> std::result::Result<MetaEstimates, MetaStatus> {
    let distribution = match inference {
        MetaInference::Normal => Distribution::Normal,
        MetaInference::ModifiedKnappHartung => Distribution::StudentT {
            df: diag.residual_df,
        },
    };
    let effects = spec
        .contrasts
        .iter()
        .map(|c| sum(c.weights.iter().zip(&fit.coefficients).map(|(c, b)| c * b)))
        .collect::<Vec<_>>();
    if effects.iter().any(|x| !x.is_finite()) {
        return Err(MetaStatus::NumericalFailure);
    }
    let mut out = MetaEstimates {
        effects,
        standard_errors: None,
        joint_covariance: None,
        statistics: None,
        p_values: None,
        confidence_intervals: None,
        distribution,
    };
    if spec.uncertainty == MetaUncertainty::None {
        return Ok(out);
    }
    let geometry = fit.contrast_geometry(&spec.contrasts)?;
    let se = geometry
        .iter()
        .map(|g| (sum(g.iter().map(|x| x * x)) * diag.covariance_scale).sqrt())
        .collect::<Vec<_>>();
    if se.iter().any(|s| !s.is_finite() || *s <= 0.) {
        return Err(MetaStatus::NumericalFailure);
    }
    if spec.uncertainty == MetaUncertainty::Joint {
        let k = geometry.len();
        let cov = Array2::from_shape_fn((k, k), |(i, j)| {
            sum(geometry[i].iter().zip(&geometry[j]).map(|(a, b)| a * b)) * diag.covariance_scale
        });
        if cov.iter().any(|v| !v.is_finite()) {
            return Err(MetaStatus::NumericalFailure);
        }
        out.joint_covariance = Some(cov);
    }
    if spec.p_values || spec.confidence_level.is_some() {
        let normal = Normal::new(0., 1.).unwrap();
        let student = StudentsT::new(0., 1., diag.residual_df as f64).unwrap();
        if spec.p_values {
            let stat = out
                .effects
                .iter()
                .zip(&se)
                .map(|(b, se)| b / se)
                .collect::<Vec<_>>();
            if stat.iter().any(|t| !t.is_finite()) {
                return Err(MetaStatus::NumericalFailure);
            }
            out.p_values = Some(
                stat.iter()
                    .map(|t| {
                        2. * match inference {
                            MetaInference::Normal => normal.sf(t.abs()),
                            MetaInference::ModifiedKnappHartung => student.sf(t.abs()),
                        }
                    })
                    .collect(),
            );
            out.statistics = Some(stat);
        }
        if let Some(level) = spec.confidence_level {
            // Evaluate the lower tail; (1+level)/2 can round to exactly one.
            let probability = (1. - level) * 0.5;
            let critical = match inference {
                MetaInference::Normal => -normal.inverse_cdf(probability),
                MetaInference::ModifiedKnappHartung => -student.inverse_cdf(probability),
            };
            let intervals = out
                .effects
                .iter()
                .zip(&se)
                .map(|(b, se)| [b - critical * se, b + critical * se])
                .collect::<Vec<_>>();
            if intervals.iter().flatten().any(|v| !v.is_finite()) {
                return Err(MetaStatus::NumericalFailure);
            }
            out.confidence_intervals = Some(intervals);
        }
    }
    out.standard_errors = Some(se);
    Ok(out)
}

#[cfg(test)]
mod numerical_tests {
    use super::*;
    #[test]
    fn analytic_derivative_matches_independent_finite_difference() {
        let rows = (0..12)
            .map(|i| fmridataset::ObservationId::new(format!("s{i}")).unwrap())
            .collect();
        let d = GroupDesign::new(
            rows,
            vec!["a".into(), "b".into()],
            Array2::from_shape_fn((12, 2), |(i, j)| if j == 0 { 1. } else { i as f64 / 11. }),
        )
        .unwrap();
        let rows = (0..12).collect::<Vec<_>>();
        let y = (0..12).map(|i| (i as f64 * 0.7).sin()).collect::<Vec<_>>();
        let v = (0..12).map(|i| 0.1 + i as f64 / 12.).collect::<Vec<_>>();
        for t in [0.05, 0.5, 3.] {
            let h = 1e-5 * (1. + t);
            let at = |t| weighted_fit(&d, &rows, &y, &v, t, Some(0), 1e-12).unwrap();
            let numerical = (at(t + h).q - at(t - h).q) / (2. * h);
            assert!((numerical + at(t).derivative_magnitude).abs() < 1e-7 * numerical.abs());
        }
    }
    #[test]
    fn unequal_pairs_derivative_denominator_and_column_scaling() {
        for a in 0..4 {
            for b in (0..4).filter(|&b| b != a) {
                for c in (0..4).filter(|&c| c != a && c != b) {
                    let order = [a, b, c, 6 - a - b - c];
                    for factors in [[1., 1.], [1e-8, -1e8]] {
                        let ids = (0..4)
                            .map(|i| fmridataset::ObservationId::new(format!("s{i}")).unwrap())
                            .collect();
                        let d = GroupDesign::new(
                            ids,
                            vec!["a".into(), "b".into()],
                            Array2::from_shape_fn((4, 2), |(i, j)| {
                                if order[i] / 2 == 1 - j {
                                    factors[j]
                                } else {
                                    0.
                                }
                            }),
                        )
                        .unwrap();
                        let y = order.map(|i| [1., 3., 2., 8.][i]);
                        let v = order.map(|i| [1e-30, 1., 2e-30, 3.][i]);
                        for t in [0., 0.5, 9.5] {
                            let f =
                                weighted_fit(&d, &[0, 1, 2, 3], &y, &v, t, None, 1e-12).unwrap();
                            let q = 4. / (1. + 2. * t) + 36. / (3. + 2. * t);
                            let derivative =
                                8. / (1. + 2. * t).powi(2) + 72. / (3. + 2. * t).powi(2);
                            assert!((f.q - q).abs() < 1e-10 * q);
                            assert!(
                                (f.derivative_magnitude - derivative).abs() < 1e-10 * derivative,
                                "{} != {}",
                                f.derivative_magnitude,
                                derivative
                            );
                            let denominator = f.dl_denominator().unwrap().0 / f.variance_scale;
                            let expected = 2. / (1. + 2. * t) + 2. / (3. + 2. * t);
                            assert!((denominator - expected).abs() < 1e-10 * expected);
                        }
                    }
                }
            }
        }
    }
}
