//! Variance-aware, independent-subject repeated-measures Gaussian models.
//!
//! This module deliberately has a different admission type from
//! [`IndependentCohort`]: repeated measurements of one participant are never
//! silently treated as independent observations.  Covariances are supplied in
//! subject blocks, and all likelihood work is done block by block.
use crate::{nonblank, Admission, Error, Estimand, GroupContrast, GroupDesign, Result};
use fmridataset::{FeatureId, FmriFrame, ObservationId};
use nalgebra::{DMatrix, DVector};
use ndarray::Array2;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepeatedSubject {
    pub observation: ObservationId,
    pub dataset: String,
    pub participant: String,
    pub measurement: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CovarianceProvenance {
    Known { method: String },
    Estimated { method: String, approximate: bool },
}
#[derive(Clone, Debug)]
pub struct RepeatedCovarianceBlock {
    pub observations: Vec<ObservationId>,
    pub values: Array2<f64>,
}
#[derive(Clone, Debug)]
pub struct RepeatedFeatureCovariance {
    pub feature: FeatureId,
    pub blocks: Vec<RepeatedCovarianceBlock>,
}

/// A subject-block covariance for every feature.  The outer vector follows
/// feature order; the inner vector follows `RepeatedCohort::subject_blocks()`.
#[derive(Clone, Debug)]
pub struct RepeatedCohort {
    frame: FmriFrame,
    subjects: Vec<RepeatedSubject>,
    estimand: Estimand,
    admission: Admission,
    effect: String,
    covariance: Vec<RepeatedFeatureCovariance>,
    covariance_provenance: CovarianceProvenance,
    blocks: Vec<Vec<usize>>,
}
impl RepeatedCohort {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame: FmriFrame,
        subjects: Vec<RepeatedSubject>,
        estimand: Estimand,
        admission: Admission,
        effect: String,
        covariance: Vec<RepeatedFeatureCovariance>,
        covariance_provenance: CovarianceProvenance,
    ) -> Result<Self> {
        let [n, nf] = frame.shape();
        if n == 0
            || nf == 0
            || subjects.len() != n
            || !nonblank(&effect)
            || ![
                &estimand.id,
                &estimand.definition,
                &estimand.units,
                &admission.target,
                &admission.evidence,
            ]
            .iter()
            .all(|x| nonblank(x))
        {
            return Err(Error::Invalid("repeated cohort dimensions or effect"));
        }
        frame.assay(&effect)?;
        let mut by_subject: HashMap<(&str, &str), usize> = HashMap::new();
        let mut measurements = HashSet::new();
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        for (i, s) in subjects.iter().enumerate() {
            if s.observation != frame.observations().ids()[i]
                || ![&s.dataset, &s.participant, &s.measurement]
                    .iter()
                    .all(|x| nonblank(x))
                || !measurements.insert((&s.dataset, &s.participant, &s.measurement))
            {
                return Err(Error::Invalid("repeated subject identity/order"));
            }
            let block = *by_subject
                .entry((&s.dataset, &s.participant))
                .or_insert_with(|| {
                    blocks.push(Vec::new());
                    blocks.len() - 1
                });
            blocks[block].push(i);
        }
        if covariance.len() != nf
            || covariance.iter().any(|f| f.blocks.len() != blocks.len())
            || covariance
                .iter()
                .enumerate()
                .any(|(j, f)| f.feature != frame.features().axis().ids()[j])
        {
            return Err(Error::Invalid(
                "repeated covariance feature/subject binding",
            ));
        }
        for feature in &covariance {
            for (b, s) in feature.blocks.iter().zip(&blocks) {
                let ids = s
                    .iter()
                    .map(|i| subjects[*i].observation.clone())
                    .collect::<Vec<_>>();
                if b.observations != ids || b.values.dim() != (s.len(), s.len()) || !spd(&b.values)
                {
                    return Err(Error::Invalid(
                        "sampling covariance must be SPD and observation-bound",
                    ));
                }
            }
        }
        let p = match &covariance_provenance {
            CovarianceProvenance::Known { method } => method,
            CovarianceProvenance::Estimated { method, .. } => method,
        };
        if !nonblank(p) {
            return Err(Error::Invalid("covariance provenance"));
        }
        Ok(Self {
            frame,
            subjects,
            estimand,
            admission,
            effect,
            covariance,
            covariance_provenance,
            blocks,
        })
    }
    pub fn frame(&self) -> &FmriFrame {
        &self.frame
    }
    pub fn subjects(&self) -> &[RepeatedSubject] {
        &self.subjects
    }
    pub fn effect_assay(&self) -> &str {
        &self.effect
    }
    pub fn estimand(&self) -> &Estimand {
        &self.estimand
    }
    pub fn admission(&self) -> &Admission {
        &self.admission
    }
    pub fn covariance_provenance(&self) -> &CovarianceProvenance {
        &self.covariance_provenance
    }
    pub fn subject_blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }
    pub fn covariance(&self, feature: usize) -> Option<&[RepeatedCovarianceBlock]> {
        self.covariance.get(feature).map(|x| x.blocks.as_slice())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepeatedLikelihood {
    Ml,
    Reml,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomEffects {
    Intercept,
    InterceptSlope,
}
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResidualVariance {
    FixedZero,
    Estimate,
}
#[derive(Clone, Debug)]
pub struct RepeatedModel {
    pub random_effects: RandomEffects,
    pub residual_variance: ResidualVariance,
    pub likelihood: RepeatedLikelihood,
    /// Required only for an intercept+slope model; values are bound to cohort rows.
    pub time: Option<Vec<f64>>,
    pub max_evaluations: usize,
    pub tolerance: f64,
    pub rank_tolerance: f64,
    /// Numeric fitting/preparation workspace; excludes resident cohort and sink retention.
    pub max_working_bytes: usize,
}
impl Default for RepeatedModel {
    fn default() -> Self {
        Self {
            random_effects: RandomEffects::Intercept,
            residual_variance: ResidualVariance::FixedZero,
            likelihood: RepeatedLikelihood::Reml,
            time: None,
            max_evaluations: 2048,
            tolerance: 1e-7,
            rank_tolerance: 1e-12,
            max_working_bytes: 64 * 1024 * 1024,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RepeatedStatus {
    Estimated,
    MissingObservation,
    InvalidCovariance,
    UnusableDesign,
    Unidentified,
    EvaluationLimit,
    NonConvergence,
    NumericalFailure,
}
#[derive(Clone, Debug)]
pub struct RepeatedDiagnostics {
    pub evaluations: usize,
    pub projected_gradient: f64,
    pub start_objectives: Vec<f64>,
    pub eta_boundary: bool,
    pub g_rank: usize,
    pub normalizing_constant: f64,
    pub numeric_working_bytes: usize,
    pub slope_center: f64,
    pub slope_scale: f64,
    /// Gradient in sampling-scaled, centered-slope PSD coordinates
    /// (g00, sqrt(2)*g01, g11, optional eta).
    pub covariance_gradient: Vec<f64>,
    pub response_variance_scale: f64,
}
#[derive(Clone, Debug)]
pub struct RepeatedEstimates {
    pub beta: Vec<f64>,
    pub beta_covariance: Array2<f64>,
    pub contrast_effects: Vec<f64>,
    pub contrast_covariance: Array2<f64>,
    pub g: Array2<f64>,
    pub eta: f64,
    pub objective: f64,
    /// Conditional asymptotic reference; covariance estimation uncertainty omitted.
    pub distribution: crate::Distribution,
}
#[derive(Clone, Debug)]
pub struct RepeatedFeature {
    pub feature: FeatureId,
    pub n_observations: usize,
    pub n_subjects: usize,
    pub status: RepeatedStatus,
    pub estimates: Option<RepeatedEstimates>,
    pub diagnostics: Option<RepeatedDiagnostics>,
}
#[derive(Clone, Debug)]
pub struct RepeatedBlock {
    pub start: usize,
    pub contrasts: Vec<String>,
    pub features: Vec<RepeatedFeature>,
}

/// Featurewise additive-covariance ML/REML. Supplied sampling covariances remain
/// fixed; inference conditional on the fitted covariance is asymptotic normal.
/// Dense work is restricted to subject blocks and the tall fixed-effect design.
#[derive(Clone, Debug)]
pub struct PreparedRepeatedModel {
    cohort: RepeatedCohort,
    design: GroupDesign,
    contrasts: Vec<GroupContrast>,
    model: RepeatedModel,
    z: Vec<DMatrix<f64>>,
    slope_center: f64,
    slope_scale: f64,
    working_bytes: usize,
    constant: Option<usize>,
}
impl PreparedRepeatedModel {
    pub fn new(
        cohort: RepeatedCohort,
        design: GroupDesign,
        contrasts: Vec<GroupContrast>,
        model: RepeatedModel,
    ) -> Result<Self> {
        let n = design.rows.len();
        let p = design.columns.len();
        if design.rows != cohort.frame.observations().ids()
            || contrasts.is_empty()
            || contrasts.iter().any(|c| c.columns != design.columns)
            || contrasts
                .iter()
                .map(|c| &c.id)
                .collect::<std::collections::HashSet<_>>()
                .len()
                != contrasts.len()
            || model.max_evaluations == 0
            || !model.tolerance.is_finite()
            || model.tolerance <= 0.
            || model.tolerance >= 0.1
            || !model.rank_tolerance.is_finite()
            || model.rank_tolerance <= 0.
            || model.rank_tolerance >= 0.1
            || n <= p
        {
            return Err(Error::Invalid("repeated model identity, design or options"));
        }
        let q = if model.random_effects == RandomEffects::Intercept {
            1
        } else {
            2
        };
        let time = match model.random_effects {
            RandomEffects::Intercept => {
                if model.time.is_some() {
                    return Err(Error::Invalid(
                        "intercept-only model cannot bind slope values",
                    ));
                }
                vec![0.; n]
            }
            RandomEffects::InterceptSlope => model
                .time
                .clone()
                .ok_or(Error::Invalid("slope model requires row-bound time"))?,
        };
        if time.len() != n || time.iter().any(|v| !v.is_finite()) {
            return Err(Error::Invalid("slope row binding"));
        }
        let slope_center = if q == 2 {
            time[0] + time.iter().map(|v| (v - time[0]) / n as f64).sum::<f64>()
        } else {
            0.
        };
        let slope_scale = if q == 2 {
            time.iter()
                .map(|v| (v - slope_center).abs())
                .fold(0., f64::max)
        } else {
            1.
        };
        if !slope_center.is_finite() || !slope_scale.is_finite() || slope_scale <= 0. {
            return Err(Error::Invalid("constant or overflowing slope"));
        }
        let z = cohort
            .blocks
            .iter()
            .map(|rows| {
                DMatrix::from_fn(rows.len(), q, |r, c| {
                    if c == 0 {
                        1.
                    } else {
                        (time[rows[r]] - slope_center) / slope_scale
                    }
                })
            })
            .collect::<Vec<_>>();
        // Includes block covariance factors/gradients, multiple tall SVD workspaces,
        // covariance-basis admission and current/line-search parameter evaluations.
        let blocks = cohort
            .blocks
            .iter()
            .map(|b| (b.len() as u128).pow(2))
            .sum::<u128>();
        let k = contrasts.len() as u128;
        let bytes = 8u128
            * (40 * blocks
                + 64 * n as u128 * (p as u128 + 4)
                + 32 * (p as u128).pow(2)
                + 8 * k * k
                + 64 * k);
        if bytes > model.max_working_bytes as u128 {
            return Err(Error::Budget);
        }
        let constant = (0..p).find(|&j| {
            design.values[(0, j)] != 0.
                && design
                    .values
                    .column(j)
                    .iter()
                    .all(|v| *v == design.values[(0, j)])
        });
        let this = Self {
            cohort,
            design,
            contrasts,
            model,
            z,
            slope_center,
            slope_scale,
            working_bytes: bytes as usize,
            constant,
        };
        this.admit()?;
        Ok(this)
    }
    pub fn cohort(&self) -> &RepeatedCohort {
        &self.cohort
    }
    pub fn design(&self) -> &GroupDesign {
        &self.design
    }
    pub fn model(&self) -> &RepeatedModel {
        &self.model
    }
    pub fn numeric_working_bytes(&self) -> usize {
        self.working_bytes
    }
    fn admit(&self) -> Result<()> {
        let x = matrix(&self.design.values);
        let geometry = Geometry::new(&x, self.model.rank_tolerance)
            .ok_or(Error::Invalid("repeated fixed design rank"))?;
        let u = &geometry.u;
        let q = self.z[0].ncols();
        let mut bases: Vec<Vec<DMatrix<f64>>> = Vec::new();
        for a in 0..q {
            for b in a..q {
                bases.push(
                    self.z
                        .iter()
                        .map(|z| {
                            DMatrix::from_fn(z.nrows(), z.nrows(), |i, j| {
                                if a == b {
                                    z[(i, a)] * z[(j, b)]
                                } else {
                                    z[(i, a)] * z[(j, b)] + z[(i, b)] * z[(j, a)]
                                }
                            })
                        })
                        .collect(),
                );
            }
        }
        if self.model.residual_variance == ResidualVariance::Estimate {
            bases.push(
                self.z
                    .iter()
                    .map(|z| DMatrix::identity(z.nrows(), z.nrows()))
                    .collect(),
            );
        }
        let k = bases.len();
        let mut du = Vec::new();
        let mut utu = Vec::new();
        for basis in &bases {
            let mut a = DMatrix::zeros(x.nrows(), x.ncols());
            for (d, rows) in basis.iter().zip(&self.cohort.blocks) {
                let ub = DMatrix::from_fn(rows.len(), u.ncols(), |i, j| u[(rows[i], j)]);
                let product = d * ub;
                for (i, &row) in rows.iter().enumerate() {
                    for j in 0..u.ncols() {
                        a[(row, j)] = product[(i, j)];
                    }
                }
            }
            utu.push(u.transpose() * &a);
            du.push(a);
        }
        for restricted in [false, true] {
            if restricted && self.model.likelihood != RepeatedLikelihood::Reml {
                continue;
            }
            let mut gram = DMatrix::zeros(k, k);
            let mut raw_norms = vec![0.; k];
            for a in 0..k {
                for b in 0..=a {
                    let raw = bases[a]
                        .iter()
                        .zip(&bases[b])
                        .map(|(a, b)| a.dot(b))
                        .sum::<f64>();
                    if a == b {
                        raw_norms[a] = raw;
                    }
                    let value = if restricted {
                        raw - 2. * du[a].dot(&du[b]) + utu[a].dot(&utu[b])
                    } else {
                        raw
                    };
                    gram[(a, b)] = value;
                    gram[(b, a)] = value;
                }
            }
            let diag = gram.diagonal();
            if diag
                .iter()
                .zip(raw_norms)
                .any(|(&v, r)| !v.is_finite() || v <= 1e-10 * r.max(f64::MIN_POSITIVE))
            {
                return Err(Error::Invalid("unidentified repeated covariance direction"));
            }
            for a in 0..k {
                for b in 0..k {
                    gram[(a, b)] /= diag[a].sqrt() * diag[b].sqrt();
                }
            }
            let eigen = gram.symmetric_eigen().eigenvalues;
            if eigen.min() <= 1e-10 * eigen.max() {
                return Err(Error::Invalid("unidentified repeated covariance basis"));
            }
        }
        Ok(())
    }
    pub fn fit_feature(&self, feature: usize) -> Result<RepeatedFeature> {
        self.fit_feature_impl(feature, None)
    }
    /// Fixed covariance evaluation has no optimizer uncertainty or convergence
    /// claim. G is in original slope coordinates and response-squared units.
    pub fn evaluate_fixed(
        &self,
        feature: usize,
        g: Array2<f64>,
        eta: f64,
    ) -> Result<RepeatedFeature> {
        self.fit_feature_impl(feature, Some((g, eta)))
    }
    pub fn fit_blocks(
        &self,
        block_features: usize,
        mut sink: impl FnMut(RepeatedBlock) -> Result<()>,
    ) -> Result<()> {
        if block_features == 0 {
            return Err(Error::Invalid("repeated block size"));
        }
        let p = self.design.columns.len() as u128;
        let k = self.contrasts.len() as u128;
        let extra = 8u128
            * block_features.min(self.cohort.frame.shape()[1]) as u128
            * (p * p + k * k + p + k + 32);
        if self.working_bytes as u128 + extra > self.model.max_working_bytes as u128 {
            return Err(Error::Budget);
        }
        for start in (0..self.cohort.frame.shape()[1]).step_by(block_features) {
            let end = start + block_features.min(self.cohort.frame.shape()[1] - start);
            let features = (start..end)
                .map(|j| self.fit_feature(j))
                .collect::<Result<Vec<_>>>()?;
            sink(RepeatedBlock {
                start,
                contrasts: self.contrasts.iter().map(|c| c.id.clone()).collect(),
                features,
            })?;
        }
        Ok(())
    }
    fn fit_feature_impl(
        &self,
        j: usize,
        fixed: Option<(Array2<f64>, f64)>,
    ) -> Result<RepeatedFeature> {
        let fid = self
            .cohort
            .frame
            .features()
            .axis()
            .ids()
            .get(j)
            .cloned()
            .ok_or(Error::Invalid("repeated feature index"))?;
        let blank = |status| RepeatedFeature {
            feature: fid.clone(),
            n_observations: self.design.rows.len(),
            n_subjects: self.cohort.blocks.len(),
            status,
            estimates: None,
            diagnostics: None,
        };
        let y = read_effect(&self.cohort, j, self.working_bytes)?;
        if y.iter().any(|v| !v.is_finite()) {
            return Ok(blank(RepeatedStatus::MissingObservation));
        }
        let mut diagonal = self.cohort.covariance[j]
            .blocks
            .iter()
            .flat_map(|b| (0..b.values.nrows()).map(|i| b.values[(i, i)]))
            .collect::<Vec<_>>();
        diagonal.sort_by(f64::total_cmp);
        let sampling_scale = diagonal[diagonal.len() / 2];
        // A fixed coordinate normalization, not a fitted/profiled variance. Include
        // response residual scale so high heterogeneity does not flatten scores.
        let anchor = if self.constant.is_some() { y[0] } else { 0. };
        let centered = y.map(|v| v - anchor);
        let magnitude = centered.iter().map(|v| v.abs()).fold(0., f64::max);
        if !magnitude.is_finite() {
            return Ok(blank(RepeatedStatus::NumericalFailure));
        }
        let residual_scale = if magnitude == 0. {
            0.
        } else {
            let yn = centered.map(|v| v / magnitude);
            let x = matrix(&self.design.values);
            let geometry = Geometry::new(&x, self.model.rank_tolerance)
                .ok_or(Error::Invalid("prepared design geometry"))?;
            let r = &yn - &x * (&geometry.inverse * &yn);
            let rms = (r.norm_squared() / y.len() as f64).sqrt() * magnitude;
            rms * rms
        };
        let scale = sampling_scale.max(residual_scale);
        if !scale.is_finite() || scale <= 0. {
            return Ok(blank(RepeatedStatus::InvalidCovariance));
        }
        let ys = centered.map(|v| v / scale.sqrt());
        if ys.iter().any(|v| !v.is_finite()) {
            return Ok(blank(RepeatedStatus::NumericalFailure));
        }
        let q = self.z[0].ncols();
        let has_eta = self.model.residual_variance == ResidualVariance::Estimate;
        let mut evaluations = 0usize;
        let (best, status, pg, starts) = if let Some((g, eta)) = fixed {
            if !eta.is_finite()
                || eta < 0.
                || (!has_eta && eta != 0.)
                || g.dim() != (q, q)
                || !symmetric_finite(&g)
            {
                return Ok(blank(RepeatedStatus::InvalidCovariance));
            }
            let g = matrix(&g);
            let magnitude = g.iter().map(|v| v.abs()).fold(0., f64::max);
            let normalized = if magnitude == 0. {
                g.clone()
            } else {
                DMatrix::from_fn(q, q, |i, k| g[(i, k)] / magnitude)
            };
            let eig = normalized.symmetric_eigen();
            if eig.eigenvalues.iter().any(|v| !v.is_finite())
                || eig.eigenvalues.min() < -64. * f64::EPSILON
            {
                return Ok(blank(RepeatedStatus::InvalidCovariance));
            }
            // Only rounding-scale negative eigenvalues of an admitted PSD input
            // are set to zero; supplied sampling covariance is never repaired.
            let g = &eig.eigenvectors
                * DMatrix::from_diagonal(&eig.eigenvalues.map(|v| v.max(0.)))
                * eig.eigenvectors.transpose()
                * magnitude;
            let transform = if q == 1 {
                DMatrix::identity(1, 1)
            } else {
                DMatrix::from_row_slice(2, 2, &[1., self.slope_center, 0., self.slope_scale])
            };
            let norm_g = &transform * g * transform.transpose() / scale;
            let params = pack(&norm_g, eta / scale, has_eta);
            evaluations = 1;
            match self.eval(j, &ys, scale, &params) {
                Some(v) => (Some(v), RepeatedStatus::Estimated, 0., Vec::new()),
                None => (
                    None,
                    RepeatedStatus::NumericalFailure,
                    f64::INFINITY,
                    Vec::new(),
                ),
            }
        } else {
            self.optimize(j, &ys, scale, &mut evaluations)
        };
        let Some(best) = best else {
            return Ok(blank(status));
        };
        let factor = if q == 1 {
            DMatrix::identity(1, 1)
        } else {
            DMatrix::from_row_slice(
                2,
                2,
                &[
                    1.,
                    -self.slope_center / self.slope_scale,
                    0.,
                    1. / self.slope_scale,
                ],
            )
        };
        let g = &factor * &best.g * factor.transpose() * scale;
        let mut beta = &best.beta * scale.sqrt();
        if let Some(c) = self.constant {
            beta[c] += anchor / self.design.values[(0, c)];
        }
        let cov = &best.cov_beta * scale;
        let ct = DMatrix::from_fn(self.contrasts.len(), beta.len(), |i, j| {
            self.contrasts[i].weights[j]
        });
        let ce = &ct * &beta;
        let cc = &ct * &cov * ct.transpose();
        let dimension = if self.model.likelihood == RepeatedLikelihood::Ml {
            y.len()
        } else {
            y.len() - self.design.columns.len()
        };
        let objective = best.objective + 0.5 * dimension as f64 * scale.ln();
        let covariance_rank = best.g.clone().symmetric_eigen().eigenvalues;
        let diagnostics = RepeatedDiagnostics {
            evaluations,
            projected_gradient: pg,
            start_objectives: starts,
            eta_boundary: best.eta == 0.,
            g_rank: covariance_rank
                .iter()
                .filter(|v| **v > 1e-8 * covariance_rank.max().max(f64::MIN_POSITIVE))
                .count(),
            normalizing_constant: 0.5 * dimension as f64 * (2. * std::f64::consts::PI).ln(),
            numeric_working_bytes: self.working_bytes,
            slope_center: self.slope_center,
            slope_scale: self.slope_scale,
            covariance_gradient: best.gradient.clone(),
            response_variance_scale: scale,
        };
        let eta_output = best.eta * scale;
        let valid = beta
            .iter()
            .chain(cov.iter())
            .chain(ce.iter())
            .chain(cc.iter())
            .chain(g.iter())
            .all(|v| v.is_finite())
            && objective.is_finite()
            && eta_output.is_finite();
        let status = if valid {
            status
        } else {
            RepeatedStatus::NumericalFailure
        };
        let estimates = if status == RepeatedStatus::Estimated {
            Some(RepeatedEstimates {
                beta: beta.iter().copied().collect(),
                beta_covariance: to_array(&cov),
                contrast_effects: ce.iter().copied().collect(),
                contrast_covariance: to_array(&cc),
                g: to_array(&g),
                eta: eta_output,
                objective,
                distribution: crate::Distribution::Normal,
            })
        } else {
            None
        };
        Ok(RepeatedFeature {
            feature: fid,
            n_observations: y.len(),
            n_subjects: self.cohort.blocks.len(),
            status,
            estimates,
            diagnostics: Some(diagnostics),
        })
    }
    fn eval(&self, j: usize, y: &DVector<f64>, scale: f64, par: &[f64]) -> Option<Evaluation> {
        let q = self.z[0].ncols();
        let (g, eta) = unpack(
            par,
            q,
            self.model.residual_variance == ResidualVariance::Estimate,
        );
        let n = self.design.rows.len();
        let p = self.design.columns.len();
        let mut wx = DMatrix::zeros(n, p);
        let mut wy = DVector::zeros(n);
        let mut blocks = Vec::new();
        let mut logdet = 0.;
        for (index, rows) in self.cohort.blocks.iter().enumerate() {
            let sampling = matrix(&self.cohort.covariance[j].blocks[index].values) / scale;
            let mut v = sampling + &self.z[index] * &g * self.z[index].transpose();
            for i in 0..v.nrows() {
                v[(i, i)] += eta;
            }
            let ch = v.cholesky()?;
            let l = ch.l();
            logdet += 2. * l.diagonal().iter().map(|x| x.ln()).sum::<f64>();
            let xb = DMatrix::from_fn(rows.len(), p, |i, k| self.design.values[(rows[i], k)]);
            let yb = DVector::from_iterator(rows.len(), rows.iter().map(|i| y[*i]));
            let ax = l.solve_lower_triangular(&xb)?;
            let ay = l.solve_lower_triangular(&yb)?;
            for (i, &r) in rows.iter().enumerate() {
                wy[r] = ay[i];
                for k in 0..p {
                    wx[(r, k)] = ax[(i, k)];
                }
            }
            let az = l.solve_lower_triangular(&self.z[index])?;
            let inverse = if self.model.residual_variance == ResidualVariance::Estimate {
                Some(ch.inverse())
            } else {
                None
            };
            blocks.push((l, ax, ay, az, inverse));
        }
        let geometry = Geometry::new(&wx, self.model.rank_tolerance)?;
        let mut beta = &geometry.inverse * &wy;
        beta += &geometry.inverse * (&wy - &wx * &beta);
        let residual = &wy - &wx * &beta;
        let cov = &geometry.inverse * geometry.inverse.transpose();
        let objective = 0.5
            * (logdet
                + residual.norm_squared()
                + if self.model.likelihood == RepeatedLikelihood::Reml {
                    geometry.log_information
                } else {
                    0.
                });
        let mut ge = 0.;
        let mut gg = DMatrix::zeros(q, q);
        for (l, ax, ay, az, inverse) in &blocks {
            let residual = ay - ax * &beta;
            let score = az.transpose() * &residual;
            let mut information = az.transpose() * az;
            if self.model.likelihood == RepeatedLikelihood::Reml {
                let cross = az.transpose() * ax;
                information -= &cross * &cov * cross.transpose();
            }
            gg += 0.5 * (information - &score * score.transpose());
            if let Some(vi) = inverse {
                let w = l.transpose().solve_upper_triangular(&residual)?;
                let mut trace = vi.trace();
                if self.model.likelihood == RepeatedLikelihood::Reml {
                    let a = l.transpose().solve_upper_triangular(ax)?;
                    trace -= (&a * &cov * a.transpose()).trace();
                }
                ge += 0.5 * (trace - w.norm_squared());
            }
        }
        let gradient = pack(
            &gg,
            ge,
            self.model.residual_variance == ResidualVariance::Estimate,
        );
        if !objective.is_finite()
            || gradient
                .iter()
                .chain(beta.iter())
                .chain(cov.iter())
                .any(|v| !v.is_finite())
        {
            return None;
        }
        Some(Evaluation {
            objective,
            beta,
            cov_beta: cov,
            g,
            eta,
            gradient,
            parameters: par.to_vec(),
        })
    }
    fn optimize(
        &self,
        j: usize,
        y: &DVector<f64>,
        scale: f64,
        count: &mut usize,
    ) -> (Option<Evaluation>, RepeatedStatus, f64, Vec<f64>) {
        let q = self.z[0].ncols();
        let eta = self.model.residual_variance == ResidualVariance::Estimate;
        let dims = if q == 1 { 1 } else { 3 } + usize::from(eta);
        let mut seeds = vec![vec![0.; dims]];
        for magnitude in [1., 10.] {
            let mut s = vec![0.; dims];
            s[0] = magnitude;
            if q == 2 {
                s[2] = magnitude;
            }
            if eta {
                *s.last_mut().unwrap() = magnitude;
            }
            seeds.push(s);
        }
        if q == 2 {
            for sign in [-1., 1.] {
                let mut s = vec![0.; dims];
                s[0] = 1.;
                s[1] = sign * std::f64::consts::SQRT_2;
                s[2] = 1.;
                seeds.push(s);
            }
        }
        let seed_count = seeds.len();
        let mut candidates = Vec::new();
        let mut objectives = Vec::new();
        for (start_index, seed) in seeds.into_iter().enumerate() {
            if *count >= self.model.max_evaluations {
                break;
            }
            let per_start =
                ((self.model.max_evaluations - *count) / (seed_count - start_index)).max(1);
            let end = (*count + per_start).min(self.model.max_evaluations);
            *count += 1;
            let Some(mut current) = self.eval(j, y, scale, &seed) else {
                continue;
            };
            let mut step = 1.;
            let mut converged = false;
            let mut last_change = 0.;
            loop {
                let pg = projected_norm(&current.parameters, &current.gradient, q, eta);
                let threshold = self.model.tolerance;
                if pg <= threshold && last_change <= self.model.tolerance.sqrt() * y.len() as f64 {
                    // Feasible coordinate probes catch a false flat stopping point.
                    let mut improving = None;
                    for k in 0..dims {
                        for sign in [-1., 1.] {
                            if *count >= end {
                                break;
                            }
                            let mut x = current.parameters.clone();
                            x[k] += sign * self.model.tolerance.sqrt() * (1. + x[k].abs());
                            project(&mut x, q, eta);
                            *count += 1;
                            if let Some(v) = self.eval(j, y, scale, &x) {
                                if v.objective
                                    < current.objective
                                        - 0.1 * self.model.tolerance * y.len() as f64
                                {
                                    improving = Some(v);
                                }
                            }
                        }
                    }
                    if let Some(v) = improving {
                        current = v;
                        last_change = f64::INFINITY;
                        continue;
                    }
                    converged = *count < end;
                    break;
                }
                if *count >= end {
                    break;
                }
                let mut accepted = None;
                for _ in 0..60 {
                    if *count >= end {
                        break;
                    }
                    let mut x = current
                        .parameters
                        .iter()
                        .zip(&current.gradient)
                        .map(|(a, g)| a - step * g)
                        .collect::<Vec<_>>();
                    project(&mut x, q, eta);
                    let displacement = x
                        .iter()
                        .zip(&current.parameters)
                        .map(|(a, b)| a - b)
                        .collect::<Vec<_>>();
                    let descent = dot_slice(&current.gradient, &displacement);
                    if !descent.is_finite() || descent >= 0. {
                        step *= 0.5;
                        continue;
                    }
                    *count += 1;
                    if let Some(v) = self.eval(j, y, scale, &x) {
                        if v.objective <= current.objective + 1e-4 * descent {
                            accepted = Some((v, displacement));
                            break;
                        }
                    }
                    step *= 0.5;
                }
                let Some((next, dx)) = accepted else {
                    break;
                };
                let dg = next
                    .gradient
                    .iter()
                    .zip(&current.gradient)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>();
                let curvature = dot_slice(&dx, &dg);
                step = if curvature > 0. {
                    (dot_slice(&dx, &dx) / curvature).clamp(1e-8, 1e8)
                } else {
                    1.
                };
                last_change = (next.objective - current.objective).abs();
                current = next;
            }
            let pg = projected_norm(&current.parameters, &current.gradient, q, eta);
            objectives.push(current.objective);
            candidates.push((current, converged, pg));
        }
        if candidates.is_empty() {
            return (
                None,
                RepeatedStatus::NumericalFailure,
                f64::INFINITY,
                objectives,
            );
        }
        candidates.sort_by(|a, b| a.0.objective.total_cmp(&b.0.objective));
        let minimum = candidates[0].0.objective;
        let selected = candidates
            .iter()
            .position(|(v, c, _)| {
                *c && v.objective <= minimum + self.model.tolerance * y.len() as f64
            })
            .unwrap_or(0);
        let (best, converged, pg) = candidates.swap_remove(selected);
        let status = if converged {
            RepeatedStatus::Estimated
        } else if *count >= self.model.max_evaluations {
            RepeatedStatus::EvaluationLimit
        } else {
            RepeatedStatus::NonConvergence
        };
        (Some(best), status, pg, objectives)
    }
}
#[derive(Clone)]
struct Evaluation {
    objective: f64,
    beta: DVector<f64>,
    cov_beta: DMatrix<f64>,
    g: DMatrix<f64>,
    eta: f64,
    gradient: Vec<f64>,
    parameters: Vec<f64>,
}
struct Geometry {
    inverse: DMatrix<f64>,
    u: DMatrix<f64>,
    log_information: f64,
}
impl Geometry {
    fn new(x: &DMatrix<f64>, tolerance: f64) -> Option<Self> {
        let mut scaled = x.clone();
        let mut scales = Vec::new();
        for j in 0..x.ncols() {
            let scale = x.column(j).iter().map(|v| v.abs()).fold(0., f64::max);
            if !scale.is_finite() || scale == 0. {
                return None;
            }
            scales.push(scale);
            scaled.column_mut(j).scale_mut(1. / scale);
        }
        let svd = scaled.svd(true, true);
        let maximum = svd.singular_values.max();
        if svd
            .singular_values
            .iter()
            .any(|v| !v.is_finite() || *v <= tolerance * maximum)
        {
            return None;
        }
        let u = svd.u?;
        let v = svd.v_t?.transpose();
        let log_information = 2.
            * (svd.singular_values.iter().map(|s| s.ln()).sum::<f64>()
                + scales.iter().map(|s| s.ln()).sum::<f64>());
        let mut inverse =
            v * DMatrix::from_diagonal(&svd.singular_values.map(|s| 1. / s)) * u.transpose();
        for (j, s) in scales.iter().enumerate() {
            inverse.row_mut(j).scale_mut(1. / s);
        }
        Some(Self {
            inverse,
            u,
            log_information,
        })
    }
}
fn matrix(a: &Array2<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(a.nrows(), a.ncols(), |i, j| a[(i, j)])
}
fn to_array(a: &DMatrix<f64>) -> Array2<f64> {
    Array2::from_shape_fn((a.nrows(), a.ncols()), |(i, j)| a[(i, j)])
}
fn symmetric_finite(a: &Array2<f64>) -> bool {
    if a.nrows() == 0 || a.nrows() != a.ncols() || a.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let scale = a.iter().map(|v| v.abs()).fold(0., f64::max);
    (0..a.nrows())
        .all(|i| (0..i).all(|j| (a[(i, j)] - a[(j, i)]).abs() <= 64. * f64::EPSILON * scale))
}
fn spd(a: &Array2<f64>) -> bool {
    symmetric_finite(a) && matrix(a).cholesky().is_some()
}
fn read_effect(c: &RepeatedCohort, j: usize, bytes: usize) -> Result<DVector<f64>> {
    use fmridataset::{ExecutionContext, MemoryBudget, ReadSelection, Selection, Values};
    let src = c.frame.assay(&c.effect)?.source();
    let ctx = ExecutionContext::new(MemoryBudget::new(bytes));
    let b = ctx.read(
        &mut ctx.open(src),
        &ReadSelection::new(src.shape(), Selection::All, Selection::Range(j..j + 1))?,
    )?;
    let values = match b.into_values() {
        Values::F64(v) => v,
        Values::F32(v) => v.into_iter().map(f64::from).collect(),
    };
    Ok(DVector::from_vec(values))
}
fn norm(v: &[f64]) -> f64 {
    v.iter().map(|v| v * v).sum::<f64>().sqrt()
}
fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn unpack(x: &[f64], q: usize, eta: bool) -> (DMatrix<f64>, f64) {
    let g = if q == 1 {
        DMatrix::from_element(1, 1, x[0])
    } else {
        DMatrix::from_row_slice(
            2,
            2,
            &[
                x[0],
                x[1] / std::f64::consts::SQRT_2,
                x[1] / std::f64::consts::SQRT_2,
                x[2],
            ],
        )
    };
    (g, if eta { *x.last().unwrap() } else { 0. })
}
fn pack(g: &DMatrix<f64>, eta: f64, has_eta: bool) -> Vec<f64> {
    let mut v = if g.nrows() == 1 {
        vec![g[(0, 0)]]
    } else {
        vec![g[(0, 0)], std::f64::consts::SQRT_2 * g[(0, 1)], g[(1, 1)]]
    };
    if has_eta {
        v.push(eta);
    }
    v
}
fn project(x: &mut [f64], q: usize, eta: bool) {
    if q == 1 {
        x[0] = x[0].max(0.);
    } else {
        let (g, _) = unpack(x, q, eta);
        let eig = g.symmetric_eigen();
        let p = &eig.eigenvectors
            * DMatrix::from_diagonal(&eig.eigenvalues.map(|v| v.max(0.)))
            * eig.eigenvectors.transpose();
        x[0] = p[(0, 0)];
        x[1] = std::f64::consts::SQRT_2 * p[(0, 1)];
        x[2] = p[(1, 1)];
    }
    if eta {
        let n = x.len();
        x[n - 1] = x[n - 1].max(0.);
    }
}
fn projected_norm(x: &[f64], g: &[f64], q: usize, eta: bool) -> f64 {
    let scale = norm(x).max(1.);
    let mut z = x
        .iter()
        .zip(g)
        .map(|(a, b)| a / scale - scale * b)
        .collect::<Vec<_>>();
    if z.iter().any(|v| !v.is_finite()) {
        return f64::INFINITY;
    }
    project(&mut z, q, eta);
    norm(
        &z.iter()
            .zip(x)
            .map(|(a, b)| a - b / scale)
            .collect::<Vec<_>>(),
    )
}
