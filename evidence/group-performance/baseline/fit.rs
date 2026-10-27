use crate::{Error, IndependentCohort, Result};
use fmridataset::{
    ExecutionContext, FeatureId, MemoryBudget, ObservationId, ReadSelection, Selection, Values,
};
use fmrireg::{FitError, LinearModel};
use ndarray::{Array1, Array2};
use statrs::distribution::{ContinuousCDF, Normal};
use std::collections::{BTreeMap, HashSet, VecDeque};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Missingness {
    RejectFeature,
    OmitObservation,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetaMethod {
    CommonEffect,
    DerSimonianLaird,
}
#[derive(Clone, Debug, PartialEq)]
pub enum Distribution {
    Normal,
    StudentT { df: usize },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Estimated,
    MissingObservation,
    InsufficientObservations,
    /// Retained design has no usable rank/residual df or cannot be factorized.
    UnusableDesign,
    NonEstimable,
    DegenerateVariance,
    NumericalFailure,
}
#[derive(Clone, Debug)]
pub struct Estimate {
    pub effect: f64,
    pub standard_error: f64,
    pub statistic: f64,
    pub p_value: f64,
    pub distribution: Distribution,
}
#[derive(Clone, Debug)]
pub struct Heterogeneity {
    pub q: f64,
    pub i_squared: f64,
    pub tau_squared: f64,
}
#[derive(Clone, Debug)]
pub struct FeatureResult {
    pub feature: FeatureId,
    pub n: usize,
    pub rank: usize,
    pub status: Vec<Status>,
    pub estimates: Vec<Option<Estimate>>,
    pub heterogeneity: Option<Heterogeneity>,
}
#[derive(Clone, Debug)]
pub struct ResultBlock {
    pub start: usize,
    pub contrasts: Vec<String>,
    pub features: Vec<FeatureResult>,
}
#[derive(Clone, Debug)]
pub struct RunReceipt {
    pub method: String,
    pub input: crate::Estimand,
    pub admission: crate::Admission,
    pub subjects: Vec<crate::Subject>,
    pub first_level: Vec<crate::FitReceipt>,
    pub variance_origins: std::sync::Arc<[crate::VarianceOrigin]>,
    pub missingness: Missingness,
    pub model: ModelReceipt,
    pub min_observations: usize,
    pub feature_count: usize,
    pub blocks: usize,
    pub factorizations: usize,
    /// Conservative numeric working-set estimate, not process RSS. Excludes
    /// resident input, returned/sink-retained outputs, allocator and LAPACK scratch.
    pub numeric_working_bytes: usize,
}
/// Numerical model bound to the ordered subjects in the run receipt.
#[derive(Clone, Debug)]
pub enum ModelReceipt {
    Ols {
        columns: Vec<String>,
        design: Array2<f64>,
        contrasts: Vec<(String, Vec<f64>)>,
        rank_tolerance: f64,
    },
    Meta {
        method: MetaMethod,
    },
}
#[derive(Clone, Copy, Debug)]
pub struct FitOptions {
    pub block_features: usize,
    pub max_working_bytes: usize,
    pub max_cached_patterns: usize,
    pub missingness: Missingness,
    pub min_observations: usize,
    pub rank_tolerance: f64,
}
impl Default for FitOptions {
    fn default() -> Self {
        Self {
            block_features: 256,
            max_working_bytes: 64 * 1024 * 1024,
            max_cached_patterns: 16,
            missingness: Missingness::RejectFeature,
            min_observations: 2,
            rank_tolerance: 1e-12,
        }
    }
}
/// Exact row-bound design. Formula expansion belongs outside numerical loops.
#[derive(Clone, Debug)]
pub struct GroupDesign {
    rows: Vec<ObservationId>,
    columns: Vec<String>,
    values: Array2<f64>,
}
impl GroupDesign {
    pub fn new(
        rows: Vec<ObservationId>,
        columns: Vec<String>,
        values: Array2<f64>,
    ) -> Result<Self> {
        if values.dim() != (rows.len(), columns.len())
            || rows.is_empty()
            || columns.is_empty()
            || rows.iter().collect::<HashSet<_>>().len() != rows.len()
            || columns.iter().any(|s| !crate::nonblank(s))
            || columns.iter().collect::<HashSet<_>>().len() != columns.len()
            || values.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Invalid("design axes or values"));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }
    pub fn intercept(cohort: &IndependentCohort) -> Self {
        Self {
            rows: cohort.frame().observations().ids().to_vec(),
            columns: vec!["intercept".into()],
            values: Array2::ones((cohort.subjects().len(), 1)),
        }
    }
    pub fn columns(&self) -> &[String] {
        &self.columns
    }
}
#[derive(Clone, Debug)]
pub struct GroupContrast {
    id: String,
    columns: Vec<String>,
    weights: Array1<f64>,
}
impl GroupContrast {
    /// Resolve named weights once; unknown/duplicate terms are errors.
    pub fn named(
        id: impl Into<String>,
        design: &GroupDesign,
        terms: &[(&str, f64)],
    ) -> Result<Self> {
        let id = id.into();
        let mut weights = Array1::zeros(design.columns.len());
        let mut seen = HashSet::new();
        if !crate::nonblank(&id) {
            return Err(Error::Invalid("empty contrast ID"));
        }
        for &(name, value) in terms {
            let i = design
                .columns
                .iter()
                .position(|s| s == name)
                .ok_or(Error::Invalid("unknown contrast term"))?;
            if !seen.insert(i) || !value.is_finite() {
                return Err(Error::Invalid("duplicate/nonfinite contrast term"));
            }
            weights[i] = value;
        }
        if weights.iter().all(|v| *v == 0.) {
            return Err(Error::Invalid("zero contrast"));
        }
        Ok(Self {
            id,
            columns: design.columns.clone(),
            weights,
        })
    }
}
fn budget(n: usize, p: usize, k: usize, o: &FitOptions, nf: usize) -> Result<usize> {
    if o.block_features == 0
        || o.max_cached_patterns == 0
        || o.min_observations < 2
        || !o.rank_tolerance.is_finite()
        || o.rank_tolerance <= 0.
        || o.rank_tolerance >= 1.
    {
        return Err(Error::Invalid("fit options"));
    }
    let b = o.block_features.min(nf) as u128;
    let n = n as u128;
    let p = p as u128;
    let k = k as u128;
    let bytes = 8
        * (8 * n * b
            + 8 * p * b
            + 16 * k * b
            + (o.max_cached_patterns as u128 + 2) * (2 * n * p + 2 * p * p));
    if bytes > o.max_working_bytes as u128 {
        return Err(Error::Budget);
    }
    Ok(bytes as usize)
}
fn read(
    cohort: &IndependentCohort,
    name: &str,
    start: usize,
    end: usize,
    bytes: usize,
) -> Result<Array2<f64>> {
    let src = cohort.frame().assay(name)?.source();
    let ctx = ExecutionContext::new(MemoryBudget::new(bytes));
    let sel = ReadSelection::new(src.shape(), Selection::All, Selection::Range(start..end))?;
    let block = ctx.read(&mut ctx.open(src), &sel)?;
    let shape = block.shape();
    let values = match block.into_values() {
        Values::F64(v) => v,
        Values::F32(v) => v.into_iter().map(f64::from).collect(),
    };
    Array2::from_shape_vec((shape[0], shape[1]), values).map_err(|_| Error::Invalid("reader shape"))
}
fn receipt(c: &IndependentCohort, o: &FitOptions, method: &str, bytes: usize) -> RunReceipt {
    RunReceipt {
        method: method.into(),
        input: c.estimand().clone(),
        admission: c.admission().clone(),
        subjects: c.subjects().to_vec(),
        first_level: c.fit_receipts().to_vec(),
        variance_origins: c.shared_origins(),
        model: ModelReceipt::Meta {
            method: MetaMethod::CommonEffect,
        },
        min_observations: o.min_observations,
        missingness: o.missingness,
        feature_count: c.frame().shape()[1],
        blocks: 0,
        factorizations: 0,
        numeric_working_bytes: bytes,
    }
}
fn blank(c: &IndependentCohort, j: usize, k: usize) -> FeatureResult {
    FeatureResult {
        feature: c.frame().features().axis().ids()[j].clone(),
        n: 0,
        rank: 0,
        status: vec![Status::InsufficientObservations; k],
        estimates: vec![None; k],
        heterogeneity: None,
    }
}
fn fit_error(e: &FitError) -> Status {
    match e {
        FitError::NotEstimable => Status::NonEstimable,
        FitError::DegenerateVariance => Status::DegenerateVariance,
        _ => Status::NumericalFailure,
    }
}

/// Fit group OLS with a bounded FIFO cache of missingness-pattern preparations.
/// Sink errors stop execution immediately. Sinks own any outputs they retain.
pub fn fit_ols(
    c: &IndependentCohort,
    d: &GroupDesign,
    contrasts: &[GroupContrast],
    o: FitOptions,
    mut sink: impl FnMut(ResultBlock) -> Result<()>,
) -> Result<RunReceipt> {
    if d.rows != c.frame().observations().ids()
        || contrasts.is_empty()
        || contrasts.iter().any(|x| x.columns != d.columns)
        || contrasts
            .iter()
            .map(|x| &x.id)
            .collect::<HashSet<_>>()
            .len()
            != contrasts.len()
    {
        return Err(Error::Invalid("design binding or contrasts"));
    }
    let [n, nf] = c.frame().shape();
    let k = contrasts.len();
    let p = d.columns.len();
    let bytes = budget(n, p, k, &o, nf)?;
    // Whole-design failure is a request error, not a missing-feature status.
    let full = LinearModel::new(d.values.view(), o.rank_tolerance)?;
    let mut cache = VecDeque::from([((0..n).collect::<Vec<_>>(), full)]);
    let mut r = receipt(c, &o, "OLS: estimated residual scale, two-sided t", bytes);
    r.factorizations = 1;
    r.model = ModelReceipt::Ols {
        columns: d.columns.clone(),
        design: d.values.clone(),
        contrasts: contrasts
            .iter()
            .map(|x| (x.id.clone(), x.weights.to_vec()))
            .collect(),
        rank_tolerance: o.rank_tolerance,
    };
    for start in (0..nf).step_by(o.block_features) {
        let end = (start + o.block_features.min(nf - start)).min(nf);
        let y = read(c, c.effect_assay(), start, end, bytes)?;
        let mut results = (start..end).map(|j| blank(c, j, k)).collect::<Vec<_>>();
        let mut patterns: BTreeMap<Vec<usize>, Vec<usize>> = BTreeMap::new();
        for j in 0..y.ncols() {
            let rows = (0..n)
                .filter(|&i| y[(i, j)].is_finite())
                .collect::<Vec<_>>();
            results[j].n = rows.len();
            if rows.len() != n && o.missingness == Missingness::RejectFeature {
                results[j].status.fill(Status::MissingObservation);
            } else if rows.len() >= o.min_observations {
                patterns.entry(rows).or_default().push(j);
            }
        }
        for (rows, columns) in patterns {
            let pos = if let Some(pos) = cache.iter().position(|(key, _)| *key == rows) {
                pos
            } else {
                let x = Array2::from_shape_fn((rows.len(), p), |(i, j)| d.values[(rows[i], j)]);
                match LinearModel::new(x.view(), o.rank_tolerance) {
                    Ok(model) => {
                        r.factorizations += 1;
                        if cache.len() == o.max_cached_patterns {
                            cache.pop_front();
                        }
                        cache.push_back((rows.clone(), model));
                        cache.len() - 1
                    }
                    Err(_) => {
                        for &j in &columns {
                            results[j].status.fill(Status::UnusableDesign);
                        }
                        continue;
                    }
                }
            };
            let model = &cache[pos].1;
            let data = Array2::from_shape_fn((rows.len(), columns.len()), |(i, j)| {
                y[(rows[i], columns[j])]
            });
            for &j in &columns {
                results[j].rank = model.rank();
            }
            match model.fit(data.view()) {
                Ok(fit) => {
                    for (ci, contrast) in contrasts.iter().enumerate() {
                        match fit.t_contrast(contrast.weights.view()) {
                            Ok(t) => {
                                for (local, &j) in columns.iter().enumerate() {
                                    put_t(&mut results[j], ci, &t, local);
                                }
                            }
                            Err(FitError::NotEstimable) => {
                                for &j in &columns {
                                    results[j].status[ci] = Status::NonEstimable
                                }
                            }
                            Err(_) => {
                                // A constant feature must not poison its neighbors.
                                for (local, &j) in columns.iter().enumerate() {
                                    let one = data.slice(ndarray::s![.., local..local + 1]);
                                    match model
                                        .fit(one)
                                        .and_then(|f| f.t_contrast(contrast.weights.view()))
                                    {
                                        Ok(t) => put_t(&mut results[j], ci, &t, 0),
                                        Err(e) => results[j].status[ci] = fit_error(&e),
                                    }
                                }
                            }
                        }
                    }
                }
                Err(_) => {
                    // Overflow may be local to one feature; isolate it.
                    for (local, &j) in columns.iter().enumerate() {
                        for (ci, contrast) in contrasts.iter().enumerate() {
                            match model
                                .fit(data.slice(ndarray::s![.., local..local + 1]))
                                .and_then(|f| f.t_contrast(contrast.weights.view()))
                            {
                                Ok(t) => put_t(&mut results[j], ci, &t, 0),
                                Err(e) => results[j].status[ci] = fit_error(&e),
                            }
                        }
                    }
                }
            }
        }
        sink(ResultBlock {
            start,
            contrasts: contrasts.iter().map(|c| c.id.clone()).collect(),
            features: results,
        })?;
        r.blocks += 1;
    }
    Ok(r)
}
fn put_t(out: &mut FeatureResult, i: usize, t: &fmrireg::TContrast, j: usize) {
    out.status[i] = Status::Estimated;
    out.estimates[i] = Some(Estimate {
        effect: t.effect[j],
        standard_error: t.standard_error[j],
        statistic: t.statistic[j],
        p_value: t.p_value[j],
        distribution: Distribution::StudentT { df: t.df },
    });
}

/// Scalar meta-analysis. Input variances are treated as fixed for inference,
/// even when the retained provenance says they were estimated at first level.
/// No dimensional variance floor is applied. Unsupported arithmetic extremes
/// produce NumericalFailure, never a silently regularized answer.
pub fn fit_meta(
    c: &IndependentCohort,
    method: MetaMethod,
    o: FitOptions,
    mut sink: impl FnMut(ResultBlock) -> Result<()>,
) -> Result<RunReceipt> {
    let name = c.variance_assay().ok_or(Error::Invalid(
        "meta-analysis requires genuine marginal variance",
    ))?;
    let [n, nf] = c.frame().shape();
    let bytes = budget(n, 1, 1, &o, nf)?;
    let mut r = receipt(
        c,
        &o,
        match method {
            MetaMethod::CommonEffect => "Common effect: fixed input variance, two-sided normal",
            MetaMethod::DerSimonianLaird => "DL random effects: plug-in two-sided normal",
        },
        bytes,
    );
    r.model = ModelReceipt::Meta { method };
    let normal = Normal::new(0., 1.).unwrap();
    for start in (0..nf).step_by(o.block_features) {
        let end = start + o.block_features.min(nf - start);
        let y = read(c, c.effect_assay(), start, end, bytes)?;
        let v = read(c, name, start, end, bytes)?;
        let mut output = Vec::with_capacity(end - start);
        for j in 0..end - start {
            let mut out = blank(c, start + j, 1);
            let rows = (0..n)
                .filter(|&i| y[(i, j)].is_finite() && v[(i, j)].is_finite() && v[(i, j)] > 0.)
                .collect::<Vec<_>>();
            out.n = rows.len();
            if rows.len() != n && o.missingness == Missingness::RejectFeature {
                out.status[0] = Status::MissingObservation;
            } else if rows.len() >= o.min_observations {
                out.rank = 1;
                let pairs = rows
                    .iter()
                    .map(|&i| (y[(i, j)], v[(i, j)]))
                    .collect::<Vec<_>>();
                match meta(&pairs, method, &normal) {
                    Some((estimate, h)) => {
                        out.status[0] = Status::Estimated;
                        out.estimates[0] = Some(estimate);
                        out.heterogeneity = Some(h);
                    }
                    None => out.status[0] = Status::NumericalFailure,
                }
            }
            output.push(out);
        }
        sink(ResultBlock {
            start,
            contrasts: vec!["mean".into()],
            features: output,
        })?;
        r.blocks += 1;
    }
    Ok(r)
}
// Neumaier sum protects cancellation in weighted means without a variance floor.
fn sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut s = 0.;
    let mut c = 0.;
    for x in values {
        let t = s + x;
        if s.abs() >= x.abs() {
            c += (s - t) + x;
        } else {
            c += (x - t) + s;
        }
        s = t;
    }
    s + c
}
fn weighted(pairs: &[(f64, f64)], tau: f64) -> Option<(f64, f64, f64, f64)> {
    let scale = pairs
        .iter()
        .map(|(_, v)| v + tau)
        .fold(f64::INFINITY, f64::min);
    let ymax = pairs.iter().map(|(y, _)| y.abs()).fold(0., f64::max);
    if !scale.is_finite() || scale <= 0. || pairs.iter().any(|(_, v)| !(v + tau).is_finite()) {
        return None;
    }
    let sw = sum(pairs.iter().map(|(_, v)| scale / (v + tau)));
    let mu = if ymax == 0. {
        0.
    } else {
        sum(pairs.iter().map(|(y, v)| (scale / (v + tau)) * (y / ymax))) / sw * ymax
    };
    let se = scale.sqrt() / sw.sqrt();
    if !mu.is_finite() || !se.is_finite() || se <= 0. {
        return None;
    }
    Some((mu, se, scale, sw))
}
fn meta(
    pairs: &[(f64, f64)],
    method: MetaMethod,
    normal: &Normal,
) -> Option<(Estimate, Heterogeneity)> {
    let (mu, _, scale, sw) = weighted(pairs, 0.)?;
    let q = sum(pairs.iter().map(|(y, v)| ((y - mu) / v.sqrt()).powi(2)));
    let df = (pairs.len() - 1) as f64;
    let mut prefix = 0.;
    let mut cross = 0.;
    for &(_, v) in pairs {
        let w = scale / v;
        cross += prefix * w;
        prefix += w;
    }
    let cs = 2. * cross / sw;
    let tau = if method == MetaMethod::DerSimonianLaird && q > df {
        if cs <= 0. {
            return None;
        }
        (q - df) / cs * scale
    } else {
        0.
    };
    if !q.is_finite() || !tau.is_finite() {
        return None;
    }
    let (effect, standard_error, _, _) = weighted(pairs, tau)?;
    let statistic = effect / standard_error;
    if !statistic.is_finite() {
        return None;
    }
    Some((
        Estimate {
            effect,
            standard_error,
            statistic,
            p_value: 2. * normal.sf(statistic.abs()),
            distribution: Distribution::Normal,
        },
        Heterogeneity {
            q,
            i_squared: if q > 0. { (1. - df / q).max(0.) } else { 0. },
            tau_squared: tau,
        },
    ))
}
