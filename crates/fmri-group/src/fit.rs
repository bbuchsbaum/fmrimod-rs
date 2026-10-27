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
    /// Successful design factorizations performed for this execution, including
    /// initial preparation on the first use of a `PreparedOls`. The intercept
    /// reducer performs none; later prepared runs count only new cache entries.
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
    MetaRegression {
        columns: Vec<String>,
        design: Array2<f64>,
        contrasts: Vec<(String, Vec<f64>)>,
        rank_tolerance: f64,
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
    pub(crate) rows: Vec<ObservationId>,
    pub(crate) columns: Vec<String>,
    pub(crate) values: Array2<f64>,
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
    pub(crate) id: String,
    pub(crate) columns: Vec<String>,
    pub(crate) weights: Array1<f64>,
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
pub(crate) fn budget(n: usize, p: usize, k: usize, o: &FitOptions, nf: usize) -> Result<usize> {
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
pub(crate) fn read(
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
pub(crate) fn receipt(
    c: &IndependentCohort,
    o: &FitOptions,
    method: &str,
    bytes: usize,
) -> RunReceipt {
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

/// Reusable, row-bound group OLS preparation. Cache and numeric workspaces are
/// owned by this value; each call rereads the supplied cohort and produces a new
/// receipt. Feature axes may differ, but ordered observation IDs must match.
///
/// The all-ones single-column design uses centered moments. Other designs retain
/// the existing solver's rank policy and explicit residual calculation.
#[derive(Debug)]
pub struct PreparedOls {
    design: GroupDesign,
    contrasts: Vec<GroupContrast>,
    options: FitOptions,
    intercept: bool,
    cache: VecDeque<(Vec<usize>, LinearModel)>,
    workspace: Option<(usize, fmrireg::FitWorkspace)>,
    gathered: Vec<f64>,
    moments: Vec<Moment>,
    retained_features: usize,
    pending_factorizations: usize,
}
impl PreparedOls {
    pub fn new(d: &GroupDesign, contrasts: &[GroupContrast], o: FitOptions) -> Result<Self> {
        if contrasts.is_empty()
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
        budget(d.rows.len(), d.columns.len(), contrasts.len(), &o, 1)?;
        let intercept = d.columns.len() == 1 && d.values.iter().all(|&x| x == 1.);
        let mut cache = VecDeque::new();
        if intercept {
            if d.rows.len() < 2 {
                return Err(
                    FitError::Invalid("no residual degrees of freedom or zero rank").into(),
                );
            }
        } else {
            // Whole-design failure remains a request error.
            cache.push_back((
                (0..d.rows.len()).collect(),
                LinearModel::new(d.values.view(), o.rank_tolerance)?,
            ));
        }
        Ok(Self {
            design: d.clone(),
            contrasts: contrasts.to_vec(),
            options: o,
            intercept,
            pending_factorizations: cache.len(),
            cache,
            workspace: None,
            gathered: Vec::new(),
            moments: Vec::new(),
            retained_features: 0,
        })
    }

    pub fn fit(
        &mut self,
        c: &IndependentCohort,
        mut sink: impl FnMut(ResultBlock) -> Result<()>,
    ) -> Result<RunReceipt> {
        if self.design.rows != c.frame().observations().ids() {
            return Err(Error::Invalid("design binding or contrasts"));
        }
        let o = self.options;
        let [n, nf] = c.frame().shape();
        let k = self.contrasts.len();
        let bytes = budget(
            n,
            self.design.columns.len(),
            k,
            &o,
            nf.max(self.retained_features),
        )?;
        self.retained_features = self.retained_features.max(o.block_features.min(nf));
        let mut r = receipt(c, &o, "OLS: estimated residual scale, two-sided t", bytes);
        // Includes preparation since the previous successful start; subsequent
        // fits count only new cache entries, never pretend reuse is a refactor.
        r.factorizations = std::mem::take(&mut self.pending_factorizations);
        r.model = ModelReceipt::Ols {
            columns: self.design.columns.clone(),
            design: self.design.values.clone(),
            contrasts: self
                .contrasts
                .iter()
                .map(|x| (x.id.clone(), x.weights.to_vec()))
                .collect(),
            rank_tolerance: o.rank_tolerance,
        };
        let all_rows = (0..n).collect::<Vec<_>>();
        for start in (0..nf).step_by(o.block_features) {
            let end = start + o.block_features.min(nf - start);
            let y = read(c, c.effect_assay(), start, end, bytes)?;
            let mut results = (start..end).map(|j| blank(c, j, k)).collect::<Vec<_>>();
            if self.intercept {
                intercept_block(
                    y.view(),
                    &self.contrasts,
                    &o,
                    &mut results,
                    &mut self.moments,
                );
            } else if y.iter().all(|x| x.is_finite()) {
                for out in &mut results {
                    out.n = n;
                }
                if n >= o.min_observations {
                    // Borrow the reader's complete block directly: no grouping
                    // keys or second copy of its numeric data.
                    let columns = (0..y.ncols()).collect::<Vec<_>>();
                    self.fit_pattern(&all_rows, &columns, y.view(), true, &mut results, &mut r)?;
                }
            } else {
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
                    self.fit_pattern(&rows, &columns, y.view(), false, &mut results, &mut r)?;
                }
            }
            sink(ResultBlock {
                start,
                contrasts: self.contrasts.iter().map(|c| c.id.clone()).collect(),
                features: results,
            })?;
            r.blocks += 1;
        }
        Ok(r)
    }

    fn fit_pattern(
        &mut self,
        rows: &[usize],
        columns: &[usize],
        y: ndarray::ArrayView2<'_, f64>,
        complete: bool,
        results: &mut [FeatureResult],
        r: &mut RunReceipt,
    ) -> Result<()> {
        let pos = if let Some(pos) = self.cache.iter().position(|(key, _)| key == rows) {
            pos
        } else {
            let x = Array2::from_shape_fn((rows.len(), self.design.columns.len()), |(i, j)| {
                self.design.values[(rows[i], j)]
            });
            match LinearModel::new(x.view(), self.options.rank_tolerance) {
                Ok(model) => {
                    r.factorizations += 1;
                    if self.cache.len() == self.options.max_cached_patterns {
                        self.cache.pop_front();
                    }
                    self.cache.push_back((rows.to_vec(), model));
                    self.cache.len() - 1
                }
                Err(_) => {
                    for &j in columns {
                        results[j].status.fill(Status::UnusableDesign);
                    }
                    return Ok(());
                }
            }
        };
        let model = &self.cache[pos].1;
        let capacity = self.options.block_features.min(r.feature_count);
        if self
            .workspace
            .as_ref()
            .is_none_or(|(n, w)| *n != rows.len() || w.capacity() < capacity)
        {
            self.workspace = Some((rows.len(), model.workspace(capacity)?));
        }
        let data = if complete {
            y.view()
        } else {
            self.gathered.clear();
            for &i in rows {
                for &j in columns {
                    self.gathered.push(y[(i, j)]);
                }
            }
            ndarray::ArrayView2::from_shape((rows.len(), columns.len()), &self.gathered).unwrap()
        };
        for &j in columns {
            results[j].rank = model.rank();
        }
        fit_pattern_outputs(
            model,
            &mut self.workspace.as_mut().unwrap().1,
            data,
            &self.contrasts,
            columns,
            results,
        );
        Ok(())
    }
}

/// Fit once. Use [`PreparedOls`] to reuse preparation across calls.
/// Sink errors stop execution immediately. Sinks own any outputs they retain.
pub fn fit_ols(
    c: &IndependentCohort,
    d: &GroupDesign,
    contrasts: &[GroupContrast],
    o: FitOptions,
    sink: impl FnMut(ResultBlock) -> Result<()>,
) -> Result<RunReceipt> {
    if d.rows != c.frame().observations().ids() {
        return Err(Error::Invalid("design binding or contrasts"));
    }
    // Enforce the whole-call budget before allocating model preparation.
    budget(
        c.subjects().len(),
        d.columns.len(),
        contrasts.len(),
        &o,
        c.frame().shape()[1],
    )?;
    PreparedOls::new(d, contrasts, o)?.fit(c, sink)
}

fn fit_pattern_outputs(
    model: &LinearModel,
    workspace: &mut fmrireg::FitWorkspace,
    data: ndarray::ArrayView2<'_, f64>,
    contrasts: &[GroupContrast],
    columns: &[usize],
    results: &mut [FeatureResult],
) {
    match model.fit_into(data, workspace) {
        Ok(fit) => {
            for (ci, contrast) in contrasts.iter().enumerate() {
                match fit.t_contrast(contrast.weights.view()) {
                    Ok(t) => {
                        for (local, &j) in columns.iter().enumerate() {
                            put_t(&mut results[j], ci, &t, local);
                        }
                    }
                    Err(FitError::NotEstimable) => {
                        for &j in columns {
                            results[j].status[ci] = Status::NonEstimable;
                        }
                    }
                    Err(_) => {
                        for (local, &j) in columns.iter().enumerate() {
                            // A degenerate feature must not poison its neighbors.
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
        Err(_) => {
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

#[derive(Clone, Copy, Debug, Default)]
struct Compensated {
    value: f64,
    correction: f64,
}
impl Compensated {
    fn add(&mut self, x: f64) {
        let t = self.value + x;
        self.correction += if self.value.abs() >= x.abs() {
            (self.value - t) + x
        } else {
            (x - t) + self.value
        };
        self.value = t;
    }
    fn total(self) -> f64 {
        self.value + self.correction
    }
}
#[derive(Clone, Copy, Debug, Default)]
struct Moment {
    n: usize,
    anchor: f64,
    offset: Compensated,
    rss: Compensated,
}
fn intercept_block(
    y: ndarray::ArrayView2<'_, f64>,
    contrasts: &[GroupContrast],
    o: &FitOptions,
    results: &mut [FeatureResult],
    moments: &mut Vec<Moment>,
) {
    moments.clear();
    moments.resize(y.ncols(), Moment::default());
    // Traverse contiguous rows, not strided feature columns. Center around an
    // observed anchor so a large response offset does not destroy small variance.
    for row in y.rows() {
        for (m, &value) in moments.iter_mut().zip(row) {
            if value.is_finite() {
                if m.n == 0 {
                    m.anchor = value;
                }
                m.n += 1;
                m.offset.add(value - m.anchor);
            }
        }
    }
    for m in moments.iter_mut() {
        m.offset.value = m.offset.total() / m.n as f64;
        m.offset.correction = 0.;
    }
    for row in y.rows() {
        for (m, &value) in moments.iter_mut().zip(row) {
            if value.is_finite() {
                let residual = (value - m.anchor) - m.offset.value;
                m.rss.add(residual * residual);
            }
        }
    }
    for (out, m) in results.iter_mut().zip(moments.iter()) {
        out.n = m.n;
        if m.n != y.nrows() && o.missingness == Missingness::RejectFeature {
            out.status.fill(Status::MissingObservation);
            continue;
        }
        if m.n < o.min_observations {
            continue;
        }
        out.rank = 1;
        let mean = m.anchor + m.offset.value;
        let variance = m.rss.total() / (m.n - 1) as f64;
        if !mean.is_finite() || !variance.is_finite() {
            out.status.fill(Status::NumericalFailure);
            continue;
        }
        if variance <= 0. {
            out.status.fill(Status::DegenerateVariance);
            continue;
        }
        let sem = variance.sqrt() / (m.n as f64).sqrt();
        let dist = statrs::distribution::StudentsT::new(0., 1., (m.n - 1) as f64).unwrap();
        for (ci, contrast) in contrasts.iter().enumerate() {
            let weight = contrast.weights[0];
            let effect = weight * mean;
            let se = weight.abs() * sem;
            if !se.is_finite() || se <= 0. {
                out.status[ci] = Status::DegenerateVariance;
                continue;
            }
            let statistic = effect / se;
            if !effect.is_finite() || !statistic.is_finite() {
                out.status[ci] = Status::NumericalFailure;
                continue;
            }
            out.status[ci] = Status::Estimated;
            out.estimates[ci] = Some(Estimate {
                effect,
                standard_error: se,
                statistic,
                p_value: 2. * dist.sf(statistic.abs()),
                distribution: Distribution::StudentT { df: m.n - 1 },
            });
        }
    }
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
    // Small feature tiles make row-major input access contiguous while keeping
    // each reducer's retained pairs contiguous. Reuse scratch for the whole run.
    let tile = 32.min(o.block_features).min(nf);
    let mut pairs = vec![(0., 0.); n * tile];
    let mut counts = vec![0; tile];
    for start in (0..nf).step_by(o.block_features) {
        let end = start + o.block_features.min(nf - start);
        let y = read(c, c.effect_assay(), start, end, bytes)?;
        let v = read(c, name, start, end, bytes)?;
        let mut output = Vec::with_capacity(end - start);
        for offset in (0..end - start).step_by(tile) {
            let width = tile.min(end - start - offset);
            counts.fill(0);
            for (ys, vs) in y.rows().into_iter().zip(v.rows()) {
                for local in 0..width {
                    let effect = ys[offset + local];
                    let variance = vs[offset + local];
                    if effect.is_finite() && variance.is_finite() && variance > 0. {
                        pairs[local * n + counts[local]] = (effect, variance);
                        counts[local] += 1;
                    }
                }
            }
            for (local, &count) in counts[..width].iter().enumerate() {
                let j = offset + local;
                let mut out = blank(c, start + j, 1);
                out.n = count;
                if count != n && o.missingness == Missingness::RejectFeature {
                    out.status[0] = Status::MissingObservation;
                } else if count >= o.min_observations {
                    out.rank = 1;
                    match meta(&pairs[local * n..local * n + count], method, &normal) {
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
    let (mu, se, scale, sw) = weighted(pairs, 0.)?;
    let q = sum(pairs.iter().map(|(y, v)| ((y - mu) / v.sqrt()).powi(2)));
    let df = (pairs.len() - 1) as f64;
    let tau = if method == MetaMethod::DerSimonianLaird && q > df {
        // Positive cross-products avoid cancellation in sum(w)-sum(w²)/sum(w).
        let mut prefix = 0.;
        let mut cross = 0.;
        for &(_, v) in pairs {
            let w = scale / v;
            cross += prefix * w;
            prefix += w;
        }
        let cs = 2. * cross / sw;
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
    let (effect, standard_error) = if tau == 0. {
        (mu, se)
    } else {
        let (effect, se, _, _) = weighted(pairs, tau)?;
        (effect, se)
    };
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
