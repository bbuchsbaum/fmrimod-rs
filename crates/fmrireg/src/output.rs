//! Compact requested-estimand output from prepared OLS and GLS models.
//!
//! The output path retains only requested linear estimands and optional residual
//! scale, allowing callers to process feature blocks without retaining betas.

use crate::{finite, FitError, GlsModel, LinearModel, Result};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2};
use std::collections::HashSet;
use std::sync::Arc;

/// The uncertainty retained with requested estimands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Uncertainty {
    /// Return estimates only.
    #[default]
    None,
    /// Retain each estimand's marginal variance geometry.
    Marginal,
    /// Retain the full covariance geometry among estimands.
    Joint,
}

/// Named linear estimands, one row of weights per requested output.
#[derive(Debug)]
pub struct EstimandSet {
    ids: Vec<String>,
    weights: Array2<f64>,
}
impl Clone for EstimandSet {
    fn clone(&self) -> Self {
        // Array2::clone may retain hidden backing elements after slice_move.
        // Prepared output contexts account for only the logical weight matrix.
        Self {
            ids: self.ids.clone(),
            weights: self.weights.to_owned(),
        }
    }
}
impl EstimandSet {
    /// Construct general named contrasts. IDs must be unique and nonblank.
    pub fn contrasts(ids: Vec<String>, weights: Array2<f64>) -> Result<Self> {
        if ids.is_empty() || weights.nrows() != ids.len() || weights.ncols() == 0 {
            return Err(FitError::Invalid("estimand dimensions"));
        }
        let mut seen = HashSet::with_capacity(ids.len());
        if ids
            .iter()
            .any(|id| id.trim().is_empty() || !seen.insert(id.clone()))
        {
            return Err(FitError::Invalid("blank or duplicate estimand ID"));
        }
        if weights.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid("nonfinite estimand weights"));
        }
        Ok(Self { ids, weights })
    }

    /// Construct coefficient-selector estimands in the supplied column space.
    pub fn coefficients(ids: Vec<String>, indices: Vec<usize>, columns: usize) -> Result<Self> {
        if columns == 0 || ids.len() != indices.len() {
            return Err(FitError::Invalid("coefficient selector dimensions"));
        }
        let mut weights = Array2::zeros((ids.len(), columns));
        for (row, column) in indices.into_iter().enumerate() {
            if column >= columns {
                return Err(FitError::Invalid("coefficient selector index"));
            }
            weights[(row, column)] = 1.0;
        }
        Self::contrasts(ids, weights)
    }
    pub fn ids(&self) -> &[String] {
        &self.ids
    }
    pub fn weights(&self) -> &Array2<f64> {
        &self.weights
    }
}

/// Requested estimands and retained uncertainty.
#[derive(Clone, Debug)]
pub struct OutputSpec {
    estimands: EstimandSet,
    uncertainty: Uncertainty,
}
impl OutputSpec {
    pub fn new(estimands: EstimandSet) -> Self {
        Self {
            estimands,
            uncertainty: Uncertainty::None,
        }
    }
    pub fn with_uncertainty(mut self, uncertainty: Uncertainty) -> Self {
        self.uncertainty = uncertainty;
        self
    }
    pub fn estimands(&self) -> &EstimandSet {
        &self.estimands
    }
    pub fn uncertainty(&self) -> Uncertainty {
        self.uncertainty
    }
}

#[derive(Clone, Debug)]
pub struct OutputContext {
    spec: OutputSpec,
    geometry: Geometry,
    residual_df: usize,
    conditional_noise: bool,
}
#[derive(Clone, Debug)]
enum Geometry {
    None,
    Marginal(Array1<f64>),
    Joint(Array2<f64>),
}
impl OutputContext {
    pub fn spec(&self) -> &OutputSpec {
        &self.spec
    }
    pub fn estimand_ids(&self) -> &[String] {
        self.spec.estimands.ids()
    }
    pub fn residual_df(&self) -> usize {
        self.residual_df
    }
    /// True for GLS: uncertainty is conditional on a fixed whitening plan and
    /// excludes uncertainty from fitting that plan.
    pub fn conditional_noise(&self) -> bool {
        self.conditional_noise
    }
}

/// A model-bound compact-output plan.
#[derive(Debug)]
pub struct PreparedOutputs<'m, M> {
    model: &'m M,
    context: Arc<OutputContext>,
}
impl<'m, M> Clone for PreparedOutputs<'m, M> {
    fn clone(&self) -> Self {
        Self {
            model: self.model,
            context: Arc::clone(&self.context),
        }
    }
}
#[allow(private_bounds)] // M is sealed to the two prepared public model types.
impl<M> PreparedOutputs<'_, M> {
    pub fn spec(&self) -> &OutputSpec {
        self.context.spec()
    }
    pub fn estimand_ids(&self) -> &[String] {
        self.context.estimand_ids()
    }
    pub fn output_count(&self) -> usize {
        self.context.spec.estimands.weights.nrows()
    }
    /// Whether execution retains RSS/(n - numerical rank) per feature.
    pub fn requires_residual_scale(&self) -> bool {
        self.context.spec.uncertainty != Uncertainty::None
    }
    /// Bytes retained in an owned result for `features` columns: requested
    /// estimates and, when requested, one residual scale per feature. Shared
    /// specification and uncertainty geometry are excluded.
    pub fn retained_numeric_bytes(&self, features: usize) -> Result<usize> {
        if features == 0 {
            return Err(FitError::Invalid("zero output feature count"));
        }
        self.output_count()
            .checked_add(usize::from(self.requires_residual_scale()))
            .and_then(|per_feature| per_feature.checked_mul(features))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f64>()))
            .filter(|&bytes| bytes <= isize::MAX as usize)
            .ok_or(FitError::Invalid("retained output byte overflow"))
    }
}

/// Numeric buffers for [`PreparedOutputs`].
#[derive(Debug)]
pub struct OutputWorkspace {
    context: Arc<OutputContext>,
    n: usize,
    p: usize,
    q: usize,
    capacity: usize,
    coefficients: Vec<f64>,
    estimates: Vec<f64>,
    fitted: Vec<f64>,
    scales: Vec<f64>,
    transformed: Vec<f64>,
}
impl OutputWorkspace {
    fn required_bytes(
        n: usize,
        p: usize,
        q: usize,
        capacity: usize,
        inference: bool,
        transformed: bool,
    ) -> Result<usize> {
        if n == 0 || p == 0 || q == 0 || capacity == 0 {
            return Err(FitError::Invalid("zero workspace dimension"));
        }
        let rows = p
            .checked_add(q)
            .and_then(|x| {
                if inference {
                    x.checked_add(n)?.checked_add(1)
                } else {
                    Some(x)
                }
            })
            .and_then(|x| {
                if transformed {
                    x.checked_add(n)
                } else {
                    Some(x)
                }
            })
            .ok_or(FitError::Invalid("workspace byte overflow"))?;
        rows.checked_mul(capacity)
            .and_then(|x| x.checked_mul(8))
            .filter(|&x| x <= isize::MAX as usize)
            .ok_or(FitError::Invalid("workspace byte overflow"))
    }
    fn zeros(n: usize) -> Result<Vec<f64>> {
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|_| FitError::Invalid("workspace allocation failed"))?;
        v.resize(n, 0.);
        Ok(v)
    }
    fn new(
        context: Arc<OutputContext>,
        n: usize,
        p: usize,
        q: usize,
        capacity: usize,
        inference: bool,
        transformed: bool,
    ) -> Result<Self> {
        Self::required_bytes(n, p, q, capacity, inference, transformed)?;
        let mul = |x: usize| {
            x.checked_mul(capacity)
                .ok_or(FitError::Invalid("workspace byte overflow"))
        };
        Ok(Self {
            context,
            n,
            p,
            q,
            capacity,
            coefficients: Self::zeros(mul(p)?)?,
            estimates: Self::zeros(mul(q)?)?,
            fitted: Self::zeros(if inference { mul(n)? } else { 0 })?,
            scales: Self::zeros(if inference { capacity } else { 0 })?,
            transformed: Self::zeros(if transformed { mul(n)? } else { 0 })?,
        })
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Numeric allocation only; excludes allocator and backend scratch.
    pub fn working_bytes(&self) -> usize {
        (self.coefficients.len()
            + self.estimates.len()
            + self.fitted.len()
            + self.scales.len()
            + self.transformed.len())
            * 8
    }
    #[allow(clippy::too_many_arguments)] // validates the explicit workspace identity and shape contract together.
    fn check(
        &self,
        context: &Arc<OutputContext>,
        n: usize,
        p: usize,
        q: usize,
        b: usize,
        inference: bool,
        transformed: bool,
    ) -> Result<()> {
        if !Arc::ptr_eq(&self.context, context)
            || self.n != n
            || self.p != p
            || self.q != q
            || b == 0
            || b > self.capacity
            || (inference && (self.fitted.len() < n * b || self.scales.len() < b))
            || (transformed && self.transformed.len() < n * b)
        {
            return Err(FitError::Invalid("workspace plan, dimensions or capacity"));
        }
        Ok(())
    }
}

/// Borrowed compact output. Arrays are overwritten when its workspace is reused.
#[derive(Debug)]
pub enum SummaryView<'p, 'w> {
    Estimates(EstimatesView<'p, 'w>),
    Marginal(MarginalView<'p, 'w>),
    Joint(JointView<'p, 'w>),
}
impl<'p, 'w> SummaryView<'p, 'w> {
    pub fn estimates(&self) -> ArrayView2<'w, f64> {
        match self {
            Self::Estimates(x) => x.estimates(),
            Self::Marginal(x) => x.estimates(),
            Self::Joint(x) => x.estimates(),
        }
    }
    pub fn uncertainty(&self) -> Uncertainty {
        self.context().spec.uncertainty
    }
    pub fn spec(&self) -> &OutputSpec {
        self.context().spec()
    }
    pub fn estimand_ids(&self) -> &[String] {
        self.context().estimand_ids()
    }
    pub fn residual_df(&self) -> usize {
        self.context().residual_df()
    }
    pub fn conditional_noise(&self) -> bool {
        self.context().conditional_noise()
    }
    pub fn context(&self) -> &OutputContext {
        match self {
            Self::Estimates(x) => x.context,
            Self::Marginal(x) => x.context,
            Self::Joint(x) => x.context,
        }
    }
    pub fn to_owned(&self) -> Summary {
        match self {
            Self::Estimates(x) => Summary::Estimates(OwnedEstimates {
                estimates: x.estimates.to_owned(),
                context: Arc::clone(x.context),
            }),
            Self::Marginal(x) => Summary::Marginal(OwnedMarginal {
                estimates: x.estimates.to_owned(),
                scales: x.scales.to_owned(),
                context: Arc::clone(x.context),
            }),
            Self::Joint(x) => Summary::Joint(OwnedJoint {
                estimates: x.estimates.to_owned(),
                scales: x.scales.to_owned(),
                context: Arc::clone(x.context),
            }),
        }
    }
}
#[derive(Debug)]
pub struct EstimatesView<'p, 'w> {
    estimates: ArrayView2<'w, f64>,
    context: &'p Arc<OutputContext>,
}
impl<'p, 'w> EstimatesView<'p, 'w> {
    pub fn estimates(&self) -> ArrayView2<'w, f64> {
        self.estimates
    }
    pub fn context(&self) -> &'p OutputContext {
        self.context
    }
}
#[derive(Debug)]
pub struct MarginalView<'p, 'w> {
    estimates: ArrayView2<'w, f64>,
    scales: ArrayView1<'w, f64>,
    context: &'p Arc<OutputContext>,
}
impl<'p, 'w> MarginalView<'p, 'w> {
    pub fn estimates(&self) -> ArrayView2<'w, f64> {
        self.estimates
    }
    pub fn residual_scales(&self) -> ArrayView1<'w, f64> {
        self.scales
    }
    pub fn context(&self) -> &'p OutputContext {
        self.context
    }
    pub fn variance(&self, estimand: usize, feature: usize) -> Result<f64> {
        scaled_geometry(self.context, estimand, estimand, feature, self.scales)
    }
    pub fn standard_error(&self, estimand: usize, feature: usize) -> Result<f64> {
        Ok(self.variance(estimand, feature)?.sqrt())
    }
}
#[derive(Debug)]
pub struct JointView<'p, 'w> {
    estimates: ArrayView2<'w, f64>,
    scales: ArrayView1<'w, f64>,
    context: &'p Arc<OutputContext>,
}
impl<'p, 'w> JointView<'p, 'w> {
    pub fn estimates(&self) -> ArrayView2<'w, f64> {
        self.estimates
    }
    pub fn residual_scales(&self) -> ArrayView1<'w, f64> {
        self.scales
    }
    pub fn context(&self) -> &'p OutputContext {
        self.context
    }
    pub fn covariance(&self, left: usize, right: usize, feature: usize) -> Result<f64> {
        scaled_geometry(self.context, left, right, feature, self.scales)
    }
    pub fn variance(&self, estimand: usize, feature: usize) -> Result<f64> {
        self.covariance(estimand, estimand, feature)
    }
    pub fn standard_error(&self, estimand: usize, feature: usize) -> Result<f64> {
        Ok(self.variance(estimand, feature)?.sqrt())
    }
}

/// Owned compact output, independent of the original model and workspace.
#[derive(Debug)]
pub enum Summary {
    Estimates(OwnedEstimates),
    Marginal(OwnedMarginal),
    Joint(OwnedJoint),
}
impl Summary {
    pub fn estimates(&self) -> &Array2<f64> {
        match self {
            Self::Estimates(x) => &x.estimates,
            Self::Marginal(x) => &x.estimates,
            Self::Joint(x) => &x.estimates,
        }
    }
    pub fn uncertainty(&self) -> Uncertainty {
        self.context().spec.uncertainty
    }
    pub fn spec(&self) -> &OutputSpec {
        self.context().spec()
    }
    pub fn estimand_ids(&self) -> &[String] {
        self.context().estimand_ids()
    }
    pub fn residual_df(&self) -> usize {
        self.context().residual_df()
    }
    pub fn conditional_noise(&self) -> bool {
        self.context().conditional_noise()
    }
    pub fn context(&self) -> &OutputContext {
        match self {
            Self::Estimates(x) => &x.context,
            Self::Marginal(x) => &x.context,
            Self::Joint(x) => &x.context,
        }
    }
    pub fn as_view(&self) -> SummaryView<'_, '_> {
        match self {
            Self::Estimates(x) => SummaryView::Estimates(EstimatesView {
                estimates: x.estimates.view(),
                context: &x.context,
            }),
            Self::Marginal(x) => SummaryView::Marginal(MarginalView {
                estimates: x.estimates.view(),
                scales: x.scales.view(),
                context: &x.context,
            }),
            Self::Joint(x) => SummaryView::Joint(JointView {
                estimates: x.estimates.view(),
                scales: x.scales.view(),
                context: &x.context,
            }),
        }
    }
}
#[derive(Debug)]
pub struct OwnedEstimates {
    estimates: Array2<f64>,
    context: Arc<OutputContext>,
}
impl OwnedEstimates {
    pub fn estimates(&self) -> &Array2<f64> {
        &self.estimates
    }
    pub fn context(&self) -> &OutputContext {
        &self.context
    }
}
#[derive(Debug)]
pub struct OwnedMarginal {
    estimates: Array2<f64>,
    scales: Array1<f64>,
    context: Arc<OutputContext>,
}
impl OwnedMarginal {
    pub fn estimates(&self) -> &Array2<f64> {
        &self.estimates
    }
    pub fn residual_scales(&self) -> &Array1<f64> {
        &self.scales
    }
    pub fn context(&self) -> &OutputContext {
        &self.context
    }
    pub fn variance(&self, e: usize, f: usize) -> Result<f64> {
        scaled_geometry(&self.context, e, e, f, self.scales.view())
    }
    pub fn standard_error(&self, e: usize, f: usize) -> Result<f64> {
        Ok(self.variance(e, f)?.sqrt())
    }
}
#[derive(Debug)]
pub struct OwnedJoint {
    estimates: Array2<f64>,
    scales: Array1<f64>,
    context: Arc<OutputContext>,
}
impl OwnedJoint {
    pub fn estimates(&self) -> &Array2<f64> {
        &self.estimates
    }
    pub fn residual_scales(&self) -> &Array1<f64> {
        &self.scales
    }
    pub fn context(&self) -> &OutputContext {
        &self.context
    }
    pub fn covariance(&self, a: usize, b: usize, f: usize) -> Result<f64> {
        scaled_geometry(&self.context, a, b, f, self.scales.view())
    }
    pub fn variance(&self, e: usize, f: usize) -> Result<f64> {
        self.covariance(e, e, f)
    }
    pub fn standard_error(&self, e: usize, f: usize) -> Result<f64> {
        Ok(self.variance(e, f)?.sqrt())
    }
}

fn scaled_geometry(
    context: &OutputContext,
    left: usize,
    right: usize,
    feature: usize,
    scales: ArrayView1<'_, f64>,
) -> Result<f64> {
    let scale = *scales
        .get(feature)
        .ok_or(FitError::Invalid("feature index"))?;
    let g = match &context.geometry {
        Geometry::None => return Err(FitError::Invalid("uncertainty was not requested")),
        Geometry::Marginal(diagonal) => {
            if left != right {
                return Err(FitError::Invalid("marginal covariance unavailable"));
            }
            *diagonal
                .get(left)
                .ok_or(FitError::Invalid("estimand index"))?
        }
        Geometry::Joint(matrix) => *matrix
            .get((left, right))
            .ok_or(FitError::Invalid("estimand index"))?,
    };
    let value = scale * g;
    if !value.is_finite() {
        return Err(FitError::Invalid("uncertainty overflow"));
    }
    if left == right && value < 0.0 {
        return Err(FitError::Invalid("negative variance"));
    }
    Ok(value)
}

fn prepare<'m, M>(
    model: &'m M,
    core: &LinearModel,
    spec: &OutputSpec,
    conditional_noise: bool,
) -> Result<PreparedOutputs<'m, M>> {
    if spec.estimands.weights.ncols() != core.design.ncols() {
        return Err(FitError::Invalid("estimand column dimensions"));
    }
    for row in spec.estimands.weights.rows() {
        // A zero row is scientifically meaningful (and has exact zero variance),
        // while all nonzero rows retain LinearModel's rank/tolerance test.
        if row.iter().any(|x| *x != 0.0) {
            core.check_contrast(row)?;
        }
    }
    let mut geometry = Geometry::None;
    if spec.uncertainty == Uncertainty::Marginal {
        // H = L X^+; diagonal G is each squared row norm of H.  This avoids
        // cancellation in L (X'X)^- L' and retains no q by n temporary.
        let mut diagonal = Array1::zeros(spec.estimands.weights.nrows());
        for (out, row) in spec.estimands.weights.rows().into_iter().enumerate() {
            let mut value = 0.0;
            for observation in 0..core.inverse.ncols() {
                let mut h = 0.0;
                for column in 0..row.len() {
                    h += row[column] * core.inverse[(column, observation)];
                }
                if !h.is_finite() {
                    return Err(FitError::Invalid("estimand covariance overflow"));
                }
                value += h * h;
            }
            if !value.is_finite() {
                return Err(FitError::Invalid("estimand covariance overflow"));
            }
            if value < 0.0 {
                return Err(FitError::Invalid("negative estimand variance geometry"));
            }
            diagonal[out] = value;
        }
        geometry = Geometry::Marginal(diagonal);
    } else if spec.uncertainty == Uncertainty::Joint {
        // A common H gives G = H H' by construction, retaining dependent and
        // zero estimands as PSD geometry without eigenvalue clamping.
        let h = spec.estimands.weights.dot(&core.inverse);
        if h.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid("estimand covariance overflow"));
        }
        let q = h.nrows();
        let mut joint = h.dot(&h.t());
        if joint.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid("estimand covariance overflow"));
        }
        // The algebraic covariance is symmetric.  Store a symmetric roundoff
        // representative, without eigenvalue clamping or rank alteration.
        for i in 0..q {
            for j in 0..i {
                let value = 0.5 * joint[(i, j)] + 0.5 * joint[(j, i)];
                if !value.is_finite() {
                    return Err(FitError::Invalid("estimand covariance overflow"));
                }
                joint[(i, j)] = value;
                joint[(j, i)] = value;
            }
        }
        if joint.diag().iter().any(|x| *x < 0.0) {
            return Err(FitError::Invalid("negative estimand variance geometry"));
        }
        geometry = Geometry::Joint(joint);
    }
    Ok(PreparedOutputs {
        model,
        context: Arc::new(OutputContext {
            spec: spec.clone(),
            geometry,
            residual_df: core.residual_df(),
            conditional_noise,
        }),
    })
}
impl LinearModel {
    /// Validate estimands at this model's numerical rank and prepare output.
    pub fn prepare_outputs(&self, spec: &OutputSpec) -> Result<PreparedOutputs<'_, Self>> {
        prepare(self, self, spec, false)
    }
}
impl<P: fmri_noise::WhiteningPlan> GlsModel<P> {
    /// Validate estimands after whitening; uncertainty is conditional on this plan.
    pub fn prepare_outputs(&self, spec: &OutputSpec) -> Result<PreparedOutputs<'_, Self>> {
        prepare(self, &self.model, spec, true)
    }
}

mod private {
    pub trait Sealed {}
    impl Sealed for crate::LinearModel {}
    impl<P: fmri_noise::WhiteningPlan> Sealed for crate::GlsModel<P> {}
}
/// Crate-private execution seam for dataset/block adapters.
pub(crate) trait OutputModel: private::Sealed {
    fn output_core(&self) -> &LinearModel;
    fn output_uses_transform(&self) -> bool;
    fn output_execute(
        &self,
        context: &Arc<OutputContext>,
        data: ArrayView2<'_, f64>,
        workspace: &mut OutputWorkspace,
    ) -> Result<()>;
}
impl OutputModel for LinearModel {
    fn output_core(&self) -> &LinearModel {
        self
    }
    fn output_uses_transform(&self) -> bool {
        false
    }
    fn output_execute(
        &self,
        context: &Arc<OutputContext>,
        data: ArrayView2<'_, f64>,
        workspace: &mut OutputWorkspace,
    ) -> Result<()> {
        self.validate_data(data)?;
        execute_buffers(self, context, data, workspace)
    }
}
impl<P: fmri_noise::WhiteningPlan> OutputModel for GlsModel<P> {
    fn output_core(&self) -> &LinearModel {
        &self.model
    }
    fn output_uses_transform(&self) -> bool {
        true
    }
    fn output_execute(
        &self,
        context: &Arc<OutputContext>,
        data: ArrayView2<'_, f64>,
        workspace: &mut OutputWorkspace,
    ) -> Result<()> {
        self.model.validate_shape(data)?;
        let (n, b) = data.dim();
        let OutputWorkspace {
            transformed,
            coefficients,
            fitted,
            scales,
            estimates,
            ..
        } = workspace;
        let mut transformed_view =
            ArrayViewMut2::from_shape((n, b), &mut transformed[..n * b]).unwrap();
        self.noise.apply_into(data, transformed_view.view_mut())?;
        if !finite::all_finite(transformed_view.view()) {
            return Err(FitError::Invalid("nonfinite whitened data"));
        }
        let whitened = transformed_view.view();
        execute_buffers_parts(
            &self.model,
            context,
            whitened,
            coefficients,
            fitted,
            scales,
            estimates,
        )
    }
}
#[allow(private_bounds)]
impl<'m, M: OutputModel> PreparedOutputs<'m, M> {
    pub fn observations(&self) -> usize {
        self.model.output_core().design.nrows()
    }
    pub fn columns(&self) -> usize {
        self.model.output_core().design.ncols()
    }
    pub fn workspace_bytes(&self, capacity: usize) -> Result<usize> {
        OutputWorkspace::required_bytes(
            self.observations(),
            self.columns(),
            self.output_count(),
            capacity,
            self.context.spec.uncertainty != Uncertainty::None,
            self.model.output_uses_transform(),
        )
    }
    pub fn workspace(&self, capacity: usize) -> Result<OutputWorkspace> {
        OutputWorkspace::new(
            self.context.clone(),
            self.observations(),
            self.columns(),
            self.output_count(),
            capacity,
            self.context.spec.uncertainty != Uncertainty::None,
            self.model.output_uses_transform(),
        )
    }
    pub fn run_into<'p, 'w>(
        &'p self,
        data: ArrayView2<'_, f64>,
        workspace: &'w mut OutputWorkspace,
    ) -> Result<SummaryView<'p, 'w>> {
        self.execute(data, workspace)?;
        self.view(workspace, data.ncols())
    }
    pub(crate) fn execute(
        &self,
        data: ArrayView2<'_, f64>,
        workspace: &mut OutputWorkspace,
    ) -> Result<()> {
        check_workspace(self, workspace, data.nrows(), data.ncols())?;
        self.model.output_execute(&self.context, data, workspace)
    }
    pub(crate) fn view<'p, 'w>(
        &'p self,
        workspace: &'w OutputWorkspace,
        b: usize,
    ) -> Result<SummaryView<'p, 'w>> {
        view_from_workspace(self, workspace, b)
    }
}
fn check_workspace<M: OutputModel>(
    plan: &PreparedOutputs<'_, M>,
    workspace: &OutputWorkspace,
    n: usize,
    b: usize,
) -> Result<()> {
    let core = plan.model.output_core();
    workspace.check(
        &plan.context,
        core.design.nrows(),
        core.design.ncols(),
        plan.output_count(),
        b,
        plan.context.spec.uncertainty != Uncertainty::None,
        plan.model.output_uses_transform(),
    )?;
    if n != core.design.nrows() {
        return Err(FitError::Invalid("data dimensions"));
    }
    Ok(())
}
fn execute_buffers(
    core: &LinearModel,
    context: &Arc<OutputContext>,
    data: ArrayView2<'_, f64>,
    workspace: &mut OutputWorkspace,
) -> Result<()> {
    let OutputWorkspace {
        coefficients,
        fitted,
        scales,
        estimates,
        ..
    } = workspace;
    execute_buffers_parts(core, context, data, coefficients, fitted, scales, estimates)
}
fn execute_buffers_parts(
    core: &LinearModel,
    context: &Arc<OutputContext>,
    data: ArrayView2<'_, f64>,
    coefficients: &mut [f64],
    fitted: &mut [f64],
    scales: &mut [f64],
    estimates_buffer: &mut [f64],
) -> Result<()> {
    let (_, b) = data.dim();
    let q = context.spec.estimands.weights.nrows();
    if context.spec.uncertainty == Uncertainty::None {
        let mut betas = ArrayViewMut2::from_shape(
            (core.design.ncols(), b),
            &mut coefficients[..core.design.ncols() * b],
        )
        .unwrap();
        core.backend
            .multiply(core.inverse.view(), data, betas.view_mut());
        if betas.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid("fit overflow; rescale inputs"));
        }
    } else {
        // Both OutputModel implementations validate shape and finiteness before
        // entering this helper (GLS validates after the caller's transform).
        // Reuse the validated kernel instead of scanning the block twice.
        core.fit_buffers_validated(data, coefficients, fitted, scales)?;
        if scales[..b].iter().any(|x| !x.is_finite() || *x < 0.) {
            return Err(FitError::Invalid("invalid residual scale"));
        }
    }
    let betas = ArrayView2::from_shape(
        (core.design.ncols(), b),
        &coefficients[..core.design.ncols() * b],
    )
    .unwrap();
    let mut estimates = ArrayViewMut2::from_shape((q, b), &mut estimates_buffer[..q * b]).unwrap();
    core.backend.multiply(
        context.spec.estimands.weights.view(),
        betas,
        estimates.view_mut(),
    );
    if estimates.iter().any(|x| !x.is_finite()) {
        return Err(FitError::Invalid(
            "estimand overflow; rescale weights or inputs",
        ));
    }
    Ok(())
}
fn view_from_workspace<'p, 'w, M: OutputModel>(
    plan: &'p PreparedOutputs<'_, M>,
    workspace: &'w OutputWorkspace,
    b: usize,
) -> Result<SummaryView<'p, 'w>> {
    check_workspace(plan, workspace, plan.observations(), b)?;
    let estimates = ArrayView2::from_shape(
        (plan.output_count(), b),
        &workspace.estimates[..plan.output_count() * b],
    )
    .unwrap();
    Ok(match plan.context.spec.uncertainty {
        Uncertainty::None => SummaryView::Estimates(EstimatesView {
            estimates,
            context: &plan.context,
        }),
        Uncertainty::Marginal => SummaryView::Marginal(MarginalView {
            estimates,
            scales: ArrayView1::from(&workspace.scales[..b]),
            context: &plan.context,
        }),
        Uncertainty::Joint => SummaryView::Joint(JointView {
            estimates,
            scales: ArrayView1::from(&workspace.scales[..b]),
            context: &plan.context,
        }),
    })
}

/// Coefficients and residual scales explicitly retained for future contrast edits.
/// Bound to the exact prepared model by an opaque identity; not serializable and
/// never accepted by a different model, even when dimensions/designs coincide.
#[derive(Debug)]
pub struct RetainedFit {
    model_identity: Arc<()>,
    coefficients: Array2<f64>,
    scales: Array1<f64>,
}
impl RetainedFit {
    pub fn features(&self) -> usize {
        self.coefficients.ncols()
    }
    pub fn numeric_bytes(&self) -> usize {
        8 * (self.coefficients.len() + self.scales.len())
    }
}
impl crate::FitView<'_, '_> {
    /// Explicitly pay for retaining all coefficients and scales for this block.
    pub fn retain_for_outputs(&self) -> Result<RetainedFit> {
        if self.coefficients.iter().any(|x| !x.is_finite())
            || self.variance.iter().any(|x| !x.is_finite() || *x < 0.)
        {
            return Err(FitError::Invalid("nonfinite retained fit"));
        }
        Ok(RetainedFit {
            model_identity: Arc::clone(&self.model.identity),
            coefficients: self.coefficients.to_owned(),
            scales: self.variance.to_owned(),
        })
    }
}
#[allow(private_bounds)]
impl<M: OutputModel> PreparedOutputs<'_, M> {
    /// Apply validated requested weights to a retained fit without reading or
    /// whitening time series. Geometry and residual df come from this model.
    /// Only output arrays allocate; `retained_numeric_bytes(range.len())` bounds
    /// their numerical storage. Backend packing remains outside that bound.
    pub fn summarize_retained(
        &self,
        fit: &RetainedFit,
        range: std::ops::Range<usize>,
    ) -> Result<Summary> {
        if !Arc::ptr_eq(&fit.model_identity, &self.model.output_core().identity) {
            return Err(FitError::Invalid(
                "retained fit belongs to a different model",
            ));
        }
        if range.start >= range.end || range.end > fit.features() {
            return Err(FitError::Invalid("retained feature range"));
        }
        self.retained_numeric_bytes(range.len())?;
        let mut estimates = Array2::zeros((self.output_count(), range.len()));
        self.model.output_core().backend.multiply(
            self.context.spec.estimands.weights.view(),
            fit.coefficients.slice(ndarray::s![.., range.clone()]),
            estimates.view_mut(),
        );
        if estimates.iter().any(|x| !x.is_finite()) {
            return Err(FitError::Invalid(
                "estimand overflow; rescale weights or inputs",
            ));
        }
        let context = Arc::clone(&self.context);
        Ok(match self.spec().uncertainty() {
            Uncertainty::None => Summary::Estimates(OwnedEstimates { estimates, context }),
            Uncertainty::Marginal => Summary::Marginal(OwnedMarginal {
                estimates,
                scales: fit.scales.slice(ndarray::s![range]).to_owned(),
                context,
            }),
            Uncertainty::Joint => Summary::Joint(OwnedJoint {
                estimates,
                scales: fit.scales.slice(ndarray::s![range]).to_owned(),
                context,
            }),
        })
    }
}
