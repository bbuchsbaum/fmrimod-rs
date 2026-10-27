//! One-step stationary AR(1) feasible GLS on explicitly retained observations.
use crate::{GlsModel, LinearModel, Result};
use fmri_noise::{
    selected_estimation::{SelectedAr1Diagnostics, SelectedAr1Estimator},
    GapPolicy, ObservationSelection, Pooling, SelectedArPlan,
};
use ndarray::ArrayView2;
/// No automatic lag reduction or stationarity clipping. The correction truncates
/// covariance after max_lag. Global pooling assumes a shared stationary ACVF;
/// Run pooling solves coupled run-specific ACVFs and requires data in every run.
pub struct SelectedAr1Options {
    pub selection: ObservationSelection,
    pub gaps: GapPolicy,
    pub pooling: Pooling,
    pub max_lag: usize,
    pub relative_tolerance: f64,
}
pub struct EstimatedSelectedAr1 {
    pub model: GlsModel<SelectedArPlan>,
    pub diagnostics: SelectedAr1Diagnostics,
}
/// Select original X and representative noise data, fit OLS, correct its residual
/// lag moments, then prepare one-step GLS. Subsequent fits take *selected* data.
/// Inference is conditional on the estimated coefficient and ignores its uncertainty.
pub fn estimate_selected_ar1(
    original_design: ArrayView2<'_, f64>,
    original_noise_data: ArrayView2<'_, f64>,
    options: SelectedAr1Options,
) -> Result<EstimatedSelectedAr1> {
    let x = options.selection.select(original_design)?;
    let y = options.selection.select(original_noise_data)?;
    let initial = LinearModel::new(x.view(), options.relative_tolerance)?;
    let fit = initial.fit(y.view())?;
    let residuals = &y - &x.dot(fit.coefficients());
    let estimator = SelectedAr1Estimator::new(
        x.view(),
        options.selection,
        options.gaps,
        options.pooling,
        options.max_lag,
        options.relative_tolerance,
    )?;
    let (noise, diagnostics) = estimator.fit(residuals.view())?;
    Ok(EstimatedSelectedAr1 {
        model: GlsModel::new(x.view(), noise, options.relative_tolerance)?,
        diagnostics,
    })
}
