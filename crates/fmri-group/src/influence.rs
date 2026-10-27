//! Exact participant deletion; no automatic exclusion or approximate rank update.
use crate::*;
use fmridataset::{FeatureId, ObservationId};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeletionHeterogeneity {
    /// Estimate tau² again using the selected FE/DL/PM method after each deletion.
    Refit,
    /// Hold full-feature tau² fixed; refit coefficients and requested uncertainty.
    /// mKH still uses the deleted fit's Q and residual degrees of freedom.
    FixedAtFullFit,
}
#[derive(Clone, Debug)]
pub struct MetaDeletion {
    pub omitted: ObservationId,
    /// None if the omitted participant was already unavailable for this feature,
    /// or if the full fit failed. Otherwise preserves the exact refit's status.
    pub fit: Option<MetaRegressionFeature>,
    /// Deleted minus full fit, in requested contrast order.
    pub effect_change: Option<Vec<f64>>,
    /// Change divided by the full fit's SE; this is not a hypothesis test.
    pub change_in_full_se: Option<Vec<f64>>,
}
#[derive(Clone, Debug)]
pub struct MetaInfluenceFeature {
    pub feature: FeatureId,
    pub full: MetaRegressionFeature,
    pub deletions: Vec<MetaDeletion>,
}
#[derive(Clone, Debug)]
pub struct MetaInfluenceBlock {
    pub start: usize,
    pub features: Vec<MetaInfluenceFeature>,
}
#[derive(Clone, Debug)]
pub struct MetaInfluenceReceipt {
    pub design: GroupDesign,
    pub outputs: MetaOutputSpec,
    pub options: MetaRegressionOptions,
    pub heterogeneity: DeletionHeterogeneity,
    pub subjects: Vec<Subject>,
    pub estimand: Estimand,
    pub admission: Admission,
    pub first_level: Vec<FitReceipt>,
    pub variance_origins: Vec<VarianceOrigin>,
    pub fits_attempted: usize,
    pub numeric_working_bytes: usize,
}

/// Refit every available subject deletion for each feature. This is deliberately
/// exact and costs O(subjects) fits per feature. The sink receives bounded blocks;
/// retained sink output and caller-owned sources are outside the numeric budget.
pub fn meta_influence(
    cohort: &IndependentCohort,
    design: &GroupDesign,
    outputs: MetaOutputSpec,
    options: MetaRegressionOptions,
    heterogeneity: DeletionHeterogeneity,
    mut sink: impl FnMut(MetaInfluenceBlock) -> Result<()>,
) -> Result<MetaInfluenceReceipt> {
    if design.rows != cohort.frame().observations().ids() {
        return Err(Error::Invalid("influence observation binding"));
    }
    let variance = cohort
        .variance_assay()
        .ok_or(Error::Invalid("influence requires sampling variance"))?;
    let [n, nf] = cohort.frame().shape();
    let p = design.columns.len();
    let k = outputs.contrasts.len();
    let b = options.execution.block_features.min(nf) as u128;
    let bytes = 8u128
        * (16 * n as u128 * b * (k as u128 * k as u128 + k as u128 + 1)
            + 32 * n as u128 * (p as u128 + 1)
            + 16 * p as u128 * p as u128);
    if bytes > options.execution.max_working_bytes as u128 {
        return Err(Error::Budget);
    }
    let bytes = bytes as usize;
    let mut plan = PreparedMetaRegression::new(design, outputs.clone(), options)?;
    let mut fits_attempted = 0;
    for start in (0..nf).step_by(options.execution.block_features) {
        let end = start + options.execution.block_features.min(nf - start);
        let y = crate::fit::read(cohort, cohort.effect_assay(), start, end, bytes)?;
        let v = crate::fit::read(cohort, variance, start, end, bytes)?;
        let mut features = Vec::with_capacity(end - start);
        for j in 0..end - start {
            let feature = cohort.frame().features().axis().ids()[start + j].clone();
            let rows = (0..n)
                .filter(|&i| y[(i, j)].is_finite() && v[(i, j)].is_finite() && v[(i, j)] > 0.)
                .collect::<Vec<_>>();
            let yy = rows.iter().map(|&i| y[(i, j)]).collect::<Vec<_>>();
            let vv = rows.iter().map(|&i| v[(i, j)]).collect::<Vec<_>>();
            let (status, estimates, diagnostics) =
                if rows.len() != n && options.execution.missingness == Missingness::RejectFeature {
                    (MetaStatus::MissingObservation, None, None)
                } else {
                    fits_attempted += 1;
                    plan.fit_retained(&rows, &yy, &vv, None)
                };
            let full = MetaRegressionFeature {
                feature: feature.clone(),
                n: rows.len(),
                rank: if estimates.is_some() { p } else { 0 },
                status,
                estimates,
                diagnostics,
            };
            let mut deletions = Vec::with_capacity(n);
            for omitted in 0..n {
                let mut deletion = MetaDeletion {
                    omitted: cohort.subjects()[omitted].observation.clone(),
                    fit: None,
                    effect_change: None,
                    change_in_full_se: None,
                };
                if full.status == MetaStatus::Estimated && rows.contains(&omitted) {
                    let kept = rows
                        .iter()
                        .copied()
                        .filter(|&i| i != omitted)
                        .collect::<Vec<_>>();
                    let yd = kept.iter().map(|&i| y[(i, j)]).collect::<Vec<_>>();
                    let vd = kept.iter().map(|&i| v[(i, j)]).collect::<Vec<_>>();
                    let fixed = match heterogeneity {
                        DeletionHeterogeneity::Refit => None,
                        DeletionHeterogeneity::FixedAtFullFit => Some(
                            full.diagnostics
                                .as_ref()
                                .ok_or(Error::Invalid("full fit diagnostics"))?
                                .tau_squared,
                        ),
                    };
                    fits_attempted += 1;
                    let (status, estimates, diagnostics) =
                        plan.fit_retained(&kept, &yd, &vd, fixed);
                    if let (Some(a), Some(b)) = (&full.estimates, &estimates) {
                        let delta = b
                            .effects
                            .iter()
                            .zip(&a.effects)
                            .map(|(b, a)| b - a)
                            .collect::<Vec<_>>();
                        deletion.change_in_full_se = a
                            .standard_errors
                            .as_ref()
                            .map(|se| delta.iter().zip(se).map(|(d, se)| d / se).collect());
                        deletion.effect_change = Some(delta);
                    }
                    deletion.fit = Some(MetaRegressionFeature {
                        feature: feature.clone(),
                        n: kept.len(),
                        rank: if estimates.is_some() { p } else { 0 },
                        status,
                        estimates,
                        diagnostics,
                    });
                }
                deletions.push(deletion);
            }
            features.push(MetaInfluenceFeature {
                feature,
                full,
                deletions,
            });
        }
        sink(MetaInfluenceBlock { start, features })?;
    }
    Ok(MetaInfluenceReceipt {
        design: design.clone(),
        outputs,
        options,
        heterogeneity,
        subjects: cohort.subjects().to_vec(),
        estimand: cohort.estimand().clone(),
        admission: cohort.admission().clone(),
        first_level: cohort.fit_receipts().to_vec(),
        variance_origins: cohort.variance_origins().to_vec(),
        fits_attempted,
        numeric_working_bytes: bytes,
    })
}
