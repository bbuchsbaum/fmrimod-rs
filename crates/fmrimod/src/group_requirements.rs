//! Translate a selected group provider into its first-level input contract.
//!
//! Group-output uncertainty is deliberately separate from the uncertainty that
//! must be retained by first-level fitting.  In particular, weighted
//! meta-regression consumes marginal first-level sampling variances even when
//! it is asked to return effects only.

use crate::outputs::{ConsumerRequirement, NamedLinearEstimand, OutputRequirementError};
use fmri_group::{FitOptions, MetaOutputSpec, MetaRegressionOptions};
use fmrireg::output::Uncertainty;

/// The first-level payload that a portable scalar cohort must retain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortableBundleRequirement {
    EffectsOnly,
    EffectsAndMarginalVariance,
}

/// A selected group-analysis provider and its actual execution options.
///
/// Group design and contrast binding remain execution concerns.  In the meta
/// variant, `outputs` retains the already bound group-output request; this
/// adapter concerns only the single scalar first-level estimand consumed by
/// either provider.
#[derive(Clone, Debug)]
pub enum GroupAnalysisSpec {
    Ols {
        options: FitOptions,
    },
    MetaRegression {
        options: MetaRegressionOptions,
        outputs: MetaOutputSpec,
    },
}

impl GroupAnalysisSpec {
    /// Whether the group provider consumes a genuine marginal first-level
    /// sampling variance for every retained scalar effect.  The scalar group
    /// provider does not consume first-level cross-estimand covariance, so this
    /// is marginal rather than joint uncertainty.
    pub fn requires_first_level_variance(&self) -> bool {
        matches!(self, Self::MetaRegression { .. })
    }

    /// The portable scalar-cohort payload implied by this provider.
    pub fn portable_bundle_requirement(&self) -> PortableBundleRequirement {
        if self.requires_first_level_variance() {
            PortableBundleRequirement::EffectsAndMarginalVariance
        } else {
            PortableBundleRequirement::EffectsOnly
        }
    }

    /// Create the first-level request for this group's one named scalar
    /// estimand.  The caller's identifiers bind it to a particular subject and
    /// resolved first-level model; `scalar_estimand` is retained unchanged.
    ///
    /// OLS consumes effects only. Weighted meta-regression consumes marginal
    /// sampling variance regardless of its requested *group-output*
    /// uncertainty, because its weights are defined by the first-level input.
    pub fn first_level_requirement(
        &self,
        consumer_id: impl Into<String>,
        subject_id: impl Into<String>,
        model_id: impl Into<String>,
        scalar_estimand: NamedLinearEstimand,
    ) -> Result<ConsumerRequirement, OutputRequirementError> {
        let uncertainty = if self.requires_first_level_variance() {
            Uncertainty::Marginal
        } else {
            Uncertainty::None
        };
        ConsumerRequirement::new(
            consumer_id,
            subject_id,
            model_id,
            vec![scalar_estimand],
            uncertainty,
        )
    }
}
