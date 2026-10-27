use crate::{nonblank, Error, Result};
use fmridataset::{
    Assay, AssayAnnotations, Axis, FeatureAxis, FmriFrame, Metadata, NumericBlock, ObservationId,
    SourceSpec, Values,
};
use std::collections::HashSet;
use std::sync::Arc;

/// Dataset-qualified participant identity; measurement never creates independence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Subject {
    pub observation: ObservationId,
    pub dataset: String,
    pub participant: String,
    pub measurement: String,
}
/// Scientific identity, not just a display label. Equivalence must be admitted
/// upstream; different definitions are never automatically joined by name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Estimand {
    pub id: String,
    pub definition: String,
    pub units: String,
}
#[derive(Clone, Debug, PartialEq)]
pub enum DegreesOfFreedom {
    Scalar(f64),
    ByFeature(Vec<f64>),
}
#[derive(Clone, Debug, PartialEq)]
pub enum VarianceOrigin {
    Known {
        method: String,
    },
    Estimated {
        df: DegreesOfFreedom,
        method: String,
        approximate: bool,
    },
    Unknown {
        reason: String,
    },
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FitReceipt {
    pub model: String,
    pub noise: String,
    pub selection: String,
}
/// Caller-supplied scientific admission; matching coordinates cannot establish
/// registration or participant independence on their own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    pub target: String,
    pub evidence: String,
}

#[derive(Clone, Debug)]
pub struct IndependentCohort {
    frame: FmriFrame,
    subjects: Vec<Subject>,
    estimand: Estimand,
    effect: String,
    variance: Option<String>,
    origins: Arc<[VarianceOrigin]>,
    fits: Vec<FitReceipt>,
    admission: Admission,
}
impl IndependentCohort {
    /// Assay constructors and frame binding enforce exact effect/variance axes.
    /// Nonfinite effects represent unavailable cells; zero effects remain valid.
    /// Positive finite variance is required by meta-analysis at each used cell.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frame: FmriFrame,
        subjects: Vec<Subject>,
        estimand: Estimand,
        effect: String,
        uncertainty: Option<(String, Vec<VarianceOrigin>)>,
        fits: Vec<FitReceipt>,
        admission: Admission,
    ) -> Result<Self> {
        let [n, f] = frame.shape();
        if n == 0
            || f == 0
            || subjects.len() != n
            || fits.len() != n
            || ![
                &estimand.id,
                &estimand.definition,
                &estimand.units,
                &admission.target,
                &admission.evidence,
            ]
            .iter()
            .all(|s| nonblank(s))
        {
            return Err(Error::Invalid(
                "empty cohort, identity, or receipt dimensions",
            ));
        }
        let mut participants = HashSet::new();
        for (i, s) in subjects.iter().enumerate() {
            if s.observation != frame.observations().ids()[i]
                || ![&s.dataset, &s.participant, &s.measurement]
                    .iter()
                    .all(|s| nonblank(s))
                || !participants.insert((&s.dataset, &s.participant))
            {
                return Err(Error::Invalid(
                    "subject order, identity, or repeated participant",
                ));
            }
        }
        if fits.iter().any(|r| {
            ![&r.model, &r.noise, &r.selection]
                .iter()
                .all(|s| nonblank(s))
        }) {
            return Err(Error::Invalid("incomplete fit receipt"));
        }
        frame.assay(&effect)?;
        let (variance, origins) = if let Some((name, origins)) = uncertainty {
            if name == effect || origins.len() != n {
                return Err(Error::Invalid("variance role or origins"));
            }
            frame.assay(&name)?;
            for origin in &origins {
                validate_origin(origin, f)?;
            }
            (Some(name), origins)
        } else {
            (None, Vec::new())
        };
        Ok(Self {
            frame,
            subjects,
            estimand,
            effect,
            variance,
            origins: origins.into(),
            fits,
            admission,
        })
    }
    pub fn frame(&self) -> &FmriFrame {
        &self.frame
    }
    pub fn subjects(&self) -> &[Subject] {
        &self.subjects
    }
    pub fn estimand(&self) -> &Estimand {
        &self.estimand
    }
    pub fn effect_assay(&self) -> &str {
        &self.effect
    }
    pub fn variance_assay(&self) -> Option<&str> {
        self.variance.as_deref()
    }
    pub fn variance_origins(&self) -> &[VarianceOrigin] {
        &self.origins
    }
    pub(crate) fn shared_origins(&self) -> Arc<[VarianceOrigin]> {
        self.origins.clone()
    }
    pub fn fit_receipts(&self) -> &[FitReceipt] {
        &self.fits
    }
    pub fn admission(&self) -> &Admission {
        &self.admission
    }

    /// Explicitly assemble scalar first-level t-contrast results. No F-statistic
    /// conversion, label-based equivalence or implicit spatial registration.
    pub fn from_first_level(rows: Vec<FirstLevelEstimate>, admission: Admission) -> Result<Self> {
        let first = rows
            .first()
            .ok_or(Error::Invalid("empty first-level collection"))?;
        let features = first.features.clone();
        let estimand = first.estimand.clone();
        let nf = features.axis().len();
        let n = rows.len();
        let mut effects = Vec::with_capacity(n * nf);
        let mut vars = Vec::with_capacity(n * nf);
        let mut subjects = Vec::with_capacity(n);
        let mut origins = Vec::with_capacity(n);
        let mut fits = Vec::with_capacity(n);
        for row in rows {
            features.space().assert_compatible(row.features.space())?;
            if row.estimand != estimand {
                return Err(Error::Invalid("different estimand definitions or units"));
            }
            subjects.push(row.subject);
            effects.extend(row.effect);
            vars.extend(row.variance);
            origins.push(row.origin);
            fits.push(row.fit);
        }
        let ids = subjects
            .iter()
            .map(|s| s.observation.clone())
            .collect::<Vec<_>>();
        let assays = [("effect", effects), ("variance", vars)]
            .into_iter()
            .map(|(name, values)| {
                Ok(Assay::new(
                    name,
                    SourceSpec::memory(NumericBlock::new([n, nf], Values::F64(values))?),
                    ids.clone(),
                    features.axis().ids().to_vec(),
                    AssayAnnotations::default(),
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let frame = FmriFrame::new(
            Axis::new(ids, Vec::new())?,
            features,
            assays,
            "effect",
            Metadata::new(),
        )?;
        Self::new(
            frame,
            subjects,
            estimand,
            "effect".into(),
            Some(("variance".into(), origins)),
            fits,
            admission,
        )
    }
}
fn validate_origin(origin: &VarianceOrigin, nf: usize) -> Result<()> {
    let valid = match origin {
        VarianceOrigin::Known { method } => nonblank(method),
        VarianceOrigin::Unknown { reason } => nonblank(reason),
        VarianceOrigin::Estimated { df, method, .. } => {
            nonblank(method)
                && match df {
                    DegreesOfFreedom::Scalar(d) => d.is_finite() && *d > 0.,
                    DegreesOfFreedom::ByFeature(d) => {
                        d.len() == nf && d.iter().all(|d| d.is_finite() && *d > 0.)
                    }
                }
        }
    };
    if valid {
        Ok(())
    } else {
        Err(Error::Invalid("variance origin or degrees of freedom"))
    }
}
#[derive(Clone, Debug)]
pub struct FirstLevelEstimate {
    subject: Subject,
    estimand: Estimand,
    features: FeatureAxis,
    effect: Vec<f64>,
    variance: Vec<f64>,
    origin: VarianceOrigin,
    fit: FitReceipt,
}
impl FirstLevelEstimate {
    /// The supplied receipt must identify the actual fitted noise/selection.
    /// `approximate` records whether first-level uncertainty is conditional on
    /// estimated nuisance/noise parameters. Marginal covariance only is retained.
    pub fn from_t(
        subject: Subject,
        estimand: Estimand,
        features: FeatureAxis,
        contrast: &fmrireg::TContrast,
        fit: FitReceipt,
        approximate: bool,
    ) -> Result<Self> {
        let n = features.axis().len();
        if contrast.effect.len() != n
            || contrast.standard_error.len() != n
            || contrast.df == 0
            || contrast.effect.iter().any(|x| !x.is_finite())
            || contrast
                .standard_error
                .iter()
                .any(|x| !x.is_finite() || *x <= 0. || !(x * x).is_finite() || x * x == 0.)
        {
            return Err(Error::Invalid(
                "invalid first-level marginal effect/uncertainty",
            ));
        }
        Ok(Self {
            subject,
            estimand,
            features,
            effect: contrast.effect.to_vec(),
            variance: contrast.standard_error.iter().map(|x| x * x).collect(),
            origin: VarianceOrigin::Estimated {
                df: DegreesOfFreedom::Scalar(contrast.df as f64),
                method: "first-level t contrast".into(),
                approximate,
            },
            fit,
        })
    }
}
