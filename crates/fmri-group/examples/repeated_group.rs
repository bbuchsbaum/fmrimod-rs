//! Three measurements per subject with explicitly identified sampling covariance.
use fmri_group::*;
use fmridataset::*;
use ndarray::Array2;
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let ids = (0..12)
        .map(|i| ObservationId::new(format!("row-{i}")))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let feature = FeatureId::new("region")?;
    let features = FeatureAxis::from_space(FeatureSpace::Index(IndexSpace::new(
        "synthetic",
        vec![feature.clone()],
    )?))?;
    let y = [-1., 0., 1., 2.].iter().flat_map(|v| vec![*v; 3]).collect();
    let assay = Assay::new(
        "effect",
        SourceSpec::memory(NumericBlock::new([12, 1], Values::F64(y))?),
        ids.clone(),
        vec![feature.clone()],
        AssayAnnotations::default(),
    )?;
    let frame = FmriFrame::new(
        Axis::new(ids.clone(), vec![])?,
        features,
        vec![assay],
        "effect",
        Metadata::new(),
    )?;
    let subjects = (0..12)
        .map(|i| RepeatedSubject {
            observation: ids[i].clone(),
            dataset: "example".into(),
            participant: format!("subject-{}", i / 3),
            measurement: format!("visit-{}", i % 3),
        })
        .collect();
    let covariance = RepeatedFeatureCovariance {
        feature,
        blocks: (0..4)
            .map(|i| RepeatedCovarianceBlock {
                observations: ids[i * 3..i * 3 + 3].to_vec(),
                values: Array2::eye(3) * 0.3,
            })
            .collect(),
    };
    let cohort = RepeatedCohort::new(
        frame,
        subjects,
        Estimand {
            id: "task-control".into(),
            definition: "same synthetic effect at each visit".into(),
            units: "a.u.".into(),
        },
        Admission {
            target: "one synthetic region".into(),
            evidence:
                "four independent participants with three explicitly correlated measurements each"
                    .into(),
        },
        "effect".into(),
        vec![covariance],
        CovarianceProvenance::Known {
            method: "synthetic sampling covariance".into(),
        },
    )?;
    let design = GroupDesign::new(ids, vec!["mean".into()], Array2::ones((12, 1)))?;
    let contrast = GroupContrast::named("mean", &design, &[("mean", 1.)])?;
    let plan = PreparedRepeatedModel::new(
        cohort,
        design,
        vec![contrast],
        RepeatedModel {
            residual_variance: ResidualVariance::Estimate,
            ..Default::default()
        },
    )?;
    let fit = plan.fit_feature(0)?;
    println!(
        "status={:?}, subjects={}, observations={}",
        fit.status, fit.n_subjects, fit.n_observations
    );
    if let Some(e) = fit.estimates {
        println!("mean={:.6}, conditional SE={:.6}, random-intercept variance={:.6}, extra variance={:.6}",e.beta[0],e.beta_covariance[(0,0)].sqrt(),e.g[(0,0)],e.eta);
    }
    Ok(())
}
