//! Synthetic first-level contrasts -> admitted cohort -> PM meta-regression with modified Knapp–Hartung inference.
use fmri_data::{FeatureAxis, FeatureId, FeatureSpace, IndexSpace, ObservationId};
use fmri_group::*;
use ndarray::{array, Array2};
fn main() -> Result<()> {
    let x = Array2::from_shape_fn(
        (24, 2),
        |(t, j)| if j == 0 { 1. } else { (t as f64 * 0.7).sin() },
    );
    let model = fmri_reg::LinearModel::new(x.view(), 1e-12)?;
    let features = FeatureAxis::from_space(FeatureSpace::Index(IndexSpace::new(
        "synthetic-parcels",
        (0..6)
            .map(|j| FeatureId::new(format!("parcel-{j}")))
            .collect::<fmri_data::Result<Vec<_>>>()?,
    )?))?;
    let mut rows = vec![];
    for subject in 0..20 {
        let response = Array2::from_shape_fn((24, 6), |(t, j)| {
            1. + (0.2 + j as f64 * 0.05 + (subject as f64 * 1.3).cos() * 0.3) * x[(t, 1)]
                + ((t * 7 + subject * 13 + j * 3) as f64).sin() * 0.2
        });
        let fit = model.fit(response.view())?;
        let contrast = fit.t_contrast(array![0., 1.].view())?;
        rows.push(FirstLevelEstimate::from_t(
            Subject {
                observation: ObservationId::new(format!("subject-{subject}/visit-1"))?,
                dataset: "synthetic".into(),
                participant: format!("subject-{subject}"),
                measurement: "visit-1".into(),
            },
            Estimand {
                id: "task-control".into(),
                definition: "unit-amplitude task minus control".into(),
                units: "percent signal".into(),
            },
            features.clone(),
            &contrast,
            FitReceipt {
                model: "intercept + synthetic task".into(),
                noise: "OLS iid residual model".into(),
                selection: "all 24 rows".into(),
            },
            false,
        )?);
    }
    let cohort = IndependentCohort::from_first_level(
        rows,
        Admission {
            target: "synthetic-parcels".into(),
            evidence: "same generated definitions and independent participant IDs".into(),
        },
    )?;
    let design = GroupDesign::new(
        cohort.frame().observations().ids().to_vec(),
        vec!["intercept".into(), "age".into()],
        Array2::from_shape_fn(
            (20, 2),
            |(i, j)| if j == 0 { 1. } else { (i as f64 - 9.5) / 10. },
        ),
    )?;
    let outputs = MetaOutputSpec {
        contrasts: vec![
            GroupContrast::named("adjusted-mean", &design, &[("intercept", 1.)])?,
            GroupContrast::named("age-slope", &design, &[("age", 1.)])?,
        ],
        uncertainty: MetaUncertainty::Joint,
        p_values: true,
        confidence_level: Some(0.95),
    };
    let options = MetaRegressionOptions {
        heterogeneity: HeterogeneityEstimator::PauleMandel,
        inference: MetaInference::ModifiedKnappHartung,
        execution: FitOptions {
            block_features: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut plan = PreparedMetaRegression::new(&design, outputs, options)?;
    let receipt = plan.fit(&cohort, |block| {
        for row in block.features {
            if let (Some(estimates), Some(diagnostics)) = (row.estimates, row.diagnostics) {
                println!(
                    "{}: effects={:?}, 95% intervals={:?}, tau²={:.6}, evaluations={}",
                    row.feature.as_str(),
                    estimates.effects,
                    estimates.confidence_intervals.unwrap(),
                    diagnostics.tau_squared,
                    diagnostics.evaluations
                );
            } else {
                println!("{}: {:?}", row.feature.as_str(), row.status);
            }
        }
        Ok(())
    })?;
    println!(
        "{} blocks, {} weighted evaluations",
        receipt.run.blocks, receipt.weighted_evaluations
    );
    Ok(())
}
