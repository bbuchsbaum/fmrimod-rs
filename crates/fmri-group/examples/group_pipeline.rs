//! Synthetic first-level contrasts -> admitted cohort -> group OLS -> whole-map BH.
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
    let design = GroupDesign::intercept(&cohort);
    let contrast = GroupContrast::named("population-mean", &design, &[("intercept", 1.)])?;
    let mut p = vec![];
    let receipt = fit_ols(
        &cohort,
        &design,
        &[contrast],
        FitOptions {
            block_features: 2,
            ..Default::default()
        },
        |block| {
            for row in block.features {
                if let Some(a) = &row.estimates[0] {
                    println!(
                        "{}: effect={:.4}, SE={:.4}, p={:.5}",
                        row.feature.as_str(),
                        a.effect,
                        a.standard_error,
                        a.p_value
                    );
                }
                p.push((row.feature, row.estimates[0].as_ref().map(|a| a.p_value)));
            }
            Ok(())
        },
    )?;
    let q = TestFamily::new("all-six-parcels", "population-mean", p)?
        .adjust(FdrMethod::BenjaminiHochberg);
    println!(
        "{} feature blocks; {} tests in BH family {}",
        receipt.blocks, q.tested, q.id
    );
    for (id, q) in q.values {
        println!("{}: q={q:?}", id.as_str());
    }
    Ok(())
}
