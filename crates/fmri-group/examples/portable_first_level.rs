//! Actual first-level fits -> incremental bundle -> disk-backed group inference.
//! Synthetic demonstration, not a calibration experiment. Output must be new.
use fmri_data::{FeatureAxis, FeatureId, FeatureSpace, IndexSpace, ObservationId};
use fmri_group::*;
use ndarray::{array, Array2};

fn main() -> Result<()> {
    let directory = std::env::args()
        .nth(1)
        .ok_or(Error::Invalid("supply a new output directory"))?;
    let (subjects, features, timepoints, block) = (8, 37, 40, 7);
    let x = Array2::from_shape_fn((timepoints, 2), |(t, j)| {
        if j == 0 {
            1.
        } else {
            (t as f64 * 0.7).sin()
        }
    });
    let model = fmri_reg::LinearModel::new(x.view(), 1e-12)?;
    let metadata = PortableMetadata {
        subjects: (0..subjects)
            .map(|i| {
                Ok(Subject {
                    observation: ObservationId::new(format!("subject-{i}/visit-1"))?,
                    dataset: "synthetic-first-level".into(),
                    participant: format!("subject-{i}"),
                    measurement: "visit-1".into(),
                })
            })
            .collect::<Result<Vec<_>>>()?,
        estimand: Estimand {
            id: "task-control".into(),
            definition: "unit-amplitude task minus control".into(),
            units: "percent signal".into(),
        },
        features: FeatureAxis::from_space(FeatureSpace::Index(IndexSpace::new(
            "synthetic-parcels",
            (0..features)
                .map(|j| FeatureId::new(format!("parcel-{j}")))
                .collect::<fmri_data::Result<Vec<_>>>()?,
        )?))?,
        variance_origins: Some(vec![
            VarianceOrigin::Estimated {
                df: DegreesOfFreedom::Scalar((timepoints - 2) as f64),
                method: "first-level OLS t contrast".into(),
                approximate: false
            };
            subjects
        ]),
        fit_receipts: vec![
            FitReceipt {
                model: "intercept + synthetic task".into(),
                noise: "OLS iid residual model".into(),
                selection: "all 40 rows".into()
            };
            subjects
        ],
        admission: Admission {
            target: "synthetic-parcels".into(),
            evidence: "generated independent participant IDs and identical feature definitions"
                .into(),
        },
    };
    let mut writer = EstimateBundleWriter::create(
        &directory,
        metadata,
        PortableWriteOptions {
            max_block_bytes: subjects * block * 8,
        },
    )?;
    for start in (0..features).step_by(block) {
        let count = block.min(features - start);
        let mut effects = Array2::zeros((subjects, count));
        let mut variances = Array2::zeros((subjects, count));
        for subject in 0..subjects {
            // In a real pipeline this is a bounded read from this subject's source.
            let response = Array2::from_shape_fn((timepoints, count), |(t, j)| {
                let feature = start + j;
                1. + (0.2 + feature as f64 * 0.03 + (subject as f64 * 1.3).cos() * 0.3) * x[(t, 1)]
                    + ((t * 7 + subject * 13 + feature * 3) as f64).sin() * 0.2
            });
            let fitted = model.fit(response.view())?;
            let contrast = fitted.t_contrast(array![0., 1.].view())?;
            assert_eq!(contrast.df, timepoints - 2);
            for j in 0..count {
                effects[(subject, j)] = contrast.effect[j];
                variances[(subject, j)] = contrast.standard_error[j].powi(2);
            }
        }
        writer.write_effect_block(start, effects.view())?;
        writer.write_variance_block(start, variances.view())?;
    }
    writer.finish()?;
    let cohort = read_cohort_bundle(&directory, PortableReadOptions::default())?;
    let design = GroupDesign::intercept(&cohort);
    let outputs = MetaOutputSpec {
        contrasts: vec![GroupContrast::named(
            "population-mean",
            &design,
            &[("intercept", 1.)],
        )?],
        uncertainty: MetaUncertainty::Marginal,
        p_values: true,
        confidence_level: Some(0.95),
    };
    let mut plan = PreparedMetaRegression::new(
        &design,
        outputs,
        MetaRegressionOptions {
            heterogeneity: HeterogeneityEstimator::PauleMandel,
            inference: MetaInference::ModifiedKnappHartung,
            execution: FitOptions {
                block_features: block,
                ..Default::default()
            },
            ..Default::default()
        },
    )?;
    let mut count = 0;
    let receipt = plan.fit(&cohort, |b| {
        for row in b.features {
            if row.status != MetaStatus::Estimated {
                return Err(Error::Invalid("example feature failed"));
            }
            count += 1;
        }
        Ok(())
    })?;
    assert_eq!(count, features);
    println!("{subjects} subjects, {features} features; actual first-level estimates written in blocks of {block}; {} group blocks, {} weighted evaluations",receipt.run.blocks,receipt.weighted_evaluations);
    Ok(())
}
