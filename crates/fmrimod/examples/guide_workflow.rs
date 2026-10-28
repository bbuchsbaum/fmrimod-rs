//! Executable companion to docs/guide: events -> estimated AR GLS -> scalar group inference.
use fmrimod::{
    cohort::{run_cohort, CohortOptions, ScalarCohortBuilder},
    dataset::{FeatureAxis, FeatureId, FeatureSpace, IndexSpace, ObservationId},
    group::{
        self, Admission, Estimand, GroupContrast, GroupDesign, HeterogeneityEstimator,
        MetaInference, MetaOutputSpec, MetaRegressionOptions, MetaUncertainty,
    },
    hrf::{
        self,
        regressor::{ConvolutionMethod, EventTable},
    },
    noise::{ArStart, Pooling},
    outputs::{DesignColumn, DesignColumns},
    regression::{
        output::{EstimandSet, OutputSpec, Uncertainty},
        MatrixBackend,
    },
    session::{LoadedSubject, ModelSpec, NoisePolicy, SubjectSession},
};
use ndarray::{array, Array2};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ANCHOR: design
    let frame = hrf::SamplingFrame::uniform(180, 1.5, 0.05)?;
    let amplitudes = array![[1.], [1.], [1.], [1.], [1.], [1.]];
    let events = EventTable {
        onsets: &[5., 35., 70., 115., 160., 210.],
        durations: &[1.],
        amplitudes: amplitudes.view(),
        run_ids: &[0; 6],
    };
    let task = events.evaluate(
        &frame,
        &hrf::hrf::SpmG1Hrf::new(5., 15., 1. / 120.),
        ConvolutionMethod::Auto,
    )?;
    let x = Array2::from_shape_fn((180, 2), |(i, j)| if j == 0 { 1. } else { task[(i, 0)] });
    let rows = (0..180)
        .map(|i| ObservationId::new(format!("scan-{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let features = FeatureAxis::from_space(FeatureSpace::Index(IndexSpace::new(
        "synthetic-regions",
        (0..4)
            .map(|j| FeatureId::new(format!("region-{j}")))
            .collect::<Result<Vec<_>, _>>()?,
    )?))?;
    // ANCHOR_END: design
    let mut subjects = Vec::new();
    let mut rng = Rng(71173);
    for subject in 0..12 {
        let random_effect = 0.25 * rng.normal();
        let mut y = Array2::zeros((180, 4));
        for j in 0..4 {
            let mut noise = 0.;
            for i in 0..180 {
                noise = 0.4 * noise + 0.15 * rng.normal();
                y[(i, j)] = 10. + (0.7 + 0.1 * j as f64 + random_effect) * x[(i, 1)] + noise;
            }
        }
        // ANCHOR: prepare
        let spec = ModelSpec {
            design: x.clone(),
            observation_ids: rows.clone(),
            columns: DesignColumns::new(vec![
                DesignColumn::new("intercept", "signal", "constant")?,
                DesignColumn::new("task", "signal", "unit-amplitude event versus baseline")?,
            ])?,
            noise: NoisePolicy::EstimateAr {
                run_lengths: vec![180],
                noise_valid: None,
                order: 1,
                correction_max_lag: 12,
                pooling: Pooling::Run,
                startup: ArStart::ZeroHistory,
                feature_indices: vec![0, 1, 2, 3],
            },
            tolerance: 1e-12,
            backend: MatrixBackend::Portable,
        };
        let loaded = LoadedSubject::from_f64(
            y,
            rows.clone(),
            features.clone(),
            format!("s{subject}"),
            "guide-v1",
        )?;
        let mut session = SubjectSession::new(loaded);
        session.prepare(&spec, 8 << 20)?;
        subjects.push(session);
        // ANCHOR_END: prepare
    }
    // ANCHOR: outputs
    // ANCHOR: request
    let request = OutputSpec::new(EstimandSet::contrasts(
        vec!["task".into()],
        array![[0., 1.]],
    )?)
    .with_uncertainty(Uncertainty::Marginal);
    let requests = vec![request; subjects.len()];
    // ANCHOR_END: request
    // ANCHOR: collect
    let mut collector = ScalarCohortBuilder::new(
        (0..12)
            .map(|i| {
                Ok(group::Subject {
                    observation: ObservationId::new(format!("s{i}"))?,
                    dataset: "synthetic-guide".into(),
                    participant: format!("participant-{i}"),
                    measurement: "visit-1".into(),
                })
            })
            .collect::<fmrimod::dataset::Result<Vec<_>>>()?,
        Estimand {
            id: "task".into(),
            definition: "unit-amplitude event versus baseline; same HRF and coding".into(),
            units: "signal".into(),
        },
        features,
        Admission {
            target: "synthetic-regions".into(),
            evidence: "independent simulated participants; same feature order, units and estimand"
                .into(),
        },
        true,
        1 << 20,
    )?;
    // ANCHOR_END: collect
    // ANCHOR: execute
    let receipt = run_cohort(
        &subjects,
        &requests,
        CohortOptions {
            workers: 2,
            block_features: 2,
            max_numeric_bytes: 16 << 20,
        },
        |i, output| {
            assert!(output.receipt().estimated_noise);
            collector.push(i, output).map_err(|e| {
                eprintln!("{e}");
                fmrimod::session::SessionError::Invalid(
                    "subject output does not match admitted cohort",
                )
            })?;
            Ok(())
        },
    )?;
    let cohort = collector.finish()?;
    assert_eq!(receipt.subjects, 12);
    println!(
        "first-level: {} subjects, {} active workers",
        receipt.subjects, receipt.active_workers
    );
    // ANCHOR_END: execute
    // ANCHOR_END: outputs
    // ANCHOR: group
    let design = GroupDesign::intercept(&cohort);
    let contrast = GroupContrast::named("population-task", &design, &[("intercept", 1.)])?;
    let outputs = MetaOutputSpec {
        contrasts: vec![contrast],
        uncertainty: MetaUncertainty::Marginal,
        p_values: true,
        confidence_level: Some(0.95),
    };
    let mut tested = 0;
    group::fit_meta_regression(
        &cohort,
        &design,
        outputs,
        MetaRegressionOptions {
            heterogeneity: HeterogeneityEstimator::PauleMandel,
            inference: MetaInference::ModifiedKnappHartung,
            ..Default::default()
        },
        |block| {
            for feature in block.features {
                let estimates = feature
                    .estimates
                    .as_ref()
                    .ok_or(group::Error::Invalid("guide group fit failed"))?;
                let ci = estimates.confidence_intervals.as_ref().unwrap()[0];
                println!(
                    "{}: effect={:.4}, CI=[{:.4},{:.4}], p={:.5}",
                    feature.feature.as_str(),
                    estimates.effects[0],
                    ci[0],
                    ci[1],
                    estimates.p_values.as_ref().unwrap()[0]
                );
                assert!(estimates.effects[0].is_finite() && ci.iter().all(|v| v.is_finite()));
                tested += 1;
            }
            Ok(())
        },
    )?;
    assert_eq!(tested, 4);
    println!("PASS guide: 12 participants, 4 group estimates; conditional first-level uncertainty");
    // ANCHOR_END: group
    Ok(())
}
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1_u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}
