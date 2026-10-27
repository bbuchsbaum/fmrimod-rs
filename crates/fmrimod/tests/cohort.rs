use fmrimod::{
    cohort::{run_cohort, CohortOptions, ScalarCohortBuilder},
    dataset::{FeatureAxis, FeatureId, FeatureSpace, IndexSpace, ObservationId},
    group,
    outputs::{DesignColumn, DesignColumns},
    regression::{
        output::{EstimandSet, OutputSpec, Uncertainty},
        MatrixBackend,
    },
    session::{LoadedSubject, ModelSpec, NoisePolicy, SessionError, SubjectSession},
};
use ndarray::{array, Array2};
fn sessions() -> Vec<SubjectSession> {
    (0..4)
        .map(|s| {
            let rows = (0..8)
                .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
                .collect::<Vec<_>>();
            let features = FeatureAxis::from_space(FeatureSpace::Index(
                IndexSpace::new(
                    "same-space",
                    (0..17)
                        .map(|i| FeatureId::new(format!("v{i}")).unwrap())
                        .collect(),
                )
                .unwrap(),
            ))
            .unwrap();
            let x =
                Array2::from_shape_fn((8, 2), |(i, j)| if j == 0 || i % 2 == 0 { 1. } else { -1. });
            let y = Array2::from_shape_fn((8, 17), |(i, j)| {
                2. + x[(i, 1)] * ((s + 1) as f64 + j as f64 / 10.)
                    + if (i / 2) % 2 == 0 { 0.25 } else { -0.25 }
            });
            let spec = ModelSpec {
                design: x,
                observation_ids: rows.clone(),
                columns: DesignColumns::new(vec![
                    DesignColumn::new("intercept", "signal", "constant").unwrap(),
                    DesignColumn::new("task", "signal", "task").unwrap(),
                ])
                .unwrap(),
                noise: NoisePolicy::Ols,
                tolerance: 1e-12,
                backend: MatrixBackend::Portable,
            };
            let mut session = SubjectSession::new(
                LoadedSubject::from_f64(y, rows, features, format!("s{s}"), "synthetic-v1")
                    .unwrap(),
            );
            session.prepare(&spec, 1_000_000).unwrap();
            session
        })
        .collect()
}
fn requests(level: Uncertainty) -> Vec<OutputSpec> {
    (0..4)
        .map(|_| {
            OutputSpec::new(EstimandSet::contrasts(vec!["task".into()], array![[0., 1.]]).unwrap())
                .with_uncertainty(level)
        })
        .collect()
}
fn builder(sessions: &[SubjectSession], variance: bool) -> ScalarCohortBuilder {
    ScalarCohortBuilder::new(
        (0..4)
            .map(|s| group::Subject {
                observation: ObservationId::new(format!("s{s}")).unwrap(),
                dataset: "synthetic".into(),
                participant: format!("p{s}"),
                measurement: "run".into(),
            })
            .collect(),
        group::Estimand {
            id: "task".into(),
            definition: "task slope".into(),
            units: "signal".into(),
        },
        sessions[0].features().clone(),
        group::Admission {
            target: "same-space".into(),
            evidence: "analytic identical fixture".into(),
        },
        variance,
        1_000_000,
    )
    .unwrap()
}
#[test]
fn ordered_workers_and_exact_budget_match_analytic_group() {
    let sessions = sessions();
    let requests = requests(Uncertainty::Marginal);
    for workers in [1, 3, 100] {
        let mut collector = builder(&sessions, true);
        let mut seen = Vec::new();
        let options = CohortOptions {
            workers,
            block_features: 4,
            max_numeric_bytes: 10_000_000,
        };
        let receipt = run_cohort(&sessions, &requests, options, |i, s| {
            seen.push(i);
            collector.push(i, s).unwrap();
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![0, 1, 2, 3]);
        assert_eq!(receipt.active_workers, workers.min(4));
        let mut callbacks = 0;
        assert!(run_cohort(
            &sessions,
            &requests,
            CohortOptions {
                max_numeric_bytes: receipt.reserved_numeric_bytes - 1,
                ..options
            },
            |_, _| {
                callbacks += 1;
                Ok(())
            }
        )
        .is_err());
        assert_eq!(callbacks, 0);
        run_cohort(
            &sessions,
            &requests,
            CohortOptions {
                max_numeric_bytes: receipt.reserved_numeric_bytes,
                ..options
            },
            |_, _| Ok(()),
        )
        .unwrap();
        let cohort = collector.finish().unwrap();
        assert!(cohort.variance_origins().iter().all(|x| matches!(
            x,
            group::VarianceOrigin::Estimated {
                df: group::DegreesOfFreedom::Scalar(6.),
                approximate: false,
                ..
            }
        )));
        let design = group::GroupDesign::intercept(&cohort);
        let contrast = group::GroupContrast::named("mean", &design, &[("intercept", 1.)]).unwrap();
        let mut count = 0;
        group::fit_ols(
            &cohort,
            &design,
            &[contrast],
            group::FitOptions::default(),
            |block| {
                for (j, feature) in block.features.iter().enumerate() {
                    let result = feature.estimates[0].as_ref().unwrap();
                    assert!((result.effect - (2.5 + (block.start + j) as f64 / 10.)).abs() < 1e-11);
                    assert!((result.standard_error - (5f64 / 12.).sqrt()).abs() < 1e-11);
                    count += 1;
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(count, 17);
    }
}
#[test]
fn estimates_only_bridge_and_errors_keep_sessions_usable() {
    let sessions = sessions();
    let req = requests(Uncertainty::None);
    let options = CohortOptions {
        workers: 3,
        block_features: 4,
        max_numeric_bytes: 10_000_000,
    };
    let mut collector = builder(&sessions, false);
    run_cohort(&sessions, &req, options, |i, s| {
        collector.push(i, s).unwrap();
        Ok(())
    })
    .unwrap();
    assert!(collector.finish().unwrap().variance_assay().is_none());
    let mut invalid = builder(&sessions, true);
    assert!(invalid
        .push(0, &sessions[0].run_owned(&req[0], 4, 1_000_000).unwrap())
        .is_err());
    assert!(invalid.finish().is_err());
    let mut calls = 0;
    assert!(run_cohort(&sessions, &req, options, |_, _| {
        calls += 1;
        Err(SessionError::Invalid("sink failure"))
    })
    .is_err());
    assert_eq!(calls, 1);
    let mut recovered = 0;
    run_cohort(&sessions, &req, options, |_, _| {
        recovered += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(recovered, 4);
    let mut mismatch = builder(&sessions, false);
    assert!(mismatch
        .push(0, &sessions[1].run_owned(&req[1], 4, 1_000_000).unwrap())
        .is_err());
}
