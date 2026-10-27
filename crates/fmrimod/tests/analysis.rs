#![cfg(feature = "nifti")]
use fmrimod::{
    analysis::*,
    dataset::{NiftiSource, ObservationId},
    ingress::NiftiSubject,
    noise::{ArStart, Pooling},
    outputs::{
        ConsumerRequirement, DesignColumn, DesignColumns, NamedLinearEstimand, NamedWeight,
        ScientificSemantics,
    },
    regression::{
        output::{Summary, Uncertainty},
        MatrixBackend,
    },
    session::{ModelSpec, NoisePolicy, SessionError, SubjectSession},
};
use ndarray::Array2;
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> (Fixture, Vec<NiftiSubject>) {
    let dir = std::env::temp_dir().join(format!(
        "analysis-fixture-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::SeqCst)
    ));
    fs::create_dir(&dir).unwrap();
    let mut sources = Vec::new();
    for subject in 0..4 {
        let mut h = vec![0u8; 352];
        h[0..4].copy_from_slice(&348i32.to_le_bytes());
        for (i, d) in [4i16, 8, 8, 8, 64, 1, 1, 1].iter().enumerate() {
            h[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        for (o, v) in [(70, 16i16), (72, 32), (254, 1)] {
            h[o..o + 2].copy_from_slice(&v.to_le_bytes());
        }
        for o in [76, 80, 84, 88, 92, 112, 280, 300, 320] {
            h[o..o + 4].copy_from_slice(&1f32.to_le_bytes());
        }
        h[108..112].copy_from_slice(&352f32.to_le_bytes());
        h[123] = 2;
        h[344..348].copy_from_slice(b"n+1\0");
        let mut state = vec![0.; 512];
        let mut rng = 17u64 + subject;
        for t in 0..64 {
            if t == 32 {
                state.fill(0.);
            }
            for (j, noise) in state.iter_mut().enumerate() {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let innovation = ((rng >> 32) as u32) as f64 / (u32::MAX as f64) - 0.5;
                *noise = 0.3 * *noise + innovation;
                let value = (1000.
                    + (subject as f64 + 1. + j as f64 / 512.) * (t as f64 * 0.2).sin()
                    + *noise) as f32;
                h.extend_from_slice(&value.to_le_bytes());
            }
        }
        let path = dir.join(format!("s{subject}.nii"));
        fs::write(&path, h).unwrap();
        sources.push(
            NiftiSubject::new(
                NiftiSource::new(
                    path,
                    Some((0..512).step_by(3).collect()),
                    Some("synthetic".into()),
                )
                .unwrap(),
                (0..64)
                    .map(|i| ObservationId::new(format!("t{i}")).unwrap())
                    .collect(),
                format!("s{subject}"),
                "generated-v1",
            )
            .unwrap(),
        );
    }
    (Fixture(dir), sources)
}
fn estimand(id: &str, column: &str, weight: f64) -> NamedLinearEstimand {
    NamedLinearEstimand::new(
        ScientificSemantics::new(
            id,
            id,
            "signal",
            "synthetic",
            "none",
            "shared across two runs",
            "v1",
        )
        .unwrap(),
        vec![NamedWeight::new(column, weight).unwrap()],
    )
    .unwrap()
}
fn requests(sources: &[NiftiSubject], scale: f64, estimated: bool) -> Vec<SubjectRequest> {
    sources
        .iter()
        .map(|s| SubjectRequest {
            model_id: "model-v1".into(),
            model: ModelSpec {
                design: Array2::from_shape_fn((64, 2), |(i, j)| {
                    if j == 0 {
                        1.
                    } else {
                        (i as f64 * 0.2).sin()
                    }
                }),
                observation_ids: s.observation_ids().to_vec(),
                columns: DesignColumns::new(vec![
                    DesignColumn::new("intercept", "signal", "constant").unwrap(),
                    DesignColumn::new("task", "signal", "task").unwrap(),
                ])
                .unwrap(),
                noise: if estimated {
                    NoisePolicy::EstimateAr {
                        run_lengths: vec![32, 32],
                        noise_valid: None,
                        order: 1,
                        correction_max_lag: 2,
                        pooling: Pooling::Global,
                        startup: ArStart::ZeroHistory,
                        feature_indices: vec![0, 40, 100],
                    }
                } else {
                    NoisePolicy::Ols
                },
                tolerance: 1e-12,
                backend: MatrixBackend::Portable,
            },
            consumers: vec![
                ConsumerRequirement::new(
                    "viewer",
                    s.subject_id(),
                    "model-v1",
                    vec![estimand("task", "task", scale)],
                    Uncertainty::None,
                )
                .unwrap(),
                ConsumerRequirement::new(
                    "group",
                    s.subject_id(),
                    "model-v1",
                    vec![
                        estimand("task", "task", scale),
                        estimand("baseline", "intercept", 1.),
                    ],
                    Uncertainty::Marginal,
                )
                .unwrap(),
            ],
        })
        .collect()
}
fn subjects(sources: &[NiftiSubject], requests: Vec<SubjectRequest>) -> Vec<AnalysisSubject> {
    sources
        .iter()
        .cloned()
        .zip(requests)
        .map(|(source, request)| AnalysisSubject { source, request })
        .collect()
}
fn policy() -> ExecutionPolicy {
    ExecutionPolicy {
        expected_use: ExpectedUse::ContrastExploration,
        retention: RetentionPolicy::CoefficientsAndScale,
        block_features: 31,
        ..ExecutionPolicy::new(10_000_000)
    }
}
#[test]
fn inspect_compiles_named_outputs_without_voxel_reads() {
    let (fixture, sources) = fixture();
    let req = requests(&sources, 1., false);
    for i in 0..4 {
        fs::remove_file(fixture.0.join(format!("s{i}.nii"))).unwrap();
    }
    let mut analysis = CohortAnalysis::new(subjects(&sources, req), policy()).unwrap();
    let plan = analysis.inspect().unwrap();
    assert_eq!(plan.storage, StorageMode::Resident);
    assert!(plan.subjects.iter().all(|s| s.data == DataAction::Load
        && s.model == ModelAction::Prepare
        && s.fit == FitAction::BuildRetainedFit
        && s.uncertainty == Uncertainty::Marginal));
    assert_eq!(
        analysis.compiled_outputs()[0].consumer_maps()[0].output_indices(),
        &[0]
    );
    assert_eq!(
        analysis.compiled_outputs()[0].consumer_maps()[1].output_indices(),
        &[0, 1]
    );
    assert!(analysis.run(|_, _, _| panic!("missing source")).is_err());
}
#[test]
fn resident_updates_reuse_exact_models_and_preserve_failed_updates() {
    let (_fixture, sources) = fixture();
    let req = requests(&sources, 1., false);
    let mut analysis = CohortAnalysis::new(subjects(&sources, req.clone()), policy()).unwrap();
    let mut baseline = Vec::new();
    let first = analysis
        .run(|_, out, _| {
            baseline.push(
                out.blocks()
                    .iter()
                    .flat_map(|(_, b)| b.estimates().row(0).to_vec())
                    .collect::<Vec<_>>(),
            );
            Ok(())
        })
        .unwrap();
    assert_eq!(
        (
            first.source_reads,
            first.model_preparations,
            first.retained_fit_builds
        ),
        (4, 4, 4)
    );
    let after = analysis.inspect().unwrap();
    assert!(after
        .subjects
        .iter()
        .all(|s| s.data == DataAction::ReuseResident
            && s.model == ModelAction::Reuse
            && s.fit == FitAction::UseRetainedFit));
    analysis.update(requests(&sources, 2., false)).unwrap();
    let second = analysis
        .run(|i, out, compiled| {
            assert_eq!(compiled.consumer_maps()[0].output_indices(), &[0]);
            for (range, b) in out.blocks() {
                for j in 0..range.len() {
                    assert!(
                        (b.estimates()[(0, j)] - 2. * baseline[i][range.start + j]).abs() < 1e-10
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(
        (
            second.source_reads,
            second.model_preparations,
            second.retained_fit_builds
        ),
        (0, 0, 0)
    );
    let mut bad = requests(&sources, 1., false);
    bad[0].model_id = "wrong".into();
    assert!(analysis.update(bad).is_err());
    assert!(analysis
        .run(|_, _, _| Err(SessionError::Invalid("sink")))
        .is_err());
    analysis.run(|_, _, _| Ok(())).unwrap();
    let mut changed = requests(&sources, 1., false);
    for r in &mut changed {
        r.model.design[(0, 1)] = 0.2;
    }
    analysis.update(changed).unwrap();
    assert!(analysis
        .inspect()
        .unwrap()
        .subjects
        .iter()
        .all(|s| s.model == ModelAction::Prepare && s.fit == FitAction::BuildRetainedFit));
    let third = analysis.run(|_, _, _| Ok(())).unwrap();
    assert_eq!(
        (
            third.source_reads,
            third.model_preparations,
            third.retained_fit_builds
        ),
        (0, 4, 4)
    );
}
#[test]
fn file_fallback_budgets_match_executor_and_low_level_outputs() {
    let (_fixture, sources) = fixture();
    let req = requests(&sources, 1., true);
    let one = ExecutionPolicy {
        workers: 1,
        block_features: 16,
        ..ExecutionPolicy::new(10_000_000)
    };
    let probe = CohortAnalysis::new(subjects(&sources, req.clone()), one).unwrap();
    let exact = probe.inspect().unwrap().reserved_numeric_bytes;
    let fallback = ExecutionPolicy {
        expected_use: ExpectedUse::ContrastExploration,
        workers: 4,
        max_numeric_bytes: exact,
        ..one
    };
    let mut analysis = CohortAnalysis::new(subjects(&sources, req.clone()), fallback).unwrap();
    let plan = analysis.inspect().unwrap();
    assert_eq!(plan.storage, StorageMode::FileBatches);
    assert_eq!(plan.workers, 1);
    assert!(plan
        .subjects
        .iter()
        .all(|s| s.model == ModelAction::EstimateNoise));
    assert!(CohortAnalysis::new(
        subjects(&sources, req.clone()),
        ExecutionPolicy {
            max_numeric_bytes: exact - 1,
            workers: 1,
            ..one
        }
    )
    .is_err());
    let compiled = analysis.compiled_outputs().to_vec();
    let mut expected = Vec::new();
    for (i, s) in sources.iter().enumerate() {
        let mut low = SubjectSession::new(s.load(usize::MAX).unwrap());
        low.prepare(&req[i].model, usize::MAX).unwrap();
        expected.push(low.run_owned(compiled[i].output(), 16, usize::MAX).unwrap());
    }
    let receipt = analysis
        .run(|i, out, _| {
            for ((_, a), (_, b)) in out.blocks().iter().zip(expected[i].blocks()) {
                assert_eq!(a.estimates(), b.estimates());
                if let (Summary::Marginal(a), Summary::Marginal(b)) = (a, b) {
                    assert_eq!(a.variance(0, 0).unwrap(), b.variance(0, 0).unwrap());
                } else {
                    panic!("missing variance");
                }
            }
            Ok(())
        })
        .unwrap();
    assert_eq!((receipt.source_reads, receipt.model_preparations), (4, 4));
    let again = analysis.run(|_, _, _| Ok(())).unwrap();
    assert_eq!(
        (
            again.source_reads,
            again.model_preparations,
            again.model_reuses
        ),
        (4, 0, 4)
    );
}
#[test]
fn policy_and_binding_errors_are_preflighted() {
    let (_fixture, sources) = fixture();
    let req = requests(&sources, 1., false);
    assert!(CohortAnalysis::new(
        subjects(&sources, req.clone()),
        ExecutionPolicy {
            storage: StoragePolicy::FileBatches,
            ..policy()
        }
    )
    .is_err());
    assert!(CohortAnalysis::new(
        subjects(&sources, req.clone()),
        ExecutionPolicy {
            external_reserved_bytes: 10_000_000,
            ..policy()
        }
    )
    .is_err());
    let mut bad = req.clone();
    bad[0].consumers[0] = ConsumerRequirement::new(
        "viewer",
        "s0",
        "model-v1",
        vec![estimand("missing", "missing", 1.)],
        Uncertainty::None,
    )
    .unwrap();
    assert!(CohortAnalysis::new(subjects(&sources, bad), policy()).is_err());
    let mut bad = req.clone();
    bad[0].model.observation_ids.swap(0, 1);
    assert!(CohortAnalysis::new(subjects(&sources, bad), policy()).is_err());
    let base = CohortAnalysis::new(subjects(&sources, req.clone()), policy())
        .unwrap()
        .inspect()
        .unwrap();
    let external = 12345;
    let with = CohortAnalysis::new(
        subjects(&sources, req),
        ExecutionPolicy {
            external_reserved_bytes: external,
            ..policy()
        },
    )
    .unwrap()
    .inspect()
    .unwrap();
    assert_eq!(
        with.reserved_numeric_bytes,
        base.reserved_numeric_bytes + external
    );
}
