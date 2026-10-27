#![cfg(feature = "nifti")]
use fmrimod::{
    dataset::{NiftiSource, ObservationId},
    ingress::{load_subjects, LoadOptions, NiftiSubject},
};
use std::{fs, path::PathBuf};
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
struct Fixture(PathBuf);
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> (Fixture, Vec<NiftiSubject>) {
    let dir = std::env::temp_dir().join(format!("fmrimod-ingress-{}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let mut sources = Vec::new();
    for subject in 0..3 {
        let mut h = vec![0u8; 352];
        h[0..4].copy_from_slice(&348i32.to_le_bytes());
        for (i, d) in [4i16, 2, 1, 1, 8, 1, 1, 1].iter().enumerate() {
            h[40 + 2 * i..42 + 2 * i].copy_from_slice(&d.to_le_bytes());
        }
        h[70..72].copy_from_slice(&16i16.to_le_bytes());
        h[72..74].copy_from_slice(&32i16.to_le_bytes());
        for i in 0..8 {
            h[76 + 4 * i..80 + 4 * i].copy_from_slice(&1f32.to_le_bytes());
        }
        h[108..112].copy_from_slice(&352f32.to_le_bytes());
        h[112..116].copy_from_slice(&1f32.to_le_bytes());
        h[123] = 2;
        h[254..256].copy_from_slice(&1i16.to_le_bytes());
        for k in [280, 300, 320] {
            h[k..k + 4].copy_from_slice(&1f32.to_le_bytes());
        }
        h[344..348].copy_from_slice(b"n+1\0");
        for i in 0..16 {
            h.extend_from_slice(&((subject * 100 + i) as f32).to_le_bytes());
        }
        let path = dir.join(format!("s{subject}.nii"));
        fs::write(&path, h).unwrap();
        sources.push(
            NiftiSubject::new(
                NiftiSource::new(path, None, Some("grid".into())).unwrap(),
                (0..8)
                    .map(|i| ObservationId::new(format!("t{i}")).unwrap())
                    .collect(),
                format!("s{subject}"),
                "v1",
            )
            .unwrap(),
        );
    }
    (Fixture(dir), sources)
}
#[test]
fn bounded_loading_is_ordered_and_preflights_before_reads() {
    let _guard = TEST_LOCK.lock().unwrap();
    let (_fixture, sources) = fixture();
    for workers in [1, 2, 9] {
        let (loaded, receipt) = load_subjects(
            &sources,
            LoadOptions {
                workers,
                max_numeric_bytes: 10000,
            },
        )
        .unwrap();
        for (i, s) in loaded.iter().enumerate() {
            assert_eq!(s.subject_id(), format!("s{i}"));
            assert_eq!(s.subject().data()[(7, 1)], (i * 100 + 15) as f64);
        }
        let options = LoadOptions {
            workers,
            max_numeric_bytes: receipt.reserved_numeric_bytes,
        };
        assert!(load_subjects(&sources, options).is_ok());
        assert!(load_subjects(
            &sources,
            LoadOptions {
                max_numeric_bytes: options.max_numeric_bytes - 1,
                ..options
            }
        )
        .is_err());
    }
    let mut duplicate = sources.clone();
    duplicate[1] = duplicate[0].clone();
    assert!(load_subjects(
        &duplicate,
        LoadOptions {
            workers: 2,
            max_numeric_bytes: 10000
        }
    )
    .is_err());
    // Header remains intact: zero budget must fail before missing voxel values.
    let file = fs::OpenOptions::new()
        .write(true)
        .open(_fixture.0.join("s0.nii"))
        .unwrap();
    file.set_len(352).unwrap();
    assert!(matches!(
        load_subjects(
            &sources,
            LoadOptions {
                workers: 2,
                max_numeric_bytes: 0
            }
        ),
        Err(fmrimod::session::SessionError::Budget { .. })
    ));
    assert!(load_subjects(
        &sources,
        LoadOptions {
            workers: 2,
            max_numeric_bytes: 10000
        }
    )
    .is_err());
}

#[test]
fn file_batches_release_data_reuse_models_and_reduce_workers_to_budget() {
    use fmrimod::{
        cohort::CohortOptions,
        ingress::{run_file_cohort, FileSession},
        outputs::{DesignColumn, DesignColumns},
        regression::{
            output::{EstimandSet, OutputSpec, Uncertainty},
            MatrixBackend,
        },
        session::{ModelSpec, NoisePolicy},
    };
    // Different directory from the other test, which may execute concurrently.
    let _guard = TEST_LOCK.lock().unwrap();
    let (_fixture, sources) = fixture();
    let specs = sources
        .iter()
        .map(|s| ModelSpec {
            design: ndarray::Array2::from_shape_fn(
                (8, 2),
                |(i, j)| if j == 0 { 1. } else { i as f64 },
            ),
            observation_ids: s.observation_ids().to_vec(),
            columns: DesignColumns::new(vec![
                DesignColumn::new("i", "1", "intercept").unwrap(),
                DesignColumn::new("t", "1", "time").unwrap(),
            ])
            .unwrap(),
            noise: NoisePolicy::Ols,
            tolerance: 1e-12,
            backend: MatrixBackend::Portable,
        })
        .collect::<Vec<_>>();
    let request = OutputSpec::new(
        EstimandSet::contrasts(vec!["slope".into()], ndarray::array![[0., 1.]]).unwrap(),
    )
    .with_uncertainty(Uncertainty::Marginal);
    let requests = vec![request; 3];
    let mut sessions = sources
        .into_iter()
        .map(FileSession::new)
        .collect::<Vec<_>>();
    let options = CohortOptions {
        workers: 1,
        block_features: 1,
        max_numeric_bytes: 100000,
    };
    let mut seen = Vec::new();
    let first = run_file_cohort(&mut sessions, &specs, &requests, options, |i, out| {
        seen.push(i);
        for (_, b) in out.blocks() {
            assert!((b.estimates()[(0, 0)] - 2.).abs() < 1e-12);
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, vec![0, 1, 2]);
    assert_eq!(
        first.resident_numeric_bytes,
        sessions
            .iter()
            .map(|s| s.resident_numeric_bytes().unwrap())
            .sum::<usize>()
    );
    let second = run_file_cohort(&mut sessions, &specs, &requests, options, |_, _| Ok(())).unwrap();
    assert!(sessions
        .iter()
        .all(|s| s.preparation_counts().prepared == 1 && s.preparation_counts().reused == 1));
    assert!(run_file_cohort(
        &mut sessions,
        &specs,
        &requests,
        CohortOptions {
            max_numeric_bytes: second.reserved_numeric_bytes - 1,
            ..options
        },
        |_, _| panic!("must preflight")
    )
    .is_err());
    assert!(sessions.iter().all(|s| s.read_count() == 2));
    let bounded = run_file_cohort(
        &mut sessions,
        &specs,
        &requests,
        CohortOptions {
            workers: 3,
            max_numeric_bytes: second.reserved_numeric_bytes,
            ..options
        },
        |_, _| Ok(()),
    )
    .unwrap();
    assert_eq!(bounded.active_workers, 1);
    let mut changed = specs.clone();
    for s in &mut changed {
        s.design[(0, 1)] = 0.1;
    }
    run_file_cohort(&mut sessions, &changed, &requests, options, |_, _| Ok(())).unwrap();
    assert!(sessions
        .iter()
        .all(|s| s.preparation_counts().prepared == 2));
    assert!(
        run_file_cohort(&mut sessions, &changed, &requests, options, |_, _| Err(
            fmrimod::session::SessionError::Invalid("sink")
        ))
        .is_err()
    );
    run_file_cohort(&mut sessions, &changed, &requests, options, |_, _| Ok(())).unwrap();
}
