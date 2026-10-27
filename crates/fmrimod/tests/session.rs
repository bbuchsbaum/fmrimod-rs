use fmrimod::noise::ArStart;
use fmrimod::{
    dataset::{Axis, FeatureAxis, FeatureId, IndexSpace, ObservationId},
    outputs::{DesignColumn, DesignColumns},
    regression::{
        output::{EstimandSet, OutputSpec},
        MatrixBackend,
    },
    session::{LoadedSubject, ModelSpec, NoisePolicy, PrepareStatus, SessionError, SubjectSession},
};
use ndarray::{array, s, ShapeBuilder};

fn subject() -> SubjectSession {
    let observations = (0..4)
        .map(|i| ObservationId::new(format!("t{i}")).unwrap())
        .collect::<Vec<_>>();
    let ids = (0..3)
        .map(|i| FeatureId::new(format!("v{i}")).unwrap())
        .collect::<Vec<_>>();
    let space = IndexSpace::new("test", ids.clone()).unwrap();
    let features = FeatureAxis::new(
        Axis::new(ids, vec![]).unwrap(),
        fmrimod::dataset::FeatureSpace::Index(space),
    )
    .unwrap();
    SubjectSession::new(
        LoadedSubject::from_f64(
            array![[1., 2., 3.], [2., 4., 5.], [3., 5., 8.], [4., 7., 9.]],
            observations,
            features,
            "s1",
            "source-1",
        )
        .unwrap(),
    )
}
fn spec(session: &SubjectSession) -> ModelSpec {
    ModelSpec {
        design: array![[1., 0.], [1., 1.], [1., 2.], [1., 3.]],
        observation_ids: session.subject().observation_ids().to_vec(),
        columns: DesignColumns::new(vec![
            DesignColumn::new("intercept", "1", "constant").unwrap(),
            DesignColumn::new("time", "1", "linear").unwrap(),
        ])
        .unwrap(),
        noise: NoisePolicy::Ols,
        tolerance: 1e-12,
        backend: MatrixBackend::Portable,
    }
}
fn output() -> OutputSpec {
    OutputSpec::new(EstimandSet::contrasts(vec!["time".into()], array![[0., 1.]]).unwrap())
}

#[test]
fn resident_session_reuses_models_and_matches_blocking() {
    let mut session = subject();
    let model = spec(&session);
    assert!(matches!(
        session.prepare(&model, usize::MAX).unwrap(),
        PrepareStatus::Prepared(_)
    ));
    assert!(matches!(
        session.prepare(&model, usize::MAX).unwrap(),
        PrepareStatus::Reused(_)
    ));
    assert_eq!(session.preparation_counts().prepared, 1);
    assert_eq!(session.preparation_counts().reused, 1);
    let all = session.run_owned(&output(), 3, usize::MAX).unwrap();
    let tiled = session.run_owned(&output(), 1, usize::MAX).unwrap();
    assert_eq!(session.read_count(), 0);
    let tiled_values = tiled
        .blocks()
        .iter()
        .flat_map(|(_, summary)| summary.estimates().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(
        all.blocks()[0]
            .1
            .estimates()
            .iter()
            .copied()
            .collect::<Vec<_>>(),
        tiled_values
    );
    assert_eq!(all.receipt().residual_df, 2);
}

#[test]
fn failed_budget_keeps_cached_model_usable() {
    let mut session = subject();
    let model = spec(&session);
    session.prepare(&model, usize::MAX).unwrap();
    let error = session
        .prepare(
            &ModelSpec {
                tolerance: 1e-8,
                ..model.clone()
            },
            0,
        )
        .unwrap_err();
    assert!(matches!(error, SessionError::Budget { .. }));
    assert_eq!(
        session
            .run_owned(&output(), 2, usize::MAX)
            .unwrap()
            .blocks()
            .len(),
        2
    );
}

#[test]
fn fixed_ar_settings_and_estimated_tolerance_participate_in_cache_identity() {
    let mut session = subject();
    let mut model = spec(&session);
    model.noise = NoisePolicy::FixedAr {
        run_lengths: vec![4],
        coefficients: vec![vec![0.2]],
        startup: ArStart::ZeroHistory,
        resets: vec![2],
    };
    session.prepare(&model, usize::MAX).unwrap();
    let mut changed = model.clone();
    changed.noise = NoisePolicy::FixedAr {
        run_lengths: vec![4],
        coefficients: vec![vec![0.2]],
        startup: ArStart::ZeroHistory,
        resets: vec![1],
    };
    assert!(matches!(
        session.prepare(&changed, usize::MAX).unwrap(),
        PrepareStatus::Prepared(_)
    ));
    changed.noise = NoisePolicy::FixedAr {
        run_lengths: vec![4],
        coefficients: vec![vec![0.2]],
        startup: ArStart::Stationary,
        resets: vec![1],
    };
    assert!(matches!(
        session.prepare(&changed, usize::MAX).unwrap(),
        PrepareStatus::Prepared(_)
    ));
    let mut estimate = spec(&session);
    estimate.noise = NoisePolicy::EstimateAr {
        run_lengths: vec![4],
        noise_valid: None,
        order: 1,
        correction_max_lag: 1,
        pooling: fmrimod::noise::Pooling::Global,
        startup: ArStart::ZeroHistory,
        feature_indices: vec![0],
    };
    estimate.tolerance = 1e-8;
    assert!(matches!(
        session.prepare(&estimate, usize::MAX),
        Err(SessionError::Invalid(
            "estimated AR requires tolerance exactly 1e-12"
        ))
    ));
}

#[test]
fn output_budget_and_sink_failure_do_not_execute_or_poison_cache() {
    let mut session = subject();
    session.prepare(&spec(&session), usize::MAX).unwrap();
    let output = output();
    let need = session.run_required_bytes(&output, 2, false).unwrap();
    let mut called = false;
    assert!(matches!(
        session.run_blocks(&output, 2, need - 1, |_, _| {
            called = true;
            Ok(())
        }),
        Err(SessionError::Budget { .. })
    ));
    assert!(!called);
    assert!(matches!(
        session.run_blocks(&output, 2, usize::MAX, |_, _| Err(SessionError::Invalid(
            "sink"
        ))),
        Err(SessionError::Invalid("sink"))
    ));
    assert_eq!(
        session
            .run_owned(&output, 2, usize::MAX)
            .unwrap()
            .blocks()
            .len(),
        2
    );
}

#[test]
fn from_f64_accepts_nonstandard_layout_without_reinterpreting_values() {
    let data = ndarray::Array2::from_shape_fn((4, 3).f(), |(r, c)| (10 * r + c) as f64);
    let observations = (0..4)
        .map(|i| ObservationId::new(format!("t{i}")).unwrap())
        .collect::<Vec<_>>();
    let ids = (0..3)
        .map(|i| FeatureId::new(format!("v{i}")).unwrap())
        .collect::<Vec<_>>();
    let features = FeatureAxis::new(
        Axis::new(ids.clone(), vec![]).unwrap(),
        fmrimod::dataset::FeatureSpace::Index(IndexSpace::new("test", ids).unwrap()),
    )
    .unwrap();
    let loaded = LoadedSubject::from_f64(data, observations, features, "s", "v").unwrap();
    assert_eq!(loaded.data()[(3, 2)], 32.);
}

#[test]
fn from_f64_compacts_a_sliced_owned_backing_array() {
    let data = ndarray::Array2::from_shape_fn((6, 3), |(r, c)| (10 * r + c) as f64)
        .slice_move(s![1..5, ..]);
    let observations = (0..4)
        .map(|i| ObservationId::new(format!("t{i}")).unwrap())
        .collect();
    let ids = (0..3)
        .map(|i| FeatureId::new(format!("v{i}")).unwrap())
        .collect::<Vec<_>>();
    let features = FeatureAxis::new(
        Axis::new(ids.clone(), vec![]).unwrap(),
        fmrimod::dataset::FeatureSpace::Index(IndexSpace::new("test", ids).unwrap()),
    )
    .unwrap();
    let loaded = LoadedSubject::from_f64(data, observations, features, "s", "v").unwrap();
    assert_eq!(loaded.data()[(0, 0)], 10.);
    assert_eq!(loaded.data_bytes(), 4 * 3 * 8);
}

#[test]
fn preparation_byte_math_reports_overflow_before_model_allocation() {
    let mut session = subject();
    let mut model = spec(&session);
    model.noise = NoisePolicy::EstimateAr {
        run_lengths: vec![4],
        noise_valid: None,
        order: 1,
        correction_max_lag: usize::MAX,
        pooling: fmrimod::noise::Pooling::Global,
        startup: ArStart::ZeroHistory,
        feature_indices: vec![0],
    };
    assert!(matches!(
        session.prepare(&model, usize::MAX),
        Err(SessionError::Overflow)
    ));
}

#[test]
fn estimated_ar_design_change_matches_fresh_session() {
    let make = || {
        let observations = (0..128)
            .map(|i| ObservationId::new(format!("t{i}")).unwrap())
            .collect::<Vec<_>>();
        let ids = (0..3)
            .map(|i| FeatureId::new(format!("v{i}")).unwrap())
            .collect::<Vec<_>>();
        let features = FeatureAxis::new(
            Axis::new(ids.clone(), vec![]).unwrap(),
            fmrimod::dataset::FeatureSpace::Index(IndexSpace::new("test", ids).unwrap()),
        )
        .unwrap();
        let mut data = ndarray::Array2::zeros((128, 3));
        let mut state = 0x9e37_79b9_u64;
        let mut previous = [0.; 3];
        for time in 0..128 {
            for feature in 0..3 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let innovation = ((state >> 11) as f64 / ((1_u64 << 53) as f64) - 0.5)
                    * (0.45 + feature as f64 * 0.07);
                previous[feature] = 0.4 * previous[feature] + innovation;
                data[(time, feature)] = 0.7 * (0.12 * time as f64).sin() + previous[feature];
            }
        }
        SubjectSession::new(
            LoadedSubject::from_f64(data, observations, features, "s", "v").unwrap(),
        )
    };
    let make_spec = |session: &SubjectSession, offset: f64| ModelSpec {
        design: ndarray::Array2::from_shape_fn((128, 2), |(r, c)| {
            if c == 0 {
                1.
            } else {
                (0.12 + offset) * r as f64
            }
            .sin()
        }),
        observation_ids: session.subject().observation_ids().to_vec(),
        columns: DesignColumns::new(vec![
            DesignColumn::new("i", "1", "c").unwrap(),
            DesignColumn::new("t", "1", "l").unwrap(),
        ])
        .unwrap(),
        noise: NoisePolicy::EstimateAr {
            run_lengths: vec![128],
            noise_valid: None,
            order: 1,
            correction_max_lag: 8,
            pooling: fmrimod::noise::Pooling::Global,
            startup: ArStart::ZeroHistory,
            feature_indices: vec![0, 1, 2],
        },
        tolerance: 1e-12,
        backend: MatrixBackend::Portable,
    };
    let mut reused = make();
    let first = make_spec(&reused, 0.);
    let first_output = reused.prepare(&first, usize::MAX).unwrap();
    let first_coefficients = first_output.receipt().noise_coefficients.clone().unwrap();
    let changed = make_spec(&reused, 0.05);
    let changed_output = reused.prepare(&changed, usize::MAX).unwrap();
    assert!(matches!(changed_output, PrepareStatus::Prepared(_)));
    let changed_coefficients = changed_output.receipt().noise_coefficients.clone().unwrap();
    let changed_values = reused
        .run_owned(
            &OutputSpec::new(EstimandSet::contrasts(vec!["t".into()], array![[0., 1.]]).unwrap()),
            3,
            usize::MAX,
        )
        .unwrap();
    let mut fresh = make();
    let fresh_spec = make_spec(&fresh, 0.05);
    let fresh_receipt = fresh.prepare(&fresh_spec, usize::MAX).unwrap();
    let fresh_values = fresh
        .run_owned(
            &OutputSpec::new(EstimandSet::contrasts(vec!["t".into()], array![[0., 1.]]).unwrap()),
            3,
            usize::MAX,
        )
        .unwrap();
    assert_eq!(
        changed_values.blocks()[0].1.estimates(),
        fresh_values.blocks()[0].1.estimates()
    );
    let fresh_coefficients = fresh_receipt.receipt().noise_coefficients.clone().unwrap();
    assert_eq!(changed_coefficients, fresh_coefficients);
    assert!(changed_coefficients
        .iter()
        .flatten()
        .zip(first_coefficients.iter().flatten())
        .any(|(new, old)| (new - old).abs() > 1e-4));
    let mut stale = make();
    let mut stale_spec = make_spec(&stale, 0.05);
    stale_spec.noise = NoisePolicy::FixedAr {
        run_lengths: vec![128],
        coefficients: first_coefficients,
        startup: ArStart::ZeroHistory,
        resets: vec![],
    };
    stale.prepare(&stale_spec, usize::MAX).unwrap();
    let stale_values = stale
        .run_owned(
            &OutputSpec::new(EstimandSet::contrasts(vec!["t".into()], array![[0., 1.]]).unwrap()),
            3,
            usize::MAX,
        )
        .unwrap();
    assert!(changed_values.blocks()[0]
        .1
        .estimates()
        .iter()
        .zip(stale_values.blocks()[0].1.estimates().iter())
        .any(|(current, old)| (current - old).abs() > 1e-6));
    let mut different_subset = make_spec(&reused, 0.05);
    if let NoisePolicy::EstimateAr {
        feature_indices, ..
    } = &mut different_subset.noise
    {
        *feature_indices = vec![0];
    }
    assert!(matches!(
        reused.prepare(&different_subset, usize::MAX).unwrap(),
        PrepareStatus::Prepared(_)
    ));
}

#[test]
fn model_spec_clone_does_not_retain_hidden_design_backing() {
    let mut model = spec(&subject());
    model.design = ndarray::Array2::ones((10_000, 2)).slice_move(s![5000..5004, ..]);
    assert_eq!(model.clone().design.into_raw_vec_and_offset().0.len(), 8);
}

#[test]
fn reader_load_moves_its_buffer_without_a_second_full_copy() {
    use fmrimod::dataset::{
        ArrayReader, DType, MemoryBudget, NumericBlock, ReadCost, ReadSelection, Values,
    };
    struct Once {
        block: Option<NumericBlock>,
        reads: usize,
    }
    impl ArrayReader for Once {
        fn shape(&self) -> [usize; 2] {
            [4, 3]
        }
        fn dtype(&self) -> DType {
            DType::F64
        }
        fn estimate(&self, _: &ReadSelection) -> fmrimod::dataset::Result<ReadCost> {
            Ok(ReadCost {
                output_bytes: 96,
                peak_bytes: 96,
            })
        }
        fn read(
            &mut self,
            _: &ReadSelection,
            budget: MemoryBudget,
        ) -> fmrimod::dataset::Result<NumericBlock> {
            budget.check(96)?;
            self.reads += 1;
            Ok(self.block.take().unwrap())
        }
    }
    let values = vec![1.; 12];
    let pointer = values.as_ptr();
    let mut reader = Once {
        block: Some(NumericBlock::new([4, 3], Values::F64(values)).unwrap()),
        reads: 0,
    };
    let source = subject();
    assert!(LoadedSubject::from_reader(
        &mut reader,
        source.subject().observation_ids().to_vec(),
        source.features().clone(),
        "s",
        "v",
        95
    )
    .is_err());
    assert_eq!(reader.reads, 0);
    let loaded = LoadedSubject::from_reader(
        &mut reader,
        source.subject().observation_ids().to_vec(),
        source.features().clone(),
        "s",
        "v",
        96,
    )
    .unwrap();
    assert_eq!(reader.reads, 1);
    assert_eq!(loaded.data().as_ptr(), pointer);
    assert_eq!(loaded.data_bytes(), 96);
}

#[test]
fn explicit_retention_preserves_outputs_budgets_and_invalidation() {
    use fmrimod::regression::output::{Summary, Uncertainty};
    for ar in [false, true] {
        let mut session = subject();
        let mut model = spec(&session);
        if ar {
            model.noise = NoisePolicy::FixedAr {
                run_lengths: vec![4],
                coefficients: vec![vec![0.2]],
                startup: ArStart::ZeroHistory,
                resets: vec![],
            };
        }
        session.prepare(&model, usize::MAX).unwrap();
        let request = output().with_uncertainty(Uncertainty::Joint);
        let baseline = session.run_owned(&request, 1, usize::MAX).unwrap();
        assert_eq!(session.retained_numeric_bytes(), 0);
        let required = session.retention_required_bytes(2).unwrap();
        assert!(session.retain_for_edits(2, required - 1).is_err());
        assert_eq!(session.retained_numeric_bytes(), 0);
        session.retain_for_edits(2, required).unwrap();
        assert_eq!(session.retained_numeric_bytes(), 3 * 3 * 8);
        session.retain_for_edits(1, 0).unwrap();
        let retained = session.run_owned(&request, 1, usize::MAX).unwrap();
        for ((_, a), (_, b)) in baseline.blocks().iter().zip(retained.blocks()) {
            assert!((a.estimates()[(0, 0)] - b.estimates()[(0, 0)]).abs() < 1e-12);
            if let (Summary::Joint(a), Summary::Joint(b)) = (a, b) {
                assert!((a.variance(0, 0).unwrap() - b.variance(0, 0).unwrap()).abs() < 1e-12);
            } else {
                panic!("lost joint output");
            }
        }
        let bound = session.run_required_bytes(&request, 1, true).unwrap();
        assert!(session.run_owned(&request, 1, bound - 1).is_err());
        session.run_owned(&request, 1, bound).unwrap();
        session.prepare(&model, usize::MAX).unwrap();
        assert!(session.retained_numeric_bytes() > 0);
        model.design[(0, 1)] = 0.3;
        assert!(session.prepare(&model, 0).is_err());
        assert!(session.retained_numeric_bytes() > 0);
        session.prepare(&model, usize::MAX).unwrap();
        assert_eq!(session.retained_numeric_bytes(), 0);
        session.retain_for_edits(2, usize::MAX).unwrap();
        session.clear_retained();
        assert_eq!(session.retained_numeric_bytes(), 0);
    }
}

#[test]
fn detached_state_checks_source_identity_and_preserves_retained_model() {
    let mut session = subject();
    let model = spec(&session);
    session.prepare(&model, usize::MAX).unwrap();
    session.retain_for_edits(2, usize::MAX).unwrap();
    let cache_bytes = session.retained_numeric_bytes();
    let state = session.detach();
    let fresh = subject();
    let data = LoadedSubject::from_f64(
        fresh.subject().data().to_owned(),
        fresh.subject().observation_ids().to_vec(),
        fresh.features().clone(),
        fresh.subject_id(),
        "source-1",
    )
    .unwrap();
    let attached = state.attach(data).unwrap();
    assert_eq!(attached.retained_numeric_bytes(), cache_bytes);
    attached.run_owned(&output(), 1, usize::MAX).unwrap();
    let state = attached.detach();
    let wrong = LoadedSubject::from_f64(
        fresh.subject().data().to_owned(),
        fresh.subject().observation_ids().to_vec(),
        fresh.features().clone(),
        fresh.subject_id(),
        "changed-source",
    )
    .unwrap();
    assert!(state.attach(wrong).is_err());
}

#[test]
fn detached_outputs_match_analytic_slopes_and_variances_without_data() {
    use fmrimod::regression::output::{Summary, Uncertainty};
    let mut session = subject();
    session.prepare(&spec(&session), usize::MAX).unwrap();
    session.retain_for_edits(2, usize::MAX).unwrap();
    let before = session.resident_numeric_bytes().unwrap();
    let data = session.data_bytes();
    let detached = session.detach();
    assert_eq!(detached.resident_numeric_bytes().unwrap(), before - data);
    let request = output().with_uncertainty(Uncertainty::Marginal);
    let budget = detached.run_required_bytes(&request, 1, true).unwrap();
    assert!(detached.run_owned(&request, 1, budget - 1).is_err());
    let out = detached.run_owned(&request, 1, budget).unwrap();
    // Ordinary straight-line least squares: Sxx=5, SSE=(0,.2,.7), df=2.
    for ((range, summary), (effect, variance)) in
        out.blocks()
            .iter()
            .zip([(1., 0.), (1.6, 0.02), (2.1, 0.07)])
    {
        assert_eq!(range.len(), 1);
        assert!((summary.estimates()[(0, 0)] - effect).abs() < 1e-12);
        if let Summary::Marginal(m) = summary {
            assert!((m.variance(0, 0).unwrap() - variance).abs() < 1e-12);
        } else {
            panic!("missing marginal uncertainty");
        }
    }
    let mut empty = subject();
    empty.prepare(&spec(&empty), usize::MAX).unwrap();
    assert!(empty.detach().run_owned(&request, 1, usize::MAX).is_err());
}
