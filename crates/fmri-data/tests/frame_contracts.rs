use fmri_data::*;
use proptest::prelude::*;

fn obs(ids: &[&str]) -> Vec<ObservationId> {
    ids.iter()
        .map(|s| ObservationId::new(*s).unwrap())
        .collect()
}
fn feat(ids: &[&str]) -> Vec<FeatureId> {
    ids.iter().map(|s| FeatureId::new(*s).unwrap()).collect()
}
fn identity() -> [[f64; 4]; 4] {
    [
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
    ]
}
fn context(bytes: usize) -> ExecutionContext {
    ExecutionContext::new(MemoryBudget::new(bytes))
}
fn toy() -> FmriFrame {
    // R matrix(seq_len(24), nrow=6): explicit row-major conversion.
    let values = (0..6)
        .flat_map(|row| (0..4).map(move |col| (1 + row + 6 * col) as f64))
        .collect();
    let observations = Axis::new(
        obs(&[
            "volume-1", "volume-2", "volume-3", "volume-4", "volume-5", "volume-6",
        ]),
        vec![
            (
                "run".into(),
                Column::Categorical {
                    codes: vec![Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)],
                    levels: vec!["run-1".into(), "run-2".into(), "unused".into()],
                    ordered: true,
                },
            ),
            (
                "censored".into(),
                Column::Bool(vec![
                    Some(false),
                    None,
                    Some(true),
                    Some(false),
                    Some(false),
                    Some(true),
                ]),
            ),
        ],
    )
    .unwrap();
    let features = FeatureAxis::from_space(FeatureSpace::Volume(
        VolumeSpace::new([2, 2, 1], identity(), None, Some("toy".into())).unwrap(),
    ))
    .unwrap();
    FmriFrame::from_source(
        "signal",
        SourceSpec::memory(NumericBlock::new([6, 4], Values::F64(values)).unwrap()),
        observations,
        features,
    )
    .unwrap()
}

#[test]
fn upstream_readme_values_ids_and_space_are_aligned() {
    let f = toy();
    let v = f
        .view_ids(
            &obs(&["volume-6", "volume-1"]),
            &feat(&["voxel-4", "voxel-2"]),
        )
        .unwrap();
    assert_eq!(v.shape(), [2, 2]);
    assert_eq!(v.assay("signal").unwrap().source().shape(), v.shape());
    assert_eq!(
        v.collect("signal", &context(32)).unwrap().values(),
        &Values::F64(vec![24., 12., 19., 7.])
    );
    assert_eq!(v.observations().ids(), obs(&["volume-6", "volume-1"]));
    assert_eq!(v.features().axis().ids(), feat(&["voxel-4", "voxel-2"]));
    assert_eq!(
        v.observations().columns()[0].1,
        Column::Categorical {
            codes: vec![Some(1), Some(0)],
            levels: vec!["run-1".into(), "run-2".into(), "unused".into()],
            ordered: true
        }
    );
    let FeatureSpace::Volume(space) = v.features().space() else {
        panic!("expected volume")
    };
    assert_eq!(space.support(), &[3, 1]);
    assert_eq!(
        space
            .reconstruct(&[24., 12.], -1., MemoryBudget::new(32))
            .unwrap(),
        vec![-1., 12., -1., 24.]
    );
    assert!(space
        .reconstruct(&[24., 12.], 0., MemoryBudget::new(31))
        .is_err());
    assert_eq!(f.shape(), [6, 4]);
}

#[test]
fn nested_views_equal_composed_view_and_serialize_visible_axes() {
    let f = toy();
    let a = f
        .view_ids(
            &obs(&["volume-6", "volume-2", "volume-1"]),
            &feat(&["voxel-4", "voxel-1", "voxel-2"]),
        )
        .unwrap();
    let b = a
        .view_ids(
            &obs(&["volume-1", "volume-6"]),
            &feat(&["voxel-2", "voxel-4"]),
        )
        .unwrap();
    let direct = f
        .view_ids(
            &obs(&["volume-1", "volume-6"]),
            &feat(&["voxel-2", "voxel-4"]),
        )
        .unwrap();
    assert_eq!(
        b.collect("signal", &context(32)).unwrap(),
        direct.collect("signal", &context(32)).unwrap()
    );
    assert_eq!(b.features().space(), direct.features().space());
    assert_eq!(
        serde_json::to_value(&b).unwrap(),
        serde_json::to_value(&direct).unwrap()
    );
}

#[test]
fn duplicate_source_gathers_are_valid_but_frame_duplicates_are_not() {
    let f = toy();
    let source = f.assay("signal").unwrap().source();
    let selected = ReadSelection::new(
        source.shape(),
        Selection::Indices(vec![5, 5, 0]),
        Selection::Indices(vec![3, 1, 3]),
    )
    .unwrap();
    assert_eq!(
        context(72)
            .read(&mut context(72).open(source), &selected)
            .unwrap()
            .values(),
        &Values::F64(vec![24., 12., 24., 24., 12., 24., 19., 7., 19.])
    );
    assert!(AxisSelection::new(Selection::Indices(vec![0, 0]), 6).is_err());
    assert!(f
        .view_ids(&obs(&["volume-1", "volume-1"]), &feat(&["voxel-1"]))
        .is_err());
    assert!(f.view_ids(&obs(&["missing"]), &feat(&["voxel-1"])).is_err());
    assert!(ReadSelection::new([6, 4], Selection::Indices(vec![6]), Selection::All).is_err());
    assert!(ReadSelection::new([6, 4], Selection::Range(4..7), Selection::All).is_err());
}

#[test]
fn empty_axes_preserve_shape_and_dtype_without_budget() {
    let f = toy();
    for (rows, cols, shape) in [
        (vec![], vec![0, 2], [0, 2]),
        (vec![0, 1], vec![], [2, 0]),
        (vec![], vec![], [0, 0]),
    ] {
        let v = f
            .view(
                &AxisSelection::new(Selection::Indices(rows), 6).unwrap(),
                &AxisSelection::new(Selection::Indices(cols), 4).unwrap(),
            )
            .unwrap();
        assert_eq!(v.shape(), shape);
        let block = v.collect("signal", &context(0)).unwrap();
        assert_eq!(block.shape(), shape);
        assert_eq!(block.dtype(), DType::F64);
        assert!(block.values().is_empty());
    }
    // Valid source-only shape: zero columns must not iterate usize::MAX rows.
    let source =
        SourceSpec::memory(NumericBlock::new([usize::MAX, 0], Values::F32(vec![])).unwrap());
    assert_eq!(
        context(0).collect(&source).unwrap().shape(),
        [usize::MAX, 0]
    );
}

#[test]
fn constructors_reject_ambiguous_alignment_and_bad_annotations() {
    assert!(ObservationId::new("").is_err());
    assert!(Axis::new(obs(&["a", "a"]), vec![]).is_err());
    assert!(Axis::new(obs(&["a"]), vec![("x".into(), Column::Int(vec![]))]).is_err());
    assert!(Axis::new(
        obs(&["a"]),
        vec![
            ("x".into(), Column::Int(vec![None])),
            ("x".into(), Column::Int(vec![None]))
        ]
    )
    .is_err());
    assert!(Axis::new(
        obs(&["a"]),
        vec![(
            "x".into(),
            Column::Categorical {
                codes: vec![Some(1)],
                levels: vec!["one".into()],
                ordered: false
            }
        )]
    )
    .is_err());
    assert!(NumericBlock::new([2, 3], Values::F64(vec![0.; 5])).is_err());
    assert!(matches!(
        NumericBlock::new([usize::MAX, 2], Values::F64(vec![])),
        Err(FmriDatasetError::SizeOverflow)
    ));
    let f = toy();
    let mut ids = f.observations().ids().to_vec();
    ids.reverse();
    let wrong = Assay::new(
        "signal",
        f.assay("signal").unwrap().source().clone(),
        ids,
        f.features().axis().ids().to_vec(),
        AssayAnnotations::default(),
    )
    .unwrap();
    assert!(FmriFrame::new(
        f.observations().clone(),
        f.features().clone(),
        vec![wrong],
        "signal",
        Metadata::new()
    )
    .is_err());
    assert!(FmriFrame::new(
        f.observations().clone(),
        f.features().clone(),
        vec![f.assays()[0].clone(), f.assays()[0].clone()],
        "signal",
        Metadata::new()
    )
    .is_err());
    assert!(FmriFrame::new(
        f.observations().clone(),
        f.features().clone(),
        f.assays().to_vec(),
        "missing",
        Metadata::new()
    )
    .is_err());
}

#[test]
fn equal_shapes_and_ids_do_not_imply_equal_spaces() {
    let a = FeatureSpace::Volume(
        VolumeSpace::new([2, 2, 1], identity(), None, Some("a".into())).unwrap(),
    );
    let b = FeatureSpace::Volume(
        VolumeSpace::new([2, 2, 1], identity(), None, Some("b".into())).unwrap(),
    );
    assert_eq!(a.ids(), b.ids());
    assert!(!a.compatible_with(&b));
    let mut affine = identity();
    affine[0][3] = 2.;
    let c =
        FeatureSpace::Volume(VolumeSpace::new([2, 2, 1], affine, None, Some("a".into())).unwrap());
    assert!(!a.compatible_with(&c));
    let d = FeatureSpace::Index(IndexSpace::new("atlas-a", feat(&["x"])).unwrap());
    let e = FeatureSpace::Index(IndexSpace::new("atlas-b", feat(&["x"])).unwrap());
    assert!(!d.compatible_with(&e));
    assert!(FeatureAxis::new(
        Axis::new(feat(&["voxel-2", "voxel-1", "voxel-3", "voxel-4"]), vec![]).unwrap(),
        a
    )
    .is_err());
    assert!(VolumeSpace::new([2, 2, 1], identity(), Some(vec![0, 0]), None).is_err());
    assert!(VolumeSpace::new([2, 2, 1], identity(), Some(vec![4]), None).is_err());
    assert!(VolumeSpace::new([0, 2, 1], identity(), None, None).is_err());
    assert!(VolumeSpace::new([2, 2, 1], [[0.; 4]; 4], None, None).is_err());
}

#[test]
fn axis_blocks_and_assay_annotations_survive_views() {
    let f = toy();
    let components = Axis::new(vec![ComponentId::new("motion-x").unwrap()], vec![]).unwrap();
    let block = AxisBlock::new(
        SourceSpec::memory(
            NumericBlock::new([6, 1], Values::F32(vec![1., 2., 3., 4., 5., 6.])).unwrap(),
        ),
        components,
        "confound",
        Some("mm".into()),
        Metadata::new(),
    )
    .unwrap();
    let observations = f
        .observations()
        .clone()
        .with_blocks(vec![("motion".into(), block)])
        .unwrap();
    let annotations = AssayAnnotations {
        role: Some("signal".into()),
        units: Some("percent".into()),
        metadata: Metadata::from([("origin".into(), serde_json::json!("fixture"))]),
    };
    let assay = Assay::new(
        "signal",
        f.assays()[0].source().clone(),
        observations.ids().to_vec(),
        f.features().axis().ids().to_vec(),
        annotations,
    )
    .unwrap();
    let frame = FmriFrame::new(
        observations,
        f.features().clone(),
        vec![assay],
        "signal",
        Metadata::new(),
    )
    .unwrap();
    let v = frame
        .view_ids(&obs(&["volume-6", "volume-1"]), &feat(&["voxel-2"]))
        .unwrap();
    assert_eq!(
        v.assays()[0].annotations().units.as_deref(),
        Some("percent")
    );
    let block = &v.observations().blocks()[0].1;
    assert_eq!(block.source().shape(), [2, 1]);
    assert_eq!(
        context(8).collect(block.source()).unwrap().values(),
        &Values::F32(vec![6., 1.])
    );
    assert_eq!(block.units(), Some("mm"));
}

#[test]
fn budget_is_checked_before_reader_invocation_and_f32_is_preserved() {
    struct Spy {
        reads: usize,
    }
    impl ArrayReader for Spy {
        fn shape(&self) -> [usize; 2] {
            [2, 2]
        }
        fn dtype(&self) -> DType {
            DType::F32
        }
        fn estimate(&self, _: &ReadSelection) -> Result<ReadCost> {
            Ok(ReadCost {
                output_bytes: 16,
                peak_bytes: 32,
            })
        }
        fn read(&mut self, _: &ReadSelection, _: MemoryBudget) -> Result<NumericBlock> {
            self.reads += 1;
            NumericBlock::new([2, 2], Values::F32(vec![0.; 4]))
        }
    }
    let mut spy = Spy { reads: 0 };
    assert!(matches!(
        context(31).read(&mut spy, &ReadSelection::all([2, 2]).unwrap()),
        Err(FmriDatasetError::BudgetExceeded { required: 32, .. })
    ));
    assert_eq!(spy.reads, 0);
    let source =
        SourceSpec::memory(NumericBlock::new([2, 2], Values::F32(vec![1., 2., 3., 4.])).unwrap());
    assert_eq!(context(16).collect(&source).unwrap().dtype(), DType::F32);
    assert!(context(15).collect(&source).is_err());
    let mut reader = context(16).open(&source);
    assert!(reader
        .read(&ReadSelection::all([2, 2]).unwrap(), MemoryBudget::new(15))
        .is_err());
    assert!(reader
        .read(&ReadSelection::all([1, 4]).unwrap(), MemoryBudget::new(16))
        .is_err());
}

proptest! {
    #[test]
    fn arbitrary_source_gathers_match_independent_coordinate_oracle(
        nr in 1usize..12, nc in 1usize..12,
        rows in prop::collection::vec(0usize..100,0..20),
        cols in prop::collection::vec(0usize..100,0..20)
    ) {
        let rows:Vec<_> = rows.into_iter().map(|r|r%nr).collect();
        let cols:Vec<_> = cols.into_iter().map(|c|c%nc).collect();
        let values = (0..nr).flat_map(|r|(0..nc).map(move |c| (1000*r+c) as f64)).collect();
        let source = SourceSpec::memory(NumericBlock::new([nr,nc],Values::F64(values)).unwrap());
        let sel = ReadSelection::new([nr,nc],Selection::Indices(rows.clone()),Selection::Indices(cols.clone())).unwrap();
        let expected:Vec<_> = rows.iter().flat_map(|&r| cols.iter().map(move |&c|(1000*r+c) as f64)).collect();
        let block = context(expected.len()*8).read(&mut context(0).open(&source),&sel).unwrap();
        prop_assert_eq!(block.shape(),[rows.len(),cols.len()]);
        prop_assert_eq!(block.values(),&Values::F64(expected));
    }
}

#[test]
fn diagnostic_serialization_preserves_nonfinite_and_missing_float_bits() {
    let nan = f64::from_bits(0x7ff8_0000_0000_1234);
    let values = Values::F64(vec![nan, f64::INFINITY, -0.0]);
    let encoded = serde_json::to_value(&values).unwrap();
    assert_eq!(
        encoded["f64_bits"],
        serde_json::json!([nan.to_bits(), f64::INFINITY.to_bits(), (-0.0f64).to_bits()])
    );
    let column = Column::Float(vec![None, Some(nan), Some(-0.0)]);
    let encoded = serde_json::to_value(&column).unwrap();
    assert_eq!(
        encoded["float64_bits"],
        serde_json::json!([null, nan.to_bits(), (-0.0f64).to_bits()])
    );
    let n32 = f32::from_bits(0x7fc0_0123);
    assert_eq!(
        serde_json::to_value(Values::F32(vec![n32])).unwrap()["f32_bits"],
        serde_json::json!([n32.to_bits()])
    );
}

#[test]
fn malformed_reader_estimates_are_rejected_without_reading() {
    struct BadEstimate;
    impl ArrayReader for BadEstimate {
        fn shape(&self) -> [usize; 2] {
            [1, 1]
        }
        fn dtype(&self) -> DType {
            DType::F64
        }
        fn estimate(&self, _: &ReadSelection) -> Result<ReadCost> {
            Ok(ReadCost {
                output_bytes: 0,
                peak_bytes: 0,
            })
        }
        fn read(&mut self, _: &ReadSelection, _: MemoryBudget) -> Result<NumericBlock> {
            panic!("read must not be invoked")
        }
    }
    assert!(matches!(
        context(0).read(&mut BadEstimate, &ReadSelection::all([1, 1]).unwrap()),
        Err(FmriDatasetError::Invalid {
            field: "reader estimate",
            ..
        })
    ));
}

#[test]
fn sampling_utility_rejects_invalid_finite_domains_and_overflow() {
    use fmri_data::sampling::SamplingFrame;
    for tr in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(SamplingFrame::uniform(2, tr, 0.1).is_err());
    }
    assert!(SamplingFrame::uniform(2, 1., f64::NAN).is_err());
    assert!(SamplingFrame::builder()
        .run_lengths(vec![usize::MAX, 1])
        .tr(vec![1.])
        .build()
        .is_err());
    assert!(SamplingFrame::builder()
        .run_lengths(vec![2])
        .tr(vec![1.])
        .start_time(vec![f64::INFINITY])
        .build()
        .is_err());
    assert!(SamplingFrame::uniform(3, f64::MAX, 0.1).is_err());
    let sampling = SamplingFrame::builder()
        .run_lengths(vec![2, 2])
        .tr(vec![2., 3.])
        .start_time(vec![1., 1.5])
        .build()
        .unwrap();
    assert_eq!(sampling.samples(), vec![1., 3., 5.5, 8.5]);
    assert_eq!(sampling.block_ids(), vec![0, 0, 1, 1]);
    assert!(sampling.tr_for_run(2).is_err());
}

#[test]
fn affine_validation_is_independent_of_coordinate_unit_scale() {
    for scale in [1e-200, 1e200] {
        let mut affine = identity();
        for (i, row) in affine.iter_mut().take(3).enumerate() {
            row[i] = scale;
        }
        assert!(VolumeSpace::new([1, 1, 1], affine, None, None).is_ok());
    }
    let mut singular = identity();
    singular[1] = singular[0];
    assert!(VolumeSpace::new([1, 1, 1], singular, None, None).is_err());
}
