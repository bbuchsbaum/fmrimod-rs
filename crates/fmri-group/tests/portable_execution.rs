use fmri_group::*;
use fmridataset::{
    Assay, AssayAnnotations, Axis, ExecutionContext, FeatureAxis, FeatureId, FeatureSpace,
    FmriFrame, IndexSpace, MemoryBudget, Metadata, NumericBlock, ObservationId, SourceSpec,
};
use ndarray::Array2;
use std::time::{SystemTime, UNIX_EPOCH};

fn metadata(n: usize, f: usize) -> PortableMetadata {
    PortableMetadata {
        subjects: (0..n)
            .map(|i| Subject {
                observation: ObservationId::new(format!("o{i}")).unwrap(),
                dataset: "portable-test".into(),
                participant: format!("p{i}"),
                measurement: "baseline".into(),
            })
            .collect(),
        estimand: Estimand {
            id: "fixture".into(),
            definition: "portable execution fixture".into(),
            units: "arbitrary".into(),
        },
        features: FeatureAxis::from_space(FeatureSpace::Index(
            IndexSpace::new(
                "fixture",
                (0..f)
                    .map(|j| FeatureId::new(format!("f{j}")).unwrap())
                    .collect(),
            )
            .unwrap(),
        ))
        .unwrap(),
        variance_origins: Some(vec![
            VarianceOrigin::Known {
                method: "known".into()
            };
            n
        ]),
        fit_receipts: vec![
            FitReceipt {
                model: "fixture".into(),
                noise: "known".into(),
                selection: "all".into(),
            };
            n
        ],
        admission: Admission {
            target: "fixture".into(),
            evidence: "identical generated support".into(),
        },
    }
}

fn fixture(dir: &std::path::Path, n: usize, f: usize, block: usize) {
    let w = (0..n)
        .map(|i| 0.25 + (i % 5) as f64 * 0.125)
        .collect::<Vec<_>>();
    let mut writer = EstimateBundleWriter::create(
        dir,
        metadata(n, f),
        PortableWriteOptions {
            max_block_bytes: n * block * 8,
        },
    )
    .unwrap();
    for start in (0..f).step_by(block) {
        let end = (start + block).min(f);
        let y = Array2::from_shape_fn((n, end - start), |(i, k)| {
            let j = start + k;
            if (i, j) == (1, 3) {
                f64::NAN
            } else {
                0.25 + ((i + 1) as f64 * (j + 1) as f64 * 0.017).sin()
                    + 0.3 * ((i + 1) as f64 * 0.23).cos()
            }
        });
        let v = Array2::from_shape_fn((n, end - start), |(i, k)| {
            let j = start + k;
            if (i, j) == (2, 4) {
                0.
            } else {
                w[i] * 2_f64.powi(j as i32 % 5 - 2)
            }
        });
        writer.write_effect_block(start, y.view()).unwrap();
        writer.write_variance_block(start, v.view()).unwrap();
    }
    writer.finish().unwrap();
}

fn memory(cohort: &IndependentCohort) -> IndependentCohort {
    let [n, f] = cohort.frame().shape();
    let context = ExecutionContext::new(MemoryBudget::new(64 << 20));
    let effect = context
        .collect(cohort.frame().assay("effect").unwrap().source())
        .unwrap();
    let variance = context
        .collect(cohort.frame().assay("variance").unwrap().source())
        .unwrap();
    let ids = cohort.frame().observations().ids().to_vec();
    let feature_ids = cohort.frame().features().axis().ids().to_vec();
    let assays = vec![
        Assay::new(
            "effect",
            SourceSpec::memory(NumericBlock::new([n, f], effect.into_values()).unwrap()),
            ids.clone(),
            feature_ids.clone(),
            AssayAnnotations::default(),
        )
        .unwrap(),
        Assay::new(
            "variance",
            SourceSpec::memory(NumericBlock::new([n, f], variance.into_values()).unwrap()),
            ids.clone(),
            feature_ids,
            AssayAnnotations::default(),
        )
        .unwrap(),
    ];
    IndependentCohort::new(
        FmriFrame::new(
            Axis::new(ids, vec![]).unwrap(),
            cohort.frame().features().clone(),
            assays,
            "effect",
            Metadata::new(),
        )
        .unwrap(),
        cohort.subjects().to_vec(),
        cohort.estimand().clone(),
        "effect".into(),
        Some(("variance".into(), cohort.variance_origins().to_vec())),
        cohort.fit_receipts().to_vec(),
        cohort.admission().clone(),
    )
    .unwrap()
}

fn collect(
    cohort: &IndependentCohort,
    method: HeterogeneityEstimator,
    block: usize,
    shared: bool,
) -> (Vec<MetaRegressionFeature>, MetaRegressionReceipt) {
    let n = cohort.subjects().len();
    let design = GroupDesign::new(
        cohort.frame().observations().ids().to_vec(),
        vec!["x0".into(), "x1".into()],
        Array2::from_shape_fn((n, 2), |(i, j)| {
            (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / n as f64).cos()
        }),
    )
    .unwrap();
    let spec = MetaOutputSpec {
        contrasts: design
            .columns()
            .iter()
            .map(|name| GroupContrast::named(name.clone(), &design, &[(name, 1.)]).unwrap())
            .collect(),
        uncertainty: MetaUncertainty::Marginal,
        p_values: false,
        confidence_level: None,
    };
    let mut plan = PreparedMetaRegression::new(
        &design,
        spec,
        MetaRegressionOptions {
            heterogeneity: method,
            execution: FitOptions {
                block_features: block,
                max_working_bytes: 64 << 20,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    if shared {
        plan = plan
            .with_shared_variances((0..n).map(|i| 0.25 + (i % 5) as f64 * 0.125).collect())
            .unwrap();
    }
    let mut values = Vec::new();
    let receipt = plan
        .fit(cohort, |b| {
            values.extend(b.features);
            Ok(())
        })
        .unwrap();
    (values, receipt)
}

fn near(a: f64, b: f64) {
    assert!(
        (a.is_nan() && b.is_nan()) || (a - b).abs() <= 2e-9 * (1. + b.abs()),
        "{a} != {b}"
    );
}

fn temporary_directory() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "fmri-group-portable-execution-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

fn same(left: &[MetaRegressionFeature], right: &[MetaRegressionFeature]) {
    assert_eq!(left.len(), right.len());
    for (a, b) in left.iter().zip(right) {
        assert_eq!(a.feature, b.feature);
        assert_eq!(a.status, b.status);
        assert_eq!(a.n, b.n);
        assert_eq!(a.rank, b.rank);
        match (&a.estimates, &b.estimates) {
            (Some(a), Some(b)) => {
                for (x, y) in a.effects.iter().zip(&b.effects) {
                    near(*x, *y);
                }
                for (x, y) in a
                    .standard_errors
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(b.standard_errors.as_ref().unwrap())
                {
                    near(*x, *y);
                }
            }
            (None, None) => {}
            _ => panic!("estimate availability differs"),
        }
        match (&a.diagnostics, &b.diagnostics) {
            (Some(a), Some(b)) => {
                near(a.q_fixed, b.q_fixed);
                near(a.q_final, b.q_final);
                near(a.tau_squared, b.tau_squared);
                assert_eq!(a.residual_df, b.residual_df);
            }
            (None, None) => {}
            _ => panic!("diagnostic availability differs"),
        }
    }
}

#[test]
fn portable_disk_and_materialized_memory_match_across_methods_blocks_and_shared_weights() {
    let root = temporary_directory();
    std::fs::create_dir(&root).unwrap();
    let directory = root.join("fixture");
    fixture(&directory, 9, 7, 3);
    let disk = read_cohort_bundle(&directory, PortableReadOptions::default()).unwrap();
    let memory = memory(&disk);

    for method in [
        HeterogeneityEstimator::CommonEffect,
        HeterogeneityEstimator::DerSimonianLaird,
        HeterogeneityEstimator::PauleMandel,
    ] {
        for block in [1, 3, 8] {
            for shared in [false, true] {
                let (expected, expected_receipt) = collect(&disk, method, block, shared);
                let (actual, actual_receipt) = collect(&memory, method, block, shared);
                same(&expected, &actual);
                assert_eq!(
                    expected_receipt.shared_features,
                    actual_receipt.shared_features
                );
                assert!(expected
                    .iter()
                    .any(|x| x.status == MetaStatus::MissingObservation));
            }
        }
    }
    std::fs::remove_dir_all(root).unwrap();
}
