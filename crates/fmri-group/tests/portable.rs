mod common;

use common::*;
use fmri_data::*;
use fmri_group::*;
use ndarray::array;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

fn path(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "fmri-group-{label}-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ))
}

#[test]
fn effects_only_round_trip_and_incomplete_is_not_openable() {
    let input = cohort(array![[1., 0.], [f64::NAN, 4.]], None);
    let dir = path("portable");
    write_cohort_bundle(&dir, &input, 1, PortableWriteOptions::default()).unwrap();
    let output = read_cohort_bundle(&dir, PortableReadOptions::default()).unwrap();
    assert_eq!(output.subjects(), input.subjects());
    assert_eq!(
        output.frame().features().space(),
        input.frame().features().space()
    );
    let block = output
        .frame()
        .collect("effect", &ExecutionContext::new(MemoryBudget::new(1024)))
        .unwrap();
    assert!(
        matches!(block.values(), Values::F64(x) if x[0] == 1. && x[1] == 0. && x[2].is_nan() && x[3] == 4.)
    );
    let incomplete = path("incomplete");
    let _writer = EstimateBundleWriter::create(
        &incomplete,
        PortableMetadata::from_cohort(&input),
        PortableWriteOptions::default(),
    )
    .unwrap();
    assert!(read_cohort_bundle(&incomplete, PortableReadOptions::default()).is_err());
    fs::remove_dir_all(dir).unwrap();
    fs::remove_dir_all(incomplete).unwrap();
}

#[test]
fn variance_checksum_and_block_order_are_enforced() {
    let input = cohort(array![[1., 2.], [3., 4.]], Some(array![[1., 1.], [2., 2.]]));
    let dir = path("variance");
    let metadata = PortableMetadata::from_cohort(&input);
    let mut writer =
        EstimateBundleWriter::create(&dir, metadata, PortableWriteOptions::default()).unwrap();
    assert!(writer
        .write_effect_block(1, array![[2.], [4.]].view())
        .is_err());
    writer
        .write_effect_block(0, array![[1.], [3.]].view())
        .unwrap();
    writer
        .write_effect_block(1, array![[2.], [4.]].view())
        .unwrap();
    writer
        .write_variance_block(0, array![[1.], [2.]].view())
        .unwrap();
    writer
        .write_variance_block(1, array![[1.], [2.]].view())
        .unwrap();
    writer.finish().unwrap();
    let payload = dir.join("variances.f64");
    let mut bytes = fs::read(&payload).unwrap();
    bytes[0] ^= 1;
    fs::write(&payload, bytes).unwrap();
    assert!(read_cohort_bundle(&dir, PortableReadOptions::default()).is_err());
    assert!(EstimateBundleWriter::create(
        &dir,
        PortableMetadata::from_cohort(&input),
        PortableWriteOptions::default()
    )
    .is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn reader_rejects_unknown_version_truncation_and_manifest_over_budget() {
    let input = cohort(array![[1.], [2.]], None);
    let dir = path("bad-manifest");
    write_cohort_bundle(&dir, &input, 1, PortableWriteOptions::default()).unwrap();
    let manifest = dir.join("manifest.json");
    let mut value: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest).unwrap()).unwrap();
    value["version"] = serde_json::json!(99);
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(read_cohort_bundle(&dir, PortableReadOptions::default()).is_err());
    value["version"] = serde_json::json!(1);
    value["metadata"]["feature_space"]["ids"] = serde_json::json!(["duplicate", "duplicate"]);
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(read_cohort_bundle(&dir, PortableReadOptions::default()).is_err());
    value["metadata"]["feature_space"]["ids"] = serde_json::json!(["f0"]);
    fs::write(&manifest, serde_json::to_vec(&value).unwrap()).unwrap();
    let payload = dir.join("effects.f64");
    let mut bytes = fs::read(&payload).unwrap();
    bytes.pop();
    fs::write(&payload, bytes).unwrap();
    assert!(read_cohort_bundle(&dir, PortableReadOptions::default()).is_err());
    assert!(read_cohort_bundle(
        &dir,
        PortableReadOptions {
            max_manifest_bytes: 1,
            ..PortableReadOptions::default()
        }
    )
    .is_err());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn volume_space_round_trip_preserves_support_order() {
    let input = cohort(array![[7., 8.], [9., 10.]], None);
    let dir = path("volume");
    let mut metadata = PortableMetadata::from_cohort(&input);
    metadata.features = FeatureAxis::from_space(FeatureSpace::Volume(
        VolumeSpace::new(
            [2, 1, 1],
            [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
            Some(vec![1, 0]),
            Some("synthetic".into()),
        )
        .unwrap(),
    ))
    .unwrap();
    let mut writer =
        EstimateBundleWriter::create(&dir, metadata, PortableWriteOptions::default()).unwrap();
    writer
        .write_effect_block(0, array![[7., 8.], [9., 10.]].view())
        .unwrap();
    writer.finish().unwrap();
    match output_space(read_cohort_bundle(&dir, PortableReadOptions::default()).unwrap()) {
        FeatureSpace::Volume(space) => {
            assert_eq!(space.support(), &[1, 0]);
            assert_eq!(space.template(), Some("synthetic"));
            assert_eq!(
                space.affine(),
                &[
                    [1., 0., 0., 0.],
                    [0., 1., 0., 0.],
                    [0., 0., 1., 0.],
                    [0., 0., 0., 1.]
                ]
            );
        }
        _ => panic!("volume space was not preserved"),
    }
    fs::remove_dir_all(dir).unwrap();
}

fn output_space(cohort: IndependentCohort) -> FeatureSpace {
    cohort.frame().features().space().clone()
}

#[test]
fn reader_rejects_unknown_nested_semantics_and_invalid_identity_before_payload_read() {
    let input = cohort(array![[1., 2.], [3., 4.]], Some(array![[1., 1.], [1., 1.]]));
    let dir = path("nested-metadata");
    write_cohort_bundle(&dir, &input, 1, PortableWriteOptions::default()).unwrap();
    let filename = dir.join("manifest.json");
    let original: serde_json::Value =
        serde_json::from_slice(&fs::read(&filename).unwrap()).unwrap();
    for section in ["feature_space", "variance_origins"] {
        let mut value = original.clone();
        if section == "feature_space" {
            value["metadata"][section]["unknown_semantics"] = true.into();
        } else {
            value["metadata"][section][0]["unknown_semantics"] = true.into();
        }
        fs::write(&filename, serde_json::to_vec(&value).unwrap()).unwrap();
        assert!(matches!(
            read_cohort_bundle(&dir, PortableReadOptions::default()),
            Err(Error::Json(_))
        ));
    }
    let mut value = original;
    value["metadata"]["subjects"][1]["participant"] =
        value["metadata"]["subjects"][0]["participant"].clone();
    fs::write(&filename, serde_json::to_vec(&value).unwrap()).unwrap();
    fs::remove_file(dir.join("effects.f64")).unwrap();
    assert!(matches!(
        read_cohort_bundle(&dir, PortableReadOptions::default()),
        Err(Error::Invalid(_))
    ));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn estimated_df_round_trip_and_metadata_preflight() {
    let input = cohort(array![[1., 2.], [3., 4.]], Some(array![[1., 1.], [1., 1.]]));
    let dir = path("origins");
    let mut metadata = PortableMetadata::from_cohort(&input);
    metadata.variance_origins = Some(vec![
        VarianceOrigin::Estimated {
            df: DegreesOfFreedom::Scalar(10.),
            method: "scalar".into(),
            approximate: false,
        },
        VarianceOrigin::Estimated {
            df: DegreesOfFreedom::ByFeature(vec![9., 8.]),
            method: "featurewise".into(),
            approximate: true,
        },
    ]);
    let mut writer =
        EstimateBundleWriter::create(&dir, metadata, PortableWriteOptions::default()).unwrap();
    writer
        .write_effect_block(0, array![[1., 2.], [3., 4.]].view())
        .unwrap();
    writer
        .write_variance_block(0, array![[1., 1.], [1., 1.]].view())
        .unwrap();
    writer.finish().unwrap();
    assert!(
        matches!(read_cohort_bundle(&dir, PortableReadOptions::default()).unwrap().variance_origins(), [VarianceOrigin::Estimated { df: DegreesOfFreedom::Scalar(scalar), approximate: false, .. }, VarianceOrigin::Estimated { df: DegreesOfFreedom::ByFeature(featurewise), approximate: true, .. }] if *scalar == 10. && *featurewise == vec![9., 8.])
    );
    let bad_dir = path("bad-metadata");
    let mut bad = PortableMetadata::from_cohort(&input);
    bad.subjects[1].participant = bad.subjects[0].participant.clone();
    assert!(EstimateBundleWriter::create(&bad_dir, bad, PortableWriteOptions::default()).is_err());
    assert!(!bad_dir.exists());
    let bad_dir = path("bad-units");
    let mut bad = PortableMetadata::from_cohort(&input);
    bad.estimand.units.clear();
    assert!(EstimateBundleWriter::create(&bad_dir, bad, PortableWriteOptions::default()).is_err());
    assert!(!bad_dir.exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn writer_rejects_wrong_shape_and_budget_before_payload_write() {
    let input = cohort(array![[1., 2.], [3., 4.]], None);
    let dir = path("budget");
    let mut writer = EstimateBundleWriter::create(
        &dir,
        PortableMetadata::from_cohort(&input),
        PortableWriteOptions { max_block_bytes: 8 },
    )
    .unwrap();
    assert!(writer
        .write_effect_block(0, array![[1., 2.], [3., 4.]].view())
        .is_err());
    assert_eq!(fs::metadata(dir.join("effects.f64")).unwrap().len(), 0);
    assert!(writer
        .write_effect_block(0, array![[1.], [2.], [3.]].view())
        .is_err());
    assert_eq!(fs::metadata(dir.join("effects.f64")).unwrap().len(), 0);
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn finish_never_overwrites_an_appearing_manifest() {
    let input = cohort(array![[1.], [2.]], None);
    let dir = path("manifest-race");
    let mut writer = EstimateBundleWriter::create(
        &dir,
        PortableMetadata::from_cohort(&input),
        PortableWriteOptions::default(),
    )
    .unwrap();
    writer
        .write_effect_block(0, array![[1.], [2.]].view())
        .unwrap();
    let manifest = dir.join("manifest.json");
    fs::write(&manifest, b"external manifest").unwrap();
    assert!(writer.finish().is_err());
    assert_eq!(fs::read(&manifest).unwrap(), b"external manifest");
    fs::remove_dir_all(dir).unwrap();
}
