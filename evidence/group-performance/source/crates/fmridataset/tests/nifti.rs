#![cfg(feature = "nifti")]

use std::fs;

use flate2::{write::GzEncoder, Compression};
use fmridataset::{ArrayReader, MemoryBudget, NiftiSource, ReadSelection, Selection, Values};
use tempfile::tempdir;

fn put_i16(bytes: &mut [u8], offset: usize, value: i16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn put_f32(bytes: &mut [u8], offset: usize, value: f32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Tiny NIfTI-1 single-file fixture. Values are x-fastest signed i16 frames.
fn fixture(units: u8, sform_code: i16, payload: &[i16]) -> Vec<u8> {
    let mut bytes = vec![0; 352];
    bytes[0..4].copy_from_slice(&348_i32.to_le_bytes());
    put_u16(&mut bytes, 40, 4);
    put_u16(&mut bytes, 42, 2);
    put_u16(&mut bytes, 44, 2);
    put_u16(&mut bytes, 46, 1);
    put_u16(&mut bytes, 48, 2);
    put_i16(&mut bytes, 70, 4); // int16
    put_i16(&mut bytes, 72, 16);
    put_f32(&mut bytes, 76, 1.0);
    put_f32(&mut bytes, 80, 2.0);
    put_f32(&mut bytes, 84, 2.0);
    put_f32(&mut bytes, 88, 2.0);
    put_f32(&mut bytes, 108, 352.0);
    put_f32(&mut bytes, 112, 2.0);
    put_f32(&mut bytes, 116, 0.5);
    bytes[123] = units;
    put_i16(&mut bytes, 254, sform_code);
    put_f32(&mut bytes, 280, 2.0);
    put_f32(&mut bytes, 300, 2.0);
    put_f32(&mut bytes, 320, 2.0);
    bytes[344..348].copy_from_slice(b"n+1\0");
    bytes.extend(payload.iter().flat_map(|v| v.to_le_bytes()));
    bytes
}

fn big_endian_fixture(payload: &[i16]) -> Vec<u8> {
    let mut bytes = fixture(2, 1, &[]);
    bytes[0..4].copy_from_slice(&348_i32.to_be_bytes());
    for (offset, value) in [(40, 4), (42, 2), (44, 2), (46, 1), (48, 2)] {
        bytes[offset..offset + 2].copy_from_slice(&(value as u16).to_be_bytes());
    }
    for (offset, value) in [(70, 4), (72, 16), (254, 1)] {
        bytes[offset..offset + 2].copy_from_slice(&(value as i16).to_be_bytes());
    }
    for (offset, value) in [
        (76, 1.0),
        (80, 2.0),
        (84, 2.0),
        (88, 2.0),
        (108, 352.0),
        (112, 2.0),
        (116, 0.5),
        (280, 2.0),
        (300, 2.0),
        (320, 2.0),
    ] {
        bytes[offset..offset + 4].copy_from_slice(&(value as f32).to_be_bytes());
    }
    bytes.extend(payload.iter().flat_map(|v| v.to_be_bytes()));
    bytes
}

fn qform_fixture(b: f32, c: f32, d: f32, qfac: f32) -> Vec<u8> {
    let mut bytes = fixture(2, 0, &[0; 8]);
    put_i16(&mut bytes, 252, 1);
    put_f32(&mut bytes, 76, qfac);
    put_f32(&mut bytes, 256, b);
    put_f32(&mut bytes, 260, c);
    put_f32(&mut bytes, 264, d);
    bytes
}

fn source(path: &std::path::Path) -> NiftiSource {
    NiftiSource::new(path, Some(vec![3, 0, 2]), Some("native-toy".into())).unwrap()
}

fn assert_affine(actual: &[[f64; 4]; 4], expected: [[f64; 4]; 4]) {
    for (actual, expected) in actual.iter().flatten().zip(expected.iter().flatten()) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

#[test]
fn streamed_reader_preserves_order_scaling_and_gzip_parity() {
    let directory = tempdir().unwrap();
    let bytes = fixture(2, 1, &[0, 1, 2, 3, 10, 11, 12, 13]);
    let nii = directory.path().join("toy.nii");
    let gz = directory.path().join("toy.nii.gz");
    let big_endian = directory.path().join("toy-be.nii");
    fs::write(&nii, &bytes).unwrap();
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    std::io::Write::write_all(&mut encoder, &bytes).unwrap();
    fs::write(&gz, encoder.finish().unwrap()).unwrap();
    fs::write(
        &big_endian,
        big_endian_fixture(&[0, 1, 2, 3, 10, 11, 12, 13]),
    )
    .unwrap();

    let selection = ReadSelection::new(
        [2, 3],
        Selection::Indices(vec![1, 0]),
        Selection::Indices(vec![2, 0, 1]),
    )
    .unwrap();
    for path in [&nii, &gz, &big_endian] {
        let input = source(path);
        assert_eq!(input.shape(), [2, 3]);
        assert_eq!(input.space().dimensions(), [2, 2, 1]);
        assert_eq!(input.space().support(), &[3, 0, 2]);
        assert_eq!(
            input.space().affine(),
            &[
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        );
        let mut reader = input.open_reader().unwrap();
        assert_eq!(reader.dtype(), fmridataset::DType::F64);
        let block = reader.read(&selection, MemoryBudget::new(128)).unwrap();
        assert_eq!(
            block.values(),
            &Values::F64(vec![24.5, 26.5, 20.5, 4.5, 6.5, 0.5])
        );
    }
}

#[test]
fn invalid_geometry_is_rejected_before_values() {
    let directory = tempdir().unwrap();
    let no_units = directory.path().join("no-units.nii");
    let no_form = directory.path().join("no-form.nii");
    let oversized_metadata = directory.path().join("oversized-metadata.nii");
    fs::write(&no_units, fixture(0, 1, &[0; 8])).unwrap();
    fs::write(&no_form, fixture(2, 0, &[0; 8])).unwrap();
    let mut oversized = fixture(2, 1, &[0; 8]);
    put_f32(&mut oversized, 108, (352 + 1024 * 1024 + 1) as f32);
    fs::write(&oversized_metadata, oversized).unwrap();
    assert!(NiftiSource::new(&no_units, None, None).is_err());
    assert!(NiftiSource::new(&no_form, None, None).is_err());
    assert!(NiftiSource::new(&oversized_metadata, None, None).is_err());
}

#[test]
fn metadata_mutation_including_scaling_invalidates_the_source() {
    let directory = tempdir().unwrap();
    let path = directory.path().join("mutable.nii");
    fs::write(&path, fixture(2, 1, &[0; 8])).unwrap();
    let input = source(&path);
    let mut changed = fixture(2, 1, &[0; 8]);
    put_f32(&mut changed, 112, 3.0);
    fs::write(&path, changed).unwrap();
    assert!(input.open_reader().is_err());
}

#[test]
fn budget_is_checked_before_truncated_values_are_read() {
    let directory = tempdir().unwrap();
    let truncated = directory.path().join("truncated.nii");
    fs::write(&truncated, fixture(2, 1, &[])).unwrap();
    let input = source(&truncated);
    let mut reader = input.open_reader().unwrap();
    let selection = ReadSelection::all(input.shape()).unwrap();
    assert!(reader.read(&selection, MemoryBudget::new(0)).is_err());
    assert!(reader.read(&selection, MemoryBudget::new(128)).is_err());
}

#[test]
fn qform_accepts_standard_rotation_reflection_and_near_unit_quaternions() {
    let directory = tempdir().unwrap();
    let rotation = directory.path().join("qform-rotation.nii");
    let reflection = directory.path().join("qform-reflection.nii");
    let near_unit = directory.path().join("qform-near-unit.nii");
    let half_sqrt = 0.5_f32.sqrt();
    fs::write(&rotation, qform_fixture(0.0, 0.0, half_sqrt, 1.0)).unwrap();
    fs::write(&reflection, qform_fixture(0.0, 0.0, 0.0, -1.0)).unwrap();
    // The reference routine normalizes this slightly oversized quaternion.
    fs::write(&near_unit, qform_fixture(1.0000001, 0.0, 0.0, 0.0)).unwrap();

    assert_affine(
        source(&rotation).space().affine(),
        [
            [0.0, -2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    assert_affine(
        source(&reflection).space().affine(),
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
    assert_affine(
        source(&near_unit).space().affine(),
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, -2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    );
}
