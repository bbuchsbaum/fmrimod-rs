use std::fs;
use std::io::Write;

use fmri_data::*;
use tempfile::NamedTempFile;

fn context(bytes: usize) -> ExecutionContext {
    ExecutionContext::new(MemoryBudget::new(bytes))
}

/// Write the fixture directly in the on-disk feature-major layout, rather than
/// deriving it from the reader implementation under test.
fn feature_major_fixture(shape: [usize; 2], values: &[f64]) -> NamedTempFile {
    let mut file = NamedTempFile::new().unwrap();
    for column in 0..shape[1] {
        for row in 0..shape[0] {
            file.write_all(&values[row * shape[1] + column].to_le_bytes())
                .unwrap();
        }
    }
    file.flush().unwrap();
    file
}

fn f64_bits(values: &Values) -> Vec<u64> {
    match values {
        Values::F64(values) => values.iter().map(|value| value.to_bits()).collect(),
        Values::F32(_) => panic!("expected f64 values"),
    }
}

#[test]
fn feature_major_file_reads_raw_little_endian_f64_into_row_major_blocks() {
    let values = vec![
        1.5,
        -0.0,
        f64::from_bits(0x7ff8_0000_0000_1234),
        4.25,
        0.0,
        -9.0,
    ];
    let file = feature_major_fixture([3, 2], &values);
    let source = SourceSpec::feature_major_f64_file(file.path(), [3, 2]).unwrap();

    assert_eq!(source.shape(), [3, 2]);
    assert_eq!(source.dtype(), DType::F64);
    let block = context(96).collect(&source).unwrap();
    assert_eq!(block.shape(), [3, 2]);
    assert_eq!(
        f64_bits(block.values()),
        values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn file_source_views_preserve_strides_repetitions_and_order() {
    let values = vec![
        0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 20.0, 21.0, 22.0, 23.0,
    ];
    let file = feature_major_fixture([3, 4], &values);
    let source = SourceSpec::feature_major_f64_file(file.path(), [3, 4]).unwrap();
    let first = source
        .view(
            ReadSelection::new(
                [3, 4],
                Selection::Indices(vec![2, 0, 2]),
                Selection::Indices(vec![3, 1, 3]),
            )
            .unwrap(),
        )
        .unwrap();
    let nested = first
        .view(
            ReadSelection::new(
                [3, 3],
                Selection::Indices(vec![1, 2]),
                Selection::Indices(vec![2, 0]),
            )
            .unwrap(),
        )
        .unwrap();

    let selection = ReadSelection::all(nested.shape()).unwrap();
    let execution = context(40);
    let mut reader = execution.open(&nested);
    let block = execution.read(&mut reader, &selection).unwrap();
    assert_eq!(block.shape(), [2, 2]);
    assert_eq!(block.values(), &Values::F64(vec![3.0, 3.0, 23.0, 23.0]));

    let empty = source
        .view(
            ReadSelection::new(
                [3, 4],
                Selection::Indices(vec![]),
                Selection::Indices(vec![2, 0]),
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(
        context(0).collect(&empty).unwrap().values(),
        &Values::F64(vec![])
    );
}

#[test]
fn file_source_budget_includes_scratch_and_precedes_revalidation_io() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let file = feature_major_fixture([3, 2], &values);
    let source = SourceSpec::feature_major_f64_file(file.path(), [3, 2]).unwrap();

    // Output needs 48 bytes; the contiguous block reader additionally owns a
    // 48-byte bounded byte scratch buffer.
    assert!(matches!(
        context(95).collect(&source),
        Err(FmriDatasetError::BudgetExceeded {
            required: 96,
            budget: 95
        })
    ));
    assert_eq!(
        context(96).collect(&source).unwrap().values(),
        &Values::F64(values)
    );

    fs::write(file.path(), [0_u8; 8]).unwrap();
    // The budget failure is determined without reopening/revalidating the now
    // truncated file. With sufficient budget, the metadata failure is clear.
    assert!(matches!(
        context(95).collect(&source),
        Err(FmriDatasetError::BudgetExceeded { .. })
    ));
    assert!(matches!(
        context(96).collect(&source),
        Err(FmriDatasetError::Invalid { .. })
    ));
}

#[test]
fn file_source_rejects_short_extra_nonregular_and_overflowing_inputs() {
    let short = NamedTempFile::new().unwrap();
    fs::write(short.path(), [0_u8; 7]).unwrap();
    assert!(matches!(
        SourceSpec::feature_major_f64_file(short.path(), [1, 1]),
        Err(FmriDatasetError::Invalid { .. })
    ));

    let extra = NamedTempFile::new().unwrap();
    fs::write(extra.path(), [0_u8; 9]).unwrap();
    assert!(matches!(
        SourceSpec::feature_major_f64_file(extra.path(), [1, 1]),
        Err(FmriDatasetError::Invalid { .. })
    ));

    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        SourceSpec::feature_major_f64_file(directory.path(), [0, 0]),
        Err(FmriDatasetError::Invalid { .. })
    ));

    let empty = NamedTempFile::new().unwrap();
    assert!(matches!(
        SourceSpec::feature_major_f64_file(empty.path(), [usize::MAX, 2]),
        Err(FmriDatasetError::SizeOverflow)
    ));
}

#[test]
fn contiguous_feature_block_crosses_scratch_boundaries_without_reordering() {
    let [rows, columns] = [97, 103];
    let values = (0..rows)
        .flat_map(|r| (0..columns).map(move |c| (1000 * r + c) as f64))
        .collect::<Vec<_>>();
    let file = feature_major_fixture([rows, columns], &values);
    let source = SourceSpec::feature_major_f64_file(file.path(), [rows, columns]).unwrap();
    let select =
        ReadSelection::new([rows, columns], Selection::All, Selection::Range(3..101)).unwrap();
    let required = rows * 98 * 8 + 65536;
    let ctx = context(required);
    let block = ctx.read(&mut ctx.open(&source), &select).unwrap();
    let expected = (0..rows)
        .flat_map(|r| (3..101).map(move |c| (1000 * r + c) as f64))
        .collect::<Vec<_>>();
    assert_eq!(block.values(), &Values::F64(expected));
    assert!(matches!(
        context(required - 1).read(&mut ctx.open(&source), &select),
        Err(FmriDatasetError::BudgetExceeded { .. })
    ));
    let select = ReadSelection::new(
        [rows, columns],
        Selection::All,
        Selection::Indices(vec![100, 2, 100]),
    )
    .unwrap();
    let block = ctx.read(&mut ctx.open(&source), &select).unwrap();
    let expected = (0..rows)
        .flat_map(|r| [100, 2, 100].map(move |c| (1000 * r + c) as f64))
        .collect::<Vec<_>>();
    assert_eq!(block.values(), &Values::F64(expected));
}
