//! Golden-style regression tests for current fmridesign APIs.

use fmridesign::baseline::{baseline_model, BasisType, InterceptType, SamplingInfo};
use fmridesign::formula::event_model;
use fmridesign::sampling::{SamplingFrame, SamplingFrameExt};
use ndarray::{concatenate, Array2, Axis};

fn create_sampling_frame(block_lengths: &[usize], tr: f64) -> SamplingFrame {
    SamplingFrame::new(block_lengths.to_vec(), vec![tr], None::<Vec<f64>>, 0.1)
        .expect("valid sampling frame")
}

fn create_sampling_info(block_lengths: &[usize], tr: f64) -> SamplingInfo {
    SamplingInfo::from_frame(create_sampling_frame(block_lengths, tr))
}

fn create_event_data(rows: &[(&str, f64)]) -> fmridesign::DataFrame {
    let mut data = fmridesign::DataFrame::new();
    let onsets: Vec<f64> = rows.iter().map(|(_, onset)| *onset).collect();
    let durations: Vec<f64> = rows.iter().map(|_| 0.0).collect();
    let conditions: Vec<String> = rows
        .iter()
        .map(|(condition, _)| (*condition).to_string())
        .collect();
    data.add_column("onset", onsets)
        .expect("valid onset column");
    data.add_column("duration", durations)
        .expect("valid duration column");
    data.add_str_column("condition", conditions)
        .expect("valid condition column");
    data
}

fn column_variance(matrix: &Array2<f64>, col: usize) -> f64 {
    let col_view = matrix.column(col);
    let mean = col_view.mean().unwrap_or(0.0);
    col_view
        .iter()
        .map(|value| {
            let centered = *value - mean;
            centered * centered
        })
        .sum::<f64>()
        / matrix.nrows() as f64
}

fn approximate_condition_number(matrix: &Array2<f64>) -> f64 {
    let mut min_norm = f64::INFINITY;
    let mut max_norm = 0.0f64;

    for col in 0..matrix.ncols() {
        let norm = matrix
            .column(col)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        min_norm = min_norm.min(norm);
        max_norm = max_norm.max(norm);
    }

    if min_norm > 0.0 {
        max_norm / min_norm
    } else {
        f64::INFINITY
    }
}

#[test]
fn golden_test_basis_functions_bspline_invariants() {
    let sampling = create_sampling_info(&[80], 2.0);
    let baseline = baseline_model(
        BasisType::BSpline,
        5,
        &sampling,
        InterceptType::Global,
        None,
    )
    .unwrap();

    let matrix = baseline.design_matrix().unwrap();
    assert_eq!(matrix.shape(), &[80, 6]);

    for col in 0..5 {
        let min_val = matrix
            .column(col)
            .iter()
            .fold(f64::INFINITY, |acc, v| acc.min(*v));
        assert!(min_val >= -1e-6, "column {} minimum {}", col, min_val);
    }

    for row in [20usize, 40, 60] {
        let basis_sum: f64 = (0..5).map(|col| matrix[[row, col]]).sum();
        assert!(
            (basis_sum - (1.0 - (1.0 - row as f64 / 79.0).powi(5))).abs() <= 1e-12,
            "row {} basis sum {}",
            row,
            basis_sum
        );
    }

    for row in 0..80 {
        assert!((matrix[[row, 5]] - 1.0).abs() <= 1e-9);
    }
}

#[test]
fn golden_test_basis_functions_orthogonal_polynomials() {
    let sampling = create_sampling_info(&[50], 2.0);
    let baseline = baseline_model(
        BasisType::Polynomial,
        3,
        &sampling,
        InterceptType::Global,
        None,
    )
    .unwrap();

    let matrix = baseline.design_matrix().unwrap();
    assert_eq!(matrix.shape(), &[50, 4]);

    for left in 0..3 {
        for right in (left + 1)..3 {
            let dot: f64 = (0..50)
                .map(|row| matrix[[row, left]] * matrix[[row, right]])
                .sum();
            assert!(dot.abs() <= 1e-6, "dot({}, {}) = {}", left, right, dot);
        }
    }

    for row in 0..50 {
        assert!((matrix[[row, 3]] - 1.0).abs() <= 1e-9);
    }
}

#[test]
fn golden_test_event_model_basic_hrf() {
    let sampling_frame = SamplingFrame::from_duration(200.0, 2.0);
    let data = create_event_data(&[("A", 10.0), ("B", 30.0), ("A", 60.0), ("B", 80.0)]);
    let model = event_model("hrf(condition)", &data, sampling_frame, 2.0).unwrap();

    let matrix = model.generate_design_matrix().unwrap();
    assert_eq!(matrix.shape(), &[100, 2]);

    for row in 0..4 {
        assert!(matrix[[row, 0]].abs() < 0.1);
        assert!(matrix[[row, 1]].abs() < 0.1);
    }

    let peak_a = matrix.column(0).iter().fold(0.0f64, |acc, v| acc.max(*v));
    let peak_b = matrix.column(1).iter().fold(0.0f64, |acc, v| acc.max(*v));
    assert!(peak_a > 0.0);
    assert!(peak_b > 0.0);
}

#[test]
fn golden_test_core_design_matrix_conditioning() {
    let sampling_frame = SamplingFrame::from_duration(200.0, 2.0);
    let sampling = SamplingInfo::from_frame(sampling_frame.clone());
    let baseline = baseline_model(
        BasisType::Polynomial,
        3,
        &sampling,
        InterceptType::Global,
        None,
    )
    .unwrap()
    .design_matrix()
    .unwrap();

    let data = create_event_data(&[
        ("condition", 0.0),
        ("condition", 10.0),
        ("condition", 20.0),
        ("condition", 30.0),
        ("condition", 40.0),
        ("condition", 50.0),
        ("condition", 60.0),
        ("condition", 70.0),
        ("condition", 80.0),
        ("condition", 90.0),
    ]);
    let event_matrix = event_model("hrf(condition)", &data, sampling_frame, 2.0)
        .unwrap()
        .generate_design_matrix()
        .unwrap();

    let combined = concatenate(Axis(1), &[baseline.view(), event_matrix.view()]).unwrap();
    let kappa = approximate_condition_number(&combined);
    assert!(kappa.is_finite());
    assert!(kappa < 1e12, "condition number {}", kappa);

    for col in 0..combined.ncols() {
        let variance = column_variance(&combined, col);
        if variance <= 1e-12 {
            let first = combined[[0, col]];
            assert!(
                combined
                    .column(col)
                    .iter()
                    .all(|value| (*value - first).abs() <= 1e-12),
                "column {} has zero variance but is not constant",
                col
            );
        } else {
            assert!(variance < 1e10, "column {} variance {}", col, variance);
        }
    }
}
