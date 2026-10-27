use approx::assert_relative_eq;
use fmrihrf::hrf::derivatives::{SpmG2Hrf, SpmG3Hrf};
use fmrihrf::Hrf;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::PathBuf;

fn load_matrix_fixture(path: &str, n_cols: usize) -> (Array1<f64>, Array2<f64>) {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push(path);

    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();

    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim().starts_with("\"time\","),
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); n_cols];

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), n_cols + 1, "expected {} value columns", n_cols);

        let t: f64 = parts[0]
            .trim_matches('"')
            .parse()
            .expect("invalid time value");
        times.push(t);

        for (j, col) in cols.iter_mut().enumerate() {
            let v: f64 = parts[j + 1]
                .trim_matches('"')
                .parse()
                .expect("invalid value");
            col.push(v);
        }
    }

    let n_rows = times.len();
    assert!(n_rows > 0, "no data rows in fixture");

    let time_array = Array1::from(times);
    let mut values = Array2::zeros((n_rows, n_cols));
    for (j, col) in cols.iter().enumerate() {
        assert_eq!(col.len(), n_rows);
        for i in 0..n_rows {
            values[[i, j]] = col[i];
        }
    }

    (time_array, values)
}

/// Cross-validation of SpmG2Hrf against R HRF_SPMG2 fixture.
#[test]
fn test_spmg2_hrf_r_fixture_parity() {
    let (times, expected) = load_matrix_fixture("tests/fixtures/spmg2_hrf_default.csv", 2);

    // Match R defaults: P1=5, P2=15, A1=1.0 / 120.0
    let hrf = SpmG2Hrf::new(5.0, 15.0, 1.0 / 120.0);
    let values = hrf.evaluate(&times).expect("SpmG2Hrf evaluation failed");

    assert_eq!(values.shape(), expected.shape());

    for i in 0..values.nrows() {
        for j in 0..values.ncols() {
            assert_relative_eq!(
                values[[i, j]],
                expected[[i, j]],
                epsilon = 1e-6,
                max_relative = 1e-4
            );
        }
    }
}

/// Cross-validation of SpmG3Hrf against R HRF_SPMG3 fixture.
#[test]
fn test_spmg3_hrf_r_fixture_parity() {
    let (times, expected) = load_matrix_fixture("tests/fixtures/spmg3_hrf_default.csv", 3);

    // Match R defaults: P1=5, P2=15, A1=1.0 / 120.0
    let hrf = SpmG3Hrf::new(5.0, 15.0, 1.0 / 120.0);
    let values = hrf.evaluate(&times).expect("SpmG3Hrf evaluation failed");

    assert_eq!(values.shape(), expected.shape());

    for i in 0..values.nrows() {
        for j in 0..values.ncols() {
            assert_relative_eq!(
                values[[i, j]],
                expected[[i, j]],
                epsilon = 1e-6,
                max_relative = 1e-4
            );
        }
    }
}
