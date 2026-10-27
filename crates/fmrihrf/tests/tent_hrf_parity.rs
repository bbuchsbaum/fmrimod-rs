use fmrihrf::hrf::tent_hrf;
use ndarray::{Array1, Array2};
use std::fs;
use std::path::PathBuf;

/// Cross-validation of `tent_hrf` against R fixture generated
/// from the authoritative R package (`hrf_tent_generator`).
#[test]
fn test_tent_hrf_r_fixture_parity() {
    // Locate the CSV fixture generated via R's hrf_tent_generator.
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures/tent_hrf_n5_span24.csv");

    let csv = fs::read_to_string(path).expect("failed to read tent HRF fixture");
    let mut lines = csv.lines();

    // Skip header
    let header = lines.next().expect("fixture has no header");
    let expected_header = "\"time\",\"b1\",\"b2\",\"b3\",\"b4\",\"b5\"";
    assert_eq!(header.trim(), expected_header);

    let mut times = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 5];

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 6, "expected 6 columns in fixture row");

        let t: f64 = parts[0]
            .trim_matches('"')
            .parse()
            .expect("invalid time value");
        times.push(t);

        for (j, col) in cols.iter_mut().enumerate() {
            let v: f64 = parts[j + 1]
                .trim_matches('"')
                .parse()
                .expect("invalid basis value");
            col.push(v);
        }
    }

    let n_rows = times.len();
    assert!(n_rows > 0, "no data rows in fixture");

    // Build ndarray structures from fixture.
    let time_array = Array1::from(times);
    let mut expected = Array2::zeros((n_rows, 5));
    for (j, col) in cols.iter().enumerate() {
        assert_eq!(col.len(), n_rows);
        for i in 0..n_rows {
            expected[[i, j]] = col[i];
        }
    }

    // Evaluate Rust tent_hrf on the same grid.
    let hrf = tent_hrf(5, Some(24.0));
    let values = hrf
        .evaluate(&time_array)
        .expect("tent_hrf evaluation failed");

    // Shape parity.
    assert_eq!(values.shape(), expected.shape());

    // Element-wise numerical parity summary (allowing for spline implementation differences).
    let mut max_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    for i in 0..n_rows {
        for j in 0..5 {
            let r_val = expected[[i, j]];
            let rust_val = values[[i, j]];
            let abs_diff = (rust_val - r_val).abs();
            max_abs_diff = max_abs_diff.max(abs_diff);

            if r_val != 0.0 {
                let rel_diff = abs_diff / r_val.abs();
                max_rel_diff = max_rel_diff.max(rel_diff);
            }
        }
    }

    // Report the bound across every row, including support boundaries.
    println!(
        "tent_hrf R fixture parity: max_abs_diff = {}, max_rel_diff = {}",
        max_abs_diff, max_rel_diff
    );

    assert!(
        max_abs_diff < 1e-12,
        "max_abs_diff too large: {}",
        max_abs_diff
    );
}
