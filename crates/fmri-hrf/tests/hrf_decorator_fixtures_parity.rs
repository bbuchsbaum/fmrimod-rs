//! Cross-validation tests for additional HRFs and decorators
//! against fixtures generated from the R `fmri_hrf` package.
//!
//! The numeric fixtures are produced by running
//! `scripts/generate_r_fixtures.R` in the R repo, which writes
//! CSV files into `tests/fixtures/`. These tests mirror the
//! scalar HRF parity checks in `tests/hrf_fixtures_parity.rs`.

use approx::assert_relative_eq;
use fmri_hrf::hrf::functions::{half_cosine_hrf_segments, spm_canonical};
use ndarray::Array1;
use std::fs;
use std::path::PathBuf;

fn load_scalar_fixture(path: &str) -> (Array1<f64>, Array1<f64>) {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push(path);

    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();

    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim().starts_with("\"time\",") || header.trim().starts_with("time,"),
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut values = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 2, "expected 2 columns in fixture row");

        let t: f64 = parts[0]
            .trim_matches('"')
            .parse()
            .expect("invalid time value");
        let v: f64 = parts[1].trim_matches('"').parse().expect("invalid value");
        times.push(t);
        values.push(v);
    }

    (Array1::from(times), Array1::from(values))
}

/// Cross-validation of 6-parameter half-cosine HRF against R fixture.
///
/// The R fixture is generated via:
/// `hrf_half_cosine(times_30, h1=1, h2=5, h3=7, h4=7, f1=0, f2=0)`.
#[test]
fn test_half_cosine_hrf_segments_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/half_cosine_hrf_default.csv");

    // Rust half_cosine_hrf_segments matches the same parameterization.
    let hrf = half_cosine_hrf_segments(1.0, 5.0, 7.0, 7.0, 0.0, 0.0);
    let values = hrf
        .evaluate(&times)
        .expect("half_cosine_hrf_segments evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of a lagged SPMG1 HRF (lag=2s) against R fixture.
///
/// The R fixture is generated via:
/// `lag_hrf(HRF_SPMG1, lag=2)` evaluated on `seq(0, 30, by=0.5)`.
#[test]
fn test_spmg1_lag2_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/spmg1_lag2_hrf.csv");

    // Use the R-matched SPMG1 formula without normalization.
    let base = || spm_canonical(None, None, None, None, None, Some(false));

    // Apply the same lag semantics as R's lag_hrf using gen_hrf_lagged.
    let lag = 2.0;
    let hrf = fmri_hrf::hrf::gen_hrf_lagged(base, &[lag]);
    let values = hrf
        .evaluate(&times)
        .expect("lagged spm_canonical evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of a blocked SPMG1 HRF (width=5s) against R fixture.
///
/// The R fixture is generated via:
/// `block_hrf(HRF_SPMG1, width=5)` evaluated on `seq(0, 30, by=0.5)`.
#[test]
fn test_spmg1_block_width5_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/spmg1_block_width5_hrf.csv");

    let base = || spm_canonical(None, None, None, None, None, Some(false));
    let hrf = fmri_hrf::hrf::gen_hrf_blocked(base, &[5.0]);
    let values = hrf
        .evaluate(&times)
        .expect("blocked spm_canonical evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of multi-lag SPMG1 basis via gen_hrf_lagged against R fixture.
///
/// The R fixture is generated via:
/// `gen_hrf_lagged(HRF_SPMG1, lag=c(0,2,4), normalize=FALSE)` on `seq(0,30,by=0.5)`.
#[test]
fn test_spmg1_gen_hrf_lagged_multi_r_fixture_parity() {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push("tests/fixtures/spmg1_gen_hrf_lagged_lags0_2_4.csv");

    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();
    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim().starts_with("\"time\",") || header.trim().starts_with("time,"),
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(), Vec::new(), Vec::new()];
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 4, "expected 4 columns (time + 3 lags)");
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

    let lags = [0.0, 2.0, 4.0];
    let base = || spm_canonical(None, None, None, None, None, Some(false));
    let hrf = fmri_hrf::hrf::gen_hrf_lagged(base, &lags);
    let times = Array1::from(times);
    let values = hrf
        .evaluate(&times)
        .expect("gen_hrf_lagged evaluation failed");
    assert_eq!(values.shape(), &[times.len(), lags.len()]);

    for (b, lag_col) in cols.iter().enumerate() {
        for i in 0..times.len() {
            let rust_val = values[[i, b]];
            let r_val = lag_col[i];
            assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
        }
    }
}

/// Cross-validation of multi-width SPMG1 basis via gen_hrf_blocked against R fixture.
///
/// The R fixture is generated via:
/// `gen_hrf_blocked(HRF_SPMG1, width=c(1,2,4), precision=.1, half_life=Inf, summate=TRUE, normalize=FALSE)`.
#[test]
fn test_spmg1_gen_hrf_blocked_multi_r_fixture_parity() {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push("tests/fixtures/spmg1_gen_hrf_blocked_widths1_2_4.csv");
    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();
    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim().starts_with("\"time\",") || header.trim().starts_with("time,"),
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(), Vec::new(), Vec::new()];
    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 4, "expected 4 columns (time + 3 widths)");
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

    let widths = [1.0, 2.0, 4.0];
    let base = || spm_canonical(None, None, None, None, None, Some(false));
    let hrf = fmri_hrf::hrf::gen_hrf_blocked(base, &widths);
    let times = Array1::from(times);
    let values = hrf
        .evaluate(&times)
        .expect("gen_hrf_blocked evaluation failed");
    assert_eq!(values.shape(), &[times.len(), widths.len()]);

    for (b, width_col) in cols.iter().enumerate() {
        for i in 0..times.len() {
            let rust_val = values[[i, b]];
            let r_val = width_col[i];
            assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
        }
    }
}
