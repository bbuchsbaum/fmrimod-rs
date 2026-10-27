use approx::assert_relative_eq;
use fmri_hrf::hrf::{gamma_hrf, gaussian_hrf, lwu_hrf, mexhat_hrf};
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

/// Cross-validation of gamma_hrf against R hrf_gamma fixture.
#[test]
fn test_gamma_hrf_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/gamma_hrf_shape6_rate1.csv");

    // R uses `rate`; current Rust implementation uses `scale` with the
    // same numeric role in the exponent. We disable normalization to
    // compare raw gamma PDF values.
    let hrf = gamma_hrf(6.0, 1.0, Some(false));
    let values = hrf.evaluate(&times).expect("gamma_hrf evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of gaussian_hrf against R hrf_gaussian fixture.
#[test]
fn test_gaussian_hrf_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/gaussian_hrf_mean6_sd2.csv");

    // R uses mean/sd; Rust uses center/width with the same semantics.
    // Disable normalization to compare raw Gaussian PDF values.
    let hrf = gaussian_hrf(6.0, 2.0, Some(false));
    let values = hrf
        .evaluate(&times)
        .expect("gaussian_hrf evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of mexhat_hrf against R hrf_mexhat fixture.
#[test]
fn test_mexhat_hrf_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/mexhat_hrf_mean6_sd2.csv");

    // R uses mean/sd; Rust uses center/sigma with equivalent semantics.
    let hrf = mexhat_hrf(2.0, 6.0);
    let values = hrf.evaluate(&times).expect("mexhat_hrf evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

/// Cross-validation of lwu_hrf against R hrf_lwu fixture with default parameters.
#[test]
fn test_lwu_hrf_r_fixture_parity() {
    let (times, expected) = load_scalar_fixture("tests/fixtures/lwu_hrf_default.csv");

    // Defaults: tau=6, sigma=2.5, rho=0.35, normalize="none" in both R and Rust.
    let hrf = lwu_hrf(None, None, None, None).expect("failed to construct lwu_hrf");
    let values = hrf.evaluate(&times).expect("lwu_hrf evaluation failed");
    assert_eq!(values.shape(), &[times.len(), 1]);

    for i in 0..times.len() {
        let rust_val = values[[i, 0]];
        let r_val = expected[i];
        assert_relative_eq!(rust_val, r_val, epsilon = 1e-10, max_relative = 1e-10);
    }
}

// Cross-validation of `inv_logit_hrf` against R's `hrf_inv_logit` is
// intentionally deferred: Rust currently exposes a simpler 2-parameter
// inverse logit (center/scale) while R defines a 5-parameter
// difference-of-logits form. A fixture-based parity test will be added
// once the full 5-parameter variant is implemented in Rust.
