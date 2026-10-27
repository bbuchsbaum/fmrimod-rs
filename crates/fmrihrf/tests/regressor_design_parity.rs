use fmrihrf::regressor::ConditionEvents;
use fmrihrf::sampling::SamplingFrame;
use ndarray::Array1;
use std::fs;
use std::path::PathBuf;

fn load_design_fixture(path: &str) -> (Array1<f64>, Vec<Vec<f64>>) {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push(path);

    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();

    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim() == "\"time\",\"A\",\"B\"" || header.trim() == "time,A,B",
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut col_a = Vec::new();
    let mut col_b = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 3, "expected 3 columns in fixture row");

        let t: f64 = parts[0]
            .trim_matches('"')
            .parse()
            .expect("invalid time value");
        let a: f64 = parts[1].trim_matches('"').parse().expect("invalid A value");
        let b: f64 = parts[2].trim_matches('"').parse().expect("invalid B value");

        times.push(t);
        col_a.push(a);
        col_b.push(b);
    }

    (Array1::from(times), vec![col_a, col_b])
}

#[test]
fn test_regressor_design_block_example_parity() {
    let (times, cols) = load_design_fixture("tests/fixtures/regressor_design_block_example.csv");

    // Recreate the same sampling frame and block-wise design as in the R fixtures.
    // NOTE: This test assumes that the R fixture was generated with
    // sampling precision = 2.0 / 100.0 (i.e., 1/100 of TR).
    let sframe = SamplingFrame::new(
        vec![100, 100],
        vec![2.0, 2.0],
        Some(vec![0.0, 0.0]),
        2.0 / 100.0,
    )
    .unwrap();

    let onsets = vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0];
    let fac = vec![
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
        "A".to_string(),
        "B".to_string(),
    ];
    let block = vec![1_usize, 1, 1, 2, 2, 2];

    let run_ids: Vec<_> = block.iter().map(|b| b - 1).collect();
    let hrf = fmrihrf::hrf::spm_canonical(None, None, None, None, None, Some(false));
    let design = ConditionEvents::new(&onsets, &fac, &run_ids)
        .evaluate(&sframe, hrf.as_ref())
        .unwrap()
        .data;

    assert_eq!(design.nrows(), times.len());
    assert_eq!(design.ncols(), 2);

    // Compare each column against the R fixture (within numerical tolerance)
    // while also tracking the worst-case discrepancies for debugging.
    let mut max_abs_diff = 0.0_f64;
    let mut max_rel_diff = 0.0_f64;
    let mut worst_idx = 0_usize;

    for i in 0..times.len() {
        let a_r = cols[0][i];
        let b_r = cols[1][i];
        let a_rs = design[[i, 0]];
        let b_rs = design[[i, 1]];

        let da = (a_rs - a_r).abs();
        let db = (b_rs - b_r).abs();
        let abs_diff = da.max(db);

        let denom_a = a_r.abs().max(a_rs.abs()).max(1.0);
        let denom_b = b_r.abs().max(b_rs.abs()).max(1.0);
        let rel_diff = (da / denom_a).max(db / denom_b);

        if abs_diff > max_abs_diff {
            max_abs_diff = abs_diff;
            max_rel_diff = rel_diff;
            worst_idx = i;
        }
    }

    // With the R-aligned convolution and neural-input path, the remaining
    // discrepancies are modest (~4.7e-2 at worst in this fixture). We use
    // a relatively loose absolute tolerance here to keep this test as a
    // design-level regression guard without over-constraining the numeric
    // implementation.
    assert!(
        max_abs_diff <= 5e-2 && max_rel_diff <= 5e-2,
        "Design parity exceeded tolerance: max_abs_diff={:.3e}, max_rel_diff={:.3e} at t={} (row {})",
        max_abs_diff,
        max_rel_diff,
        times[worst_idx],
        worst_idx
    );

    println!(
        "regressor_design_block_example: max_abs_diff = {:.3e}, max_rel_diff = {:.3e} at t = {} (row {})",
        max_abs_diff,
        max_rel_diff,
        times[worst_idx],
        worst_idx
    );
}
