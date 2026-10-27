use approx::assert_relative_eq;
use fmrihrf::hrf::derivatives::SpmG3Hrf;
use fmrihrf::regressor::ConditionEvents;
use fmrihrf::sampling::SamplingFrame;
use ndarray::Array1;
use std::fs;
use std::path::PathBuf;

fn load_spmg3_design_fixture(path: &str) -> (Array1<f64>, Vec<Vec<f64>>) {
    let mut full_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    full_path.push(path);

    let csv = fs::read_to_string(&full_path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", full_path.display(), e));
    let mut lines = csv.lines();

    let header = lines.next().expect("fixture has no header");
    assert!(
        header.trim() == "\"time\",\"A_c1\",\"A_c2\",\"A_c3\",\"B_c1\",\"B_c2\",\"B_c3\""
            || header.trim() == "time,A_c1,A_c2,A_c3,B_c1,B_c2,B_c3",
        "unexpected header in {}: {}",
        full_path.display(),
        header
    );

    let mut times = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); 6];

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        assert_eq!(parts.len(), 7, "expected 7 columns in fixture row");

        let t: f64 = parts[0]
            .trim_matches('"')
            .parse()
            .expect("invalid time value");
        times.push(t);

        for j in 0..6 {
            let v: f64 = parts[j + 1]
                .trim_matches('"')
                .parse()
                .expect("invalid value");
            cols[j].push(v);
        }
    }

    (Array1::from(times), cols)
}

/// Cross-validation of block-wise regressor_design with HRF_SPMG3 (multi-basis)
/// against the R fixture.
#[test]
fn test_regressor_design_spmg3_block_example_parity() {
    let (times, cols) =
        load_spmg3_design_fixture("tests/fixtures/regressor_design_spmg3_block_example.csv");

    // Same sampling frame and block-wise design as the scalar example, but
    // with a multi-basis HRF (SPMG3).
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

    // Use the analytic SPMG3 HRF (matches HRF_SPMG3).
    let hrf = Box::new(SpmG3Hrf::new(5.0, 15.0, 1.0 / 120.0));

    let run_ids: Vec<_> = block.iter().map(|b| b - 1).collect();
    let basis_major = ConditionEvents::new(&onsets, &fac, &run_ids)
        .evaluate(&sframe, hrf.as_ref())
        .unwrap()
        .data;
    let design = basis_major.select(ndarray::Axis(1), &[0, 2, 4, 1, 3, 5]);

    assert_eq!(design.nrows(), times.len());
    assert_eq!(design.ncols(), 6);

    // Compare each column against the R fixture.
    for (col_idx, col_r) in cols.iter().enumerate() {
        for i in 0..times.len() {
            let r_val = col_r[i];
            let rs_val = design[[i, col_idx]];
            assert_relative_eq!(rs_val, r_val, epsilon = 1e-10, max_relative = 1e-8);
        }
    }
}
