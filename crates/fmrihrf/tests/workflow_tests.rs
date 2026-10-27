//! Integration tests for common workflows.

use fmrihrf::{
    error::Result,
    hrf::{get_hrf, make_hrf, spm_canonical},
    regressor::{regressor, regressor_set},
    sampling::SamplingFrame,
};
use ndarray::Array1;

#[test]
fn test_basic_glm_workflow() -> Result<()> {
    // 1. Create a sampling frame
    let sampling = SamplingFrame::uniform(200, 2.0, 0.1)?;

    // 2. Create event onsets for two conditions
    let condition_a_onsets = vec![10.0, 50.0, 90.0, 130.0, 170.0];
    let condition_b_onsets = vec![30.0, 70.0, 110.0, 150.0];

    // 3. Create regressors with canonical HRF
    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    let reg_a = regressor(condition_a_onsets, Some(hrf), None, None, None, None)?;

    let hrf2 = spm_canonical(None, None, None, None, None, Some(true));
    let reg_b = regressor(condition_b_onsets, Some(hrf2), None, None, None, None)?;

    // 4. Generate design matrix
    let time_points = sampling.samples();
    let time_array = Array1::from_vec(time_points);
    let dm_a = reg_a.evaluate(&time_array, Some(0.1), None)?;
    let dm_b = reg_b.evaluate(&time_array, Some(0.1), None)?;

    // 5. Verify design matrix properties
    assert_eq!(dm_a.nrows(), 200);
    assert_eq!(dm_a.ncols(), 1); // Single basis function
    assert_eq!(dm_b.nrows(), 200);
    assert_eq!(dm_b.ncols(), 1);

    // Check that responses peak after stimulus onset
    let peak_indices_a: Vec<_> = (0..200)
        .filter(|&i| {
            i > 0 && i < 199 && dm_a[[i, 0]] > dm_a[[i - 1, 0]] && dm_a[[i, 0]] > dm_a[[i + 1, 0]]
        })
        .collect();

    // Should have peaks near each event (with some delay)
    assert!(!peak_indices_a.is_empty());

    Ok(())
}

#[test]
fn test_regressor_set_workflow() -> Result<()> {
    // Create a mixed design with multiple conditions
    let all_onsets = vec![
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
        80.0, 85.0, 90.0, 95.0, 100.0,
    ];

    let conditions = vec![
        "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C",
        "A", "B",
    ];

    let condition_labels: Vec<String> = conditions.iter().map(|&s| s.to_string()).collect();

    // Create regressor set
    let hrf = get_hrf("spm_canonical")?;
    let rset = regressor_set(
        all_onsets,
        condition_labels,
        Some(hrf),
        None,
        None,
        None,
        None,
    )?;

    // Verify structure
    assert_eq!(rset.levels.len(), 3); // A, B, C
    assert_eq!(rset.regressors.len(), 3);

    // Each condition should have the correct number of events
    let events_per_condition = vec![7, 7, 6]; // A:7, B:7, C:6
    for (i, (reg, &expected)) in rset
        .regressors
        .iter()
        .zip(events_per_condition.iter())
        .enumerate()
    {
        assert_eq!(
            reg.onsets.len(),
            expected,
            "Condition {} should have {} events",
            rset.levels[i],
            expected
        );
    }

    // Create design matrix
    let time_grid = Array1::linspace(0.0, 110.0, 220);
    let design = rset.evaluate(&time_grid, Some(0.1), None)?;

    assert_eq!(design.nrows(), 220);
    assert_eq!(design.ncols(), 3); // Three conditions

    Ok(())
}

#[test]
fn test_multi_basis_hrf_workflow() -> Result<()> {
    // Use HRF with derivatives
    let hrf = get_hrf("spm_with_derivatives")?;
    assert_eq!(hrf.nbasis(), 3); // Main + temporal derivative + dispersion derivative

    let onsets = vec![10.0, 30.0, 50.0];
    let reg = regressor(
        onsets,
        Some(hrf),
        Some(vec![2.0]), // 2 second duration
        None,
        None,
        None,
    )?;

    let time_grid = Array1::linspace(0.0, 60.0, 300);
    let dm = reg.evaluate(&time_grid, Some(0.1), None)?;

    assert_eq!(dm.nrows(), 300);
    assert_eq!(dm.ncols(), 3); // Three basis functions

    // First column should be the main response
    // Second column should be the derivative
    // The derivative should cross zero near the main response peak

    Ok(())
}

#[test]
fn test_string_based_hrf_creation() -> Result<()> {
    // Test various string formats
    let hrf1 = make_hrf("spm_canonical")?;
    assert_eq!(hrf1.nbasis(), 1);

    let hrf2 = make_hrf("bspline(N=7,degree=3)")?;
    assert_eq!(hrf2.nbasis(), 7);

    let hrf3 = make_hrf("gamma(shape=6.0,rate=1.0)")?;
    assert_eq!(hrf3.nbasis(), 1);

    // Test with FIR HRF (multi-basis)
    let hrf4 = make_hrf("fir(N=10,span=20.0)")?;
    assert_eq!(hrf4.nbasis(), 10);

    // Evaluate to ensure they work
    let t = Array1::linspace(0.0, 20.0, 100);
    let vals1 = hrf1.evaluate(&t)?;
    let vals2 = hrf2.evaluate(&t)?;
    let vals3 = hrf3.evaluate(&t)?;
    let vals4 = hrf4.evaluate(&t)?;

    assert_eq!(vals1.ncols(), 1);
    assert_eq!(vals2.ncols(), 7);
    assert_eq!(vals3.ncols(), 1);
    assert_eq!(vals4.ncols(), 10);

    Ok(())
}

#[test]
fn test_block_design_workflow() -> Result<()> {
    // Create a block design with extended stimulus duration
    let block_onsets = vec![20.0, 80.0, 140.0];
    let block_duration = 20.0; // 20 second blocks

    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    let reg = regressor(
        block_onsets.clone(),
        Some(hrf),
        Some(vec![block_duration]), // Block duration
        None,
        None,
        None,
    )?;

    let time_grid = Array1::linspace(0.0, 200.0, 400);
    let dm = reg.evaluate(&time_grid, Some(0.1), None)?;

    // For block designs, the response should be sustained during the block
    // Check that the response is elevated during block periods
    for (i, &onset) in block_onsets.iter().enumerate() {
        // Sample points during the block (offset by HRF delay ~5s)
        let _block_start_idx = ((onset + 5.0) * 2.0) as usize; // 2 Hz sampling
        let block_middle_idx = ((onset + 15.0) * 2.0) as usize;

        if block_middle_idx < dm.nrows() {
            // Response should be sustained during the block
            assert!(
                dm[[block_middle_idx, 0]] > 0.5,
                "Block {} response should be elevated during stimulation",
                i
            );
        }
    }

    Ok(())
}

#[test]
fn test_concatenated_runs_workflow() -> Result<()> {
    // Simulate multi-run experiment
    let run1 = SamplingFrame::new(vec![100], vec![2.0], None::<Vec<f64>>, 0.1)?;
    let run2 = SamplingFrame::new(vec![100], vec![2.0], None::<Vec<f64>>, 0.1)?;

    let combined = run1.concatenate(&run2)?;

    assert_eq!(combined.n_blocks(), 2);
    assert_eq!(combined.n_scans(), 200);

    // Second run should start after first run ends
    let samples = combined.samples();
    assert!(samples[100] > samples[99] + 2.0); // Gap between runs

    // Create regressors that span both runs
    let onsets = vec![
        // Run 1 events
        10.0, 30.0, 50.0, 70.0, 90.0, // Run 2 events (accounting for run offset)
        210.0, 230.0, 250.0, 270.0, 290.0,
    ];

    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let reg = regressor(onsets, Some(hrf), None, None, None, None)?;

    let samples_array = Array1::from_vec(samples);
    let dm = reg.evaluate(&samples_array, Some(0.1), None)?;
    assert_eq!(dm.nrows(), 200);

    Ok(())
}

#[test]
fn test_custom_hrf_workflow() -> Result<()> {
    use fmrihrf::hrf::empirical_hrf;

    // Create empirical HRF
    let t = Array1::linspace(0.0, 20.0, 100);
    let hrf_vals = (&t * (-&t / 4.0)).mapv(f64::exp) * 0.5; // Simple gamma-like

    let custom_hrf = empirical_hrf(&t, &hrf_vals, Some("my_custom_hrf"), Some(20.0))?;

    // Use the custom HRF in a regressor
    let onsets = vec![10.0, 30.0, 50.0];
    let reg = regressor(
        onsets,
        Some(custom_hrf),
        None,
        None,
        Some(20.0), // Specify span since it's empirical
        None,
    )?;

    let eval_grid = Array1::linspace(0.0, 70.0, 350);
    let dm = reg.evaluate(&eval_grid, Some(0.1), None)?;

    assert_eq!(dm.nrows(), 350);
    assert_eq!(dm.ncols(), 1);

    // Verify response characteristics
    // Should have peaks after each onset
    for &onset in &[10.0, 30.0, 50.0] {
        let peak_window_start = ((onset + 2.0) * 5.0) as usize; // Expecting peak 2-6s after onset
        let peak_window_end = (((onset + 6.0) * 5.0) as f64).min(349.0) as usize;

        let window_max = (peak_window_start..=peak_window_end)
            .map(|i| dm[[i, 0]])
            .fold(0.0, f64::max);

        assert!(
            window_max > 0.1,
            "Should have response peak after onset at {}",
            onset
        );
    }

    Ok(())
}
