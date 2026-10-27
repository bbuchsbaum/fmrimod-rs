//! Vignette 01: Introduction to fmridesign-rs
//! ============================================
//!
//! Port of R's `a_01_introduction.Rmd`.
//!
//! This vignette walks through the complete workflow for building an fMRI
//! design matrix in Rust, starting from raw experimental event data and
//! ending with a combined event + baseline model.
//!
//! Run with:
//!   cargo run --example vignette_01_introduction

use fmridesign::baseline::{baseline_model, BasisType, InterceptType, SamplingInfo};
use fmridesign::core::{BaselineModel as CoreBaselineModel, EventModel};
use fmridesign::events::{EventFactor, EventTerm, HrfEvent};
use fmridesign::hrf::HrfSpec;
use fmridesign::sampling::{multi_block_sampling_frame, uniform_sampling_frame};
use fmridesign::types::EventSpec;
use ndarray::{concatenate, Axis};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vignette 01: Introduction to fmridesign-rs ===\n");

    // ---------------------------------------------------------------
    // 1. Create a Sampling Frame
    // ---------------------------------------------------------------
    // In R: sampling_frame(blocklens = c(150, 150), TR = 2)
    //
    // A sampling frame describes the temporal grid of fMRI acquisition.
    // Here we simulate 2 runs of 150 scans each, TR = 2 seconds.
    println!("--- Step 1: Create a sampling frame ---");

    let sf = multi_block_sampling_frame(&[150, 150], &[2.0], None)?;

    println!("  Number of blocks (runs): {}", sf.n_blocks());
    println!("  Total scans:             {}", sf.n_scans());
    println!("  Block lengths:           {:?}", sf.blocklens);
    println!();

    // ---------------------------------------------------------------
    // 2. Define experimental events
    // ---------------------------------------------------------------
    // In R: event_factor("condition", onset, blocknum, levels = c("face","scene","object"))
    //
    // We create a simple categorical factor with 3 conditions.
    // Events are distributed across the 2 runs.
    println!("--- Step 2: Define experimental events ---");

    // Run 1 events (first 150 scans = 0..300s, sampling starts at TR/2)
    let mut all_onsets = Vec::new();
    let mut all_conditions = Vec::new();
    let mut all_durations = Vec::new();

    let conditions = ["face", "scene", "object"];

    // Generate 5 events per condition per run, spaced 18s apart
    for run in 0..2 {
        let run_offset = run as f64 * 300.0; // 150 scans * 2s = 300s per run
        for (cond_idx, &cond) in conditions.iter().enumerate() {
            for trial in 0..5 {
                let onset = run_offset + 10.0 + (cond_idx * 5 + trial) as f64 * 18.0;
                all_onsets.push(onset);
                all_conditions.push(cond);
                all_durations.push(1.0); // 1-second events (brief stimulus)
            }
        }
    }

    let n_events = all_onsets.len();
    println!("  Conditions:       {:?}", conditions);
    println!("  Events per run:   {}", n_events / 2);
    println!("  Total events:     {}", n_events);

    // Create the EventFactor
    let factor = EventFactor::new(
        "condition",
        all_onsets.clone(),
        all_conditions.clone(),
        all_durations.clone(),
    )?;

    println!("  Unique levels:    {:?}", factor.levels);
    println!(
        "  Face onsets:      {:?}",
        &factor.get_level_onsets("face")[..3]
    );
    println!(
        "  Scene onsets:     {:?}",
        &factor.get_level_onsets("scene")[..3]
    );
    println!();

    // ---------------------------------------------------------------
    // 3. Wrap in an EventTerm and build the EventModel
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition), data, block = ~blocknum, sampling_frame)
    //
    // An EventTerm wraps one or more EventSpecs. For a simple main-effect
    // design, we use a single factor inside a non-interaction term.
    println!("--- Step 3: Build the EventModel ---");

    let term = EventTerm::new(vec![Box::new(factor)], false)?;
    let model = EventModel::new(vec![Box::new(term)], sf.clone())?;

    println!("  {}", model);

    let col_names = model.column_names();
    println!("  Column names: {:?}", col_names);

    // Generate the design matrix
    let dm = model.generate_design_matrix()?;
    println!(
        "  Design matrix shape: {} rows x {} cols",
        dm.nrows(),
        dm.ncols()
    );

    // Show a few rows around the first event
    println!("\n  First 10 rows of the design matrix:");
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}",
        "row", &col_names[0], &col_names[1], &col_names[2]
    );
    for i in 0..10.min(dm.nrows()) {
        print!("  {:>6}", i);
        for j in 0..dm.ncols() {
            print!("  {:>8.4}", dm[[i, j]]);
        }
        println!();
    }
    println!();

    // ---------------------------------------------------------------
    // 3b. HRF Convolution
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition), ...)
    //
    // Wrapping an event in HrfEvent convolves the indicator matrix with a
    // hemodynamic response function. This is the key step that transforms
    // binary event indicators into predicted BOLD signal shapes.
    println!("--- Step 3b: Apply HRF convolution ---");

    let factor_for_hrf = EventFactor::new(
        "condition",
        all_onsets.clone(),
        all_conditions.clone(),
        all_durations.clone(),
    )?;

    // Wrap with canonical SPM HRF (equivalent to R's basis = "spmg1")
    let hrf_spec = HrfSpec::Named {
        name: "spmg1".to_string(),
        params: std::collections::HashMap::new(),
    };
    let hrf_event = HrfEvent::new(Box::new(factor_for_hrf), hrf_spec);

    let hrf_term = EventTerm::new(vec![Box::new(hrf_event)], false)?;
    let hrf_model = EventModel::new(vec![Box::new(hrf_term)], sf.clone())?;

    let hrf_dm = hrf_model.generate_design_matrix()?;
    let hrf_names = hrf_model.column_names();

    println!(
        "  HRF-convolved design: {} rows x {} cols",
        hrf_dm.nrows(),
        hrf_dm.ncols()
    );
    println!("  Columns: {:?}", hrf_names);

    // Show peak values (HRF-convolved regressors have smooth shapes)
    for (j, name) in hrf_names.iter().enumerate() {
        let col = hrf_dm.column(j);
        let max_val = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_val = col.iter().cloned().fold(f64::INFINITY, f64::min);
        println!(
            "    {:<20} range=[{:>8.4}, {:>8.4}]",
            name, min_val, max_val
        );
    }

    // Show a few rows around the first event onset (~10s = row 5 at TR=2)
    println!("\n  Rows 3-12 of HRF-convolved design (around first event):");
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}",
        "row", &hrf_names[0], &hrf_names[1], &hrf_names[2]
    );
    for i in 3..12.min(hrf_dm.nrows()) {
        print!("  {:>6}", i);
        for j in 0..hrf_dm.ncols() {
            print!("  {:>8.4}", hrf_dm[[i, j]]);
        }
        println!();
    }
    println!();

    // ---------------------------------------------------------------
    // 4. Create a Baseline Model
    // ---------------------------------------------------------------
    // In R: baseline_model(basis = "poly", degree = 3, sframe = sampling_frame)
    //
    // The baseline model captures low-frequency drift and run-wise intercepts.
    println!("--- Step 4: Create a baseline model ---");

    let sinfo = SamplingInfo::from_frame(sf.clone());

    let baseline = baseline_model(
        BasisType::Polynomial,
        3,
        &sinfo,
        InterceptType::Runwise,
        None,
    )?;

    let baseline_dm = baseline.design_matrix()?;
    let baseline_cols = baseline.column_names();

    println!(
        "  Baseline shape:   {} rows x {} cols",
        baseline_dm.nrows(),
        baseline_dm.ncols()
    );
    println!("  Baseline columns: {:?}", baseline_cols);
    println!();

    // ---------------------------------------------------------------
    // 5. Combine Event + Baseline into Full Design Matrix
    // ---------------------------------------------------------------
    // In R: the event_model() automatically includes the baseline, but here
    // we demonstrate explicit combination.
    println!("--- Step 5: Combine into full design matrix ---");

    let full_dm = concatenate(Axis(1), &[dm.view(), baseline_dm.view()])?;

    let mut full_cols = col_names.clone();
    full_cols.extend(baseline_cols.clone());

    println!(
        "  Full design matrix: {} rows x {} cols",
        full_dm.nrows(),
        full_dm.ncols()
    );
    println!("  All columns:");
    for (i, name) in full_cols.iter().enumerate() {
        println!("    [{:>2}] {}", i, name);
    }
    println!();

    // ---------------------------------------------------------------
    // 6. Inspect column properties
    // ---------------------------------------------------------------
    println!("--- Step 6: Column summary statistics ---");

    for (j, name) in full_cols.iter().enumerate() {
        let col = full_dm.column(j);
        let mean = col.mean().unwrap_or(0.0);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let nonzero = col.iter().filter(|&&v| v.abs() > 1e-10).count();
        println!(
            "  {:<30} mean={:>8.4}  min={:>8.4}  max={:>8.4}  nonzero={:>4}",
            name, mean, min, max, nonzero
        );
    }
    println!();

    // ---------------------------------------------------------------
    // 7. Alternative: Use core::DesignMatrix to combine automatically
    // ---------------------------------------------------------------
    // The DesignMatrix wrapper in core.rs combines event + baseline.
    println!("--- Step 7: Using core::DesignMatrix for automatic combination ---");

    // Rebuild the event model
    let factor2 = EventFactor::new("condition", all_onsets, all_conditions, all_durations)?;
    let term2 = EventTerm::new(vec![Box::new(factor2)], false)?;
    let model2 = EventModel::new(vec![Box::new(term2)], sf.clone())?;

    // Use core::BaselineModel (simpler polynomial wrapper)
    let core_baseline = CoreBaselineModel::polynomial(3);
    println!("  Core baseline: {}", core_baseline);

    // Combine via core::DesignMatrix
    let mut combined = fmridesign::core::DesignMatrix::from_event_model(model2);
    let combined_dm = combined.with_baseline(core_baseline);

    println!("  Combined: {}", combined_dm);
    println!();

    println!("=== Vignette 01 complete ===");
    Ok(())
}
