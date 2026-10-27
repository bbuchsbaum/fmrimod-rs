//! Vignette 2: Building fMRI Regressors
//!
//! This example mirrors the R vignette "Building fMRI Regressors"
//! from the fmrihrf package. It demonstrates:
//!
//! - Creating regressors from event onsets
//! - Evaluating regressors on a time grid
//! - Varying event durations and summation
//! - Parametric modulation (amplitude scaling)
//! - Multi-basis regressors (B-spline, SPMG3)
//! - Shifting regressors in time
//!
//! Run with: cargo run --example vignette_02_regressor

use fmrihrf::hrf::*;
use fmrihrf::regressor::*;
use ndarray::Array1;

fn main() {
    println!("=== Vignette 2: Building fMRI Regressors ===\n");

    // ------------------------------------------------------------------
    // Basic regressor from event onsets
    // ------------------------------------------------------------------
    println!("## Basic Regressor from Event Onsets\n");

    let onsets: Vec<f64> = (0..11).map(|i| i as f64 * 12.0).collect();
    let hrf = spm_canonical(None, None, None, None, None, None);

    let reg = regressor(onsets.clone(), Some(hrf), None, None, None, None).unwrap();
    println!(
        "  Regressor: {} events, nbasis={}, span={}",
        reg.onsets.len(),
        reg.hrf.nbasis(),
        reg.span
    );
    println!("  Onsets: {:?}\n", &reg.onsets[..5]);

    // ------------------------------------------------------------------
    // Evaluating a regressor
    // ------------------------------------------------------------------
    println!("## Evaluating a Regressor\n");

    let tr = 2.0;
    let scan_times = Array1::from_vec((0..71).map(|i| i as f64 * tr).collect::<Vec<_>>());

    let pred = reg.evaluate(&scan_times, Some(0.33), Some("conv")).unwrap();
    println!(
        "  Grid: {} time points (TR={}s, 0 to {}s)",
        scan_times.len(),
        tr,
        scan_times[scan_times.len() - 1]
    );
    println!("  Output shape: {:?}", pred.shape());

    let max_val = pred
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_idx = pred
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!(
        "  Peak response: {:.4} at t={:.0}s\n",
        max_val, scan_times[max_idx]
    );

    // ------------------------------------------------------------------
    // Varying event durations
    // ------------------------------------------------------------------
    println!("## Varying Event Durations\n");

    let onsets_dur: Vec<f64> = (0..6).map(|i| i as f64 * 12.0).collect();
    let durations: Vec<f64> = (1..=6).map(|i| i as f64).collect();
    let hrf_dur = spm_canonical(None, None, None, None, None, None);

    let reg_dur = regressor(
        onsets_dur.clone(),
        Some(hrf_dur),
        Some(durations.clone()),
        None,
        None,
        None,
    )
    .unwrap();
    println!("  Events with durations: {:?}", durations);

    let grid_dur = Array1::from_vec((0..51).map(|i| i as f64 * tr).collect::<Vec<_>>());
    let pred_dur = reg_dur.evaluate(&grid_dur, None, None).unwrap();
    println!("  Output shape: {:?}", pred_dur.shape());

    // Show peak amplitudes increasing with duration
    for (i, &onset) in onsets_dur.iter().enumerate() {
        let window_start = (onset / tr) as usize;
        let window_end = ((onset + 15.0) / tr).min(grid_dur.len() as f64 - 1.0) as usize;
        let local_max = pred_dur.column(0).as_slice().unwrap()[window_start..=window_end]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!(
            "    Event {} (dur={}s): local peak = {:.4}",
            i + 1,
            durations[i],
            local_max
        );
    }

    // Duration with summate=false
    let hrf_nosum = spm_canonical(None, None, None, None, None, None);
    let reg_nosum = regressor(
        onsets_dur.clone(),
        Some(hrf_nosum),
        Some(durations.clone()),
        None,
        None,
        Some(false),
    )
    .unwrap();
    let pred_nosum = reg_nosum.evaluate(&grid_dur, None, None).unwrap();
    let max_sum = pred_dur
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_nosum = pred_nosum
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!("\n  summate=true max:  {:.4}", max_sum);
    println!("  summate=false max: {:.4}\n", max_nosum);

    // ------------------------------------------------------------------
    // Parametric modulation (varying amplitudes)
    // ------------------------------------------------------------------
    println!("## Parametric Modulation (Varying Amplitudes)\n");

    let onsets_amp: Vec<f64> = (0..11).map(|i| i as f64 * 12.0).collect();
    let amplitudes: Vec<f64> = (1..=11).map(|i| i as f64 - 6.0).collect(); // centered: -5..+5

    let hrf_amp = spm_canonical(None, None, None, None, None, None);
    let reg_amp = regressor(
        onsets_amp,
        Some(hrf_amp),
        None,
        Some(amplitudes.clone()),
        None,
        None,
    )
    .unwrap();

    println!("  Amplitudes (centered): {:?}", amplitudes);

    let grid_amp = Array1::from_vec((0..76).map(|i| i as f64 * tr).collect::<Vec<_>>());
    let pred_amp = reg_amp.evaluate(&grid_amp, None, None).unwrap();
    let amp_max = pred_amp
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let amp_min = pred_amp
        .column(0)
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!(
        "  Response range: [{:.4}, {:.4}]  (negative for negative amplitudes)\n",
        amp_min, amp_max
    );

    // ------------------------------------------------------------------
    // Multi-basis regressors
    // ------------------------------------------------------------------
    println!("## Regressors with HRF Basis Sets\n");

    let onsets_bs = vec![10.0, 30.0, 50.0, 70.0, 90.0];
    let bs_hrf = bspline_hrf(5, None, None);
    println!("  B-spline HRF: nbasis={}", bs_hrf.nbasis());

    let reg_bs = regressor(onsets_bs, Some(bs_hrf), None, None, None, None).unwrap();
    let grid_bs = Array1::linspace(0.0, 120.0, 61);
    let pred_bs = reg_bs.evaluate(&grid_bs, None, None).unwrap();

    println!(
        "  Output shape: {:?}  (time_points x n_basis)",
        pred_bs.shape()
    );
    for col in 0..pred_bs.ncols() {
        let col_max = pred_bs
            .column(col)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("    Basis {}: peak = {:.4}", col + 1, col_max);
    }
    println!();

    // ------------------------------------------------------------------
    // Shifting regressors
    // ------------------------------------------------------------------
    println!("## Shifting Regressors\n");

    let hrf_shift = spm_canonical(None, None, None, None, None, None);
    let reg_orig = regressor(
        vec![10.0, 30.0, 50.0],
        Some(hrf_shift),
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let reg_shifted = reg_orig.shift(5.0);

    println!("  Original onsets:  {:?}", reg_orig.onsets);
    println!("  Shifted (+5s):    {:?}", reg_shifted.onsets);

    let grid_shift = Array1::linspace(0.0, 80.0, 41);
    let pred_orig = reg_orig.evaluate(&grid_shift, None, None).unwrap();
    let pred_shift = reg_shifted.evaluate(&grid_shift, None, None).unwrap();

    let orig_peak = pred_orig
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let shift_peak = pred_shift
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("  Original first peak at t={:.0}s", grid_shift[orig_peak]);
    println!(
        "  Shifted first peak at t={:.0}s  (delayed by ~5s)",
        grid_shift[shift_peak]
    );

    println!("\n=== Done ===");
}
