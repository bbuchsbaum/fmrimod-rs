//! Vignette 4: Advanced HRF Modeling and Design
//!
//! This example mirrors the R vignette "Advanced HRF Modeling and Design"
//! from the fmrihrf package. It demonstrates:
//!
//! - HRF libraries (systematic parameter exploration)
//! - Reconstruction matrices (from coefficients to HRF shapes)
//! - Regressor sets (multi-condition designs)
//! - Regressor design (complex block designs with sampling frames)
//!
//! Run with: cargo run --example vignette_04_advanced

use fmrihrf::hrf::*;
use fmrihrf::penalty::penalty_matrix;
use fmrihrf::reconstruction::{reconstruct_hrf, reconstruction_matrix};
use fmrihrf::regressor::*;
use fmrihrf::sampling::SamplingFrame;
use ndarray::Array1;

fn main() {
    println!("=== Vignette 4: Advanced HRF Modeling and Design ===\n");

    // ------------------------------------------------------------------
    // HRF Libraries: Systematic Parameter Exploration
    // ------------------------------------------------------------------
    println!("## HRF Libraries: Parameter Exploration\n");

    // Create a library of gamma HRFs with different parameters
    let shapes = [4.0, 6.0, 8.0];
    let rates = [0.8, 1.0, 1.2];
    let t = Array1::linspace(0.0, 20.0, 201);

    println!("  Gamma HRF library (shape x rate):");
    println!(
        "  {:>8} {:>8} {:>10} {:>10}",
        "shape", "rate", "peak_time", "peak_val"
    );
    for &shape in &shapes {
        for &rate in &rates {
            let hrf = gamma_hrf(shape, 1.0 / rate, None);
            let vals = hrf.evaluate(&t).unwrap();
            let peak_idx = vals
                .column(0)
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;
            println!(
                "  {:8.0} {:8.1} {:10.1} {:10.4}",
                shape,
                rate,
                t[peak_idx],
                vals[[peak_idx, 0]]
            );
        }
    }

    // Library of lagged SPM HRFs
    println!("\n  Lagged SPM library:");
    println!("  {:>6} {:>10}", "lag", "peak_time");
    for lag in -2..=4 {
        let base = spm_canonical(None, None, None, None, None, None);
        let lagged = lag_hrf(base, lag as f64);
        let vals = lagged.evaluate(&t).unwrap();
        let peak_idx = vals
            .column(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!("  {:6.0} {:10.1}", lag, t[peak_idx]);
    }
    println!();

    // ------------------------------------------------------------------
    // Reconstruction Matrices
    // ------------------------------------------------------------------
    println!("## Reconstruction Matrices\n");

    // Use a B-spline basis
    let bs5 = bspline_hrf(5, None, Some(30.0));
    let recon_times = Array1::linspace(0.0, 30.0, 301);
    let r_matrix = reconstruction_matrix(&*bs5, &recon_times).unwrap();
    println!(
        "  B-spline(5) reconstruction matrix: {:?}",
        r_matrix.shape()
    );
    println!("  (301 time points x 5 basis functions)\n");

    // Reconstruct an HRF from coefficients
    // Coefficients emphasize the middle basis functions (peaked response)
    let coeffs = Array1::from_vec(vec![0.1, 0.5, 1.0, 0.5, 0.1]);
    let reconstructed = reconstruct_hrf(&r_matrix, &coeffs).unwrap();
    let recon_peak_idx = reconstructed
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("  Coefficients: {:?}", coeffs.as_slice().unwrap());
    println!(
        "  Reconstructed HRF peak at t={:.1}s, value={:.4}",
        recon_times[recon_peak_idx], reconstructed[recon_peak_idx]
    );

    // Show how different coefficients change the shape
    let coeff_sets = vec![
        ("Early peak", vec![1.0, 0.5, 0.0, 0.0, 0.0]),
        ("Late peak", vec![0.0, 0.0, 0.0, 0.5, 1.0]),
        ("Broad", vec![0.5, 0.8, 1.0, 0.8, 0.5]),
        ("Biphasic", vec![1.0, 0.3, -0.5, 0.0, 0.0]),
    ];
    println!("\n  Varying coefficients:");
    for (label, c) in &coeff_sets {
        let coeffs = Array1::from_vec(c.clone());
        let recon = reconstruct_hrf(&r_matrix, &coeffs).unwrap();
        let peak_idx = recon
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let min_idx = recon
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!(
            "    {:12}: peak at t={:.1}s ({:.3}), min at t={:.1}s ({:.3})",
            label, recon_times[peak_idx], recon[peak_idx], recon_times[min_idx], recon[min_idx]
        );
    }
    println!();

    // ------------------------------------------------------------------
    // Penalty Matrices
    // ------------------------------------------------------------------
    println!("## Penalty Matrices for Regularization\n");

    let pen = penalty_matrix(&*bs5, 2, None).unwrap();
    println!("  B-spline(5) penalty matrix (order 2): {:?}", pen.shape());
    println!(
        "  Diagonal: [{:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
        pen[[0, 0]],
        pen[[1, 1]],
        pen[[2, 2]],
        pen[[3, 3]],
        pen[[4, 4]]
    );
    println!();

    // ------------------------------------------------------------------
    // Regressor Sets: Multi-Condition Designs
    // ------------------------------------------------------------------
    println!("## Regressor Sets: Multi-Condition Designs\n");

    // Simulate a simple 2-condition experiment
    let onsets_a = vec![5.0, 25.0, 45.0, 65.0, 85.0];
    let onsets_b = vec![15.0, 35.0, 55.0, 75.0, 95.0];

    let hrf_a = spm_canonical(None, None, None, None, None, None);
    let hrf_b = spm_canonical(None, None, None, None, None, None);

    let reg_a = regressor(onsets_a, Some(hrf_a), None, None, None, None).unwrap();
    let reg_b = regressor(onsets_b, Some(hrf_b), None, None, None, None).unwrap();

    let grid = Array1::linspace(0.0, 120.0, 61); // TR=2s

    let pred_a = reg_a.evaluate(&grid, None, None).unwrap();
    let pred_b = reg_b.evaluate(&grid, None, None).unwrap();

    println!(
        "  Condition A: {} events, regressor shape {:?}",
        reg_a.onsets.len(),
        pred_a.shape()
    );
    println!(
        "  Condition B: {} events, regressor shape {:?}",
        reg_b.onsets.len(),
        pred_b.shape()
    );

    // Correlation between conditions
    let mean_a: f64 = pred_a.column(0).iter().sum::<f64>() / pred_a.nrows() as f64;
    let mean_b: f64 = pred_b.column(0).iter().sum::<f64>() / pred_b.nrows() as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..pred_a.nrows() {
        let da = pred_a[[i, 0]] - mean_a;
        let db = pred_b[[i, 0]] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let corr = cov / (var_a.sqrt() * var_b.sqrt());
    println!(
        "  Correlation between A and B: {:.3}  (alternating design reduces collinearity)\n",
        corr
    );

    // ------------------------------------------------------------------
    // Regressor Design with Sampling Frame
    // ------------------------------------------------------------------
    println!("## Regressor Design with Sampling Frame\n");

    // Create a sampling frame: 2 runs, 100 scans each, TR=2s
    let sframe = SamplingFrame::new(vec![100, 100], vec![2.0], None::<Vec<f64>>, 0.1).unwrap();

    println!(
        "  Sampling frame: {} blocks, {} total scans, TR=2s",
        sframe.n_blocks(),
        sframe.n_scans()
    );
    println!("  Block lengths: {:?}", sframe.blocklens);

    let samples = sframe.samples();
    println!(
        "  First 5 scan times: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
        samples[0], samples[1], samples[2], samples[3], samples[4]
    );
    println!("  Last scan time: {:.1}", samples[samples.len() - 1]);

    // Global onsets: convert block-relative to global times
    // Block 1 starts at t=0, block 2 starts after block 1 ends
    let block1_start = 0.0;
    let block2_start = sframe.blocklens[0] as f64 * sframe.tr[0];
    let block1_onsets = vec![10.0, 30.0, 50.0];
    let block2_onsets: Vec<f64> = vec![10.0, 40.0, 70.0]
        .iter()
        .map(|&t| t + block2_start)
        .collect();
    let global: Vec<f64> = [
        block1_onsets
            .iter()
            .map(|&t| t + block1_start)
            .collect::<Vec<_>>(),
        block2_onsets.clone(),
    ]
    .concat();
    println!("\n  Block 2 starts at t={:.0}s", block2_start);
    println!("  Global onsets: {:?}", global);

    // Build design matrix from global onsets
    let hrf_design = spm_canonical(None, None, None, None, None, None);
    let reg_design = regressor(global, Some(hrf_design), None, None, None, None).unwrap();
    let grid_design = Array1::from_vec(samples);
    let design = reg_design
        .evaluate(&grid_design, Some(0.33), Some("conv"))
        .unwrap();
    println!(
        "  Design matrix shape: {:?}  (scans x basis_functions)",
        design.shape()
    );

    println!("\n=== Done ===");
}
