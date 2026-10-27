//! Comprehensive demonstration of fmri_hrf-rs functionality.
//!
//! Run with: cargo run --example comprehensive_demo

use fmri_hrf::hrf::{
    bspline_hrf, fir_hrf, fourier_hrf, gamma_hrf, get_hrf, spm_canonical, spm_hrf_with_derivatives,
    HrfExt, HrfGenerator,
};
use fmri_hrf::penalty::{penalty_matrix, PenaltyOptions};
use fmri_hrf::reconstruction::{reconstruct_hrf, reconstruction_matrix};
use fmri_hrf::regressor::{
    design_from_regressor_set, design_matrix_from_set, regressor, regressor_set, DesignOptions,
};
use fmri_hrf::sampling::{SamplingFrame, SamplingFrameBuilder};
use ndarray::Array1;

fn main() {
    println!("=== fmri_hrf-rs Comprehensive Demo ===\n");

    demo_basic_hrf();
    demo_hrf_types();
    demo_hrf_generators();
    demo_sampling_frames();
    demo_regressors();
    demo_design_matrices();
    demo_penalty_matrices();
    demo_reconstruction();
}

fn demo_basic_hrf() {
    println!("1. Basic HRF Evaluation");
    println!("-----------------------");

    // Create canonical HRF
    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    // Evaluate at time points
    let times = Array1::linspace(0.0, 20.0, 21);
    let values = hrf.evaluate(&times).unwrap();

    // Find peak
    let peak_idx = values
        .column(0)
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;

    println!("  HRF name: {}", hrf.name());
    println!("  Number of basis functions: {}", hrf.nbasis());
    println!("  Temporal span: {} seconds", hrf.span());
    println!("  Peak time: {} seconds", times[peak_idx]);
    println!("  Peak value: {:.4}", values[[peak_idx, 0]]);

    // Evaluate with block duration
    let block_values = hrf.evaluate_block(&times, 3.0, 0.1, true, false).unwrap();
    println!(
        "  Block (3s) peak value: {:.4}",
        block_values.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!();
}

fn demo_hrf_types() {
    println!("2. Different HRF Types");
    println!("----------------------");

    // SPM with derivatives
    let spm_deriv = spm_hrf_with_derivatives();
    println!(
        "  SPM with derivatives: {} basis functions",
        spm_deriv.nbasis()
    );

    // Gamma HRF
    let gamma = gamma_hrf(6.0, 1.0, Some(true));
    println!("  Gamma HRF: peak at {} seconds", gamma.params()[0].1);

    // B-spline basis
    let bspline = bspline_hrf(6, Some(3), Some(20.0));
    println!(
        "  B-spline: {} basis functions, order {}",
        bspline.nbasis(),
        bspline.params()[0].1
    );

    // Fourier basis
    let fourier = fourier_hrf(7, Some(20.0));
    println!("  Fourier: {} basis functions", fourier.nbasis());

    // FIR basis
    let fir = fir_hrf(10, Some(20.0));
    println!("  FIR: {} time bins", fir.nbasis());

    // Get HRF from registry
    let registered = get_hrf("gaussian").unwrap();
    println!("  Registry lookup 'gaussian': {}", registered.name());
    println!();
}

fn demo_hrf_generators() {
    println!("3. HRF Generators");
    println!("-----------------");

    let base_hrf1 = spm_canonical(None, None, None, None, None, Some(true));
    let base_hrf2 = spm_canonical(None, None, None, None, None, Some(true));
    let base_hrf3 = spm_canonical(None, None, None, None, None, Some(true));

    // Create lagged HRF
    let lagged = HrfGenerator::new(base_hrf1).with_lag(2.0).build();

    // Create blocked HRF
    let blocked = HrfGenerator::new(base_hrf2).with_block(3.0).build();

    // Create lagged, blocked, and normalized HRF
    let complex = HrfGenerator::new(base_hrf3)
        .with_lag(1.0)
        .with_block(2.0)
        .normalized()
        .build();

    let _times = Array1::linspace(0.0, 20.0, 201);

    println!("  Lagged HRF: {} lag", lagged.name());
    println!("  Blocked HRF: {} block", blocked.name());
    println!("  Complex HRF: {}", complex.name());
    println!();
}

fn demo_sampling_frames() {
    println!("4. Sampling Frames");
    println!("------------------");

    // Uniform sampling
    let uniform = SamplingFrame::uniform(100, 2.0, 0.1).unwrap();
    println!("  Uniform: {} samples, TR={}", uniform.samples().len(), 2.0);

    // Multi-run with different TRs
    let multi_run = SamplingFrameBuilder::default()
        .blocklens(vec![50, 50, 50])
        .tr(vec![2.0, 2.5, 2.0])
        .start_time(vec![0.0, 0.0, 0.0])
        .build()
        .unwrap();

    println!("  Multi-run: {} total samples", multi_run.samples().len());
    println!("  Run lengths: {:?}", multi_run.blocklens);
    println!("  TRs: {:?}", multi_run.tr);

    // Get sample times
    let samples = multi_run.samples();
    println!("  First 5 samples: {:?}", &samples[..5]);
    println!("  Last 5 samples: {:?}", &samples[samples.len() - 5..]);
    println!();
}

fn demo_regressors() {
    println!("5. Event Regressors");
    println!("-------------------");

    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    // Simple impulse regressor
    let hrf1 = spm_canonical(None, None, None, None, None, Some(true));
    let impulse_reg = regressor(
        vec![10.0, 30.0, 50.0],
        Some(hrf1),
        None, // impulse events
        None, // default amplitude
        None, // default span
        None, // default summate
    )
    .unwrap();

    // Block regressor with varying durations
    let hrf2 = spm_canonical(None, None, None, None, None, Some(true));
    let block_reg = regressor(
        vec![10.0, 30.0, 50.0],
        Some(hrf2),
        Some(vec![2.0, 4.0, 3.0]), // different durations
        Some(vec![1.0, 2.0, 1.5]), // different amplitudes
        None,
        None,
    )
    .unwrap();

    // Multi-condition regressor set
    let rset = regressor_set(
        vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        vec!["face", "house", "face", "house", "face", "house"]
            .into_iter()
            .map(|s| s.to_string())
            .collect(),
        Some(hrf),
        Some(vec![1.0]), // 1s duration for all
        None,
        None,
        None,
    )
    .unwrap();

    println!("  Impulse regressor: {} events", impulse_reg.onsets.len());
    println!("  Block regressor: durations {:?}", block_reg.durations);
    println!("  Regressor set: {} conditions", rset.levels.len());
    println!("  Conditions: {:?}", rset.levels);

    // Evaluate regressors
    let grid = Array1::linspace(0.0, 60.0, 601);
    let impulse_result = impulse_reg.evaluate(&grid, None, None).unwrap();
    let block_result = block_reg.evaluate(&grid, None, None).unwrap();

    println!(
        "  Impulse response max: {:.4}",
        impulse_result.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!(
        "  Block response max: {:.4}",
        block_result.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );
    println!();
}

fn demo_design_matrices() {
    println!("6. Design Matrices");
    println!("------------------");

    let hrf = spm_canonical(None, None, None, None, None, Some(true));

    // Create regressor set
    let onsets = vec![10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0];
    let conditions = vec!["A", "B", "C", "A", "B", "C", "A", "B"];
    let rset = regressor_set(
        onsets,
        conditions.into_iter().map(|s| s.to_string()).collect(),
        Some(hrf),
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // Create design matrix
    let grid = Array1::linspace(0.0, 60.0, 301);
    let design = design_from_regressor_set(&rset, &grid, None).unwrap();

    println!("  Design matrix shape: {:?}", design.shape());
    println!("  Conditions: {:?}", rset.levels);

    // With intercept
    let options = DesignOptions::builder()
        .include_intercept(true)
        .precision(0.1)
        .build();

    let design_with_intercept = design_from_regressor_set(&rset, &grid, Some(options)).unwrap();
    println!(
        "  With intercept shape: {:?}",
        design_with_intercept.shape()
    );

    // Create DesignMatrix with names
    let dm = design_matrix_from_set(&rset, &grid, None).unwrap();
    println!("  Column names: {:?}", dm.column_names);

    // Access columns
    if let Some(col_a) = dm.column_by_name("A") {
        println!("  Column 'A' sum: {:.4}", col_a.sum());
    }
    println!();
}

fn demo_penalty_matrices() {
    println!("7. Penalty Matrices");
    println!("-------------------");

    // B-spline penalty
    let bspline = bspline_hrf(8, Some(3), Some(20.0));
    let bspline_penalty = penalty_matrix(bspline.as_ref(), 2, None).unwrap();

    println!("  B-spline penalty shape: {:?}", bspline_penalty.shape());
    println!(
        "  B-spline penalty rank: ~{}",
        bspline_penalty
            .diag()
            .iter()
            .filter(|&&x| x.abs() > 1e-10)
            .count()
    );

    // Fourier penalty
    let fourier = fourier_hrf(9, Some(20.0));
    let fourier_penalty = penalty_matrix(fourier.as_ref(), 2, None).unwrap();

    println!("  Fourier penalty shape: {:?}", fourier_penalty.shape());

    // Custom penalty options
    let options = PenaltyOptions { shrink_deriv: 0.1 };

    let custom_penalty = penalty_matrix(bspline.as_ref(), 2, Some(options)).unwrap();
    println!("  Custom penalty trace: {:.4}", custom_penalty.diag().sum());
    println!();
}

fn demo_reconstruction() {
    println!("8. Reconstruction Matrices");
    println!("--------------------------");

    // Multi-basis HRF
    let hrf = spm_hrf_with_derivatives();
    let times = Array1::linspace(0.0, 20.0, 101);

    // Get reconstruction matrix
    let r_matrix = reconstruction_matrix(hrf.as_ref(), &times).unwrap();
    println!("  Reconstruction matrix shape: {:?}", r_matrix.shape());

    // Reconstruct with different coefficients
    let coeffs1 = Array1::from_vec(vec![1.0, 0.0, 0.0]); // Canonical only
    let coeffs2 = Array1::from_vec(vec![1.0, 0.5, 0.0]); // With temporal derivative
    let coeffs3 = Array1::from_vec(vec![1.0, 0.0, 0.5]); // With dispersion derivative

    let recon1 = reconstruct_hrf(&r_matrix, &coeffs1).unwrap();
    let recon2 = reconstruct_hrf(&r_matrix, &coeffs2).unwrap();
    let recon3 = reconstruct_hrf(&r_matrix, &coeffs3).unwrap();

    // Find peaks
    let peak1 = recon1
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| times[i])
        .unwrap();
    let peak2 = recon2
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| times[i])
        .unwrap();
    let peak3 = recon3
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| times[i])
        .unwrap();

    println!("  Canonical only peak: {} s", peak1);
    println!("  With temporal deriv peak: {} s", peak2);
    println!("  With dispersion deriv peak: {} s", peak3);
    println!();

    println!("Demo complete!");
}
