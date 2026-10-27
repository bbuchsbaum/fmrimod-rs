//! Vignette 1: Hemodynamic Response Functions
//!
//! This example mirrors the R vignette "Hemodynamic Response Functions"
//! from the fmri_hrf package. It demonstrates:
//!
//! - Pre-defined HRF objects (SPM canonical, Gaussian, etc.)
//! - Modifying HRF parameters
//! - Block/sustained stimulus modeling
//! - Temporal lag decorators
//! - Multi-basis HRF sets (SPM derivatives, B-splines, sine)
//! - Other HRF shapes (gamma, Mexican hat, inverse logit)
//! - Boxcar and weighted HRFs
//! - Empirical HRFs
//!
//! Run with: cargo run --example vignette_01_hrf

use fmri_hrf::hrf::*;
use ndarray::Array1;

fn main() {
    println!("=== Vignette 1: Hemodynamic Response Functions ===\n");

    // ------------------------------------------------------------------
    // Pre-defined HRF objects
    // ------------------------------------------------------------------
    println!("## Pre-defined HRF Objects\n");

    let spmg1 = spm_canonical(None, None, None, None, None, None);
    let gauss = gaussian_hrf(6.0, 2.0, None);

    println!(
        "SPM Canonical: name={}, nbasis={}, span={}",
        spmg1.name(),
        spmg1.nbasis(),
        spmg1.span()
    );
    println!(
        "Gaussian:      name={}, nbasis={}, span={}",
        gauss.name(),
        gauss.nbasis(),
        gauss.span()
    );

    let t = Array1::linspace(0.0, 25.0, 251);
    let spm_vals = spmg1.evaluate(&t).unwrap();
    let gauss_vals = gauss.evaluate(&t).unwrap();

    // Find peaks
    let spm_peak_idx = spm_vals
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let gauss_peak_idx = gauss_vals
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    println!(
        "SPM peak at t={:.1}s, Gaussian peak at t={:.1}s\n",
        t[spm_peak_idx], t[gauss_peak_idx]
    );

    // ------------------------------------------------------------------
    // Modifying HRF parameters with gen_hrf
    // ------------------------------------------------------------------
    println!("## Modifying HRF Parameters\n");

    let gauss_7_3 = gaussian_hrf(7.0, 3.0, None);
    let gauss_5_2 = gaussian_hrf(5.0, 2.0, None);
    let gauss_4_1 = gaussian_hrf(4.0, 1.0, None);

    for (label, hrf) in [
        ("mean=7,sd=3", &gauss_7_3),
        ("mean=5,sd=2", &gauss_5_2),
        ("mean=4,sd=1", &gauss_4_1),
    ] {
        let vals = hrf.evaluate(&t).unwrap();
        let peak_idx = vals
            .column(0)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        println!("  Gaussian({}): peak at t={:.1}s", label, t[peak_idx]);
    }
    println!();

    // ------------------------------------------------------------------
    // Block HRF — modeling sustained stimuli
    // ------------------------------------------------------------------
    println!("## Modeling Event Duration with block_hrf\n");

    let base = spm_canonical(None, None, None, None, None, None);
    let blocked_3s = block_hrf(base, 3.0);

    let base2 = spm_canonical(None, None, None, None, None, None);
    let blocked_10s = block_hrf(base2, 10.0);

    let bvals_3 = blocked_3s.evaluate(&t).unwrap();
    let bvals_10 = blocked_10s.evaluate(&t).unwrap();

    let peak_3 = bvals_3
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let peak_10 = bvals_10
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    println!("  Block 3s peak amplitude:  {:.4}", peak_3);
    println!(
        "  Block 10s peak amplitude: {:.4}  (larger due to summation)\n",
        peak_10
    );

    // ------------------------------------------------------------------
    // Lag decorator
    // ------------------------------------------------------------------
    println!("## Modeling Temporal Shifts with lag_hrf\n");

    let base_lag = spm_canonical(None, None, None, None, None, None);
    let lagged = lag_hrf(base_lag, 3.0);

    let lag_vals = lagged.evaluate(&t).unwrap();
    let lag_peak_idx = lag_vals
        .column(0)
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    println!("  Original SPM peak at t={:.1}s", t[spm_peak_idx]);
    println!(
        "  Lagged (3s) peak at t={:.1}s  (shifted ~3s later)\n",
        t[lag_peak_idx]
    );

    // ------------------------------------------------------------------
    // Multi-basis HRF sets
    // ------------------------------------------------------------------
    println!("## Multivariate HRFs: Basis Sets\n");

    // SPM with derivatives (3 basis functions)
    let spmg3 = spm_hrf_with_derivatives();
    let spmg3_vals = spmg3.evaluate(&t).unwrap();
    println!(
        "  SPMG3: nbasis={}, shape={:?}",
        spmg3.nbasis(),
        spmg3_vals.shape()
    );

    // B-spline basis
    let bs5 = bspline_hrf(5, None, None);
    let bs_vals = bs5.evaluate(&t).unwrap();
    println!(
        "  B-spline(5): nbasis={}, shape={:?}",
        bs5.nbasis(),
        bs_vals.shape()
    );

    // Sine basis
    let sine5 = sine_basis_hrf(5, 24.0);
    let sine_vals = sine5.evaluate(&t).unwrap();
    println!(
        "  Sine(5): nbasis={}, shape={:?}",
        sine5.nbasis(),
        sine_vals.shape()
    );

    // FIR basis
    let fir12 = fir_hrf(12, None);
    let fir_vals = fir12.evaluate(&t).unwrap();
    println!(
        "  FIR(12): nbasis={}, shape={:?}\n",
        fir12.nbasis(),
        fir_vals.shape()
    );

    // ------------------------------------------------------------------
    // Other HRF shapes
    // ------------------------------------------------------------------
    println!("## Other HRF Shapes\n");

    let gam = gamma_hrf(6.0, 1.0, None);
    let mex = mexhat_hrf(2.0, 6.0);
    let inv = inv_logit_hrf(6.0, 1.0, 16.0, 1.0, 0.0);
    let lwu = lwu_hrf(None, None, None, None).unwrap();

    for (label, hrf) in [
        ("Gamma(6,1)", &*gam as &dyn Hrf),
        ("MexHat(2,6)", &*mex as &dyn Hrf),
        ("InvLogit(6,1,16,1)", &*inv as &dyn Hrf),
        ("LWU(default)", &*lwu as &dyn Hrf),
    ] {
        let vals = hrf.evaluate(&t).unwrap();
        let max_val = vals
            .column(0)
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_val = vals.column(0).iter().cloned().fold(f64::INFINITY, f64::min);
        println!("  {}: range=[{:.4}, {:.4}]", label, min_val, max_val);
    }
    println!();

    // ------------------------------------------------------------------
    // Boxcar and weighted HRFs
    // ------------------------------------------------------------------
    println!("## Boxcar and Weighted HRFs\n");

    let boxcar = boxcar_hrf(5.0, 1.0, false);
    let bc_vals = boxcar
        .evaluate(&Array1::from_vec(vec![0.0, 2.5, 4.99, 5.0, 10.0]))
        .unwrap();
    println!(
        "  Boxcar(5s): at t=0: {}, t=2.5: {}, t=4.99: {}, t=5.0: {}",
        bc_vals[[0, 0]],
        bc_vals[[1, 0]],
        bc_vals[[2, 0]],
        bc_vals[[3, 0]]
    );

    let weights = vec![0.0, 0.5, 1.0, 0.5, 0.0];
    let times = vec![0.0, 2.0, 4.0, 6.0, 8.0];
    let wt_hrf = weighted_hrf(weights, times, WeightedMethod::Linear, false).unwrap();
    let wt_vals = wt_hrf
        .evaluate(&Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0]))
        .unwrap();
    println!(
        "  Weighted(linear): at t=0: {:.1}, t=2: {:.1}, t=4: {:.1}, t=6: {:.1}, t=8: {:.1}",
        wt_vals[[0, 0]],
        wt_vals[[1, 0]],
        wt_vals[[2, 0]],
        wt_vals[[3, 0]],
        wt_vals[[4, 0]]
    );
    println!();

    // ------------------------------------------------------------------
    // Empirical HRF
    // ------------------------------------------------------------------
    println!("## Creating Empirical HRFs\n");

    let emp_t = Array1::from_vec(vec![0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0]);
    let emp_y = Array1::from_vec(vec![0.0, 0.2, 0.8, 1.0, 0.7, 0.3, -0.1, 0.0]);
    let emp = empirical_hrf(&emp_t, &emp_y, Some("measured_hrf"), None).unwrap();

    let emp_vals = emp.evaluate(&Array1::linspace(0.0, 14.0, 15)).unwrap();
    println!(
        "  Empirical HRF: name={}, nbasis={}",
        emp.name(),
        emp.nbasis()
    );
    println!(
        "  Values at 1s intervals: {:?}",
        emp_vals
            .column(0)
            .iter()
            .map(|v| format!("{:.3}", v))
            .collect::<Vec<_>>()
    );

    println!("\n=== Done ===");
}
