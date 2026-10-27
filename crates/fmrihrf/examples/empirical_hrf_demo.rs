//! Example demonstrating empirical HRF functionality.

use fmrihrf::hrf::{empirical_hrf, spm_canonical};
use ndarray::{array, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Empirical HRF Demo ===\n");

    // Example 1: Create empirical HRF from measured data points
    println!("1. Creating empirical HRF from measured data:");

    // Simulate some measured HRF data points
    let t_measured = array![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.];
    let y_measured = array![0., 0.2, 0.8, 1.0, 0.7, 0.3, 0., -0.1, -0.05, 0., 0.];

    println!("   Measured time points: {:?}", t_measured);
    println!("   Measured HRF values: {:?}", y_measured);

    // Create empirical HRF
    let emp_hrf = empirical_hrf(&t_measured, &y_measured, Some("measured_hrf"), None)?;

    println!("   Created empirical HRF: {}", emp_hrf.name());
    println!("   Number of basis functions: {}", emp_hrf.nbasis());
    println!("   Temporal span: {} seconds", emp_hrf.span());

    // Evaluate at finer time resolution
    let t_eval = Array1::linspace(0., 20., 101);
    let response = emp_hrf.evaluate(&t_eval)?;

    println!("   Evaluated at {} time points", t_eval.len());
    println!(
        "   Peak value: {:.3}",
        response.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
    );

    // Example 2: Compare with canonical HRF
    println!("\n2. Comparing empirical HRF with SPM canonical:");

    let canonical_hrf = spm_canonical(None, None, None, None, None, Some(true));
    let canonical_response = canonical_hrf.evaluate(&t_eval)?;

    // Find peaks
    let emp_peak = response.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let can_peak = canonical_response.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    println!("   Empirical peak: {:.3}", emp_peak);
    println!("   Canonical peak: {:.3}", can_peak);

    // Example 3: Create empirical HRF with custom span
    println!("\n3. Creating empirical HRF with custom span:");

    let t_short = array![0., 5., 10.];
    let y_short = array![0., 1., 0.];

    let emp_hrf_custom = empirical_hrf(&t_short, &y_short, Some("custom_span_hrf"), Some(30.0))?;

    println!("   Created HRF with {} data points", t_short.len());
    println!("   Custom span: {} seconds", emp_hrf_custom.span());

    // Evaluate beyond original data range
    let t_extended = Array1::linspace(0., 30., 31);
    let response_extended = emp_hrf_custom.evaluate(&t_extended)?;

    println!("   Values at t=0, 5, 10, 15, 20, 25, 30:");
    for i in [0, 5, 10, 15, 20, 25, 30] {
        println!("     t={:2}: {:.3}", i, response_extended[[i, 0]]);
    }

    // Example 4: Import HRF from external software
    println!("\n4. Importing HRF from external measurements:");

    // Simulate HRF data that might come from FSL, SPM, or AFNI
    let external_times =
        array![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 12., 14., 16., 18., 20.];
    let external_values = array![
        0.000, 0.087, 0.350, 0.630, 0.850, 0.950, 0.980, 0.950, 0.870, 0.750, 0.600, 0.350, 0.150,
        0.020, -0.020, 0.000
    ];

    let imported_hrf = empirical_hrf(
        &external_times,
        &external_values,
        Some("imported_hrf"),
        None,
    )?;

    println!("   Imported HRF with {} data points", external_times.len());
    println!(
        "   Using {} interpolation",
        if external_times.len() >= 4 {
            "cubic spline"
        } else {
            "linear"
        }
    );

    // Evaluate at standard fMRI TR
    let tr = 2.0;
    let n_scans = 15;
    let scan_times = Array1::from_vec((0..n_scans).map(|i| i as f64 * tr).collect());
    let fmri_response = imported_hrf.evaluate(&scan_times)?;

    println!("   Response at TR={} for {} scans:", tr, n_scans);
    for (i, t) in scan_times.iter().enumerate() {
        println!(
            "     Scan {:2} (t={:4.1}s): {:.3}",
            i + 1,
            t,
            fmri_response[[i, 0]]
        );
    }

    // Example 5: Handle edge cases
    println!("\n5. Edge cases and extrapolation:");

    let t_limited = array![5., 10., 15.];
    let y_limited = array![0.5, 1.0, 0.5];

    let limited_hrf = empirical_hrf(&t_limited, &y_limited, Some("limited_range"), None)?;

    // Test extrapolation behavior
    let t_test = array![0., 2.5, 5., 10., 15., 20., 25.];
    let response_test = limited_hrf.evaluate(&t_test)?;

    println!(
        "   Data range: {} to {} seconds",
        t_limited[0],
        t_limited[t_limited.len() - 1]
    );
    println!("   Extrapolation behavior:");
    for (i, &t) in t_test.iter().enumerate() {
        let status = if t < t_limited[0] || t > t_limited[t_limited.len() - 1] {
            " (extrapolated to 0)"
        } else {
            ""
        };
        println!("     t={:4.1}: {:.3}{}", t, response_test[[i, 0]], status);
    }

    println!("\n=== Demo Complete ===");

    Ok(())
}
