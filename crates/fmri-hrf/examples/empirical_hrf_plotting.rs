//! Example demonstrating empirical HRF plotting.

#[cfg(feature = "plotting")]
use fmri_hrf::hrf::{empirical_hrf, gamma_hrf, spm_canonical};
#[cfg(feature = "plotting")]
use fmri_hrf::plotting::{plot_hrf, plot_hrf_comparison, PlotConfig};
#[cfg(feature = "plotting")]
use ndarray::{array, Array1};

#[cfg(feature = "plotting")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Empirical HRF Plotting Demo ===\n");

    // Create output directory
    std::fs::create_dir_all("plots")?;

    // Example 1: Plot measured HRF data
    println!("1. Plotting empirical HRF from measured data:");

    let t_measured = array![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 24., 28., 32.];
    let y_measured = array![0., 0.1, 0.6, 0.95, 0.8, 0.4, 0.05, -0.15, -0.1, -0.02, 0., 0., 0., 0.];

    let emp_hrf = empirical_hrf(&t_measured, &y_measured, Some("Measured HRF"), None)?;

    let t_eval = Array1::linspace(0., 35., 351);

    let config = PlotConfig {
        title: "Empirical HRF from Measured Data".to_string(),
        xlabel: "Time (s)".to_string(),
        ylabel: "BOLD Response".to_string(),
        width: 800,
        height: 600,
        ..Default::default()
    };

    plot_hrf(
        emp_hrf.as_ref(),
        &t_eval,
        "plots/empirical_measured.html",
        Some(config),
    )?;
    println!("   Saved to plots/empirical_measured.html");

    // Example 2: Compare empirical with theoretical HRFs
    println!("\n2. Comparing empirical HRF with theoretical models:");

    let canonical = spm_canonical(None, None, None, None, None, Some(true));
    let gamma = gamma_hrf(6.0, 1.0, Some(true));

    let hrfs = vec![
        ("Empirical (Measured)".to_string(), emp_hrf),
        ("SPM Canonical".to_string(), canonical),
        ("Gamma (α=6, β=1)".to_string(), gamma),
    ];

    let config = PlotConfig {
        title: "Empirical vs Theoretical HRFs".to_string(),
        ..Default::default()
    };

    plot_hrf_comparison(
        &hrfs,
        &t_eval,
        "plots/empirical_comparison.html",
        Some(config),
    )?;
    println!("   Saved to plots/empirical_comparison.html");

    // Example 3: Show interpolation quality
    println!("\n3. Demonstrating interpolation quality:");

    // Sparse measurements
    let t_sparse = array![0., 4., 8., 12., 16., 20.];
    let y_sparse = array![0., 0.8, 0.9, 0.2, -0.05, 0.];

    let sparse_hrf = empirical_hrf(&t_sparse, &y_sparse, Some("Sparse Data"), None)?;

    // Dense measurements
    let t_dense = Array1::linspace(0., 20., 21);
    let y_dense = t_dense.mapv(|t: f64| {
        let a = 0.9;
        let b = 0.1;
        a * t.powf(2.5) * (-b * t).exp()
    });
    let dense_hrf = empirical_hrf(&t_dense, &y_dense, Some("Dense Data"), None)?;

    let hrfs = vec![
        ("Sparse (6 points)".to_string(), sparse_hrf),
        ("Dense (21 points)".to_string(), dense_hrf),
    ];

    let config = PlotConfig {
        title: "Interpolation Quality: Sparse vs Dense Sampling".to_string(),
        ..Default::default()
    };

    plot_hrf_comparison(
        &hrfs,
        &t_eval,
        "plots/empirical_interpolation.html",
        Some(config),
    )?;
    println!("   Saved to plots/empirical_interpolation.html");

    // Example 4: Import and plot HRF from different sources
    println!("\n4. Simulating HRF import from different software:");

    // Simulate FSL-style HRF (3-parameter gamma)
    let t_fsl = Array1::linspace(0., 30., 31);
    let y_fsl = t_fsl.mapv(|t: f64| {
        if t == 0.0 {
            0.0
        } else {
            let k1 = 6.0;
            let k2 = 12.0;
            let k3 = 0.35;
            t.powf(k1 - 1.0) * (-t / k2).exp() - k3 * t.powf(k1 - 1.0) * (-t / (k2 * 2.0)).exp()
        }
    });
    let fsl_hrf = empirical_hrf(&t_fsl, &y_fsl, Some("FSL-style Import"), None)?;

    // Simulate AFNI-style HRF
    let t_afni = Array1::linspace(0., 25., 26);
    let y_afni = t_afni
        .mapv(|t: f64| 0.8 * (t / 4.0) * (-t / 4.0).exp() - 0.2 * (t / 8.0) * (-t / 8.0).exp());
    let afni_hrf = empirical_hrf(&t_afni, &y_afni, Some("AFNI-style Import"), None)?;

    let hrfs = vec![
        ("FSL-style Import".to_string(), fsl_hrf),
        ("AFNI-style Import".to_string(), afni_hrf),
        (
            "Native SPM".to_string(),
            spm_canonical(None, None, None, None, None, Some(true)),
        ),
    ];

    let config = PlotConfig {
        title: "Imported HRFs from Different Software".to_string(),
        ..Default::default()
    };

    plot_hrf_comparison(&hrfs, &t_eval, "plots/empirical_imports.html", Some(config))?;
    println!("   Saved to plots/empirical_imports.html");

    // Example 5: Edge case visualization
    println!("\n5. Visualizing extrapolation behavior:");

    // Limited range data
    let t_limited = array![4., 8., 12., 16.];
    let y_limited = array![0.3, 1.0, 0.5, 0.1];

    let limited_hrf = empirical_hrf(&t_limited, &y_limited, Some("Limited Range (4-16s)"), None)?;

    // Full range reference
    let t_full = Array1::linspace(0., 20., 21);
    let y_full = t_full.mapv(|t: f64| {
        let peak = 8.0;
        let width = 4.0;
        (-((t - peak) / width).powi(2)).exp()
    });
    let full_hrf = empirical_hrf(&t_full, &y_full, Some("Full Range Reference"), None)?;

    let hrfs = vec![
        ("Limited Range (extrapolates to 0)".to_string(), limited_hrf),
        ("Full Range Reference".to_string(), full_hrf),
    ];

    let config = PlotConfig {
        title: "Extrapolation Behavior of Empirical HRF".to_string(),
        ..Default::default()
    };

    let t_wide = Array1::linspace(-2., 25., 271);
    plot_hrf_comparison(
        &hrfs,
        &t_wide,
        "plots/empirical_extrapolation.html",
        Some(config),
    )?;
    println!("   Saved to plots/empirical_extrapolation.html");

    println!("\n=== All plots saved to plots/ directory ===");
    println!("Open the HTML files in a web browser to view interactive plots.");

    Ok(())
}

#[cfg(not(feature = "plotting"))]
fn main() {
    println!("This example requires the 'plotting' feature.");
    println!("Run with: cargo run --example empirical_hrf_plotting --features plotting");
}
