//! Demonstration of plotting functionality for fMRI HRF visualization.
//!
//! Run with: cargo run --example plotting_demo --features plotting

#[cfg(feature = "plotting")]
use fmrihrf::{
    hrf::{bspline_hrf, fourier_hrf, gamma_hrf, spm_canonical, spm_hrf_with_derivatives},
    plotting::{plot_design_matrix, plot_hrf, plot_hrf_comparison, PlotConfig},
    regressor::{design_matrix_from_set, regressor_set},
};
#[cfg(feature = "plotting")]
use ndarray::Array1;
#[cfg(feature = "plotting")]
use std::fs;

#[cfg(feature = "plotting")]
fn main() {
    println!("=== fmrihrf-rs Plotting Demo ===\n");

    // Create output directory
    fs::create_dir_all("plots").unwrap();

    demo_single_hrf();
    demo_hrf_comparison();
    demo_multi_basis_hrf();
    demo_design_matrix();
    demo_interactive_html();

    println!("\nPlots saved to ./plots/ directory!");
}

#[cfg(feature = "plotting")]
fn demo_single_hrf() {
    println!("1. Plotting single HRF");

    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let times = Array1::linspace(0.0, 30.0, 301);

    let config = PlotConfig {
        title: "SPM Canonical HRF".to_string(),
        ylabel: "BOLD Response".to_string(),
        ..Default::default()
    };

    // Save as SVG first (no font issues)
    plot_hrf(
        hrf.as_ref(),
        &times,
        "plots/spm_canonical.svg",
        Some(config.clone()),
    )
    .unwrap();

    // Save as PNG
    plot_hrf(
        hrf.as_ref(),
        &times,
        "plots/spm_canonical.png",
        Some(config),
    )
    .unwrap();

    println!("  ✓ Saved spm_canonical.png and .svg");
}

#[cfg(feature = "plotting")]
fn demo_hrf_comparison() {
    println!("2. Comparing multiple HRFs");

    let times = Array1::linspace(0.0, 30.0, 301);

    let hrfs = vec![
        (
            "SPM Canonical".to_string(),
            spm_canonical(None, None, None, None, None, Some(true)),
        ),
        ("Gamma (α=6)".to_string(), gamma_hrf(6.0, 1.0, Some(true))),
        ("Gamma (α=8)".to_string(), gamma_hrf(8.0, 1.0, Some(true))),
        ("Gamma (α=10)".to_string(), gamma_hrf(10.0, 1.0, Some(true))),
    ];

    let config = PlotConfig {
        title: "HRF Model Comparison".to_string(),
        ylabel: "Normalized Response".to_string(),
        ..Default::default()
    };

    plot_hrf_comparison(&hrfs, &times, "plots/hrf_comparison.png", Some(config)).unwrap();

    println!("  ✓ Saved hrf_comparison.png");
}

#[cfg(feature = "plotting")]
fn demo_multi_basis_hrf() {
    println!("3. Plotting multi-basis HRF");

    let hrf = spm_hrf_with_derivatives();
    let times = Array1::linspace(0.0, 30.0, 301);

    let config = PlotConfig {
        title: "SPM HRF with Temporal and Dispersion Derivatives".to_string(),
        ylabel: "Basis Functions".to_string(),
        ..Default::default()
    };

    plot_hrf(
        hrf.as_ref(),
        &times,
        "plots/spm_derivatives.png",
        Some(config),
    )
    .unwrap();

    println!("  ✓ Saved spm_derivatives.png");
}

#[cfg(feature = "plotting")]
fn demo_design_matrix() {
    println!("4. Plotting design matrix");

    // Create experimental design
    let onsets = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
    let conditions = vec!["A", "B", "C", "A", "B", "C", "A", "B"];

    let hrf = spm_canonical(None, None, None, None, None, Some(true));
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

    let grid = Array1::linspace(0.0, 100.0, 501);
    let dm = design_matrix_from_set(&rset, &grid, None).unwrap();

    let config = PlotConfig {
        title: "fMRI Design Matrix".to_string(),
        width: 600,
        height: 800,
        ..Default::default()
    };

    plot_design_matrix(
        &dm.data,
        "plots/design_matrix.png",
        Some(config),
        Some(&dm.column_names),
    )
    .unwrap();

    println!("  ✓ Saved design_matrix.png");
}

#[cfg(feature = "plotting")]
fn demo_interactive_html() {
    println!("5. Creating interactive HTML plots");

    // Basis function comparison
    let times = Array1::linspace(0.0, 30.0, 301);

    let hrfs = vec![
        (
            "B-spline (6 basis)".to_string(),
            bspline_hrf(6, Some(3), Some(30.0)),
        ),
        ("Fourier (7 basis)".to_string(), fourier_hrf(7, Some(30.0))),
    ];

    let config = PlotConfig {
        title: "Basis Function Comparison".to_string(),
        ylabel: "Amplitude".to_string(),
        ..Default::default()
    };

    plot_hrf_comparison(&hrfs, &times, "plots/basis_comparison.html", Some(config)).unwrap();

    // Design matrix heatmap
    let onsets = vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0];
    let conditions = vec![
        "Face", "House", "Object", "Face", "House", "Object", "Face", "House", "Object", "Face",
    ];

    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let rset = regressor_set(
        onsets,
        conditions.into_iter().map(|s| s.to_string()).collect(),
        Some(hrf),
        Some(vec![2.0]), // 2s block duration
        None,
        None,
        None,
    )
    .unwrap();

    let grid = Array1::linspace(0.0, 60.0, 301);
    let dm = design_matrix_from_set(&rset, &grid, None).unwrap();

    let config = PlotConfig {
        title: "Interactive Design Matrix".to_string(),
        width: 800,
        height: 600,
        ..Default::default()
    };

    plot_design_matrix(
        &dm.data,
        "plots/design_matrix_interactive.html",
        Some(config),
        Some(&dm.column_names),
    )
    .unwrap();

    println!("  ✓ Saved basis_comparison.html and design_matrix_interactive.html");
}

#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("This example requires the 'plotting' feature.");
    eprintln!("Run with: cargo run --example plotting_demo --features plotting");
}
