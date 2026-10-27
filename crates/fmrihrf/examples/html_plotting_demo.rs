//! HTML-based plotting demonstration for fMRI HRF visualization.
//! Uses plotly for interactive plots without font dependencies.
//!
//! Run with: cargo run --example html_plotting_demo --features plotting

#[cfg(feature = "plotting")]
use fmrihrf::{
    hrf::{bspline_hrf, fir_hrf, fourier_hrf, gamma_hrf, spm_canonical, spm_hrf_with_derivatives},
    plotting::{plot_design_matrix, plot_hrf, plot_hrf_comparison, PlotConfig},
    regressor::{design_matrix_from_set, regressor_set},
};
#[cfg(feature = "plotting")]
use ndarray::Array1;
#[cfg(feature = "plotting")]
use std::fs;

#[cfg(feature = "plotting")]
fn main() {
    println!("=== fmrihrf-rs Interactive HTML Plotting Demo ===\n");

    // Create output directory
    fs::create_dir_all("plots").unwrap();

    demo_single_hrf();
    demo_hrf_comparison();
    demo_multi_basis_hrf();
    demo_design_matrix();
    demo_basis_comparison();

    println!("\nHTML plots saved to ./plots/ directory!");
    println!("Open these files in a web browser to interact with the plots:");
    println!("  - plots/spm_canonical.html");
    println!("  - plots/hrf_comparison.html");
    println!("  - plots/spm_derivatives.html");
    println!("  - plots/design_matrix.html");
    println!("  - plots/basis_comparison.html");
}

#[cfg(feature = "plotting")]
fn demo_single_hrf() {
    println!("1. Plotting single HRF");

    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let times = Array1::linspace(0.0, 30.0, 301);

    let config = PlotConfig {
        title: "SPM Canonical HRF".to_string(),
        ylabel: "BOLD Response".to_string(),
        width: 800,
        height: 500,
        ..Default::default()
    };

    plot_hrf(
        hrf.as_ref(),
        &times,
        "plots/spm_canonical.html",
        Some(config),
    )
    .unwrap();

    println!("  ✓ Created spm_canonical.html");
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
        width: 900,
        height: 600,
        ..Default::default()
    };

    plot_hrf_comparison(&hrfs, &times, "plots/hrf_comparison.html", Some(config)).unwrap();

    println!("  ✓ Created hrf_comparison.html");
}

#[cfg(feature = "plotting")]
fn demo_multi_basis_hrf() {
    println!("3. Plotting multi-basis HRF");

    let hrf = spm_hrf_with_derivatives();
    let times = Array1::linspace(0.0, 30.0, 301);

    let config = PlotConfig {
        title: "SPM HRF with Temporal and Dispersion Derivatives".to_string(),
        ylabel: "Basis Functions".to_string(),
        width: 900,
        height: 600,
        ..Default::default()
    };

    plot_hrf(
        hrf.as_ref(),
        &times,
        "plots/spm_derivatives.html",
        Some(config),
    )
    .unwrap();

    println!("  ✓ Created spm_derivatives.html");
}

#[cfg(feature = "plotting")]
fn demo_design_matrix() {
    println!("4. Plotting design matrix");

    // Create experimental design with multiple conditions
    let onsets = vec![
        5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
        80.0, 85.0, 90.0, 95.0, 100.0,
    ];
    let conditions = vec![
        "Face", "House", "Object", "Rest", "Face", "House", "Object", "Rest", "Face", "House",
        "Object", "Rest", "Face", "House", "Object", "Rest", "Face", "House", "Object", "Rest",
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

    let grid = Array1::linspace(0.0, 120.0, 601);
    let dm = design_matrix_from_set(&rset, &grid, None).unwrap();

    let config = PlotConfig {
        title: "fMRI Design Matrix Heatmap".to_string(),
        width: 800,
        height: 1000,
        ..Default::default()
    };

    plot_design_matrix(
        &dm.data,
        "plots/design_matrix.html",
        Some(config),
        Some(&dm.column_names),
    )
    .unwrap();

    println!("  ✓ Created design_matrix.html");
}

#[cfg(feature = "plotting")]
fn demo_basis_comparison() {
    println!("5. Comparing basis function sets");

    let times = Array1::linspace(0.0, 30.0, 301);

    let hrfs = vec![
        (
            "B-spline (6 basis)".to_string(),
            bspline_hrf(6, Some(3), Some(30.0)),
        ),
        ("Fourier (7 basis)".to_string(), fourier_hrf(7, Some(30.0))),
        ("FIR (10 bins)".to_string(), fir_hrf(10, Some(30.0))),
    ];

    // For multi-basis HRFs, we'll just plot the first basis function
    let config = PlotConfig {
        title: "Basis Function Set Comparison (First Basis Only)".to_string(),
        ylabel: "Amplitude".to_string(),
        xlabel: "Time (s)".to_string(),
        width: 900,
        height: 600,
        ..Default::default()
    };

    plot_hrf_comparison(&hrfs, &times, "plots/basis_comparison.html", Some(config)).unwrap();

    println!("  ✓ Created basis_comparison.html");
}

#[cfg(not(feature = "plotting"))]
fn main() {
    eprintln!("This example requires the 'plotting' feature.");
    eprintln!("Run with: cargo run --example html_plotting_demo --features plotting");
}
