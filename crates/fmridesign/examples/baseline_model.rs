//! Baseline Model Example
//! ======================
//!
//! This example demonstrates how to create baseline models for fMRI designs.

use fmridesign::baseline::{baseline_model, BasisType, InterceptType, SamplingInfo};
use fmridesign::sampling::uniform_sampling_frame;
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a sampling frame
    let n_scans = 200;
    let tr = 2.0;
    let sampling_frame = uniform_sampling_frame(n_scans, tr, Some(0.0))?;

    // Create SamplingInfo wrapper
    let sampling_info = SamplingInfo::from_frame(sampling_frame);

    // Example 1: Constant baseline (intercept only)
    println!("Example 1: Constant baseline");
    let baseline1 = baseline_model(
        BasisType::Constant,
        1,
        &sampling_info,
        InterceptType::Global,
        None,
    )?;

    let dm1 = baseline1.design_matrix()?;
    println!("  Shape: {:?}", dm1.shape());
    println!("  Columns: {:?}", baseline1.column_names());
    println!();

    // Example 2: Polynomial drift
    println!("Example 2: Polynomial drift (degree 3)");
    let baseline2 = baseline_model(
        BasisType::Polynomial,
        3,
        &sampling_info,
        InterceptType::Global,
        None,
    )?;

    let dm2 = baseline2.design_matrix()?;
    println!("  Shape: {:?}", dm2.shape());
    println!("  Columns: {:?}", baseline2.column_names());
    println!();

    // Example 3: B-spline drift
    println!("Example 3: B-spline drift (degree 4)");
    let baseline3 = baseline_model(
        BasisType::BSpline,
        4,
        &sampling_info,
        InterceptType::Runwise,
        None,
    )?;

    let dm3 = baseline3.design_matrix()?;
    println!("  Shape: {:?}", dm3.shape());
    println!("  Columns: {:?}", baseline3.column_names());
    println!();

    // Example 4: With nuisance regressors
    println!("Example 4: Polynomial drift with nuisance regressors");

    // Create mock nuisance regressors (e.g., motion parameters)
    let nuisance = vec![Array2::from_shape_fn((n_scans, 6), |(i, j)| {
        (i as f64 * 0.01 + j as f64 * 0.1).sin()
    })];

    let baseline4 = baseline_model(
        BasisType::Polynomial,
        2,
        &sampling_info,
        InterceptType::Global,
        Some(nuisance),
    )?;

    let dm4 = baseline4.design_matrix()?;
    println!("  Shape: {:?}", dm4.shape());
    println!("  Columns: {:?}", baseline4.column_names());

    // Show first few column names
    let cols = baseline4.column_names();
    println!("  First few columns:");
    for (i, name) in cols.iter().take(5).enumerate() {
        println!("    {}: {}", i, name);
    }

    Ok(())
}
