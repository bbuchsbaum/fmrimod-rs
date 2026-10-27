//! Vignette 02: Baseline Models
//! ==============================
//!
//! Port of R's `a_03_baseline_model.Rmd`.
//!
//! Baseline models capture nuisance variance in fMRI time series --
//! low-frequency scanner drift, run-wise intercepts, and known confounds
//! such as head motion parameters. This vignette demonstrates how to
//! construct baseline models with different basis functions and how to
//! combine them with nuisance regressors.
//!
//! Run with:
//!   cargo run --example vignette_02_baseline

use fmri_design::baseline::{
    baseline, baseline_model, block, nuisance, BaselineSpec, BasisType, InterceptType, SamplingInfo,
};
use fmri_design::sampling::{multi_block_sampling_frame, uniform_sampling_frame};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vignette 02: Baseline Models ===\n");

    // ---------------------------------------------------------------
    // 1. Single-run baseline with different basis types
    // ---------------------------------------------------------------
    // In R: baseline_model(basis = "poly", degree = 3, sframe)
    println!("--- Step 1: Single-run baselines ---\n");

    let sf_single = uniform_sampling_frame(200, 2.0, None)?;
    let sinfo_single = SamplingInfo::from_frame(sf_single);

    // 1a. Constant baseline (intercept only)
    // In R: baseline_model(basis = "constant", sframe)
    println!("  [1a] Constant baseline (intercept only)");
    let baseline_const = baseline_model(
        BasisType::Constant,
        1,
        &sinfo_single,
        InterceptType::Global,
        None,
    )?;
    let dm_const = baseline_const.design_matrix()?;
    println!("    Shape:   {:?}", dm_const.shape());
    println!("    Columns: {:?}", baseline_const.column_names());
    println!(
        "    All values equal 1: {}",
        dm_const.iter().all(|&v| v == 1.0)
    );
    println!();

    // 1b. Polynomial baseline
    // In R: baseline_model(basis = "poly", degree = 3, sframe)
    println!("  [1b] Polynomial baseline (degree 3)");
    let baseline_poly = baseline_model(
        BasisType::Polynomial,
        3,
        &sinfo_single,
        InterceptType::Global,
        None,
    )?;
    let dm_poly = baseline_poly.design_matrix()?;
    println!("    Shape:   {:?}", dm_poly.shape());
    println!("    Columns: {:?}", baseline_poly.column_names());
    print_column_ranges(&dm_poly, &baseline_poly.column_names());
    println!();

    // 1c. B-spline baseline
    // In R: baseline_model(basis = "bs", degree = 5, sframe)
    println!("  [1c] B-spline baseline (5 basis functions)");
    let baseline_bs = baseline_model(
        BasisType::BSpline,
        5,
        &sinfo_single,
        InterceptType::Global,
        None,
    )?;
    let dm_bs = baseline_bs.design_matrix()?;
    println!("    Shape:   {:?}", dm_bs.shape());
    println!("    Columns: {:?}", baseline_bs.column_names());
    print_column_ranges(&dm_bs, &baseline_bs.column_names());
    println!();

    // 1d. Natural spline baseline
    // In R: baseline_model(basis = "ns", degree = 4, sframe)
    println!("  [1d] Natural spline baseline (4 basis functions)");
    let baseline_ns = baseline_model(
        BasisType::NaturalSpline,
        4,
        &sinfo_single,
        InterceptType::Global,
        None,
    )?;
    let dm_ns = baseline_ns.design_matrix()?;
    println!("    Shape:   {:?}", dm_ns.shape());
    println!("    Columns: {:?}", baseline_ns.column_names());
    print_column_ranges(&dm_ns, &baseline_ns.column_names());
    println!();

    // ---------------------------------------------------------------
    // 2. Multi-run baseline with run-wise intercepts
    // ---------------------------------------------------------------
    // In R: baseline_model(basis = "poly", degree = 2, sframe,
    //                       intercept = "runwise")
    println!("--- Step 2: Multi-run baseline with run-wise intercepts ---\n");

    let sf_multi = multi_block_sampling_frame(&[100, 120, 80], &[2.0], None)?;
    let sinfo_multi = SamplingInfo::from_frame(sf_multi);

    println!(
        "  Three runs: 100, 120, 80 scans (total: {})",
        sinfo_multi.n_scans()
    );

    // Polynomial drift with runwise intercept
    let baseline_multi = baseline_model(
        BasisType::Polynomial,
        2,
        &sinfo_multi,
        InterceptType::Runwise,
        None,
    )?;
    let dm_multi = baseline_multi.design_matrix()?;
    let cols_multi = baseline_multi.column_names();

    println!("  Shape:   {:?}", dm_multi.shape());
    println!("  Columns: {:?}", cols_multi);
    println!();

    // Show that drift regressors are block-diagonal
    println!("  Block-diagonal structure (sample rows from each run):");
    let run_boundaries = [0, 100, 220]; // cumulative scan offsets
    for (run_idx, &start) in run_boundaries.iter().enumerate() {
        print!("    Run {} (row {:>3}): ", run_idx + 1, start);
        for j in 0..dm_multi.ncols().min(10) {
            print!("{:>7.3}", dm_multi[[start, j]]);
        }
        println!();
    }
    println!();

    // ---------------------------------------------------------------
    // 3. Adding nuisance regressors (motion parameters)
    // ---------------------------------------------------------------
    // In R: baseline_model(..., nuisance_list = list(run1_motion, run2_motion, run3_motion))
    println!("--- Step 3: Baseline with nuisance regressors ---\n");

    // Simulate 6 motion parameters per run (x, y, z, roll, pitch, yaw)
    let run_lengths = [100usize, 120, 80];
    let nuisance_matrices: Vec<Array2<f64>> = run_lengths
        .iter()
        .map(|&n| {
            Array2::from_shape_fn((n, 6), |(i, j)| {
                // Simulate slow drift in translation and small rotations
                let t = i as f64 / n as f64;
                match j {
                    0 => 0.1 * t.sin(),           // x translation
                    1 => 0.05 * (2.0 * t).cos(),  // y translation
                    2 => 0.08 * (0.5 * t).sin(),  // z translation
                    3 => 0.002 * (3.0 * t).sin(), // roll
                    4 => 0.001 * (1.5 * t).cos(), // pitch
                    5 => 0.003 * (0.7 * t).sin(), // yaw
                    _ => 0.0,
                }
            })
        })
        .collect();

    let baseline_with_nuis = baseline_model(
        BasisType::Polynomial,
        2,
        &sinfo_multi,
        InterceptType::Runwise,
        Some(nuisance_matrices),
    )?;

    let dm_nuis = baseline_with_nuis.design_matrix()?;
    let cols_nuis = baseline_with_nuis.column_names();

    println!("  Shape:   {:?}", dm_nuis.shape());
    println!("  Total columns: {}", cols_nuis.len());
    println!();
    println!("  Column breakdown:");

    // Group columns by type
    let drift_cols: Vec<_> = cols_nuis
        .iter()
        .filter(|c| c.starts_with("base_"))
        .collect();
    let intercept_cols: Vec<_> = cols_nuis
        .iter()
        .filter(|c| c.starts_with("intercept"))
        .collect();
    let nuis_cols: Vec<_> = cols_nuis
        .iter()
        .filter(|c| c.starts_with("nuis_"))
        .collect();

    println!("    Drift columns ({}):", drift_cols.len());
    for c in &drift_cols {
        println!("      {}", c);
    }
    println!("    Intercept columns ({}):", intercept_cols.len());
    for c in &intercept_cols {
        println!("      {}", c);
    }
    println!("    Nuisance columns ({}):", nuis_cols.len());
    for c in &nuis_cols[..nuis_cols.len().min(6)] {
        println!("      {}", c);
    }
    if nuis_cols.len() > 6 {
        println!("      ... and {} more", nuis_cols.len() - 6);
    }
    println!();

    // ---------------------------------------------------------------
    // 4. Convenience constructors
    // ---------------------------------------------------------------
    // In R: baseline(degree = 3, basis = "poly")
    println!("--- Step 4: Convenience constructors ---\n");

    let poly_spec = baseline(3, "poly", None)?;
    println!(
        "  baseline(3, \"poly\"): degree={}, basis={:?}, intercept={:?}",
        poly_spec.degree, poly_spec.basis, poly_spec.intercept
    );

    let bs_spec = baseline(5, "bs", Some("my_drift"))?;
    println!(
        "  baseline(5, \"bs\", \"my_drift\"): name={}, basis={:?}",
        bs_spec.name, bs_spec.basis
    );

    let nuis_spec = nuisance("motion", Some(vec!["x", "y", "z", "roll", "pitch", "yaw"]));
    println!(
        "  nuisance(\"motion\"): name={}, columns={:?}",
        nuis_spec.name, nuis_spec.columns
    );

    let block_spec = block(vec![100, 100, 100], Some(vec!["run1", "run2", "run3"]));
    println!(
        "  block([100,100,100]): lengths={:?}, names={:?}",
        block_spec.block_lens, block_spec.names
    );
    println!();

    // ---------------------------------------------------------------
    // 5. Comparing basis types visually
    // ---------------------------------------------------------------
    println!("--- Step 5: Comparing drift basis types ---\n");

    let sf_compare = uniform_sampling_frame(100, 2.0, None)?;
    let sinfo_compare = SamplingInfo::from_frame(sf_compare);

    let bases: Vec<(&str, BasisType, usize)> = vec![
        ("Polynomial(3)", BasisType::Polynomial, 3),
        ("B-spline(5)", BasisType::BSpline, 5),
        ("B-spline(10)", BasisType::BSpline, 10),
    ];

    for (label, btype, degree) in &bases {
        let model = baseline_model(*btype, *degree, &sinfo_compare, InterceptType::Global, None)?;
        let dm = model.design_matrix()?;
        let cols = model.column_names();

        // Show range of each basis function
        println!("  {} ({} columns):", label, cols.len());
        for (j, name) in cols.iter().enumerate() {
            let col = dm.column(j);
            let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let variance: f64 = {
                let mean = col.mean().unwrap_or(0.0);
                col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64
            };
            println!(
                "    {:<35} range=[{:>8.4}, {:>8.4}]  var={:.6}",
                name, min, max, variance
            );
        }
        println!();
    }

    println!("=== Vignette 02 complete ===");
    Ok(())
}

/// Helper: print the range of each column in a design matrix.
fn print_column_ranges(dm: &Array2<f64>, col_names: &[String]) {
    for (j, name) in col_names.iter().enumerate() {
        let col = dm.column(j);
        let min = col.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = col.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!("    {:<35} range=[{:>8.4}, {:>8.4}]", name, min, max);
    }
}
