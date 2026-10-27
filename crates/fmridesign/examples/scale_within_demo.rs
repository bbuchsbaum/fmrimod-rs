//! Example demonstrating the ScaleWithin transform for group-wise scaling
//!
//! This example shows how to scale data within groups independently,
//! which is useful when you have multiple runs or sessions that should
//! be normalized separately.

use fmridesign::basis::{Scale, ScaleWithin};
use fmridesign::core::DataFrame;
use fmridesign::types::BasisFunction;
use ndarray::{array, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ScaleWithin Transform Demo ===\n");

    // Example 1: Direct usage of ScaleWithin
    println!("1. Direct ScaleWithin transformation:");

    // Create data with two groups (e.g., two scanning runs)
    let groups: Vec<String> = vec![
        "run1", "run1", "run1", "run1", "run1", "run2", "run2", "run2", "run2", "run2",
    ]
    .into_iter()
    .map(|s| s.to_string())
    .collect();

    // Data with different means and scales between runs
    let values = array![
        // Run 1: mean ~10, std ~2
        8.0, 9.0, 10.0, 11.0, 12.0, // Run 2: mean ~50, std ~10
        40.0, 45.0, 50.0, 55.0, 60.0
    ];

    // Create ScaleWithin transform
    let scale_within = ScaleWithin::new(groups.clone(), true, true);

    // Apply transformation
    let scaled = scale_within.evaluate(&values.view())?;

    println!("  Original values:");
    println!("    Run 1: {:?}", &values.slice(ndarray::s![0..5]));
    println!("    Run 2: {:?}", &values.slice(ndarray::s![5..10]));

    println!("\n  Scaled values:");
    println!("    Run 1: {:?}", &scaled.slice(ndarray::s![0..5, 0]));
    println!("    Run 2: {:?}", &scaled.slice(ndarray::s![5..10, 0]));

    // Show group statistics
    if let Some(stats) = scale_within.group_stats() {
        println!("\n  Group statistics:");
        for (group, stat) in stats {
            println!(
                "    {}: mean={:.2}, std={:.2}, n={}",
                group, stat.mean, stat.std, stat.n
            );
        }
    }

    // Example 2: Using ScaleWithin in an event model with formula
    println!("\n2. ScaleWithin in event model:");

    // Create event data with multiple runs
    let mut data = DataFrame::new();

    // Events across two runs
    let onsets = vec![1.0, 5.0, 10.0, 15.0, 21.0, 25.0, 30.0, 35.0];
    let conditions = vec!["A", "B", "A", "B", "A", "B", "A", "B"];
    let runs = vec![
        "run1", "run1", "run1", "run1", "run2", "run2", "run2", "run2",
    ];
    let durations = vec![2.0; 8];

    // Add event data
    data.insert(
        "onset",
        onsets
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.as_str())
            .collect(),
    );
    data.insert("condition", conditions);
    data.insert("run", runs.clone());
    data.insert(
        "duration",
        durations
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.as_str())
            .collect(),
    );

    // Add continuous covariate (motion) with different scales per run
    let n_scans = 40;
    let motion: Vec<f64> = (0..n_scans)
        .map(|i| {
            if i < 20 {
                // Run 1: small motion
                0.1 * (i as f64).sin()
            } else {
                // Run 2: larger motion
                0.5 * ((i - 20) as f64).sin() + 2.0
            }
        })
        .collect();

    // Add run indicators for each scan
    let run_indicators: Vec<String> = (0..n_scans)
        .map(|i| if i < 20 { "run1" } else { "run2" }.to_string())
        .collect();

    data.insert(
        "motion",
        motion
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .iter()
            .map(|s| s.as_str())
            .collect(),
    );
    data.insert(
        "scan_run",
        run_indicators.iter().map(|s| s.as_str()).collect(),
    );

    // Example 3: Manual scaling within runs
    println!("\n3. Manual within-run scaling of motion covariate:");

    let motion_array = Array1::from(motion.clone());
    let scale_within_motion = ScaleWithin::new(run_indicators, true, true);
    let scaled_motion = scale_within_motion.evaluate(&motion_array.view())?;

    println!("  Original motion (first 5 scans per run):");
    println!("    Run 1: {:?}", &motion[0..5]);
    println!("    Run 2: {:?}", &motion[20..25]);

    println!("\n  Scaled motion (first 5 scans per run):");
    println!(
        "    Run 1: {:?}",
        &scaled_motion.slice(ndarray::s![0..5, 0])
    );
    println!(
        "    Run 2: {:?}",
        &scaled_motion.slice(ndarray::s![20..25, 0])
    );

    // Show the effect of scaling
    if let Some(stats) = scale_within_motion.group_stats() {
        println!("\n  Motion statistics by run:");
        for (run, stat) in stats {
            println!("    {}: mean={:.4}, std={:.4}", run, stat.mean, stat.std);
        }
    }

    // Example 4: Comparing with global scaling
    println!("\n4. Comparison with global scaling:");

    let global_scale = Scale::new(true, true);
    let global_scaled = global_scale.evaluate(&motion_array.view())?;

    println!("  Global scaling statistics:");
    let global_mean = motion_array.mean().unwrap();
    let global_std = motion_array.std(0.0);
    println!(
        "    Overall: mean={:.4}, std={:.4}",
        global_mean, global_std
    );

    println!("\n  Effect on first scan of each run:");
    println!("    Run 1 original: {:.4}", motion[0]);
    println!("    Run 1 within-scaled: {:.4}", scaled_motion[[0, 0]]);
    println!("    Run 1 global-scaled: {:.4}", global_scaled[[0, 0]]);
    println!("    Run 2 original: {:.4}", motion[20]);
    println!("    Run 2 within-scaled: {:.4}", scaled_motion[[20, 0]]);
    println!("    Run 2 global-scaled: {:.4}", global_scaled[[20, 0]]);

    println!("\n✓ ScaleWithin demonstration complete!");

    Ok(())
}
