//! Example demonstrating EventBasis for continuous parametric events
//!
//! This example shows how to use EventBasis to model continuous variables
//! (like ratings, reaction times, or physiological measures) that vary
//! across events and can have non-linear relationships with brain activity.

use fmri_design::basis::{BSplineBasis, PolynomialBasis, Scale, ScaleWithin};
use fmri_design::events::EventBasis;
use fmri_design::sampling::{SamplingFrame, SamplingFrameExt};
use fmri_design::types::{BasisFunction, EventSpec};
use ndarray::{array, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== EventBasis Demo ===\n");

    // Example 1: Linear rating modulation
    println!("1. Linear rating modulation:");

    // Simulate rating data (1-7 scale)
    let onsets = vec![5.0, 15.0, 25.0, 35.0, 45.0, 55.0];
    let ratings = vec![2.0, 5.0, 7.0, 3.0, 6.0, 1.0];
    let durations = vec![2.0; 6];

    // Create linear basis (degree 1)
    let linear_basis = PolynomialBasis::new(1).with_intercept(false);

    let rating_event = EventBasis::new(
        "rating",
        onsets.clone(),
        ratings.clone(),
        Box::new(linear_basis),
        durations.clone(),
    )?;

    println!("  Event: {}", rating_event.name());
    println!("  Number of basis functions: {}", rating_event.n_basis());
    println!("  Basis names: {:?}", rating_event.basis_names());

    // Example 2: Quadratic relationship
    println!("\n2. Quadratic rating modulation:");

    // Use polynomial basis of degree 2 to capture U-shaped relationships
    let quad_basis = PolynomialBasis::new(2).with_intercept(false);

    let quad_rating_event = EventBasis::new(
        "rating_quad",
        onsets.clone(),
        ratings.clone(),
        Box::new(quad_basis),
        durations.clone(),
    )?;

    println!("  Event: {}", quad_rating_event.name());
    println!(
        "  Number of basis functions: {}",
        quad_rating_event.n_basis()
    );
    println!("  Basis names: {:?}", quad_rating_event.basis_names());

    // Example 3: Spline basis for flexible non-linear relationships
    println!("\n3. Spline basis for flexible modeling:");

    // B-spline with 4 basis functions
    let spline_basis = BSplineBasis::new(4); // 4 basis functions

    let spline_rating_event = EventBasis::new(
        "rating_spline",
        onsets.clone(),
        ratings.clone(),
        Box::new(spline_basis),
        durations.clone(),
    )?;

    println!("  Event: {}", spline_rating_event.name());
    println!(
        "  Number of basis functions: {}",
        spline_rating_event.n_basis()
    );
    println!("  Basis names: {:?}", spline_rating_event.basis_names());

    // Example 4: Reaction time with scaling
    println!("\n4. Reaction time modeling with scaling:");

    // Simulate reaction times (in seconds)
    let rt_onsets = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let reaction_times = vec![0.45, 0.72, 0.38, 0.91, 0.55];

    // Scale RT values to z-scores
    let scale_transform = Scale::new(true, true);
    let rt_scaled = scale_transform.evaluate(&array![0.45, 0.72, 0.38, 0.91, 0.55].view())?;
    let rt_values: Vec<f64> = rt_scaled.column(0).to_vec();

    let rt_event = EventBasis::new(
        "reaction_time",
        rt_onsets.clone(),
        rt_values,
        Box::new(PolynomialBasis::new(1).with_intercept(false)),
        vec![0.0; 5], // Impulse events
    )?;

    println!("  Original RTs: {:?}", reaction_times);
    println!("  Scaled RTs: {:?}", rt_scaled.column(0).to_vec());
    println!("  Event: {}", rt_event.name());

    // Example 5: Multiple runs with ScaleWithin
    println!("\n5. Multi-run modeling with within-run scaling:");

    // Simulate data from two runs
    let all_onsets = vec![
        // Run 1
        5.0, 15.0, 25.0, 35.0, // Run 2
        105.0, 115.0, 125.0, 135.0,
    ];
    let all_ratings = vec![
        // Run 1 (lower overall)
        2.0, 3.0, 4.0, 3.0, // Run 2 (higher overall)
        5.0, 6.0, 7.0, 6.0,
    ];
    let runs = vec![
        "run1", "run1", "run1", "run1", "run2", "run2", "run2", "run2",
    ];

    // Apply within-run scaling
    let scale_within = ScaleWithin::new(runs.iter().map(|s| s.to_string()).collect(), true, true);
    let ratings_scaled =
        scale_within.evaluate(&array![2.0, 3.0, 4.0, 3.0, 5.0, 6.0, 7.0, 6.0].view())?;
    let scaled_values: Vec<f64> = ratings_scaled.column(0).to_vec();

    let multirun_event = EventBasis::new(
        "rating_multirun",
        all_onsets.clone(),
        scaled_values.clone(),
        Box::new(PolynomialBasis::new(1).with_intercept(false)),
        vec![2.0; 8],
    )?;

    println!("  Original ratings:");
    println!("    Run 1: {:?}", &all_ratings[0..4]);
    println!("    Run 2: {:?}", &all_ratings[4..8]);
    println!("  Scaled ratings:");
    println!("    Run 1: {:?}", &scaled_values[0..4]);
    println!("    Run 2: {:?}", &scaled_values[4..8]);

    // Example 6: Creating a design matrix
    println!("\n6. Design matrix generation:");

    // Create sampling frame
    let sampling_frame = SamplingFrame::from_duration(140.0, 2.0);
    let time_points_vec = sampling_frame.samples();
    let time_points = Array1::from(time_points_vec.clone());

    // Generate design matrix for the quadratic rating event
    let dm = quad_rating_event.design_matrix(&time_points)?;

    println!("  Time points: {} (TR=2s)", time_points.len());
    println!("  Design matrix shape: {:?}", dm.shape());
    println!("  Non-zero timepoints:");

    for t in 0..time_points.len() {
        let row_sum: f64 = (0..dm.ncols()).map(|c| dm[[t, c]].abs()).sum();
        if row_sum > 0.0 {
            println!(
                "    t={:.1}s: linear={:.2}, quadratic={:.2}",
                time_points[t],
                dm[[t, 0]],
                dm[[t, 1]]
            );
        }
    }

    // Example 7: Converting to DataFrame
    println!("\n7. DataFrame representation:");

    let df = rating_event.to_dataframe();
    println!("  Columns: {:?}", df.column_names());

    if let Some(onset_col) = df.get_column("onset") {
        println!("  First 3 events:");
        for i in 0..3.min(onset_col.len()) {
            println!(
                "    Event {}: onset={}, rating_value={}, rating_basis1={}",
                i + 1,
                onset_col[i],
                df.get_column("rating_value").unwrap()[i],
                df.get_column("rating_basis1").unwrap()[i]
            );
        }
    }

    // Example 8: Complex model with multiple EventBasis
    println!("\n8. Complex model with multiple parametric modulators:");

    // Create events with different basis expansions
    let confidence = EventBasis::new(
        "confidence",
        vec![8.0, 18.0, 28.0, 38.0],
        vec![0.3, 0.8, 0.5, 0.9], // Confidence ratings (0-1)
        Box::new(PolynomialBasis::new(1).with_intercept(false)),
        vec![2.0; 4],
    )?;

    let difficulty = EventBasis::new(
        "difficulty",
        vec![12.0, 22.0, 32.0, 42.0],
        vec![1.0, 3.0, 2.0, 4.0],       // Difficulty levels (1-4)
        Box::new(BSplineBasis::new(3)), // Non-linear difficulty effect
        vec![3.0; 4],
    )?;

    println!(
        "  Confidence event: {} basis functions",
        confidence.n_basis()
    );
    println!(
        "  Difficulty event: {} basis functions",
        difficulty.n_basis()
    );

    // Example 9: Orthogonalized parametric modulators
    println!("\n9. Orthogonalized parametric modulators:");

    // When you have correlated modulators (e.g., RT and accuracy)
    let trial_onsets = vec![5.0, 10.0, 15.0, 20.0, 25.0];
    let rt = vec![0.5, 0.7, 0.4, 0.8, 0.6];
    let accuracy = vec![1.0, 0.0, 1.0, 0.0, 1.0]; // Binary accuracy

    // Note: In practice, you might want to orthogonalize these
    let rt_basis = EventBasis::new(
        "rt",
        trial_onsets.clone(),
        rt,
        Box::new(PolynomialBasis::new(1).with_intercept(false)),
        vec![0.0; 5],
    )?;

    let accuracy_basis = EventBasis::new(
        "accuracy",
        trial_onsets,
        accuracy,
        Box::new(PolynomialBasis::new(1).with_intercept(false)),
        vec![0.0; 5],
    )?;

    println!("  RT modulator created");
    println!("  Accuracy modulator created");
    println!("  Note: Consider orthogonalizing correlated modulators");

    println!("\n✓ EventBasis demonstration complete!");

    Ok(())
}
