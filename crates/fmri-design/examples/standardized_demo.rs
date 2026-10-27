//! Example demonstrating the Standardized transform with fit/transform pattern
//!
//! This example shows how to use the Standardized transform to fit
//! standardization parameters on training data and apply them to new data.
//! This is particularly useful in fMRI analysis when you want to standardize
//! test data using statistics from training data.

use fmri_design::basis::{Scale, Standardized};
use fmri_design::types::BasisFunction;
use ndarray::{array, Array1};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Standardized Transform Demo ===\n");

    // Example 1: Basic fit/transform pattern
    println!("1. Fit on training data, transform test data:");

    // Training data (e.g., from baseline scans)
    let training_data = vec![10.0, 12.0, 11.0, 13.0, 10.5, 11.5, 12.5, 11.0, 10.0, 12.0];

    // Test data (e.g., from task scans)
    let test_data = vec![15.0, 8.0, 11.0, 14.0, 9.0];

    // Create and fit standardized transform
    let mut standardized = Standardized::new(true, true);
    standardized.fit(&training_data)?;

    println!("  Training data statistics:");
    println!("    Mean: {:.2}", standardized.fitted_mean().unwrap());
    println!("    Std:  {:.2}", standardized.fitted_std().unwrap());

    // Transform test data using fitted parameters
    let test_array = Array1::from(test_data.clone());
    let transformed = standardized.evaluate(&test_array.view())?;

    println!("\n  Test data transformation:");
    for (i, (&orig, &trans)) in test_data
        .iter()
        .zip(transformed.column(0).iter())
        .enumerate()
    {
        println!("    Test[{}]: {:.1} -> {:.3}", i, orig, trans);
    }

    // Example 2: Comparing with auto-fit behavior
    println!("\n2. Auto-fit behavior (when not pre-fitted):");

    let unfitted_standardized = Standardized::new(true, true);
    let auto_transformed = unfitted_standardized.evaluate(&test_array.view())?;

    println!("  When transforming without pre-fitting:");
    println!("  (uses statistics from the data being transformed)");
    for (i, (&orig, &trans)) in test_data
        .iter()
        .zip(auto_transformed.column(0).iter())
        .enumerate()
    {
        println!("    Test[{}]: {:.1} -> {:.3}", i, orig, trans);
    }

    // Example 3: Cross-validation scenario
    println!("\n3. Cross-validation scenario:");

    // Simulate k-fold cross-validation
    let all_data = vec![
        // Fold 1
        10.0, 11.0, 12.0, 13.0, // Fold 2
        20.0, 21.0, 22.0, 23.0, // Fold 3
        30.0, 31.0, 32.0, 33.0,
    ];

    let fold_size = 4;
    let n_folds = 3;

    for fold in 0..n_folds {
        println!("\n  Fold {} as test:", fold + 1);

        // Split data into train and test
        let test_start = fold * fold_size;
        let test_end = (fold + 1) * fold_size;

        let mut train_data = Vec::new();
        let mut test_fold_data = Vec::new();

        for (i, &val) in all_data.iter().enumerate() {
            if i >= test_start && i < test_end {
                test_fold_data.push(val);
            } else {
                train_data.push(val);
            }
        }

        // Fit on training folds
        let mut cv_standardized = Standardized::new(true, true);
        cv_standardized.fit(&train_data)?;

        // Transform test fold
        let test_fold_array = Array1::from(test_fold_data.clone());
        let transformed_fold = cv_standardized.evaluate(&test_fold_array.view())?;

        println!(
            "    Train mean: {:.1}, std: {:.1}",
            cv_standardized.fitted_mean().unwrap(),
            cv_standardized.fitted_std().unwrap()
        );
        println!("    Test data: {:?}", test_fold_data);
        println!(
            "    Transformed: [{:.3}, {:.3}, {:.3}, {:.3}]",
            transformed_fold[[0, 0]],
            transformed_fold[[1, 0]],
            transformed_fold[[2, 0]],
            transformed_fold[[3, 0]]
        );
    }

    // Example 4: Center-only and scale-only modes
    println!("\n4. Different standardization modes:");

    let data = vec![100.0, 110.0, 120.0, 130.0, 140.0];

    // Center only
    let mut center_only = Standardized::new(true, false);
    center_only.fit(&data)?;

    // Scale only
    let mut scale_only = Standardized::new(false, true);
    scale_only.fit(&data)?;

    // Both
    let mut both = Standardized::new(true, true);
    both.fit(&data)?;

    let test_value = array![125.0];

    println!("\n  Original value: 125.0");
    println!(
        "  Center only: {:.3}",
        center_only.evaluate(&test_value.view())?[[0, 0]]
    );
    println!(
        "  Scale only: {:.3}",
        scale_only.evaluate(&test_value.view())?[[0, 0]]
    );
    println!("  Both: {:.3}", both.evaluate(&test_value.view())?[[0, 0]]);

    // Example 5: Handling edge cases
    println!("\n5. Edge cases:");

    // Constant data (zero variance)
    let constant_data = vec![5.0, 5.0, 5.0, 5.0];
    let mut const_standardized = Standardized::new(true, true);
    const_standardized.fit(&constant_data)?;

    println!("  Constant data (all 5.0):");
    println!("    Mean: {:.1}", const_standardized.fitted_mean().unwrap());
    println!(
        "    Std: {:.1} (treated as 1.0 to avoid division by zero)",
        const_standardized.fitted_std().unwrap()
    );

    let test_const = array![5.0, 10.0];
    let trans_const = const_standardized.evaluate(&test_const.view())?;
    println!(
        "    Transform [5.0, 10.0] -> [{:.1}, {:.1}]",
        trans_const[[0, 0]],
        trans_const[[1, 0]]
    );

    // Example 6: Comparison with regular Scale
    println!("\n6. Standardized vs Scale comparison:");

    let train = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let test = array![6.0, 7.0, 8.0];

    // Standardized with fit
    let mut std_fit = Standardized::new(true, true);
    std_fit.fit(&train)?;
    let std_result = std_fit.evaluate(&test.view())?;

    // Regular Scale (always uses current data stats)
    let scale = Scale::new(true, true);
    let scale_result = scale.evaluate(&test.view())?;

    println!("  Test data: [6.0, 7.0, 8.0]");
    println!("  Standardized (fit on [1,2,3,4,5]):");
    println!(
        "    Results: [{:.3}, {:.3}, {:.3}]",
        std_result[[0, 0]],
        std_result[[1, 0]],
        std_result[[2, 0]]
    );
    println!("  Scale (uses test data stats):");
    println!(
        "    Results: [{:.3}, {:.3}, {:.3}]",
        scale_result[[0, 0]],
        scale_result[[1, 0]],
        scale_result[[2, 0]]
    );

    println!("\n✓ Standardized demonstration complete!");

    Ok(())
}
