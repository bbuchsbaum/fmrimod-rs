//! Vignette 04: Contrasts and Validation
//! ========================================
//!
//! Port of R's `a_05_contrasts.Rmd`.
//!
//! Contrasts define the hypotheses we want to test in an fMRI analysis.
//! This vignette covers pair contrasts (A vs B), unit contrasts, polynomial
//! (trend) contrasts, one-way (main effect) contrasts, F-contrasts,
//! sliding-window contrasts, and validation/collinearity checks.
//!
//! Run with:
//!   cargo run --example vignette_04_contrasts

use fmridesign::contrast::spec::*;
use fmridesign::contrast::weights::{contrast_weights, Contrast as ComputedContrast, TermInfo};
use fmridesign::contrast::{
    pairwise_contrast_specs, pairwise_contrasts, sliding_window_contrast_specs, Contrast,
};
use fmridesign::core::EventModel;
use fmridesign::events::{EventFactor, EventTerm};
use fmridesign::sampling::uniform_sampling_frame;
use fmridesign::types::{ContrastSpec as OldContrastSpec, EventSpec};
use fmridesign::validation::{check_collinearity, validate_contrasts};
use ndarray::{array, Array1, Array2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vignette 04: Contrasts and Validation ===\n");

    // ---------------------------------------------------------------
    // Setup: Create a design with 4 conditions
    // ---------------------------------------------------------------
    // In R: event_model(onset ~ hrf(condition), ...)
    //
    // We build a simple 4-level design to demonstrate contrasts against.
    println!("--- Setup: 4-condition design ---\n");

    let sf = uniform_sampling_frame(200, 2.0, None)?;

    let condition_levels = ["faces", "houses", "objects", "scrambled"];
    let mut onsets = Vec::new();
    let mut conditions = Vec::new();
    let mut durations = Vec::new();

    for (i, &cond) in condition_levels.iter().enumerate() {
        for trial in 0..6 {
            onsets.push(5.0 + (i * 6 + trial) as f64 * 14.0);
            conditions.push(cond);
            durations.push(1.0);
        }
    }

    let factor = EventFactor::new("condition", onsets.clone(), conditions, durations)?;
    let term = EventTerm::new(vec![Box::new(factor.clone())], false)?;
    let model = EventModel::new(vec![Box::new(term.clone())], sf.clone())?;
    let dm = model.generate_design_matrix()?;
    let col_names = model.column_names();

    println!("  Design matrix: {} x {}", dm.nrows(), dm.ncols());
    println!("  Columns: {:?}", col_names);
    println!();

    // ---------------------------------------------------------------
    // 1. Pair contrasts (A vs B)
    // ---------------------------------------------------------------
    // In R: pair_contrast(~ condition == "faces", ~ condition == "houses", name = "faces_vs_houses")
    //
    // The most common contrast: compare the mean response between two
    // conditions. Weights are +1/n_a for condition A and -1/n_b for B.
    println!("--- Example 1: Pair contrasts ---\n");

    // Using the legacy Contrast API (simple string matching)
    let c_fh = Contrast::pair("faces_vs_houses", "faces", "houses");
    let w_fh = c_fh.compute_weights(&col_names)?;
    print_weights("  faces vs houses", &col_names, &w_fh);

    let c_fo = Contrast::pair("faces_vs_objects", "faces", "objects");
    let w_fo = c_fo.compute_weights(&col_names)?;
    print_weights("  faces vs objects", &col_names, &w_fo);

    let c_fs = Contrast::pair("faces_vs_scrambled", "faces", "scrambled");
    let w_fs = c_fs.compute_weights(&col_names)?;
    print_weights("  faces vs scrambled", &col_names, &w_fs);
    println!();

    // Using the spec-based API
    println!("  Using spec-based API (pair_contrast):");
    let spec_fh = pair_contrast("faces_vs_houses", "faces", "houses", None);
    println!(
        "    Spec: name='{}', type='{}'",
        spec_fh.name,
        spec_fh.spec_type()
    );

    // Compute weights via the TermInfo-based system
    let spec_enum = ContrastSpecEnum::Pair(spec_fh);
    let computed = contrast_weights(&spec_enum, &term)?;
    println!(
        "    Computed weights: {:?}",
        computed.weights.column(0).to_vec()
    );
    println!("    Condition names:  {:?}", computed.condnames);
    println!();

    // ---------------------------------------------------------------
    // 2. Unit contrasts
    // ---------------------------------------------------------------
    // In R: unit_contrast(~ condition == "faces", name = "faces_activation")
    //
    // A unit contrast tests whether a condition (or group of conditions)
    // is significantly different from zero. Weights sum to 1.
    println!("--- Example 2: Unit contrasts ---\n");

    let c_unit_face = Contrast::simple("faces_activation", "faces");
    let w_unit_face = c_unit_face.compute_weights(&col_names)?;
    print_weights("  faces activation (unit)", &col_names, &w_unit_face);
    println!("    Sum of weights: {:.4}", w_unit_face.sum());

    // Unit contrast for all conditions (grand mean)
    let spec_all = unit_contrast("grand_mean", None::<&str>);
    let computed_all = contrast_weights(&ContrastSpecEnum::Unit(spec_all), &term)?;
    println!("  Grand mean (unit, all conditions):");
    println!("    Weights: {:?}", computed_all.weights.column(0).to_vec());
    println!("    Sum: {:.4}", computed_all.weights.column(0).sum());
    println!();

    // ---------------------------------------------------------------
    // 3. Polynomial (trend) contrasts
    // ---------------------------------------------------------------
    // In R: poly_contrast(~ condition, degree = 2, name = "linear+quad_trend")
    //
    // Polynomial contrasts test for linear, quadratic, or higher-order
    // trends across ordered levels. Useful when conditions represent
    // graded stimulus parameters (e.g., intensity levels).
    println!("--- Example 3: Polynomial contrasts ---\n");

    let spec_poly = poly_contrast("trend", "condition", 2);
    println!(
        "  Polynomial contrast: degree={}, factor='{}'",
        spec_poly.degree, spec_poly.a.expr
    );

    let computed_poly = contrast_weights(&ContrastSpecEnum::Poly(spec_poly), &term)?;
    println!("  Weight matrix shape: {:?}", computed_poly.weights.dim());
    println!("  Conditions: {:?}", computed_poly.condnames);
    for j in 0..computed_poly.weights.ncols() {
        let degree_label = match j {
            0 => "linear",
            1 => "quadratic",
            _ => "higher",
        };
        println!(
            "    {} weights: {:?}",
            degree_label,
            computed_poly
                .weights
                .column(j)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }
    println!();

    // ---------------------------------------------------------------
    // 4. One-way (main effect) contrasts
    // ---------------------------------------------------------------
    // In R: oneway_contrast(~ condition, name = "main_effect")
    //
    // A one-way contrast tests the overall effect of a factor (are any
    // conditions different from any others?). It produces Helmert-coded
    // sub-contrasts.
    println!("--- Example 4: One-way contrasts ---\n");

    let spec_oneway = oneway_contrast("main_effect", "condition", None);
    println!("  One-way contrast: factor='{}'", spec_oneway.a.expr);

    let computed_oneway = contrast_weights(&ContrastSpecEnum::Oneway(spec_oneway), &term)?;
    println!("  Weight matrix: {:?}", computed_oneway.weights.dim());
    println!("  Conditions: {:?}", computed_oneway.condnames);
    for j in 0..computed_oneway.weights.ncols() {
        println!(
            "    Sub-contrast {}: {:?}",
            j + 1,
            computed_oneway
                .weights
                .column(j)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }
    println!();

    // ---------------------------------------------------------------
    // 5. All pairwise contrasts
    // ---------------------------------------------------------------
    // In R: pairwise_contrasts(model, ~ condition)
    //
    // Generates every possible pair contrast: faces-houses, faces-objects,
    // faces-scrambled, houses-objects, houses-scrambled, objects-scrambled.
    println!("--- Example 5: All pairwise contrasts ---\n");

    let levels_str: Vec<String> = condition_levels.iter().map(|s| s.to_string()).collect();
    let pw_specs = pairwise_contrast_specs("condition", &levels_str);
    println!("  {} pairwise contrasts generated:", pw_specs.len());
    for spec in &pw_specs {
        println!("    {}", spec.name());
    }

    // Also demonstrate the legacy API
    let pw_legacy = pairwise_contrasts("condition", &col_names);
    println!(
        "\n  Legacy pairwise_contrasts: {} contrasts",
        pw_legacy.len()
    );
    for c in &pw_legacy {
        let w = c.compute_weights(&col_names)?;
        println!("    {}: sum={:.6}", c.name(), w.sum());
    }
    println!();

    // ---------------------------------------------------------------
    // 6. F-contrasts
    // ---------------------------------------------------------------
    // In R: Fcontrast(list(c1, c2, c3), name = "omnibus")
    //
    // An F-contrast combines multiple sub-contrasts for a joint
    // (omnibus) test. This is equivalent to a multi-row contrast matrix.
    println!("--- Example 6: F-contrasts ---\n");

    // Build sub-contrasts
    let sub1 = pair_contrast("fh", "faces", "houses", None);
    let sub2 = pair_contrast("fo", "faces", "objects", None);
    let sub3 = pair_contrast("fs", "faces", "scrambled", None);

    let f_spec = f_contrast_spec(
        "faces_omnibus",
        vec![
            ContrastSpecEnum::Pair(sub1),
            ContrastSpecEnum::Pair(sub2),
            ContrastSpecEnum::Pair(sub3),
        ],
    );
    println!(
        "  F-contrast '{}': {} sub-contrasts",
        f_spec.name,
        f_spec.sub_contrasts.len()
    );

    let computed_f = contrast_weights(&ContrastSpecEnum::FContrast(f_spec), &term)?;
    println!("  Weight matrix: {:?}", computed_f.weights.dim());
    println!("  Is F-contrast: {}", computed_f.is_fcontrast());
    for j in 0..computed_f.weights.ncols() {
        println!(
            "    Column {}: {:?}",
            j,
            computed_f
                .weights
                .column(j)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }
    println!();

    // ---------------------------------------------------------------
    // 7. Sliding window contrasts
    // ---------------------------------------------------------------
    // In R: sliding_window_contrasts(~ condition, window = 1)
    //
    // Sliding-window contrasts compare adjacent levels, useful for
    // testing monotonic or step-wise changes across ordered conditions.
    println!("--- Example 7: Sliding window contrasts ---\n");

    // The spec-based sliding window generates formula expressions (e.g.,
    // condition == 'faces') which are designed for the full event_model
    // pipeline. Here we show the generated contrast specifications.
    let sw_contrasts = sliding_window_contrast_specs("condition", &levels_str);
    println!(
        "  {} sliding-window contrasts (window=1):",
        sw_contrasts.len()
    );
    for spec in &sw_contrasts {
        println!("    {}", spec.name());
    }

    // We can also build sliding windows manually using pair_contrast
    // with direct level names (matching the condition names in the term).
    println!("\n  Manual sliding-window contrasts (using pair_contrast):");
    for i in 0..condition_levels.len() - 1 {
        let a = condition_levels[i];
        let b = condition_levels[i + 1];
        let name = format!("{}_vs_{}", a, b);
        let sw_spec = pair_contrast(&name, a, b, None);
        let computed_sw = contrast_weights(&ContrastSpecEnum::Pair(sw_spec), &term)?;
        println!(
            "    {}: {:?}",
            name,
            computed_sw
                .weights
                .column(0)
                .iter()
                .map(|v| format!("{:.4}", v))
                .collect::<Vec<_>>()
        );
    }

    // Wider window
    let sw2_contrasts = sliding_window_contrasts("condition", &levels_str, 2);
    println!(
        "\n  {} sliding-window contrasts (window=2):",
        sw2_contrasts.len()
    );
    for spec in &sw2_contrasts {
        println!("    {}", spec.name());
    }
    println!();

    // ---------------------------------------------------------------
    // 8. Validate contrasts
    // ---------------------------------------------------------------
    // In R: validate_contrasts(model, contrast_list)
    //
    // Validation checks whether contrasts are estimable (in the row-space
    // of X), sum to zero, and are orthogonal to the intercept.
    println!("--- Example 8: Validate contrasts ---\n");

    // Build a simple design matrix with an intercept column
    let n = dm.nrows();
    let intercept = Array2::ones((n, 1));
    let mut x_with_intercept = Array2::zeros((n, dm.ncols() + 1));
    for j in 0..dm.ncols() {
        x_with_intercept.column_mut(j).assign(&dm.column(j));
    }
    x_with_intercept
        .column_mut(dm.ncols())
        .assign(&intercept.column(0));

    let mut x_col_names = col_names.clone();
    x_col_names.push("Intercept".to_string());

    // Validate a good pair contrast (sum-to-zero)
    let w_good = array![[1.0], [-1.0], [0.0], [0.0], [0.0]];
    let results_good = validate_contrasts(&x_with_intercept, &w_good, &x_col_names, 1e-8);
    println!("  Good contrast [1, -1, 0, 0] (faces vs houses):");
    for r in &results_good {
        println!("    estimable:              {}", r.estimable);
        println!("    sum_to_zero:            {}", r.sum_to_zero);
        println!("    orthogonal_to_intercept:{}", r.orthogonal_to_intercept);
        println!("    nonzero_weights:        {}", r.nonzero_weights);
    }

    // Validate a contrast that does NOT sum to zero
    let w_bad = array![[1.0], [0.0], [0.0], [0.0], [0.0]];
    let results_bad = validate_contrasts(&x_with_intercept, &w_bad, &x_col_names, 1e-8);
    println!("\n  Unit contrast [1, 0, 0, 0] (faces only -- does NOT sum to zero):");
    for r in &results_bad {
        println!("    estimable:              {}", r.estimable);
        println!("    sum_to_zero:            {}", r.sum_to_zero);
        println!("    orthogonal_to_intercept:{}", r.orthogonal_to_intercept);
    }

    // Validate an F-contrast matrix
    let w_f = array![[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0], [0.0, 0.0], [0.0, 0.0],];
    let results_f = validate_contrasts(&x_with_intercept, &w_f, &x_col_names, 1e-8);
    println!("\n  F-contrast (2 sub-contrasts):");
    for r in &results_f {
        println!(
            "    {} - type={}, estimable={}, sum_to_zero={}, full_rank={:?}",
            r.name, r.contrast_type, r.estimable, r.sum_to_zero, r.full_rank
        );
    }
    println!();

    // ---------------------------------------------------------------
    // 9. Check collinearity
    // ---------------------------------------------------------------
    // In R: check_collinearity(model, threshold = 0.9)
    //
    // High correlations between regressors indicate multicollinearity,
    // which inflates standard errors and makes individual contrasts
    // unreliable.
    println!("--- Example 9: Check collinearity ---\n");

    // Our well-designed matrix should have low collinearity
    let collin = check_collinearity(&dm, &col_names, 0.9);
    println!("  Design matrix collinearity check (threshold=0.9):");
    println!("    OK: {}", collin.ok);
    println!("    High-correlation pairs: {}", collin.pairs.len());
    if !collin.pairs.is_empty() {
        for (a, b, r) in &collin.pairs {
            println!("      {} <-> {}: r = {:.4}", a, b, r);
        }
    }

    // Create a deliberately collinear design for demonstration
    println!("\n  Deliberately collinear design:");
    let x_collinear = array![
        [1.0, 1.01, 0.0],
        [2.0, 2.02, 0.0],
        [3.0, 3.03, 1.0],
        [4.0, 4.04, 1.0],
        [5.0, 5.05, 2.0],
    ];
    let collinear_names = vec![
        "reg1".to_string(),
        "reg2_similar".to_string(),
        "reg3".to_string(),
    ];

    let collin_bad = check_collinearity(&x_collinear, &collinear_names, 0.9);
    println!("    OK: {}", collin_bad.ok);
    for (a, b, r) in &collin_bad.pairs {
        println!("    WARNING: {} <-> {} have r = {:.6}", a, b, r);
    }

    // Also check with a lower threshold
    let collin_strict = check_collinearity(&dm, &col_names, 0.3);
    println!("\n  Strict check (threshold=0.3):");
    println!("    High-correlation pairs: {}", collin_strict.pairs.len());
    for (a, b, r) in &collin_strict.pairs {
        println!("      {} <-> {}: r = {:.4}", a, b, r);
    }
    println!();

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    println!("--- Summary ---\n");
    println!("  Contrast types demonstrated:");
    println!("    1. Pair contrast      - Compare two conditions (A vs B)");
    println!("    2. Unit contrast      - Test single condition vs zero");
    println!("    3. Polynomial         - Linear/quadratic trends across levels");
    println!("    4. One-way (Helmert)  - Overall main effect of a factor");
    println!("    5. All pairwise       - Every combination of pairs");
    println!("    6. F-contrast         - Joint test of multiple sub-contrasts");
    println!("    7. Sliding window     - Adjacent-level comparisons");
    println!("    8. Validation         - Estimability, sum-to-zero, intercept");
    println!("    9. Collinearity       - Correlation threshold checks");
    println!();

    println!("=== Vignette 04 complete ===");
    Ok(())
}

/// Helper: print non-zero contrast weights.
fn print_weights(label: &str, col_names: &[String], weights: &Array1<f64>) {
    println!("{}:", label);
    for (name, &w) in col_names.iter().zip(weights.iter()) {
        if w.abs() > 1e-10 {
            println!("      {:<15} {:>+.4}", name, w);
        }
    }
    println!("      sum = {:.6}", weights.sum());
}
