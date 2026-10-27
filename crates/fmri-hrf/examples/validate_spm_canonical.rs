// Validation example for SPM canonical HRF
// Generates reference values to compare against R implementation

use fmri_hrf::hrf::{spm_canonical, HrfExt};
use ndarray::Array1;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("SPM Canonical HRF Validation");
    println!("=============================\n");

    // Generate time points (0 to 30 seconds, 0.1s intervals)
    // This matches the R reference generation
    let times = Array1::linspace(0.0, 30.0, 301);

    // Generate HRF with default parameters (no normalization to see raw values)
    println!("Generating Rust SPM canonical HRF...");
    println!("Parameters: a1=6, a2=16, b1=1, b2=1, c=1/6");
    println!("Formula: gamma_pdf(t; a1, b1) - c * gamma_pdf(t; a2, b2)\n");

    let hrf = spm_canonical(None, None, None, None, None, Some(false));
    let values = hrf.evaluate(&times)?;
    let values_1d = values.column(0);

    // Find peak
    let (peak_idx, &peak_val) = values_1d
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_time = times[peak_idx];

    println!("Results:");
    println!("  Peak time:  {:.1} seconds", peak_time);
    println!("  Peak value: {:.10}", peak_val);
    println!(
        "  Min value:  {:.10}",
        values_1d.iter().fold(f64::INFINITY, |a, &b| a.min(b))
    );
    println!();

    // Print first few values for manual inspection
    println!("First 10 time points:");
    for i in 0..10 {
        println!("  t={:4.1}s: {:.10}", times[i], values_1d[i]);
    }
    println!("  ...");

    // Save to CSV
    let output_path = "/tmp/spm_canonical_rust.csv";
    let mut file = File::create(output_path)?;
    writeln!(file, "time,rust_standard")?;
    for (t, v) in times.iter().zip(values_1d.iter()) {
        writeln!(file, "{:.1},{:.15}", t, v)?;
    }

    println!("\nSaved Rust values to: {}", output_path);
    println!("\nTo compare with R:");
    println!("  cd /Users/bbuchsbaum/code/rust/fmri_hrf-rs");
    println!("  ./scripts/validate_spm_canonical.sh");

    Ok(())
}
