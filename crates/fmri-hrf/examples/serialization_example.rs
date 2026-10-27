//! Example of using serialization features.
//!
//! Run with: cargo run --example serialization_example --features serde

#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmri_hrf::hrf::spm_canonical;
    use fmri_hrf::regressor::{regressor, Regressor, RegressorData};
    use fmri_hrf::sampling::SamplingFrame;
    use serde_json;

    println!("=== Serialization Example ===\n");

    // Example 1: Serialize a SamplingFrame
    println!("1. SamplingFrame Serialization:");
    let sampling_frame = SamplingFrame::new(
        vec![100, 100],   // Two runs of 100 scans each
        vec![2.0],        // TR = 2 seconds
        None::<Vec<f64>>, // Default start time (TR/2)
        0.1,              // Precision
    )?;

    let json = serde_json::to_string_pretty(&sampling_frame)?;
    println!("Serialized SamplingFrame:\n{}\n", json);

    // Deserialize back
    let deserialized: SamplingFrame = serde_json::from_str(&json)?;
    println!(
        "Deserialized: {} blocks, {} total scans\n",
        deserialized.n_blocks(),
        deserialized.n_scans()
    );

    // Example 2: Serialize a Regressor (via RegressorData)
    println!("2. Regressor Serialization:");
    let hrf = spm_canonical(None, None, None, None, None, Some(true));
    let regressor = regressor(
        vec![10.0, 30.0, 50.0, 70.0], // Event onsets
        Some(hrf),
        Some(vec![1.0]),                // Duration
        Some(vec![1.0, 2.0, 1.5, 1.0]), // Amplitudes
        None,
        None,
    )?;

    // Convert to serializable data
    let reg_data = regressor.to_data();
    let json = serde_json::to_string_pretty(&reg_data)?;
    println!("Serialized Regressor:\n{}\n", json);

    // Deserialize and reconstruct
    let deserialized_data: RegressorData = serde_json::from_str(&json)?;
    let reconstructed = Regressor::from_data(deserialized_data)?;
    println!(
        "Reconstructed: {} events, HRF: {}\n",
        reconstructed.onsets.len(),
        reconstructed.hrf.name()
    );

    // Example 3: Save to file
    println!("3. Saving to file:");
    let filename = "regressor_data.json";
    std::fs::write(filename, &json)?;
    println!("Saved regressor data to {}", filename);

    // Load from file
    let loaded_json = std::fs::read_to_string(filename)?;
    let loaded_data: RegressorData = serde_json::from_str(&loaded_json)?;
    let loaded_regressor = Regressor::from_data(loaded_data)?;
    println!("Loaded from file: {} events", loaded_regressor.onsets.len());

    // Clean up
    std::fs::remove_file(filename)?;
    println!("\nCleaned up temporary file.");

    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    eprintln!("This example requires the 'serde' feature.");
    eprintln!("Run with: cargo run --example serialization_example --features serde");
    std::process::exit(1);
}
