//! DataFrame demonstration example
//!
//! Shows how to use DataFrames with conditional compilation

#[cfg(feature = "dataframe")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmridesign::prelude::*;

    println!("🐻‍❄️ Polars DataFrame Demo");

    // Create a DataFrame using the builder
    let mut df = DataFrameBuilder::new()
        .with_column("onset", vec![1.0, 2.0, 3.0, 4.0])?
        .with_str_column("condition", vec!["A", "B", "A", "B"])?
        .build();

    println!(
        "✅ Created DataFrame with {} rows and {} columns",
        df.len(),
        df.column_names().len()
    );
    println!("📋 Columns: {:?}", df.column_names());

    // Add another column
    df.add_column("duration", vec![1.0, 1.0, 1.0, 1.0])?;
    println!("➕ Added duration column: {:?}", df.column_names());

    // Get data back
    if let Some(onsets) = df.get_column("onset") {
        println!("🎯 Onsets: {:?}", onsets);
    }

    if let Some(conditions) = df.get_str_column("condition") {
        println!("🏷️  Conditions: {:?}", conditions);
    }

    // Filter data
    let filtered = df.filter(&[true, false, true, false])?;
    println!("🔍 Filtered to {} rows", filtered.len());

    // Get unique values
    let unique_conditions = df.unique("condition")?;
    println!("🔤 Unique conditions: {:?}", unique_conditions);

    println!("✨ DataFrame operations completed successfully!");

    Ok(())
}

#[cfg(not(feature = "dataframe"))]
fn main() {
    println!("📝 Basic DataFrame Demo");
    use fmridesign::core::DataFrame;

    let mut df = DataFrame::new();

    if let Ok(()) = df.add_column("onset", vec![1.0, 2.0, 3.0, 4.0]) {
        println!("✅ Created basic DataFrame with {} rows", df.len());
    }

    if let Ok(()) = df.add_str_column(
        "condition",
        vec![
            "A".to_string(),
            "B".to_string(),
            "A".to_string(),
            "B".to_string(),
        ],
    ) {
        println!("📋 Added condition column");
    }

    if let Some(onsets) = df.get_column("onset") {
        println!("🎯 Onsets: {:?}", onsets);
    }

    println!("✨ Basic DataFrame operations completed successfully!");
}
