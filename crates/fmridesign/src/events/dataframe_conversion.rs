//! DataFrame conversion utilities for events.

use crate::core::DataFrame;
use crate::events::{EventFactor, EventMatrix, EventVariable};
use crate::types::ValidationError;
use ndarray::Array2;

/// Trait for types that can be created from a DataFrame.
pub trait FromDataFrame: Sized {
    /// Create from a DataFrame.
    fn from_dataframe(df: &DataFrame) -> Result<Self, ValidationError>;
}

impl FromDataFrame for EventFactor {
    fn from_dataframe(df: &DataFrame) -> Result<Self, ValidationError> {
        // Required columns
        let onsets = df
            .get_column("onset")
            .ok_or_else(|| ValidationError::InvalidParameter("Missing 'onset' column".to_string()))?
            .clone();

        let durations = df
            .get_column("duration")
            .ok_or_else(|| {
                ValidationError::InvalidParameter("Missing 'duration' column".to_string())
            })?
            .clone();

        // Find the categorical column (first string column that's not onset/duration)
        let str_columns = df.string_columns();
        let (name, values) = str_columns
            .iter()
            .find(|(name, _)| *name != "onset" && *name != "duration")
            .ok_or_else(|| {
                ValidationError::InvalidParameter("No categorical column found".to_string())
            })?;

        let values: Vec<&str> = values.iter().map(|s| s.as_str()).collect();

        EventFactor::new(name.clone(), onsets, values, durations)
    }
}

impl FromDataFrame for EventVariable {
    fn from_dataframe(df: &DataFrame) -> Result<Self, ValidationError> {
        // Required columns
        let onsets = df
            .get_column("onset")
            .ok_or_else(|| ValidationError::InvalidParameter("Missing 'onset' column".to_string()))?
            .clone();

        let durations = df
            .get_column("duration")
            .ok_or_else(|| {
                ValidationError::InvalidParameter("Missing 'duration' column".to_string())
            })?
            .clone();

        // Find the value column (first numeric column that's not onset/duration)
        let numeric_columns = df.numeric_columns();
        let (name, values) = numeric_columns
            .iter()
            .find(|(name, _)| *name != "onset" && *name != "duration" && !name.ends_with("_raw"))
            .ok_or_else(|| {
                ValidationError::InvalidParameter("No value column found".to_string())
            })?;

        // Create without transformation (raw values = values)
        let mut event = EventVariable::new(name.clone(), onsets, values.clone(), durations)?;

        // Disable centering to keep original values
        event = event.with_center(false);

        Ok(event)
    }
}

impl FromDataFrame for EventMatrix {
    fn from_dataframe(df: &DataFrame) -> Result<Self, ValidationError> {
        // Required columns
        let onsets = df
            .get_column("onset")
            .ok_or_else(|| ValidationError::InvalidParameter("Missing 'onset' column".to_string()))?
            .clone();

        let durations = df
            .get_column("duration")
            .ok_or_else(|| {
                ValidationError::InvalidParameter("Missing 'duration' column".to_string())
            })?
            .clone();

        // Collect all numeric columns that aren't onset/duration
        let numeric_columns = df.numeric_columns();
        let matrix_columns: Vec<(String, Vec<f64>)> = numeric_columns
            .into_iter()
            .filter(|(name, _)| *name != "onset" && *name != "duration")
            .collect();

        if matrix_columns.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "No matrix columns found".to_string(),
            ));
        }

        let n_rows = onsets.len();
        let n_cols = matrix_columns.len();
        let column_names: Vec<String> = matrix_columns
            .iter()
            .map(|(name, _)| name.clone())
            .collect();

        // Build matrix by columns
        let mut values_vec = Vec::with_capacity(n_rows * n_cols);
        for row in 0..n_rows {
            for (_, col_values) in &matrix_columns {
                values_vec.push(col_values[row]);
            }
        }

        let values = Array2::from_shape_vec((n_rows, n_cols), values_vec).map_err(|_| {
            ValidationError::InvalidParameter("Failed to create matrix".to_string())
        })?;

        // Use first column name as event name if it has a common prefix
        let name = if column_names.len() > 1 {
            // Try to find common prefix
            let first = &column_names[0];
            let prefix_len = first
                .chars()
                .zip(column_names[1].chars())
                .take_while(|(a, b)| a == b)
                .count();

            if prefix_len > 0 {
                first[..prefix_len].trim_end_matches('_').to_string()
            } else {
                "matrix".to_string()
            }
        } else {
            column_names[0].clone()
        };

        EventMatrix::new(name, onsets, values, durations)?.with_column_names(column_names)
    }
}

/// Create events from a DataFrame based on column types.
pub fn events_from_dataframe(
    df: &DataFrame,
) -> Result<Vec<Box<dyn crate::types::EventSpec>>, ValidationError> {
    let mut events: Vec<Box<dyn crate::types::EventSpec>> = Vec::new();

    // Check for required columns
    let col_names = df.column_names();
    if !col_names.contains(&"onset".to_string()) || !col_names.contains(&"duration".to_string()) {
        return Err(ValidationError::InvalidParameter(
            "DataFrame must contain 'onset' and 'duration' columns".to_string(),
        ));
    }

    // Process string columns as categorical events
    let str_columns = df.string_columns();
    for (name, _) in &str_columns {
        if name != "onset" && name != "duration" {
            // Create a sub-DataFrame with just this column
            let mut sub_df = DataFrame::new();
            sub_df.add_column("onset", df.get_column("onset").unwrap().clone())?;
            sub_df.add_column("duration", df.get_column("duration").unwrap().clone())?;
            sub_df.add_str_column(name, df.get_str_column(name).unwrap().clone())?;

            let event = EventFactor::from_dataframe(&sub_df)?;
            events.push(Box::new(event));
        }
    }

    // Process numeric columns
    let numeric_columns = df.numeric_columns();
    let numeric_cols: Vec<String> = numeric_columns
        .keys()
        .filter(|name| *name != "onset" && *name != "duration" && !name.ends_with("_raw"))
        .cloned()
        .collect();

    if numeric_cols.len() == 1 {
        // Single numeric column - create EventVariable
        let col_name = &numeric_cols[0];
        let mut sub_df = DataFrame::new();
        sub_df.add_column("onset", df.get_column("onset").unwrap().clone())?;
        sub_df.add_column("duration", df.get_column("duration").unwrap().clone())?;
        sub_df.add_column(col_name, df.get_column(col_name).unwrap().clone())?;

        let event = EventVariable::from_dataframe(&sub_df)?;
        events.push(Box::new(event));
    } else if numeric_cols.len() > 1 {
        // Multiple numeric columns - could be matrix
        // For now, create as EventMatrix
        let event = EventMatrix::from_dataframe(df)?;
        events.push(Box::new(event));
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EventSpec;

    use crate::dataframe_polars::DataFrameBuilder;

    #[test]
    fn test_event_factor_from_dataframe() {
        let df = DataFrameBuilder::new()
            .with_column("onset", vec![1.0, 2.0, 3.0])
            .unwrap()
            .with_column("duration", vec![1.0, 1.0, 1.0])
            .unwrap()
            .with_str_column("condition", vec!["A", "B", "A"])
            .unwrap()
            .build();

        let event = EventFactor::from_dataframe(&df).unwrap();
        assert_eq!(event.name, "condition");
        assert_eq!(
            event.values,
            vec!["A".to_string(), "B".to_string(), "A".to_string()]
        );
    }

    #[test]
    fn test_event_variable_from_dataframe() {
        let df = DataFrameBuilder::new()
            .with_column("onset", vec![1.0, 2.0, 3.0])
            .unwrap()
            .with_column("duration", vec![1.0, 1.0, 1.0])
            .unwrap()
            .with_column("rating", vec![2.5, 3.5, 4.5])
            .unwrap()
            .build();

        let event = EventVariable::from_dataframe(&df).unwrap();
        assert_eq!(event.name, "rating");
        // Should preserve original values (no centering)
        assert_eq!(event.values.to_vec(), vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_roundtrip_conversion() {
        // Create event, convert to DataFrame, then back
        let original = EventFactor::new(
            "stimulus",
            vec![1.0, 2.0, 3.0],
            vec!["face", "house", "face"],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();

        let df = original.to_dataframe();
        let recovered = EventFactor::from_dataframe(&df).unwrap();

        assert_eq!(original.name, recovered.name);
        assert_eq!(original.values, recovered.values);
        assert_eq!(original.onsets, recovered.onsets);
    }
}
