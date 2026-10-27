//! Import functionality for loading event data from files.

use crate::core::{DataFrame, EventModel};
use crate::events::{events_from_dataframe, EventFactor, EventMatrix, EventVariable};
use crate::types::{EventSpec, ValidationError};
use serde::{Deserialize, Serialize};
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Options for importing data
#[derive(Debug, Clone)]
pub struct ImportOptions {
    /// Column delimiter for CSV (default: ',')
    pub delimiter: char,
    /// Whether the first row contains headers
    pub has_headers: bool,
    /// Columns to treat as numeric (if None, auto-detect)
    pub numeric_columns: Option<Vec<String>>,
    /// Columns to treat as string (if None, auto-detect)
    pub string_columns: Option<Vec<String>>,
    /// Skip rows at the beginning
    pub skip_rows: usize,
}

impl Default for ImportOptions {
    fn default() -> Self {
        Self {
            delimiter: ',',
            has_headers: true,
            numeric_columns: None,
            string_columns: None,
            skip_rows: 0,
        }
    }
}

/// Read a CSV file into a DataFrame
pub fn read_csv<P: AsRef<Path>>(
    path: P,
    options: Option<ImportOptions>,
) -> Result<DataFrame, ValidationError> {
    let opts = options.unwrap_or_default();
    let file =
        File::open(path.as_ref()).map_err(|e| ValidationError::FileSystemError(e.to_string()))?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    // Skip initial rows if requested
    for _ in 0..opts.skip_rows {
        lines.next();
    }

    // Read headers
    let headers = if opts.has_headers {
        match lines.next() {
            Some(Ok(line)) => line
                .split(opts.delimiter)
                .map(|s| s.trim().to_string())
                .collect::<Vec<_>>(),
            Some(Err(e)) => return Err(ValidationError::FileSystemError(e.to_string())),
            None => {
                return Err(ValidationError::InvalidParameter(
                    "Empty CSV file".to_string(),
                ))
            }
        }
    } else {
        // Generate default headers
        Vec::new()
    };

    // Read data rows
    let mut data_rows: Vec<Vec<String>> = Vec::new();
    for line_result in lines {
        let line = line_result.map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        if line.trim().is_empty() {
            continue;
        }

        let values: Vec<String> = line
            .split(opts.delimiter)
            .map(|s| s.trim().to_string())
            .collect();

        // Generate headers if not provided
        if !opts.has_headers && headers.is_empty() && !data_rows.is_empty() {
            return Err(ValidationError::InvalidParameter(
                "Headers must be provided or generated before data rows".to_string(),
            ));
        }

        data_rows.push(values);
    }

    if data_rows.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "No data rows found".to_string(),
        ));
    }

    // Generate headers if needed
    let headers = if headers.is_empty() {
        (0..data_rows[0].len())
            .map(|i| format!("col{}", i))
            .collect()
    } else {
        headers
    };

    // Build DataFrame column by column
    let mut df = DataFrame::new();

    for (col_idx, header) in headers.iter().enumerate() {
        let column_data: Vec<String> = data_rows
            .iter()
            .map(|row| row.get(col_idx).cloned().unwrap_or_default())
            .collect();

        // Determine if column should be numeric or string
        let is_numeric = if let Some(ref numeric_cols) = opts.numeric_columns {
            numeric_cols.contains(header)
        } else if let Some(ref string_cols) = opts.string_columns {
            !string_cols.contains(header)
        } else {
            // Auto-detect: try parsing as numeric
            column_data
                .iter()
                .all(|s| s.is_empty() || s.parse::<f64>().is_ok())
        };

        if is_numeric {
            let numeric_data: Vec<f64> = column_data
                .iter()
                .map(|s| {
                    if s.is_empty() {
                        0.0
                    } else {
                        s.parse().unwrap_or(0.0)
                    }
                })
                .collect();
            df.add_column(header, numeric_data)?;
        } else {
            df.add_str_column(header, column_data)?;
        }
    }

    Ok(df)
}

/// JSON representation of event data
#[derive(Debug, Serialize, Deserialize)]
struct EventDataJson {
    /// Event name
    name: String,
    /// Event type (categorical, continuous, matrix)
    event_type: String,
    /// Onset times
    onsets: Vec<f64>,
    /// Event durations
    durations: Vec<f64>,
    /// Values (for categorical or continuous events)
    values: Option<Vec<serde_json::Value>>,
    /// Matrix data (for matrix events)
    matrix: Option<Vec<Vec<f64>>>,
    /// Column names (for matrix events)
    columns: Option<Vec<String>>,
}

/// Read events from a JSON file
pub fn read_json_events<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
    let file =
        File::open(path.as_ref()).map_err(|e| ValidationError::FileSystemError(e.to_string()))?;
    let reader = BufReader::new(file);

    let events_data: Vec<EventDataJson> = serde_json::from_reader(reader)
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    let mut events: Vec<Box<dyn EventSpec>> = Vec::new();

    for event_data in events_data {
        match event_data.event_type.as_str() {
            "categorical" => {
                if let Some(values) = event_data.values {
                    let string_values: Vec<&str> =
                        values.iter().map(|v| v.as_str().unwrap_or("")).collect();

                    let event = EventFactor::new(
                        event_data.name,
                        event_data.onsets,
                        string_values,
                        event_data.durations,
                    )?;
                    events.push(Box::new(event));
                }
            }
            "continuous" => {
                if let Some(values) = event_data.values {
                    let numeric_values: Vec<f64> =
                        values.iter().map(|v| v.as_f64().unwrap_or(0.0)).collect();

                    let event = EventVariable::new(
                        event_data.name,
                        event_data.onsets,
                        numeric_values,
                        event_data.durations,
                    )?;
                    events.push(Box::new(event));
                }
            }
            "matrix" => {
                if let Some(matrix_vec) = event_data.matrix {
                    // Convert Vec<Vec<f64>> to Array2<f64>
                    let n_rows = matrix_vec.len();
                    let n_cols = if n_rows > 0 { matrix_vec[0].len() } else { 0 };

                    let mut flat_data = Vec::with_capacity(n_rows * n_cols);
                    for row in &matrix_vec {
                        if row.len() != n_cols {
                            return Err(ValidationError::InvalidParameter(
                                "Inconsistent matrix column count".to_string(),
                            ));
                        }
                        flat_data.extend_from_slice(row);
                    }

                    let matrix = ndarray::Array2::from_shape_vec((n_rows, n_cols), flat_data)
                        .map_err(|_| {
                            ValidationError::InvalidParameter("Invalid matrix shape".to_string())
                        })?;

                    let mut event = EventMatrix::new(
                        event_data.name,
                        event_data.onsets,
                        matrix,
                        event_data.durations,
                    )?;

                    // Set column names if provided
                    if let Some(columns) = event_data.columns {
                        event = event.with_column_names(columns)?;
                    }

                    events.push(Box::new(event));
                }
            }
            _ => {
                return Err(ValidationError::InvalidParameter(format!(
                    "Unknown event type: {}",
                    event_data.event_type
                )));
            }
        }
    }

    Ok(events)
}

/// Read a JSON file into a DataFrame
pub fn read_json_dataframe<P: AsRef<Path>>(path: P) -> Result<DataFrame, ValidationError> {
    let file =
        File::open(path.as_ref()).map_err(|e| ValidationError::FileSystemError(e.to_string()))?;
    let reader = BufReader::new(file);

    // Read as array of records
    let records: Vec<HashMap<String, serde_json::Value>> = serde_json::from_reader(reader)
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    if records.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "Empty JSON file".to_string(),
        ));
    }

    // Get all unique keys
    let mut all_keys: Vec<String> = Vec::new();
    for record in &records {
        for key in record.keys() {
            if !all_keys.contains(key) {
                all_keys.push(key.clone());
            }
        }
    }

    // Build DataFrame column by column
    let mut df = DataFrame::new();

    for key in &all_keys {
        // Collect values for this column
        let mut numeric_values: Vec<f64> = Vec::new();
        let mut string_values: Vec<String> = Vec::new();
        let mut is_numeric = true;

        for record in &records {
            match record.get(key) {
                Some(serde_json::Value::Number(n)) => {
                    numeric_values.push(n.as_f64().unwrap_or(0.0));
                    string_values.push(n.to_string());
                }
                Some(serde_json::Value::Bool(b)) => {
                    numeric_values.push(if *b { 1.0 } else { 0.0 });
                    string_values.push(b.to_string());
                }
                Some(serde_json::Value::String(s)) => {
                    if let Ok(n) = s.parse::<f64>() {
                        numeric_values.push(n);
                    } else {
                        is_numeric = false;
                    }
                    string_values.push(s.clone());
                }
                Some(serde_json::Value::Null) | None => {
                    numeric_values.push(0.0);
                    string_values.push(String::new());
                }
                _ => {
                    is_numeric = false;
                    string_values.push(String::new());
                }
            }
        }

        if is_numeric && numeric_values.len() == records.len() {
            df.add_column(key, numeric_values)?;
        } else {
            df.add_str_column(key, string_values)?;
        }
    }

    Ok(df)
}

/// Load events from a CSV file
///
/// The CSV file should have columns for:
/// - onset: event onset times
/// - duration: event durations
/// - Additional columns for event values
///
/// # Example
/// ```ignore
/// let events = load_events_csv("events.csv", None)?;
/// ```
pub fn load_events_csv<P: AsRef<Path>>(
    path: P,
    options: Option<ImportOptions>,
) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
    let df = read_csv(path, options)?;
    events_from_dataframe(&df)
}

/// Load events from a JSON file
///
/// The JSON file can be either:
/// 1. An array of event objects with explicit type information
/// 2. An array of records that will be converted to a DataFrame
///
/// # Example
/// ```ignore
/// let events = load_events_json("events.json")?;
/// ```
pub fn load_events_json<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Box<dyn EventSpec>>, ValidationError> {
    // Try to read as explicit event format first
    match read_json_events(&path) {
        Ok(events) => Ok(events),
        Err(_) => {
            // Fall back to DataFrame format
            let df = read_json_dataframe(path)?;
            events_from_dataframe(&df)
        }
    }
}

/// Import trait for EventModel
pub trait EventModelImport {
    /// Load events from a CSV file
    fn from_csv(
        path: impl AsRef<Path>,
        options: Option<ImportOptions>,
    ) -> Result<Self, ValidationError>
    where
        Self: Sized;

    /// Load events from a JSON file
    fn from_json(path: impl AsRef<Path>) -> Result<Self, ValidationError>
    where
        Self: Sized;
}

impl EventModelImport for crate::core::EventModel {
    fn from_csv(
        path: impl AsRef<Path>,
        options: Option<ImportOptions>,
    ) -> Result<Self, ValidationError> {
        let events = load_events_csv(path, options)?;
        let sampling_frame = fmrihrf::SamplingFrame::uniform(100, 2.0, 0.1)
            .map_err(|e| ValidationError::InvalidParameter(e.to_string()))?;

        EventModel::new(events, sampling_frame)
    }

    fn from_json(path: impl AsRef<Path>) -> Result<Self, ValidationError> {
        let events = load_events_json(path)?;
        let sampling_frame = fmrihrf::SamplingFrame::uniform(100, 2.0, 0.1)
            .map_err(|e| ValidationError::InvalidParameter(e.to_string()))?;

        EventModel::new(events, sampling_frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_import_options() {
        let opts = ImportOptions {
            delimiter: '\t',
            has_headers: false,
            numeric_columns: Some(vec!["value".to_string()]),
            string_columns: Some(vec!["name".to_string()]),
            skip_rows: 1,
        };

        assert_eq!(opts.delimiter, '\t');
        assert!(!opts.has_headers);
        assert_eq!(opts.skip_rows, 1);
    }
}
