//! Export functionality for design matrices and event data.

use crate::core::{DataFrame, EventModel};
use crate::design::DesignMatrix;
use crate::types::ValidationError;
use serde::{Deserialize, Serialize};
use serde_json;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    /// CSV format
    Csv,
    /// JSON format
    Json,
    /// Tab-separated values
    Tsv,
}

/// Options for exporting data
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Include column headers in CSV/TSV
    pub include_headers: bool,
    /// Number of decimal places for floating point values
    pub precision: usize,
    /// Pretty print JSON
    pub pretty_json: bool,
    /// Delimiter for CSV (default: ',')
    pub delimiter: char,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            include_headers: true,
            precision: 6,
            pretty_json: true,
            delimiter: ',',
        }
    }
}

/// Write a design matrix to CSV file
pub fn write_csv<P: AsRef<Path>>(
    path: P,
    design_matrix: &DesignMatrix,
    options: Option<ExportOptions>,
) -> Result<(), ValidationError> {
    let opts = options.unwrap_or_default();
    let csv_string = to_csv_string(design_matrix, Some(opts))?;

    let mut file =
        File::create(path.as_ref()).map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    file.write_all(csv_string.as_bytes())
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    Ok(())
}

/// Convert a design matrix to CSV string
pub fn to_csv_string(
    design_matrix: &DesignMatrix,
    options: Option<ExportOptions>,
) -> Result<String, ValidationError> {
    let opts = options.unwrap_or_default();
    let mut output = String::new();

    // Add headers if requested
    if opts.include_headers {
        let headers = design_matrix.column_names.join(&opts.delimiter.to_string());
        output.push_str(&headers);
        output.push('\n');
    }

    // Add data rows
    let matrix = &design_matrix.data;
    for row in matrix.rows() {
        let row_str = row
            .iter()
            .map(|&val| format!("{:.prec$}", val, prec = opts.precision))
            .collect::<Vec<_>>()
            .join(&opts.delimiter.to_string());
        output.push_str(&row_str);
        output.push('\n');
    }

    Ok(output)
}

/// Write a design matrix to JSON file
pub fn write_json<P: AsRef<Path>>(
    path: P,
    design_matrix: &DesignMatrix,
    options: Option<ExportOptions>,
) -> Result<(), ValidationError> {
    let opts = options.unwrap_or_default();
    let json_string = to_json_string(design_matrix, Some(opts))?;

    let mut file =
        File::create(path.as_ref()).map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    file.write_all(json_string.as_bytes())
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    Ok(())
}

/// Serializable representation of a design matrix
#[derive(Debug, Serialize, Deserialize)]
struct DesignMatrixJson {
    /// Matrix data as nested arrays
    data: Vec<Vec<f64>>,
    /// Column names
    columns: Vec<String>,
    /// Number of rows
    n_rows: usize,
    /// Number of columns
    n_cols: usize,
    /// Metadata
    metadata: Option<serde_json::Value>,
}

/// Convert a design matrix to JSON string
pub fn to_json_string(
    design_matrix: &DesignMatrix,
    options: Option<ExportOptions>,
) -> Result<String, ValidationError> {
    let opts = options.unwrap_or_default();

    // Convert matrix to nested vectors
    let matrix = &design_matrix.data;
    let data: Vec<Vec<f64>> = matrix
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect();

    // Get column names
    let columns = design_matrix.column_names.clone();

    let json_repr = DesignMatrixJson {
        data,
        columns,
        n_rows: matrix.nrows(),
        n_cols: matrix.ncols(),
        metadata: None,
    };

    let json_string = if opts.pretty_json {
        serde_json::to_string_pretty(&json_repr)
    } else {
        serde_json::to_string(&json_repr)
    }
    .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

    Ok(json_string)
}

/// Extension trait for EventModel export
pub trait EventModelExport {
    /// Export events to CSV
    fn to_csv(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError>;

    /// Export events to JSON
    fn to_json(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError>;
}

impl EventModelExport for EventModel {
    fn to_csv(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError> {
        let opts = options.unwrap_or_default();
        let mut output = String::new();

        // Header
        if opts.include_headers {
            output.push_str(&format!(
                "event_name{}event_type{}onset{}duration{}value\n",
                opts.delimiter, opts.delimiter, opts.delimiter, opts.delimiter
            ));
        }

        // Events
        for event in self.events() {
            let name = event.name();
            let event_type = format!("{:?}", event.event_type());
            let onsets = event.onsets();
            let durations = event.durations();

            // Get values based on event type
            let values: Vec<String> = if let Some(vals) = event.values() {
                vals.to_vec()
            } else {
                vec!["".to_string(); onsets.len()]
            };

            for i in 0..onsets.len() {
                output.push_str(&format!(
                    "{}{}{}{}{:.prec$}{}{:.prec$}{}{}\n",
                    name,
                    opts.delimiter,
                    event_type,
                    opts.delimiter,
                    onsets[i],
                    opts.delimiter,
                    durations[i],
                    opts.delimiter,
                    values.get(i).unwrap_or(&"".to_string()),
                    prec = opts.precision
                ));
            }
        }

        let mut file = File::create(path.as_ref())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        file.write_all(output.as_bytes())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        Ok(())
    }

    fn to_json(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError> {
        let opts = options.unwrap_or_default();

        #[derive(Serialize)]
        struct EventJson {
            name: String,
            event_type: String,
            onsets: Vec<f64>,
            durations: Vec<f64>,
            values: Option<Vec<String>>,
        }

        #[derive(Serialize)]
        struct EventModelJson {
            name: Option<String>,
            events: Vec<EventJson>,
            sampling_rate: Option<f64>,
        }

        let events: Vec<EventJson> = self
            .events()
            .iter()
            .map(|event| EventJson {
                name: event.name().to_string(),
                event_type: format!("{:?}", event.event_type()),
                onsets: event.onsets().to_vec(),
                durations: event.durations().to_vec(),
                values: event.values().map(|v| v.to_vec()),
            })
            .collect();

        let model_json = EventModelJson {
            name: Some(self.name.clone()),
            events,
            sampling_rate: Some(1.0 / self.sampling_frame.tr[0]),
        };

        let json_string = if opts.pretty_json {
            serde_json::to_string_pretty(&model_json)
        } else {
            serde_json::to_string(&model_json)
        }
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        let mut file = File::create(path.as_ref())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        file.write_all(json_string.as_bytes())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        Ok(())
    }
}

/// Extension trait for DataFrame export
pub trait DataFrameExport {
    /// Export DataFrame to CSV
    fn to_csv(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError>;

    /// Export DataFrame to JSON
    fn to_json(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError>;
}

impl DataFrameExport for DataFrame {
    fn to_csv(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError> {
        let opts = options.unwrap_or_default();
        let mut output = String::new();

        // Header
        if opts.include_headers {
            let headers = self.column_names().join(&opts.delimiter.to_string());
            output.push_str(&headers);
            output.push('\n');
        }

        // Data
        let n_rows = self.len();
        for row_idx in 0..n_rows {
            let row_values: Vec<String> = self
                .column_names()
                .iter()
                .map(|col_name| {
                    if let Some(col) = self.get_column(col_name) {
                        if row_idx < col.len() {
                            format!("{:.prec$}", col[row_idx], prec = opts.precision)
                        } else {
                            "".to_string()
                        }
                    } else if let Some(col) = self.get_str_column(&col_name) {
                        if row_idx < col.len() {
                            col[row_idx].clone()
                        } else {
                            "".to_string()
                        }
                    } else {
                        "".to_string()
                    }
                })
                .collect();

            output.push_str(&row_values.join(&opts.delimiter.to_string()));
            output.push('\n');
        }

        let mut file = File::create(path.as_ref())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        file.write_all(output.as_bytes())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        Ok(())
    }

    fn to_json(
        &self,
        path: impl AsRef<Path>,
        options: Option<ExportOptions>,
    ) -> Result<(), ValidationError> {
        let opts = options.unwrap_or_default();

        // Convert to records format
        let mut records: Vec<serde_json::Map<String, serde_json::Value>> = Vec::new();

        let n_rows = self.len();
        for row_idx in 0..n_rows {
            let mut record = serde_json::Map::new();

            for col_name in self.column_names() {
                let value = if let Some(col) = self.get_column(&col_name) {
                    if row_idx < col.len() {
                        serde_json::Value::Number(
                            serde_json::Number::from_f64(col[row_idx])
                                .unwrap_or(serde_json::Number::from(0)),
                        )
                    } else {
                        serde_json::Value::Null
                    }
                } else if let Some(col) = self.get_str_column(&col_name) {
                    if row_idx < col.len() {
                        serde_json::Value::String(col[row_idx].clone())
                    } else {
                        serde_json::Value::Null
                    }
                } else {
                    serde_json::Value::Null
                };

                record.insert(col_name.clone(), value);
            }

            records.push(record);
        }

        let json_string = if opts.pretty_json {
            serde_json::to_string_pretty(&records)
        } else {
            serde_json::to_string(&records)
        }
        .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        let mut file = File::create(path.as_ref())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        file.write_all(json_string.as_bytes())
            .map_err(|e| ValidationError::FileSystemError(e.to_string()))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventFactor;
    use crate::sampling::{SamplingFrame, SamplingFrameExt};
    use ndarray::array;

    #[test]
    fn test_csv_export() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

        let csv = to_csv_string(&dm, None).unwrap();
        assert!(csv.contains("A,B"));
        assert!(csv.contains("1.000000,2.000000"));
        assert!(csv.contains("3.000000,4.000000"));
    }

    #[test]
    fn test_json_export() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

        let json = to_json_string(&dm, None).unwrap();
        let parsed: DesignMatrixJson = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.columns, vec!["A", "B"]);
        assert_eq!(parsed.n_rows, 2);
        assert_eq!(parsed.n_cols, 2);
        assert_eq!(parsed.data[0], vec![1.0, 2.0]);
    }

    #[test]
    fn test_custom_delimiter() {
        let matrix = array![[1.0, 2.0]];
        let dm = DesignMatrix::new(matrix, vec!["A".to_string(), "B".to_string()]);

        let opts = ExportOptions {
            delimiter: '\t',
            ..Default::default()
        };

        let csv = to_csv_string(&dm, Some(opts)).unwrap();
        assert!(csv.contains("A\tB"));
        assert!(csv.contains("1.000000\t2.000000"));
    }
}
