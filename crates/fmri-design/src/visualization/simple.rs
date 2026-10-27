//! Simple visualization implementation that works with basic features.

use crate::core::EventModel;
use crate::types::ValidationError;
use ndarray::{Array2, Axis};
use std::fmt::Write as _;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn column_stats(matrix: &Array2<f64>, col_idx: usize) -> (f64, f64, f64, f64) {
    let col = matrix.column(col_idx);
    let n = col.len() as f64;

    let mean = if n > 0.0 {
        col.iter().sum::<f64>() / n
    } else {
        0.0
    };

    let variance = if n > 0.0 {
        col.iter()
            .map(|value| {
                let centered = *value - mean;
                centered * centered
            })
            .sum::<f64>()
            / n
    } else {
        0.0
    };

    let min = col.iter().fold(f64::INFINITY, |acc, value| acc.min(*value));
    let max = col
        .iter()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(*value));

    (min, max, mean, variance)
}

fn pearson_correlation(matrix: &Array2<f64>, left: usize, right: usize) -> f64 {
    let left_col = matrix.column(left);
    let right_col = matrix.column(right);

    let left_mean = left_col.mean().unwrap_or(0.0);
    let right_mean = right_col.mean().unwrap_or(0.0);

    let mut covariance = 0.0;
    let mut left_var = 0.0;
    let mut right_var = 0.0;

    for (left_value, right_value) in left_col.iter().zip(right_col.iter()) {
        let dl = *left_value - left_mean;
        let dr = *right_value - right_mean;
        covariance += dl * dr;
        left_var += dl * dl;
        right_var += dr * dr;
    }

    if left_var <= f64::EPSILON || right_var <= f64::EPSILON {
        0.0
    } else {
        covariance / (left_var.sqrt() * right_var.sqrt())
    }
}

/// Render a compact text summary for a design matrix.
pub fn design_summary(matrix: &Array2<f64>, column_names: &[String]) -> String {
    let mut output = String::new();
    let _ = writeln!(
        &mut output,
        "Design matrix summary ({} x {})",
        matrix.nrows(),
        matrix.ncols()
    );
    let _ = writeln!(
        &mut output,
        "{:<24} {:>10} {:>10} {:>10} {:>10}",
        "column", "min", "max", "mean", "variance"
    );

    for (col_idx, col_name) in column_names.iter().enumerate() {
        let (min, max, mean, variance) = column_stats(matrix, col_idx);
        let _ = writeln!(
            &mut output,
            "{:<24} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            col_name, min, max, mean, variance,
        );
    }

    output
}

/// Render a compact text summary of highly correlated column pairs.
pub fn correlation_summary(
    matrix: &Array2<f64>,
    column_names: &[String],
    threshold: f64,
) -> String {
    let mut output = String::new();
    let _ = writeln!(
        &mut output,
        "Correlation summary (|r| >= {:.3})",
        threshold.abs()
    );

    let mut matches = 0usize;
    for left in 0..matrix.len_of(Axis(1)) {
        for right in (left + 1)..matrix.len_of(Axis(1)) {
            let correlation = pearson_correlation(matrix, left, right);
            if correlation.abs() >= threshold.abs() {
                matches += 1;
                let _ = writeln!(
                    &mut output,
                    "{} <-> {}: r={:.4}",
                    column_names[left], column_names[right], correlation,
                );
            }
        }
    }

    if matches == 0 {
        let _ = writeln!(&mut output, "No column pairs exceeded the threshold.");
    }

    output
}

/// Simple design matrix visualization that outputs to CSV.
///
/// This is a basic implementation that works without complex plotting libraries.
pub fn design_matrix_to_csv(
    model: &EventModel,
    output_path: impl AsRef<Path>,
) -> Result<(), ValidationError> {
    let matrix = model.generate_design_matrix()?;

    let mut file = File::create(output_path)
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to create file: {}", e)))?;

    let column_names = model.column_names();
    let time_points = model.sampling_frame.samples();

    // Write header
    writeln!(file, "time,{}", column_names.join(","))
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write header: {}", e)))?;

    // Write data
    for (i, time) in time_points.iter().enumerate() {
        write!(file, "{}", time).map_err(|e| {
            ValidationError::FileSystemError(format!("Failed to write time: {}", e))
        })?;

        for j in 0..matrix.ncols() {
            write!(file, ",{}", matrix[[i, j]]).map_err(|e| {
                ValidationError::FileSystemError(format!("Failed to write value: {}", e))
            })?;
        }
        writeln!(file).map_err(|e| {
            ValidationError::FileSystemError(format!("Failed to write newline: {}", e))
        })?;
    }

    Ok(())
}

/// Simple correlation matrix visualization that outputs to CSV.
pub fn correlation_matrix_to_csv(
    model: &EventModel,
    output_path: impl AsRef<Path>,
) -> Result<(), ValidationError> {
    let matrix = model.generate_design_matrix()?;

    // matrix is already defined above
    let n_cols = matrix.ncols();

    // Compute correlation matrix
    let mut corr_matrix = ndarray::Array2::zeros((n_cols, n_cols));

    for i in 0..n_cols {
        for j in 0..n_cols {
            if i == j {
                corr_matrix[[i, j]] = 1.0;
            } else if i < j {
                let corr = pearson_correlation(&matrix, i, j);
                corr_matrix[[i, j]] = corr;
                corr_matrix[[j, i]] = corr;
            }
        }
    }

    // Write to CSV
    let mut file = File::create(output_path)
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to create file: {}", e)))?;

    // Get column names
    let column_names = model.column_names();

    // Write header
    writeln!(file, ",{}", column_names.join(","))
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write header: {}", e)))?;

    // Write data
    for (i, name) in column_names.iter().enumerate() {
        write!(file, "{}", name).map_err(|e| {
            ValidationError::FileSystemError(format!("Failed to write row name: {}", e))
        })?;

        for j in 0..n_cols {
            write!(file, ",{:.3}", corr_matrix[[i, j]]).map_err(|e| {
                ValidationError::FileSystemError(format!("Failed to write correlation: {}", e))
            })?;
        }
        writeln!(file).map_err(|e| {
            ValidationError::FileSystemError(format!("Failed to write newline: {}", e))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_design_summary_contains_all_column_names() {
        let matrix = array![[0.0, 1.0, 2.0], [1.0, 2.0, 4.0], [0.0, 3.0, 6.0],];
        let column_names = vec!["cond_A".to_string(), "cond_B".to_string(), "rt".to_string()];

        let summary = design_summary(&matrix, &column_names);

        assert!(summary.contains("cond_A"));
        assert!(summary.contains("cond_B"));
        assert!(summary.contains("rt"));
    }

    #[test]
    fn test_correlation_summary_detects_known_pair() {
        let matrix = array![
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0],
            [2.0, 4.0, 1.0],
            [3.0, 6.0, 0.0],
        ];
        let column_names = vec![
            "task".to_string(),
            "task_scaled".to_string(),
            "other".to_string(),
        ];

        let summary = correlation_summary(&matrix, &column_names, 0.95);

        assert!(summary.contains("task <-> task_scaled"));
        assert!(summary.contains("r=1.0000"));
    }
}

/// Simple model summary that outputs to text file.
pub fn model_summary_to_txt(
    model: &EventModel,
    output_path: impl AsRef<Path>,
) -> Result<(), ValidationError> {
    let mut file = File::create(output_path)
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to create file: {}", e)))?;

    // Write summary
    writeln!(file, "EventModel: {}\n", model.name)
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write summary: {}", e)))?;

    writeln!(file, "Number of events: {}", model.n_events())
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write info: {}", e)))?;

    writeln!(file, "Number of samples: {}", model.n_samples())
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write info: {}", e)))?;

    writeln!(file, "Number of columns: {}", model.n_columns())
        .map_err(|e| ValidationError::FileSystemError(format!("Failed to write info: {}", e)))?;

    // Write column names
    let column_names = model.column_names();
    if !column_names.is_empty() {
        writeln!(file, "\nColumn Indices:")
            .map_err(|e| ValidationError::FileSystemError(format!("Failed to write: {}", e)))?;

        writeln!(file, "\nColumn Names:")
            .map_err(|e| ValidationError::FileSystemError(format!("Failed to write: {}", e)))?;
        for (i, name) in column_names.iter().enumerate() {
            writeln!(file, "  {}: {}", i, name).map_err(|e| {
                ValidationError::FileSystemError(format!("Failed to write column name: {}", e))
            })?;
        }
    }

    Ok(())
}
