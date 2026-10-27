//! Correlation matrix visualization for design matrices.

use crate::core::EventModel;
use crate::types::ValidationError;
use ndarray::Array2;
use plotters::prelude::*;
use std::path::Path;

/// Visualize correlation matrix of design matrix regressors.
///
/// Creates a heatmap showing the correlation between all pairs of
/// regressors in the design matrix.
///
/// # Arguments
/// * `model` - Event model containing the design matrix
/// * `output_path` - Path to save the output image
/// * `figsize` - Figure size as (width, height) in pixels
///
/// # Example
/// ```ignore
/// use fmri_design::visualization::correlation_map;
///
/// // Basic correlation map
/// correlation_map(&model, "correlation.png", (800, 800))?;
/// ```
pub fn correlation_map(
    model: &EventModel,
    output_path: impl AsRef<Path>,
    figsize: (u32, u32),
) -> Result<(), ValidationError> {
    let matrix = model.generate_design_matrix()?;
    let column_names = model.column_names();

    // Compute correlation matrix
    let corr_matrix = compute_correlation_matrix(&matrix)?;
    let n_regressors = corr_matrix.nrows();

    // Create drawing backend
    let root = BitMapBackend::new(output_path.as_ref(), figsize).into_drawing_area();

    root.fill(&WHITE).map_err(|e| {
        ValidationError::ComputationError(format!("Failed to fill background: {}", e))
    })?;

    // Calculate layout
    let margin = 60;
    let colorbar_width = 60;
    let plot_area = root.margin(margin, margin, margin, margin);

    // Split into main plot and colorbar
    let plot_dim = plot_area.get_pixel_range();
    let plot_width = (plot_dim.0.end - plot_dim.0.start) as i32;
    let (main_area, colorbar_area) = plot_area.split_horizontally(plot_width - colorbar_width);

    // Draw correlation heatmap
    let main_dim = main_area.get_pixel_range();
    let main_width = (main_dim.0.end - main_dim.0.start) as f64;
    let main_height = (main_dim.1.end - main_dim.1.start) as f64;
    let cell_size = main_width.min(main_height) / n_regressors as f64;

    for i in 0..n_regressors {
        for j in 0..n_regressors {
            let corr = corr_matrix[[i, j]];
            let color = correlation_color(corr);

            let x0 = (j as f64 * cell_size) as i32;
            let x1 = ((j + 1) as f64 * cell_size) as i32;
            let y0 = (i as f64 * cell_size) as i32;
            let y1 = ((i + 1) as f64 * cell_size) as i32;

            main_area
                .draw(&Rectangle::new([(x0, y0), (x1, y1)], color.filled()))
                .map_err(|e| {
                    ValidationError::ComputationError(format!("Failed to draw rectangle: {}", e))
                })?;

            // Draw grid lines
            main_area
                .draw(&Rectangle::new([(x0, y0), (x1, y1)], WHITE.stroke_width(1)))
                .map_err(|e| {
                    ValidationError::ComputationError(format!("Failed to draw rectangle: {}", e))
                })?;
        }
    }

    // Add labels
    let font_size = (12.0 * (20.0 / n_regressors as f64).sqrt()).max(8.0) as u32;
    let font = ("sans-serif", font_size).into_font();

    for i in 0..n_regressors {
        let x = (i as f64 * cell_size + cell_size / 2.0) as i32;
        let y = (n_regressors as f64 * cell_size) as i32 + 5;

        // X-axis labels (bottom)
        if i < column_names.len() {
            let name = &column_names[i];
            let display_name = if name.len() > 15 {
                format!("{}...", &name[..15])
            } else {
                name.clone()
            };

            main_area
                .draw(&Text::new(
                    display_name.clone(),
                    (x - (display_name.len() as i32 * 3), y),
                    font.clone().into_text_style(&main_area),
                ))
                .map_err(|e| {
                    ValidationError::ComputationError(format!("Failed to draw rectangle: {}", e))
                })?;
        }

        // Y-axis labels (left)
        if i < column_names.len() {
            let name = &column_names[i];
            let display_name = if name.len() > 15 {
                format!("{}...", &name[..15])
            } else {
                name.clone()
            };

            let y_pos = (i as f64 * cell_size + cell_size / 2.0) as i32;
            main_area
                .draw(&Text::new(
                    display_name,
                    (-margin + 5, y_pos - 5),
                    font.clone().into_text_style(&main_area),
                ))
                .map_err(|e| {
                    ValidationError::ComputationError(format!("Failed to draw rectangle: {}", e))
                })?;
        }
    }

    // Title
    let title_font = ("sans-serif", 20).into_font();
    main_area
        .draw(&Text::new(
            format!("Regressor Correlation Matrix: {}", model.name),
            (main_width as i32 / 2 - 150, -margin + 20),
            title_font.into_text_style(&main_area),
        ))
        .map_err(|e| ValidationError::ComputationError(format!("Failed to draw title: {}", e)))?;

    // Draw colorbar
    draw_correlation_colorbar(&colorbar_area).map_err(|e| {
        ValidationError::ComputationError(format!("Failed to draw colorbar: {}", e))
    })?;

    root.present().map_err(|e| {
        ValidationError::ComputationError(format!("Failed to present chart: {}", e))
    })?;

    Ok(())
}

/// Compute correlation matrix from design matrix.
fn compute_correlation_matrix(matrix: &Array2<f64>) -> Result<Array2<f64>, ValidationError> {
    let n_cols = matrix.ncols();
    let mut corr = Array2::zeros((n_cols, n_cols));

    for i in 0..n_cols {
        for j in 0..n_cols {
            if i == j {
                corr[[i, j]] = 1.0;
            } else if i < j {
                let col_i = matrix.column(i);
                let col_j = matrix.column(j);

                // Compute correlation coefficient
                let corr_val = pearson_correlation(&col_i.to_vec(), &col_j.to_vec())?;
                corr[[i, j]] = corr_val;
                corr[[j, i]] = corr_val; // Symmetric
            }
        }
    }

    Ok(corr)
}

/// Compute Pearson correlation coefficient.
fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64, ValidationError> {
    if x.len() != y.len() {
        return Err(ValidationError::LengthMismatch {
            expected: x.len(),
            actual: y.len(),
        });
    }

    let n = x.len() as f64;

    // Compute means
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    // Compute covariance and standard deviations
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    // Avoid division by zero
    if var_x == 0.0 || var_y == 0.0 {
        return Ok(0.0);
    }

    Ok(cov / (var_x.sqrt() * var_y.sqrt()))
}

/// Map correlation value to color (blue-white-red diverging).
fn correlation_color(corr: f64) -> RGBColor {
    // Map [-1, 1] to [0, 1]
    let normalized = (corr + 1.0) / 2.0;

    if normalized < 0.5 {
        // Blue to white
        let intensity = normalized * 2.0;
        RGBColor((intensity * 255.0) as u8, (intensity * 255.0) as u8, 255)
    } else {
        // White to red
        let intensity = (1.0 - normalized) * 2.0;
        RGBColor(255, (intensity * 255.0) as u8, (intensity * 255.0) as u8)
    }
}

/// Draw colorbar for correlation values.
fn draw_correlation_colorbar(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_steps = 100;
    let bar_width = 20;
    let margin = 10;

    let area_dim = area.get_pixel_range();
    let bar_height = (area_dim.1.end - area_dim.1.start) as f64 - 2.0 * margin as f64;
    let step_height = bar_height / n_steps as f64;

    // Draw gradient from -1 to 1
    for i in 0..n_steps {
        let normalized = i as f64 / (n_steps - 1) as f64;
        let corr_value = 2.0 * (1.0 - normalized) - 1.0; // Map to [-1, 1]
        let color = correlation_color(corr_value);

        let y0 = (margin as f64 + i as f64 * step_height) as i32;
        let y1 = (margin as f64 + (i + 1) as f64 * step_height) as i32;

        area.draw(&Rectangle::new(
            [(margin, y0), (margin + bar_width, y1)],
            color.filled(),
        ))?;
    }

    // Draw border
    area.draw(&Rectangle::new(
        [
            (margin, margin),
            (
                margin + bar_width,
                (area_dim.1.end - area_dim.1.start) as i32 - margin,
            ),
        ],
        BLACK.stroke_width(1),
    ))?;

    // Add labels
    let font = ("sans-serif", 12).into_font();

    // Top: +1
    area.draw(&Text::new(
        "1.0",
        (margin + bar_width + 5, margin),
        font.clone().into_text_style(area),
    ))?;

    // Middle: 0
    area.draw(&Text::new(
        "0.0",
        (
            margin + bar_width + 5,
            (area_dim.1.end - area_dim.1.start) as i32 / 2 - 6,
        ),
        font.clone().into_text_style(area),
    ))?;

    // Bottom: -1
    area.draw(&Text::new(
        "-1.0",
        (
            margin + bar_width + 5,
            (area_dim.1.end - area_dim.1.start) as i32 - margin - 12,
        ),
        font.clone().into_text_style(area),
    ))?;

    // Label
    area.draw(&Text::new(
        "Correlation",
        (margin - 5, margin - 20),
        font.into_text_style(area),
    ))?;

    Ok(())
}
