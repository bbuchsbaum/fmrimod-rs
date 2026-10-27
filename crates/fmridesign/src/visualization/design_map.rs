//! Design matrix visualization functions.

use crate::core::EventModel;
use crate::types::ValidationError;
use plotters::prelude::*;
use std::path::Path;

// Helper macro for error conversion
macro_rules! plot_err {
    ($expr:expr) => {
        $expr.map_err(|e| ValidationError::ComputationError(e.to_string()))
    };
}

/// Configuration for design matrix visualization.
#[derive(Debug, Clone)]
pub struct DesignMapConfig {
    /// Figure size as (width, height) in pixels
    pub figsize: (u32, u32),
    /// Whether to draw separators between blocks/runs
    pub block_separators: bool,
    /// Whether to rotate x-axis labels
    pub rotate_x_text: bool,
    /// Color map to use
    pub colormap: ColorMap,
    /// Value limits for color scale (min, max)
    pub value_limits: Option<(f64, f64)>,
    /// Midpoint for diverging color scale
    pub midpoint: Option<f64>,
}

impl Default for DesignMapConfig {
    fn default() -> Self {
        Self {
            figsize: (800, 600),
            block_separators: true,
            rotate_x_text: true,
            colormap: ColorMap::Viridis,
            value_limits: None,
            midpoint: None,
        }
    }
}

/// Color map options for visualization.
#[derive(Debug, Clone, Copy)]
pub enum ColorMap {
    /// Blue to red diverging colormap
    BlueRed,
    /// Viridis sequential colormap
    Viridis,
    /// Grayscale
    Gray,
    /// Red to blue diverging (reversed)
    RedBlue,
}

impl ColorMap {
    /// Map a normalized value [0, 1] to RGB color.
    fn map(&self, value: f64) -> RGBColor {
        let v = value.clamp(0.0, 1.0);

        match self {
            ColorMap::BlueRed => {
                if v < 0.5 {
                    // Blue to white
                    let intensity = v * 2.0;
                    RGBColor((intensity * 255.0) as u8, (intensity * 255.0) as u8, 255)
                } else {
                    // White to red
                    let intensity = (1.0 - v) * 2.0;
                    RGBColor(255, (intensity * 255.0) as u8, (intensity * 255.0) as u8)
                }
            }
            ColorMap::RedBlue => {
                if v < 0.5 {
                    // Red to white
                    let intensity = v * 2.0;
                    RGBColor(255, (intensity * 255.0) as u8, (intensity * 255.0) as u8)
                } else {
                    // White to blue
                    let intensity = (1.0 - v) * 2.0;
                    RGBColor((intensity * 255.0) as u8, (intensity * 255.0) as u8, 255)
                }
            }
            ColorMap::Viridis => {
                // Simplified viridis colormap
                let r = (68.0 + v * (1.0 - 68.0 / 255.0) * 255.0) as u8;
                let g = ((1.0 + v * (231.0 / 255.0 - 1.0 / 255.0)) * 255.0) as u8;
                let b = ((84.0 + v * (55.0 / 255.0 - 84.0 / 255.0)) * 255.0) as u8;
                RGBColor(r, g, b)
            }
            ColorMap::Gray => {
                let gray = (v * 255.0) as u8;
                RGBColor(gray, gray, gray)
            }
        }
    }
}

/// Visualize design matrix as a heatmap.
///
/// Creates a heatmap visualization of the design matrix showing the
/// intensity of each regressor at each time point.
///
/// # Arguments
/// * `model` - Event model containing the design matrix to visualize
/// * `output_path` - Path to save the output image
/// * `config` - Visualization configuration
///
/// # Example
/// ```ignore
/// use fmridesign::visualization::{design_map, DesignMapConfig};
///
/// // Basic usage
/// design_map(&model, "design_matrix.png", &DesignMapConfig::default())?;
///
/// // Custom configuration
/// let config = DesignMapConfig {
///     figsize: (1200, 800),
///     colormap: ColorMap::BlueRed,
///     midpoint: Some(0.0),
///     ..Default::default()
/// };
/// design_map(&model, "design_matrix_custom.png", &config)?;
/// ```
pub fn design_map(
    model: &EventModel,
    output_path: impl AsRef<Path>,
    config: &DesignMapConfig,
) -> Result<(), ValidationError> {
    // Get design matrix
    let matrix = model.generate_design_matrix()?;

    let column_names = model.column_names();
    let (n_scans, n_regressors) = (matrix.nrows(), matrix.ncols());

    // Create drawing backend
    let root = BitMapBackend::new(output_path.as_ref(), config.figsize).into_drawing_area();

    plot_err!(root.fill(&WHITE))?;

    // Calculate layout
    let margin = 50;
    let colorbar_width = 60;
    let plot_area = root.margin(margin, margin, margin, margin);

    // Split into main plot and colorbar
    let plot_dim = plot_area.get_pixel_range();
    let plot_width = (plot_dim.0.end - plot_dim.0.start) as i32;
    let (main_area, colorbar_area) = plot_area.split_horizontally(plot_width - colorbar_width);

    // Get value range
    let (vmin, vmax) = if let Some(limits) = config.value_limits {
        limits
    } else {
        let min = matrix.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = matrix.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (min, max)
    };

    // Adjust for midpoint if specified
    let (norm_vmin, norm_vmax) = if let Some(midpoint) = config.midpoint {
        let range = (vmax - midpoint).abs().max((vmin - midpoint).abs());
        (midpoint - range, midpoint + range)
    } else {
        (vmin, vmax)
    };

    // Draw heatmap
    let main_dim = main_area.get_pixel_range();
    let cell_width = (main_dim.0.end - main_dim.0.start) as f64 / n_regressors as f64;
    let cell_height = (main_dim.1.end - main_dim.1.start) as f64 / n_scans as f64;

    for i in 0..n_scans {
        for j in 0..n_regressors {
            let value = matrix[[i, j]];
            let normalized = if norm_vmax > norm_vmin {
                (value - norm_vmin) / (norm_vmax - norm_vmin)
            } else {
                0.5
            };

            let color = config.colormap.map(normalized);

            let x0 = (j as f64 * cell_width) as i32;
            let x1 = ((j + 1) as f64 * cell_width) as i32;
            let y0 = (i as f64 * cell_height) as i32;
            let y1 = ((i + 1) as f64 * cell_height) as i32;

            plot_err!(main_area.draw(&Rectangle::new([(x0, y0), (x1, y1)], color.filled(),)))?;
        }
    }

    // Draw axes and labels
    let mut chart = ChartBuilder::on(&main_area)
        .caption(
            &format!("Design Matrix: {}", model.name),
            ("sans-serif", 20).into_font(),
        )
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..n_regressors as f64, 0f64..n_scans as f64)
        .map_err(|e| ValidationError::ComputationError(e.to_string()))?;

    // Configure mesh
    chart
        .configure_mesh()
        .x_desc("Regressors")
        .y_desc("Scan Number")
        .x_labels(n_regressors.min(20))
        .y_labels(10)
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < column_names.len() {
                // Truncate long names
                let name = &column_names[idx];
                if name.len() > 10 {
                    format!("{}...", &name[..10])
                } else {
                    name.clone()
                }
            } else {
                String::new()
            }
        })
        .draw()
        .map_err(|e| ValidationError::ComputationError(format!("Failed to draw: {}", e)))?;

    // Draw colorbar
    draw_colorbar(&colorbar_area, config.colormap, norm_vmin, norm_vmax).map_err(|e| {
        ValidationError::ComputationError(format!("Failed to draw colorbar: {}", e))
    })?;

    root.present().map_err(|e| {
        ValidationError::ComputationError(format!("Failed to present chart: {}", e))
    })?;

    Ok(())
}

/// Draw a colorbar for the heatmap.
fn draw_colorbar(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    colormap: ColorMap,
    vmin: f64,
    vmax: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    let n_steps = 100;
    let bar_width = 20;
    let margin = 10;

    let area_dim = area.get_pixel_range();
    let bar_height = (area_dim.1.end - area_dim.1.start) as f64 - 2.0 * margin as f64;
    let step_height = bar_height / n_steps as f64;

    // Draw color gradient
    for i in 0..n_steps {
        let normalized = i as f64 / (n_steps - 1) as f64;
        let color = colormap.map(1.0 - normalized); // Flip for top-to-bottom

        let y0 = (margin as f64 + i as f64 * step_height) as i32;
        let y1 = (margin as f64 + (i + 1) as f64 * step_height) as i32;

        area.draw(&Rectangle::new(
            [(margin, y0), (margin + bar_width, y1)],
            color.filled(),
        ))
        .map_err(|e| ValidationError::ComputationError(format!("Failed to draw series: {}", e)))?;
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

    area.draw(&Text::new(
        format!("{:.2}", vmax),
        (margin + bar_width + 5, margin),
        font.clone().into_text_style(area),
    ))?;

    area.draw(&Text::new(
        format!("{:.2}", vmin),
        (
            margin + bar_width + 5,
            (area_dim.1.end - area_dim.1.start) as i32 - margin - 12,
        ),
        font.clone().into_text_style(area),
    ))?;

    area.draw(&Text::new(
        format!("{:.2}", (vmax + vmin) / 2.0),
        (
            margin + bar_width + 5,
            (area_dim.1.end - area_dim.1.start) as i32 / 2 - 6,
        ),
        font.into_text_style(area),
    ))?;

    Ok(())
}

/// Plot time series of design matrix regressors.
///
/// Creates line plots showing the time course of regressors in the
/// design matrix.
///
/// # Arguments
/// * `model` - Event model containing the design matrix
/// * `output_path` - Path to save the output image
/// * `term_names` - Optional list of term names to plot (None = all)
/// * `figsize` - Figure size as (width, height) in pixels
///
/// # Example
/// ```ignore
/// use fmridesign::visualization::plot_design_matrix;
///
/// // Plot all regressors
/// plot_design_matrix(&model, "timeseries.png", None, (1000, 600))?;
///
/// // Plot specific terms only
/// plot_design_matrix(&model, "condition_timeseries.png", Some(&["condition"]), (800, 400))?;
/// ```
pub fn plot_design_matrix(
    model: &EventModel,
    output_path: impl AsRef<Path>,
    term_names: Option<&[&str]>,
    figsize: (u32, u32),
) -> Result<(), ValidationError> {
    let matrix = model.generate_design_matrix()?;
    let column_names = model.column_names();
    let time_points = model.sampling_frame.samples();

    // Select columns to plot
    let columns_to_plot: Vec<usize> = if let Some(terms) = term_names {
        // Find columns matching term names
        column_names
            .iter()
            .enumerate()
            .filter(|(_, name)| terms.iter().any(|&term| name.starts_with(term)))
            .map(|(i, _)| i)
            .collect()
    } else {
        // Plot all columns
        (0..matrix.ncols()).collect()
    };

    if columns_to_plot.is_empty() {
        return Err(ValidationError::InvalidParameter(
            "No columns found matching specified terms".to_string(),
        ));
    }

    // Create plot
    let root = BitMapBackend::new(output_path.as_ref(), figsize).into_drawing_area();

    root.fill(&WHITE).map_err(|e| {
        ValidationError::ComputationError(format!("Failed to fill background: {}", e))
    })?;

    // Find value range
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;

    for &col in &columns_to_plot {
        let column = matrix.column(col);
        y_min = y_min.min(
            *column
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
        y_max = y_max.max(
            *column
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap(),
        );
    }

    // Add some padding
    let y_range = y_max - y_min;
    y_min -= y_range * 0.1;
    y_max += y_range * 0.1;

    let mut chart = ChartBuilder::on(&root)
        .caption(
            &format!("Design Matrix Time Series: {}", model.name),
            ("sans-serif", 20).into_font(),
        )
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(40)
        .build_cartesian_2d(
            time_points[0]..time_points[time_points.len() - 1],
            y_min..y_max,
        )
        .map_err(|e| ValidationError::ComputationError(format!("Failed to build chart: {}", e)))?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Value")
        .draw()
        .map_err(|e| ValidationError::ComputationError(format!("Failed to draw: {}", e)))?;

    // Plot each column
    let colors = [&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &BLACK];

    for (idx, &col) in columns_to_plot.iter().enumerate() {
        let color = colors[idx % colors.len()];
        let column_data: Vec<(f64, f64)> = time_points
            .iter()
            .zip(matrix.column(col).iter())
            .map(|(&t, &v)| (t, v))
            .collect();

        chart
            .draw_series(LineSeries::new(column_data, color))
            .map_err(|e| {
                ValidationError::ComputationError(format!("Failed to draw series: {}", e))
            })?
            .label(&column_names[col])
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 10, y)], color));
    }

    // Draw legend if not too many columns
    if columns_to_plot.len() <= 10 {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()
            .map_err(|e| ValidationError::ComputationError(format!("Failed to draw: {}", e)))?;
    }

    root.present().map_err(|e| {
        ValidationError::ComputationError(format!("Failed to present chart: {}", e))
    })?;

    Ok(())
}

/// Create summary visualization of event model.
///
/// Generates both design matrix heatmap and time series plots.
///
/// # Arguments
/// * `model` - Event model to visualize
/// * `output_prefix` - Prefix for output files
/// * `config` - Visualization configuration
///
/// # Example
/// ```ignore
/// use fmridesign::visualization::{plot_model_summary, DesignMapConfig};
///
/// plot_model_summary(&model, "model", &DesignMapConfig::default())?;
/// // Creates: model_design_map.png, model_timeseries.png, model_correlation.png
/// ```
pub fn plot_model_summary(
    model: &EventModel,
    output_prefix: impl AsRef<str>,
    config: &DesignMapConfig,
) -> Result<(), ValidationError> {
    let prefix = output_prefix.as_ref();

    // Design matrix heatmap
    design_map(model, format!("{}_design_map.png", prefix), config)?;

    // Time series plot
    plot_design_matrix(
        model,
        format!("{}_timeseries.png", prefix),
        None,
        config.figsize,
    )?;

    // Also create correlation map
    crate::visualization::correlation_map(
        model,
        format!("{}_correlation.png", prefix),
        config.figsize,
    )?;

    Ok(())
}
