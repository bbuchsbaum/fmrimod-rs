//! Plotting utilities for fMRI HRF visualization.

use crate::error::{HrfError, Result};
use crate::hrf::Hrf;
use crate::types::TimeArray;
use ndarray::Array2;
use plotters::element::Rectangle;
use plotters::prelude::*;
use plotters::series::LineSeries;
use std::path::Path;

/// Plot configuration options.
#[derive(Debug, Clone)]
pub struct PlotConfig {
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
    /// Title
    pub title: String,
    /// X-axis label
    pub xlabel: String,
    /// Y-axis label
    pub ylabel: String,
    /// Line width
    pub line_width: u32,
    /// Show grid
    pub grid: bool,
    /// DPI for bitmap outputs
    pub dpi: f64,
}

impl Default for PlotConfig {
    fn default() -> Self {
        PlotConfig {
            width: 800,
            height: 600,
            title: String::new(),
            xlabel: "Time (s)".to_string(),
            ylabel: "Response".to_string(),
            line_width: 2,
            grid: true,
            dpi: 100.0,
        }
    }
}

/// Plot HRF curves to a file.
pub fn plot_hrf<P: AsRef<Path>>(
    hrf: &dyn Hrf,
    times: &TimeArray,
    output_path: P,
    config: Option<PlotConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    validate_plot(&config, times)?;
    let values = hrf.evaluate(times)?;
    validate_values(&values, times.len(), hrf.nbasis())?;

    // Determine output format from extension
    let path = output_path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "svg" => plot_hrf_svg(times, &values, path, &config),
        "png" => plot_hrf_bitmap(times, &values, path, &config),
        "html" => plot_hrf_html(times, &values, path, &config, hrf.name()),
        _ => Err(HrfError::InvalidInput(
            "plot extension must be png, svg, or html".into(),
        )),
    }
}

/// Plot multiple HRFs for comparison.
pub fn plot_hrf_comparison<P: AsRef<Path>>(
    hrfs: &[(String, Box<dyn Hrf>)],
    times: &TimeArray,
    output_path: P,
    config: Option<PlotConfig>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    let path = output_path.as_ref();

    validate_plot(&config, times)?;
    if hrfs.is_empty() {
        return Err(HrfError::InvalidInput(
            "comparison needs at least one HRF".into(),
        ));
    }
    // Collect all HRF values
    let mut all_values = Vec::new();
    for (name, hrf) in hrfs {
        let values = hrf.evaluate(times)?;
        validate_values(&values, times.len(), hrf.nbasis())?;
        all_values.push((name.clone(), values));
    }

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "svg" => plot_multiple_svg(times, &all_values, path, &config),
        "png" => plot_multiple_bitmap(times, &all_values, path, &config),
        "html" => plot_multiple_html(times, &all_values, path, &config),
        _ => Err(HrfError::InvalidInput(
            "plot extension must be png, svg, or html".into(),
        )),
    }
}

/// Plot design matrix as a heatmap.
pub fn plot_design_matrix<P: AsRef<Path>>(
    design: &Array2<f64>,
    output_path: P,
    config: Option<PlotConfig>,
    column_names: Option<&[String]>,
) -> Result<()> {
    let config = config.unwrap_or_default();
    validate_config(&config)?;
    if design.is_empty() || design.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::InvalidInput(
            "plot matrix must be nonempty and finite".into(),
        ));
    }
    if column_names.is_some_and(|names| names.len() != design.ncols()) {
        return Err(HrfError::InvalidInput(
            "column names must match the matrix".into(),
        ));
    }
    let path = output_path.as_ref();

    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match ext {
        "svg" => plot_matrix_svg(design, path, &config, column_names),
        "png" => plot_matrix_bitmap(design, path, &config, column_names),
        "html" => plot_matrix_html(design, path, &config, column_names),
        _ => Err(HrfError::InvalidInput(
            "plot extension must be png, svg, or html".into(),
        )),
    }
}

fn validate_config(config: &PlotConfig) -> Result<()> {
    if config.width == 0 || config.height == 0 {
        return Err(HrfError::InvalidInput(
            "plot dimensions must be positive".into(),
        ));
    }
    Ok(())
}
fn validate_plot(config: &PlotConfig, times: &TimeArray) -> Result<()> {
    validate_config(config)?;
    if times.len() < 2
        || times.iter().any(|v| !v.is_finite())
        || times.windows(2).into_iter().any(|w| w[1] <= w[0])
    {
        return Err(HrfError::InvalidInput(
            "plot times must be finite, increasing, and contain at least two points".into(),
        ));
    }
    Ok(())
}
fn validate_values(values: &Array2<f64>, rows: usize, cols: usize) -> Result<()> {
    if cols == 0 || values.dim() != (rows, cols) || values.iter().any(|v| !v.is_finite()) {
        return Err(HrfError::InvalidInput(
            "HRF plot values must be finite and match the advertised shape".into(),
        ));
    }
    Ok(())
}
fn value_range(values: &Array2<f64>) -> std::ops::Range<f64> {
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if min == max {
        min - 0.5..max + 0.5
    } else {
        min..max
    }
}

// Helper macro to convert plotting errors
macro_rules! plot_err {
    ($e:expr) => {
        $e.map_err(|e| HrfError::PlottingError(format!("{}", e)))
    };
}

// Helper functions for different output formats

fn plot_hrf_svg(
    times: &TimeArray,
    values: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
) -> Result<()> {
    let root = SVGBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(times[0]..times[times.len() - 1], value_range(values))
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    if config.grid {
        plot_err!(chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw())?;
    }

    // Plot each basis function
    for col_idx in 0..values.ncols() {
        let data: Vec<(f64, f64)> = times
            .iter()
            .zip(values.column(col_idx))
            .map(|(&t, &v)| (t, v))
            .collect();

        plot_err!(chart.draw_series(LineSeries::new(data, Palette99::pick(col_idx),)))?
            .label(format!("Basis {}", col_idx + 1))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 10, y)], Palette99::pick(col_idx))
            });
    }

    if values.ncols() > 1 {
        plot_err!(chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw())?;
    }

    plot_err!(root.present())?;
    Ok(())
}

fn plot_hrf_bitmap(
    times: &TimeArray,
    values: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
) -> Result<()> {
    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(times[0]..times[times.len() - 1], value_range(values))
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    if config.grid {
        plot_err!(chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw())?;
    }

    // Plot each basis function
    for col_idx in 0..values.ncols() {
        let data: Vec<(f64, f64)> = times
            .iter()
            .zip(values.column(col_idx))
            .map(|(&t, &v)| (t, v))
            .collect();

        plot_err!(chart.draw_series(LineSeries::new(data, Palette99::pick(col_idx),)))?
            .label(format!("Basis {}", col_idx + 1))
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 10, y)], Palette99::pick(col_idx))
            });
    }

    if values.ncols() > 1 {
        plot_err!(chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw())?;
    }

    plot_err!(root.present())?;
    Ok(())
}

fn plot_hrf_html(
    times: &TimeArray,
    values: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
    hrf_name: &str,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut traces = Vec::new();

    // Create traces for each basis function
    for col_idx in 0..values.ncols() {
        let y_data: Vec<f64> = values.column(col_idx).to_vec();
        let trace = plotly::Scatter::new(times.to_vec(), y_data)
            .name(format!("Basis {}", col_idx + 1))
            .mode(plotly::common::Mode::Lines);
        traces.push(trace);
    }

    let layout = plotly::Layout::new()
        .title(plotly::common::Title::new(&config.title))
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::new(&config.xlabel)))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::new(&config.ylabel)))
        .height(config.height as usize)
        .width(config.width as usize);

    let mut plot = plotly::Plot::new();
    for trace in traces {
        plot.add_trace(trace);
    }
    plot.set_layout(layout);

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{} - {}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>{}</h1>
    <div id="plot"></div>
    <script>
        {}
    </script>
</body>
</html>"#,
        hrf_name,
        config.title,
        config.title,
        plot.to_inline_html(Some("plot"))
    );

    let mut file = File::create(path)?;
    file.write_all(html.as_bytes())?;

    Ok(())
}

fn plot_multiple_svg(
    times: &TimeArray,
    all_values: &[(String, Array2<f64>)],
    path: &Path,
    config: &PlotConfig,
) -> Result<()> {
    let root = SVGBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    // Find global min/max
    let (y_min, y_max) = all_values.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), (_, values)| {
            (
                min.min(values.fold(f64::INFINITY, |a, &b| a.min(b))),
                max.max(values.fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            )
        },
    );

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(times[0]..times[times.len() - 1], y_min..y_max)
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    if config.grid {
        plot_err!(chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw())?;
    }

    // Plot each HRF
    for (idx, (name, values)) in all_values.iter().enumerate() {
        // For multi-basis HRFs, just plot the first basis
        let data: Vec<(f64, f64)> = times
            .iter()
            .zip(values.column(0))
            .map(|(&t, &v)| (t, v))
            .collect();

        plot_err!(chart.draw_series(LineSeries::new(data, Palette99::pick(idx),)))?
            .label(name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 10, y)], Palette99::pick(idx))
            });
    }

    plot_err!(chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw())?;

    plot_err!(root.present())?;
    Ok(())
}

fn plot_multiple_bitmap(
    times: &TimeArray,
    all_values: &[(String, Array2<f64>)],
    path: &Path,
    config: &PlotConfig,
) -> Result<()> {
    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    // Find global min/max
    let (y_min, y_max) = all_values.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY),
        |(min, max), (_, values)| {
            (
                min.min(values.fold(f64::INFINITY, |a, &b| a.min(b))),
                max.max(values.fold(f64::NEG_INFINITY, |a, &b| a.max(b))),
            )
        },
    );

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(times[0]..times[times.len() - 1], y_min..y_max)
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    if config.grid {
        plot_err!(chart
            .configure_mesh()
            .x_desc(&config.xlabel)
            .y_desc(&config.ylabel)
            .draw())?;
    }

    // Plot each HRF
    for (idx, (name, values)) in all_values.iter().enumerate() {
        // For multi-basis HRFs, just plot the first basis
        let data: Vec<(f64, f64)> = times
            .iter()
            .zip(values.column(0))
            .map(|(&t, &v)| (t, v))
            .collect();

        plot_err!(chart.draw_series(LineSeries::new(data, Palette99::pick(idx),)))?
            .label(name)
            .legend(move |(x, y)| {
                PathElement::new(vec![(x, y), (x + 10, y)], Palette99::pick(idx))
            });
    }

    plot_err!(chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw())?;

    plot_err!(root.present())?;
    Ok(())
}

fn plot_multiple_html(
    times: &TimeArray,
    all_values: &[(String, Array2<f64>)],
    path: &Path,
    config: &PlotConfig,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let mut traces = Vec::new();

    // Create traces for each HRF
    for (name, values) in all_values {
        // For multi-basis HRFs, just plot the first basis
        let y_data: Vec<f64> = values.column(0).to_vec();
        let trace = plotly::Scatter::new(times.to_vec(), y_data)
            .name(name)
            .mode(plotly::common::Mode::Lines);
        traces.push(trace);
    }

    let layout = plotly::Layout::new()
        .title(plotly::common::Title::new(&config.title))
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::new(&config.xlabel)))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::new(&config.ylabel)))
        .height(config.height as usize)
        .width(config.width as usize);

    let mut plot = plotly::Plot::new();
    for trace in traces {
        plot.add_trace(trace);
    }
    plot.set_layout(layout);

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>{}</h1>
    <div id="plot"></div>
    <script>
        {}
    </script>
</body>
</html>"#,
        config.title,
        config.title,
        plot.to_inline_html(Some("plot"))
    );

    let mut file = File::create(path)?;
    file.write_all(html.as_bytes())?;

    Ok(())
}

fn plot_matrix_svg(
    matrix: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
    _column_names: Option<&[String]>,
) -> Result<()> {
    let root = SVGBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    let (rows, cols) = matrix.dim();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..cols as f64, 0f64..rows as f64)
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    plot_err!(chart
        .configure_mesh()
        .x_desc("Columns")
        .y_desc("Time Points")
        .y_label_formatter(&|y| format!("{:.0}", rows as f64 - y))
        .draw())?;

    // Find min/max for color scaling
    let min_val = matrix.fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = matrix.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    // Draw heatmap
    let cell_size = 1.0;
    for ((row, col), &value) in matrix.indexed_iter() {
        let norm_value = if range > 0.0 {
            (value - min_val) / range
        } else {
            0.5
        };

        let color = if norm_value < 0.5 {
            RGBColor(
                (255.0 * (2.0 * norm_value)) as u8,
                (255.0 * (2.0 * norm_value)) as u8,
                255,
            )
        } else {
            RGBColor(
                255,
                (255.0 * (2.0 * (1.0 - norm_value))) as u8,
                (255.0 * (2.0 * (1.0 - norm_value))) as u8,
            )
        };

        plot_err!(chart.draw_series(std::iter::once(Rectangle::new(
            [
                (col as f64, (rows - row - 1) as f64),
                (col as f64 + cell_size, (rows - row) as f64),
            ],
            color.filled()
        ))))?;
    }

    plot_err!(root.present())?;
    Ok(())
}

fn plot_matrix_bitmap(
    matrix: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
    _column_names: Option<&[String]>,
) -> Result<()> {
    let root = BitMapBackend::new(path, (config.width, config.height)).into_drawing_area();
    plot_err!(root.fill(&WHITE))?;

    let (rows, cols) = matrix.dim();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(40)
        .build_cartesian_2d(0f64..cols as f64, 0f64..rows as f64)
        .map_err(|e| HrfError::PlottingError(format!("Failed to build chart: {}", e)))?;

    plot_err!(chart
        .configure_mesh()
        .x_desc("Columns")
        .y_desc("Time Points")
        .y_label_formatter(&|y| format!("{:.0}", rows as f64 - y))
        .draw())?;

    // Find min/max for color scaling
    let min_val = matrix.fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = matrix.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max_val - min_val;

    // Draw heatmap
    let cell_size = 1.0;
    for ((row, col), &value) in matrix.indexed_iter() {
        let norm_value = if range > 0.0 {
            (value - min_val) / range
        } else {
            0.5
        };

        let color = if norm_value < 0.5 {
            RGBColor(
                (255.0 * (2.0 * norm_value)) as u8,
                (255.0 * (2.0 * norm_value)) as u8,
                255,
            )
        } else {
            RGBColor(
                255,
                (255.0 * (2.0 * (1.0 - norm_value))) as u8,
                (255.0 * (2.0 * (1.0 - norm_value))) as u8,
            )
        };

        plot_err!(chart.draw_series(std::iter::once(Rectangle::new(
            [
                (col as f64, (rows - row - 1) as f64),
                (col as f64 + cell_size, (rows - row) as f64),
            ],
            color.filled()
        ))))?;
    }

    plot_err!(root.present())?;
    Ok(())
}

fn plot_matrix_html(
    matrix: &Array2<f64>,
    path: &Path,
    config: &PlotConfig,
    column_names: Option<&[String]>,
) -> Result<()> {
    use std::fs::File;
    use std::io::Write;

    let (rows, cols) = matrix.dim();

    // Convert matrix to 2D vector for plotly
    let z_data: Vec<Vec<f64>> = (0..rows)
        .map(|row| (0..cols).map(|col| matrix[[row, col]]).collect())
        .collect();

    let y_labels: Vec<String> = (0..rows).map(|i| format!("{}", i)).collect();
    let x_labels: Vec<String> = if let Some(names) = column_names {
        names.to_vec()
    } else {
        (0..cols).map(|i| format!("Col {}", i)).collect()
    };

    let trace = plotly::HeatMap::new(x_labels.clone(), y_labels.clone(), z_data).color_scale(
        plotly::common::ColorScale::Palette(plotly::common::ColorScalePalette::Blues),
    );

    let layout = plotly::Layout::new()
        .title(plotly::common::Title::new(&config.title))
        .x_axis(plotly::layout::Axis::new().title(plotly::common::Title::new("Columns")))
        .y_axis(plotly::layout::Axis::new().title(plotly::common::Title::new("Time Points")))
        .height(config.height as usize)
        .width(config.width as usize);

    let mut plot = plotly::Plot::new();
    plot.add_trace(trace);
    plot.set_layout(layout);

    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>{}</h1>
    <div id="plot"></div>
    <script>
        {}
    </script>
</body>
</html>"#,
        config.title,
        config.title,
        plot.to_inline_html(Some("plot"))
    );

    let mut file = File::create(path)?;
    file.write_all(html.as_bytes())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hrf::spm_canonical;
    use ndarray::Array1;
    use tempfile::tempdir;

    #[test]
    fn test_plot_hrf_svg() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_hrf.svg");

        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 20.0, 201);

        let config = PlotConfig {
            title: "Test HRF".to_string(),
            ..Default::default()
        };

        plot_hrf(hrf.as_ref(), &times, &path, Some(config)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_comparison() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_comparison.html");

        let hrfs = vec![
            (
                "SPM Canonical".to_string(),
                crate::hrf::spm_canonical(None, None, None, None, None, Some(true)),
            ),
            (
                "Gamma".to_string(),
                crate::hrf::gamma_hrf(6.0, 1.0, Some(true)),
            ),
        ];

        let times = Array1::linspace(0.0, 20.0, 201);

        let config = PlotConfig {
            title: "HRF Comparison".to_string(),
            ..Default::default()
        };

        plot_hrf_comparison(&hrfs, &times, &path, Some(config)).unwrap();
        assert!(path.exists());
    }

    #[test]
    fn test_plot_edge_cases() {
        let dir = tempdir().unwrap();

        // Test 1: Plot with single time point
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::from_vec(vec![5.0]);
        let path1 = dir.path().join("single_point.svg");

        let result = plot_hrf(hrf.as_ref(), &times, &path1, None);
        // Should handle gracefully even if plot might be degenerate
        assert!(result.is_err());

        // Test 2: Plot with NaN values
        #[derive(Debug)]
        struct NanHrf;
        impl crate::hrf::Hrf for NanHrf {
            fn evaluate(
                &self,
                t: &crate::types::TimeArray,
            ) -> crate::error::Result<crate::types::HrfValues> {
                let mut values = Array1::zeros(t.len());
                values[t.len() / 2] = f64::NAN;
                Ok(values.insert_axis(ndarray::Axis(1)))
            }
            fn name(&self) -> &str {
                "nan_hrf"
            }
            fn nbasis(&self) -> usize {
                1
            }
            fn span(&self) -> f64 {
                10.0
            }
            fn params(&self) -> Vec<(&str, f64)> {
                vec![]
            }
        }

        let nan_hrf = NanHrf;
        let times2 = Array1::linspace(0.0, 10.0, 100);
        let path2 = dir.path().join("nan_values.png");
        let result2 = plot_hrf(&nan_hrf, &times2, &path2, None);
        // Should handle NaN gracefully
        assert!(result2.is_err());

        // Test 3: Empty HRF list for comparison
        let empty_hrfs: Vec<(String, Box<dyn crate::hrf::Hrf>)> = vec![];
        let path3 = dir.path().join("empty_comparison.svg");
        let result3 = plot_hrf_comparison(&empty_hrfs, &times2, &path3, None);
        // Should handle empty list
        assert!(result3.is_err());

        // Test 4: Design matrix with extreme dimensions
        let large_matrix = Array2::zeros((1000, 100));
        let path4 = dir.path().join("large_matrix.png");
        let result4 = plot_design_matrix(&large_matrix, &path4, None, None);
        assert!(result4.is_ok(), "{result4:?}");

        // Test 5: Very long column names
        let matrix = Array2::ones((10, 3));
        let long_names = vec!["A".repeat(100), "B".repeat(200), "C".repeat(300)];
        let path5 = dir.path().join("long_names.html");
        let result5 = plot_design_matrix(&matrix, &path5, None, Some(&long_names));
        assert!(result5.is_ok());
    }

    #[test]
    fn test_plot_file_extensions() {
        let dir = tempdir().unwrap();
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 20.0, 100);

        // Test various file extensions
        let extensions = vec!["png", "svg", "html", "pdf", "jpg", ""];

        for (i, ext) in extensions.iter().enumerate() {
            let filename = if ext.is_empty() {
                format!("test{}", i)
            } else {
                format!("test{}.{}", i, ext)
            };

            let path = dir.path().join(filename);
            let result = plot_hrf(hrf.as_ref(), &times, &path, None);

            // Only explicitly supported formats are accepted.
            assert_eq!(
                result.is_ok(),
                matches!(*ext, "png" | "svg" | "html"),
                "{ext}: {result:?}"
            );
        }
    }

    #[test]
    fn test_plot_config_validation() {
        let dir = tempdir().unwrap();
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 20.0, 100);

        // Test with zero dimensions
        let config1 = PlotConfig {
            width: 0,
            height: 0,
            ..Default::default()
        };
        let path1 = dir.path().join("zero_dim.png");
        let result1 = plot_hrf(hrf.as_ref(), &times, &path1, Some(config1));
        // Should handle or error gracefully
        assert!(result1.is_err());

        // Test with extreme line width
        let config2 = PlotConfig {
            line_width: 1000,
            ..Default::default()
        };
        let path2 = dir.path().join("thick_line.svg");
        let result2 = plot_hrf(hrf.as_ref(), &times, &path2, Some(config2));
        assert!(result2.is_ok());
    }
}
