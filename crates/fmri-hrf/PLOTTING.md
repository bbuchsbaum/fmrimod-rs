# Plotting Guide for fmrihrf-rs

This guide explains how to create plots and visualizations with fmrihrf-rs.

## Setup

Add the `plotting` feature to your `Cargo.toml`:

```toml
[dependencies]
fmrihrf = { version = "0.1.0", features = ["plotting"] }
```

## Output Formats

### HTML (Recommended)
Interactive plots using Plotly.js. Best for exploring data.

```rust
use fmri_hrf::plotting::{plot_hrf, PlotConfig};

let config = PlotConfig {
    title: "My HRF Plot".to_string(),
    width: 800,
    height: 600,
    ..Default::default()
};

plot_hrf(hrf.as_ref(), &times, "output/plot.html", Some(config))?;
```

**Advantages:**
- Interactive (zoom, pan, hover tooltips)
- No system dependencies
- Works everywhere
- Exports to various formats from browser

### SVG
Vector graphics for publications.

```rust
plot_hrf(hrf.as_ref(), &times, "output/plot.svg", Some(config))?;
```

**Note:** May have font rendering issues on some systems. If you encounter font errors, use HTML format instead.

### PNG
Bitmap images. Requires system fonts.

```rust
plot_hrf(hrf.as_ref(), &times, "output/plot.png", Some(config))?;
```

**Note:** Currently has font loading issues on many systems. Use HTML or SVG instead.

## Plot Types

### 1. Single HRF Plot

```rust
use fmri_hrf::hrf::spm_canonical;
use fmri_hrf::plotting::{plot_hrf, PlotConfig};
use ndarray::Array1;

let hrf = spm_canonical(None, None, None, None, None, Some(true));
let times = Array1::linspace(0.0, 30.0, 301);

let config = PlotConfig {
    title: "SPM Canonical HRF".to_string(),
    ylabel: "BOLD Response".to_string(),
    ..Default::default()
};

plot_hrf(hrf.as_ref(), &times, "hrf.html", Some(config))?;
```

### 2. HRF Comparison

```rust
use fmri_hrf::plotting::plot_hrf_comparison;

let hrfs = vec![
    ("SPM".to_string(), spm_canonical(...)),
    ("Gamma".to_string(), gamma_hrf(6.0, 1.0, Some(true))),
];

plot_hrf_comparison(&hrfs, &times, "comparison.html", Some(config))?;
```

### 3. Design Matrix Heatmap

```rust
use fmri_hrf::plotting::plot_design_matrix;
use fmri_hrf::regressor::{design_matrix_from_set};

let dm = design_matrix_from_set(&regressor_set, &times, None)?;

plot_design_matrix(
    &dm.data,
    "design_matrix.html",
    Some(config),
    Some(&dm.column_names),
)?;
```

## Jupyter Notebook Support

Install the Rust kernel:

```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```

In a Jupyter notebook:

```rust
:dep fmrihrf = { path = "/path/to/fmrihrf-rs", features = ["plotting"] }
:dep ndarray = "0.15"

use fmri_hrf::{hrf::spm_canonical, plotting::plot_hrf};
use ndarray::Array1;

let hrf = spm_canonical(None, None, None, None, None, Some(true));
let times = Array1::linspace(0.0, 20.0, 201);

// Create plot
plot_hrf(hrf.as_ref(), &times, "output.html", None).unwrap();

// Display inline (if using evcxr features)
:html <iframe src="output.html" width="800" height="600"></iframe>
```

## Examples

See the examples directory for complete demos:
- `examples/html_plotting_demo.rs` - Interactive HTML plots
- `examples/plotting_demo_simple.rs` - SVG output examples

Run with:
```bash
cargo run --example html_plotting_demo --features plotting
```

## Troubleshooting

### Font Errors
If you see "Font loading error: FontUnavailable", use HTML format instead of PNG/SVG.

### Missing Plots
Ensure the `plotting` feature is enabled in your Cargo.toml.

### Browser Issues
HTML plots require JavaScript. Use a modern browser with JavaScript enabled.