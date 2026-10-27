> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# fmrihrf-rs

Rust implementation of fMRI Hemodynamic Response Function (HRF) and Regressor Tools.

This is a Rust port of the R package [fmrihrf](https://github.com/bbuchsbaum/fmrihrf), providing high-performance HRF modeling and design matrix generation for fMRI analysis. The R package is the authoritative reference implementation; a Python port also exists but has diverged from the R source.

The current core parity evidence is recorded in [PORT_PARITY_AUDIT.md](PORT_PARITY_AUDIT.md).
That report distinguishes numerical parity from API availability and documents
remaining differences; the older percentage estimates are historical.

For multiple runs, prefer the borrowed `EventTable` or `ConditionEvents` API.
Run indices are zero based, onsets are local to each run, and columns are grouped
by basis and then predictor. Runs without events retain zero rows, and HRF tails
never enter another run.

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmrihrf::{hrf::spm_hrf_with_derivatives, regressor::ConditionEvents, SamplingFrame};
    let frame = SamplingFrame::new(vec![100, 80], vec![2.0], None, 0.05)?;
    let conditions = vec!["A".to_string(), "B".to_string()];
    let hrf = spm_hrf_with_derivatives();
    let design = ConditionEvents::new(&[0.13, 4.27], &conditions, &[0, 1])
        .with_durations(&[0.0, 0.15])
        .evaluate(&frame, hrf.as_ref())?;
    assert_eq!(design.data.dim(), (180, 6));
    Ok(())
}
```

`conv` / `auto` selects direct or FFT convolution from projected input density.
`direct`, `fft`, and the independent quadrature engine `loop` remain explicit
choices. Raw HRF amplitudes are the default. `normalize_hrf_with` offers typed,
fixed-reference normalization without changing scale when the query grid changes.

## Features

### Core Functionality
- ✅ **HRF Functions**: Hemodynamic response functions
  - SPM canonical (with and without derivatives)
  - Gamma (single and double)
  - Gaussian
  - B-spline basis
  - Fourier basis
  - Finite Impulse Response (FIR)
  - Mexican Hat
  - Sine basis
- ✅ **HRF Generators**: Dynamic HRF creation with decorators
  - Lag/delay
  - Block convolution
  - Normalization
- ✅ **HRF Registry**: Global registry for HRF lookup by name
- ✅ **Empirical HRF**: Create HRFs from measured data with interpolation
- ✅ **Sampling Framework**: Flexible fMRI timing specification
- ✅ **Regressor System**: Event-related design matrix creation
  - Neural input generation
  - Convolution (direct and FFT)
  - Multi-condition support
- ✅ **Penalty Matrices**: Regularization for HRF estimation
- ✅ **Reconstruction Matrices**: Basis function evaluation
- ✅ **Visualization** (optional feature):
  - Static plots (PNG, SVG) via plotters
  - Interactive HTML plots via plotly
  - Design matrix heatmaps
  - Jupyter notebook support via evcxr

### Performance
- Immutable HRF identities allow safe kernel caching; arbitrary custom HRFs bypass it.
- Pure Rust linear algebra; no mandatory BLAS or Fortran toolchain.
- Efficient algorithms (O(E+N) neural input generation)
- FFT-based convolution for long time series
- Zero-copy operations where possible

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
fmrihrf = "0.1.0"

# Optional features
fmrihrf = { version = "0.1.0", features = ["plotting"] }  # For visualization
```

## Usage Examples

### Basic HRF Evaluation

```rust
use fmrihrf::hrf::{spm_canonical, HrfExt};
use ndarray::Array1;

// Create SPM canonical HRF
let hrf = spm_canonical(None, None, None, None, None, Some(true));

// Evaluate at time points
let times = Array1::linspace(0.0, 30.0, 301);
let values = hrf.evaluate(&times).unwrap();

// Evaluate with 3-second block duration
let block_values = hrf.evaluate_block(&times, 3.0, 0.1, true, false).unwrap();
```

### Creating Design Matrices

```rust
use fmrihrf::regressor::{regressor, regressor_set, design_from_regressor_set};
use fmrihrf::hrf::spm_canonical;
use ndarray::Array1;

// Single condition design
let hrf = spm_canonical(None, None, None, None, None, Some(true));
let reg = regressor(
    vec![10.0, 30.0, 50.0],  // event onsets
    Some(hrf),
    Some(vec![2.0]),         // duration
    None,                    // amplitude (default 1.0)
    None,                    // span (default from HRF)
    None,                    // summate (default true)
).unwrap();

let times = Array1::linspace(0.0, 80.0, 801);
let design = design_from_regressor(&reg, &times, None).unwrap();

// Multi-condition design
let onsets = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
let conditions = vec!["A", "B", "C", "A", "B", "C"];
let rset = regressor_set(
    onsets,
    conditions.into_iter().map(|s| s.to_string()).collect(),
    Some(hrf),
    None, None, None, None,
).unwrap();

let multi_design = design_from_regressor_set(&rset, &times, None).unwrap();
```

### Using Different HRF Types

```rust
use fmrihrf::hrf::{bspline_hrf, fourier_hrf, fir_hrf, get_hrf, empirical_hrf};
use ndarray::array;

// B-spline basis (4 basis functions)
let bspline = bspline_hrf(4, Some(3), Some(20.0));

// Fourier basis (7 basis functions)
let fourier = fourier_hrf(7, Some(20.0));

// FIR basis (10 time bins)
let fir = fir_hrf(10, Some(20.0));

// Get HRF by name from registry
let canonical = get_hrf("spm_canonical").unwrap();

// Create empirical HRF from measured data
let t_measured = array![0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20.];
let y_measured = array![0., 0.2, 0.8, 1.0, 0.7, 0.3, 0., -0.1, -0.05, 0., 0.];
let empirical = empirical_hrf(&t_measured, &y_measured, Some("measured_hrf"), None).unwrap();
```

### Advanced HRF Operations

```rust
use fmrihrf::hrf::{hrf_from_coefficients, bspline_hrf, HrfExt, DerivativeMethod};
use fmrihrf::regressor::regressor;
use ndarray::Array1;

// Create HRF from weighted basis functions (for parameter estimation)
let basis_hrf = bspline_hrf(5, Some(3), Some(20.0));
let coefficients = vec![0.1, 0.5, 1.0, 0.5, 0.1];  // Estimated from data
let weighted_hrf = hrf_from_coefficients(basis_hrf, &coefficients).unwrap();

// Evaluate derivatives (useful for optimization and gradient-based methods)
let hrf = spm_canonical(None, None, None, None, None, Some(true));
let t = Array1::linspace(0.0, 20.0, 201);
let deriv = hrf.evaluate_deriv(&t, DerivativeMethod::Auto).unwrap();

// Generate neural input signal (before HRF convolution)
let reg = regressor(vec![1.0, 3.0, 5.0], Some(hrf), Some(vec![0.5]), None, None, None).unwrap();
let neural = reg.neural_input(Some(0.0), Some(10.0), Some(0.1)).unwrap();
let time = &neural["time"];
let signal = &neural["neural_input"];
```

### HRF Generators

```rust
use fmrihrf::hrf::{HrfGenerator, spm_canonical};

let base_hrf = spm_canonical(None, None, None, None, None, Some(true));

// Create lagged version
let lagged = HrfGenerator::new(base_hrf)
    .lag(2.0)
    .build();

// Create blocked and normalized version
let blocked = HrfGenerator::new(base_hrf)
    .block(3.0, 0.1)
    .normalize()
    .build();
```

### Penalty Matrices

```rust
use fmrihrf::penalty::{penalty_matrix, PenaltyOptions};
use fmrihrf::hrf::bspline_hrf;

let hrf = bspline_hrf(6, Some(3), Some(20.0));
let penalty = penalty_matrix(
    hrf.as_ref(),
    2,  // second-order penalty
    Some(PenaltyOptions::default()),
).unwrap();
```

### HRF Evaluation Caching

```rust
use fmrihrf::hrf::spm_canonical;
use fmrihrf::utils::{cached_hrf_eval, clear_hrf_cache, set_cache_size, get_cache_stats};

let hrf = spm_canonical(None, None, None, None, None, Some(true));

// First evaluation (cache miss) - ~3.9µs
let values1 = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

// Second evaluation (cache hit) - ~370ns - 10.5x faster!
let values2 = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

// Check cache performance
let stats = get_cache_stats();
println!("Hit rate: {:.1}%", stats.hit_rate());
println!("Hits: {}, Misses: {}", stats.hits, stats.misses);

// Manage cache
set_cache_size(256);  // Increase cache size
clear_hrf_cache();    // Clear all cached entries

// Caching is automatically used in regressor convolution!
```

### Visualization

```rust
#[cfg(feature = "plotting")]
use fmrihrf::plotting::{plot_hrf, plot_hrf_comparison, PlotConfig};

// Plot single HRF to PNG/SVG
let config = PlotConfig {
    title: "SPM Canonical HRF".to_string(),
    ylabel: "BOLD Response".to_string(),
    ..Default::default()
};

plot_hrf(hrf.as_ref(), &times, "output/hrf.png", Some(config)).unwrap();

// Create interactive HTML plot
let hrfs = vec![
    ("SPM".to_string(), spm_canonical(...)),
    ("Gamma".to_string(), gamma_hrf(6.0, 1.0, Some(true))),
];

plot_hrf_comparison(&hrfs, &times, "output/comparison.html", None).unwrap();
```

### Jupyter Notebook Support

Install the Rust kernel for Jupyter:

```bash
cargo install evcxr_jupyter
evcxr_jupyter --install
```

Then use fmrihrf-rs in Jupyter notebooks! See the `notebooks/` directory for examples.

## Building

```bash
# Standard build
cargo build --release

# Run tests
cargo test

# Run benchmarks (when available)
cargo bench
```

## Performance

The Rust implementation provides significant performance improvements over Python:
- Basic HRF evaluation: ~10-15x faster
- Convolution operations: ~20x faster with BLAS
- Design matrix creation: ~15x faster
- Memory usage: ~3x lower

## R Compatibility: SPM Canonical Formula

**Important Note**: This implementation prioritizes exact compatibility with the R package `fmrihrf` over mathematical "correctness."

The SPM canonical HRF uses the R package formula:
```
hrf(t) = exp(-t) * (A1 * t^P1 - C * t^P2)
```

where:
- `A1 = 0.0833` (R package value)
- `P1 = 5` (exponent for positive component)
- `P2 = 15` (exponent for negative component)
- `C = 1.274527e-13` (coefficient for negative component)

### Why This Matters

The R package uses `A1 = 0.0833`, which is **10x larger** than the mathematically "correct" value derived from gamma probability density functions (`1/Γ(6) ≈ 0.00833333`). However:

1. **The R value is standard**: SPM software and the R package both use this value
2. **Breaking changes are costly**: Using the "correct" value would break all existing analyses
3. **Users expect R compatibility**: This implementation matches R output exactly (RMSE < 1e-15)

For details on the formula validation and decision rationale, see:
- `VALIDATION_RESULTS_P0-1.md` - Numerical validation results
- `SPM_CANONICAL_FORMULA_ANALYSIS.md` - Mathematical analysis
- `tests/cross_validation_spm.rs` - Cross-validation tests

### For Advanced Users

If you need the mathematically derived formula (10x smaller amplitude), you can:
1. Use the standard function and scale the output by 0.1
2. Check for future API additions that may provide the alternative formula

## Documentation

Generate and view the API documentation:

```bash
cargo doc --open
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is dual-licensed under either:

- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Acknowledgments

This is a Rust port of the R package [fmrihrf](https://github.com/bbuchsbaum/fmrihrf) by Brad Buchsbaum. The R package is the authoritative reference implementation.