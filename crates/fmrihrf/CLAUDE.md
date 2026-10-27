# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust implementation of fMRI Hemodynamic Response Function (HRF) and Regressor Tools, ported from the R package `fmrihrf` (located at `~/code/fmrihrf`). The library provides high-performance HRF modeling and design matrix generation for fMRI analysis with BLAS-accelerated linear algebra and efficient algorithms.

**Important**: The original source is the R package. While a Python port (`pyfmrihrf`) exists, it has drifted from the R source. The Rust implementation should maintain parity with the **R package**, not the Python version. The R package is the authoritative reference implementation.

## Development Commands

### Build and Test
```bash
# Standard build
cargo build

# Release build with optimizations
cargo build --release

# Run all tests
cargo test

# Run specific test
cargo test <test_name>

# Run tests with plotting feature
cargo test --features plotting

# Run benchmarks
cargo bench

# Generate and view documentation
cargo doc --open
```

### Running Examples
```bash
# Run example with plotting
cargo run --example html_plotting_demo --features plotting
cargo run --example comprehensive_demo
```

### Development Notes
- The project uses BLAS via ndarray with `openblas-static` feature
- Local dependency on `splines` crate at `../splines` (see Cargo.toml:26)
- The crate has optional `plotting`, `serde`, and `parallel` features
- Release builds use LTO and codegen-units=1 for maximum optimization

## Architecture Overview

### Core Design Pattern: Trait-Based HRF System

The architecture centers on the `Hrf` trait (src/hrf/core.rs) which defines the contract for all hemodynamic response functions:

```rust
pub trait Hrf: Debug + Send + Sync {
    fn evaluate(&self, t: &TimeArray) -> Result<HrfValues>;
    fn name(&self) -> &str;
    fn nbasis(&self) -> usize;
    fn span(&self) -> f64;
    fn params(&self) -> Vec<(&str, f64)>;
}
```

**Extension Trait**: `HrfExt` provides additional functionality like `evaluate_block()` for sustained stimuli with block duration support.

### Module Structure

**hrf/** - Core HRF functionality
- `core.rs` - Hrf and HrfExt traits defining the interface
- `functions.rs` - Concrete HRF implementations (SPM, gamma, gaussian, b-spline, fourier, FIR, wavelets)
- `generators.rs` - HrfGenerator builder pattern for composing HRFs with transformations (lag, block, normalize)
- `registry.rs` - Global registry using lazy_static for HRF lookup by name
- `empirical.rs` - Create HRFs from measured data with interpolation
- `derivatives.rs` - Analytic derivative implementations
- `library.rs` - Predefined HRF configurations (currently minimal)

**regressor/** - Convolution and design matrix tools
- `core.rs` - Regressor struct associating event onsets with HRFs
- `convolution.rs` - Direct and FFT-based convolution operations
- `design.rs` - Design matrix creation for single/multi-condition experiments
- `neural_input.rs` - Neural input signal generation from event timing (O(E+N) algorithm)

**Other modules:**
- `sampling.rs` - SamplingFrame for fMRI timing specification (TR, blocklens, precision)
- `penalty.rs` - Penalty matrix generation for regularization
- `reconstruction.rs` - Reconstruction matrix for basis function evaluation
- `error.rs` - Unified error handling with thiserror
- `types.rs` - Type aliases (HrfValues, Matrix, TimeArray, Vector)
- `plotting.rs` - Visualization via plotters (static) and plotly (interactive HTML)

### Key Patterns

**Registry Pattern**: Global HRF registry using `lazy_static!` HashMap allows lookup by name (e.g., `get_hrf("spm_canonical")`).

**Builder Pattern**: `HrfGenerator` allows chaining transformations:
```rust
HrfGenerator::new(base_hrf)
    .lag(2.0)
    .block(3.0, 0.1)
    .normalize()
    .build()
```

**Arc<Box<dyn Hrf>>**: Regressors store HRFs as `Arc<Box<dyn Hrf>>` for cheap cloning and shared ownership across thread boundaries.

**Result-based Error Handling**: All fallible operations return `Result<T, HrfError>` with specific error variants.

### Type System

- `TimeArray` = `Array1<f64>` - 1D time points
- `HrfValues` = `Array2<f64>` - 2D matrix (time_points × n_basis)
- `Vector` = `Array1<f64>` - 1D vector
- `Matrix` = `Array2<f64>` - 2D matrix

HRF evaluation always returns 2D arrays where columns represent basis functions (even single-basis HRFs return (n, 1) shaped arrays).

### Critical Implementation Details

**Convolution**: Supports both direct discrete convolution and FFT-based convolution for long time series (controlled by `use_fft` parameter and automatic threshold).

**Neural Input Generation**: Uses efficient O(E+N) algorithm with binary search for onset mapping to time grid indices.

**Block Evaluation**: `evaluate_block()` creates high-resolution grid, convolves HRF with boxcar function, then downsamples to requested time points. Supports both summation and max modes.

**Sparse Matrices**: Uses `sprs` crate for penalty matrices and sparse operations.

**B-splines**: Delegates to external `splines` crate for Cox-de Boor recursion.

## Feature Parity Status

**Current Status**: 73% complete, in active development to match R package `fmrihrf`

**R Package Location**: `/Users/bbuchsbaum/code/fmrihrf`

**Complete Features:**
- SPM canonical HRF (with/without derivatives)
- Gamma, Gaussian, B-spline, Fourier, FIR, Mexican hat, sine basis functions
- Regressor system with convolution
- Design matrix generation
- Penalty and reconstruction matrices
- Empirical HRF from data
- Core registry system

**Critical Gaps (P0):**
- SPM canonical formula validation (see SPM_CANONICAL_FORMULA_ANALYSIS.md)
- S3 generic system → Rust traits mapping
- `hrf_from_coefficients()` function
- HRF evaluation caching
- `global_onsets()` method
- SPMG2 column count fix

**Key Documentation:**
- `R_GAP_ANALYSIS_2025.md` - Comprehensive gap analysis (73% complete, 27 critical/high priority gaps)
- `IMPLEMENTATION_PLAN.md` - Detailed roadmap with 4 phases (12-17 weeks to 95% parity)
- `SPM_CANONICAL_FORMULA_ANALYSIS.md` - Critical SPM formula investigation
- `QUICK_START_IMPLEMENTATION.md` - Immediate action guide
- `GAP_ANALYSIS_2025.md` - Historical Python comparison (superseded by R analysis)

**R Package Structure** (source of truth):
- `R/hrf.R` - Main HRF S3 class and methods (~1100 lines)
- `R/hrf-functions.R` - HRF function implementations
- `R/hrf_decorators.R` - Lag, block, normalize decorators
- `R/all_generic.R` - Generic function definitions (S3 methods)
- `R/evaluate-helpers.R` - Evaluation utilities
- `R/reg-constructor.R` - Regressor construction
- `R/reg-methods.R` - Regressor methods
- `R/regressor-design.R` - Design matrix creation
- `src/neural_input.cpp` - C++ neural input generation (via Rcpp)
- `tests/testthat/` - Comprehensive test suite

**Next Steps:**
1. Run validation: `cargo run --example validate_spm_canonical`
2. Review SPM_CANONICAL_FORMULA_ANALYSIS.md
3. Follow QUICK_START_IMPLEMENTATION.md
4. Refer to IMPLEMENTATION_PLAN.md for full roadmap

## Testing Strategy

Tests are organized in `tests/`:
- `integration_test.rs` - End-to-end workflow tests
- `cross_validation.rs` - Cross-validation against R package outputs
- `edge_case_tests.rs` - Boundary conditions and error cases
- `test_serialization.rs` - Serde feature tests
- `workflow_tests.rs` - Common usage patterns

**Testing approach:**
- Unit tests in each module file
- Property-based tests with proptest for invariants
- Cross-validation with R package outputs (JSON/CSV-based)
- Benchmark comparisons against R (using Rcpp benchmarks as baseline)
- Reference R package location: `/Users/bbuchsbaum/code/fmrihrf`

## Common Pitfalls

1. **HRF evaluation shape**: Always returns 2D (time × basis), even for single-basis HRFs
2. **Registry initialization**: Registry is lazy-initialized on first access; must call functions to populate
3. **Arc<Box<dyn Hrf>> cloning**: Clone the Arc, not the underlying HRF
4. **Plotting fonts**: PNG/SVG may have font issues; prefer HTML output (see PLOTTING.md)
5. **Local splines dependency**: The project depends on `splines = { path = "../splines" }`, not a crates.io version
6. **FFT threshold**: Automatic FFT selection may need tuning for specific use cases

## R Package Compatibility Notes

This is a port of the R package `fmrihrf`. When implementing new features:
- **Source of Truth**: R package at `/Users/bbuchsbaum/code/fmrihrf`
- Follow R package API patterns where sensible (adapted for Rust idioms)
- R uses S3 classes; Rust uses traits and structs
- R uses `memoise` for caching; Rust uses custom caching with Mutex
- Use `Result<T>` instead of R's error/warning system
- Cross-validate outputs with R package using JSON/CSV serialization
- R package has C++ code via Rcpp - compare performance against those implementations
- See RUST_CONVERSION_STRATEGY.md for type mappings from R to Rust

**Key Differences from R**:
- R is dynamically typed; Rust is statically typed
- R uses S3 generic dispatch; Rust uses trait-based dispatch
- R arrays are column-major; ndarray can be configured for either
- R uses `Matrix` package for sparse matrices; Rust uses `sprs` crate

## Performance Characteristics

**Target performance vs R:**
- Basic HRF evaluation: ~5-10x faster than pure R, comparable to Rcpp
- Convolution with BLAS: ~10-20x faster than pure R
- Design matrix creation: ~10-15x faster than pure R
- Memory usage: Comparable to R with good memory management

**Comparison Points**:
- Pure R implementations (baseline)
- Rcpp implementations in R package (high-performance baseline)
- Target: Match or exceed Rcpp performance

Use `cargo bench` to validate optimizations.
