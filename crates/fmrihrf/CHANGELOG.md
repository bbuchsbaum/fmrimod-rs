> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Core R parity audit — 2026-09-05

- Correct fractional impulse placement, integrated blocks shorter than precision,
  duration averages, overlapping events, pre-window tails, and empty events.
- Add typed direct/FFT/automatic/reference methods and borrowed run-aware event
  tables; preserve custom HRFs and every SPMG basis column.
- Use raw R amplitudes and fixed-reference normalization; correct Mexican hat
  scaling, causal support, tent splines, half-cosine segments, and registry aliases.
- Key cached evaluations by immutable identity and exact floating-point inputs.
  Unknown custom HRFs bypass caching. Correct run offsets and validate timing.
- Replace mandatory OpenBLAS/ndarray-linalg with the existing nalgebra dependency.
  This removes the public ndarray-linalg error conversion along with that backend.
- Repair ordinary tests and doctests, regenerate source-backed R fixtures with
  explicit precision, tighten tent parity to 1e-12, and add a release benchmark.
- Validate plotting inputs before rendering; optional bitmap plots use system
  font support. Reject unsupported file extensions.
- `SamplingFrame::new` now takes `Option<Vec<f64>>` for start time so `None`
  needs no type annotation; arrays can be passed with `.to_vec()`.
- Explicit spans are respected. New run-aware tables use zero-based run indices
  and basis-major columns; legacy R-like APIs retain their signatures.
- See PORT_PARITY_AUDIT.md for measured scope, provenance, and unverified features.


### Added
- **HRF Evaluation Caching** (P0-4: Performance Optimization)
  - LRU cache for HRF evaluations with configurable size (default 128)
  - Cache statistics tracking (hits, misses, evictions, hit rate)
  - Thread-safe caching using Arc<Mutex<>>
  - Automatic integration with regressor convolution
  - **10.5x speedup** for warm cache lookups (370ns vs 3.9µs)
  - **8.5x speedup** for repeated evaluations (100 calls: 36µs vs 311µs)
  - Public API: `cached_hrf_eval()`, `clear_hrf_cache()`, `set_cache_size()`, `get_cache_stats()`, `reset_cache_stats()`
  - Comprehensive benchmarks comparing cached vs uncached performance
- **Empirical HRF functionality** - Create HRFs from measured data points
  - Linear interpolation for sparse data (< 4 points)
  - Cubic spline interpolation for dense data (≥ 4 points)
  - Automatic extrapolation to zero outside data range
  - Full compatibility with HRF trait system
  - `empirical_hrf()` and `gen_empirical_hrf()` functions
  - Comprehensive tests and examples
  - Interactive plotting support
- **Cross-validation test suite** for SPM canonical HRF against R package
  - Validates numerical equivalence (RMSE < 1e-15)
  - Tests normalization, peak location, and backward compatibility
  - See `tests/cross_validation_spm.rs`
- **Enhanced HRF Trait System** (P0-2: S3 Generic System → Rust Traits)
  - `hrf_from_coefficients()` function for creating weighted HRF combinations
    - Create single-basis HRFs from weighted sums of basis functions
    - Critical for parameter estimation workflows
    - Works with any HRF type (basis sets, canonical, custom)
    - Returns `WeightedHrf` struct that implements `Hrf` trait
  - `evaluate_deriv()` method in `HrfExt` trait for polymorphic derivatives
    - Compute derivatives for any HRF type via trait method
    - Uses numerical differentiation (central differences)
    - Works with both concrete types and trait objects
    - Supports multi-basis HRFs (returns derivative for each basis)
    - Comprehensive test coverage including trait object usage
  - `neural_input()` method on `Regressor` for neural signal generation
    - Generate boxcar time series representing neural activity before HRF convolution
    - Efficient O(E+N) algorithm using difference arrays
    - Useful for visualization and understanding event timing
    - Method syntax matches R package S3 generic pattern

### Changed
- **SPM Canonical HRF Formula** (P0-1: Critical Correctness Fix)
  - Changed from gamma PDF formulation to R package's simplified exponential-polynomial formula
  - Now uses `exp(-t) * (0.0833 * t^5 - 1.274527e-13 * t^15)` to match R exactly
  - **Note**: This uses A1=0.0833 (R package value), which is 10x larger than the mathematically
    derived value from gamma PDFs. This is intentional for R compatibility.
  - Maintains backward compatibility: old 6-parameter signature still works (parameters are ignored)
  - Achieves exact numerical equivalence with R package (RMSE < 1e-15)
  - See `VALIDATION_RESULTS_P0-1.md` for detailed analysis and rationale
- Updated Cargo.toml to include `line_series` feature for plotters
- Enhanced README with empirical HRF documentation and examples
- Enhanced README with SPM canonical formula documentation explaining R compatibility choice

### Fixed
- Fixed plotting module compilation by adding missing plotters features

## [0.1.0] - Initial Release

### Core Features
- **HRF Models**: SPM canonical, gamma, gaussian, and custom basis functions
- **Basis Sets**: B-spline, Fourier, FIR basis functions
- **HRF Generators**: Dynamic HRF creation with decorators (lag, block, normalize)
- **HRF Registry**: Global registry for HRF lookup by name
- **Sampling Framework**: Flexible fMRI timing specification
- **Regressor System**: Event-related design matrix creation
- **Penalty Matrices**: Regularization for HRF estimation
- **Reconstruction Matrices**: Basis function evaluation
- **Visualization**: Optional plotting feature with HTML, SVG, and PNG output

### Performance
- BLAS-accelerated linear algebra via ndarray
- Efficient algorithms (O(E+N) neural input generation)
- FFT-based convolution for long time series
- Zero-copy operations where possible