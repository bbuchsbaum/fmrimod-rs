# Splines

A fast, safe Rust implementation of the R splines package functionality.

This crate provides functions and data structures for creating and working with:

- **B-splines** (`BSpline`) - Basis splines for regression and smoothing
- **Natural cubic splines** (`NaturalSpline`) - Splines with linear extrapolation
- **Interpolation splines** (`InterpolationSpline`) - For exact data fitting
- **Periodic splines** (`PeriodicSpline`) - For cyclic data
- **Polynomial spline representations** (`PolySpline`) - Alternative representations

## Features

- **Type-safe**: Leverages Rust's type system for correctness
- **Performance**: SIMD-optimized operations via ndarray
- **Memory-safe**: Zero unsafe code
- **R-compatible**: Numerical equivalence with R's splines package
- **Comprehensive**: Full API coverage of R splines functionality

## Quick Start

```rust
use splines::{BSpline, Spline};

// Create sample data
let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];

// Create a cubic B-spline with 5 degrees of freedom
let spline = BSpline::new(&x, Some(5), None, 3, false, None)?;

// Generate design matrix for new points
let new_x = vec![1.5, 2.5, 3.5];
let design_matrix = spline.design_matrix(&new_x)?;

// Or use the builder pattern
let spline = BSpline::builder()
    .degree(3)
    .df(6)
    .intercept(true)
    .build(&x)?;
    
# Ok::<(), Box<dyn std::error::Error>>(())
```

## License

Licensed under either of:

- Apache License, Version 2.0
- MIT License

at your option.