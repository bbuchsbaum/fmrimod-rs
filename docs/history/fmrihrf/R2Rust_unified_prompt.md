# R to Rust Package Conversion - Claude Code Agent Guide

## Core Objective
Convert R packages to Rust with maximum feature parity, performance, and idiomatic Rust design. Produce Rust crates that match R outputs exactly while leveraging Rust's zero-cost abstractions, memory safety, and performance advantages.

## Phase 0: Discovery & Assessment

### Initial Codebase Analysis
1. **Repository Scan**
   - Run `R CMD build && R CMD check` to identify existing issues
   - Use `getNamespaceExports(pkg)` to list all exported objects
   - Create dependency tree with `renv::dependencies()` or `pkgnet::CreatePackageReport()`
   - Tag dependencies as R-core vs CRAN vs GitHub

2. **Complexity Assessment**
   - For each exported object, assign translation risk:
     - **High**: NSE, active bindings, R6, Shiny UI (requires macro design)
     - **Medium**: S3/S4 dispatch, complex vectorization, C/C++ integration
     - **Low**: Pure functions without side effects

3. **Performance Profiling**
   ```bash
   # Profile R code for optimization targets
   rg -I -n "(for|while|apply|lapply|sapply|mapply)" R/ | tee loop_patterns.txt
   rg -I -n "(\\.Call|\\.C|\\.Fortran|\\.External)" R/ | tee ffi_calls.txt
   ```

4. **Side Effects Inventory**
   ```bash
   rg -I -n "(write|save|assign\\s*\\(|options\\(|par\\(|plot\\(|png\\(|pdf\\()" R/ | tee side_effect_hits.txt
   ```

5. **Baseline Testing**
   - Extract existing `testthat` tests
   - Freeze examples from help files as golden tests
   - Set reproducibility: `set.seed(1)` at test start
   - Record performance baselines with `bench::mark`

## Phase 1: Architecture Mapping

### R to Rust Concept Translation

| R Concept | Rust Equivalent | Implementation |
|-----------|------------------|----------------|
| S4 class w/ validity | `struct` + `impl` with validation | Use builder pattern or TryFrom |
| Simple S4 slots | `struct` fields | Direct field mapping with types |
| Generic + methods | Trait objects or generics | `trait` definitions with implementations |
| Vectorized ops | `ndarray` or `polars` | Use SIMD-optimized operations |
| Data frames | `polars::DataFrame` | Match columns/dtypes with strong typing |
| NSE verbs | Procedural macros | Custom derive macros for DSL |
| Formula interface | Custom parser + AST | Use `nom` or `pest` for parsing |
| Pipe %>% | Method chaining | Builder pattern or fluent interfaces |

### Naming Conventions
- Functions: `snake_case` (idiomatic Rust)
- Types/Structs: `PascalCase`
- Constants: `SCREAMING_SNAKE_CASE`
- R dots to underscores: `as.mask` → `as_mask`
- Private functions: `mod internal` or `pub(crate)`
- Document all 1-based to 0-based index shifts

### Error Mapping
- R `stop()` → Rust `Result<T, E>` with custom error types
- R `warning()` → Rust `log::warn!()` or custom warning system
- R `try()` → Rust `Result` pattern matching

### Type System Advantages
```rust
// Leverage Rust's type system for R's dynamic types
#[derive(Debug, Clone)]
pub enum RValue {
    Logical(Vec<bool>),
    Integer(Vec<i32>),
    Numeric(Vec<f64>),
    Character(Vec<String>),
    Complex(Vec<Complex64>),
    Raw(Vec<u8>),
    List(Vec<RValue>),
    Null,
}

// Use generics for type-safe operations
pub trait RVector<T> {
    fn length(&self) -> usize;
    fn get(&self, index: usize) -> Option<&T>;
    fn set(&mut self, index: usize, value: T) -> Result<(), RError>;
}
```

## Phase 2: Implementation Strategy

### Project Structure
```
package_name/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── core/
│   │   ├── mod.rs
│   │   ├── vectors.rs
│   │   └── dataframes.rs
│   ├── utils/
│   │   ├── mod.rs
│   │   └── helpers.rs
│   ├── r_api/
│   │   ├── mod.rs
│   │   └── compatibility.rs
│   └── errors.rs
├── tests/
│   ├── integration/
│   │   ├── mod.rs
│   │   └── r_comparison.rs
│   └── fixtures/
│       ├── mod.rs
│       └── test_data.rs
├── examples/
│   ├── basic_usage.rs
│   └── advanced_features.rs
├── benches/
│   └── performance.rs
└── docs/
    └── src/
        └── lib.md
```

### Cargo.toml Configuration
```toml
[package]
name = "r_package_name"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <email@example.com>"]
description = "Rust implementation of R package functionality"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"
documentation = "https://docs.rs/r_package_name"
categories = ["science", "mathematics"]
keywords = ["statistics", "data-science", "r-compatibility"]

[dependencies]
# Core data structures
ndarray = "0.15"
polars = { version = "0.35", features = ["lazy", "csv", "json"] }
num-complex = "0.4"
num-traits = "0.2"

# Serialization
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Parsing (for formula interface)
nom = "7.1"
pest = "2.7"
pest_derive = "2.7"

# Performance
rayon = "1.8"
crossbeam = "0.8"

# Logging
log = "0.4"
env_logger = "0.10"

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
tempfile = "3.8"
approx = "0.5"

[features]
default = ["std"]
std = []
parallel = ["rayon", "polars/lazy"]
python-interop = ["pyo3"]

[[bench]]
name = "performance"
harness = false

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
```

### Migration Loop Pattern
1. **Pick leaf module** (fewest dependencies)
2. **Manual translation** (avoid auto-transpilers)
3. **Implement core traits** first
4. **Write Rust tests** using R fixtures
5. **Run cross-language validation**
6. **Optimize after correctness confirmed**

### Implementation Checklist
- [ ] Study R source code thoroughly
- [ ] Define custom error types with `thiserror`
- [ ] Implement core data structures
- [ ] Create trait bounds for generic operations
- [ ] Test against R output with `approx::assert_relative_eq!`
- [ ] Add comprehensive documentation with examples
- [ ] Implement `Debug`, `Clone`, `PartialEq` where appropriate
- [ ] Use `#[derive(serde::Serialize, serde::Deserialize)]` for data interchange

## Phase 3: Verification Strategy

### Testing Layers

| Layer | Technique | Success Criterion |
|-------|-----------|-------------------|
| Unit | Built-in `#[test]` | All pass on stable Rust |
| Property | `proptest` | Invariants hold across input space |
| Integration | Cross-language fixtures | `approx::assert_relative_eq!` passes |
| Performance | `criterion` benchmarks | ≤20% slower than R or justified |
| Memory | `valgrind` + `miri` | No undefined behavior |
| API | `cargo doc` + examples | Full documentation coverage |

### Cross-Testing Infrastructure
```rust
// tests/integration/r_comparison.rs
use approx::assert_relative_eq;
use serde_json::Value;

#[test]
fn test_function_equivalence() {
    // Load R fixture data
    let r_input: Value = load_r_fixture("function_input.json");
    let r_output: Vec<f64> = load_r_fixture("function_output.json");
    
    // Convert to Rust types
    let rust_input = RValue::from_json(r_input).unwrap();
    
    // Run Rust implementation
    let rust_output = rust_function(&rust_input).unwrap();
    
    // Compare results
    assert_relative_eq!(
        rust_output.as_slice(),
        r_output.as_slice(),
        epsilon = 1e-12
    );
}

// Helper for loading R-generated test fixtures
fn load_r_fixture<T: serde::de::DeserializeOwned>(name: &str) -> T {
    let path = format!("tests/fixtures/{}", name);
    let content = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&content).unwrap()
}
```

### Property-Based Testing
```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_vectorized_operation_properties(
        a in prop::collection::vec(any::<f64>(), 1..100),
        b in prop::collection::vec(any::<f64>(), 1..100)
    ) {
        let result = vector_add(&a, &b);
        
        // Test properties that should hold
        prop_assert_eq!(result.len(), a.len().max(b.len()));
        
        // Test commutativity
        let result_rev = vector_add(&b, &a);
        prop_assert!(result.iter().zip(result_rev.iter())
            .all(|(x, y)| (x - y).abs() < 1e-10));
    }
}
```

## Phase 4: Documentation & Packaging

### Documentation Strategy
1. **rustdoc**: Comprehensive API documentation with examples
2. **mdBook**: User guide and migration documentation
3. **Benchmarks**: Performance comparisons with R
4. **Examples**: Executable code demonstrating features

### rustdoc Configuration
```rust
#![doc = include_str!("../README.md")]
#![doc(html_logo_url = "https://example.com/logo.png")]
#![doc(html_favicon_url = "https://example.com/favicon.ico")]
#![doc(html_root_url = "https://docs.rs/package_name/")]

/// Central data structure representing R vectors
/// 
/// This struct provides a type-safe interface to R's vectorized operations
/// while maintaining zero-cost abstractions where possible.
/// 
/// # Examples
/// 
/// ```
/// use package_name::RVector;
/// 
/// let vec = RVector::new(vec![1.0, 2.0, 3.0]);
/// let doubled = vec.map(|x| x * 2.0);
/// assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0]);
/// ```
/// 
/// # R Compatibility
/// 
/// This implementation matches R's behavior for:
/// - Vector recycling rules
/// - NA value handling
/// - Attribute preservation
/// 
/// # Performance
/// 
/// Uses SIMD operations where possible via `ndarray` backend.
/// Typical performance is 2-5x faster than equivalent R code.
pub struct RVector<T> {
    data: Vec<T>,
    attributes: HashMap<String, RValue>,
}
```

### Migration Guide Structure
```rust
// docs/src/migration.md
# Migrating from R

## Function Mapping

| R Function | Rust Equivalent | Notes |
|------------|-----------------|-------|
| `c(1, 2, 3)` | `RVector::new(vec![1, 2, 3])` | Type-safe construction |
| `length(x)` | `x.len()` | Zero-cost |
| `sum(x)` | `x.sum()` | SIMD-optimized |
| `x[i]` | `x.get(i-1)` | 0-based indexing |
| `x[x > 0]` | `x.filter(|&v| v > 0.0)` | Lazy evaluation |

## Type System Benefits

R's dynamic typing is replaced with Rust's compile-time guarantees:

```rust
// This won't compile - caught at compile time
let x = RVector::new(vec![1, 2, 3]);
let y = RVector::new(vec!["a", "b", "c"]);
let result = x + y; // Error: mismatched types
```

## Performance Comparisons

See `benches/` directory for detailed benchmarks. Summary:

- Vector operations: 2-5x faster
- Data frame operations: 1.5-3x faster  
- Memory usage: 50-80% less
- Startup time: 100x faster (no interpreter)
```

### Packaging for crates.io
```toml
# Complete Cargo.toml for publication
[package]
name = "r_package_name"
version = "0.1.0"
edition = "2021"
rust-version = "1.70"
authors = ["Your Name <email@example.com>"]
description = "Fast, safe Rust implementation of R package functionality"
license = "MIT OR Apache-2.0"
repository = "https://github.com/user/repo"
documentation = "https://docs.rs/r_package_name"
homepage = "https://github.com/user/repo"
readme = "README.md"
categories = ["science", "mathematics", "algorithms"]
keywords = ["statistics", "data-science", "r-compatibility", "numerical"]
exclude = ["tests/fixtures/*", "benches/data/*"]

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]
```

## Phase 5: Performance Optimization

### Optimization Strategy
1. **Profile first** with `cargo flamegraph` and `perf`
2. **Use SIMD** via `ndarray` or manual vectorization
3. **Parallel processing** with `rayon` for large datasets
4. **Memory layout optimization** for cache efficiency
5. **Compile-time optimization** with const generics

### Performance Patterns
```rust
// SIMD-optimized vector operations
use ndarray::prelude::*;

pub fn vector_add_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
    let a_array = Array1::from_slice(a);
    let b_array = Array1::from_slice(b);
    
    // Automatic SIMD via ndarray
    let result = &a_array + &b_array;
    result.to_vec()
}

// Parallel processing for large datasets
use rayon::prelude::*;

pub fn parallel_apply<F>(data: &[f64], func: F) -> Vec<f64>
where
    F: Fn(f64) -> f64 + Sync + Send,
{
    data.par_iter()
        .map(|&x| func(x))
        .collect()
}

// Zero-cost abstractions with iterators
pub fn filter_map_collect<T, U, F, P>(
    data: &[T],
    predicate: P,
    mapper: F,
) -> Vec<U>
where
    F: Fn(&T) -> U,
    P: Fn(&T) -> bool,
    T: Clone,
{
    data.iter()
        .filter(|x| predicate(x))
        .map(mapper)
        .collect()
}
```

### Memory Optimization
```rust
// Use SmallVec for small vectors to avoid heap allocation
use smallvec::SmallVec;

// Pre-allocate with known capacity
pub fn preallocated_computation(n: usize) -> Vec<f64> {
    let mut result = Vec::with_capacity(n);
    // ... computation
    result
}

// Use Cow for flexible ownership
use std::borrow::Cow;

pub fn flexible_function(data: Cow<[f64]>) -> Vec<f64> {
    // Works with both owned and borrowed data
    data.iter().map(|&x| x * 2.0).collect()
}
```

## Critical Implementation Rules

### Always
- Study R source code before implementing
- Use `Result<T, E>` for all fallible operations
- Implement appropriate traits (`Debug`, `Clone`, `PartialEq`)
- Handle NA/NaN values explicitly with `Option<T>`
- Test against R outputs with appropriate tolerances
- Use `#[doc]` attributes with examples
- Leverage Rust's type system for correctness

### Never
- Use `unwrap()` or `expect()` in library code
- Ignore compiler warnings
- Skip property-based testing for core functions
- Create global mutable state
- Use `unsafe` without extensive documentation
- Panic in library functions (use `Result` instead)

## Workflow Commands

### Initial Analysis
```bash
# Inventory R package structure
find . -name "*.R" -exec wc -l {} + | sort -n
rg "^#'" R/ | head -20  # Find roxygen documentation
rg "UseMethod|setClass|setMethod" R/  # Find S3/S4 usage

# Check for performance bottlenecks
rg "for\\s*\\(|while\\s*\\(" R/  # Find loops
rg "\\.Call|\\.C|\\.Fortran" R/  # Find C/Fortran calls
```

### Development Workflow
```bash
# Fast development cycle
cargo check          # Quick syntax check
cargo test           # Run tests
cargo clippy         # Linting
cargo fmt            # Format code

# Performance testing
cargo bench          # Run benchmarks
cargo flamegraph     # Profile hot paths

# Documentation
cargo doc --open     # Build and view docs
```

### Cross-Language Testing
```bash
# Generate R test fixtures
Rscript scripts/generate_fixtures.R

# Run equivalence tests
cargo test --test r_comparison

# Performance comparison
cargo bench --bench vs_r
```

## Success Metrics
- ✅ 100% R function coverage
- ✅ >95% test coverage (`cargo tarpaulin`)
- ✅ R-Rust output equivalence (within numerical tolerance)
- ✅ Performance within 20% of R (or demonstrably faster)
- ✅ Zero unsafe code (or fully documented unsafe blocks)
- ✅ Full rustdoc documentation with examples
- ✅ Passes `cargo clippy` with no warnings
- ✅ Compiles on stable Rust

## Agent-Specific Instructions

1. **Start with exploration**: Use `ls`, `rg`, `find` to understand R structure
2. **Create type-safe design**: Define core types and traits first
3. **Implement incrementally**: One module at a time with full testing
4. **Use Rust idioms**: Leverage iterators, `Result`, and traits
5. **Maintain R compatibility**: Match function signatures and behavior
6. **Document thoroughly**: Use rustdoc with examples
7. **Profile and optimize**: Use `criterion` for benchmarking

## Common Patterns Reference

### R S3 Dispatch → Rust Traits
```rust
// R: UseMethod("print")
// Rust: Trait with implementations
pub trait Display {
    fn display(&self) -> String;
}

impl Display for MyStruct {
    fn display(&self) -> String {
        format!("MyStruct {{ value: {} }}", self.value)
    }
}
```

### R Vectorization → Rust Iterators
```rust
// R: x + y (with recycling)
// Rust: Explicit iterator patterns
pub fn vector_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() == b.len() {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    } else {
        // Handle recycling explicitly
        let (long, short) = if a.len() > b.len() { (a, b) } else { (b, a) };
        long.iter()
            .enumerate()
            .map(|(i, &x)| x + short[i % short.len()])
            .collect()
    }
}
```

### R Factors → Rust Enums
```rust
// R: factor(x, levels = c("low", "medium", "high"))
// Rust: Type-safe enum
#[derive(Debug, Clone, PartialEq)]
pub enum Level {
    Low,
    Medium,
    High,
}

impl From<&str> for Level {
    fn from(s: &str) -> Self {
        match s {
            "low" => Level::Low,
            "medium" => Level::Medium,
            "high" => Level::High,
            _ => panic!("Invalid level: {}", s),
        }
    }
}
```

### R Data Frames → Rust Structs
```rust
// R: data.frame(x = c(1, 2, 3), y = c("a", "b", "c"))
// Rust: Strongly typed struct
#[derive(Debug, Clone)]
pub struct DataFrame {
    pub x: Vec<f64>,
    pub y: Vec<String>,
}

impl DataFrame {
    pub fn new(x: Vec<f64>, y: Vec<String>) -> Result<Self, DataError> {
        if x.len() != y.len() {
            return Err(DataError::LengthMismatch);
        }
        Ok(DataFrame { x, y })
    }
    
    pub fn nrows(&self) -> usize {
        self.x.len()
    }
}
```

This guide provides a comprehensive, Rust-specific approach to R package conversion, emphasizing type safety, performance, and idiomatic Rust patterns while maintaining R compatibility.