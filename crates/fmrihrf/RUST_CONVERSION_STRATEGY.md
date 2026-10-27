# pyfmrihrf to Rust Conversion Strategy

## Phase 0: Discovery & Assessment Summary

### 1. Codebase Structure
The pyfmrihrf library is a well-structured Python implementation of fMRI HRF (Hemodynamic Response Function) and regressor tools.

**Key Modules:**
- `hrf/` - Core HRF functionality
  - `core.py` - Base HRF class (ABC with dataclass)
  - `functions.py` - HRF implementations (SPM, gamma, gaussian, etc.)
  - `generators.py` - HRF generators
  - `decorators.py` - HRF decorators (lag, block, normalize)
  - `library.py` - Pre-defined HRF objects
  - `registry.py` - HRF registry system
  - `penalty.py` - Penalty matrix calculations
  - `reconstruction.py` - Reconstruction matrix
- `regressor/` - Convolution and design matrix tools
  - `core.py` - Regressor implementation
  - `convolution.py` - Convolution operations
  - `design.py` - Design matrix creation
- `sampling.py` - SamplingFrame for fMRI timing
- `utils/` - Utilities (caching, misc)

### 2. Dependency Analysis
**Core Dependencies:**
- numpy (heavy usage for array operations)
- scipy (stats, signal processing, sparse matrices)
- pandas (data handling)
- scikit-learn (some utilities)
- numba (JIT compilation for performance)
- matplotlib (visualization)

**Complexity Assessment:**
- **Low Risk**: Pure functions (most HRF functions), data structures
- **Medium Risk**: ABC inheritance pattern, dataclasses, decorators
- **High Risk**: Dynamic registry system, numba JIT compilation

### 3. Dynamic Features
- Minimal dynamic features detected
- Uses `@dataclass` decorator extensively
- Uses `@abstractmethod` for base class
- Registry pattern with runtime registration
- No metaclasses or complex reflection

### 4. Testing Infrastructure
- Comprehensive test suite with pytest
- Cross-validation tests with R implementation
- Performance benchmarks
- ~170 test files across main tests and cross-testing

## Phase 1: Architecture Mapping

### Type Translations

| Python Component | Rust Equivalent | Notes |
|-----------------|-----------------|-------|
| `@dataclass` HRF | `struct` with `#[derive(Debug, Clone)]` | Add builder pattern |
| ABC inheritance | Trait-based design | `HRF` trait with implementations |
| numpy arrays | `ndarray::Array` or custom types | Consider nalgebra for linear algebra |
| scipy.sparse | `sprs` or `nalgebra::sparse` | For sparse matrices |
| Registry pattern | `lazy_static!` HashMap | Static registration |
| Decorators | Function wrappers/combinators | Use closure patterns |

### Core Type Mappings

```rust
// Python: HRF base class
// Rust: HRF trait
pub trait HRF: Debug + Clone {
    fn evaluate(&self, t: &Array1<f64>) -> Array2<f64>;
    fn name(&self) -> &str;
    fn nbasis(&self) -> usize;
    fn span(&self) -> f64;
}

// Python: SamplingFrame dataclass
// Rust: struct with validation
#[derive(Debug, Clone)]
pub struct SamplingFrame {
    blocklens: Vec<usize>,
    tr: Vec<f64>,
    start_time: Vec<f64>,
    precision: f64,
}
```

## Phase 2: Implementation Strategy

### Project Structure
```
fmrihrf-rs/
├── Cargo.toml
├── README.md
├── src/
│   ├── lib.rs
│   ├── error.rs
│   ├── types.rs
│   ├── hrf/
│   │   ├── mod.rs
│   │   ├── core.rs      # HRF trait and implementations
│   │   ├── functions.rs # HRF function implementations
│   │   ├── generators.rs
│   │   ├── decorators.rs
│   │   ├── library.rs
│   │   └── registry.rs
│   ├── regressor/
│   │   ├── mod.rs
│   │   ├── core.rs
│   │   └── convolution.rs
│   ├── sampling.rs
│   └── utils/
│       └── mod.rs
├── tests/
├── benches/
└── examples/
```

### Implementation Order
1. **Core Types** (Week 1)
   - Error types
   - SamplingFrame
   - Basic HRF trait

2. **HRF Functions** (Week 2-3)
   - SPM canonical
   - Gamma HRF
   - Gaussian HRF
   - Other basis functions

3. **Generators & Registry** (Week 4)
   - HRF generators
   - Static registry

4. **Regressor Module** (Week 5)
   - Convolution operations
   - Design matrix

5. **Optimization & Benchmarks** (Week 6)
   - SIMD optimizations
   - Parallel processing with rayon

### Key Considerations

1. **Performance Optimizations**
   - Use `ndarray` with BLAS backend
   - Consider `rayon` for parallel operations
   - SIMD for convolution operations
   - Zero-copy where possible

2. **API Design**
   - Keep similar API for easy migration
   - Add builder patterns for complex types
   - Use Result<T, E> for all fallible operations

3. **Testing Strategy**
   - Port all Python tests
   - Add property-based tests with proptest
   - Cross-validation with Python outputs
   - Benchmark against Python implementation

## Phase 3: Critical Implementation Details

### Memory Management
- Use `Arc<dyn HRF>` for shared HRF objects
- Implement `Clone` for all public types
- Consider arena allocators for temporary computations

### Error Handling
```rust
#[derive(Debug, thiserror::Error)]
pub enum HRFError {
    #[error("Invalid time points: {0}")]
    InvalidTimePoints(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Numerical error: {0}")]
    NumericalError(String),
}
```

### Performance Targets
- 10x faster than Python for basic operations
- 50x faster for convolution operations
- Memory usage: comparable or better

### Compatibility Layer
- Consider PyO3 bindings for gradual migration
- JSON serialization for cross-validation

## Next Steps

1. Set up Rust project structure
2. Implement core types and traits
3. Port SPM canonical HRF as proof of concept
4. Set up cross-validation framework
5. Iterate through remaining modules

## Success Metrics
- ✅ All tests passing
- ✅ Performance improvements demonstrated
- ✅ API documentation complete
- ✅ Cross-validation with Python
- ✅ Published to crates.io