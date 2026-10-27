# Feature Parity Report: Python pyfmrihrf vs Rust fmrihrf-rs

Generated: 2025-07-01

## Executive Summary

This report provides a comprehensive comparison between the Python `pyfmrihrf` and Rust `fmrihrf-rs` implementations. The analysis shows that the Rust implementation covers most core functionality but lacks several advanced features present in the Python version.

**Overall Completeness: ~70%**

## 1. Core HRF Functions Comparison

| Function | Python | Rust | Status | Notes |
|----------|--------|------|--------|-------|
| `spm_canonical` | ✅ | ✅ | ✅ Complete | Full implementation with normalization |
| `gamma_hrf` | ✅ | ✅ | ✅ Complete | Shape/rate parameterization |
| `gaussian_hrf` | ✅ | ✅ | ✅ Complete | Mean/SD parameterization |
| `bspline_hrf` | ✅ | ✅ | ✅ Complete | B-spline basis with Cox-de Boor recursion |
| `fourier_hrf` | ✅ | ✅ | ✅ Complete | Sin/cos basis functions |
| `fir_basis` | ✅ | ✅ | ✅ Complete | Finite impulse response basis |
| `mexhat_hrf` | ✅ | ✅ | ✅ Complete | Mexican hat wavelet |
| `sine_hrf` | ✅ | ✅ | ✅ Complete | Simple sine wave |
| `half_cosine_hrf` | ✅ | ✅ | ✅ Complete | Half-cosine implementation |
| `inv_logit_hrf` | ✅ | ✅ | ✅ Complete | Inverse logit (sigmoid) |
| `lwu_hrf` | ✅ | ❌ | ❌ Missing | Lindquist-Wager-Ungerleider basis |
| `hrf_time` | ✅ | ❌ | ❌ Missing | Linear time HRF |
| `hrf_mexhat` (variant) | ✅ | ❌ | ❌ Missing | Alternative Mexican hat implementation |
| `hrf_inv_logit` (variant) | ✅ | ❌ | ❌ Missing | Double sigmoid variant |
| `hrf_half_cosine` (variant) | ✅ | ❌ | ❌ Missing | Four-segment half-cosine |
| `hrf_sine` (basis) | ✅ | ❌ | ❌ Missing | Sine basis set |
| `hrf_lwu` (variant) | ✅ | ❌ | ❌ Missing | LWU with normalization options |
| `hrf_basis_lwu` | ✅ | ❌ | ❌ Missing | LWU basis with derivatives |

## 2. Module-by-Module Comparison

### 2.1 HRF Module

| Component | Python Files | Rust Files | Completeness | Missing Features |
|-----------|-------------|------------|--------------|------------------|
| Core | `core.py` | `core.rs`, `mod.rs` | 85% | - `evaluate()` method with duration parameter<br>- `from_coefficients()` method<br>- `bind_basis()` function |
| Functions | `functions.py` | `functions.rs` | 70% | - LWU variants<br>- Time-based HRF<br>- Alternative implementations |
| Generators | `generators.py` | `generators.rs` | 60% | - `gen_hrf()` with string/dict specs<br>- `make_hrf()` parser<br>- `daguerre_generator()`<br>- Parameter parsing from strings |
| Registry | `registry.py` | `registry.rs` | 90% | - Parametric HRF registration<br>- Aliases support |
| Empirical | `empirical.py` | `empirical.rs` | 95% | Near complete |
| Library | `library.py`, `hrf_library.py` | - | 0% | No predefined HRF library in Rust |
| Decorators | `decorators.py` | (in `generators.rs`) | 50% | - `normalize_hrf` decorator<br>- `scale_hrf` decorator<br>- Decorator chaining pattern |
| Derivatives | `derivatives.py` | (partial in `functions.rs`) | 30% | - Analytic derivatives<br>- Dedicated derivative functions |
| SPM HRF | `spm_hrf.py` | - | 0% | - SPMG1/2/3 classes with analytic derivatives |
| Penalty | `penalty.py` | `penalty.rs` | 100% | Complete |
| Reconstruction | `reconstruction.py` | `reconstruction.rs` | 100% | Complete |

**HRF Module Completeness: ~65%**

### 2.2 Regressor Module

| Component | Python Files | Rust Files | Completeness | Missing Features |
|-----------|-------------|------------|--------------|------------------|
| Core | `core.py` | `core.rs` | 85% | - `evaluate()` method<br>- `as_matrix()` method<br>- Automatic HRF loading from registry |
| Convolution | `convolution.py` | `convolution.rs` | 90% | - Edge case handling differences |
| Design | `design.py` | `design.rs` | 70% | - `append()` method<br>- Column naming utilities<br>- Flexible concatenation |
| Neural Input | `neural_input.py` | `neural_input.rs` | 80% | - More activation functions<br>- Parameter validation |

**Regressor Module Completeness: ~80%**

### 2.3 Other Modules

| Module | Python | Rust | Completeness | Missing Features |
|--------|--------|------|--------------|------------------|
| Sampling | `sampling.py` | `sampling.rs` | 95% | - Minor API differences |
| Utils | `utils/` (misc.py, cache.py) | `utils/` | 10% | - Caching mechanisms<br>- Utility functions |
| Types | - | `types.rs` | N/A | Rust-specific type definitions |
| Error | - | `error.rs` | N/A | Rust-specific error handling |
| Plotting | - | `plotting.rs` | N/A | Rust-specific, not in Python |

**Other Modules Completeness: ~50%**

## 3. Feature Completeness by Category

### 3.1 Core Functionality (90%)
- ✅ Basic HRF evaluation
- ✅ Multi-basis support
- ✅ Time point evaluation
- ✅ Parameter handling
- ❌ Duration-based evaluation (evaluate method)
- ❌ Coefficient-based HRF construction

### 3.2 HRF Types (70%)
- ✅ SPM canonical (single basis)
- ✅ SPM with derivatives (multi-basis)
- ✅ Gamma distributions
- ✅ Gaussian
- ✅ B-spline basis
- ✅ Fourier basis
- ✅ FIR basis
- ✅ Basic wavelets
- ❌ LWU family
- ❌ Time-based HRFs
- ❌ Complex composite HRFs

### 3.3 Advanced Features (40%)
- ✅ Empirical HRF from data
- ✅ HRF registry
- ✅ Basic decorators (lag, block)
- ❌ Full decorator pattern
- ❌ Analytic derivatives
- ❌ SPM HRF classes (SPMG1/2/3)
- ❌ String-based HRF specification
- ❌ Caching mechanisms
- ❌ Comprehensive HRF library

### 3.4 Design Matrix Construction (80%)
- ✅ Regressor creation
- ✅ Design matrix assembly
- ✅ Convolution operations
- ✅ Multi-block support
- ❌ Advanced concatenation methods
- ❌ Column naming utilities

### 3.5 Utility Functions (20%)
- ✅ Basic error handling
- ✅ Type definitions
- ❌ Caching utilities
- ❌ Miscellaneous helpers
- ❌ Validation utilities

## 4. Missing Critical Features

### High Priority
1. **LWU HRF family** - Important for certain analyses
2. **String-based HRF specification** - User convenience
3. **SPM HRF classes with analytic derivatives** - Performance and accuracy
4. **evaluate() with duration** - Block design support
5. **Predefined HRF library** - User convenience

### Medium Priority
1. **Full decorator pattern** - Flexibility
2. **Coefficient-based HRF construction** - Advanced usage
3. **Alternative HRF implementations** - Compatibility
4. **Advanced design matrix methods** - Convenience
5. **Caching mechanisms** - Performance

### Low Priority
1. **Utility functions** - Nice to have
2. **Additional activation functions** - Rarely used
3. **Minor API differences** - Can be adapted

## 5. Implementation Recommendations

### Phase 1: Core Features (1-2 weeks)
1. Implement LWU HRF family
2. Add evaluate() with duration support
3. Implement coefficient-based HRF construction
4. Add string-based HRF specification parser

### Phase 2: Advanced Features (2-3 weeks)
1. Create SPM HRF classes with analytic derivatives
2. Implement full decorator pattern
3. Add predefined HRF library
4. Enhance design matrix utilities

### Phase 3: Polish (1 week)
1. Add remaining utility functions
2. Implement caching where beneficial
3. Ensure API compatibility
4. Complete documentation

## 6. Summary Statistics

| Category | Completeness | Priority |
|----------|--------------|----------|
| Core HRF Functions | 70% | High |
| HRF Module | 65% | High |
| Regressor Module | 80% | Medium |
| Sampling | 95% | Low |
| Utils | 10% | Low |
| **Overall** | **~70%** | - |

## 7. Conclusion

The Rust implementation provides a solid foundation with ~70% feature parity. Core functionality is well-implemented, but several advanced features from the Python version are missing. The recommended implementation phases would bring the Rust version to near-complete parity within 4-6 weeks of development effort.

Key strengths of the Rust implementation:
- Type safety and error handling
- Performance characteristics
- Core functionality coverage

Key gaps requiring attention:
- LWU HRF family
- Advanced HRF features
- Utility functions
- API convenience methods