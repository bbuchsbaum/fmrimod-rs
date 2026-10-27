# Comprehensive Gap Analysis: R fmrihrf vs Rust fmrihrf-rs
**Date**: 2025-11-17
**Source of Truth**: R package at `/Users/bbuchsbaum/code/fmrihrf`
**Analysis Method**: 7 Parallel Sub-Agent Analysis

## Executive Summary

This comprehensive gap analysis compares the **authoritative R package** `fmrihrf` with the Rust port `fmrihrf-rs` across 7 major functional areas. The R package is the source of truth; the Python version (`pyfmrihrf`) has drifted from the R source.

### Overall Completeness: **73%**

| Section | Completeness | Critical Gaps | High Priority Gaps |
|---------|--------------|---------------|-------------------|
| 1. HRF Core & S3 System | 46% | 3 | 4 |
| 2. HRF Functions & Implementations | 72% | 1 | 6 |
| 3. HRF Decorators & Generators | 80% | 0 | 3 |
| 4. HRF Advanced Features | 75% | 2 | 3 |
| 5. Regressor Core & Construction | 75% | 3 | 3 |
| 6. Regressor Sets & Design | 90% | 1 | 2 |
| 7. Sampling Frame & Infrastructure | 82% | 1 | 2 |

---

## Section 1: HRF Core & S3 System (46% Complete)

### Critical Missing Features

**1. S3 Generic Method System** - **CRITICAL**
- R has 17+ S3 generic methods with polymorphic dispatch
- Missing in Rust: `penalty_matrix()`, `reconstruction_matrix()`, accessors for Reg/RegSet
- **Impact**: Core to R's API design philosophy
- **Severity**: CRITICAL

**2. Reg/RegSet Trait System** - **CRITICAL**
- R has S3 methods for Reg: `evaluate()`, `shift()`, `onsets()`, `amplitudes()`, `durations()`, `nbasis()`, `neural_input()`
- Rust has struct with methods but no trait abstraction
- **Impact**: Cannot write generic code over regressors
- **Severity**: CRITICAL

**3. `hrf_from_coefficients()` Function** - **CRITICAL**
- R: `hrf_from_coefficients.HRF(hrf, h)` creates new callable HRF from weighted basis
- Rust: Missing entirely (partial in reconstruction module)
- **Impact**: Essential for model building and HRF recovery
- **Severity**: CRITICAL

### High Priority Gaps

4. **`bind_basis()` function** - Combines multiple HRFs into multi-basis sets
5. **`evaluate()` amplitude parameter** - R's evaluate() scales by amplitude; Rust requires external scaling
6. **Decorator parameter exposure** - `block_hrf()` missing `half_life`, configurable `precision`
7. **Multi-basis normalization** - R normalizes per-column; Rust uses global max

### Test Coverage
- R: 603 lines (30+ test cases)
- Rust: ~200 lines (6-7 test cases)
- **Coverage ratio: ~35%**

---

## Section 2: HRF Functions & Implementations (72% Complete)

### Critical Issues

**1. SPM Canonical HRF Formula Mismatch** - **CRITICAL**

R uses polynomial-exponential:
```r
exp(-t)*(A1*t^P1 - C*t^P2)  # P1=5, P2=15, A1=0.0833
```

Rust uses gamma difference:
```rust
g1 - c*g2  # gamma distributions
```

**These are COMPLETELY DIFFERENT formulas** producing different HRF shapes!

### High Priority Gaps

2. **Missing HRF variants**:
   - `hrf_ident()` - Identity/impulse function
   - 5-parameter `hrf_inv_logit()` - Difference of two logits (R has 5 params, Rust has 2)
   - 6-parameter `hrf_half_cosine()` - 4-segment model (R has 6 params, Rust has 2)
   - `hrf_tent_generator()` - Piecewise linear basis
   - `hrf_toeplitz()` - Matrix convolution representation

3. **SPMG2 column count mismatch**: R returns 2 columns, Rust returns 3

4. **Parameter naming inconsistencies**:
   - `hrf_gamma`: R uses `rate`, Rust uses `scale` (inverse relationship)
   - `hrf_gaussian`: R uses `mean/sd`, Rust uses `center/width`
   - Different defaults: R b-spline span=24, Rust span=20

### Function Coverage: 72%
- Implemented: 18 of 25 core functions
- Partial/Different: 7 functions
- Missing: 7 functions

---

## Section 3: HRF Decorators & Generators (80% Complete)

### High Priority Gaps

1. **`block_hrf()` missing parameters**:
   - Missing: `half_life` (exponential decay during blocks)
   - Missing: configurable `precision` (hardcoded to 0.1)
   - Missing: `summate` parameter control
   - **Impact**: Cannot model sustained stimuli with adaptation

2. **Multi-basis normalization bug**:
   - R: Normalizes each basis independently (per-column)
   - Rust: Normalizes globally (finds max across all columns)
   - **Impact**: Incorrect scaled values for multi-basis HRFs

3. **`empirical_hrf()` missing**:
   - R: Create HRFs from (t, y) data points using interpolation
   - Rust: Not implemented
   - **Impact**: Cannot create data-driven HRFs

### Medium Priority

4. **Multi-lag convenience**: R's `gen_hrf_lagged(lag=c(1,2,3))` accepts vector
5. **`gen_hrf_blocked()` missing** - Multi-width blocked HRF generator
6. **`list_available_hrfs()` metadata** - R returns rich data.frame; Rust only names

### Completeness: 80%
Rust string parsing is actually more robust than R's approach.

---

## Section 4: HRF Advanced Features (75% Complete)

### Critical Gaps

**1. `hrf_from_coefficients()` S3 Method** - **CRITICAL**
- R: Returns callable HRF object from weighted basis
- Rust: Only has `reconstruct_hrf()` which returns pre-computed values
- **Impact**: Cannot create reusable HRFs from fitted coefficients
- **Severity**: HIGH

**2. SPMG3 Third Derivative** - **MEDIUM**
- R: Implements third derivative (numerically)
- Rust: Not implemented
- **Impact**: Incomplete derivative support for SPMG3

### High Priority Gaps

3. **Penalty matrix coverage**: Actually **95% complete** - Well implemented!
4. **Reconstruction matrices**: Actually **110% complete** - Rust has MORE features!

### Excellent Areas
- ✅ Penalty matrix system (95% parity + enhancements)
- ✅ Reconstruction matrices (110% - more features than R)
- ✅ SPM G1/G2 derivatives (full parity)
- ✅ Comprehensive test coverage for penalty/reconstruction

---

## Section 5: Regressor Core & Construction (75% Complete)

### Critical Gaps

**1. Missing Evaluation Methods** - **HIGH**
- R provides 4 strategies: `conv`, `fft`, `Rconv` (R-based), `loop` (fallback)
- Rust has only 2: `conv`, `fft`
- Missing `Rconv` and `loop` methods limit flexibility

**2. No HRF Evaluation Caching** - **CRITICAL**
- R: Uses `memoise` with LRU cache (50MB, configurable)
- Rust: Re-evaluates HRF every time
- **Impact**: Major performance regression for repeated evaluations

**3. No Sparse Matrix Support** - **MEDIUM**
- R: `evaluate.Reg(..., sparse=TRUE)` returns sparse matrices
- Rust: Only dense matrices
- **Impact**: Memory inefficiency for large designs

### High Priority

4. **Neural input method on Regressor**: R has `neural_input.Reg()` S3 method
5. **User-friendly Display**: R has rich `print.Reg()` with `cli` package
6. **Missing accessor methods**: R has explicit `onsets()`, `durations()`, etc.

### Positive Notes
- ✅ Neural input generation is MORE efficient (O(E+N) vs R's O(N×E))
- ✅ Core structure matches R design
- ✅ Shift functionality complete

---

## Section 6: Regressor Sets & Design (90% Complete)

### Critical Gap

**1. Block-Wise Design Matrix Creation** - **CRITICAL**
- R: `regressor_design(onsets, fac, block, sframe, ...)`
  - Converts block-relative onsets to global time
  - Uses `global_onsets.sampling_frame()` for conversion
- Rust: Function with same name but DIFFERENT meaning
- **Impact**: Cannot create designs from block-relative onsets

### High Priority

2. **Sparse matrix support**: R supports `sparse=TRUE` parameter
3. **Factor level ordering**: R preserves factor order; Rust sorts alphabetically

### Medium Priority

4. **API naming ambiguity**: Rust's `regressor_design()` differs from R's
5. **Column naming** from conditions

### Excellent Areas
- ✅ RegressorSet construction (90%)
- ✅ Design matrix creation (85%)
- ✅ DesignMatrix struct with metadata (Rust enhancement)
- ✅ Builder patterns (Rust enhancement)

---

## Section 7: Sampling Frame & Infrastructure (82% Complete)

### Critical Gap

**1. `global_onsets()` Method Missing** - **HIGH**

R's `global_onsets.sampling_frame()`:
```r
global_onsets <- function(x, onsets, blockids, ...) {
  block_durations <- x$blocklens * x$TR
  cumulative_time <- c(0, cumsum(block_durations))
  onsets + cumulative_time[blockids]
}
```

- Converts block-relative event times to global experiment time
- Essential for multi-block designs
- **Status**: Missing from Rust

### High Priority

2. **Precision validation**: R checks `precision < min(TR)`; Rust doesn't
3. **Sparse Toeplitz matrix**: R supports `sparse=TRUE`; Rust doesn't

### Excellent Areas
- ✅ Core SamplingFrame (95%)
- ✅ Builder pattern (Rust enhancement)
- ✅ Serialization support (Rust enhancement)
- ✅ Thread-safe caching (Rust enhancement)
- ✅ `concatenate()`, `real_time()`, `total_time()` methods

### C++ Performance Baseline
R has optimized C++ implementations (via Rcpp):
- Difference array for impulse trains: O(E+N)
- FFT-based convolution: O(N log N)
- ⚠️ Need to verify Rust performance matches

---

## Consolidated Priority Recommendations

### 🔴 P0: Critical (Must Implement for Core Functionality)

1. **Resolve SPM Canonical Formula Mismatch** (Section 2)
   - CRITICAL: Different formulas produce different HRF shapes
   - Determine which is correct or document as intentional difference
   - Files: `R/hrf-functions.R:164`, `src/hrf/functions.rs:72`

2. **Implement S3-Equivalent Generic Methods** (Section 1)
   - Add trait system for Reg/RegSet with methods
   - Implement `penalty_matrix()` dispatch system
   - Implement `reconstruction_matrix()` dispatch
   - Essential for API compatibility

3. **Add `hrf_from_coefficients()`** (Sections 1, 4)
   - Create callable HRF from weighted basis combinations
   - Essential for model building workflows
   - Returns reusable HRF object (not just values)

4. **Implement HRF Caching** (Section 5)
   - Add memoization for HRF evaluations
   - Use LRU cache with configurable size
   - Major performance impact

5. **Add `global_onsets()` Method** (Sections 6, 7)
   - Convert block-relative to global time
   - Essential for block-wise design matrices
   - Part of SamplingFrame

6. **Fix SPMG2 Column Count** (Section 2)
   - R returns 2 columns, Rust returns 3
   - Breaking API difference

### 🟠 P1: High Priority (Important for Feature Parity)

7. **Add Missing Evaluation Methods** (Section 5)
   - Implement `Rconv` method (R-based convolution)
   - Implement `loop` method (event-by-event fallback)
   - Provides flexibility for edge cases

8. **Implement `block_hrf()` Parameters** (Sections 1, 3)
   - Add `half_life` parameter for exponential decay
   - Make `precision` configurable
   - Expose `summate` parameter

9. **Add Missing HRF Functions** (Section 2)
   - 5-parameter `hrf_inv_logit()`
   - 6-parameter `hrf_half_cosine()`
   - `hrf_toeplitz()`
   - `hrf_tent_generator()`

10. **Fix Multi-Basis Normalization** (Sections 1, 3)
    - Change from global to per-column normalization
    - Affects multi-basis HRF scaling

11. **Add Sparse Matrix Support** (Sections 5, 6, 7)
    - Support `sparse` parameter in evaluation
    - Use `sprs` crate
    - Memory efficiency for large designs

12. **Implement Block-Wise `regressor_design()`** (Section 6)
    - Create design from block-relative onsets
    - Integrate with SamplingFrame
    - Match R's API

### 🟡 P2: Medium Priority (Feature Completeness)

13. **Add `bind_basis()` function** (Section 1)
14. **Implement `empirical_hrf()`** (Section 3)
15. **Add SPMG3 third derivative** (Section 4)
16. **Add accessor trait methods** (Section 5)
17. **Fix parameter naming inconsistencies** (Section 2)
18. **Preserve factor level ordering** (Section 6)
19. **Add precision validation** (Section 7)

### 🟢 P3: Low Priority (Polish & Usability)

20. User-friendly Display traits
21. Additional test coverage
22. Documentation improvements
23. Performance benchmarking vs Rcpp

---

## Implementation Roadmap

### Phase 1: Critical Correctness (4-6 weeks)
**Goal**: Fix breaking issues and core missing functionality

- Week 1-2: Resolve SPM canonical formula, fix SPMG2 columns
- Week 3-4: Implement S3-equivalent trait system (Reg/RegSet traits)
- Week 5-6: Add `hrf_from_coefficients()` and `bind_basis()`

### Phase 2: Performance & Methods (3-4 weeks)
**Goal**: Match R's performance and evaluation flexibility

- Week 7-8: Implement HRF caching, add missing evaluation methods
- Week 9-10: Add sparse matrix support across modules

### Phase 3: Feature Parity (3-4 weeks)
**Goal**: Complete missing features

- Week 11: Implement `global_onsets()` and block-wise design
- Week 12: Fix `block_hrf()` parameters and normalization
- Week 13-14: Add missing HRF functions and variants

### Phase 4: Polish (2-3 weeks)
**Goal**: Test coverage, documentation, performance

- Week 15-16: Comprehensive testing and cross-validation
- Week 17: Performance benchmarking vs Rcpp
- Week 18: Documentation and examples

**Total Estimated Effort**: 12-17 weeks for 95%+ parity

---

## Cross-Validation Strategy

### Testing Approach

1. **Numerical Equivalence Tests**
   - Generate test cases in R, export to JSON/CSV
   - Import in Rust tests and compare outputs
   - Tolerance: 1e-6 for floating point

2. **Property-Based Tests**
   - Use R's test cases as property specifications
   - Verify invariants hold in Rust

3. **Performance Benchmarks**
   - Compare Rust vs Rcpp implementations
   - Target: Match or exceed Rcpp performance
   - Use R's `microbenchmark` package

4. **Integration Tests**
   - Full GLM workflows
   - Multi-block experiments
   - Various HRF types and combinations

### Test Data Generation

Create R script to generate reference outputs:
```r
library(fmrihrf)
library(jsonlite)

# Generate test cases
test_cases <- list(
  spm_canonical = list(
    times = seq(0, 20, by=0.1),
    values = hrf_spmg1(seq(0, 20, by=0.1))
  ),
  # ... more test cases
)

write_json(test_cases, "test_data/r_reference_outputs.json")
```

---

## Areas Where Rust Exceeds R

### 1. Type Safety
- ✅ Compile-time type checking
- ✅ No runtime type errors
- ✅ Clear error messages

### 2. Memory Safety
- ✅ No null pointer issues
- ✅ Thread-safe by default
- ✅ No memory leaks

### 3. Modern Infrastructure
- ✅ Builder patterns for ergonomics
- ✅ Explicit caching with control
- ✅ Serialization support (JSON)
- ✅ Better error handling (Result types)

### 4. Additional Features
- ✅ More reconstruction utilities
- ✅ Enhanced penalty matrix validation
- ✅ DesignMatrix struct with metadata
- ✅ SamplingFrame concatenation
- ✅ Real-time calculation methods

### 5. Performance Potential
- ✅ Zero-cost abstractions
- ✅ LLVM optimizations
- ✅ Native performance (no VM overhead)
- ⚠️ Need benchmarks to verify actual gains

---

## Key Differences: R vs Rust Paradigms

### S3 Classes vs Traits

**R (S3 System)**:
```r
# Generic definition
penalty_matrix <- function(x, ...) UseMethod("penalty_matrix")

# Method implementation
penalty_matrix.BSpline_HRF <- function(x, order=2, ...) { ... }
penalty_matrix.SPMG2_HRF <- function(x, order=2, ...) { ... }

# Runtime dispatch based on class
```

**Rust (Trait System)**:
```rust
// Trait definition
pub trait PenaltyMatrix {
    fn penalty_matrix(&self, order: usize) -> Result<Array2<f64>>;
}

// Implementation for specific type
impl PenaltyMatrix for BSplineHrf { ... }
impl PenaltyMatrix for SpmG2Hrf { ... }

// Compile-time dispatch (monomorphization)
// Or runtime dispatch via trait objects: Box<dyn PenaltyMatrix>
```

**Translation Strategy**:
- S3 generics → Rust traits
- S3 methods → Trait implementations
- S3 dispatch → Trait objects or monomorphization

### Data Structures

| Concept | R | Rust |
|---------|---|------|
| **Vectors** | `numeric()`, `integer()` | `Vec<f64>`, `Vec<i32>` |
| **Matrices** | `matrix()` | `Array2<f64>` (ndarray) |
| **Lists** | `list()` | `HashMap`, `Vec`, structs |
| **Factors** | `factor()` | `Vec<String>` + ordering |
| **Attributes** | `attr(x, "name")` | Struct fields |
| **NULL** | `NULL` | `Option<T>` |
| **NA** | `NA_real_` | `NaN`, `Option<T>` |

---

## Testing and Validation Plan

### Phase 1: Unit Test Parity
- Port all R test cases to Rust
- Current: ~35% coverage
- Target: 95% coverage

### Phase 2: Cross-Validation
- Generate R outputs for standardized inputs
- Compare Rust outputs within tolerance
- Document intentional differences

### Phase 3: Performance Validation
- Benchmark Rust vs Rcpp
- Identify performance regressions
- Optimize critical paths

### Phase 4: Integration Testing
- Full fMRI analysis workflows
- Multi-subject, multi-run studies
- Validate scientific correctness

---

## Documentation Requirements

### User Documentation
1. **Migration Guide**: R to Rust
   - API translation table
   - Common patterns conversion
   - Examples side-by-side

2. **Differences Document**
   - Intentional design differences
   - Breaking changes with rationale
   - Workarounds for R patterns

3. **Feature Parity Matrix**
   - Function-by-function comparison
   - Status indicators
   - Alternative approaches

### Developer Documentation
4. **Architecture Guide**
   - S3 to trait translation patterns
   - Memory management approach
   - Error handling conventions

5. **Contributing Guide**
   - How to port R features
   - Testing requirements
   - Cross-validation process

---

## Conclusion

The Rust implementation has achieved **73% feature parity** with the authoritative R package. The architecture is sound, with several areas where Rust actually exceeds R's capabilities (type safety, memory safety, modern patterns). However, there are critical gaps that must be addressed:

**Blocking Issues:**
1. SPM canonical formula mismatch (different HRF shapes!)
2. Missing S3-equivalent generic method system
3. Missing `hrf_from_coefficients()` functionality
4. No HRF evaluation caching (performance regression)
5. Missing block-wise design matrix creation

**High Priority:**
- Missing evaluation method diversity
- Sparse matrix support
- Several HRF variants and parameters
- Multi-basis normalization bug

**Strengths:**
- ✅ Solid architecture with trait-based design
- ✅ Better type and memory safety
- ✅ Modern infrastructure (builders, serialization)
- ✅ Some areas exceed R (reconstruction, validation)
- ✅ Performance potential (pending verification)

**Estimated Effort to 95% Parity**: 12-17 weeks of focused development

The roadmap above provides a clear path forward, prioritizing correctness issues before feature additions, then focusing on performance and polish.
