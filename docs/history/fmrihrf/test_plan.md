# fmrihrf-rs Test Plan

This document tracks the porting status of all tests from the R package `fmrihrf` (located at `~/code/fmrihrf/tests/testthat`) to the Rust implementation.

## Overview

- **Total R Test Files**: 14
- **Total R Test Cases**: ~95 individual tests
- **Rust Test Files**: 5 (cross_validation.rs, edge_case_tests.rs, integration_test.rs, test_serialization.rs, workflow_tests.rs)
- **Status**: In Progress

Legend:
- ✅ = Implemented and passing in Rust
- 🚧 = Partially implemented in Rust
- ❌ = Not yet implemented in Rust

---

## Test File: test_acquisition_onsets.R (9 tests)

Tests for `acquisition_onsets()` function that computes global acquisition onset times for fMRI scans.

- [ ] ❌ **test_1**: acquisition_onsets returns same values as samples(global=TRUE)
  - Single block consistency
  - Multiple blocks consistency
  - Variable TR consistency

- [ ] ❌ **test_2**: acquisition_onsets handles default start_time correctly
  - First onset should be at TR/2
  - Spacing should be TR

- [ ] ❌ **test_3**: acquisition_onsets handles custom start_time correctly
  - start_time = 0
  - start_time = custom value
  - Variable start_time per block

- [ ] ❌ **test_4**: acquisition_onsets handles variable TR correctly
  - Different TR per block
  - Correct timing calculation

- [ ] ❌ **test_5**: acquisition_onsets works with multi-block designs
  - Three blocks scenario
  - Correct transitions between blocks

- [ ] ❌ **test_6**: acquisition_onsets edge cases
  - Single scan
  - Very short TR
  - Large number of blocks

- [ ] ❌ **test_7**: acquisition_onsets matches expected timing for standard fMRI
  - Standard fMRI parameters (150 scans, TR=2)
  - Verify timing at block boundaries

---

## Test File: test_deriv.R (7 tests)

Tests for HRF derivative calculations.

- [ ] ❌ **test_1**: deriv works for SPMG1 HRF
  - Returns numeric vector
  - Derivative is 0 at t=0
  - Matches analytic derivative

- [ ] ❌ **test_2**: deriv works for SPMG2 HRF
  - Returns matrix with 2 columns
  - First column matches SPMG1 derivative
  - Second column is second derivative

- [ ] ❌ **test_3**: deriv works for SPMG3 HRF
  - Returns matrix with 3 columns
  - First two columns match SPMG2
  - Third column is third derivative

- [ ] ❌ **test_4**: deriv works with numeric differentiation for other HRF types
  - Gaussian HRF derivative
  - Gamma HRF derivative

- [ ] ❌ **test_5**: deriv works for multi-basis HRFs
  - B-spline HRF derivatives
  - Correct matrix dimensions

- [ ] ❌ **test_6**: deriv handles edge cases
  - Single time point
  - Empty time vector
  - Negative times (should be 0)

---

## Test File: test_fft_and_toeplitz.R (3 tests)

Tests for FFT-based evaluation and Toeplitz matrix construction.

- [ ] 🚧 **test_1**: evaluate.Reg fft method matches conv
  - FFT convolution matches direct convolution
  - _Note: Partial implementation in edge_case_tests.rs_

- [ ] ❌ **test_2**: hrf_toeplitz constructs correct Toeplitz matrix
  - Dense matrix construction
  - Sparse matrix construction
  - Matrix values match manual construction

---

## Test File: test_hrf.R (49 tests)

Comprehensive tests for HRF objects and functions. This is the largest test file.

### Basic HRF Structure Tests (4 tests)

- [ ] 🚧 **test_1**: HRF_GAMMA has correct structure and properties
  - Inherits HRF class
  - Has correct attributes (name, param_names)
  - Function evaluation produces correct dimensions
  - Values are non-negative
  - _Note: Some coverage in cross_validation.rs_

- [ ] ✅ **test_2**: HRF_SPMG1 has correct structure and properties
  - Inherits HRF class, has correct attributes
  - Function evaluation, peak timing around 5-6 seconds
  - _Note: Implemented in cross_validation.rs and integration_test.rs_

- [ ] 🚧 **test_3**: HRF_SPMG2 has correct structure and properties
  - Inherits HRF class, nbasis = 2
  - Returns matrix with 2 columns
  - _Note: Partial coverage in workflow_tests.rs_

- [ ] 🚧 **test_4**: HRF_GAUSSIAN has correct structure and properties
  - Inherits HRF class, correct attributes
  - Non-negative values

- [ ] 🚧 **test_5**: HRF_BSPLINE has correct structure and properties
  - Default 5 basis functions
  - Returns correct matrix dimensions
  - _Note: Some coverage in cross_validation.rs_

### HRF Evaluation Tests (3 tests)

- [ ] ❌ **test_6**: evaluate.HRF handles different duration scenarios
  - Zero duration
  - Non-zero duration

- [ ] ❌ **test_7**: evaluate.HRF handles different durations and summation correctly
  - Non-zero duration increases response
  - Summation vs non-summation
  - Precision effects

- [ ] ❌ **test_8**: evaluate.HRF validates grid and precision
  - Empty grid error
  - NA values in grid error
  - Invalid precision errors

### HRF Generator and Decorator Tests (8 tests)

- [ ] 🚧 **test_9**: gen_hrf handles lag and width correctly
  - Lag shifts peak timing
  - Width (block duration) increases response width
  - Combined lag and width
  - _Note: Some coverage in edge_case_tests.rs test_hrf_decorators_complex()_

- [ ] ❌ **test_10**: gen_hrf_set combines HRFs correctly
  - Creates basis set from multiple HRFs
  - Correct structure and nbasis
  - Peaks are correctly shifted

- [ ] 🚧 **test_11**: lag_hrf correctly lags an HRF object
  - Basic structure preserved
  - Peak timing shifted by lag amount
  - Zero lag returns unchanged
  - Multi-basis HRF support
  - _Note: Partial implementation in decorators_

- [ ] 🚧 **test_12**: block_hrf correctly blocks an HRF object
  - Basic structure preserved
  - Summation vs max modes
  - Normalization support
  - Half-life decay
  - Negligible width returns unchanged
  - _Note: Partial implementation in decorators_

- [ ] ❌ **test_13**: normalise_hrf correctly normalises an HRF object
  - Peak value equals 1
  - Already normalized HRF remains normalized
  - Multi-basis support

- [ ] ❌ **test_14**: normalised multi-basis HRF evaluated at single point returns matrix
  - Single point evaluation returns correct dimensions

- [ ] ❌ **test_15**: gen_hrf correctly sets nbasis for function inputs
  - Single basis functions (gaussian, spmg1)
  - Single basis HRF objects
  - Multi-basis HRF objects (spmg2, spmg3)
  - Functions with parameters determining nbasis (bspline)

- [ ] ❌ **test_16**: normalize in evaluate.HRF preserves matrix dimensions
  - Multi-basis HRF normalization
  - Single point evaluation

### Empirical HRF Tests (1 test)

- [ ] ✅ **test_17**: gen_empirical_hrf creates valid HRF
  - Correct structure and attributes
  - Interpolation works
  - Extrapolation returns 0
  - _Note: Implemented in edge_case_tests.rs_

### HRF Attribute Tests (1 test)

- [ ] 🚧 **test_18**: HRF objects maintain correct attributes
  - All standard HRFs have required attributes
  - Evaluation produces correct dimensions
  - _Note: Partial coverage across multiple test files_

### as_hrf Tests (3 tests)

- [ ] ❌ **test_19**: as_hrf creates valid HRF objects
  - Creates HRF from function with parameters
  - Correct attributes
  - Function evaluation works
  - Invalid parameters trigger warnings
  - Default values work
  - Multi-basis support

- [ ] ❌ **test_20**: as_hrf captures parameters in closures
  - Parameters correctly captured for gamma HRF
  - Parameters correctly captured for gaussian HRF
  - Different from default parameters

- [ ] ❌ **test_21**: as_hrf validates parameter names
  - Invalid parameter names trigger warnings
  - Mix of valid/invalid parameters
  - Valid parameters don't warn

### bind_basis Tests (1 test)

- [ ] ❌ **test_22**: bind_basis combines HRF objects correctly
  - Combines multiple single-basis HRFs
  - Correct attributes (name, nbasis, span)
  - Function evaluation correct
  - Multi-basis input support
  - Single element binding

### hrf_library Tests (1 test)

- [ ] ❌ **test_23**: hrf_library produces distinct basis functions with parameters
  - Creates library from parameter grid
  - Correct structure and nbasis
  - Columns are distinct (not collinear)
  - Multiple non-zero singular values

### Decorator Enforcement Tests (2 tests)

- [ ] ❌ **test_24**: lag_hrf and block_hrf enforce finite parameters
  - Inf and NA rejected for lag
  - Inf and NA rejected for width/precision
  - Inf allowed for half_life

---

## Test File: test_hrf_from_coefficients.R (1 test)

Tests for creating HRFs from linear combinations of basis functions.

- [ ] ❌ **test_1**: hrf_from_coefficients combines basis correctly
  - Creates single-basis HRF from multi-basis HRF
  - Correct structure and attributes
  - Linear combination matches expected result

---

## Test File: test_hrf_library.R (2 tests)

Tests for `hrf_library()` function that creates HRF libraries from parameter grids.

- [ ] ❌ **test_1**: hrf_library forwards extra arguments
  - Parameter grid with extra arguments
  - Correct structure and nbasis
  - Function evaluation matches expected

- [ ] ❌ **test_2**: duplicate parameter names cause an error
  - Error when parameter in grid conflicts with extra argument

---

## Test File: test_hrf_lwu_extra.R (2 tests)

Tests for Lindquist-Wager-Wager (LWU) HRF model.

- [ ] ❌ **test_1**: hrf_lwu computes response and normalisation
  - Basic computation
  - Height normalization
  - Area normalization (currently warns)

- [ ] ❌ **test_2**: hrf_basis_lwu returns derivatives
  - Returns 4-column matrix
  - First column matches hrf_lwu
  - Finite-difference check for derivatives

---

## Test File: test_penalty_matrix.R (4 tests)

Tests for penalty matrix generation for regularization.

- [ ] ✅ **test_1**: bspline penalty is roughness-based
  - Non-diagonal matrix for b-spline
  - Correct dimensions
  - _Note: Implemented - penalty matrix exists in src/penalty.rs_

- [ ] ❌ **test_2**: SPMG3 penalty shrinks derivatives
  - Diagonal matrix with different shrinkage values
  - First element (canonical) = 0, derivatives shrunk

- [ ] ❌ **test_3**: fourier penalty increases with frequency
  - Diagonal matrix with frequency-squared weighting

- [ ] ✅ **test_4**: default HRF penalty is identity
  - Identity matrix for simple HRFs
  - _Note: Implemented in src/penalty.rs_

---

## Test File: test_reg_constructor.R (2 tests)

Tests for Regressor (Reg) constructor validation.

- [ ] 🚧 **test_1**: Reg validates inputs
  - Negative onsets rejected
  - NA onsets rejected
  - Zero/negative span rejected
  - Mismatched duration/amplitude lengths rejected
  - _Note: Some validation exists in src/regressor/core.rs_

- [ ] ❌ **test_2**: single_trial_regressor constructs single event
  - Creates single-event regressor
  - Correct structure

- [ ] ❌ **test_3**: single_trial_regressor validates lengths
  - Rejects multiple onsets
  - Rejects multiple durations/amplitudes

---

## Test File: test_regressor.R (13 tests)

Tests for Regressor evaluation and construction.

- [ ] 🚧 **test_1**: regressor constructs valid Reg objects
  - Basic construction
  - Correct attributes
  - Span from HRF
  - _Note: Partial coverage in workflow_tests.rs_

- [ ] ❌ **test_2**: events with zero amplitude are filtered
  - Zero amplitudes filtered out
  - All-zero amplitudes handled

- [ ] ❌ **test_3**: NA inputs trigger errors
  - Single NA onset special case
  - Multiple NA values rejected

- [ ] 🚧 **test_4**: invalid inputs are rejected
  - Negative/Inf values rejected
  - _Note: Some validation in regressor construction_

- [ ] ❌ **test_5**: single_trial_regressor returns a length-1 Reg
  - Creates single-event regressor

- [ ] ❌ **test_6**: shift.Reg shifts onsets correctly
  - Basic shifting
  - Preserves other attributes

- [ ] ❌ **test_7**: shift.Reg accepts offset argument
  - Alternative parameter name

- [ ] ❌ **test_8**: shift.Reg errors without shift specification
  - Missing shift amount rejected

- [ ] ✅ **test_9**: evaluate.Reg computes convolution correctly
  - Box HRF convolution
  - _Note: Implemented in convolution module_

- [ ] ❌ **test_10**: Rconv handles non-zero constant durations
  - Block durations with convolution
  - Rconv matches conv method

- [ ] ❌ **test_11**: unsorted grid triggers warning and sorted output
  - Warning for unsorted grid
  - Output is sorted

- [ ] 🚧 **test_12**: evaluate.Reg validates grid and precision
  - Empty grid rejected
  - NA values rejected
  - Invalid precision rejected
  - _Note: Some validation in convolution module_

---

## Test File: test_regressor_design.R (2 tests)

Tests for design matrix generation.

- [ ] ✅ **test_1**: regressor_design produces expected design
  - Multi-condition design matrix
  - Correct dimensions
  - _Note: Implemented in src/regressor/design.rs_

- [ ] ✅ **test_2**: regressor_design returns sparse matrix
  - Sparse matrix output support
  - _Note: Sparse matrix support exists_

---

## Test File: test_regressor_set.R (2 tests)

Tests for RegSet (multi-condition regressor sets).

- [ ] ✅ **test_1**: regressor_set constructs and evaluates
  - Multi-condition construction
  - Correct evaluation
  - _Note: Implemented in workflow_tests.rs test_regressor_set_workflow()_

- [ ] ✅ **test_2**: regressor_set produces sparse matrix
  - Sparse matrix output
  - _Note: Sparse matrix support exists_

---

## Test File: test_sampling_frame.R (11 tests)

Tests for SamplingFrame which defines fMRI acquisition timing.

- [ ] ✅ **test_1**: sampling_frame constructor works correctly
  - Basic construction
  - Different TRs per block
  - Input validation (negative values, mismatched lengths, NA values, type errors)
  - _Note: Implemented in src/sampling.rs and cross_validation.rs_

- [ ] 🚧 **test_2**: samples.sampling_frame works correctly
  - Relative timing
  - Global timing
  - Block selection
  - Memoization
  - _Note: Partial implementation in SamplingFrame_

- [ ] ❌ **test_3**: global_onsets works correctly
  - Basic functionality
  - Error conditions for non-integer block ids

- [ ] ❌ **test_4**: print.sampling_frame works correctly
  - Display output includes structure, timing, duration

- [ ] ✅ **test_5**: sampling_frame handles edge cases
  - Single block
  - Very short block
  - Different start times
  - High precision
  - _Note: Covered in edge_case_tests.rs_

- [ ] ❌ **test_6**: sampling_frame maintains temporal consistency
  - Uniform spacing within blocks
  - Correct block transitions

- [ ] ❌ **test_7**: samples with blockids parameter works correctly for local and global timing
  - Block filtering for local timing
  - Block filtering for global timing
  - Multiple blocks selection
  - Non-sequential block selection

- [ ] ❌ **test_8**: samples with varying TR and start_time per block work correctly
  - Different TR per block
  - Different start_time per block
  - Global timing respects both

- [ ] ❌ **test_9**: edge cases for blockids parameter
  - Empty blockids
  - Out of order blockids
  - Repeated blockids
  - Single block from middle

---

## Test File: test_utils.R (1 test)

Tests for utility functions.

- [ ] ❌ **test_1**: recycle_or_error recycles or errors correctly
  - Scalar recycling to length n
  - Vector of correct length unchanged
  - Vector of wrong length errors

---

## Summary Statistics

### By Test File

| Test File | Total Tests | ✅ Pass | 🚧 Partial | ❌ Not Impl | Priority |
|-----------|-------------|---------|------------|-------------|----------|
| test_acquisition_onsets.R | 7 | 0 | 0 | 7 | Medium |
| test_deriv.R | 6 | 0 | 0 | 6 | Low |
| test_fft_and_toeplitz.R | 3 | 0 | 1 | 2 | High |
| test_hrf.R | 24 | 2 | 7 | 15 | High |
| test_hrf_from_coefficients.R | 1 | 0 | 0 | 1 | Low |
| test_hrf_library.R | 2 | 0 | 0 | 2 | Low |
| test_hrf_lwu_extra.R | 2 | 0 | 0 | 2 | Low |
| test_penalty_matrix.R | 4 | 2 | 0 | 2 | Medium |
| test_reg_constructor.R | 3 | 0 | 1 | 2 | High |
| test_regressor.R | 12 | 1 | 2 | 9 | High |
| test_regressor_design.R | 2 | 2 | 0 | 0 | ✅ Complete |
| test_regressor_set.R | 2 | 2 | 0 | 0 | ✅ Complete |
| test_sampling_frame.R | 9 | 1 | 1 | 7 | High |
| test_utils.R | 1 | 0 | 0 | 1 | Low |

### Overall Progress

- **Total Test Cases**: 78
- **Fully Implemented**: 10 (13%)
- **Partially Implemented**: 12 (15%)
- **Not Implemented**: 56 (72%)

---

## High Priority Test Gaps

These tests are critical for core functionality and should be implemented first:

1. **HRF Evaluation and Decorators** (test_hrf.R)
   - Complete lag_hrf and block_hrf testing
   - normalise_hrf implementation
   - as_hrf parameter handling
   - bind_basis functionality

2. **FFT Convolution** (test_fft_and_toeplitz.R)
   - FFT vs direct convolution equivalence
   - Toeplitz matrix construction

3. **Regressor Validation** (test_regressor.R)
   - Complete input validation
   - Zero amplitude filtering
   - shift.Reg functionality
   - Grid validation

4. **SamplingFrame** (test_sampling_frame.R)
   - samples() method with blockids
   - global_onsets() functionality
   - Variable TR and start_time per block

5. **Regressor Constructor** (test_reg_constructor.R)
   - Complete input validation
   - single_trial_regressor support

---

## Medium Priority Test Gaps

Secondary functionality that enhances usability:

1. **Acquisition Onsets** (test_acquisition_onsets.R)
   - Complete acquisition_onsets() implementation
   - Edge cases and multi-block support

2. **Penalty Matrices** (test_penalty_matrix.R)
   - SPMG3 derivative shrinkage
   - Fourier frequency weighting

---

## Low Priority Test Gaps

Advanced features and utilities:

1. **Derivatives** (test_deriv.R)
   - Analytic derivatives for HRFs
   - Numeric differentiation fallback

2. **HRF From Coefficients** (test_hrf_from_coefficients.R)
   - Linear combination of basis functions

3. **HRF Library** (test_hrf_library.R)
   - Parameter grid library generation

4. **LWU Model** (test_hrf_lwu_extra.R)
   - Lindquist-Wager HRF implementation

5. **Utils** (test_utils.R)
   - recycle_or_error utility

---

## Notes

- Test priority is based on:
  1. Core functionality required for basic use
  2. Features mentioned in project documentation
  3. Dependencies (tests for foundational features first)
  4. Frequency of use in typical fMRI analysis

- Many Rust tests exist but test different aspects than the R tests. Cross-validation tests verify output equivalence, but don't test all edge cases and error conditions.

- The R package has more comprehensive validation and error handling tests that should be ported to ensure robust behavior.
