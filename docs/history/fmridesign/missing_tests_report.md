# Missing Tests Report: Python vs Rust fmridesign

This report compares the test coverage between the Python (`pyfmridesign`) and Rust (`fmridesign-rs`) implementations, identifying tests that need to be ported to the Rust version.

## Summary

The Python test suite has **27 test files** covering various functionality, while the Rust implementation has **12 test files**. Many critical test categories are missing in the Rust implementation.

## Missing Test Files in Rust

### 1. **Baseline Model Tests** (`test_baseline_model.py`, `test_baseline_specs.py`)
- **Missing Coverage:**
  - Constant baseline tests
  - Polynomial baseline with orthogonalization
  - B-spline baseline tests
  - Natural spline baseline tests
  - Multi-block baseline handling
  - Global vs runwise intercepts
  - Nuisance regressor integration
  - Block subsetting functionality
  - BaselineSpec creation and validation

### 2. **Covariate Tests** (`test_covariate.py`)
- **Missing Coverage:**
  - CovariateTerm creation and validation
  - CovariateEvent functionality
  - Interpolation of covariate values
  - Integration with EventModel
  - No HRF convolution for covariates
  - Covariate naming with prefixes

### 3. **Visualization Tests** (`test_visualization.py`, `test_plot_contrasts.py`)
- **Missing Coverage:**
  - Design matrix plotting
  - Correlation map visualization
  - Model summary plots
  - Contrast weight visualization
  - F-contrast plotting with multiple components
  - Color scale handling (diverging vs one-sided)
  - Custom colormap support

### 4. **Trialwise Modeling Tests** (`test_trialwise.py`)
- **Missing Coverage:**
  - Single-trial regressor generation
  - Trial mean computation
  - Custom trial labeling
  - Integration with regular event terms
  - Large dataset handling (100+ trials)
  - Zero-padded trial naming

### 5. **Naming Utilities Tests** (`test_naming.py`)
- **Status:** Partially covered (has unit tests in `naming.rs`)
- **Missing Coverage:**
  - Short name abbreviation (`shortnames` function)
  - Additional edge cases for sanitization
  - Integration tests with real model data

### 6. **Regressor Extraction Tests** (`test_regressors.py`)
- **Missing Coverage:**
  - Term-specific regressor extraction
  - Regex pattern matching for regressors
  - Baseline inclusion/exclusion
  - Dictionary output format
  - Multi-term extraction

### 7. **Design Matrix Tests** (`test_design_matrix.py`)
- **Missing Coverage:**
  - Direct design matrix creation
  - Column selection and ordering
  - Matrix properties validation
  - Integration with various model types

### 8. **Cell and Condition Tests** (`test_cells_conditions.py`)
- **Missing Coverage:**
  - Cell enumeration for single categorical events
  - Cell enumeration for categorical interactions
  - Cell handling for continuous variables
  - Condition extraction with proper naming
  - Empty cell dropping functionality
  - Count attributes in cell dataframes
  - Mixed categorical/continuous interactions

### 9. **HRF Integration Tests** (`test_hrf_integration.py`)
- **Missing Coverage:**
  - Different HRF basis functions
  - HRF parameter customization
  - Multiple basis function support
  - HRF convolution validation

### 10. **Import Tests** (`test_imports.py`)
- **Missing Coverage:**
  - Public API surface validation
  - Module structure verification

### 11. **Utility Function Tests** (`test_utils.py`)
- **Missing Coverage:**
  - Helper function validation
  - Data transformation utilities
  - Validation functions

### 12. **Generics Tests** (`test_generics.py`)
- **Missing Coverage:**
  - Generic type handling
  - Type conversion functions

### 13. **Event Table Tests** (`test_event_table.py`)
- **Missing Coverage:**
  - Event table creation from data
  - Event aggregation
  - Duration handling

### 14. **Convolve Design Tests** (`test_convolve_design.py`)
- **Missing Coverage:**
  - Design matrix convolution
  - Multi-channel convolution
  - Edge case handling

### 15. **As HRF Tests** (`test_as_hrf.py`)
- **Missing Coverage:**
  - HRF basis function conversion
  - Parameter validation

## Partially Covered Areas

### 1. **Basis Functions** 
- Rust has `test_basis_functions.rs` but missing:
  - Natural splines
  - Fourier basis
  - Custom basis functions
  - Basis function derivatives

### 2. **Formula Parsing**
- Rust has `test_formula_parser.rs` but missing:
  - Complex interaction terms
  - Nested functions
  - Error recovery
  - Formula validation edge cases

### 3. **Events**
- Rust has `test_events.rs` but missing:
  - Event aggregation
  - Duration handling variations
  - Multi-run event handling

### 4. **Contrasts**
- Rust has `test_contrast_weights.rs` and `test_fcontrast.rs` but missing:
  - Contrast estimation
  - Contrast variance computation
  - Multiple comparison corrections
  - Orthogonal contrast validation

## Integration Test Gaps

The Python suite includes many integration tests that combine multiple components:

1. **Baseline + Event Model Integration**
2. **Covariate + Event Model Integration**
3. **Visualization of Complex Models**
4. **Multi-run Design Matrix Construction**
5. **Formula + HRF + Baseline Integration**

## Recommendations

### Priority 1 (Core Functionality)
1. Port baseline model tests
2. Port covariate tests
3. Port naming utility tests
4. Add comprehensive integration tests

### Priority 2 (Analysis Features)
1. Port regressor extraction tests
2. Port cells/conditions tests
3. Port design matrix manipulation tests
4. Port trialwise modeling tests

### Priority 3 (Visualization & Utilities)
1. Port visualization tests (if implementing viz in Rust)
2. Port utility function tests
3. Port import/API surface tests

### Testing Infrastructure Needs
1. Add fixture generation similar to Python's `conftest.py`
2. Implement test data generation utilities
3. Add property-based testing for mathematical functions
4. Add benchmarking tests for performance-critical paths

## Modules with Internal Unit Tests

The following Rust modules have internal unit tests but may need additional integration testing:

- `afni.rs` - AFNI HRF functions
- `baseline.rs` - Baseline model functionality
- `covariate.rs` - Covariate handling
- `naming.rs` - Naming utilities
- `regressors.rs` - Regressor extraction
- `trialwise.rs` - Trial-wise modeling
- `convolve.rs` - Convolution operations
- `convolve_design.rs` - Design matrix convolution
- `design.rs` - Design matrix construction
- `sampling.rs` - Sampling frame handling
- `events/` - Various event-related modules
- `formula/` - Formula parsing and DSL
- `basis/` - Basis function transformations
- `utils/` - Various utility modules

These modules have basic unit tests but lack comprehensive integration tests that verify their behavior in combination with other components.

## Test Organization Suggestions

Consider organizing Rust tests to mirror Python structure:
```
tests/
├── unit/
│   ├── basis/
│   ├── contrast/
│   ├── events/
│   └── ...
├── integration/
│   ├── model_building/
│   ├── design_matrix/
│   └── ...
└── fixtures/
    └── test_data/
```

This would make it easier to track parity and ensure comprehensive coverage.

## Summary Statistics

### Current Test Coverage
- **Python**: 27 test files with ~2,500+ test cases
- **Rust**: 12 test files (integration) + 27 modules with internal unit tests

### Critical Gaps
1. **No baseline model integration tests** - Core functionality for fMRI analysis
2. **No covariate support tests** - Essential for including nuisance regressors
3. **Limited design matrix manipulation tests** - Key for model building
4. **No visualization tests** - Important for model validation
5. **Missing comprehensive integration tests** - Critical for ensuring components work together

### Estimated Effort
- **High Priority Tests**: ~40-60 hours of development
- **Medium Priority Tests**: ~20-30 hours of development  
- **Low Priority Tests**: ~10-20 hours of development
- **Total Estimated Effort**: ~70-110 hours to achieve test parity

The most critical missing pieces are baseline model tests and covariate support, as these are fundamental to fMRI design matrix construction.