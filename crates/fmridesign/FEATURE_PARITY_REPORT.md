# Feature Parity Report: pyfmridesign (Python) vs fmridesign-rs (Rust)

This comprehensive report summarizes the feature parity analysis between the Python and Rust implementations of the fMRI design matrix library.

## Executive Summary

The Rust implementation has achieved approximately **75% feature parity** with the Python version. Core functionality is implemented and working, but several important features and refinements are missing.

### ✅ Implemented Features
- Core event types (EventFactor, EventVariable, EventMatrix, EventTerm)
- Basic EventModel with design matrix generation
- Polynomial and B-spline basis functions
- Basic transform functions (Scale, RobustScale, Ident)
- Contrast specifications and weight computation
- Formula parsing (actually more advanced than Python)
- AFNI GLT export
- Sub-basis functionality
- Regressor extraction utilities
- DataFrame conversion for events

### ❌ Missing Critical Features
1. **HRF Convolution in EventModel** - Essential for fMRI analysis
2. **Baseline Models** - Polynomial drift, nuisance regressors
3. **Covariate Support** - For motion parameters and other nuisances
4. **Functional Formula Interface** - The pipe operator (`|`) syntax
5. **Visualization** - Design matrix and correlation plots

## Detailed Gap Analysis

### 1. Core Event Types and Models

#### EventModel (High Priority)
- **Missing**: `_convolve_hrf()` method for HRF convolution
- **Missing**: Column naming system (`_get_term_tag()`, `_get_cond_tags()`)
- **Missing**: `cells()` and `conditions()` methods
- **Missing**: Proper handling of covariate and trialwise terms
- **Missing**: Column index tracking for terms

#### Event Types (Medium Priority)
- **EventFactor**: Missing `from_dataframe()` method
- **EventVariable**: Missing `bin_values()` for discretization
- **EventMatrix**: Missing callable transform support
- **EventBasis**: Missing `to_matrix()`, `change_basis()` methods
- **EventTerm**: Missing sophisticated mixed-type interaction handling

### 2. Basis Functions and Transforms

#### Missing Basis Functions
- **NaturalPolynomialBasis**: Implementation incomplete (missing centering)
- **NaturalSplineBasis**: Implementation incomplete (missing boundary constraints)
- **EventBasis**: Not implemented (for basis expansion of events)

#### Missing Transforms
- **ScaleWithin**: Group-wise standardization
- **Standardized**: Missing fit/transform pattern

#### Missing Methods
- `predict()` method incomplete across basis functions
- `get_range()` method for parameter bounds
- `basis_names` property for column naming

### 3. Contrast and Formula Systems

#### Contrast System
- Missing `contrast()` and `contrast_set()` constructor functions
- Where clause evaluation not implemented
- Less informative error messages

#### Formula System
- **Critical**: Missing functional interface (`functional.py` equivalent)
- Missing pipe operator (`|`) support
- Missing `@` operator in DSL
- Missing `compose()` function
- Missing `BasisNamespace` and `ChainedTransform`

### 4. Utilities and Helper Functions

#### Missing Utilities
- Advanced event splitting with structure preservation
- Generic singledispatch functions: `cells()`, `conditions()`, `labels()`, `levels()`
- `variance_explained()` calculation
- `event_table()` for event combinations

#### Missing Export Formats
- CSV/TSV export
- JSON export
- Excel export
- HDF5 export
- Event table export

#### Missing Visualization
- `design_map()` - Heatmap visualization
- `correlation_map()` - Correlation matrix plots
- `plot_design_matrix()` - Time series plots
- `plot_model_summary()` - Combined visualizations

### 5. Test Coverage

#### Test Statistics
- Python: 27 test files, ~2000+ test cases
- Rust: 12 integration tests + 116 unit tests

#### Critical Missing Tests
- Baseline model tests
- Covariate support tests
- Visualization tests
- Trialwise modeling tests
- Complex formula parsing tests
- Mixed-type interaction tests

## Implementation Priorities

### Phase 1: Critical Functionality (2-3 weeks)
1. **HRF Convolution** in EventModel
2. **Baseline Models** implementation
3. **Covariate Support** 
4. **Column Naming System**

### Phase 2: User Experience (2-3 weeks)
1. **Functional Formula Interface**
2. **Complete DSL** with missing operators
3. **Better error messages**
4. **CSV/JSON export**

### Phase 3: Advanced Features (2-3 weeks)
1. **Visualization** with plotters
2. **Missing basis functions** (Natural poly/spline)
3. **ScaleWithin transform**
4. **Event table functionality**

### Phase 4: Testing and Polish (1-2 weeks)
1. **Port missing tests**
2. **Integration tests**
3. **Documentation**
4. **Examples**

## Recommendations

1. **Immediate Action**: Implement HRF convolution and baseline models as these are essential for any real fMRI analysis.

2. **User Experience**: The functional formula interface is what makes the Python version user-friendly. This should be high priority.

3. **Testing**: Increase test coverage before adding more features to ensure stability.

4. **Documentation**: Create migration guide for Python users.

5. **Consider Dependencies**: 
   - Use `polars` for DataFrame functionality
   - Use `plotters` for visualization
   - Consider `pyo3` for Python interop

## Conclusion

The Rust implementation has a solid foundation with good type safety and performance characteristics. However, it needs approximately 6-10 weeks of focused development to achieve full feature parity with the Python version. The most critical missing pieces are HRF convolution, baseline models, and the functional formula interface.

The good news is that the Rust implementation has some advantages:
- Better type safety
- More robust formula parser
- Cleaner architecture in some areas
- Better performance potential

With the identified gaps addressed, the Rust version could become the preferred implementation for performance-critical fMRI analyses.