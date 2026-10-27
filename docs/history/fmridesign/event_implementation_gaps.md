# Event Implementation Gaps: Python vs Rust

This document provides a detailed comparison of the Python pyfmridesign event implementation vs the Rust fmridesign-rs implementation, highlighting missing methods, properties, and functionality.

## EventFactor (Categorical Events)

### Missing Methods/Properties in Rust:
1. **`from_dataframe` class method** - Creates EventFactor from pandas DataFrame
2. **`to_dataframe` method** - Converts to DataFrame (exists but different signature in Rust)
3. **`__repr__` method** - String representation with level preview
4. **BaseEvent properties**:
   - `n_events` property (Python has this as base class property)

### Implementation Differences:
- Python uses pandas.Categorical for values, Rust uses Vec<String>
- Python has more sophisticated DataFrame integration
- Python's design_matrix returns ndarray with different API

## EventVariable (Continuous Events)

### Missing Methods/Properties in Rust:
1. **`bin_values` method** - Convert continuous values to categorical by binning
2. **`from_dataframe` class method** - Creates EventVariable from DataFrame
3. **`to_dataframe` method** - Includes both transformed and raw values columns
4. **BaseEvent integration** - Python inherits from BaseEvent with validation lifecycle

### Implementation Differences:
- Python stores both `values` and `raw_values` as properties
- Python has more sophisticated transformation pipeline in `_transform_values`
- Python's DataFrame output includes both transformed and raw values

## EventMatrix (Multi-column Events)

### Missing Methods/Properties in Rust:
1. **`from_dataframe` class method** - Creates EventMatrix from DataFrame with flexible column selection
2. **`apply_transform` with callable support** - Python supports both matrix multiplication and callable transforms
3. **Column access by name** - Python's `get_column` accepts both int and str indices
4. **Default column name generation** - More sophisticated in Python

### Implementation Differences:
- Python has better integration with pandas DataFrames
- Python's transform method supports both arrays and callables
- Column naming is more flexible in Python

## EventBasis (Basis Function Events)

### Missing Methods/Properties in Rust:
1. **`to_matrix` method** - Convert to EventMatrix representation
2. **`change_basis` method** - Create new EventBasis with different basis functions
3. **`from_dataframe` class method** - Creates EventBasis from DataFrame
4. **`expanded_values` property** - Stores the expanded values after basis application
5. **Basis validation** - Python validates that basis has 'evaluate' method

### Implementation Differences:
- Python stores both original values and expanded values
- Python has better basis function integration
- More flexible basis function handling in Python

## EventTerm (Term Combinations)

### Missing Methods/Properties in Rust:
1. **`cells` method** - Extract cells (factor combinations) from the term
2. **`conditions` method** - Extract condition names from the term
3. **`split_by_level` method** - For splitting categorical terms
4. **Cached design matrix** - Python uses CacheMixin for performance
5. **Mixed interaction handling** - More sophisticated categorical × continuous interactions
6. **`create_interaction` helper function** - Convenience function for creating interactions

### Implementation Differences:
- Python has more sophisticated interaction computation methods:
  - `_categorical_interaction`
  - `_continuous_interaction`
  - `_mixed_interaction`
- Python caches computed design matrices
- Python has better support for mixed-type interactions

## EventModel

### Missing Methods/Properties in Rust:
1. **`cells` method** - Extract cells from all terms
2. **`conditions` method** - Extract condition names from all terms
3. **`shortnames` method** - Generate shortened column names
4. **`longnames` method** - Alias for column_names
5. **`get_event_onsets` method** - Get onset times for specific event
6. **`get_regressor` method** - Get specific column by name or index
7. **`to_dataframe` method** - Convert design matrix to DataFrame
8. **`summary` method** - Get model summary string
9. **`column_indices` property** - Maps term names to column indices
10. **HRF convolution** - `_convolve_hrf` method with basis function support
11. **Covariate term handling** - Special handling for CovariateTerm
12. **Trialwise term handling** - Special handling for trialwise terms

### Implementation Differences:
- Python has much more sophisticated column naming with `make_term_tag`, `make_cond_tag`, etc.
- Python supports caching of design matrix
- Python has better integration with formula parsing
- Python supports both string and HRFProtocol for HRF specification

## General Missing Features

### 1. Base Classes
- **BaseEvent** - Abstract base class with validation lifecycle (`__post_init__` pattern)
- **CacheMixin** - For caching expensive computations

### 2. Helper Functions
- **`create_event`** - Factory function for creating events by type string
- **`events_from_dataframe`** - Create multiple events from DataFrame with specifications
- **`create_interaction`** - Helper for creating interaction terms
- **`event_table`** - Create event table from various sources

### 3. Validation
- Python has more comprehensive validation with custom error messages
- Python validates at multiple lifecycle points
- Python uses `validate_onsets` and `validate_durations` consistently

### 4. Integration
- Better pandas DataFrame integration throughout
- More sophisticated column naming utilities
- Better formula/DSL integration

### 5. Event Type Mapping
- Python has `EVENT_TYPES` dict mapping strings to classes
- Supports aliases ('factor'/'categorical', 'variable'/'continuous')

## Critical Missing Functionality

1. **HRF Convolution in EventModel** - The Rust implementation doesn't properly handle HRF convolution when building design matrices
2. **Mixed-type Interactions** - Limited support for categorical × continuous interactions
3. **Basis Function Expansion** - EventBasis is incomplete in Rust
4. **Column Naming System** - The sophisticated naming system from Python is not fully implemented
5. **Caching** - No caching mechanism for expensive computations
6. **DataFrame Integration** - Limited pandas-like DataFrame support

## Recommendations

1. **High Priority** - Core functionality:
   - HRF convolution in EventModel (`_convolve_hrf` method)
   - Mixed-type interactions in EventTerm (categorical × continuous)
   - `from_dataframe` class methods for all event types
   - Column naming system (make_term_tag, make_cond_tag, level_token, continuous_token)
   - `cells` and `conditions` methods for EventTerm and EventModel

2. **Medium Priority** - Important features:
   - `bin_values` method in EventVariable
   - `to_matrix` and `change_basis` methods in EventBasis
   - `get_regressor` and `to_dataframe` methods in EventModel
   - Factory functions (`create_event`, `events_from_dataframe`)
   - Caching mechanism for expensive operations

3. **Lower Priority** - Nice to have:
   - `summary` method in EventModel
   - `shortnames` and `longnames` methods
   - Better `__repr__` implementations
   - event_table functionality

4. **Structural Improvements**:
   - Create an EventBase trait with common functionality (n_events property)
   - Add caching support (possibly as a separate trait)
   - Improve validation with better error messages
   - Better DataFrame integration

## Implementation Strategy

1. Start with the column naming system as it's used throughout
2. Implement HRF convolution to make EventModel functional
3. Add DataFrame conversion methods for better usability
4. Implement cells/conditions for feature parity
5. Add remaining methods based on user needs

## Testing Gaps

The Rust implementation should add tests for:
- Mixed-type interactions
- HRF convolution
- Column naming edge cases
- DataFrame round-trip conversion
- Cells and conditions extraction
- Caching behavior