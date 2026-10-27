# Contrast and Formula Implementation Comparison Report

## Executive Summary

This report compares the Python (pyfmridesign) and Rust (fmridesign-rs) implementations of contrast specification and formula parsing functionality. The analysis reveals several missing features in the Rust implementation that need to be addressed for feature parity.

## 1. Contrast Implementation Analysis

### 1.1 Contrast Specification Types

#### Python Implementation
The Python implementation provides the following contrast specification types:
- `ContrastSpec` (base class)
- `ContrastFormulaSpec` - Formula-based contrasts (e.g., ~ A - B)
- `UnitContrastSpec` - Weights sum to 1
- `PairContrastSpec` - Pairwise contrasts (A vs B)
- `ColumnContrastSpec` - Column-based using regex patterns
- `PolyContrastSpec` - Polynomial contrasts for ordered factors
- `OnewayContrastSpec` - One-way ANOVA contrasts
- `InteractionContrastSpec` - Interaction contrasts
- `ContrastSet` - Collection of contrasts

#### Rust Implementation
The Rust implementation has all the same types but with some differences:
- Uses an enum `ContrastSpecEnum` for type dispatch
- Missing the base `ContrastSpec` class hierarchy (uses trait instead)
- Has all the same concrete types

**Status: ✅ Complete** - All contrast specification types are implemented.

### 1.2 Contrast Constructor Functions

#### Python Implementation
Provides convenience functions:
- `contrast()` - General formula-based contrast
- `unit_contrast()` - Unit contrast
- `pair_contrast()` - Pairwise contrast
- `column_contrast()` - Column regex contrast
- `poly_contrast()` - Polynomial contrast
- `oneway_contrast()` - One-way ANOVA
- `interaction_contrast()` - Interaction contrast
- `contrast_set()` - Create contrast set
- `pairwise_contrasts()` - Generate all pairwise contrasts
- `one_against_all_contrast()` - One vs rest contrasts

#### Rust Implementation
Has most constructor functions but missing:
- ❌ `contrast()` - General formula-based contrast constructor
- ❌ `contrast_set()` - Convenience function for creating sets

**Status: ⚠️ Mostly Complete** - Missing 2 constructor functions.

### 1.3 Contrast Weight Computation

#### Python Implementation
- Uses `@singledispatch` for generic `contrast_weights()` function
- Implements weight computation for all contrast types
- Helper functions:
  - `_calculate_mask_weights()` - Core weight calculation
  - `_get_conditions()` - Extract condition names
  - `_get_cells()` - Get categorical combinations
- Returns `Contrast` object with weights and metadata
- Handles basis expansion for interactions
- Formula evaluation (placeholder implementation)

#### Rust Implementation
- Uses pattern matching on `ContrastSpecEnum`
- Implements weight computation for all contrast types
- Similar helper functions with slightly different names
- `TermInfo` trait for abstraction
- Returns `Contrast` struct
- Handles basis expansion
- Formula evaluation (also placeholder)

**Key Differences:**
1. ❌ Python has better error messages and warnings
2. ❌ Python's formula evaluation is more complete (though still placeholder)
3. ❌ Python handles `where` clause evaluation (Rust has TODO)
4. ❌ Python has more sophisticated polynomial contrast computation

**Status: ⚠️ Mostly Complete** - Core functionality exists but missing refinements.

### 1.4 Missing Contrast Features in Rust

1. **Where Clause Evaluation**: Python attempts to evaluate `where` clauses on cells, Rust has TODOs
2. **Formula Parsing in Contrasts**: Both have placeholder implementations, but Python's is more complete
3. **Better Error Messages**: Python provides more informative warnings and errors
4. **Contrast Plotting**: Python has `plot_contrasts.py`, Rust has no visualization for contrasts
5. **F-Contrast Support**: Python has `fcontrast.py`, Rust has basic `fcontrast.rs` but less complete

## 2. Formula Parsing Implementation Analysis

### 2.1 Parser Implementation

#### Python Implementation
- Uses combination of regex and AST parsing
- Supports:
  - LHS ~ RHS syntax
  - Function calls with args and kwargs
  - Addition (+) and interaction (:) operators
  - String/numeric/boolean literals
  - Nested expressions
- Returns `Formula` object with parsed AST
- `FormulaEvaluator` for converting to event specifications

#### Rust Implementation
- Uses `nom` parser combinators (more robust)
- Supports:
  - Same syntax as Python
  - Better error handling
  - Type-safe AST representation
- Direct conversion to `Term` objects
- More functions supported: `hrf`, `trialwise`, `poly`, `bs`, `scale`, `covariate`

**Status: ✅ Complete** - Rust parser is actually more complete than Python.

### 2.2 DSL (Domain-Specific Language) Interface

#### Python Implementation (`dsl.py`)
- `EventVar` class with operators:
  - `*` for interactions
  - `@` for transformations
- `Transform` classes for HRF, Basis, etc.
- `ChainedTransform` for multiple transformations
- Global namespaces: `hrf`, `basis`
- Natural syntax: `condition @ hrf.spm_canonical`

#### Rust Implementation (`dsl.rs`)
- `EventVar` struct with methods (not operators)
- `*` operator for interactions
- `+` operator for formula building
- Method chaining: `.hrf()`, `.basis()`
- `HrfNamespace` for convenient access
- Less operator-based, more method-based

**Key Differences:**
1. ❌ Rust missing `@` operator for transformations
2. ❌ Rust missing `ChainedTransform` equivalent
3. ❌ Rust missing `BasisNamespace`
4. ✅ Rust has `+` operator for formula building (Python doesn't)

**Status: ⚠️ Partially Complete** - Different design philosophy, missing some operators.

### 2.3 Functional Interface

#### Python Implementation (`functional.py`)
- `PipeTerm` class supporting `|` operator
- Transformation functions:
  - `hrf()`, `basis()`, `poly()`, `spline()`, `scale()`, `name()`
- `compose()` for combining transformations
- Elegant syntax: `term('x') | poly(3) | hrf('gamma')`

#### Rust Implementation
- ❌ **No functional interface implemented**
- No pipe operator support
- No compose functionality

**Status: ❌ Missing** - Entire functional interface is missing in Rust.

### 2.4 Formula Features Comparison

#### Supported in Both:
- Basic formula parsing (variables, functions, operators)
- HRF specification
- Polynomial basis
- Trialwise regressors
- Interaction terms
- Multiple terms with +

#### Python Only:
- ❌ None (Rust parser is more complete)

#### Rust Only:
- ✅ B-spline basis (`bs()` function)
- ✅ Scale transformation
- ✅ Covariate function
- ✅ Better error messages from parser

## 3. Missing Tests

### Contrast Tests Missing in Rust:
1. ❌ Comprehensive weight calculation tests
2. ❌ Edge case handling (empty conditions, single level)
3. ❌ Basis expansion in contrasts
4. ❌ Where clause evaluation
5. ❌ ContrastSet operations
6. ❌ Error handling for invalid patterns

### Formula Tests Missing in Rust:
1. ❌ DSL operator tests (only basic tests exist)
2. ❌ Complex formula parsing edge cases
3. ❌ Formula evaluation to event model
4. ❌ Integration with contrast formulas

## 4. Recommendations

### High Priority
1. **Implement Functional Interface**: Add pipe operator support and transformation functions
2. **Complete DSL Implementation**: Add `@` operator and missing namespaces
3. **Where Clause Evaluation**: Implement formula evaluation for subsetting
4. **Improve Error Messages**: Match Python's informative warnings

### Medium Priority
1. **Add Missing Constructor Functions**: `contrast()` and `contrast_set()`
2. **Enhance Formula Evaluation**: Complete the placeholder implementations
3. **Add Comprehensive Tests**: Especially for edge cases and error conditions
4. **Contrast Visualization**: Add plotting capabilities

### Low Priority
1. **Refine Polynomial Contrasts**: Match Python's orthogonalization approach
2. **F-Contrast Enhancement**: Expand F-contrast support
3. **Documentation**: Add more examples and usage guides

## 5. Implementation Plan

### Phase 1: Functional Interface (Priority: High)
```rust
// Create functional.rs module with:
- PipeTerm trait
- Implement pipe operator |
- Transformation functions (hrf, poly, etc.)
- Compose function
```

### Phase 2: DSL Enhancement (Priority: High)
```rust
// Enhance dsl.rs with:
- Implement @ operator for EventVar
- Add BasisNamespace
- Add ChainedTransform
- Improve operator ergonomics
```

### Phase 3: Contrast Completion (Priority: Medium)
```rust
// Add to contrast/spec.rs:
- contrast() constructor function
- contrast_set() convenience function

// Enhance contrast/weights.rs:
- Where clause evaluation
- Better error messages
- Improved polynomial computation
```

### Phase 4: Testing and Documentation (Priority: Medium)
```rust
// Add comprehensive tests for:
- All contrast weight calculations
- Edge cases and error conditions
- DSL and functional interfaces
- Formula parsing edge cases
```

## Conclusion

The Rust implementation has solid foundations for both contrast and formula functionality, but lacks some of the ergonomic features and refinements of the Python version. The most significant gaps are:

1. **No functional interface** (pipe operators)
2. **Incomplete DSL** (missing @ operator and some namespaces)
3. **Missing where clause evaluation**
4. **Less polished error handling**

However, the Rust implementation also has advantages:
- More robust parser using nom
- Better type safety
- More formula functions implemented
- Cleaner separation of concerns

With the recommended improvements, the Rust implementation would achieve full feature parity while maintaining its advantages in performance and type safety.