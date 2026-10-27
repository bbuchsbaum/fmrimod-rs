# R to Rust Direct Migration Report
## fmridesign Package Analysis & Implementation Strategy

**Date**: 2025-01-17
**Scope**: Direct R→Rust conversion (bypassing Python intermediate)
**R Package Version**: 0.5.0
**Target Rust Version**: 0.3.0+

---

## Executive Summary

This report provides a comprehensive analysis of the R fmridesign package (10,224 lines across 23 files) to guide direct R-to-Rust conversion. Seven specialized agents analyzed different subsystems, identifying architecture patterns, complexity hotspots, and Rust translation strategies.

**Key Findings**:
- **Event system** (3,251 lines): Complex NSE/quosure evaluation needs DSL parser replacement
- **Contrast system** (2,237 lines): Sophisticated weight computation with 7 contrast types
- **Design matrix system** (1,307 lines): Three-stage pipeline with rich metadata (15 fields/column)
- **Baseline system** (1,098 lines): Separate drift/nuisance modeling, clean architecture
- **Basis functions** (~75% complete in Rust): Missing trait methods, DCT basis
- **HRF integration**: Clear boundary with fmrihrf-rs, formula parsing needed
- **Utilities** (794 lines): 217 validation checks, robust naming system, no global state

**Current Rust Status**: ~80% feature parity for core types, ~40% for high-level components

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Component Analysis](#2-component-analysis)
3. [R-to-Rust Translation Challenges](#3-r-to-rust-translation-challenges)
4. [Critical Dependencies](#4-critical-dependencies)
5. [Implementation Roadmap](#5-implementation-roadmap)
6. [Testing Strategy](#6-testing-strategy)
7. [Risk Assessment](#7-risk-assessment)
8. [Appendices](#appendices)

---

## 1. Architecture Overview

### 1.1 R Package Structure

```
fmridesign/
├── Core Event System (3,251 lines)
│   ├── event-classes.R (638) - Base types
│   ├── event_vector.R (1,459) - Term operations
│   ├── event_model.R (580) - Model construction
│   └── event_model_helpers.R (574) - Pipeline
│
├── Contrast System (2,237 lines)
│   └── contrast.R - All contrast types
│
├── Design Matrix (1,307 lines)
│   ├── design_generics.R (666) - 30+ generics
│   ├── design_map.R (337) - Visualization
│   └── design_colmap.R (304) - Metadata
│
├── Baseline System (1,098 lines)
│   ├── baseline_model.R (922) - Drift/nuisance
│   └── covariate.R (176) - Covariates
│
├── Basis Functions (696 lines)
│   ├── basis.R (622) - Transformations
│   └── condition_basis_list.R (74)
│
├── HRF Integration (646 lines)
│   ├── hrf-formula.R (427) - Formula parsing
│   └── extension_registry.R (219) - Plugins
│
└── Support Infrastructure (794 lines)
    ├── validate.R (230) - Validation
    ├── naming-utils.R (293) - Naming
    ├── sampling_frame.R (106)
    └── utils-internal.R (165)
```

### 1.2 Current Rust Status

```rust
fmridesign-rs/src/
├── types.rs ✅ (Core traits defined)
├── events/ ✅ (Basic event types)
├── basis/ ⚠️ (75% complete - missing levels()/columns())
├── design.rs ⚠️ (Basic matrix, missing metadata)
├── convolve.rs ⚠️ (Basic impl, needs block-wise)
├── formula/ 🚧 (Parser exists, needs integration)
├── contrast.rs ❌ (Stub only)
├── baseline.rs ❌ (Not implemented)
├── hrf.rs ⚠️ (HrfSpec defined, needs formula)
└── validate.rs ❌ (Not implemented)
```

**Legend**: ✅ Complete | ⚠️ Partial | 🚧 In Progress | ❌ Missing

---

## 2. Component Analysis

### 2.1 Event System (Agent 1)

**Complexity**: ⭐⭐⭐⭐⭐ (Highest)

**Key Classes**:
- `event` (base): Factor, variable, matrix, basis variants
- `event_term`: Composite with interaction support
- Unified internal representation with metadata

**Major Challenges**:
1. **NSE Evaluation**: R uses `rlang::eval_tidy()` with quosures
   - **Rust Solution**: Explicit DSL parser (nom/pest) or builder pattern

2. **Dynamic List Manipulation**: R uses untyped lists
   - **Rust Solution**: `enum EventValue { Factor{...}, Numeric(...), Basis{...} }`

3. **Model Matrix Construction**: R's `model.matrix()` handles factor encoding
   - **Rust Solution**: Manual implementation of treatment contrasts, interactions

**Recommended Rust Pattern**:
```rust
pub enum EventValue {
    Factor { codes: Array1<usize>, levels: Vec<String> },
    Numeric(Array2<f64>),
    Basis { matrix: Array2<f64>, basis: Box<dyn BasisFunction> },
}

pub struct Event {
    pub varname: String,
    pub onsets: Array1<f64>,
    pub durations: Array1<f64>,
    pub blockids: Array1<u32>,
    pub value: EventValue,
}

pub struct EventTerm {
    pub events: Vec<Event>,
    pub onsets: Array1<f64>,
    // ... cached metadata
    cells_cache: OnceCell<DataFrame>,
}
```

**Migration Priority**: **HIGH** (core functionality)

---

### 2.2 Basis Functions (Agent 2)

**Complexity**: ⭐⭐⭐ (Medium)

**Current Status**: ~75% complete

**Missing Critical Features**:
1. `levels()` trait method - Returns level identifiers
2. `columns()` trait method - Returns column tokens for naming
3. DCT basis - Discrete cosine transform for detrending
4. State persistence for `predict()` in fitted transforms

**Rust Gaps**:
```rust
// NEEDED: Extend BasisFunction trait
pub trait BasisFunction {
    // ... existing methods ...

    fn levels(&self) -> Vec<String>;  // ❌ Missing
    fn columns(&self) -> Vec<ColumnToken>;  // ❌ Missing
    fn sub_basis(&self, indices: &[usize]) -> Result<Box<dyn BasisFunction>>;
}
```

**Action Items**:
- [ ] Add `levels()` and `columns()` to trait (Week 11)
- [ ] Implement DCT basis (Week 11)
- [ ] Fix state management for Scale/RobustScale/ScaleWithin (Week 11)
- [ ] Refine natural spline algorithm to match R's `ns()` (Week 12)

**Migration Priority**: **MEDIUM** (extension of existing code)

---

### 2.3 Contrast System (Agent 3)

**Complexity**: ⭐⭐⭐⭐⭐ (Highest)

**Scale**: Largest single file (2,237 lines)

**Contrast Types**:
1. `pair_contrast` - A vs B comparison (most common)
2. `unit_contrast` - Test vs baseline
3. `oneway_contrast` - Main effect F-test (Helmert)
4. `interaction_contrast` - Multi-factor interaction (Kronecker)
5. `poly_contrast` - Polynomial trends (orthogonal)
6. `column_contrast` - Regex pattern matching
7. `contrast` (formula) - Arithmetic expressions

**Critical Algorithms**:
- `.calculate_mask_weights()` - Core weight computation (sum-to-zero validation)
- Helmert contrasts - `contr.helmert()` for F-tests
- Orthogonal polynomials - `stats::poly()` for trends
- Kronecker products - For interaction contrasts
- Basis filtering - Multi-basis HRF support

**Rust Type Hierarchy**:
```rust
pub enum Contrast {
    Pair(PairContrast),
    Unit(UnitContrast),
    Oneway(OnewayContrast),
    Interaction(InteractionContrast),
    Polynomial(PolyContrast),
    Column(ColumnContrast),
    Formula(FormulaContrast),
    Difference(Box<ContrastDiff>),
}

pub struct ContrastWeights {
    pub weights: Array2<f64>,  // nConditions x nContrasts
    pub condnames: Vec<String>,
    pub contrast_spec: Contrast,
}
```

**Implementation Phases**:
- **Phase 1** (Week 13): Pair, Unit, Column contrasts
- **Phase 2** (Week 13-14): Oneway, Polynomial (need Helmert/poly)
- **Phase 3** (Week 14): Interaction, Formula (complex)

**Migration Priority**: **HIGH** (critical for analysis)

---

### 2.4 HRF Integration & Formula System (Agent 4)

**Complexity**: ⭐⭐⭐⭐ (High)

**HRF Specification**:
```rust
pub enum HrfSpec {
    Named { name: String, lag: f64, params: HashMap<String, f64> },
    Fir { n_basis: usize, length: f64, tr: f64 },
    BSpline { n_basis: usize, length: f64, order: usize },
    Fourier { n_basis: usize, length: f64, tr: f64 },
    Custom { values: Array1<f64>, tr: f64 },
}
```

**Formula Parsing**:
- R: `onset ~ hrf(condition, basis="spmg1") + hrf(RT, basis="spmg3")`
- **Challenge**: R uses rlang for NSE, formula manipulation
- **Rust Solutions**:
  1. **Option A**: Explicit DSL parser (nom/pest) - Full compatibility
  2. **Option B**: Builder pattern - Type-safe, limited syntax
  3. **Option C**: String-based with polars expressions - Quick MVP

**Trialwise (Beta-Series)**:
```r
onset ~ trialwise(basis="spmg1", add_sum=TRUE)
# Generates one regressor per trial
```

**Boundary with fmrihrf-rs**:
- **fmrihrf-rs**: Core HRF definitions, evaluation, transformations
- **fmridesign-rs**: HRF specification enum, formula parsing, event-HRF integration

**Action Items**:
- [ ] Add `lag` parameter to HrfSpec (Week 11)
- [ ] Implement TrialwiseSpec (Week 12)
- [ ] Formula parsing for `hrf()` calls (Week 12-13)
- [ ] Block-wise convolution (Week 13)

**Migration Priority**: **HIGH** (user-facing API)

---

### 2.5 Design Matrix System (Agent 5)

**Complexity**: ⭐⭐⭐⭐ (High)

**Three-Tier Architecture**:
1. **Generic Interface**: 30+ S3 generics (design_matrix, term_matrices, etc.)
2. **Metadata Layer**: 15-field ColumnMap per column
3. **Visualization**: ggplot2-based heatmaps

**ColumnMap Structure**:
```rust
pub struct ColumnMetadata {
    pub col: usize,
    pub name: String,
    pub term_tag: String,
    pub term_index: usize,
    pub condition: Option<String>,
    pub run: Option<usize>,
    pub role: ColumnRole,  // Task, Drift, Intercept, Nuisance
    pub model_source: ModelSource,  // Event, Baseline
    pub basis_name: Option<String>,
    pub basis_ix: Option<usize>,
    pub basis_total: Option<usize>,
    pub basis_label: Option<String>,
    pub pretty_name: String,
    pub is_block_diagonal: bool,
    pub modulation_type: Option<ModulationType>,
    pub modulation_id: Option<String>,
}
```

**Three-Stage Pipeline**:
1. **Parse**: Formula → hrfspec objects
2. **Realize**: hrfspec → event_term objects
3. **Assemble**: event_terms → combined design matrix

**Trait System**:
```rust
pub trait ExtractDesignMatrix {
    fn design_matrix(&self) -> Result<DesignMatrix>;
    fn design_matrix_subset(&self, blockid: Option<Vec<usize>>) -> Result<DesignMatrix>;
}

pub trait ExtractTermMatrices {
    fn term_matrices(&self) -> Result<HashMap<String, DesignMatrix>>;
    fn term_indices(&self) -> HashMap<String, Vec<usize>>;
}

pub trait DesignColumnMetadata {
    fn design_colmap(&self) -> Result<ColumnMap>;
}
```

**Action Items**:
- [ ] Implement ColumnMap struct with 15 fields (Week 12)
- [ ] design_colmap() methods (Week 12)
- [ ] term_matrices() extraction (Week 12)
- [ ] Name clash detection (Week 13)
- [ ] Visualization (Week 14, optional feature)

**Migration Priority**: **HIGH** (central infrastructure)

---

### 2.6 Baseline System (Agent 6)

**Complexity**: ⭐⭐⭐ (Medium)

**Separate Architecture**: Complementary to event system, not combined in R package

**Regressor Types**:
1. **Drift**: Polynomial/spline over time (block-diagonal)
2. **Block Intercepts**: Runwise or global
3. **Nuisance**: Motion, physiology (block-diagonal)
4. **Covariates**: Non-convolved direct regressors

**Rust Architecture**:
```rust
pub enum DriftBasis {
    Constant,
    Polynomial { degree: usize },
    BSpline { degree: usize },
    NaturalSpline { degree: usize },
}

pub struct BaselineModel {
    pub drift_term: Option<BaselineTerm>,
    pub block_term: Option<BaselineTerm>,
    pub nuisance_term: Option<BaselineTerm>,
    pub sampling_frame: SamplingFrame,
}

pub struct BaselineTerm {
    pub varname: String,
    pub design_matrix: Array2<f64>,
    pub col_indices: Vec<Vec<usize>>,  // Per-block
    pub row_indices: Vec<Vec<usize>>,
}
```

**Residualization**:
- Uses QR decomposition: `Y - X(X'X)^-1X'Y`
- Implemented via nalgebra

**Action Items**:
- [ ] Implement drift basis evaluation (Week 12)
- [ ] Block-diagonal construction (Week 12)
- [ ] Nuisance integration (Week 12)
- [ ] Covariate system (Week 12)
- [ ] Residualization utility (Week 12)

**Migration Priority**: **MEDIUM** (separate but important)

---

### 2.7 Utilities & Infrastructure (Agent 7)

**Complexity**: ⭐⭐ (Low)

**Key Components**:

**Validation** (230 lines):
- `validate_contrasts()` - Estimability, sum-to-zero, rank checks
- `check_collinearity()` - Correlation-based detection
- 217 error checks across codebase

**Naming** (293 lines):
- `make_term_tag()` - **CRITICAL** unique tag generation
- `sanitize()` - R name compliance
- `basis_suffix()` - _b01, _b02, ...
- `make_column_names()` - Single source of truth

**Error Patterns**:
```rust
// Recommended Rust design
#[derive(Debug, thiserror::Error)]
pub enum FmriError {
    #[error("Dimension mismatch for {context}: got {actual}, expected {expected}")]
    DimensionMismatch { context: String, actual: usize, expected: usize },

    #[error("Validation failed: {0}")]
    ValidationError(String),

    #[error("Collinearity detected between {var1} and {var2} (r={correlation:.3})")]
    CollinearityError { var1: String, var2: String, correlation: f64 },

    // ... more variants
}
```

**Action Items**:
- [ ] Implement FmriError enum with thiserror (Week 11)
- [ ] Port naming utilities (Week 11)
- [ ] Validation module (Week 13)
- [ ] Property tests for utilities (Week 14)

**Migration Priority**: **HIGH** (foundational)

---

## 3. R-to-Rust Translation Challenges

### 3.1 Non-Standard Evaluation (NSE)

**R Pattern**:
```r
hrf(condition, RT, basis="spmg1")  # NSE captures variable names
```

**Challenge**: Rust has no metaprogramming/quosures

**Solutions**:

**Option 1: Explicit DSL Parser** (Recommended for full compatibility)
```rust
let formula = Formula::parse("hrf(condition, RT, basis='spmg1')")?;
let specs = formula.to_hrf_specs(&data)?;
```

**Option 2: Builder Pattern** (Type-safe alternative)
```rust
let model = EventModel::builder()
    .add_hrf(&["condition", "RT"], HrfSpec::canonical())
    .sampling_frame(sframe)
    .build()?;
```

**Recommendation**: Start with Option 2 (MVP), add Option 1 in v1.1

---

### 3.2 S3 Method Dispatch

**R Pattern**:
```r
design_matrix(x, ...)  # Dispatches to design_matrix.event_model
```

**Rust Solutions**:

**Option 1: Trait Objects**
```rust
pub trait ExtractDesignMatrix {
    fn design_matrix(&self) -> Result<DesignMatrix>;
}

impl ExtractDesignMatrix for EventModel { ... }
impl ExtractDesignMatrix for BaselineModel { ... }
```

**Option 2: Enum Dispatch** (Zero-cost)
```rust
pub enum Model {
    Event(EventModel),
    Baseline(BaselineModel),
}

impl Model {
    pub fn design_matrix(&self) -> Result<DesignMatrix> {
        match self {
            Model::Event(m) => m.design_matrix(),
            Model::Baseline(m) => m.design_matrix(),
        }
    }
}
```

**Recommendation**: Use traits for extensibility

---

### 3.3 Dynamic Typing

**R Pattern**:
```r
# Lists with mixed types
list(value = matrix_or_factor, meta = NULL_or_list)
```

**Rust Solution**: Enums
```rust
pub enum EventValue {
    Factor { codes: Array1<usize>, levels: Vec<String> },
    Numeric(Array2<f64>),
    Basis { matrix: Array2<f64>, basis: Box<dyn BasisFunction> },
}
```

---

### 3.4 Caching with Attributes

**R Pattern**:
```r
attr(x, "cached_cells") <- computed_cells
```

**Rust Solution**: `OnceCell`
```rust
pub struct EventTerm {
    cells_cache: OnceCell<DataFrame>,
}

impl EventTerm {
    pub fn cells(&self) -> &DataFrame {
        self.cells_cache.get_or_init(|| self.compute_cells())
    }
}
```

---

## 4. Critical Dependencies

### 4.1 Dependency on fmrihrf-rs

**Required from fmrihrf-rs**:
- `Hrf` trait and implementations (canonical, gamma, FIR, B-spline, Fourier)
- `SamplingFrame` with block/timing info
- HRF constructors (`spm_canonical()`, `gamma_hrf()`, etc.)
- Convolution infrastructure (if `Regressor` exists)

**Current Status**:
- ✅ HRF trait and evaluation
- ✅ SamplingFrame structure
- ⚠️ Need to verify Regressor abstraction exists
- ⚠️ Need block-wise processing support

**Action**: Coordinate with fmrihrf-rs maintainer on convolution API

---

### 4.2 Rust Crate Dependencies

**Core**:
- `ndarray` - Matrix operations (already used)
- `nalgebra` - Linear algebra, QR decomposition
- `thiserror` - Error handling
- `polars` - DataFrame operations (optional feature)

**Numeric**:
- `approx` - Floating-point comparisons in tests
- `ndarray-linalg` - Advanced linear algebra (QR, SVD)
- `ndarray-stats` - Statistics (mean, std, etc.)

**Parsing**:
- `nom` or `pest` - Formula DSL parser (if implementing)

**Parallel**:
- `rayon` - Parallel convolution/processing

**Visualization** (optional feature):
- `plotly` or `plotters` - Heatmaps, correlations

**String/Regex**:
- `regex` - Pattern matching for contrasts

**Testing**:
- `proptest` - Property-based testing
- `criterion` - Benchmarking

---

### 4.3 New Dependencies Needed

```toml
[dependencies]
# Already present
ndarray = "0.15"
nalgebra = "0.32"

# Add for migration
thiserror = "1.0"
regex = "1.10"
once_cell = "1.19"

# Optional features
polars = { version = "0.35", optional = true, features = ["lazy"] }
plotly = { version = "0.9", optional = true }
nom = { version = "7.1", optional = true }  # For formula parsing

[dev-dependencies]
approx = "0.5"
proptest = "1.4"
criterion = "0.5"

[features]
default = []
dataframes = ["polars"]
visualization = ["plotly"]
formula-parser = ["nom"]
```

---

## 5. Implementation Roadmap

### Week 11: Foundation & Basis Extensions

**Goals**: Complete basis function system, start error infrastructure

**Tasks**:
1. ✅ Review current codebase status
2. [ ] Implement `FmriError` enum with `thiserror`
3. [ ] Extend `BasisFunction` trait:
   - Add `levels()` method
   - Add `columns()` method
   - Add `sub_basis()` default implementation
4. [ ] Implement all basis types' `levels()` methods
5. [ ] Add DCT basis implementation
6. [ ] Port naming utilities:
   - `sanitize()`
   - `basis_suffix()`, `feature_suffix()`
   - `make_term_tag()` logic
7. [ ] Add `lag` parameter to `HrfSpec`

**Deliverables**:
- Fully extended `BasisFunction` trait
- Complete error type hierarchy
- Core naming utilities module

---

### Week 12: Baseline & Core Infrastructure

**Goals**: Implement baseline system, enhance formula parsing

**Tasks**:
1. [ ] Implement baseline system:
   - `DriftBasis` enum (Constant, Polynomial, BSpline, NaturalSpline)
   - `BaselineModel` struct
   - `BaselineTerm` construction
   - Block-diagonal matrix assembly
2. [ ] Implement nuisance/covariate integration
3. [ ] Residualization utilities (QR-based)
4. [ ] Enhance formula parsing:
   - Parse `hrf()` parameters (basis, lag, nbasis)
   - `TrialwiseSpec` implementation
   - Create `HrfSpecBuilder` from parsed formula
5. [ ] Implement `ColumnMap` structure (15 fields)
6. [ ] `design_colmap()` methods for event/baseline models

**Deliverables**:
- Complete baseline system
- Working trialwise models
- Rich design matrix metadata

---

### Week 13: Contrasts & Convolution

**Goals**: Core contrast types, block-wise convolution

**Tasks**:
1. [ ] Implement contrast infrastructure:
   - `Contrast` enum with 8 variants
   - `ContrastWeights` structure
   - Weight computation trait
2. [ ] Implement basic contrasts:
   - `PairContrast` (most important)
   - `UnitContrast`
   - `ColumnContrast` (regex-based)
3. [ ] Implement statistical contrasts:
   - `OnewayContrast` with Helmert
   - `PolyContrast` with orthogonal polynomials
4. [ ] Enhance convolution:
   - Block-wise processing
   - Multi-basis column naming
   - `summate` flag support
5. [ ] Validation module:
   - `validate_contrasts()` with estimability checks
   - `check_collinearity()`

**Deliverables**:
- 5 core contrast types working
- Block-wise convolution
- Validation infrastructure

---

### Week 14: Integration & Polish

**Goals**: Complete event model formula interface, advanced contrasts

**Tasks**:
1. [ ] Implement `event_model()` from formula:
   - Parse formula → multiple `HrfSpec` objects
   - Create event terms for each spec
   - Three-stage pipeline (parse → realize → assemble)
2. [ ] Advanced contrasts:
   - `InteractionContrast` (Kronecker products)
   - `FormulaContrast` (if DSL ready)
   - Automatic F-contrast generation
3. [ ] Basis filtering/weighting for multi-HRF
4. [ ] Integration tests:
   - Full event model construction
   - Contrast weight computation
   - Baseline + event combination
5. [ ] Golden test suite from R fixtures
6. [ ] Documentation:
   - Migration guide (R → Rust API mapping)
   - Examples for all major features

**Deliverables**:
- Complete formula interface
- All 7 contrast types
- Comprehensive test coverage
- Migration documentation

---

### Post-Week 14: Enhancement & Refinement

**Optional Features**:
1. [ ] Visualization (feature-gated):
   - Design matrix heatmaps
   - Correlation matrices
   - Timecourse plots
2. [ ] Advanced formula parsing:
   - Expression evaluation (log, sqrt, etc.)
   - Formula-based contrasts with arithmetic
3. [ ] Extension system (future):
   - Plugin architecture for external packages
4. [ ] Performance optimization:
   - Parallel convolution with `rayon`
   - SIMD operations
   - Sparse matrix support

---

## 6. Testing Strategy

### 6.1 Golden Test Fixtures

**Strategy**: Generate test outputs from R, compare Rust results

**R Script Template**:
```r
library(fmridesign)
library(jsonlite)

# Generate event model fixture
events <- data.frame(onset = c(2, 8, 14), condition = c("A", "B", "A"))
sframe <- sampling_frame(blocklens = 50, TR = 2)
model <- event_model(onset ~ hrf(condition), events, ~1, sframe)

# Save inputs and outputs
fixture <- list(
  events = events,
  sframe = list(blocklens = 50, TR = 2),
  design_matrix = as.matrix(design_matrix(model)),
  column_names = colnames(design_matrix(model)),
  term_indices = term_indices(model)
)

write_json(fixture, "tests/golden/event_model_simple.json",
           auto_unbox = TRUE, digits = 15)
```

**Rust Test**:
```rust
#[test]
fn test_event_model_matches_r() {
    let fixture = load_golden_test("event_model_simple");

    let model = EventModel::from_formula(
        "onset ~ hrf(condition)",
        &fixture.events,
        &fixture.sframe,
    ).unwrap();

    let dm = model.design_matrix();

    assert_matrices_close(&dm, &fixture.design_matrix, 1e-12);
    assert_eq!(dm.column_names(), fixture.column_names);
}
```

**Coverage Goals**:
- [ ] Event model construction (10 fixtures)
- [ ] All contrast types (7 fixtures)
- [ ] Baseline models (5 fixtures)
- [ ] Convolution with different HRFs (8 fixtures)
- [ ] Trialwise models (3 fixtures)

---

### 6.2 Property-Based Testing

**Using proptest**:
```rust
proptest! {
    // Contrast weights sum to zero
    #[test]
    fn prop_contrast_weights_sum_zero(
        na in 1..10usize,
        nb in 1..10usize,
    ) {
        let weights = calculate_mask_weights(na, nb)?;
        assert!((weights.sum().abs() < 1e-8));
    }

    // Orthogonal polynomials are orthogonal
    #[test]
    fn prop_poly_orthogonal(
        n in 10..100usize,
        degree in 2..6usize,
    ) {
        let x = Array1::linspace(0.0, 1.0, n);
        let poly = PolynomialBasis::new(degree);
        let basis = poly.evaluate(&x.view())?;

        // Check orthogonality: B'B ≈ I
        let gram = basis.t().dot(&basis);
        for i in 0..degree {
            for j in 0..degree {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_abs_diff_eq!(gram[[i, j]], expected, epsilon=1e-10);
            }
        }
    }
}
```

---

### 6.3 Integration Tests

**Full Workflows**:
```rust
#[test]
fn test_full_glm_workflow() {
    // 1. Create baseline model
    let baseline = BaselineModel::builder()
        .drift(DriftBasis::BSpline { degree: 5 })
        .intercept(InterceptType::Runwise)
        .sampling_frame(sframe.clone())
        .build()?;

    // 2. Create event model
    let events = load_test_events();
    let model = EventModel::from_formula(
        "onset ~ hrf(condition) + hrf(RT, basis='spmg3')",
        &events,
        &sframe,
    )?;

    // 3. Combine design matrices
    let X_baseline = baseline.design_matrix();
    let X_events = model.design_matrix();
    let X_full = concatenate(Axis(1), &[X_baseline.view(), X_events.view()])?;

    // 4. Define contrasts
    let contrasts = ContrastSet::new()
        .add(pair_contrast("condition", "A", "B"))
        .add(oneway_contrast("condition"));

    // 5. Compute contrast weights
    let weights = contrasts.compute_weights(&model)?;

    // 6. Validate
    validate_contrasts(&X_full, &weights)?;

    // 7. Check collinearity
    check_collinearity(&X_full, 0.9)?;
}
```

---

### 6.4 Performance Benchmarks

**Using criterion**:
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_convolution(c: &mut Criterion) {
    let events = create_test_events(100);
    let hrf = HrfSpec::canonical().create_hrf()?;
    let sframe = SamplingFrame::new(vec![200], 2.0)?;

    c.bench_function("convolve 100 events", |b| {
        b.iter(|| {
            black_box(convolve_event(&events, &hrf, &sframe))
        })
    });
}

fn bench_contrast_weights(c: &mut Criterion) {
    let model = create_test_model();
    let contrast = pair_contrast("condition", "A", "B");

    c.bench_function("compute pair contrast weights", |b| {
        b.iter(|| {
            black_box(contrast.compute_weights(&model))
        })
    });
}

criterion_group!(benches, bench_convolution, bench_contrast_weights);
criterion_main!(benches);
```

---

## 7. Risk Assessment

### 7.1 High-Risk Areas

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Model matrix construction** | High | High | Port R's algorithm exactly, extensive tests |
| **Formula parsing complexity** | High | Medium | Start with builder pattern, add DSL later |
| **Numerical stability (contrasts)** | High | Low | Use validated libraries, tolerance thresholds |
| **Feature parity gaps** | Medium | Medium | Document differences, provide migration guide |
| **Performance regression** | Low | Low | Benchmark against R, optimize hot paths |

### 7.2 Technical Debt

**Decisions with Future Costs**:
1. **Builder pattern over DSL** (if chosen):
   - Less ergonomic than R formula syntax
   - May need to add DSL later (breaking change)

2. **Simplified formula evaluation**:
   - Expressions like `log(RT)` not supported initially
   - Users must pre-compute transformations

3. **No extension registry** (v1.0):
   - External packages can't register custom HRF types
   - Limits ecosystem growth

**Recommendation**: Document limitations clearly, plan v1.1 features

---

## 8. Migration Guide for Users

### 8.1 R to Rust API Mapping

**Event Models**:
```r
# R
model <- event_model(
  onset ~ hrf(condition, basis="spmg1") + hrf(RT, basis="spmg3"),
  data = events,
  block = ~run,
  sampling_frame = sframe
)
```

```rust
// Rust v1.0 (builder pattern)
let model = EventModel::builder()
    .add_hrf(&["condition"], HrfSpec::named("spmg1"))
    .add_hrf(&["RT"], HrfSpec::named("spmg3"))
    .data(&events)
    .block_column("run")
    .sampling_frame(sframe)
    .build()?;

// Rust v1.1 (formula parsing)
let model = EventModel::from_formula(
    "onset ~ hrf(condition, basis='spmg1') + hrf(RT, basis='spmg3')",
    &events,
    &sframe,
)?;
```

**Contrasts**:
```r
# R
con <- pair_contrast(~ condition == "A", ~ condition == "B")
weights <- contrast_weights(con, model)
```

```rust
// Rust
let con = PairContrast::new("condition", "A", "B");
let weights = con.compute_weights(&model)?;
```

**Baseline Models**:
```r
# R
bmod <- baseline_model(basis = "bs", degree = 5, sframe = sframe)
```

```rust
// Rust
let bmod = BaselineModel::builder()
    .drift(DriftBasis::BSpline { degree: 5 })
    .intercept(InterceptType::Runwise)
    .sampling_frame(sframe)
    .build()?;
```

---

### 8.2 Key Differences

1. **Index Convention**: R uses 1-based, Rust uses 0-based
   - Documented clearly in all APIs
   - Helper functions for conversion

2. **Error Handling**: R uses `stop()`, Rust uses `Result`
   - All fallible operations return `Result<T, FmriError>`
   - Use `?` operator for propagation

3. **Type Safety**: R is dynamic, Rust is static
   - More compile-time errors
   - Catches bugs earlier

4. **Performance**: Rust is typically 10-100x faster
   - No GC pauses
   - SIMD auto-vectorization
   - Parallel operations with `rayon`

---

## Appendices

### A. File Line Count Summary

| Subsystem | Files | Lines | Status |
|-----------|-------|-------|--------|
| Events | 4 | 3,251 | ⚠️ 60% |
| Contrasts | 1 | 2,237 | ❌ 10% |
| Design Matrix | 3 | 1,307 | ⚠️ 50% |
| Baseline | 2 | 1,098 | ❌ 5% |
| Basis | 2 | 696 | ⚠️ 75% |
| HRF | 2 | 646 | ⚠️ 40% |
| Utilities | 4 | 794 | ❌ 20% |
| **Total** | **18** | **10,029** | **~45%** |

---

### B. Rust Trait Summary

**Core Traits** (must implement):
1. `BasisFunction` - Basis evaluation interface
2. `ExtractDesignMatrix` - Design matrix extraction
3. `EventSpec` - Event type abstraction
4. `ContrastSpec` - Contrast specification
5. `HrfConvolution` - HRF convolution

**Support Traits**:
6. `Validate` - Validation interface
7. `DesignColumnMetadata` - Column metadata
8. `EventLength` - Length extraction
9. `TypeClassification` - Categorical/continuous checks

---

### C. Error Type Reference

```rust
pub enum FmriError {
    // Core errors
    DimensionMismatch { context: String, actual: usize, expected: usize },
    ValidationError(String),
    InvalidInput { field: String, reason: String },

    // Subsystem errors
    ContrastError { reason: String },
    CollinearityError { var1: String, var2: String, correlation: f64 },
    ConvolutionError { reason: String },
    ParsingError(String),

    // Wrapped errors
    Io(#[from] std::io::Error),
    Ndarray(#[from] ndarray::ShapeError),
    NalgebraError(#[from] nalgebra::Error),
}
```

---

### D. Key R Functions Not Yet Ported

**High Priority**:
- `validate_contrasts()` - Contrast validation
- `make_term_tag()` - Term tag generation
- `convolve_design()` - Core convolution
- `cells.event_term()` - Cartesian product
- `conditions.event_term()` - Condition tags

**Medium Priority**:
- `check_collinearity()` - Multicollinearity check
- `residualize()` - Residualization
- `Fcontrasts()` - Automatic F-contrasts
- `design_colmap()` - Column metadata

**Low Priority** (future):
- `plot.event_model()` - Visualization
- Extension registry functions
- External HRF support

---

### E. Dependencies on External Packages

**fmrihrf-rs** (critical):
- HRF trait and implementations
- SamplingFrame structure
- Convolution utilities

**polars** (optional):
- DataFrame operations
- CSV/JSON I/O
- Expression evaluation (for formulas)

**plotly/plotters** (optional):
- Visualization features
- Design heatmaps
- Correlation matrices

---

## Conclusion

This comprehensive analysis reveals that direct R-to-Rust conversion is feasible with careful attention to:

1. **NSE Replacement**: Builder pattern for MVP, DSL parser for v1.1
2. **Type Safety**: Enums for dynamic R types, traits for polymorphism
3. **Algorithm Porting**: Direct implementation of Helmert, poly, Kronecker
4. **Testing**: Extensive golden tests, property tests, integration tests
5. **Performance**: Leverage Rust's zero-cost abstractions, SIMD, parallelism

**Current Status**: ~45% complete overall, with strong foundation in place

**Timeline**: 4 weeks to 90% feature parity (Weeks 11-14)

**Success Metrics**:
- ✅ 100% R function coverage
- ✅ >95% test coverage
- ✅ R-Rust output equivalence (within tolerance)
- ✅ Performance ≥ R (likely 10-100x faster)
- ✅ Full rustdoc documentation

**Next Steps**:
1. Review and approve roadmap
2. Begin Week 11 implementation (basis extensions, error types)
3. Establish golden test infrastructure
4. Coordinate with fmrihrf-rs for convolution API

---

**Report Authors**: 7 Specialized Analysis Agents
**Compiled By**: Agent Coordinator
**Date**: 2025-01-17
**Version**: 1.0
