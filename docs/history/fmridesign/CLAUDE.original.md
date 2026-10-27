# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **fmridesign-rs**, a Rust implementation of fMRI experimental design matrices. It's a **direct port of the R fmridesign package** (located at `/Users/bbuchsbaum/code/fmridesign`), bypassing the Python intermediate version. The project aims to achieve complete feature parity with the R implementation while leveraging Rust's performance, safety, and type system benefits.

**Strategic Shift**: As of January 2025, the project has pivoted from Python→Rust to **R→Rust direct conversion**. The R implementation is more mature and feature-complete than the Python version, making it the authoritative reference.

## Development Status

**Current Status**: ~45% R feature parity (reassessed after R→Rust analysis)

According to `R_TO_RUST_MIGRATION_REPORT.md`, the implementation plan has been revised:

**Completed** (baseline from Python port):
- ✅ Core types and traits (80%)
- ✅ Basis functions (75%) - Cox-de Boor B-splines complete, missing levels()/columns()
- ✅ Event system (60%) - Basic types complete, needs NSE→DSL conversion
- ✅ Design matrices (50%) - Basic construction, missing 15-field metadata
- ⚠️ HRF integration (40%) - HrfSpec defined, formula parsing needed
- ⚠️ Contrast system (10%) - Stub only, 7 types to implement
- ❌ Baseline system (5%) - Drift/nuisance modeling not implemented

**Next 4 Weeks** (R parity focus):
- **Week 11**: Basis extensions, error types, naming utilities
- **Week 12**: Baseline system, ColumnMap metadata, trialwise
- **Week 13**: Contrast system (5 core types), block-wise convolution, validation
- **Week 14**: Formula interface, all contrasts, golden tests, documentation

## Build and Development Commands

### Rust Project (fmridesign-rs)

```bash
# Build
cargo build                      # Debug build
cargo build --release           # Release build
cargo build --all-features      # Build with all features

# Test
cargo test                      # Run all tests
cargo test --lib               # Library tests only
cargo test --tests             # Integration tests only
cargo test test_name           # Run specific test
cargo test -- --nocapture      # Show println! output

# Lint and Format
cargo fmt                      # Format code
cargo fmt --check             # Check formatting
cargo clippy                  # Lint code
cargo clippy -- -D warnings   # Treat warnings as errors

# Documentation
cargo doc                     # Build documentation
cargo doc --open             # Build and open in browser
cargo doc --no-deps          # Skip dependencies

# Examples
cargo run --example basic_design
cargo run --example <name>

# Benchmarks
cargo bench                   # Run all benchmarks
cargo bench design_matrix     # Run specific benchmark
```

### R Reference Project (fmridesign)

```bash
# Build and check (from R project directory)
R CMD build .                # Build package
R CMD check fmridesign_*.tar.gz  # Check package

# Interactive R session
R
> library(devtools)
> load_all(".")              # Load package for testing
> test()                     # Run all tests
> document()                 # Generate documentation

# Test
Rscript -e "devtools::test()"           # Run all tests
Rscript -e "testthat::test_dir('tests')"  # Run specific directory

# Generate golden test fixtures
Rscript scripts/generate_rust_fixtures.R  # Export to JSON for Rust tests

# Documentation
R CMD Rd2pdf .               # Build PDF documentation
```

## Architecture Overview

### Core Module Structure

1. **Types and Traits** (`src/types.rs`)
   - `EventSpec` trait - Core abstraction for all event types
   - `BasisFunction` trait - For polynomial/spline basis expansions
   - `ExtractDesignMatrix` trait - For design matrix extraction
   - `Process` trait - Replaces Python's single dispatch system

2. **Event System** (`src/events/`)
   - `EventFactor` - Categorical events with levels
   - `EventVariable` - Continuous events with optional centering/scaling
   - `EventMatrix` - Multi-column event data
   - `EventBasis` - Events expanded with basis functions
   - `EventTerm` - Interaction terms between events

3. **Basis Functions** (`src/basis/`)
   - `PolynomialBasis` - Raw and orthogonal polynomial bases
   - `NaturalPolynomialBasis` - Centered polynomial basis
   - `BSplineBasis` - B-spline basis functions
   - `NaturalSplineBasis` - Natural cubic splines

4. **Design System** (`src/design.rs`)
   - `DesignMatrix` - Core matrix with column names and timing
   - `DesignMatrixBuilder` - Flexible construction API
   - Matrix operations (hstack, subset, etc.)

5. **Convolution** (`src/convolve.rs`, `src/convolve_design.rs`)
   - Integration with `fmrihrf-rs` for HRF models
   - Event-to-regressor convolution

### Key Design Patterns

1. **Trait-Based Polymorphism**: Replaces Python's protocol types and single dispatch
2. **Builder Pattern**: For complex object construction (e.g., DesignMatrixBuilder)
3. **Result-Based Error Handling**: All fallible operations return `Result<T, FmriError>`
4. **Zero-Copy Operations**: Where possible, views instead of clones
5. **Feature Flags**: Optional dependencies (visualization, dataframes)

## Important Implementation Notes

### R to Rust Mapping

| R Concept | Rust Implementation | Notes |
|-----------|---------------------|-------|
| S3 dispatch | Trait implementations | Replace `UseMethod()` with trait objects or enums |
| S4 classes | `struct` + `impl` | Slots → fields, validity → TryFrom |
| NSE/quosures | DSL parser or builder pattern | `rlang::eval_tidy()` → explicit parsing |
| Formulas (`~`) | `nom` parser combinators | Parse "onset ~ hrf(...)" syntax |
| Lists (untyped) | Enums | `enum EventValue { Factor{...}, Numeric(...) }` |
| Factors | Enum + Vec<String> | Levels as strings, codes as indices |
| Data frames | `polars::DataFrame` | Optional feature flag |
| `model.matrix()` | Manual implementation | Treatment contrasts, interactions |
| `attr(x, "name")` | Struct fields or `HashMap` | Metadata as explicit fields |
| `NULL` | `Option<T>` | Explicit absence |
| `stop()` | `Result<T, FmriError>` | All fallible ops return Result |

### Key Dependencies

**Core**:
- **fmrihrf-rs**: Local dependency at `../fmrihrf-rs` for HRF functionality
- **ndarray**: Primary array library
- **nalgebra**: Linear algebra (QR decomposition for contrasts/residualization)
- **thiserror**: Error handling (NEW - for FmriError enum)
- **regex**: Pattern matching for column contrasts (NEW)
- **once_cell**: Lazy caching (NEW - for cells/conditions)

**Optional Features**:
- **polars**: DataFrame support (feature: `dataframes`)
- **nom** or **pest**: Formula DSL parsing (feature: `formula-parser`)
- **plotly** or **plotters**: Visualization (feature: `visualization`)
- **rayon**: Parallel processing (feature: `parallel`)

### HRF Functionality Separation

**IMPORTANT**: HRF functionality is split between `fmrihrf-rs` and `fmridesign-rs`:

**fmrihrf-rs** (core HRF library at `../fmrihrf-rs`) handles:
- Core HRF definitions and evaluation (Gamma, SPM canonical, etc.)
- HRF basis functions (FIR, Fourier, B-splines)
- Custom HRF from arrays (via `fmrihrf::empirical_hrf`)
- Basic HRF operations and transformations

**fmridesign-rs** (this library) handles:
- Model-specific HRF usage (HrfSpec enum for model configuration)
- Integration of HRFs with event models
- HRF selection and parameterization for design matrices
- Convolution of events with HRFs in the modeling context

This mirrors the separation in Python where `pyfmrihrf` provides core HRF functionality while `pyfmridesign` handles the modeling aspects. When implementing new features, ensure core HRF evaluation goes in fmrihrf-rs while modeling integration stays here.

### Testing Strategy

1. **Unit Tests**: In each module file
2. **Integration Tests**: In `tests/` directory
3. **Compatibility Tests**: Compare output with Python implementation
4. **Property Tests**: Using `proptest` for invariant checking
5. **Benchmarks**: Using `criterion` in `benches/`

### Common Development Tasks

**When implementing new features**:
1. **Check R implementation first** at `/Users/bbuchsbaum/code/fmridesign/R/`
2. Review the relevant agent report in `R_TO_RUST_MIGRATION_REPORT.md`
3. Follow existing trait patterns (see Architecture Overview)
4. Ensure comprehensive test coverage (unit + integration + golden)
5. Generate golden test fixtures from R (see Testing Strategy)
6. Update progress in migration report

**When fixing issues**:
1. Add a failing test first
2. Implement the fix
3. **Verify against R behavior** (primary reference)
4. Check Python behavior if relevant (secondary reference)
5. Run full test suite: `cargo test && cargo clippy`

**When adding R golden tests**:
1. Write R script to generate test data: `library(fmridesign); ... ; write_json(...)`
2. Save to `tests/golden/test_name.json` with inputs and expected outputs
3. Write Rust test loading JSON and comparing results with tolerance
4. Document test purpose and R function being validated

## Current Focus Areas

Based on R→Rust migration analysis, **4-week sprint to 90% R parity**:

### Week 11: Foundation & Basis Extensions
**Priority: HIGH** - Foundational infrastructure
- [ ] Implement `FmriError` enum with thiserror (all error types from Agent 7 report)
- [ ] Extend `BasisFunction` trait: add `levels()`, `columns()`, `sub_basis()`
- [ ] Implement DCT basis (discrete cosine transform for detrending)
- [ ] Port naming utilities: `sanitize()`, `make_term_tag()`, `basis_suffix()`
- [ ] Add `lag` parameter to `HrfSpec` enum
- [ ] Property tests for utilities

**Deliverables**: Error infrastructure, complete BasisFunction API, naming system

### Week 12: Baseline System & Metadata
**Priority: HIGH** - Core infrastructure
- [ ] Implement baseline system:
  - `DriftBasis` enum (Constant, Polynomial, BSpline, NaturalSpline)
  - `BaselineModel` with drift/block/nuisance terms
  - Block-diagonal matrix construction
- [ ] Implement `TrialwiseSpec` for beta-series models
- [ ] Enhance formula parsing: parse `hrf()` parameters, integrate with events
- [ ] Implement `ColumnMap` struct with 15 metadata fields
- [ ] `design_colmap()` methods for event/baseline models
- [ ] Residualization utilities (QR-based)

**Deliverables**: Complete baseline system, rich metadata, trialwise support

### Week 13: Contrasts & Convolution
**Priority: CRITICAL** - User-facing features
- [ ] Implement contrast infrastructure:
  - `Contrast` enum with 8 variants
  - `ContrastWeights` struct
  - Weight computation trait
- [ ] Implement 5 core contrast types:
  - `PairContrast` (A vs B - most important)
  - `UnitContrast` (baseline test)
  - `ColumnContrast` (regex patterns)
  - `OnewayContrast` (F-test with Helmert)
  - `PolyContrast` (trends with orthogonal polynomials)
- [ ] Block-wise convolution with `fmrihrf-rs`
- [ ] Multi-basis column naming (_b01, _b02, ...)
- [ ] Validation module: `validate_contrasts()`, `check_collinearity()`

**Deliverables**: 5 contrast types working, block-wise processing, validation

### Week 14: Integration & Testing
**Priority: HIGH** - Complete the implementation
- [ ] Implement `event_model()` from formula:
  - Full three-stage pipeline (parse → realize → assemble)
  - Multiple HRF specs in single formula
- [ ] Advanced contrasts:
  - `InteractionContrast` (Kronecker products)
  - `FormulaContrast` (if DSL ready, else defer)
  - Automatic F-contrast generation
- [ ] Golden test suite: 30+ fixtures from R (see Agent reports for categories)
- [ ] Integration tests: full workflows (event + baseline + contrasts)
- [ ] Documentation:
  - Migration guide (R API → Rust API mapping)
  - Examples for all major features
  - Update README with R reference status

**Deliverables**: Complete formula interface, 7 contrast types, test coverage, docs

### Post-Sprint: Polish & Extensions
**Priority: MEDIUM** - Enhancement phase
- [ ] Visualization (feature-gated): heatmaps, correlation matrices, timecourses
- [ ] Advanced formula evaluation: expressions like `log(RT)`, arithmetic contrasts
- [ ] Performance optimization: parallel convolution, SIMD, sparse matrices
- [ ] Extension registry (if needed for ecosystem)

## Cross-Project Workflow

**Primary Reference**: R implementation at `/Users/bbuchsbaum/code/fmridesign`

When working across projects:
1. **Use R implementation as authoritative reference** for behavior and algorithms
2. Consult Python implementation at `/Users/bbuchsbaum/code/pyfmridesign` as secondary reference
3. **Generate golden test fixtures from R**:
   ```r
   library(fmridesign)
   library(jsonlite)

   # Example: Generate event model fixture
   result <- event_model(onset ~ hrf(condition), data, ~run, sframe)
   fixture <- list(
     input = list(events = data, sframe = list(...)),
     output = list(
       design_matrix = as.matrix(design_matrix(result)),
       column_names = colnames(design_matrix(result))
     )
   )
   write_json(fixture, "tests/golden/event_model_basic.json",
              auto_unbox = TRUE, digits = 15)
   ```
4. Port tests from R's `tests/testthat/` to Rust
5. Document API differences in migration notes (R uses 1-based, Rust uses 0-based, etc.)
6. When algorithms differ, prefer R's approach (it's more mature)

**7-Agent Analysis**: See `R_TO_RUST_MIGRATION_REPORT.md` for:
- Agent 1: Event System (3,251 lines analyzed)
- Agent 2: Basis Functions (696 lines)
- Agent 3: Contrast System (2,237 lines)
- Agent 4: HRF Integration (646 lines)
- Agent 5: Design Matrix System (1,307 lines)
- Agent 6: Baseline System (1,098 lines)
- Agent 7: Utilities & Infrastructure (794 lines)

## Performance Considerations

1. **Parallelization**: Use `rayon` feature for parallel operations
2. **SIMD**: Leverage ndarray's SIMD support
3. **Memory**: Prefer views over clones where possible
4. **Caching**: Consider caching expensive computations (e.g., basis evaluations)

## Documentation Standards

1. **Module-level**: Explain purpose and main types
2. **Type-level**: Document invariants and usage
3. **Function-level**: Include examples for public APIs
4. **Test names**: Use descriptive names explaining what's tested

## Git Workflow Notes

The Rust project currently shows all files as untracked (new repository). When committing:
1. Ensure no sensitive data or credentials
2. Follow conventional commit messages (e.g., "feat: implement PairContrast", "fix: handle NA values in basis")
3. **Reference R implementation** as primary source: "Port validate_contrasts from R (validate.R:40-177)"
4. Update progress in `R_TO_RUST_MIGRATION_REPORT.md` roadmap section
5. Link to specific R files/lines when porting: `fmridesign/R/contrast.R:123-145`

## Additional Resources

- **Migration Report**: `R_TO_RUST_MIGRATION_REPORT.md` - Comprehensive 7-agent analysis
- **R Package**: `/Users/bbuchsbaum/code/fmridesign` - Primary reference (10,224 lines)
- **Python Package**: `/Users/bbuchsbaum/code/pyfmridesign` - Secondary reference
- **HRF Library**: `/Users/bbuchsbaum/code/rust/fmrihrf-rs` - Core HRF functionality
- **R2Rust Guide**: `R2Rust_unified_prompt.md` - Translation patterns and best practices