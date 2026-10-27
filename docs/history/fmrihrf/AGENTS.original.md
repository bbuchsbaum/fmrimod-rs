## Agent Guidelines for `fmrihrf-rs`

Scope: This AGENTS file applies to the entire `fmrihrf-rs` repository.

### 1. Source of Truth

- The authoritative reference implementation for this crate is the R package in `~/code/fmrihrf`.
- When implementing or modifying functionality, agents should:
  - Inspect the corresponding R code in `~/code/fmrihrf` to understand the intended API, behavior, and edge cases.
  - Prefer matching the R package’s semantics and numerical behavior over the Python port where they differ.
  - Consult `R_GAP_ANALYSIS_2025.md` and `IMPLEMENTATION_PLAN.md` in this repo before making changes.

### 2. Coding Style and Design

- Write idiomatic Rust, following existing patterns in `src/hrf`, `src/regressor`, and `src/sampling`.
- Preserve the public API shape where possible; when deviating from R for Rust ergonomics, document the rationale in:
  - Inline doc comments, and
  - The relevant planning/analysis document (e.g., `IMPLEMENTATION_PLAN.md`, `R_GAP_ANALYSIS_2025.md`, or `CHANGELOG.md`).
- Prefer small, focused modules and functions; avoid introducing new dependencies unless clearly justified.
- Keep error handling consistent by using the crate’s `Result` and `HrfError` types.

### 3. Testing and Validation

- For any non-trivial behavior change:
  - Add or update Rust tests under `tests/` or the relevant `src/**/mod.rs` test modules.
  - Where feasible, cross-validate numerics against the R package in `~/code/fmrihrf` (e.g., via fixtures generated in R).
- Do not weaken existing tests without an explicit, documented reason.

### 4. Documentation and Examples

- Maintain or improve existing documentation:
  - Public APIs should have `///` doc comments and, when helpful, small usage examples.
  - Update `README.md`, `CHANGELOG.md`, and planning documents when you introduce new public behavior or close items from the implementation plan.
- Keep examples in `examples/` compiling and representative of the recommended usage.

### 5. Scope Discipline

- Focus edits on the feature or bug you are addressing; avoid drive-by refactors across unrelated modules.
- If you discover unrelated issues, you may:
  - Add a short note to the relevant planning doc (e.g., `GAP_ANALYSIS_2025.md`, `IMPLEMENTATION_PLAN.md`), or
  - Leave a brief `TODO` comment with enough context for future work.

