# Quick Start: Implementing R Parity

**Goal**: Get started immediately on achieving R package parity
**Time to first result**: ~30 minutes
**Recommended first task**: P0-1 (SPM Canonical Formula Validation)

---

## Prerequisites

Ensure you have:
- [x] Rust toolchain installed (`cargo --version`)
- [x] R installed with `fmrihrf` package (`library(fmrihrf)`)
- [x] Python 3 with pandas and numpy (for validation scripts)
- [x] This repository cloned

---

## Immediate Action: Validate SPM Canonical Formula

The most critical gap identified is a potential formula mismatch in the SPM canonical HRF. Let's validate this first.

### Step 1: Review the Analysis (5 minutes)

```bash
cd /Users/bbuchsbaum/code/rust/fmrihrf-rs
cat SPM_CANONICAL_FORMULA_ANALYSIS.md
```

**Key Finding**: R and Rust use mathematically equivalent formulas, but R uses a computationally simplified form with pre-computed constants. There's a potential 10x amplitude difference (A1=0.0833 vs 0.00833333).

### Step 2: Run Numerical Validation (10 minutes)

```bash
# Generate Rust reference values
cargo run --release --example validate_spm_canonical

# Run full validation comparing R and Rust
./scripts/validate_spm_canonical.sh
```

**Expected Output**:
```
SPM Canonical HRF Formula Validation
====================================

Step 1: Generating R reference values...
R Implementation Results:
-------------------------
Original (A1=0.0833):  Peak at t=5.0, value=X.XXXXXX
Corrected (A1=0.00833): Peak at t=5.0, value=X.XXXXXX
✓ R reference values generated successfully

Step 2: Generating Rust values...
Rust Implementation Results:
----------------------------
Standard formula: Peak at t=5.0, value=X.XXXXXX
✓ Rust values generated successfully

Step 3: Comparing R and Rust values...
Numerical Comparison:
======================================================================
Rust (standard) vs R (A1=0.0833):
  RMSE:           X.XXXXXXe-XX
  Max abs diff:   X.XXXXXXe-XX
  Correlation:    X.XXXXXXXXXX
...
```

### Step 3: Interpret Results (5 minutes)

The validation will show one of three outcomes:

**Outcome A**: Rust matches R original (A1=0.0833)
- **Action**: Rust is correct as-is, close this issue
- **Status**: ✅ No changes needed

**Outcome B**: Rust matches R corrected (A1=0.00833333)
- **Action**: R has 10x wrong amplitude, need to decide if Rust should match it
- **Decision needed**: Match R exactly or use correct formula?

**Outcome C**: Neither matches exactly
- **Action**: Deeper investigation needed
- **Next step**: Compare against SPM12 MATLAB source

### Step 4: Decision Point (5 minutes)

Based on validation results, decide:

#### Option 1: Match R Exactly (Recommended for compatibility)
- Implement R's simplified formula in Rust
- Ensures bit-for-bit compatibility with R
- User command: "Implement R's simplified SPM formula"

#### Option 2: Keep Rust Standard (Recommended for correctness)
- Document the difference
- Add note to README about formula
- User command: "Document SPM formula difference"

#### Option 3: Support Both
- Add `SpmFormulation` enum with Standard/Simplified variants
- Let users choose
- User command: "Add support for both SPM formulations"

---

## Next Steps After Validation

### If Formula Needs Fix: Implement P0-1 (1-2 days)

```bash
# Create feature branch
git checkout -b fix/spm-canonical-formula

# Edit src/hrf/functions.rs
# Add simplified formula matching R

# Run tests
cargo test

# Create cross-validation test
cargo test --test cross_validation_spm

# Commit
git commit -m "fix(hrf): match R SPM canonical formula (P0-1)"
```

### Continue with P0-2: S3 Traits (1-2 weeks)

Once P0-1 is resolved, move to the next critical item:

```bash
# See IMPLEMENTATION_PLAN.md for details
cat IMPLEMENTATION_PLAN.md | grep -A 30 "P0-2: S3 Generic System"
```

---

## All Key Documents

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `README.md` | Project overview | First time |
| `CLAUDE.md` | Development guide | Setting up development |
| `R_GAP_ANALYSIS_2025.md` | Comprehensive gap analysis | Understanding all gaps |
| `SPM_CANONICAL_FORMULA_ANALYSIS.md` | Detailed formula analysis | Before fixing P0-1 |
| `IMPLEMENTATION_PLAN.md` | Full roadmap | Planning work |
| `QUICK_START_IMPLEMENTATION.md` | This file | Getting started NOW |

---

## Development Workflow

### Daily Workflow

```bash
# 1. Check current status
cat IMPLEMENTATION_PLAN.md | grep "Progress:"

# 2. Pick next task (refer to P0 list)
# 3. Create feature branch
git checkout -b feature/p0-X-description

# 4. Implement
# 5. Test locally
cargo test

# 6. Cross-validate against R (if applicable)
./scripts/validate_feature.sh  # Create per-feature

# 7. Commit
git commit -m "feat: implement feature (P0-X)"

# 8. Update progress in IMPLEMENTATION_PLAN.md
```

### Testing Strategy

For each feature implementation:

1. **Unit tests** (Rust internal correctness)
   ```rust
   #[test]
   fn test_feature_basic() { ... }
   ```

2. **Cross-validation tests** (Match R output)
   ```rust
   #[test]
   fn test_feature_matches_r() {
       let r_ref = load_r_fixture("feature_output.csv");
       let rust_output = compute_feature();
       assert_approx_eq!(rust_output, r_ref, 1e-10);
   }
   ```

3. **Performance benchmarks** (Verify speedup)
   ```rust
   #[bench]
   fn bench_feature(b: &mut Bencher) { ... }
   ```

---

## Getting Help

### If You're Stuck

1. **Review the analysis documents** - Most questions are answered there
2. **Check R source code** - `/Users/bbuchsbaum/code/fmrihrf/R/`
3. **Look at existing implementations** - Similar features in Rust codebase
4. **Run validation scripts** - Often reveals the issue

### Common Issues

**Issue**: "R formula doesn't match Rust"
- **Solution**: Check SPM_CANONICAL_FORMULA_ANALYSIS.md for known differences

**Issue**: "Don't know which parameters to use"
- **Solution**: Look at R defaults in R/hrf.R line 710+ (HRF_SPMG1, etc.)

**Issue**: "Not sure how to translate R's S3 method"
- **Solution**: See IMPLEMENTATION_PLAN.md P0-2 for trait design

**Issue**: "Test tolerance too strict/loose"
- **Solution**: Use 1e-10 for exact formulas, 1e-6 for iterative algorithms

---

## Quick Reference: File Locations

### R Package
- Main HRF implementations: `/Users/bbuchsbaum/code/fmrihrf/R/hrf-functions.R`
- HRF class system: `/Users/bbuchsbaum/code/fmrihrf/R/hrf.R`
- S3 generics: `/Users/bbuchsbaum/code/fmrihrf/R/all_generic.R`
- Regressor: `/Users/bbuchsbaum/code/fmrihrf/R/reg-constructor.R`
- Tests: `/Users/bbuchsbaum/code/fmrihrf/tests/testthat/`

### Rust Implementation
- HRF core: `src/hrf/core.rs`
- HRF functions: `src/hrf/functions.rs`
- HRF generators: `src/hrf/generators.rs`
- Regressor: `src/regressor/core.rs`
- Tests: `tests/`
- Examples: `examples/`

---

## Commands You'll Use Frequently

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_spm_canonical

# Run benchmarks
cargo bench

# Build documentation
cargo doc --open

# Run example
cargo run --example validate_spm_canonical

# Format code
cargo fmt

# Lint code
cargo clippy

# Check without building
cargo check
```

---

## Progress Tracking

As you complete tasks, update `IMPLEMENTATION_PLAN.md`:

```markdown
### Phase 1 Progress: 1/6 (17%)

- [x] P0-1: SPM Canonical Formula (3d) ✅ Completed 2025-11-17
- [ ] P0-2: S3 Generic System (7-10d)
- [ ] P0-3: `hrf_from_coefficients()` (3-5d)
- [ ] P0-4: HRF Caching (5-7d)
- [ ] P0-5: `global_onsets()` (2-3d)
- [ ] P0-6: SPMG2 Column Count (2-3d)
```

---

## Time Estimates

Realistic timeline for key milestones:

- **Week 1**: Complete P0-1 (SPM fix), start P0-2 (S3 traits)
- **Week 2-3**: Complete P0-2 (S3 traits)
- **Week 4**: Complete P0-3, P0-5, P0-6
- **Week 5-6**: Complete P0-4 (caching)
- **Week 7-12**: P1 items
- **Week 13-17**: P2 items

**Total to 95% parity**: 12-17 weeks of focused development

---

## Success Criteria

You'll know you're making progress when:

✅ Validation scripts show < 1e-10 difference from R
✅ All tests pass with `cargo test`
✅ Benchmarks show 10-20x speedup vs R
✅ Documentation is complete (`cargo doc`)
✅ Examples run without errors

---

## Ready to Start?

Run the validation NOW:

```bash
cd /Users/bbuchsbaum/code/rust/fmrihrf-rs
cargo run --release --example validate_spm_canonical
./scripts/validate_spm_canonical.sh
```

Then report results and we'll proceed with the fix! 🚀
