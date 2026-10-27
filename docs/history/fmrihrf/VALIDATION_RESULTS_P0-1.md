# P0-1 Validation Results: SPM Canonical Formula
**Date**: 2025-11-17
**Status**: ✅ VALIDATED - Fix Required

---

## Validation Summary

Comprehensive numerical validation comparing R package `fmrihrf` and Rust implementation confirms the predicted formula difference.

### Key Findings

**R Package (Original)**:
- Uses: `A1 = 0.0833`
- Peak value: `1.7539456206`
- Formula: `exp(-t) * (0.0833 * t^5 - 1.274527e-13 * t^15)`

**R Package (Corrected)**:
- Uses: `A1 = 0.00833333`
- Peak value: `0.1754410920`
- Formula: `exp(-t) * (0.00833333 * t^5 - 1.274527e-13 * t^15)`

**Rust Implementation**:
- Uses: Standard difference-of-gammas
- Peak value: `0.1754411622`
- Formula: `gamma_pdf(t; 6, 1) - (1/6) * gamma_pdf(t; 16, 1)`

### Numerical Comparison

| Metric | R (original) vs Rust | R (corrected) vs Rust |
|--------|---------------------|----------------------|
| RMSE | 5.75e-01 | **2.55e-08** |
| Max abs diff | 1.58e+00 | **7.02e-08** |
| Correlation | 0.996439 | **1.000000** |
| Peak ratio | 9.997344x | **0.9999996x** |

**Conclusion**: Rust matches R with corrected A1 within numerical precision (RMSE < 1e-7).

---

## Mathematical Analysis

### The Discrepancy

R uses `A1 = 0.0833`, which is **10x larger** than the mathematically correct value:

```
Mathematically correct: A1 = 1/Γ(6) = 1/120 = 0.008333...
R uses:                 A1 = 0.0833
Ratio:                  10.0
```

### Why This Matters

The SPM canonical HRF is typically defined as a **difference of two gamma probability density functions**:

```
hrf(t) = gamma_pdf(t; a1, b1) - c * gamma_pdf(t; a2, b2)

where:
  gamma_pdf(t; a, b) = [b^a / Γ(a)] * t^(a-1) * exp(-b*t)
```

For the canonical parameters (a1=6, a2=16, b1=1, b2=1, c=1/6), this simplifies to:

```
hrf(t) = exp(-t) * [(1/Γ(6)) * t^5 - (1/(6*Γ(16))) * t^15]
       = exp(-t) * [0.00833333 * t^5 - 1.274527e-13 * t^15]
```

R's formula uses `0.0833` instead of `0.00833333`, creating a **10x amplitude scaling**.

---

## Sample Values at Key Time Points

| Time (s) | R (original) | R (corrected) | Rust | Difference |
|----------|--------------|---------------|------|------------|
| 0.0 | 0.00000000 | 0.00000000 | 0.00000000 | 0.000e+00 |
| 1.0 | 0.03064436 | 0.00306566 | 0.00306566 | 1.226e-09 |
| 2.0 | 0.36074973 | 0.03608939 | 0.03608941 | 1.444e-08 |
| 3.0 | 1.00778477 | 0.10081868 | 0.10081872 | 4.033e-08 |
| 4.0 | 1.56230684 | 0.15629088 | 0.15629095 | 6.252e-08 |
| 5.0 | 1.75394562 | 0.17544109 | 0.17544116 | 7.018e-08 |
| 6.0 | 1.60544038 | 0.16047453 | 0.16047460 | 6.422e-08 |
| 8.0 | 0.91416587 | 0.09009930 | 0.09009933 | 3.630e-08 |
| 10.0 | 0.37239507 | 0.03204692 | 0.03204693 | 1.382e-08 |
| 15.0 | 0.00227750 | -0.01513685 | -0.01513686 | -3.092e-09 |
| 20.0 | -0.00805872 | -0.00855318 | -0.00855318 | -1.927e-09 |
| 30.0 | -0.00017094 | -0.00017111 | -0.00017111 | -3.875e-11 |

Rust matches R (corrected) with sub-nanosecond precision at the peak.

---

## Investigation: Is R's Value Intentional?

### Evidence from R Package

**Documentation** (R/hrf-functions.R:153):
```r
#' @param A1 Amplitude scaling factor for the positive gamma function component;
#'           normally fixed at .0833
```

The word "normally" suggests this is a conventional/standard value, not necessarily the mathematically derived one.

### Possible Explanations

**Hypothesis 1: Intentional Scaling**
- SPM software might use this scaling for convention
- The 10x scaling could be for display/visualization purposes
- Peak value ~1.75 is more convenient than ~0.175

**Hypothesis 2: Historical Typo**
- Missing decimal place: `0.0833` instead of `0.00833`
- Propagated through literature and software
- Now the "standard" everyone uses

**Hypothesis 3: Different Normalization**
- R formula might assume different normalization
- Standard gamma PDF vs alternative formulations

### Recommendation: Check SPM12 Source

To definitively resolve this, we should check:
1. SPM12 MATLAB source code
2. Original SPM papers
3. Widely-used fMRI analysis software (FSL, AFNI)

However, for practical purposes:
- **R is the source of truth for this project**
- **Users expect R compatibility**
- **The 10x difference will break all analyses if Rust doesn't match**

---

## Decision: Match R Exactly

### Chosen Approach: Option 1 (Modified)

**Primary**: Match R exactly for compatibility
**Secondary**: Provide mathematically correct version as alternative

### Implementation Plan

1. **Modify Rust `spm_canonical()` to use R's formula**
   - Change to simplified exponential-polynomial form
   - Use `A1 = 0.0833` by default
   - Add parameter to optionally use corrected value

2. **Add `spm_canonical_mathematically_correct()` variant**
   - Uses `A1 = 1/Γ(6) ≈ 0.00833333`
   - For users who want theoretical accuracy

3. **Document the difference clearly**
   - Add extensive comments explaining the 10x difference
   - Document in README.md and API docs
   - Reference this validation report

4. **Add feature flag (optional future work)**
   - `spm_formula = "r_compatible"` (default)
   - `spm_formula = "mathematically_correct"`

### Justification

**For R compatibility:**
- ✅ Users migrating from R expect same results
- ✅ Cross-validation tests will pass
- ✅ Design matrices will match R exactly
- ✅ Statistical analyses will be consistent

**Against mathematical correctness:**
- ❌ 10x different from standard literature
- ❌ May confuse users familiar with SPM theory
- ❌ Perpetuates possible historical error

**Mitigation:**
- Document extensively
- Provide correct version as alternative
- Consider submitting issue to R package maintainer

---

## Implementation Steps

### Step 1: Update `src/hrf/functions.rs`

```rust
/// SPM canonical HRF using R package formula.
///
/// **Important**: This implementation matches the R package `fmrihrf`, which uses
/// `A1 = 0.0833`. This is 10x larger than the mathematically correct value
/// `1/Γ(6) ≈ 0.00833333`. The R formula is:
///
/// ```text
/// hrf(t) = exp(-t) * (0.0833 * t^5 - 1.274527e-13 * t^15)
/// ```
///
/// For the mathematically correct formula, use `spm_canonical_correct()`.
///
/// # R Compatibility
///
/// This function produces identical output to R's `hrf_spmg1()` function
/// within numerical precision (RMSE < 1e-10).
pub fn spm_canonical(...) -> Box<dyn Hrf> {
    // Use R's simplified formula for compatibility
    // A1 = 0.0833 (R package value, 10x larger than theoretical)
    // C = 1.274527e-13 (derived from 1/(6*Γ(16)))
}

/// SPM canonical HRF using mathematically correct formula.
///
/// This uses the theoretically correct value `A1 = 1/Γ(6) ≈ 0.00833333`,
/// which is 10x smaller than the R package's value.
///
/// Use this if you need mathematical accuracy over R compatibility.
pub fn spm_canonical_correct(...) -> Box<dyn Hrf> {
    // Use mathematically correct value
    // A1 = 1/Γ(6) = 1/120 ≈ 0.00833333
}
```

### Step 2: Add Cross-Validation Test

```rust
// tests/cross_validation_spm.rs
#[test]
fn test_spm_canonical_matches_r() {
    let r_ref = load_r_fixture("spm_canonical_r_original.csv");
    let times = Array1::linspace(0.0, 30.0, 301);
    let hrf = spm_canonical(None, None, None, None, None, Some(false));
    let rust_values = hrf.evaluate(&times).unwrap();

    // Should match R within numerical precision
    assert_approx_eq!(rust_values.column(0), r_ref, 1e-10);
}
```

### Step 3: Update Documentation

- README.md: Add note about formula choice
- API docs: Explain R compatibility
- CHANGELOG.md: Document as breaking change if needed

---

## Acceptance Criteria

- [x] Validation completed
- [ ] Rust implementation matches R original (A1=0.0833)
- [ ] RMSE < 1e-10 between Rust and R
- [ ] Cross-validation test passes
- [ ] Documentation updated
- [ ] Alternative correct version provided
- [ ] Breaking change noted if applicable

---

## Related Documents

- `SPM_CANONICAL_FORMULA_ANALYSIS.md` - Detailed mathematical analysis
- `IMPLEMENTATION_PLAN.md` - Overall roadmap (this is P0-1)
- `/tmp/spm_canonical_r_reference.csv` - R reference values
- `/tmp/spm_canonical_rust.csv` - Rust reference values

---

## Next Steps

1. ✅ Validation complete
2. ⏭️ Implement fix in `src/hrf/functions.rs`
3. ⏭️ Add cross-validation test
4. ⏭️ Update documentation
5. ⏭️ Run full test suite
6. ⏭️ Move to P0-2 (S3 trait system)

---

End of Validation Report
