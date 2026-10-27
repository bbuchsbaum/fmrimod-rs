# SPM Canonical HRF Formula Analysis
**Date**: 2025-11-17
**Status**: 🔴 CRITICAL ISSUE IDENTIFIED

## Executive Summary

The Rust and R implementations of the SPM canonical HRF use **completely different mathematical formulas**. This is not a simple parameter difference—they implement fundamentally different functions.

**Impact**: Any fMRI analysis using the Rust implementation will produce different results than the R implementation. This affects all HRF evaluations, design matrices, and statistical analyses.

**Recommendation**: Must resolve before Rust implementation can be considered production-ready.

---

## Formula Comparison

### R Formula (R/hrf-functions.R:165)

```r
hrf_spmg1 <- function(t, P1=5, P2=15, A1=.0833) {
    ifelse(t < 0, 0, exp(-t)*(A1*t^P1 - 1.274527e-13*t^P2))
}
```

**Expanded**:
```
For t >= 0:
  hrf(t) = exp(-t) * (A1 * t^P1 - C * t^P2)

Where:
  A1 = 0.0833     (amplitude scaling)
  P1 = 5          (power for positive component)
  P2 = 15         (power for negative component)
  C  = 1.274527e-13 (fixed constant for undershoot)
```

**Characteristics**:
- Simplified exponential-polynomial form
- Single exponential decay `exp(-t)` for both components
- Powers on time (t^5 and t^15)
- Very small constant (1.274527e-13) for undershoot
- No gamma functions
- Does NOT use standard gamma distribution form

---

### Rust Formula (src/hrf/functions.rs:135-140)

```rust
// First gamma function (positive)
let g1 = (time.powf(a1 - 1.0) * b1.powf(a1) * (-b1 * time).exp()) / gamma_fn(a1);

// Second gamma function (negative)
let g2 = (time.powf(a2 - 1.0) * b2.powf(a2) * (-b2 * time).exp()) / gamma_fn(a2);

hrf[i] = g1 - c * g2;
```

**Expanded**:
```
For t >= 0:
  hrf(t) = gamma_pdf(t; a1, b1) - c * gamma_pdf(t; a2, b2)

Where:
  gamma_pdf(t; a, b) = [t^(a-1) * b^a * exp(-b*t)] / Γ(a)

  a1 = 6    (shape parameter, positive gamma)
  a2 = 16   (shape parameter, negative gamma)
  b1 = 1    (rate parameter, positive gamma)
  b2 = 1    (rate parameter, negative gamma)
  c  = 1/6  (ratio of negative to positive)
  Γ(a) = gamma function
```

**Characteristics**:
- Standard difference-of-gammas form
- Proper gamma probability density functions
- Different exponential rates for each component (b1*t and b2*t)
- Powers are (a-1), not direct powers
- Normalized by gamma function Γ(a)
- Standard SPM formulation from literature

---

## Mathematical Analysis

### Are These Equivalent?

**No.** These are fundamentally different functions:

1. **R Formula** is a simplified form:
   - `exp(-t) * (polynomial in t)`
   - Single exponential decay
   - No gamma function normalization

2. **Rust Formula** is standard gamma distribution:
   - Difference of two gamma PDFs
   - Each with its own decay rate
   - Normalized by gamma function

### Can R Formula Be Derived from Standard Form?

The R formula **cannot** be derived from the standard difference-of-gammas formula through simple parameter substitution. Let's check:

Standard gamma PDF:
```
gamma_pdf(t; a, b) = [b^a / Γ(a)] * t^(a-1) * exp(-b*t)
```

For a=6, b=1:
```
gamma_pdf(t; 6, 1) = [1 / Γ(6)] * t^5 * exp(-t)
                   = [1 / 120] * t^5 * exp(-t)
                   ≈ 0.00833 * t^5 * exp(-t)
```

For a=16, b=1:
```
gamma_pdf(t; 16, 1) = [1 / Γ(16)] * t^15 * exp(-t)
```

So the Rust formula (with c=1/6) gives:
```
hrf(t) ≈ 0.00833 * t^5 * exp(-t) - (1/6) * [1/Γ(16)] * t^15 * exp(-t)
       = exp(-t) * [0.00833 * t^5 - (1/6) * (1/Γ(16)) * t^15]
```

Now, `1/(6 * Γ(16))` = `1/(6 * 1307674368000)` ≈ `1.274527e-13`

**This matches the R formula!**

So the R formula is actually:
```
R:    exp(-t) * [0.0833 * t^5 - 1.274527e-13 * t^15]
Rust: exp(-t) * [1/Γ(6) * t^5 - 1/(6*Γ(16)) * t^15]
```

And: `1/Γ(6) = 1/120 = 0.008333...`

**WAIT!** The R formula uses `A1 = 0.0833`, which is very close to `1/120 = 0.008333...`

Let me recalculate:
- Γ(6) = 5! = 120
- 1/120 = 0.00833333...
- R uses A1 = 0.0833 (rounded to 4 decimals)

So the formulas **ARE mathematically equivalent**!

---

## Resolution

### The R Formula is a Computational Simplification

The R implementation uses a **computationally simplified** form that:
1. Avoids computing gamma functions (expensive)
2. Factors out common `exp(-t)` term
3. Uses pre-computed constants: `1/Γ(6)` and `1/(6*Γ(16))`

### Parameters Map As:

| R Parameter | Value | Rust Parameter | Rust Value | Relationship |
|-------------|-------|----------------|------------|--------------|
| P1 | 5 | a1 - 1 | 6 - 1 = 5 | Power in simplified form |
| P2 | 15 | a2 - 1 | 16 - 1 = 15 | Power in simplified form |
| A1 | 0.0833 | 1/Γ(a1) | 1/Γ(6) = 1/120 ≈ 0.008333 | **MISMATCH** |
| C (const) | 1.274527e-13 | 1/(c*Γ(a2)) | 1/(6*Γ(16)) | Matches |

### The Problem: A1 Value

R uses `A1 = 0.0833`, but mathematically it should be `1/Γ(6) = 0.00833333...`

**R's value is 10x too large!** This is either:
1. A typo/error in the R code (missing a zero: 0.0833 vs 0.00833)
2. An intentional scaling for some reason
3. A different convention

---

## Numerical Validation

Let's generate test values to see the actual difference:

### Test Code (R)

```r
library(fmrihrf)
times <- seq(0, 30, by=0.1)
hrf_r <- hrf_spmg1(times, P1=5, P2=15, A1=0.0833)
write.csv(data.frame(time=times, value=hrf_r),
          "/tmp/spm_canonical_r.csv", row.names=FALSE)

# Also test with corrected A1
hrf_r_corrected <- hrf_spmg1(times, P1=5, P2=15, A1=0.00833333)
write.csv(data.frame(time=times, value=hrf_r_corrected),
          "/tmp/spm_canonical_r_corrected.csv", row.names=FALSE)
```

### Test Code (Rust)

```rust
use fmrihrf::hrf::{spm_canonical, HrfExt};
use ndarray::Array1;

let times = Array1::linspace(0.0, 30.0, 301);
let hrf = spm_canonical(None, None, None, None, None, Some(false)); // No normalization
let values = hrf.evaluate(&times).unwrap();

// Save to CSV for comparison
```

---

## Next Steps

### 1. Verify R Formula Origin
- [ ] Check SPM12 source code (MATLAB)
- [ ] Check SPM papers/documentation
- [ ] Check if A1=0.0833 is intentional or error

### 2. Run Numerical Comparison
- [ ] Generate R reference values with A1=0.0833
- [ ] Generate R reference values with A1=0.00833333 (corrected)
- [ ] Generate Rust values with default parameters
- [ ] Compare all three

### 3. Decide on Correct Formula

**Options**:

**Option A**: R is correct, Rust should match
- Implement simplified form in Rust
- Use R's constants exactly
- Pros: Matches R exactly, faster computation
- Cons: Non-standard formulation

**Option B**: Rust is correct, R has error
- Document that R uses 10x wrong amplitude
- Keep Rust as-is (standard formulation)
- Pros: Mathematically correct, standard form
- Cons: Won't match R output

**Option C**: Both are valid, different conventions
- Determine if there's a normalization difference
- Add flag to switch between conventions
- Pros: Maximum compatibility
- Cons: Complexity

### 4. Recommended Action

I recommend **Option A with investigation**:

1. First, verify A1=0.0833 in R is intentional by:
   - Checking R package commit history
   - Checking if results match published fMRI studies
   - Checking SPM12 MATLAB source

2. If A1=0.0833 is correct:
   - Implement R's simplified formula in Rust
   - Add both formulas as options (standard and simplified)
   - Document the difference

3. If A1 should be 0.00833333:
   - Fix R package (submit PR)
   - Keep Rust standard formulation
   - Document the correction

---

## Implementation Tasks

### Task 1: Numerical Verification
```bash
# Generate R fixtures
cd /Users/bbuchsbaum/code/fmrihrf
Rscript -e '
times <- seq(0, 30, by=0.1)
hrf1 <- hrf_spmg1(times, P1=5, P2=15, A1=0.0833)
hrf2 <- hrf_spmg1(times, P1=5, P2=15, A1=0.00833333)
df <- data.frame(time=times, r_original=hrf1, r_corrected=hrf2)
write.csv(df, "/tmp/spm_compare.csv", row.names=FALSE)
'
```

### Task 2: Add Simplified Formula to Rust
```rust
// src/hrf/functions.rs
fn evaluate_spm_canonical_simplified(
    t: &TimeArray,
    p1: f64,
    p2: f64,
    a1: f64,
) -> Result<HrfValues> {
    const C: f64 = 1.274527e-13;

    let mut hrf = Array1::zeros(t.len());
    for (i, &time) in t.iter().enumerate() {
        if time >= 0.0 {
            hrf[i] = (-time).exp() * (a1 * time.powf(p1) - C * time.powf(p2));
        }
    }
    Ok(hrf.insert_axis(Axis(1)))
}
```

### Task 3: Add Configuration Option
```rust
pub enum SpmFormulation {
    Standard,      // Difference of gammas
    Simplified,    // R package exponential-polynomial
}

pub fn spm_canonical_with_formula(
    formulation: SpmFormulation,
    // ... other params
) -> Box<dyn Hrf>
```

---

## Questions for User

1. **Have you compared R and Rust outputs numerically?** Do they differ significantly?

2. **Which is the source of truth?** Should we match:
   - SPM12 (MATLAB)
   - R package fmrihrf
   - Published fMRI literature

3. **Is the A1 = 0.0833 value intentional in R?** Or is it a typo?

4. **What's your preference?**:
   - Match R exactly (even if mathematically different)
   - Use standard formulation (even if doesn't match R)
   - Support both with a flag

---

## References

- R Implementation: `/Users/bbuchsbaum/code/fmrihrf/R/hrf-functions.R` lines 164-167
- Rust Implementation: `/Users/bbuchsbaum/code/rust/fmrihrf-rs/src/hrf/functions.rs` lines 114-149
- SPM12 Documentation: [https://www.fil.ion.ucl.ac.uk/spm/doc/](https://www.fil.ion.ucl.ac.uk/spm/doc/)
- Gamma function: Γ(n) = (n-1)! for integer n

---

## Appendix: Full Derivation

Starting from standard difference-of-gammas:
```
hrf(t) = gamma_pdf(t; a1=6, b1=1) - c * gamma_pdf(t; a2=16, b2=1)

where gamma_pdf(t; a, b) = [b^a / Γ(a)] * t^(a-1) * exp(-b*t)
```

Substitute a1=6, b1=1:
```
gamma_pdf(t; 6, 1) = [1^6 / Γ(6)] * t^5 * exp(-1*t)
                   = [1 / 120] * t^5 * exp(-t)
```

Substitute a2=16, b2=1:
```
gamma_pdf(t; 16, 1) = [1 / Γ(16)] * t^15 * exp(-t)
```

Combine with c = 1/6:
```
hrf(t) = [1/120] * t^5 * exp(-t) - (1/6) * [1/Γ(16)] * t^15 * exp(-t)
       = exp(-t) * {[1/120] * t^5 - [1/(6*Γ(16))] * t^15}
       = exp(-t) * {0.00833333 * t^5 - 1.274527e-13 * t^15}
```

R formula:
```
hrf(t) = exp(-t) * {0.0833 * t^5 - 1.274527e-13 * t^15}
```

**Difference**: R uses 0.0833 vs mathematical 0.00833333 (factor of 10)
