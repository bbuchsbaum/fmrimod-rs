# Validation Scripts

This directory contains scripts for validating the Rust implementation against the R package.

## Available Scripts

### `validate_spm_canonical.sh`

**Purpose**: Validate SPM canonical HRF formula between R and Rust implementations

**What it does**:
1. Generates R reference values with two A1 parameters (0.0833 and 0.00833333)
2. Generates Rust values using standard difference-of-gammas formula
3. Compares numerically and reports RMSE, max difference, and correlation
4. Determines which A1 value matches Rust implementation

**Usage**:
```bash
# From repository root
./scripts/validate_spm_canonical.sh

# Or with full path
/Users/bbuchsbaum/code/rust/fmrihrf-rs/scripts/validate_spm_canonical.sh
```

**Prerequisites**:
- R with `fmrihrf` package installed
- Python 3 with pandas and numpy
- Rust toolchain
- Repository built: `cargo build --release`

**Output Files**:
- `/tmp/spm_canonical_r_reference.csv` - R reference values
- `/tmp/spm_canonical_rust.csv` - Rust values
- Console output with numerical comparison

**Related Documentation**:
- `SPM_CANONICAL_FORMULA_ANALYSIS.md` - Detailed formula analysis
- `IMPLEMENTATION_PLAN.md` - P0-1 task details

---

## Creating New Validation Scripts

Template for feature-specific validation:

```bash
#!/bin/bash
# validate_<feature>.sh
set -e

echo "Validating <feature>..."

# Step 1: Generate R reference
cd /Users/bbuchsbaum/code/fmrihrf
Rscript -e '
  library(fmrihrf)
  # ... R code to generate reference ...
  write.csv(result, "/tmp/<feature>_r.csv", row.names=FALSE)
'

# Step 2: Generate Rust values
cd /Users/bbuchsbaum/code/rust/fmrihrf-rs
cargo run --example validate_<feature>

# Step 3: Compare
python3 <<PYTHON
import pandas as pd
import numpy as np
r_data = pd.read_csv("/tmp/<feature>_r.csv")
rust_data = pd.read_csv("/tmp/<feature>_rust.csv")
# ... comparison logic ...
PYTHON
```

Make executable:
```bash
chmod +x scripts/validate_<feature>.sh
```

---

## Validation Workflow

For each feature requiring R parity:

1. **Create R reference generator**
   - R script in `/Users/bbuchsbaum/code/fmrihrf`
   - Saves to `/tmp/<feature>_r.csv`

2. **Create Rust example**
   - Example in `examples/validate_<feature>.rs`
   - Saves to `/tmp/<feature>_rust.csv`

3. **Create validation script**
   - Script in `scripts/validate_<feature>.sh`
   - Compares R and Rust outputs
   - Reports numerical differences

4. **Add cross-validation test**
   - Test in `tests/cross_validation/<feature>.rs`
   - Loads R fixture
   - Asserts numerical equivalence

---

## Numerical Tolerances

Use these tolerances for cross-validation:

| Algorithm Type | Tolerance | Reason |
|----------------|-----------|--------|
| Exact formulas | 1e-10 | Should match exactly (floating point precision) |
| Iterative algorithms | 1e-6 | May have convergence differences |
| Spline fitting | 1e-6 | Different spline libraries may differ slightly |
| FFT operations | 1e-8 | Numerical precision in FFT |
| Matrix operations | 1e-10 | BLAS should be exact |

---

## Debugging Failed Validations

If validation fails:

1. **Check CSV files**
   ```bash
   head -20 /tmp/<feature>_r.csv
   head -20 /tmp/<feature>_rust.csv
   ```

2. **Plot differences**
   ```python
   import matplotlib.pyplot as plt
   plt.plot(r_data['time'], r_data['value'], label='R')
   plt.plot(rust_data['time'], rust_data['value'], '--', label='Rust')
   plt.legend()
   plt.savefig('/tmp/comparison.png')
   ```

3. **Check parameter values**
   - Ensure R and Rust use same parameters
   - Check for parameter naming differences (rate vs scale, etc.)

4. **Review formulas**
   - Compare mathematical formulas in R and Rust source
   - Check for simplifications or optimizations

---

## Future Scripts

Planned validation scripts:

- [ ] `validate_gamma_hrf.sh` - Gamma HRF variants
- [ ] `validate_gaussian_hrf.sh` - Gaussian HRF
- [ ] `validate_bspline_hrf.sh` - B-spline basis
- [ ] `validate_fourier_hrf.sh` - Fourier basis
- [ ] `validate_fir_hrf.sh` - FIR basis
- [ ] `validate_regressor.sh` - Regressor construction
- [ ] `validate_design_matrix.sh` - Design matrix generation
- [ ] `validate_convolution.sh` - Convolution operations
- [ ] `validate_neural_input.sh` - Neural input generation
- [ ] `validate_penalty_matrix.sh` - Penalty matrices

---

## Dependencies

### R Dependencies
```r
install.packages("fmrihrf")
```

### Python Dependencies
```bash
pip install pandas numpy matplotlib
```

### Rust Dependencies
```bash
cargo build --release
```

---

## Continuous Integration

To run all validations in CI:

```bash
#!/bin/bash
# run_all_validations.sh
set -e

for script in scripts/validate_*.sh; do
    echo "Running $script..."
    $script || exit 1
done

echo "All validations passed!"
```

---

## Notes

- Validation scripts should be fast (< 1 minute each)
- Use `/tmp/` for temporary files
- Scripts should be idempotent (can run multiple times)
- Always clean up temporary files at the end
- Log all steps for debugging
- Use colors for output (green=success, red=failure, yellow=warning)
