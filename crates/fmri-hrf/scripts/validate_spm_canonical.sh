#!/bin/bash
# Validation script for SPM canonical HRF formula comparison
# Compares R and Rust implementations numerically

set -e

echo "==================================================================="
echo "SPM Canonical HRF Formula Validation"
echo "==================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Generate R reference values
echo "Step 1: Generating R reference values..."
echo ""

cd /Users/bbuchsbaum/code/fmrihrf

Rscript -e '
library(fmrihrf)

# Generate time points (0 to 30 seconds, 0.1s intervals)
times <- seq(0, 30, by=0.1)

# Generate HRF values with different A1 parameters
hrf_original <- hrf_spmg1(times, P1=5, P2=15, A1=0.0833)
hrf_corrected <- hrf_spmg1(times, P1=5, P2=15, A1=0.00833333)

# Find peaks
peak_idx_orig <- which.max(hrf_original)
peak_time_orig <- times[peak_idx_orig]
peak_val_orig <- hrf_original[peak_idx_orig]

peak_idx_corr <- which.max(hrf_corrected)
peak_time_corr <- times[peak_idx_corr]
peak_val_corr <- hrf_corrected[peak_idx_corr]

cat("R Implementation Results:\n")
cat("-------------------------\n")
cat(sprintf("Original (A1=0.0833):  Peak at t=%.1f, value=%.6f\n",
            peak_time_orig, peak_val_orig))
cat(sprintf("Corrected (A1=0.00833): Peak at t=%.1f, value=%.6f\n",
            peak_time_corr, peak_val_corr))
cat(sprintf("Peak value ratio: %.2f\n", peak_val_orig / peak_val_corr))
cat("\n")

# Save to CSV
df <- data.frame(
  time = times,
  r_original = hrf_original,
  r_corrected = hrf_corrected
)
write.csv(df, "/tmp/spm_canonical_r_reference.csv", row.names=FALSE)

cat("Saved R reference values to: /tmp/spm_canonical_r_reference.csv\n")
'

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ R reference values generated successfully${NC}"
else
    echo -e "${RED}✗ Failed to generate R reference values${NC}"
    exit 1
fi

echo ""
echo "-------------------------------------------------------------------"
echo ""

# Step 2: Generate Rust values
echo "Step 2: Generating Rust values..."
echo ""

cd /Users/bbuchsbaum/code/rust/fmrihrf-rs

# Create temporary Rust test program
cat > /tmp/test_spm_canonical.rs << 'EOF'
use fmri_hrf::hrf::{spm_canonical, HrfExt};
use ndarray::Array1;
use std::fs::File;
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate time points (0 to 30 seconds, 0.1s intervals)
    let times = Array1::linspace(0.0, 30.0, 301);

    // Generate HRF with default parameters (no normalization)
    let hrf = spm_canonical(None, None, None, None, None, Some(false));
    let values = hrf.evaluate(&times)?;
    let values_1d = values.column(0);

    // Find peak
    let (peak_idx, &peak_val) = values_1d.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();
    let peak_time = times[peak_idx];

    println!("Rust Implementation Results:");
    println!("----------------------------");
    println!("Standard formula (a1=6, a2=16, b1=1, b2=1, c=1/6):");
    println!("  Peak at t={:.1}, value={:.6}", peak_time, peak_val);
    println!();

    // Save to CSV
    let mut file = File::create("/tmp/spm_canonical_rust.csv")?;
    writeln!(file, "time,rust_standard")?;
    for (t, v) in times.iter().zip(values_1d.iter()) {
        writeln!(file, "{:.1},{}", t, v)?;
    }

    println!("Saved Rust values to: /tmp/spm_canonical_rust.csv");

    Ok(())
}
EOF

# Compile and run
rustc --edition 2021 \
    -L target/release/deps \
    --extern fmrihrf=target/release/libfmrihrf.rlib \
    --extern ndarray=/tmp/test_spm_canonical.rs \
    -o /tmp/test_spm_canonical \
    /tmp/test_spm_canonical.rs 2>/dev/null

if [ $? -eq 0 ]; then
    /tmp/test_spm_canonical
    echo -e "${GREEN}✓ Rust values generated successfully${NC}"
else
    echo -e "${YELLOW}⚠ Could not compile test program. Running via cargo instead...${NC}"

    # Alternative: use cargo test
    cargo run --release --example validate_spm_canonical 2>/dev/null || {
        echo -e "${YELLOW}⚠ Example not found. Skipping Rust generation.${NC}"
        echo "   You can manually create the example in examples/validate_spm_canonical.rs"
    }
fi

echo ""
echo "-------------------------------------------------------------------"
echo ""

# Step 3: Compare values
echo "Step 3: Comparing R and Rust values..."
echo ""

if [ -f /tmp/spm_canonical_r_reference.csv ] && [ -f /tmp/spm_canonical_rust.csv ]; then
    python3 << 'PYTHON'
import pandas as pd
import numpy as np

# Load data
r_data = pd.read_csv('/tmp/spm_canonical_r_reference.csv')
rust_data = pd.read_csv('/tmp/spm_canonical_rust.csv')

# Merge on time
merged = pd.merge(r_data, rust_data, on='time', how='inner')

# Compare
print("Numerical Comparison:")
print("=" * 70)
print()

# Compare Rust vs R original
diff_orig = merged['rust_standard'] - merged['r_original']
rmse_orig = np.sqrt(np.mean(diff_orig**2))
max_diff_orig = np.max(np.abs(diff_orig))
corr_orig = np.corrcoef(merged['rust_standard'], merged['r_original'])[0, 1]

print("Rust (standard) vs R (A1=0.0833):")
print(f"  RMSE:           {rmse_orig:.6e}")
print(f"  Max abs diff:   {max_diff_orig:.6e}")
print(f"  Correlation:    {corr_orig:.10f}")
print()

# Compare Rust vs R corrected
diff_corr = merged['rust_standard'] - merged['r_corrected']
rmse_corr = np.sqrt(np.mean(diff_corr**2))
max_diff_corr = np.max(np.abs(diff_corr))
corr_corr = np.corrcoef(merged['rust_standard'], merged['r_corrected'])[0, 1]

print("Rust (standard) vs R (A1=0.00833333):")
print(f"  RMSE:           {rmse_corr:.6e}")
print(f"  Max abs diff:   {max_diff_corr:.6e}")
print(f"  Correlation:    {corr_corr:.10f}")
print()

# Scale ratio
r_orig_peak = merged['r_original'].max()
r_corr_peak = merged['r_corrected'].max()
rust_peak = merged['rust_standard'].max()

print("Peak Values:")
print(f"  R (A1=0.0833):     {r_orig_peak:.6f}")
print(f"  R (A1=0.00833):    {r_corr_peak:.6f}")
print(f"  Rust (standard):   {rust_peak:.6f}")
print()
print(f"  Ratio (R_orig / Rust):  {r_orig_peak / rust_peak:.2f}")
print(f"  Ratio (R_corr / Rust):  {r_corr_peak / rust_peak:.2f}")
print()

# Verdict
print("=" * 70)
if rmse_corr < 1e-10:
    print("✓ Rust matches R with corrected A1=0.00833333 (difference < 1e-10)")
    print("  Conclusion: R has 10x wrong amplitude scaling")
elif rmse_orig < 1e-10:
    print("✓ Rust matches R with original A1=0.0833 (difference < 1e-10)")
    print("  Conclusion: R's A1=0.0833 is correct, Rust needs adjustment")
else:
    print("⚠ Neither R formula matches Rust exactly")
    print(f"  Best match: {'corrected A1' if rmse_corr < rmse_orig else 'original A1'}")
    print(f"  RMSE: {min(rmse_corr, rmse_orig):.6e}")

PYTHON
else
    echo -e "${YELLOW}⚠ Could not compare values (missing CSV files)${NC}"
fi

echo ""
echo "==================================================================="
echo "Validation complete!"
echo ""
echo "Files generated:"
echo "  - /tmp/spm_canonical_r_reference.csv (R reference values)"
echo "  - /tmp/spm_canonical_rust.csv (Rust values)"
echo ""
echo "Next steps:"
echo "  1. Review SPM_CANONICAL_FORMULA_ANALYSIS.md for detailed analysis"
echo "  2. Check IMPLEMENTATION_PLAN.md for implementation tasks"
echo "  3. Decide on correct formula (see analysis document)"
echo "==================================================================="
