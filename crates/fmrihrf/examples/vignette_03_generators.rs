//! Vignette 3: HRF Generators
//!
//! This example mirrors the R vignette "HRF Generators" from the fmrihrf
//! package. It demonstrates:
//!
//! - Why generators exist (flexible parameterization of basis sets)
//! - Creating B-spline, FIR, tent, fourier, and daguerre bases
//! - The gen_hrf string-based API
//! - Listing available HRFs
//!
//! Run with: cargo run --example vignette_03_generators

use fmrihrf::hrf::*;
use ndarray::Array1;

fn main() {
    println!("=== Vignette 3: HRF Generators ===\n");

    // ------------------------------------------------------------------
    // Why generators?
    // ------------------------------------------------------------------
    println!("## Why Generators?\n");
    println!("  Pre-defined HRFs like spm_canonical() are ready to use.");
    println!("  Generators let you customize nbasis and span at creation time.\n");

    // List available HRFs from the library
    let available = list_predefined_hrfs();
    println!("  Available predefined HRFs ({} total):", available.len());
    for name in &available {
        print!("    {}", name);
        if let Some(hrf) = get_predefined_hrf(name) {
            print!(" (nbasis={})", hrf.nbasis());
        }
        println!();
    }
    println!();

    // ------------------------------------------------------------------
    // Creating a basis with a generator
    // ------------------------------------------------------------------
    println!("## Creating Basis Sets with Generators\n");

    let t = Array1::linspace(0.0, 30.0, 301);

    // B-spline with 8 basis functions
    let bs8 = bspline_generator(8, None, None).build();
    let bs8_vals = bs8.evaluate(&t).unwrap();
    println!(
        "  B-spline(8): nbasis={}, shape={:?}",
        bs8.nbasis(),
        bs8_vals.shape()
    );

    // Tent (linear B-spline) with 6 basis functions
    let tent6 = bspline_hrf(6, Some(1), Some(30.0));
    let tent_vals = tent6.evaluate(&t).unwrap();
    println!(
        "  Tent(6):     nbasis={}, shape={:?}",
        tent6.nbasis(),
        tent_vals.shape()
    );

    // FIR with 10 bins
    let fir10 = fir_generator(10, None).build();
    let fir_vals = fir10.evaluate(&t).unwrap();
    println!(
        "  FIR(10):     nbasis={}, shape={:?}",
        fir10.nbasis(),
        fir_vals.shape()
    );

    // Fourier with 7 basis functions
    let four7 = fourier_generator(7, None).build();
    let four_vals = four7.evaluate(&t).unwrap();
    println!(
        "  Fourier(7):  nbasis={}, shape={:?}",
        four7.nbasis(),
        four_vals.shape()
    );

    // Daguerre (Laguerre polynomials) with 5 basis functions
    let dag5 = daguerre_generator(Some(5), Some(30.0), Some(4.0)).build();
    let dag_vals = dag5.evaluate(&t).unwrap();
    println!(
        "  Daguerre(5): nbasis={}, shape={:?}",
        dag5.nbasis(),
        dag_vals.shape()
    );
    println!();

    // ------------------------------------------------------------------
    // Examining basis function values
    // ------------------------------------------------------------------
    println!("## Examining Basis Function Values\n");

    // Show first few rows of the B-spline evaluation
    println!("  B-spline(8) first 5 time points:");
    println!(
        "  {:>6} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "t", "B1", "B2", "B3", "B4", "B5"
    );
    for i in (0..50).step_by(10) {
        print!("  {:6.1}", t[i]);
        for j in 0..5.min(bs8_vals.ncols()) {
            print!(" {:8.4}", bs8_vals[[i, j]]);
        }
        println!();
    }
    println!();

    // Show FIR — each basis is a boxcar in a time bin
    println!("  FIR(10) — each basis is a boxcar in a consecutive time bin:");
    let bin_width = fir10.span() / fir10.nbasis() as f64;
    println!("  Bin width: {:.1}s, span: {:.0}s", bin_width, fir10.span());
    for col in 0..fir10.nbasis() {
        let active_count = fir_vals.column(col).iter().filter(|&&v| v > 0.5).count();
        println!(
            "    Basis {}: active at {} time points",
            col + 1,
            active_count
        );
    }
    println!();

    // ------------------------------------------------------------------
    // Using gen_hrf string API
    // ------------------------------------------------------------------
    println!("## Using gen_hrf String API\n");

    let specs = [
        "spmg1",
        "gaussian",
        "bspline(N=8,span=30)",
        "fir(N=10,span=20)",
        "daguerre(n=5,span=30)",
    ];

    for spec in &specs {
        match gen_hrf(spec, 0.0, 0.0, false) {
            Ok(hrf) => println!(
                "  gen_hrf(\"{}\"): name={}, nbasis={}",
                spec,
                hrf.name(),
                hrf.nbasis()
            ),
            Err(e) => println!("  gen_hrf(\"{}\"): error: {}", spec, e),
        }
    }
    println!();

    // ------------------------------------------------------------------
    // Generator with decorators
    // ------------------------------------------------------------------
    println!("## Generators with Decorators\n");

    // Create a lagged B-spline basis
    let bs_lagged = bspline_generator(5, None, None).with_lag(2.0).build();
    println!(
        "  B-spline(5) + lag(2s): name={}, nbasis={}",
        bs_lagged.name(),
        bs_lagged.nbasis()
    );

    // Create a blocked and normalized HRF
    let spm_blocked = HrfGenerator::new(spm_canonical(None, None, None, None, None, None))
        .with_block(5.0)
        .normalized()
        .build();
    let sb_vals = spm_blocked.evaluate(&t).unwrap();
    let sb_max = sb_vals
        .column(0)
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  SPM + block(5s) + normalize: peak = {:.4} (should be ~1.0)",
        sb_max
    );

    println!("\n=== Done ===");
}
