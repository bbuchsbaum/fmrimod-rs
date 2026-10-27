//! Reproducible throughput and direct/FFT agreement; no timing assertions.
use fmri_hrf::{hrf::spm_hrf_with_derivatives, regressor::convolve_hrf};
use ndarray::Array1;
use std::{hint::black_box, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let grid = Array1::from_iter((0..600).map(|i| i as f64 * 1.5 + 0.2));
    let onsets: Vec<_> = (0..240).map(|i| i as f64 * 3.7 + 0.13).collect();
    let amplitudes: Vec<_> = (0..240)
        .map(|i| if i % 3 == 0 { -0.5 } else { 1.0 })
        .collect();
    let hrf = spm_hrf_with_derivatives();
    for blocks in [false, true] {
        let durations: Vec<_> = (0..240)
            .map(|i| {
                if blocks {
                    0.15 + (i % 5) as f64 * 0.47
                } else {
                    0.0
                }
            })
            .collect();
        for precision in [0.1, 0.02] {
            let direct = convolve_hrf(
                &grid,
                &onsets,
                &durations,
                &amplitudes,
                hrf.as_ref(),
                hrf.span(),
                precision,
                "direct",
                true,
            )?;
            let fft = convolve_hrf(
                &grid,
                &onsets,
                &durations,
                &amplitudes,
                hrf.as_ref(),
                hrf.span(),
                precision,
                "fft",
                true,
            )?;
            let error = direct
                .iter()
                .zip(&fft)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(error < 1e-9, "direct/FFT disagree: {error}");
            for method in ["auto", "direct", "fft", "loop"] {
                let iterations = if method == "loop" { 5 } else { 30 };
                let mut timings = Vec::new();
                for _ in 0..iterations {
                    let start = Instant::now();
                    black_box(convolve_hrf(
                        black_box(&grid),
                        &onsets,
                        &durations,
                        &amplitudes,
                        hrf.as_ref(),
                        hrf.span(),
                        precision,
                        method,
                        true,
                    )?);
                    timings.push(start.elapsed().as_secs_f64() * 1e3);
                }
                timings.sort_by(f64::total_cmp);
                println!("blocks={blocks}, precision={precision}, method={method}, scans=600, events=240, basis=3, median_ms={:.6}, direct_fft_max_error={error:.3e}",timings[timings.len()/2]);
            }
        }
    }
    Ok(())
}
