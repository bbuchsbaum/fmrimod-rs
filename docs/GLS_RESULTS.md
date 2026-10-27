# AR-GLS measured results

All times are warm medians over three repetitions. Compute includes whitening, fitting and a t contrast; synthetic generation is excluded. Setup is measured separately.

| T | P | Features | AR | Rust compute ms | R compute ms | Ratio R/Rust | Rust/R RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 300 | 8 | 1000 | 1 | 1.99 | 3.00 | 1.51 | 39.4 / 263.2 |
| 300 | 8 | 1000 | 4 | 1.84 | 4.00 | 2.18 | 43.3 / 154.0 |
| 300 | 8 | 50000 | 1 | 75.27 | 202.00 | 2.68 | 84.6 / 494.4 |
| 300 | 8 | 50000 | 4 | 91.69 | 262.00 | 2.86 | 91.6 / 702.8 |
| 300 | 32 | 1000 | 1 | 2.26 | 5.00 | 2.21 | 34.4 / 141.1 |
| 300 | 32 | 1000 | 4 | 2.59 | 6.00 | 2.32 | 27.0 / 152.4 |
| 300 | 32 | 50000 | 1 | 102.95 | 341.00 | 3.31 | 98.7 / 650.2 |
| 300 | 32 | 50000 | 4 | 125.31 | 375.00 | 2.99 | 89.7 / 663.8 |
| 1200 | 8 | 1000 | 1 | 4.34 | 28.00 | 6.45 | 41.7 / 680.2 |
| 1200 | 8 | 1000 | 4 | 5.82 | 29.00 | 4.98 | 40.3 / 690.4 |
| 1200 | 8 | 50000 | 1 | 214.65 | 955.00 | 4.45 | 53.2 / 757.0 |
| 1200 | 8 | 50000 | 4 | 292.56 | 1137.00 | 3.89 | 53.1 / 718.6 |
| 1200 | 32 | 1000 | 1 | 6.54 | 39.00 | 5.96 | 45.8 / 430.5 |
| 1200 | 32 | 1000 | 4 | 8.12 | 44.00 | 5.42 | 44.7 / 646.6 |
| 1200 | 32 | 50000 | 1 | 337.16 | 2147.00 | 6.37 | 60.7 / 784.3 |
| 1200 | 32 | 50000 | 4 | 382.44 | 2306.00 | 6.03 | 61.3 / 728.9 |

## Block and worker sweep

Stage timers sum time across workers; they are not parallel wall time. Wall time includes generated input and thread launch.

| Block | Workers | Wall median ms | Wall min–max ms | RSS MiB |
|---:|---:|---:|---:|---:|
| 1000 | 1 | 1058.4 | 1057.6–1073.4 | 61.3 |
| 128 | 1 | 1111.4 | 1104.8–1139.3 | 58.1 |
| 128 | 2 | 548.5 | 548.4–572.2 | 48.0 |
| 128 | 4 | 283.1 | 282.4–285.7 | 44.7 |
| 1000 | 2 | 571.4 | 570.3–572.6 | 86.8 |
| 1000 | 4 | 306.6 | 305.2–309.5 | 145.5 |
| 4096 | 1 | 1161.8 | 1148.7–1165.3 | 238.6 |
| 4096 | 2 | 593.0 | 590.4–597.3 | 348.0 |
| 4096 | 4 | 321.0 | 315.0–323.3 | 581.1 |

## Optimization comparison

Same benchmark harness; baseline vs contiguous residual-variance accumulation. Single-worker, 50,000 features, AR(4).

| T | P | Baseline fit ms | Final fit ms | Fit ratio |
|---:|---:|---:|---:|---:|
| 300 | 8 | 43.22 | 36.18 | 1.19 |
| 300 | 32 | 64.49 | 65.73 | 0.98 |
| 1200 | 8 | 137.36 | 103.28 | 1.33 |
| 1200 | 32 | 240.61 | 199.32 | 1.21 |

## Largest-case phases

1,200 timepoints × 32 regressors × 50,000 features, AR(4), one worker, block 1,000. Setup is a single cold observation per process; block phases are warm medians. Noise estimation uses 32 representative features in both implementations.

| Phase | Rust ms | R ms |
|---|---:|---:|
| Setup: correction_prepare | 8.316 | 786.000 |
| Setup: design_whiten | 0.118 | 2.000 |
| Setup: gls_prepare | 1.243 | 0.000 |
| Setup: initial_prepare | 1.107 | 6.000 |
| Setup: noise_fit | 1.571 | 6.000 |
| Setup: residual | 0.250 | 2.000 |
| Warm: generate | 675.798 | 1311.000 |
| Warm: whiten | 172.448 | 450.000 |
| Warm: fit | 199.317 | 1858.000 |
| Warm: contrast | 10.933 | 7.000 |
| Warm: wall | 1058.371 | 3620.000 |

First block pass wall time: Rust 1078.1 ms; R 3650.0 ms. R elapsed timer resolution is about 1 ms; a zero setup entry means below timer resolution.

## Fixed-block memory growth

Same 1,200 × 32 AR(4) model, block 1,000, one worker; fresh process per row. RSS includes setup, warm-up, repetitions and streamed output validation.

| Features | Wall ms | RSS MiB |
|---:|---:|---:|
| 1000 | 22.3 | 47.8 |
| 50000 | 1081.3 | 61.8 |
| 200000 | 4360.1 | 62.2 |
