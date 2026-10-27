//! Benchmarks for HRF operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fmri_hrf::hrf::{bspline_hrf, fourier_hrf, spm_canonical, HrfExt};
use fmri_hrf::regressor::{design_from_regressor, regressor};
use fmri_hrf::utils::{cached_hrf_eval, clear_hrf_cache, reset_cache_stats};
use ndarray::Array1;

fn benchmark_spm_canonical(c: &mut Criterion) {
    c.bench_function("spm_canonical_evaluation", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 30.0, 301);

        b.iter(|| {
            let _values = hrf.evaluate(black_box(&times)).unwrap();
        });
    });
}

fn benchmark_spm_block(c: &mut Criterion) {
    c.bench_function("spm_canonical_block_3s", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::linspace(0.0, 30.0, 301);

        b.iter(|| {
            let _values = hrf
                .evaluate_block(
                    black_box(&times),
                    black_box(3.0),
                    black_box(0.1),
                    true,
                    false,
                )
                .unwrap();
        });
    });
}

fn benchmark_bspline(c: &mut Criterion) {
    c.bench_function("bspline_evaluation", |b| {
        let hrf = bspline_hrf(6, Some(3), Some(20.0));
        let times = Array1::linspace(0.0, 20.0, 201);

        b.iter(|| {
            let _values = hrf.evaluate(black_box(&times)).unwrap();
        });
    });
}

fn benchmark_fourier(c: &mut Criterion) {
    c.bench_function("fourier_evaluation", |b| {
        let hrf = fourier_hrf(7, Some(20.0));
        let times = Array1::linspace(0.0, 20.0, 201);

        b.iter(|| {
            let _values = hrf.evaluate(black_box(&times)).unwrap();
        });
    });
}

fn benchmark_regressor(c: &mut Criterion) {
    c.bench_function("regressor_evaluation", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(
            vec![10.0, 30.0, 50.0, 70.0, 90.0],
            Some(hrf),
            Some(vec![2.0]),
            None,
            None,
            None,
        )
        .unwrap();
        let grid = Array1::linspace(0.0, 120.0, 1201);

        b.iter(|| {
            let _result = reg.evaluate(black_box(&grid), None, None).unwrap();
        });
    });
}

fn benchmark_design_matrix(c: &mut Criterion) {
    c.bench_function("design_matrix_creation", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg = regressor(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            Some(hrf),
            Some(vec![1.0]),
            None,
            None,
            None,
        )
        .unwrap();
        let grid = Array1::linspace(0.0, 120.0, 1201);

        b.iter(|| {
            let _design = design_from_regressor(black_box(&reg), black_box(&grid), None).unwrap();
        });
    });
}

fn benchmark_cache_cold_vs_warm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_cold_vs_warm");

    // Cold cache - first evaluation
    group.bench_function("cold_cache", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));

        b.iter(|| {
            clear_hrf_cache();
            let _values =
                cached_hrf_eval(black_box(&*hrf), black_box(20.0), black_box(0.1)).unwrap();
        });
    });

    // Warm cache - repeated evaluations
    group.bench_function("warm_cache", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        // Pre-warm the cache
        let _ = cached_hrf_eval(&*hrf, 20.0, 0.1).unwrap();

        b.iter(|| {
            let _values =
                cached_hrf_eval(black_box(&*hrf), black_box(20.0), black_box(0.1)).unwrap();
        });
    });

    // Uncached baseline - direct evaluation
    group.bench_function("uncached_baseline", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::from_vec(
            (0..)
                .map(|i| i as f64 * 0.1)
                .take_while(|&t| t <= 20.0)
                .collect(),
        );

        b.iter(|| {
            let _values = hrf.evaluate(black_box(&times)).unwrap();
        });
    });

    group.finish();
}

fn benchmark_cache_hit_rate(c: &mut Criterion) {
    c.bench_function("cache_hit_rate_90percent", |b| {
        clear_hrf_cache();
        reset_cache_stats();

        // Create 10 different HRFs
        let hrfs: Vec<_> = (0..10)
            .map(|i| {
                spm_canonical(
                    Some(6.0 + i as f64 * 0.1),
                    None,
                    None,
                    None,
                    None,
                    Some(true),
                )
            })
            .collect();

        b.iter(|| {
            // Simulate 90% hit rate by accessing same HRF 9 times out of 10
            for _ in 0..9 {
                let _ =
                    cached_hrf_eval(black_box(&*hrfs[0]), black_box(20.0), black_box(0.1)).unwrap();
            }
            // 1 miss out of 10
            let idx = (std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
                % 10) as usize;
            let _ =
                cached_hrf_eval(black_box(&*hrfs[idx]), black_box(20.0), black_box(0.1)).unwrap();
        });
    });
}

fn benchmark_cache_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_sizes");

    for size in [10, 50, 100, 200].iter() {
        group.bench_with_input(BenchmarkId::new("varied_hrfs", size), size, |b, &size| {
            clear_hrf_cache();

            // Create varied HRFs
            let hrfs: Vec<_> = (0..size)
                .map(|i| {
                    spm_canonical(
                        Some(5.0 + i as f64 * 0.05),
                        None,
                        None,
                        None,
                        None,
                        Some(true),
                    )
                })
                .collect();

            b.iter(|| {
                // Access HRFs in a pattern
                for i in 0..10 {
                    let idx = i % size;
                    let _ =
                        cached_hrf_eval(black_box(&*hrfs[idx]), black_box(20.0), black_box(0.1))
                            .unwrap();
                }
            });
        });
    }

    group.finish();
}

fn benchmark_cache_vs_direct_repeated(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_vs_direct_repeated");

    // Repeated evaluations with caching
    group.bench_function("cached_100_evaluations", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        clear_hrf_cache();

        b.iter(|| {
            for _ in 0..100 {
                let _ = cached_hrf_eval(black_box(&*hrf), black_box(20.0), black_box(0.1)).unwrap();
            }
        });
    });

    // Repeated evaluations without caching
    group.bench_function("direct_100_evaluations", |b| {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let times = Array1::from_vec(
            (0..)
                .map(|i| i as f64 * 0.1)
                .take_while(|&t| t <= 20.0)
                .collect(),
        );

        b.iter(|| {
            for _ in 0..100 {
                let _ = hrf.evaluate(black_box(&times)).unwrap();
            }
        });
    });

    group.finish();
}

fn benchmark_cache_with_different_params(c: &mut Criterion) {
    c.bench_function("cache_different_params", |b| {
        clear_hrf_cache();

        // Create HRFs with different parameters
        let hrf1 = spm_canonical(None, None, None, None, None, Some(true));
        let hrf2 = spm_canonical(Some(7.0), None, None, None, None, Some(true));
        let hrf3 = bspline_hrf(6, Some(3), Some(20.0));

        b.iter(|| {
            // Each should be cached separately
            let _ = cached_hrf_eval(black_box(&*hrf1), black_box(20.0), black_box(0.1)).unwrap();
            let _ = cached_hrf_eval(black_box(&*hrf2), black_box(20.0), black_box(0.1)).unwrap();
            let _ = cached_hrf_eval(black_box(&*hrf3), black_box(20.0), black_box(0.1)).unwrap();

            // Second access should hit cache
            let _ = cached_hrf_eval(black_box(&*hrf1), black_box(20.0), black_box(0.1)).unwrap();
            let _ = cached_hrf_eval(black_box(&*hrf2), black_box(20.0), black_box(0.1)).unwrap();
            let _ = cached_hrf_eval(black_box(&*hrf3), black_box(20.0), black_box(0.1)).unwrap();
        });
    });
}

criterion_group!(
    benches,
    benchmark_spm_canonical,
    benchmark_spm_block,
    benchmark_bspline,
    benchmark_fourier,
    benchmark_regressor,
    benchmark_design_matrix,
    benchmark_cache_cold_vs_warm,
    benchmark_cache_hit_rate,
    benchmark_cache_sizes,
    benchmark_cache_vs_direct_repeated,
    benchmark_cache_with_different_params
);
criterion_main!(benches);
