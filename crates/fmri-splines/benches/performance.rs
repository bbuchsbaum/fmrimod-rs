use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fmri_splines::BSpline;

fn benchmark_bspline_creation(c: &mut Criterion) {
    let x: Vec<f64> = (0..1000).map(|i| i as f64 / 100.0).collect();

    c.bench_function("bspline_creation", |b| {
        b.iter(|| BSpline::new(black_box(&x), Some(10), None, 3, false, None).unwrap());
    });
}

fn benchmark_bspline_prediction(c: &mut Criterion) {
    let x: Vec<f64> = (0..100).map(|i| i as f64 / 10.0).collect();
    let spline = BSpline::new(&x, Some(10), None, 3, false, None).unwrap();
    let test_x: Vec<f64> = (0..50).map(|i| i as f64 / 5.0).collect();

    c.bench_function("bspline_prediction", |b| {
        b.iter(|| spline.predict(black_box(&test_x)).unwrap());
    });
}

criterion_group!(
    benches,
    benchmark_bspline_creation,
    benchmark_bspline_prediction
);
criterion_main!(benches);
