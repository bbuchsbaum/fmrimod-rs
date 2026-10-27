use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fmridesign::core::EventModel;

fn bench_design_matrix_generation(c: &mut Criterion) {
    c.bench_function("create empty event model", |b| {
        b.iter(|| {
            let model = EventModel::new();
            black_box(model);
        });
    });
}

criterion_group!(benches, bench_design_matrix_generation);
criterion_main!(benches);
