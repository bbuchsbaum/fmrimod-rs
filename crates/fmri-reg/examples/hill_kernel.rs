//! Paired hill-climb kernel probe. Inputs/preparation excluded; all f64 checks retained.
use fmri_noise::{ArPlan, ArStart};
use fmri_reg::{GlsModel, LinearModel, MatrixBackend};
use ndarray::Array2;
use std::{hint::black_box, time::Instant};
fn measure(mut f: impl FnMut(), iterations: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_secs_f64() / iterations as f64
}
fn sample(mut f: impl FnMut()) -> Vec<f64> {
    for _ in 0..3 {
        f();
    }
    let first = measure(&mut f, 1);
    let iterations = (0.015 / first).ceil().clamp(1., 10000.) as usize;
    (0..7).map(|_| measure(&mut f, iterations)).collect()
}
fn finite_reduce(x: &[f64]) -> bool {
    x.iter().fold(true, |finite, x| finite & x.is_finite())
}
fn main() {
    let backend = if std::env::var("FMRIREG_BACKEND").as_deref() == Ok("portable") {
        MatrixBackend::Portable
    } else {
        MatrixBackend::default()
    };
    let mut records = Vec::new();
    let order = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "4".into())
        .parse::<usize>()
        .unwrap();
    for (n, p, b) in [
        (300, 8, 128),
        (300, 8, 1000),
        (300, 32, 1000),
        (1200, 8, 1000),
        (1200, 32, 32),
        (1200, 32, 128),
        (1200, 32, 1000),
        (1200, 32, 4096),
    ] {
        let x = Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                1.
            } else {
                ((i + 1) as f64 * (j + 1) as f64 * 0.013).sin()
            }
        });
        let y = Array2::from_shape_fn((n, b), |(i, j)| {
            let z = (i as u64 + 1)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((j as u64 + 1).wrapping_mul(1442695040888963407));
            (z >> 11) as f64 / (1_u64 << 53) as f64
        });
        let phi: Vec<_> = (0..order)
            .map(|k| {
                if k % 2 == 0 {
                    0.4 / (k + 1) as f64
                } else {
                    -0.1 / (k + 1) as f64
                }
            })
            .collect();
        let plan = ArPlan::new(
            vec![n / 2, n / 2],
            vec![phi.clone(), phi],
            ArStart::ZeroHistory,
        )
        .unwrap()
        .with_resets(&[n / 4, 3 * n / 4])
        .unwrap();
        let xw = plan.apply(x.view()).unwrap();
        let mut yw = plan.apply(y.view()).unwrap();
        let model = LinearModel::new(xw.view(), 1e-12)
            .unwrap()
            .with_backend(backend);
        let gls = GlsModel::new(x.view(), plan.clone(), 1e-12)
            .unwrap()
            .with_backend(backend);
        let mut workspace = model.workspace(b).unwrap();
        let mut gls_workspace = gls.workspace(b).unwrap();
        let finite_any = sample(|| {
            black_box(black_box(&y).iter().all(|x| x.is_finite()));
        });
        let finite_fold = sample(|| {
            black_box(finite_reduce(black_box(y.as_slice().unwrap())));
        });
        let whiten = sample(|| {
            plan.apply_into(black_box(y.view()), yw.view_mut()).unwrap();
            black_box(&yw);
        });
        let fit = sample(|| {
            black_box(
                model
                    .fit_into(black_box(yw.view()), &mut workspace)
                    .unwrap(),
            );
        });
        let full = sample(|| {
            black_box(
                gls.fit_into(black_box(y.view()), &mut gls_workspace)
                    .unwrap(),
            );
        });
        let result = gls.fit_into(y.view(), &mut gls_workspace).unwrap();
        records.push(serde_json::json!({"n":n,"p":p,"block":b,"order":order,"backend":format!("{backend:?}"),"finite_any":finite_any,"finite_fold":finite_fold,"whiten":whiten,"fit":fit,"full":full,"beta_sum":result.coefficients().sum(),"variance_sum":result.residual_variance().sum()}));
    }
    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}
