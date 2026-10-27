//! Experimental algebra kernels; no production dispatch changes.
#[path = "../src/backend.rs"]
mod backend;
use backend::MatrixBackend;
use fmri_noise::{Ar1Plan, ArPlan, ArStart, FirstSample};
use fmri_reg::GlsModel;
use nalgebra::DMatrix;
use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Zip};
use std::{hint::black_box, time::Instant};

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_dgemm(
        order: i32,
        ta: i32,
        tb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

fn subtract_product(
    backend: MatrixBackend,
    a: ArrayView2<f64>,
    b: ArrayView2<f64>,
    mut c: ArrayViewMut2<f64>,
) {
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!(c.dim(), (a.nrows(), b.ncols()));
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if backend == MatrixBackend::Accelerate {
        assert!(a.is_standard_layout() && b.is_standard_layout() && c.is_standard_layout());
        let m = i32::try_from(a.nrows()).unwrap();
        let n = i32::try_from(b.ncols()).unwrap();
        let k = i32::try_from(a.ncols()).unwrap();
        // SAFETY: checked contiguous, disjoint views, compatible dimensions and LP64 sizes.
        unsafe {
            cblas_dgemm(
                101,
                111,
                111,
                m,
                n,
                k,
                -1.,
                a.as_ptr(),
                k,
                b.as_ptr(),
                n,
                1.,
                c.as_mut_ptr(),
                n,
            );
        }
        return;
    }
    let _ = backend;
    ndarray::linalg::general_mat_mul(-1., &a, &b, 1., &mut c);
}

fn finite(x: ArrayView2<f64>) -> bool {
    x.as_slice()
        .unwrap()
        .chunks(256)
        .all(|s| s.iter().fold(true, |ok, x| ok & x.is_finite()))
}
fn inverse(x: &Array2<f64>) -> Array2<f64> {
    let (n, p) = x.dim();
    let svd = DMatrix::from_fn(n, p, |i, j| x[(i, j)]).svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut d = Array2::zeros((p, n));
    for k in 0..p {
        if svd.singular_values[k] <= svd.singular_values[0] * 1e-12 {
            continue;
        }
        for i in 0..p {
            for j in 0..n {
                d[(i, j)] += vt[(k, i)] * u[(j, k)] / svd.singular_values[k];
            }
        }
    }
    d
}
// Zero-history AR4, reset at each quarter, matching the benchmark's plan.
fn compose(d: &Array2<f64>, phi: &[f64; 4]) -> Array2<f64> {
    let (p, n) = d.dim();
    let mut a = d.clone();
    for t in 0..n {
        let offset = t % (n / 4);
        for k in 0..4.min(offset) {
            for j in 0..p {
                a[(j, t - k - 1)] -= d[(j, t)] * phi[k];
            }
        }
    }
    a
}
fn energy(residual: &Array2<f64>, phi: &[f64; 4], sum: &mut Array1<f64>) {
    let n = residual.nrows();
    sum.fill(0.);
    for t in 0..n {
        match (t % (n / 4)).min(4) {
            0 => Zip::from(&mut *sum)
                .and(residual.row(t))
                .for_each(|s, &a| *s += a * a),
            1 => Zip::from(&mut *sum)
                .and(residual.row(t))
                .and(residual.row(t - 1))
                .for_each(|s, &a, &b| {
                    let r = a - phi[0] * b;
                    *s += r * r;
                }),
            2 => Zip::from(&mut *sum)
                .and(residual.row(t))
                .and(residual.row(t - 1))
                .and(residual.row(t - 2))
                .for_each(|s, &a, &b, &c| {
                    let r = a - (phi[0] * b + phi[1] * c);
                    *s += r * r;
                }),
            3 => Zip::from(&mut *sum)
                .and(residual.row(t))
                .and(residual.row(t - 1))
                .and(residual.row(t - 2))
                .and(residual.row(t - 3))
                .for_each(|s, &a, &b, &c, &d| {
                    let r = a - (phi[0] * b + phi[1] * c + phi[2] * d);
                    *s += r * r;
                }),
            4 => Zip::from(&mut *sum)
                .and(residual.row(t))
                .and(residual.row(t - 1))
                .and(residual.row(t - 2))
                .and(residual.row(t - 3))
                .and(residual.row(t - 4))
                .for_each(|s, &a, &b, &c, &d, &e| {
                    let r = a - (phi[0] * b + phi[1] * c + phi[2] * d + phi[3] * e);
                    *s += r * r;
                }),
            _ => unreachable!(),
        }
    }
}
fn sample(mut f: impl FnMut()) -> Vec<f64> {
    for _ in 0..3 {
        f();
    }
    let start = Instant::now();
    f();
    let iterations = (0.015 / start.elapsed().as_secs_f64())
        .ceil()
        .clamp(1., 10000.) as usize;
    (0..7)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..iterations {
                f();
            }
            start.elapsed().as_secs_f64() / iterations as f64
        })
        .collect()
}
fn main() {
    let backend = if std::env::var("FMRIREG_BACKEND").as_deref() == Ok("portable") {
        MatrixBackend::Portable
    } else {
        MatrixBackend::default()
    };
    let public_backend = if backend == MatrixBackend::Portable {
        fmri_reg::MatrixBackend::Portable
    } else {
        fmri_reg::MatrixBackend::default()
    };
    if std::env::args().any(|s| s == "--stress-composition") {
        let mut records = Vec::new();
        for exponent in [10, 20, 30, 35, 38, 40] {
            for offset_exp in [0, 20, 32] {
                for phi in [0.4_f64, -0.9, 0.999999] {
                    for stationary in [false, true] {
                        let n = 32;
                        let resets = [0, 1, 8, 16, 19];
                        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
                            let t = if i % 2 == 0 { 1. } else { -1. };
                            let z = if (i / 2) % 2 == 0 { 1. } else { -1. };
                            match j {
                                0 => 1.,
                                1 => t,
                                _ => t + 2.0_f64.powi(-exponent) * z,
                            }
                        });
                        let y = Array2::from_shape_fn((n, 1), |(i, _)| {
                            let t = if i % 2 == 0 { 1. } else { -1. };
                            let z = if (i / 2) % 2 == 0 { 1. } else { -1. };
                            let noise = if (i / 4) % 2 == 0 { 1. } else { -1. };
                            2.0_f64.powi(offset_exp) - 3. * t + 0.5 * z + 0.125 * noise
                        });
                        let plan = Ar1Plan::new(
                            vec![n],
                            vec![phi],
                            if stationary {
                                FirstSample::Stationary
                            } else {
                                FirstSample::ZeroHistory
                            },
                        )
                        .unwrap()
                        .with_resets(&resets)
                        .unwrap();
                        let model = GlsModel::new(x.view(), plan.clone(), 1e-12)
                            .unwrap()
                            .with_backend(public_backend);
                        let expected = model.fit(y.view()).unwrap();
                        let d = inverse(&plan.apply(x.view()).unwrap());
                        // Sparse adjoint construction, not a dense W multiplication.
                        let a = Array2::from_shape_fn((3, n), |(j, i)| {
                            let scale = if stationary && resets.contains(&i) {
                                (1. - phi * phi).sqrt()
                            } else {
                                1.
                            };
                            let mut value = d[(j, i)] * scale;
                            if i + 1 < n && !resets.contains(&(i + 1)) {
                                value -= phi * d[(j, i + 1)];
                            }
                            value
                        });
                        let mut beta = Array2::zeros((3, 1));
                        backend.multiply(a.view(), y.view(), beta.view_mut());
                        let mut residual = Array2::zeros((n, 1));
                        backend.multiply(x.view(), beta.view(), residual.view_mut());
                        Zip::from(&mut residual)
                            .and(&y)
                            .for_each(|r, &v| *r = v - *r);
                        let rw = plan.apply(residual.view()).unwrap();
                        let variance =
                            rw.iter().map(|v| v * v).sum::<f64>() / expected.residual_df() as f64;
                        records.push(serde_json::json!({"delta_exponent":exponent,"offset_exponent":offset_exp,"phi":phi,"stationary":stationary,"rank":n-expected.residual_df(),"backend":format!("{backend:?}"),"baseline_variance":expected.residual_variance()[0],"composed_variance":variance,"baseline_beta":expected.coefficients().iter().copied().collect::<Vec<_>>(),"composed_beta":beta.iter().copied().collect::<Vec<_>>()}));
                    }
                }
            }
        }
        println!("{}", serde_json::to_string_pretty(&records).unwrap());
        return;
    }
    if std::env::args().any(|s| s == "--stress-product") {
        let mut records = Vec::new();
        for p in [2, 8, 32, 64, 128, 256, 512, 1024] {
            for b in [1, 32, 128] {
                let a = Array2::from_elem((4, p), 2.0_f64.powi(50));
                let beta = Array2::from_shape_fn((p, b), |(k, _)| if k < p / 2 { 1. } else { -1. });
                let mut prediction = Array2::zeros((4, b));
                backend.multiply(a.view(), beta.view(), prediction.view_mut());
                let mut residual = Array2::ones((4, b));
                subtract_product(backend, a.view(), beta.view(), residual.view_mut());
                records.push(serde_json::json!({"p":p,"block":b,"backend":format!("{backend:?}"),"separate_max_error":prediction.iter().map(|v| v.abs()).fold(0.,f64::max),"inplace_max_error":residual.iter().map(|v| (v-1.).abs()).fold(0.,f64::max),"expected_residual":1.,"inplace_first_residual":residual[(0,0)]}));
            }
        }
        println!("{}", serde_json::to_string_pretty(&records).unwrap());
        return;
    }
    let mut records = Vec::new();
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
        let phi = [0.4, -0.05, 0.4 / 3., -0.025];
        let plan = ArPlan::new(
            vec![n / 2, n / 2],
            vec![phi.to_vec(), phi.to_vec()],
            ArStart::ZeroHistory,
        )
        .unwrap()
        .with_resets(&[n / 4, 3 * n / 4])
        .unwrap();
        let model = GlsModel::new(x.view(), plan.clone(), 1e-12)
            .unwrap()
            .with_backend(public_backend);
        let xw = plan.apply(x.view()).unwrap();
        let d = inverse(&xw);
        let start = Instant::now();
        let a = compose(&d, &phi);
        let composition_seconds = start.elapsed().as_secs_f64();
        let mut beta = Array2::zeros((p, b));
        let mut residual = Array2::zeros((n, b));
        let mut variance = Array1::zeros(b);
        let mut workspace = model.workspace(b).unwrap();
        let expected = model.fit(y.view()).unwrap();
        let mut timings = [Vec::new(), Vec::new(), Vec::new()];
        let mut errors = [0.0_f64; 3];
        for method in [0, 1, 2, 2, 1, 0] {
            let values = sample(|| {
                match method {
                    0 => {
                        black_box(model.fit_into(black_box(y.view()), &mut workspace).unwrap());
                    }
                    1 => {
                        assert!(finite(y.view()));
                        backend.multiply(a.view(), black_box(y.view()), beta.view_mut());
                        backend.multiply(x.view(), beta.view(), residual.view_mut());
                        Zip::from(&mut residual)
                            .and(&y)
                            .for_each(|r, &v| *r = v - *r);
                        assert!(finite(residual.view()));
                        energy(&residual, &phi, &mut variance);
                        variance /= (n - p) as f64;
                        assert!(beta.iter().chain(variance.iter()).all(|v| v.is_finite()));
                        black_box((&beta, &variance));
                    }
                    2 => {
                        plan.apply_into(black_box(y.view()), residual.view_mut())
                            .unwrap();
                        // Match the production model's second transformed-input validation.
                        assert!(finite(residual.view()));
                        backend.multiply(d.view(), residual.view(), beta.view_mut());
                        subtract_product(backend, xw.view(), beta.view(), residual.view_mut());
                        variance.fill(0.);
                        for row in residual.rows() {
                            Zip::from(&mut variance)
                                .and(row)
                                .for_each(|s, &r| *s += r * r);
                        }
                        variance /= (n - p) as f64;
                        assert!(beta.iter().chain(variance.iter()).all(|v| v.is_finite()));
                        black_box((&beta, &variance));
                    }
                    _ => unreachable!(),
                }
            });
            timings[method].extend(values);
            if method > 0 {
                let err = beta
                    .iter()
                    .chain(variance.iter())
                    .zip(
                        expected
                            .coefficients()
                            .iter()
                            .chain(expected.residual_variance().iter()),
                    )
                    .map(|(a, b)| (a - b).abs() / (1. + b.abs()))
                    .fold(0.0, f64::max);
                assert!(err < 2e-8, "method {method} error {err}");
                errors[method] = err;
            }
        }
        records.push(serde_json::json!({"n":n,"p":p,"block":b,"backend":format!("{backend:?}"),"composition_seconds":composition_seconds,"baseline":timings[0],"composed":timings[1],"inplace":timings[2],"max_scaled_errors":errors,"baseline_workspace_bytes":workspace.working_bytes(),"candidate_numeric_workspace_bytes":8*(n+p+1)*b}));
    }
    println!("{}", serde_json::to_string_pretty(&records).unwrap());
}
