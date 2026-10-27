//! Full-rank OLS probe using Accelerate's implicit Householder QR.
//!
//! This is deliberately an example, not a library backend: it factors a fixed
//! design once with `dgeqrf`, views row-major responses as a column-major
//! transpose and applies `Q` on the right with `dormqr`, solves for beta with
//! `dtrsm`, and obtains RSS from transformed trailing rows.  The contiguous
//! response copy is part of each timed fit; factor preparation is reported
//! separately.  It does not define rank policy.

#[cfg(all(target_os = "macos", feature = "accelerate"))]
mod probe {
    use fmri_reg::{LinearModel, MatrixBackend};
    use ndarray::{Array2, ArrayView2};
    use std::{hint::black_box, time::Instant};

    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        fn dgeqrf_(
            m: *const i32,
            n: *const i32,
            a: *mut f64,
            lda: *const i32,
            tau: *mut f64,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );
        fn dormqr_(
            side: *const u8,
            trans: *const u8,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            a: *const f64,
            lda: *const i32,
            tau: *const f64,
            c: *mut f64,
            ldc: *const i32,
            work: *mut f64,
            lwork: *const i32,
            info: *mut i32,
        );
        fn dtrsm_(
            side: *const u8,
            uplo: *const u8,
            transa: *const u8,
            diag: *const u8,
            m: *const i32,
            n: *const i32,
            alpha: *const f64,
            a: *const f64,
            lda: *const i32,
            b: *mut f64,
            ldb: *const i32,
        );
        fn dlarft_(
            direct: *const u8,
            storev: *const u8,
            n: *const i32,
            k: *const i32,
            v: *const f64,
            ldv: *const i32,
            tau: *const f64,
            t: *mut f64,
            ldt: *const i32,
        );
        fn dlarfb_(
            side: *const u8,
            trans: *const u8,
            direct: *const u8,
            storev: *const u8,
            m: *const i32,
            n: *const i32,
            k: *const i32,
            v: *const f64,
            ldv: *const i32,
            t: *const f64,
            ldt: *const i32,
            c: *mut f64,
            ldc: *const i32,
            work: *mut f64,
            ldwork: *const i32,
        );
    }

    fn lapack_ok(info: i32, routine: &'static str) {
        assert_eq!(info, 0, "{routine} failed with INFO={info}");
    }

    fn queried_work_len(value: f64, routine: &'static str) -> usize {
        assert!(
            value.is_finite() && value >= 1.0,
            "{routine} invalid workspace query {value}"
        );
        let len = value.ceil();
        assert!(
            len <= i32::MAX as f64,
            "{routine} workspace exceeds LAPACK integer range"
        );
        len as usize
    }

    // Kept identical to the production finite scan so its cost is represented
    // comparably in the candidate's timed input validation.
    fn all_finite(values: ArrayView2<'_, f64>) -> bool {
        match values.as_slice_memory_order() {
            Some(values) => values
                .chunks(256)
                .all(|chunk| chunk.iter().fold(true, |finite, &v| finite & v.is_finite())),
            None => values.iter().all(|v| v.is_finite()),
        }
    }

    /// Column-major QR factor of the row-major ndarray design.
    struct QrOls {
        n: usize,
        p: usize,
        qr: Vec<f64>,
        tau: Vec<f64>,
        t: Vec<f64>,
        blocked: bool,
    }

    impl QrOls {
        fn prepare(x: ArrayView2<'_, f64>) -> Self {
            let (n, p) = x.dim();
            assert!(n > p, "QR probe needs residual degrees of freedom");
            assert!(all_finite(x));
            let mut qr = vec![0.0; n * p];
            // This is the one-time row-major -> LAPACK column-major conversion.
            for j in 0..p {
                for i in 0..n {
                    qr[j * n + i] = x[(i, j)];
                }
            }
            let mut tau = vec![0.0; p];
            let (m, k, lda) = (n as i32, p as i32, n as i32);
            let mut info = 0;
            let mut query = [0.0];
            let query_lwork = -1;
            unsafe {
                dgeqrf_(
                    &m,
                    &k,
                    qr.as_mut_ptr(),
                    &lda,
                    tau.as_mut_ptr(),
                    query.as_mut_ptr(),
                    &query_lwork,
                    &mut info,
                );
            }
            lapack_ok(info, "dgeqrf workspace query");
            let factor_lwork = queried_work_len(query[0], "dgeqrf");
            let mut factor_work = vec![0.0; factor_lwork];
            unsafe {
                dgeqrf_(
                    &m,
                    &k,
                    qr.as_mut_ptr(),
                    &lda,
                    tau.as_mut_ptr(),
                    factor_work.as_mut_ptr(),
                    &(factor_lwork as i32),
                    &mut info,
                );
            }
            lapack_ok(info, "dgeqrf");
            // Compact WY form Q = I - V T V' for the complete forward QR
            // reflector product.  This is preparation-only and has no dense Q.
            let mut t = vec![0.0; p * p];
            unsafe {
                dlarft_(
                    b"F".as_ptr(),
                    b"C".as_ptr(),
                    &m,
                    &k,
                    qr.as_ptr(),
                    &lda,
                    tau.as_ptr(),
                    t.as_mut_ptr(),
                    &k,
                );
            }
            Self {
                n,
                p,
                qr,
                tau,
                t,
                blocked: std::env::var("FMRIREG_QR_BLOCKED").as_deref() == Ok("1"),
            }
        }

        fn workspace(&self, block: usize) -> QrWorkspace {
            assert!(block > 0 && block <= i32::MAX as usize);
            let mut transformed = vec![0.0; self.n * block];
            let mut query = [0.0];
            let (m, n, k, lda, ldc, lwork) = (
                block as i32,
                self.n as i32,
                self.p as i32,
                self.n as i32,
                block as i32,
                -1_i32,
            );
            let mut info = 0;
            unsafe {
                dormqr_(
                    b"R".as_ptr(),
                    b"N".as_ptr(),
                    &m,
                    &n,
                    &k,
                    self.qr.as_ptr(),
                    &lda,
                    self.tau.as_ptr(),
                    transformed.as_mut_ptr(),
                    &ldc,
                    query.as_mut_ptr(),
                    &lwork,
                    &mut info,
                );
            }
            lapack_ok(info, "dormqr workspace query");
            QrWorkspace {
                transformed,
                variance: vec![0.0; block],
                work: vec![0.0; queried_work_len(query[0], "dormqr")],
                block_work: vec![0.0; block * self.p],
                block,
            }
        }

        /// `data` is row-major.  That byte layout is column-major `block x n`,
        /// i.e. Y transpose.  Thus right-applying Q produces `(Q'Y)'` without
        /// a layout conversion. The copy and finite validation are timed.
        fn fit_into(&self, data: ArrayView2<'_, f64>, ws: &mut QrWorkspace) {
            let (n, block) = data.dim();
            assert_eq!((n, block), (self.n, ws.block));
            assert!(all_finite(data));
            ws.transformed.copy_from_slice(
                data.as_slice()
                    .expect("probe requires contiguous row-major data"),
            );
            let (m, nrhs, k, lda, ldc, lwork) = (
                block as i32,
                self.n as i32,
                self.p as i32,
                self.n as i32,
                block as i32,
                ws.work.len() as i32,
            );
            if self.blocked {
                // `transformed` is B x N column-major, and WORK is B x P;
                // D/LARFB therefore applies the compact WY Q on the right.
                unsafe {
                    dlarfb_(
                        b"R".as_ptr(),
                        b"N".as_ptr(),
                        b"F".as_ptr(),
                        b"C".as_ptr(),
                        &m,
                        &nrhs,
                        &k,
                        self.qr.as_ptr(),
                        &lda,
                        self.t.as_ptr(),
                        &k,
                        ws.transformed.as_mut_ptr(),
                        &ldc,
                        ws.block_work.as_mut_ptr(),
                        &m,
                    );
                }
            } else {
                let mut info = 0;
                unsafe {
                    dormqr_(
                        b"R".as_ptr(),
                        b"N".as_ptr(),
                        &m,
                        &nrhs,
                        &k,
                        self.qr.as_ptr(),
                        &lda,
                        self.tau.as_ptr(),
                        ws.transformed.as_mut_ptr(),
                        &ldc,
                        ws.work.as_mut_ptr(),
                        &lwork,
                        &mut info,
                    );
                }
                lapack_ok(info, "dormqr");
            }
            let alpha = 1.0;
            let beta_columns = self.p as i32;
            unsafe {
                dtrsm_(
                    b"R".as_ptr(),
                    b"U".as_ptr(),
                    b"T".as_ptr(),
                    b"N".as_ptr(),
                    &m,
                    &beta_columns,
                    &alpha,
                    self.qr.as_ptr(),
                    &lda,
                    ws.transformed.as_mut_ptr(),
                    &ldc,
                );
            }
            ws.variance.fill(0.0);
            for row in self.p..self.n {
                let transformed_row = &ws.transformed[row * block..(row + 1) * block];
                for (sum, &value) in ws.variance.iter_mut().zip(transformed_row) {
                    *sum += value * value;
                }
            }
            for value in &mut ws.variance {
                *value /= (self.n - self.p) as f64;
            }
            assert!(self
                .coefficients(ws)
                .iter()
                .chain(ws.variance.iter())
                .all(|v| v.is_finite()));
        }

        fn coefficients<'a>(&self, ws: &'a QrWorkspace) -> &'a [f64] {
            &ws.transformed[..self.p * ws.block]
        }

        fn mode(&self) -> &'static str {
            if self.blocked {
                "dlarfb_compact_wy"
            } else {
                "dormqr"
            }
        }
    }

    struct QrWorkspace {
        transformed: Vec<f64>,
        variance: Vec<f64>,
        work: Vec<f64>,
        block_work: Vec<f64>,
        block: usize,
    }

    fn measure(mut f: impl FnMut(), iterations: usize) -> f64 {
        let start = Instant::now();
        for _ in 0..iterations {
            f();
        }
        start.elapsed().as_secs_f64() / iterations as f64
    }

    fn iterations(mut f: impl FnMut()) -> usize {
        for _ in 0..3 {
            f();
        }
        (0.015 / measure(&mut f, 1)).ceil().clamp(1.0, 10_000.0) as usize
    }

    /// Each sample is measured in ABBA order, reducing directional thermal and
    /// cache drift while retaining independently timed reference/candidate fits.
    fn abba(mut a: impl FnMut(), mut b: impl FnMut()) -> (Vec<f64>, Vec<f64>) {
        for _ in 0..3 {
            a();
            b();
        }
        let count = iterations(&mut a).max(iterations(&mut b));
        let mut aa = Vec::with_capacity(14);
        let mut bb = Vec::with_capacity(14);
        for _ in 0..7 {
            aa.push(measure(&mut a, count));
            bb.push(measure(&mut b, count));
            bb.push(measure(&mut b, count));
            aa.push(measure(&mut a, count));
        }
        (aa, bb)
    }

    fn assert_close(label: &str, actual: &[f64], expected: &[f64], rel: f64, abs: f64) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            let tolerance = abs + rel * want.abs().max(got.abs());
            assert!(
                got.is_finite() && (got - want).abs() <= tolerance,
                "{label}[{i}]: got {got:e}, expected {want:e}, tolerance {tolerance:e}"
            );
        }
    }

    fn design(n: usize, p: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                1.0
            } else {
                ((i + 1) as f64 * (j + 1) as f64 * 0.013).sin()
            }
        })
    }

    fn responses(n: usize, block: usize) -> Array2<f64> {
        Array2::from_shape_fn((n, block), |(i, j)| {
            let z = (i as u64 + 1)
                .wrapping_mul(6364136223846793005)
                .wrapping_add((j as u64 + 1).wrapping_mul(1442695040888963407));
            (z >> 11) as f64 / (1_u64 << 53) as f64
        })
    }

    fn verify_against_model(x: &Array2<f64>, y: &Array2<f64>, qr: &QrOls, ws: &mut QrWorkspace) {
        let model = LinearModel::new(x.view(), 1e-12)
            .unwrap()
            .with_backend(MatrixBackend::default());
        assert_eq!(
            model.rank(),
            x.ncols(),
            "probe only supports full-rank designs"
        );
        let mut reference_ws = model.workspace(y.ncols()).unwrap();
        let reference = model.fit_into(y.view(), &mut reference_ws).unwrap();
        qr.fit_into(y.view(), ws);
        assert_close(
            "coefficients",
            qr.coefficients(ws),
            reference.coefficients().as_slice().unwrap(),
            2e-11,
            2e-12,
        );
        assert_close(
            "residual variance",
            &ws.variance,
            reference.residual_variance().as_slice().unwrap(),
            2e-10,
            2e-14,
        );
    }

    fn analytic_sanity() -> serde_json::Value {
        let x =
            Array2::from_shape_vec((5, 2), vec![1., -2., 1., -1., 1., 0., 1., 1., 1., 2.]).unwrap();
        // All values are exactly representable, including the perturbation;
        // this tests a large offset without an unobservable sub-ULP residual.
        let beta = [2_f64.powi(20), -2_f64.powi(12)];
        // Orthogonal to both columns, so the known beta and RSS are exact OLS
        // targets before the deliberately difficult large-offset addition.
        let noise = [0.25, -0.5, 0.0, 0.5, -0.25];
        let y = Array2::from_shape_fn((5, 1), |(i, _)| {
            x[(i, 0)] * beta[0] + x[(i, 1)] * beta[1] + noise[i]
        });
        let qr = QrOls::prepare(x.view());
        let mut ws = qr.workspace(1);
        qr.fit_into(y.view(), &mut ws);
        // The design is centred: the analytic intercept is exactly the large offset.
        assert_close(
            "large-offset beta",
            qr.coefficients(&ws),
            &beta,
            2e-14,
            1e-9,
        );
        let expected_variance = noise.iter().map(|v| v * v).sum::<f64>() / 3.0;
        let observed_variance = ws.variance[0];
        let relative_variance_error =
            (observed_variance - expected_variance).abs() / expected_variance;
        let y_norm = y.iter().map(|value| value * value).sum::<f64>().sqrt();
        let residual_norm = (expected_variance * 3.0).sqrt();
        // Heuristic scale-aware acceptance tolerance for this deliberately
        // large-offset specimen. `64` is not a proved QR error constant, and
        // the expression omits design-factorization error; retain the observed
        // error below rather than presenting this as a numerical guarantee.
        let conditioning_scaled_allowance = 64.0 * f64::EPSILON * y_norm / residual_norm;
        assert!(
            relative_variance_error <= conditioning_scaled_allowance,
            "large-offset variance relative error {relative_variance_error:e} exceeds conditioning-scaled allowance {conditioning_scaled_allowance:e}"
        );
        verify_against_model(&x, &y, &qr, &mut ws);
        serde_json::json!({
            "expected_variance": expected_variance,
            "observed_variance": observed_variance,
            "relative_variance_error": relative_variance_error,
            "conditioning_scaled_allowance": conditioning_scaled_allowance,
            "y_norm": y_norm,
            "residual_norm": residual_norm,
        })
    }

    pub fn run() {
        let analytic = analytic_sanity();
        let mut records = Vec::new();
        for (n, p, block) in [
            (300, 8, 32),
            (300, 8, 128),
            (300, 8, 1000),
            (300, 8, 4096),
            (300, 32, 32),
            (300, 32, 128),
            (300, 32, 1000),
            (300, 32, 4096),
            (1200, 8, 32),
            (1200, 8, 128),
            (1200, 8, 1000),
            (1200, 8, 4096),
            (1200, 32, 32),
            (1200, 32, 128),
            (1200, 32, 1000),
            (1200, 32, 4096),
        ] {
            let x = design(n, p);
            let y = responses(n, block);
            let preparation = measure(
                || {
                    black_box(QrOls::prepare(x.view()));
                },
                1,
            );
            let qr = QrOls::prepare(x.view());
            let mut qr_ws = qr.workspace(block);
            verify_against_model(&x, &y, &qr, &mut qr_ws);
            let model = LinearModel::new(x.view(), 1e-12)
                .unwrap()
                .with_backend(MatrixBackend::default());
            let mut model_ws = model.workspace(block).unwrap();
            let (model_samples, qr_samples) = abba(
                || {
                    black_box(model.fit_into(black_box(y.view()), &mut model_ws).unwrap());
                },
                || {
                    qr.fit_into(black_box(y.view()), &mut qr_ws);
                    black_box((qr.coefficients(&qr_ws), &qr_ws.variance));
                },
            );
            records.push(serde_json::json!({
                "n": n, "p": p, "block": block, "backend": format!("{:?}", MatrixBackend::default()),
                "qr_mode": qr.mode(),
                "preparation_seconds": preparation,
                "model_fit_seconds_abba": model_samples,
                "implicit_qr_fit_seconds_abba": qr_samples,
                "fit_includes_response_packing": true,
                "validation": "all coefficient and residual-variance entries checked; full-rank only"
            }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "analytic_sanity": analytic,
                "records": records,
            }))
            .unwrap()
        );
    }
}

#[cfg(all(target_os = "macos", feature = "accelerate"))]
fn main() {
    probe::run();
}

#[cfg(not(all(target_os = "macos", feature = "accelerate")))]
fn main() {
    eprintln!("algebra_qr requires macOS and `--features accelerate`");
}
