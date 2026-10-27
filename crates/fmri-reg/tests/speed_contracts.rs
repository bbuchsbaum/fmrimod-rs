use fmri_noise::{Ar1Plan, FirstSample};
use fmri_reg::{GlsModel, LinearModel, MatrixBackend};
use ndarray::{array, s, Array2, ShapeBuilder};

fn backends() -> Vec<MatrixBackend> {
    #[allow(unused_mut)]
    let mut backends = vec![MatrixBackend::Portable];
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    backends.push(MatrixBackend::Accelerate);
    backends
}

#[test]
fn analytic_variance_and_reuse_with_tail_blocks_and_strides() {
    for backend in backends() {
        let n = 128;
        let x = Array2::ones((n, 1));
        let model = LinearModel::new(x.view(), 1e-12)
            .unwrap()
            .with_backend(backend);
        let y = Array2::from_shape_fn((n, 9), |(i, j)| {
            2_f64.powi(j as i32 - 4) * (1048576. + if i % 2 == 0 { 0.25 } else { -0.25 })
        });
        let mut workspace = model.workspace(9).unwrap();
        let bytes = workspace.working_bytes();
        let pointer = model
            .fit_into(y.view(), &mut workspace)
            .unwrap()
            .coefficients()
            .as_ptr();
        for width in [9, 1, 5, 9] {
            // Reversed rows and a noncontiguous feature slice exercise fallback.
            let data = y.slice(s![..;-1, ..width]);
            let fit = model.fit_into(data, &mut workspace).unwrap();
            assert_eq!(fit.coefficients().as_ptr(), pointer);
            for j in 0..width {
                let scale = 2_f64.powi(j as i32 - 4);
                let expected = scale * scale * 0.0625 * n as f64 / (n - 1) as f64;
                assert!((fit.residual_variance()[j] / expected - 1.).abs() < 1e-12);
                assert!((fit.coefficients()[(0, j)] / (1048576. * scale) - 1.).abs() < 1e-13);
            }
            let t = fit.t_contrast(array![1.].view()).unwrap();
            let f = fit.f_contrast(array![[1.]].view()).unwrap();
            for (t, f) in t.statistic.iter().zip(&f.statistic) {
                assert!((f / (t * t) - 1.).abs() < 1e-12);
            }
            assert_eq!(workspace.working_bytes(), bytes);
        }
        // Column-major data must also retain the same mathematical result.
        let colmajor = Array2::from_shape_fn((n, 9).f(), |(i, j)| y[(i, j)]);
        let fit = model.fit_into(colmajor.view(), &mut workspace).unwrap();
        assert!((fit.coefficients()[(0, 0)] - 65536.).abs() < 1e-8);
    }
}

#[test]
fn workspaces_reject_invalid_inputs_and_recover_after_error() {
    let x = Array2::ones((8, 1));
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    assert!(model.workspace(0).is_err());
    assert!(model.workspace(usize::MAX).is_err());
    let mut workspace = model.workspace(2).unwrap();
    assert!(model
        .fit_into(Array2::ones((7, 2)).view(), &mut workspace)
        .is_err());
    assert!(model
        .fit_into(Array2::ones((8, 3)).view(), &mut workspace)
        .is_err());
    assert!(model
        .fit_into(Array2::from_elem((8, 2), f64::NAN).view(), &mut workspace)
        .is_err());
    // Finite inputs whose residual squares overflow must fail too.
    let huge = Array2::from_shape_fn((8, 2), |(i, _)| if i % 2 == 0 { 1e200 } else { -1e200 });
    assert!(model.fit_into(huge.view(), &mut workspace).is_err());
    let y = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64);
    let fit = model.fit_into(y.view(), &mut workspace).unwrap();
    assert!((fit.coefficients()[(0, 0)] - 3.5).abs() < 1e-12);
    let gls = GlsModel::new(
        x.view(),
        Ar1Plan::new(vec![8], vec![0.4], FirstSample::Stationary).unwrap(),
        1e-12,
    )
    .unwrap();
    assert!(gls.fit_into(y.view(), &mut workspace).is_err());
}

#[test]
fn reused_gls_matches_independent_analytic_two_run_solution() {
    // At phi=0.5 with stationary startup, intercept GLS weights per run
    // are 0.5 at endpoints, 0.25 inside. Each run has length 4.
    let x = Array2::ones((8, 1));
    let y = array![
        [1., 8.],
        [2., 7.],
        [4., 6.],
        [5., 5.],
        [2., 4.],
        [3., 3.],
        [7., 2.],
        [8., 1.]
    ];
    for backend in backends() {
        let plan = Ar1Plan::new(vec![4, 4], vec![0.5, 0.5], FirstSample::Stationary).unwrap();
        let model = GlsModel::new(x.view(), plan, 1e-12)
            .unwrap()
            .with_backend(backend);
        let mut workspace = model.workspace(2).unwrap();
        for _ in 0..3 {
            let fit = model.fit_into(y.view(), &mut workspace).unwrap();
            for j in 0..2 {
                let weights = [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25, 0.5];
                let beta = (0..8).map(|i| weights[i] * y[(i, j)]).sum::<f64>() / 3.;
                let residual: Vec<_> = (0..8).map(|i| y[(i, j)] - beta).collect();
                // Independent tridiagonal precision quadratic form.
                let mut rss = 0.;
                for start in [0, 4] {
                    rss += residual[start].powi(2) + residual[start + 3].powi(2);
                    for r in &residual[start + 1..start + 3] {
                        rss += 1.25 * r.powi(2);
                    }
                    for i in start..start + 3 {
                        rss -= residual[i] * residual[i + 1];
                    }
                }
                assert!((fit.coefficients()[(0, j)] - beta).abs() < 1e-12);
                assert!((fit.residual_variance()[j] - rss / 7.).abs() < 1e-12);
            }
        }
    }
}
