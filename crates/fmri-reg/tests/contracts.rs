use fmri_noise::{Ar1Plan, FirstSample};
use fmri_reg::{FitError, GlsModel, LinearModel};
use ndarray::{array, s, Array2};
use serde_json::Value;
fn matrix(v: &Value) -> Array2<f64> {
    let rows = v.as_array().unwrap();
    let n = rows.len();
    let p = rows[0].as_array().unwrap().len();
    Array2::from_shape_fn((n, p), |(i, j)| rows[i][j].as_f64().unwrap())
}
fn close<'a>(actual: impl IntoIterator<Item = &'a f64>, expected: &Value) {
    let expected: Vec<f64> = if expected[0].is_array() {
        expected
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect()
    } else {
        expected
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect()
    };
    let actual: Vec<_> = actual.into_iter().copied().collect();
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 2e-11 * (1.0 + b.abs()), "{a} != {b}");
    }
}
#[test]
fn r_ols_contrasts_and_explicit_covariance_gls() {
    let f: Value = serde_json::from_str(include_str!("fixtures/estimation.json")).unwrap();
    let x = matrix(&f["X"]);
    let y = matrix(&f["Y"]);
    let c = matrix(&f["C"]);
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let fit = model.fit(y.view()).unwrap();
    close(fit.coefficients(), &f["ols"]["beta"]);
    close(fit.residual_variance(), &f["ols"]["variance"]);
    assert_eq!(fit.residual_df(), 21);
    let t = fit.t_contrast(c.row(0)).unwrap();
    let ft = fit.f_contrast(c.view()).unwrap();
    close(&t.effect, &f["ols"]["effect"]);
    close(&t.standard_error, &f["ols"]["se"]);
    close(&t.statistic, &f["ols"]["t"]);
    close(&t.p_value, &f["ols"]["p"]);
    close(&ft.statistic, &f["ols"]["F"]);
    close(&ft.p_value, &f["ols"]["Fp"]);
    let noise = Ar1Plan::new(vec![12, 12], vec![0.4, -0.25], FirstSample::Stationary).unwrap();
    close(&noise.apply(x.view()).unwrap(), &f["whitened_X"]);
    close(&noise.apply(y.view()).unwrap(), &f["whitened_Y"]);
    let gls = GlsModel::new(x.view(), noise, 1e-12).unwrap();
    let fit = gls.fit(y.view()).unwrap();
    close(fit.coefficients(), &f["dense_gls_beta"]);
    close(fit.residual_variance(), &f["dense_gls_variance"]);
    let t = fit.t_contrast(c.row(0)).unwrap();
    close(&t.statistic, &f["gls"]["t"]);
    close(&t.p_value, &f["gls"]["p"]);
    let ft = fit.f_contrast(c.view()).unwrap();
    close(&ft.statistic, &f["gls"]["F"]);
    // F with one row is t squared.
    let f1 = fit.f_contrast(c.slice(s![0..1, ..])).unwrap();
    for (f, t) in f1.statistic.iter().zip(&t.statistic) {
        assert!((f - t * t).abs() < 1e-9);
    }
}
#[test]
fn rank_deficient_estimability_and_chunk_equivalence() {
    let x = array![
        [1., 0., 0.],
        [1., 1., 1.],
        [1., 2., 2.],
        [1., 3., 3.],
        [1., 4., 4.]
    ];
    let y = array![[1., 2.], [2., 1.], [2., 4.], [4., 3.], [6., 6.]];
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    assert_eq!(model.rank(), 2);
    let fit = model.fit(y.view()).unwrap();
    assert_eq!(fit.residual_df(), 3);
    assert!(fit.t_contrast(array![0., 1., 1.].view()).is_ok());
    assert!(matches!(
        fit.t_contrast(array![0., 1., 0.].view()),
        Err(FitError::NotEstimable)
    ));
    for j in 0..2 {
        let block = model.fit(y.slice(s![.., j..j + 1])).unwrap();
        for i in 0..3 {
            assert!((block.coefficients()[(i, 0)] - fit.coefficients()[(i, j)]).abs() < 1e-12);
        }
    }
    assert!(fit
        .f_contrast(array![[0., 1., 1.], [0., 2., 2.]].view())
        .is_err());
}
#[test]
fn analytic_intercept_and_invalid_inputs() {
    let x = Array2::ones((4, 1));
    let y = array![[1.], [2.], [4.], [5.]];
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let fit = model.fit(y.view()).unwrap();
    assert!((fit.coefficients()[(0, 0)] - 3.).abs() < 1e-13);
    assert!((fit.residual_variance()[0] - 10. / 3.).abs() < 1e-13);
    assert!(LinearModel::new(array![[1., 0.], [0., 1.]].view(), 1e-12).is_err());
    assert!(LinearModel::new(array![[f64::NAN], [1.]].view(), 1e-12).is_err());
    assert!(model.fit(array![[1.]].view()).is_err());
    assert!(model
        .fit(Array2::from_elem((4, 1), f64::INFINITY).view())
        .is_err());
    assert!(fit.t_contrast(array![0.].view()).is_err());
}
#[test]
fn ill_conditioned_full_rank_design_recovers_known_coefficients() {
    let x = Array2::from_shape_fn((20, 3), |(i, j)| match j {
        0 => 1.,
        1 => i as f64 / 20.,
        _ => i as f64 / 20. + 1e-6 * (i as f64).sin(),
    });
    let beta = array![[2.], [-3.], [4.]];
    let y = x.dot(&beta);
    let m = LinearModel::new(x.view(), 1e-12).unwrap();
    assert_eq!(m.rank(), 3);
    let fit = m.fit(y.view()).unwrap();
    for (a, b) in fit.coefficients().iter().zip(&beta) {
        assert!((a - b).abs() < 2e-8);
    }
}

#[cfg(feature = "dataset")]
#[test]
fn bounded_reader_matches_dense_and_refuses_budget_before_sink() {
    use fmri_data::{ExecutionContext, MemoryBudget, NumericBlock, SourceSpec, Values};
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.], [1., 4.]];
    let y = Array2::from_shape_fn((5, 7), |(i, j)| {
        i as f64 + j as f64 + ((i + j) as f64).sin()
    });
    let source = SourceSpec::memory(
        NumericBlock::new([5, 7], Values::F64(y.iter().copied().collect())).unwrap(),
    );
    let ctx = ExecutionContext::new(MemoryBudget::new(10000));
    let mut reader = ctx.open(&source);
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let dense = model.fit(y.view()).unwrap();
    let mut seen = Vec::new();
    fmri_reg::dataset::fit_blocks(&model, &mut reader, 3, 10000, |range, fit| {
        for (local, global) in range.enumerate() {
            seen.push(global);
            for i in 0..2 {
                assert!(
                    (fit.coefficients()[(i, local)] - dense.coefficients()[(i, global)]).abs()
                        < 1e-12
                );
            }
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, (0..7).collect::<Vec<_>>());
    let mut called = false;
    assert!(
        fmri_reg::dataset::fit_blocks(&model, &mut reader, 3, 10, |_, _| {
            called = true;
            Ok(())
        })
        .is_err()
    );
    assert!(!called);
}
#[test]
fn feasible_gls_uses_corrected_noise_and_preserves_feature_block_fits() {
    use fmri_reg::Ar1Options;
    let f: Value = serde_json::from_str(include_str!(
        "../../fmri-noise/tests/fixtures/correction.json"
    ))
    .unwrap();
    let x = matrix(&f["design"]);
    let residuals = matrix(&f["residuals"]);
    let beta = Array2::from_shape_fn((4, 2), |(i, j)| (i + j + 1) as f64);
    let y = &residuals + &x.dot(&beta);
    let mut valid = vec![true; 120];
    for c in f["censor"].as_array().unwrap() {
        valid[c.as_u64().unwrap() as usize - 1] = false;
    }
    let model = GlsModel::estimate_ar1(
        x.view(),
        y.view(),
        Ar1Options {
            run_lengths: vec![60, 60],
            noise_valid: Some(valid),
            correction_max_lag: 8,
            pooling: fmri_noise::Pooling::Run,
        },
        1e-12,
    )
    .unwrap();
    for (i, p) in model.noise_plan().coefficients().iter().enumerate() {
        assert!((p - f["phi"][i].as_f64().unwrap()).abs() < 2e-11);
    }
    let full = model.fit(y.view()).unwrap();
    for j in 0..2 {
        let part = model.fit(y.slice(s![.., j..j + 1])).unwrap();
        for i in 0..4 {
            assert!((part.coefficients()[(i, 0)] - full.coefficients()[(i, j)]).abs() < 1e-11);
        }
    }
}

#[test]
fn contrast_overflow_errors_instead_of_returning_misleading_inference() {
    let x = Array2::ones((4, 1));
    let y = array![[1e10], [2e10], [4e10], [5e10]];
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let fit = model.fit(y.view()).unwrap();
    assert!(fit.t_contrast(array![1e150].view()).is_err());
    assert!(fit.f_contrast(array![[1e150]].view()).is_err());
}

#[test]
fn residual_variance_preserves_small_errors_on_large_offsets_across_blocks() {
    // Exact alternating residuals around a large offset expose cancellation in
    // RSS shortcuts. Powers of two give an analytic variance at several scales.
    let n = 1024;
    let x = Array2::ones((n, 1));
    let scales: Vec<_> = [-100, -10, 0, 10, 100].map(|e| 2.0_f64.powi(e)).to_vec();
    let y = Array2::from_shape_fn((n, scales.len()), |(i, j)| {
        scales[j] * (1048576.0 + if i % 2 == 0 { 0.25 } else { -0.25 })
    });
    let model = LinearModel::new(x.view(), 1e-12).unwrap();
    let full = model.fit(y.view()).unwrap();
    for (j, scale) in scales.iter().enumerate() {
        let expected = scale * scale * 0.0625 * n as f64 / (n - 1) as f64;
        assert!((full.residual_variance()[j] / expected - 1.0).abs() < 1e-12);
        let one = model.fit(y.slice(s![.., j..j + 1])).unwrap();
        assert!((one.residual_variance()[0] / expected - 1.0).abs() < 1e-12);
    }
}
