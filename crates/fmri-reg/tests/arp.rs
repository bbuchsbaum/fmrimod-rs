use fmri_noise::{ArPlan, ArStart, Pooling};
use fmri_reg::{ArOptions, GlsModel};
use ndarray::Array2;
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/arp.json")).unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    if let Some(a) = v.as_array() {
        a.iter().map(|v| v.as_f64().unwrap()).collect()
    } else {
        vec![v.as_f64().unwrap()]
    }
}
fn matrix(v: &Value) -> Array2<f64> {
    let a = v.as_array().unwrap();
    Array2::from_shape_fn((a.len(), a[0].as_array().unwrap().len()), |(i, j)| {
        a[i][j].as_f64().unwrap()
    })
}
fn close<'a>(a: impl IntoIterator<Item = &'a f64>, b: impl IntoIterator<Item = &'a f64>, tol: f64) {
    let a: Vec<_> = a.into_iter().collect();
    let b: Vec<_> = b.into_iter().collect();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert!((*a - b).abs() < tol * (1.0 + b.abs()), "{a} != {b}");
    }
}
#[test]
fn conditional_feasible_gls_matches_current_r_full_fit() {
    let f = fixture();
    let x = matrix(&f["X"]);
    let noise = matrix(&f["noise"]);
    let mut valid = vec![true; 160];
    for i in f["censor"].as_array().unwrap() {
        valid[i.as_u64().unwrap() as usize - 1] = false;
    }
    for c in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["corrected"] == true)
    {
        let model = GlsModel::estimate_ar(
            x.view(),
            noise.view(),
            ArOptions {
                run_lengths: vec![80, 80],
                noise_valid: Some(valid.clone()),
                order: c["order"].as_u64().unwrap() as usize,
                correction_max_lag: 10,
                pooling: if c["pooling"] == "run" {
                    Pooling::Run
                } else {
                    Pooling::Global
                },
                startup: ArStart::ZeroHistory,
            },
            1e-12,
        )
        .unwrap();
        let fit = model.fit(noise.view()).unwrap();
        close(fit.coefficients(), &matrix(&c["beta"]), 2e-8);
        close(fit.residual_variance(), &vector(&c["variance"]), 2e-8);
    }
}
#[test]
fn stationary_gls_matches_independent_dense_covariance_solve() {
    let f = fixture();
    for c in f["known"].as_array().unwrap() {
        let x = matrix(&c["X"]);
        let y = matrix(&c["Y"]);
        let starts: Vec<_> = c["resets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect();
        let plan = ArPlan::new(
            vec![x.nrows()],
            vec![vector(&c["phi"])],
            ArStart::Stationary,
        )
        .unwrap()
        .with_resets(&starts)
        .unwrap();
        let model = GlsModel::new(x.view(), plan, 1e-12).unwrap();
        let fit = model.fit(y.view()).unwrap();
        close(fit.coefficients(), &matrix(&c["beta"]), 2e-8);
        close(fit.residual_variance(), &vector(&c["variance"]), 2e-8);
    }
}
#[cfg(feature = "dataset")]
#[test]
fn aligned_gls_blocks_preserve_ids_and_reject_permuted_rows() {
    use fmri_data::{
        Assay, AssayAnnotations, ExecutionContext, FeatureId, MemoryBudget, NumericBlock,
        ObservationId, SourceSpec, Values,
    };
    use fmri_reg::dataset::{fit_blocks, BoundModel};
    let x = Array2::from_shape_fn((16, 2), |(i, j)| if j == 0 { 1.0 } else { i as f64 / 16.0 });
    let y = Array2::from_shape_fn((16, 7), |(i, j)| (i as f64 * 0.7 + j as f64).sin());
    let ids: Vec<_> = (0..16)
        .map(|i| ObservationId::new(format!("run-1/scan-{i}")).unwrap())
        .collect();
    let features: Vec<_> = (0..7)
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    let source = SourceSpec::memory(
        NumericBlock::new([16, 7], Values::F64(y.iter().copied().collect())).unwrap(),
    );
    let assay = Assay::new(
        "bold",
        source.clone(),
        ids.clone(),
        features.clone(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let plan = ArPlan::new(vec![16], vec![vec![0.5, -0.2]], ArStart::Stationary).unwrap();
    let model = GlsModel::new(x.view(), plan, 1e-12).unwrap();
    let dense = model.fit(y.view()).unwrap();
    let bound = BoundModel::new(&model, ids.clone()).unwrap();
    let mut seen = Vec::new();
    bound
        .fit_assay(&assay, 3, 10000, |ids, fit| {
            let start = seen.len();
            seen.extend_from_slice(ids);
            for j in 0..ids.len() {
                for i in 0..2 {
                    assert!(
                        (fit.coefficients()[(i, j)] - dense.coefficients()[(i, start + j)]).abs()
                            < 1e-12
                    );
                }
            }
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, features);
    let mut swapped = ids.clone();
    swapped.swap(0, 1);
    assert!(BoundModel::new(&model, swapped)
        .unwrap()
        .fit_assay(&assay, 3, 10000, |_, _| panic!("must reject before sink"))
        .is_err());
    let mut duplicates = ids;
    duplicates[1] = duplicates[0].clone();
    assert!(BoundModel::new(&model, duplicates).is_err());
    // Budget that would cover OLS but cannot cover the additional GLS transformed buffer.
    let ctx = ExecutionContext::new(MemoryBudget::new(10000));
    let mut reader = ctx.open(&source);
    assert!(fit_blocks(&model, &mut reader, 3, 1000, |_, _| panic!(
        "budget must fail before sink"
    ))
    .is_err());
}
