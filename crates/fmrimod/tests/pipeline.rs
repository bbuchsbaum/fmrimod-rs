#![cfg(feature = "formula")]
use fmrimod::{
    dataset::*,
    design,
    noise::{ArPlan, ArStart},
    regression::{dataset::BoundModel, GlsModel, LinearModel},
};
use ndarray::{array, s, Array1, Array2};
use serde_json::Value;
fn vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}
fn mat(v: &Value) -> Array2<f64> {
    let r = v.as_array().unwrap();
    Array2::from_shape_fn((r.len(), r[0].as_array().unwrap().len()), |(i, j)| {
        r[i][j].as_f64().unwrap()
    })
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 2e-8 * (1.0 + b.abs()), "{a} != {b}");
}
#[test]
fn fractional_multirun_multibasis_design_to_streamed_gls_and_contrasts() {
    let f: Value = serde_json::from_str(include_str!(
        "../../fmridesign/tests/fixtures/r_core_contract.json"
    ))
    .unwrap();
    let r: Value = serde_json::from_str(include_str!("fixtures/pipeline.json")).unwrap();
    let lens: Vec<usize> = f["frame"]["blocklens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let frame = design::sampling::SamplingFrame::new(
        lens.clone(),
        vec(&f["frame"]["tr"]),
        Some(vec(&f["frame"]["start_time"])),
        0.05,
    )
    .unwrap();
    let rows = f["events"].as_array().unwrap();
    let mut data = design::DataFrame::new();
    for name in ["onset", "duration"] {
        data.add_column(
            name,
            rows.iter().map(|r| r[name].as_f64().unwrap()).collect(),
        )
        .unwrap();
    }
    for name in ["condition", "block"] {
        data.add_str_column(
            name,
            rows.iter()
                .map(|r| r[name].as_str().unwrap().to_owned())
                .collect(),
        )
        .unwrap();
    }
    let event =
        design::formula::event_model("onset ~ hrf(condition, \"spmg3\")", &data, frame, 1.5)
            .unwrap();
    let task = event.generate_design_matrix().unwrap();
    let n = task.nrows();
    let mut x = Array2::zeros((n, 9));
    let mut first = 0;
    for (run, &len) in lens.iter().enumerate() {
        x.slice_mut(s![first..first + len, run]).fill(1.);
        first += len;
    }
    x.slice_mut(s![.., 3..]).assign(&task);
    for (a, b) in x.iter().zip(mat(&r["X"]).iter()) {
        close(*a, *b);
    }
    let names = event.column_names();
    assert_eq!(
        &names[..2],
        &["condition.A_b01".to_owned(), "condition.B_b01".to_owned()]
    );
    let phi = r["phi"].as_array().unwrap().iter().map(vec).collect();
    let plan = ArPlan::new(lens, phi, ArStart::Stationary).unwrap();
    let model = GlsModel::new(x.view(), plan, 1e-12).unwrap();
    let y = mat(&r["Y"]);
    let ids: Vec<_> = (0..n)
        .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
        .collect();
    let features: Vec<_> = (0..y.ncols())
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    let assay = Assay::new(
        "bold",
        SourceSpec::memory(
            NumericBlock::new([n, y.ncols()], Values::F64(y.iter().copied().collect())).unwrap(),
        ),
        ids.clone(),
        features.clone(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let bound = BoundModel::new(&model, ids).unwrap();
    let c = array![0., 0., 0., 1., -1., 0., 0., 0., 0.];
    let beta = mat(&r["beta"]);
    let mut seen = 0;
    bound
        .fit_assay(&assay, 3, 100000, |ids, fit| {
            let tc = fit.t_contrast(c.view())?;
            assert_eq!(ids, &features[seen..seen + ids.len()]);
            for j in 0..ids.len() {
                for k in 0..9 {
                    close(fit.coefficients()[(k, j)], beta[(k, seen + j)]);
                }
                for (name, value) in [
                    ("variance", fit.residual_variance()[j]),
                    ("effect", tc.effect[j]),
                    ("se", tc.standard_error[j]),
                    ("statistic", tc.statistic[j]),
                    ("p_value", tc.p_value[j]),
                ] {
                    close(value, r[name][seen + j].as_f64().unwrap());
                }
            }
            seen += ids.len();
            Ok(())
        })
        .unwrap();
    assert_eq!(seen, 7);
    // Aliasing does not turn an arbitrary individual coefficient into an estimable effect.
    let mut singular = Array2::zeros((n, 10));
    singular.slice_mut(s![.., ..9]).assign(&x);
    singular.column_mut(9).assign(&x.column(8));
    let rankdef = LinearModel::new(singular.view(), 1e-12).unwrap();
    assert_eq!(rankdef.rank(), 9);
    let fit = rankdef.fit(y.view()).unwrap();
    let mut bad = Array1::zeros(10);
    bad[8] = 1.;
    assert!(fit.t_contrast(bad.view()).is_err());
    bad[9] = 1.;
    assert!(fit.t_contrast(bad.view()).is_ok());
}
