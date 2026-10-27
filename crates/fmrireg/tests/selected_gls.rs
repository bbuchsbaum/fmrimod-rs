use fmri_noise::{ArPlan, ArStart, GapPolicy, ObservationSelection, SelectedArPlan, WhiteningPlan};
use fmrireg::{GlsModel, LinearModel};
use ndarray::{array, s, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/selected_gls.json")).unwrap()
}
fn vector(v: &Value) -> Vec<f64> {
    if let Some(a) = v.as_array() {
        a.iter().map(|x| x.as_f64().unwrap()).collect()
    } else {
        vec![v.as_f64().unwrap()]
    }
}
fn matrix(v: &Value) -> Array2<f64> {
    let rows = v.as_array().unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].as_array().unwrap().len()), |(i, j)| {
        rows[i][j].as_f64().unwrap()
    })
}
fn selection() -> ObservationSelection {
    let f = fixture();
    ObservationSelection::new(
        vec![14, 11, 7],
        f["keep"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_u64().unwrap() as usize)
            .collect(),
    )
    .unwrap()
}
fn plan(case: &Value) -> SelectedArPlan {
    SelectedArPlan::new(
        selection(),
        case["coefficients"]
            .as_array()
            .unwrap()
            .iter()
            .map(vector)
            .collect(),
        if case["start"] == "stationary" {
            ArStart::Stationary
        } else {
            ArStart::ZeroHistory
        },
        if case["policy"] == "preserve" {
            GapPolicy::Preserve
        } else {
            GapPolicy::Restart
        },
    )
    .unwrap()
}
fn close(a: impl IntoIterator<Item = f64>, b: impl IntoIterator<Item = f64>) {
    let a: Vec<_> = a.into_iter().collect();
    let b: Vec<_> = b.into_iter().collect();
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 2e-9 * (1. + y.abs()), "{x} != {y}");
    }
}
#[test]
fn selected_arp_matches_dense_r_covariance_gls_and_inference() {
    let f = fixture();
    let xs = selection().select(matrix(&f["x"]).view()).unwrap();
    let ys = selection().select(matrix(&f["y"]).view()).unwrap();
    for case in f["cases"].as_array().unwrap() {
        let noise = plan(case);
        let w = noise.apply(Array2::eye(xs.nrows()).view()).unwrap();
        let sigma = matrix(&case["covariance"]);
        close(
            w.dot(&sigma).dot(&w.t()).iter().copied(),
            Array2::eye(xs.nrows()).iter().copied(),
        );
        let transformed = noise.apply(xs.view()).unwrap();
        let linear = LinearModel::new(transformed.view(), 1e-12).unwrap();
        close(
            linear.unit_covariance().iter().copied(),
            matrix(&case["unit_covariance"]).iter().copied(),
        );
        let model = GlsModel::new(xs.view(), noise, 1e-12).unwrap();
        let fit = model.fit(ys.view()).unwrap();
        close(
            fit.coefficients().iter().copied(),
            matrix(&case["beta"]).iter().copied(),
        );
        close(
            fit.residual_variance().iter().copied(),
            vector(&case["variance"]),
        );
        assert_eq!(fit.residual_df(), case["df"].as_u64().unwrap() as usize);
        let t = fit.t_contrast(array![0., 1., -1.].view()).unwrap();
        for (actual, key) in [
            (&t.effect, "effect"),
            (&t.standard_error, "se"),
            (&t.statistic, "t"),
            (&t.p_value, "t_p"),
        ] {
            close(actual.iter().copied(), vector(&case[key]));
        }
        let fc = fit
            .f_contrast(array![[0., 1., 0.], [0., 0., 1.]].view())
            .unwrap();
        close(fc.statistic.iter().copied(), vector(&case["F"]));
        close(fc.p_value.iter().copied(), vector(&case["F_p"]));
        let mut workspace = model.workspace(ys.ncols()).unwrap();
        let reuse = model.fit_into(ys.view(), &mut workspace).unwrap();
        close(
            reuse.coefficients().iter().copied(),
            fit.coefficients().iter().copied(),
        );
    }
}
#[test]
fn no_gaps_matches_contiguous_kernels_and_selection_handles_strides() {
    let lengths = vec![9, 7];
    let coefficients = vec![vec![0.4, -0.13, 0.05], vec![-0.6]];
    let input = Array2::from_shape_fn((16, 8), |(i, j)| ((i * 3 + j) as f64).sin());
    let data = input.slice(s![..,..;2]);
    for start in [ArStart::Stationary, ArStart::ZeroHistory] {
        let reference = ArPlan::new(lengths.clone(), coefficients.clone(), start)
            .unwrap()
            .apply(data)
            .unwrap();
        for policy in [GapPolicy::Preserve, GapPolicy::Restart] {
            let plan = SelectedArPlan::new(
                ObservationSelection::new(lengths.clone(), (0..16).collect()).unwrap(),
                coefficients.clone(),
                start,
                policy,
            )
            .unwrap();
            close(
                plan.apply(data).unwrap().iter().copied(),
                reference.iter().copied(),
            );
            let mut out = Array2::zeros((16, 8));
            plan.apply_into(data, out.slice_mut(s![..,..;2])).unwrap();
            close(
                out.slice(s![..,..;2]).iter().copied(),
                reference.iter().copied(),
            );
        }
    }
    let selection = ObservationSelection::new(vec![16], vec![1, 4, 12]).unwrap();
    assert_eq!(selection.select(data).unwrap().row(1), data.row(4));
}
#[test]
fn invalid_selection_coefficients_shapes_and_short_runs() {
    for rows in [vec![], vec![1, 1], vec![2, 1], vec![4]] {
        assert!(ObservationSelection::new(vec![4], rows).is_err());
    }
    assert!(ObservationSelection::new(vec![0, 4], vec![1]).is_err());
    let selected = ObservationSelection::new(vec![4], vec![1, 3]).unwrap();
    for phi in [vec![1.], vec![f64::NAN], vec![2., 0.]] {
        assert!(SelectedArPlan::new(
            selected.clone(),
            vec![phi],
            ArStart::Stationary,
            GapPolicy::Preserve
        )
        .is_err());
    }
    assert!(SelectedArPlan::new(
        selected.clone(),
        vec![],
        ArStart::Stationary,
        GapPolicy::Preserve
    )
    .is_err());
    let plan = SelectedArPlan::new(
        selected.clone(),
        vec![vec![0.4, -0.1, 0.03]],
        ArStart::Stationary,
        GapPolicy::Preserve,
    )
    .unwrap();
    assert!(plan.apply(Array2::ones((4, 1)).view()).is_err());
    assert!(plan.apply(array![[1.], [f64::NAN]].view()).is_err());
    assert!(plan
        .apply_into(
            Array2::ones((2, 1)).view(),
            Array2::zeros((1, 1)).view_mut()
        )
        .is_err());
    assert!(selected
        .select(array![[f64::NAN], [1.], [f64::NAN], [2.]].view())
        .is_ok());
    let one = SelectedArPlan::new(
        ObservationSelection::new(vec![1, 3], vec![0, 3]).unwrap(),
        vec![vec![0.4, -0.1, 0.03]; 2],
        ArStart::Stationary,
        GapPolicy::Preserve,
    )
    .unwrap();
    assert!(one.apply(Array2::ones((2, 1)).view()).is_ok());
}
#[cfg(feature = "dataset")]
#[test]
fn selected_assay_streaming_preserves_ids_and_matches_dense() {
    use fmridataset::{
        Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
    };
    use fmrireg::{
        dataset::{BlockOptions, BoundModel},
        selection::select_assay,
    };
    let f = fixture();
    let original_x = matrix(&f["x"]);
    let mut y = matrix(&f["y"]);
    let selected = selection();
    let ids: Vec<_> = (0..32)
        .map(|i| ObservationId::new(format!("original-scan-{i}")).unwrap())
        .collect();
    let features: Vec<_> = (0..5)
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    // Excluded response rows must not leak into the selected likelihood.
    for i in 0..32 {
        if !selected.retained_indices().contains(&i) {
            y.row_mut(i).fill(f64::NAN);
        }
    }
    let assay = Assay::new(
        "bold",
        SourceSpec::memory(
            NumericBlock::new([32, 5], Values::F64(y.iter().copied().collect())).unwrap(),
        ),
        ids.clone(),
        features.clone(),
        AssayAnnotations::default(),
    )
    .unwrap();
    let view = select_assay(&assay, &ids, &selected).unwrap();
    let expected_ids: Vec<_> = selected
        .retained_indices()
        .iter()
        .map(|&i| ids[i].clone())
        .collect();
    assert_eq!(view.observation_ids(), expected_ids);
    let mut wrong = ids.clone();
    wrong.swap(0, 1);
    assert!(select_assay(&assay, &wrong, &selected).is_err());
    for case in f["cases"].as_array().unwrap() {
        let model = GlsModel::new(
            selected.select(original_x.view()).unwrap().view(),
            plan(case),
            1e-12,
        )
        .unwrap();
        let dense = model
            .fit(selected.select(y.view()).unwrap().view())
            .unwrap();
        let bound = BoundModel::new(&model, expected_ids.clone()).unwrap();
        for workers in [1, 3] {
            let mut offset = 0;
            bound
                .fit_assay_with_options(
                    &view,
                    BlockOptions {
                        block_features: 2,
                        workers,
                        working_bytes: 100_000,
                    },
                    |ids, fit| {
                        assert_eq!(ids, &features[offset..offset + ids.len()]);
                        close(
                            fit.coefficients().iter().copied(),
                            dense
                                .coefficients()
                                .slice(s![.., offset..offset + ids.len()])
                                .iter()
                                .copied(),
                        );
                        close(
                            fit.residual_variance().iter().copied(),
                            dense
                                .residual_variance()
                                .slice(s![offset..offset + ids.len()])
                                .iter()
                                .copied(),
                        );
                        let c = array![0., 1., -1.];
                        let tc = fit.t_contrast(c.view())?;
                        let reference = dense.t_contrast(c.view())?;
                        close(
                            tc.statistic.iter().copied(),
                            reference
                                .statistic
                                .slice(s![offset..offset + ids.len()])
                                .iter()
                                .copied(),
                        );
                        offset += ids.len();
                        Ok(())
                    },
                )
                .unwrap();
            assert_eq!(offset, 5);
        }
        assert!(bound
            .fit_assay_with_options(
                &view,
                BlockOptions {
                    block_features: 2,
                    workers: 1,
                    working_bytes: 1
                },
                |_, _| Ok(())
            )
            .is_err());
    }
}

#[test]
fn excluding_a_run_uses_retained_rank_and_rejects_its_contrast() {
    let selection = ObservationSelection::new(vec![4, 4], vec![4, 5, 7]).unwrap();
    let x = Array2::from_shape_fn((8, 3), |(i, j)| match j {
        0 => {
            if i < 4 {
                1.
            } else {
                0.
            }
        }
        1 => {
            if i >= 4 {
                1.
            } else {
                0.
            }
        }
        _ => i as f64,
    });
    let noise = SelectedArPlan::new(
        selection.clone(),
        vec![vec![0.4, -0.1]; 2],
        ArStart::Stationary,
        GapPolicy::Preserve,
    )
    .unwrap();
    let model = GlsModel::new(selection.select(x.view()).unwrap().view(), noise, 1e-12).unwrap();
    let fit = model.fit(array![[1.], [3.], [2.]].view()).unwrap();
    assert_eq!(fit.residual_df(), 1); // 3 retained observations minus rank 2.
    assert!(matches!(
        fit.t_contrast(array![1., 0., 0.].view()),
        Err(fmrireg::FitError::NotEstimable)
    ));
    assert!(fit.t_contrast(array![0., 1., 0.].view()).is_ok());
}
