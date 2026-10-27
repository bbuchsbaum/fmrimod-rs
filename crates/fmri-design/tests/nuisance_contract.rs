use fmri_design::{
    baseline::InterceptType,
    design::{hstack_design_matrices, DesignMatrix},
    nuisance::*,
};
use fmri_hrf::SamplingFrame;
use ndarray::{array, s, Array1, Array2};
use serde_json::Value;
fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/nuisance_contract.json")).unwrap()
}
fn vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn mat(v: &Value) -> Array2<f64> {
    let rows = v.as_array().unwrap();
    Array2::from_shape_fn((rows.len(), rows[0].as_array().unwrap().len()), |(i, j)| {
        rows[i][j].as_f64().unwrap()
    })
}
fn frame() -> SamplingFrame {
    SamplingFrame::new(vec![12, 9], vec![0.7, 1.1], Some(vec![0.15, 0.2]), 0.025).unwrap()
}
fn close(a: &Array2<f64>, b: &Array2<f64>) {
    assert_eq!(a.dim(), b.dim());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() < 2e-11 * (1. + y.abs()), "{x} != {y}");
    }
}
#[test]
fn current_r_polynomial_intercepts_and_named_covariates() {
    let f = fixture();
    let frame = frame();
    for case in f["polynomials"].as_array().unwrap() {
        let intercept = match case["intercept"].as_str().unwrap() {
            "none" => InterceptType::None,
            "global" => InterceptType::Global,
            _ => InterceptType::Runwise,
        };
        let d = drift_design(&frame, DriftBasis::Polynomial { degree: 2 }, intercept).unwrap();
        close(&d.data, &mat(&case["expected"]));
        assert_eq!(d.row_times.to_vec(), frame.samples());
    }
    let values: Vec<_> = f["nuisance_values"]
        .as_array()
        .unwrap()
        .iter()
        .map(mat)
        .collect();
    let d = nuisance_design(
        &frame,
        &[
            NuisanceRun {
                values: values[0].view(),
                names: &["x", "y"],
            },
            NuisanceRun {
                values: values[1].view(),
                names: &["x"],
            },
        ],
    )
    .unwrap();
    close(&d.data, &mat(&f["nuisance_expected"]));
    assert_eq!(
        d.column_names,
        ["nuisance.run1.x", "nuisance.run1.y", "nuisance.run2.x"]
    );
}
#[test]
fn dct_matches_r_and_is_orthonormal_run_local_without_dc() {
    let d = drift_design(
        &frame(),
        DriftBasis::Dct { cutoff_seconds: 4. },
        InterceptType::None,
    )
    .unwrap();
    close(&d.data, &mat(&fixture()["dct"]));
    close(&d.data.t().dot(&d.data), &Array2::eye(8));
    for col in d.data.columns() {
        assert!(col.sum().abs() < 1e-13);
    }
    assert!(d.data.slice(s![12.., ..4]).iter().all(|&x| x == 0.));
    assert!(d.data.slice(s![..12, 4..]).iter().all(|&x| x == 0.));
    let exact = SamplingFrame::uniform(8, 1., 0.1).unwrap();
    assert_eq!(
        drift_design(
            &exact,
            DriftBasis::Dct { cutoff_seconds: 8. },
            InterceptType::None
        )
        .unwrap()
        .ncols(),
        2
    );
    assert_eq!(
        drift_design(
            &exact,
            DriftBasis::Dct {
                cutoff_seconds: 100.
            },
            InterceptType::None
        )
        .unwrap()
        .ncols(),
        0
    );
    assert!(drift_design(
        &exact,
        DriftBasis::Dct { cutoff_seconds: 2. },
        InterceptType::None
    )
    .is_err());
}
fn solve(x: &Array2<f64>, y: &Array2<f64>) -> (Array2<f64>, usize, f64) {
    let a = nalgebra::DMatrix::from_fn(x.nrows(), x.ncols(), |i, j| x[(i, j)]);
    let b = nalgebra::DMatrix::from_fn(y.nrows(), y.ncols(), |i, j| y[(i, j)]);
    let svd = a.clone().svd(true, true);
    let rank = svd.rank(1e-11);
    let beta = svd.solve(&b, 1e-11).unwrap();
    let rss = (&b - a * &beta).norm_squared();
    (
        Array2::from_shape_fn((beta.nrows(), beta.ncols()), |(i, j)| beta[(i, j)]),
        rank,
        rss,
    )
}
#[test]
fn iid_spikes_and_row_exclusion_match_r_coefficients_rss_and_rank_df() {
    let f = fixture();
    let f = &f["ols"];
    let frame = frame();
    let mut mask = vec![false; 21];
    for i in f["censored"].as_array().unwrap() {
        mask[i.as_u64().unwrap() as usize - 1] = true;
    }
    let plan = CensorPlan::new(&frame, &mask).unwrap();
    let x = DesignMatrix::with_times(
        mat(&f["x"]),
        vec!["intercept".into(), "trend".into(), "task".into()],
        Array1::from_vec(frame.samples()),
    );
    let y = Array2::from_shape_vec((21, 1), vec(&f["y"])).unwrap();
    let full = hstack_design_matrices(&[x.clone(), plan.spikes().unwrap()]).unwrap();
    let selected = plan.select_design(&x).unwrap();
    let sy = plan.select_rows(y.view()).unwrap();
    let (beta, rank, rss) = solve(&full.data, &y);
    let (sb, srank, srss) = solve(&selected.data, &sy);
    close(
        &beta,
        &Array2::from_shape_vec((6, 1), vec(&f["spike_beta"])).unwrap(),
    );
    close(
        &sb,
        &Array2::from_shape_vec((3, 1), vec(&f["selected_beta"])).unwrap(),
    );
    close(&beta.slice(s![..3, ..]).to_owned(), &sb);
    assert_eq!(rank, f["spike_rank"].as_u64().unwrap() as usize);
    assert_eq!(srank, 3);
    assert_eq!(21 - rank, selected.nrows() - srank);
    assert_eq!(21 - rank, f["df"].as_u64().unwrap() as usize);
    assert!((rss - srss).abs() < 1e-12);
    assert!((rss - f["rss"].as_f64().unwrap()).abs() < 1e-12);
    for (i, &original) in plan.retained_indices().iter().enumerate() {
        assert_eq!(selected.row_times[i], frame.samples()[original]);
    }
    // Excluding an entire run makes its intercept unidentifiable: rank, not column
    // count, determines residual df in both parameterizations.
    let mut mask = vec![false; 21];
    mask[..12].fill(true);
    let plan = CensorPlan::new(&frame, &mask).unwrap();
    let base = drift_design(&frame, DriftBasis::None, InterceptType::Runwise).unwrap();
    let full = hstack_design_matrices(&[base.clone(), plan.spikes().unwrap()]).unwrap();
    let (_, rank, _) = solve(&full.data, &y);
    let selected = plan.select_design(&base).unwrap();
    let (_, srank, _) = solve(&selected.data, &plan.select_rows(y.view()).unwrap());
    assert_eq!((rank, srank), (13, 1));
    assert_eq!(21 - rank, selected.nrows() - srank);
}
#[test]
fn invalid_inputs_and_selection_contracts() {
    let f = frame();
    let none = drift_design(&f, DriftBasis::None, InterceptType::None).unwrap();
    assert_eq!(none.data.dim(), (21, 0));
    for cutoff in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(drift_design(
            &f,
            DriftBasis::Dct {
                cutoff_seconds: cutoff
            },
            InterceptType::None
        )
        .is_err());
    }
    assert!(drift_design(
        &f,
        DriftBasis::Polynomial { degree: 9 },
        InterceptType::None
    )
    .is_err());
    assert!(CensorPlan::new(&f, &[false]).is_err());
    assert!(CensorPlan::new(&f, &[true; 21]).is_err());
    let plan = CensorPlan::new(&f, &[false; 21]).unwrap();
    assert_eq!(plan.spikes().unwrap().ncols(), 0);
    let bad = Array2::from_elem((12, 1), f64::NAN);
    let good = Array2::zeros((9, 1));
    assert!(nuisance_design(
        &f,
        &[
            NuisanceRun {
                values: bad.view(),
                names: &["a"]
            },
            NuisanceRun {
                values: good.view(),
                names: &["b"]
            }
        ]
    )
    .is_err());
    let two = Array2::zeros((12, 2));
    assert!(nuisance_design(
        &f,
        &[
            NuisanceRun {
                values: two.view(),
                names: &["a", "a"]
            },
            NuisanceRun {
                values: good.view(),
                names: &["b"]
            }
        ]
    )
    .is_err());
    assert!(plan.select_rows(array![[1.], [2.]].view()).is_err());
    let mut wrong = none;
    wrong.row_times[0] += 0.1;
    assert!(plan.select_design(&wrong).is_err());
    let mut mask = [false; 21];
    mask[1] = true;
    let plan = CensorPlan::new(&f, &mask).unwrap();
    let mut data = Array2::ones((21, 1));
    data[(1, 0)] = f64::NAN;
    assert!(plan.select_rows(data.view()).is_ok());
    data[(0, 0)] = f64::NAN;
    assert!(plan.select_rows(data.view()).is_err());
}
