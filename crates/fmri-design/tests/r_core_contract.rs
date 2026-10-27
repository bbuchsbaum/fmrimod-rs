use fmri_design::baseline::{baseline_model, BasisType, InterceptType, SamplingInfo};
use fmri_design::convolve::{convolve, convolve_at_times, ConvolveOptions};
use fmri_design::events::{EventFactor, EventVariable, HrfEvent};
use fmri_design::hrf::HrfSpec;
use fmri_design::prelude::*;
use fmri_design::sampling::SamplingFrame;
use ndarray::{array, Array1, Array2};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/r_core_contract.json")).unwrap()
}
fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn frame(f: &Value) -> SamplingFrame {
    let x = &f["frame"];
    SamplingFrame::new(
        x["blocklens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect::<Vec<_>>(),
        floats(&x["tr"]),
        Some(floats(&x["start_time"])),
        x["precision"].as_f64().unwrap(),
    )
    .unwrap()
}
fn compare(got: &Array2<f64>, expected: &Value, label: &str) {
    let rows = expected.as_array().unwrap();
    assert_eq!(
        got.dim(),
        (rows.len(), rows[0].as_array().unwrap().len()),
        "{label}"
    );
    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row.as_array().unwrap().iter().enumerate() {
            let expected = value.as_f64().unwrap();
            assert!(
                (got[[i, j]] - expected).abs() < 2e-9 * (1.0 + expected.abs()),
                "{label} [{i},{j}]: {} != {expected}",
                got[[i, j]]
            );
        }
    }
}

#[test]
fn formula_event_designs_match_current_r_values_and_basis_order() {
    let f = fixture();
    let frame = frame(&f);
    for (actual, expected) in frame.samples().iter().zip(floats(&f["frame"]["samples"])) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let rows = f["events"].as_array().unwrap();
    let mut data = fmri_design::DataFrame::new();
    for name in ["onset", "duration"] {
        data.add_column(
            name,
            rows.iter().map(|row| row[name].as_f64().unwrap()).collect(),
        )
        .unwrap();
    }
    data.add_str_column(
        "condition",
        rows.iter()
            .map(|row| row["condition"].as_str().unwrap().to_string())
            .collect(),
    )
    .unwrap();
    data.add_str_column(
        "block",
        rows.iter()
            .map(|row| row["block"].as_str().unwrap().to_string())
            .collect(),
    )
    .unwrap();
    for design in f["designs"].as_array().unwrap() {
        let basis = design["basis"].as_str().unwrap();
        let model = fmri_design::formula::event_model(
            &format!("onset ~ hrf(condition, \"{basis}\")"),
            &data,
            frame.clone(),
            1.5,
        )
        .unwrap();
        let values = model.generate_design_matrix().unwrap();
        compare(&values, &design["expected"], basis);
        // Rust omits the redundant term prefix; basis and condition order agree.
        let expected: Vec<_> = design["names"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| {
                v.as_str()
                    .unwrap()
                    .strip_prefix("condition_")
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(model.column_names(), expected);
        assert_eq!(model.n_columns(), values.ncols());
        assert_eq!(model.generate_design_matrix_cached().unwrap(), values);
    }
}

#[test]
fn runwise_polynomial_baselines_match_current_r() {
    let f = fixture();
    for case in f["baselines"].as_array().unwrap() {
        let sampling = SamplingInfo::from_frame(frame(&f));
        let model = baseline_model(
            match case["basis"].as_str().unwrap_or("poly") {
                "bs" => BasisType::BSpline,
                "ns" => BasisType::NaturalSpline,
                _ => BasisType::Polynomial,
            },
            case["degree"].as_u64().unwrap() as usize,
            &sampling,
            InterceptType::Runwise,
            None,
        )
        .unwrap();
        compare(
            &model.design_matrix().unwrap(),
            &case["expected"],
            "baseline",
        );
    }
}

#[test]
fn duplicate_impulses_add_and_short_blocks_remain_blocks() {
    let event = EventVariable::new(
        "amplitude",
        vec![0.13, 0.13],
        vec![1.0, 2.0],
        vec![0.0, 0.0],
    )
    .unwrap()
    .with_center(false);
    let hrf = fmri_hrf::hrf::SpmG3Hrf::new(5.0, 15.0, 0.0833);
    let grid = array![2.13, 5.13, 9.13];
    let expected = hrf.evaluate(&array![2.0, 5.0, 9.0]).unwrap() * 3.0;
    let got = convolve_at_times(&event, &hrf, &grid, 24.0, 0.01, "loop").unwrap();
    for (a, b) in got.iter().zip(&expected) {
        assert!((a - b).abs() < 1e-12);
    }
    let options = ConvolveOptions {
        total_duration: Some(15.0),
        precision: 0.01,
        ..Default::default()
    };
    assert_eq!(convolve(&event, &hrf, &options).unwrap().ncols(), 3);
}

#[test]
fn empty_runs_and_absent_conditions_keep_their_columns_without_tail_leakage() {
    let factor = EventFactor::new(
        "condition",
        vec![17.13, 0.27],
        vec!["A", "B"],
        vec![2.7, 0.15],
    )
    .unwrap()
    .with_levels(vec!["A".into(), "B".into(), "C".into()])
    .unwrap();
    let frame = SamplingFrame::new(vec![10, 7, 12], vec![2.0], Some(vec![0.0]), 0.05).unwrap();
    let event = HrfEvent::new(Box::new(factor), HrfSpec::canonical())
        .with_runs(frame.clone(), vec![0, 2])
        .unwrap();
    let values = event.design_matrix(&Array1::from(frame.samples())).unwrap();
    assert_eq!(values.dim(), (29, 3));
    assert!(values
        .slice(ndarray::s![10..17, ..])
        .iter()
        .all(|&v| v == 0.0));
    assert!(values.column(2).iter().all(|&v| v == 0.0));
    assert!(values.slice(ndarray::s![17.., 0]).iter().all(|&v| v == 0.0));
}

#[test]
fn cached_models_do_not_confuse_amplitudes() {
    let make = |amplitude| {
        let event = EventVariable::new("same", vec![1.13], vec![amplitude], vec![1.0])
            .unwrap()
            .with_center(false);
        EventModel::new(
            vec![Box::new(HrfEvent::new(
                Box::new(event),
                HrfSpec::canonical(),
            ))],
            SamplingFrame::new(vec![20], vec![1.0], Some(vec![0.0]), 0.05).unwrap(),
        )
        .unwrap()
    };
    let a = make(1.0).generate_design_matrix_cached().unwrap();
    let b = make(2.0).generate_design_matrix_cached().unwrap();
    assert!(a.iter().any(|x| x.abs() > 0.1));
    for (a, b) in a.iter().zip(&b) {
        assert!((2.0 * a - b).abs() < 1e-12);
    }
}

#[test]
fn signed_modulators_and_factor_interactions_match_r() {
    let f = fixture();
    let mut data = fmri_design::DataFrame::new();
    let rows = f["modulated_events"].as_array().unwrap();
    for name in ["onset", "duration", "rating"] {
        data.add_column(
            name,
            rows.iter().map(|row| row[name].as_f64().unwrap()).collect(),
        )
        .unwrap();
    }
    for name in ["condition", "block"] {
        data.add_str_column(
            name,
            rows.iter()
                .map(|row| row[name].as_str().unwrap().to_string())
                .collect(),
        )
        .unwrap();
    }
    for case in f["modulations"].as_array().unwrap() {
        let term = case["term"].as_str().unwrap();
        let model = fmri_design::formula::event_model(
            &format!("onset ~ hrf({term})"),
            &data,
            frame(&f),
            1.5,
        )
        .unwrap();
        compare(
            &model.generate_design_matrix().unwrap(),
            &case["expected"],
            term,
        );
    }
}

#[test]
fn cached_convolution_preserves_sub_millisecond_timing_amplitudes_and_hrf_identity() {
    use fmri_design::convolve_cached::convolve_cached;
    let a = EventVariable::new("same", vec![0.1301], vec![1.0], vec![0.15]).unwrap();
    let b = EventVariable::new("same", vec![0.1302], vec![2.0], vec![0.15]).unwrap();
    let grid = array![1.5001, 2.3, 4.0];
    let h1 = fmri_hrf::hrf::gamma_hrf(6.0, 1.0, None);
    let h2 = fmri_hrf::hrf::gamma_hrf(8.0, 1.0, None);
    let options = ConvolveOptions::default();
    for event in [&a, &b] {
        for hrf in [&h1, &h2] {
            for grid in [&grid, &(&grid + 0.0001)] {
                let expected = convolve_at_times(
                    event,
                    hrf.as_ref(),
                    grid,
                    options.hrf_duration,
                    options.precision,
                    &options.method,
                )
                .unwrap();
                assert_eq!(
                    convolve_cached(event, hrf.as_ref(), grid, &options).unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn spline_fits_and_predictions_match_r_including_boundaries() {
    use fmri_design::basis::{BSplineBasis, NaturalSplineBasis};
    let f = fixture();
    for case in f["spline_cases"].as_array().unwrap() {
        let df = case["df"].as_u64().unwrap() as usize;
        let intercept = case["intercept"].as_bool().unwrap();
        let kind = case["kind"].as_str().unwrap();
        let basis: Box<dyn BasisFunction> = if kind == "bs" {
            Box::new(BSplineBasis::new(df).with_intercept(intercept))
        } else {
            Box::new(NaturalSplineBasis::new(df).with_intercept(intercept))
        };
        let x = Array1::from(floats(&case["x"]));
        compare(&basis.evaluate(&x.view()).unwrap(), &case["expected"], kind);
        let xnew = Array1::from(floats(&case["xnew"]));
        compare(
            &basis.evaluate(&xnew.view()).unwrap(),
            &case["predicted"],
            &format!("{kind} prediction"),
        );
    }
}

#[test]
fn sampled_convolution_rejects_ambiguous_or_invalid_time_grids() {
    use fmri_design::convolve_design::{convolve_design_matrix, ConvolveDesignOptions};
    let hrf = fmri_hrf::hrf::gamma_hrf(6.0, 1.0, None);
    let matrix = array![[1.0], [0.0], [0.0]];
    for times in [
        array![0.0, 1.0, 3.0],
        array![0.0, 0.0, 1.0],
        array![0.0, f64::NAN, 2.0],
        array![0.0, 1.0],
    ] {
        let options = ConvolveDesignOptions {
            time_points: Some(times),
            ..Default::default()
        };
        assert!(convolve_design_matrix(&matrix, hrf.as_ref(), &options).is_err());
    }
    assert!(fmri_design::convolve::get_hrf_array(hrf.as_ref(), 1e300, 1e300).is_err());
}
