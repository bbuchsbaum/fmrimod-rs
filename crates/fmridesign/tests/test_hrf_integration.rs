use fmridesign::core::{DataFrame, EventModel};
use fmridesign::covariate::CovariateEvent;
use fmridesign::events::{EventFactor, EventTerm};
use fmridesign::formula::event_model;
use fmridesign::sampling::SamplingFrameExt;

#[test]
fn test_hrf_convolution_canonical() {
    // Create simple event data
    let mut data = DataFrame::new();
    data.add_column("onset", vec![0.0, 10.0, 20.0]).unwrap();
    data.add_column("duration", vec![1.0, 1.0, 1.0]).unwrap();
    data.add_str_column(
        "condition",
        vec!["A".to_string(), "B".to_string(), "A".to_string()],
    )
    .unwrap();

    // Create model with canonical HRF
    // Note: event_model creates a sampling frame starting at TR/2 by default
    // Since our events start at t=0, we need to account for this
    let model = event_model(
        "hrf(condition, 'canonical')",
        &data,
        fmridesign::sampling::SamplingFrame::from_duration(40.0, 2.0),
        2.0,
    )
    .unwrap();

    let dm = model.generate_design_matrix().unwrap();

    // Check that design matrix has expected shape
    assert_eq!(dm.shape(), &[20, 2]); // 20 timepoints, 2 conditions

    // Check that convolution produces reasonable values
    // Peak should be around 5-6 seconds after event
    let col_a = dm.column(0);
    let col_b = dm.column(1);

    // First event at t=0 should have peak somewhere between 4-8 seconds (scans 2-4)
    let peak_idx_1 = col_a
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        peak_idx_1 >= 1 && peak_idx_1 <= 5,
        "Peak at scan {} is outside expected range",
        peak_idx_1
    );

    // Column B should have values around the event at t=10
    let peak_idx_2 = col_b
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert!(
        peak_idx_2 >= 5 && peak_idx_2 <= 8,
        "Peak at scan {} is outside expected range",
        peak_idx_2
    );
}

#[test]
fn test_hrf_convolution_gamma() {
    // Create continuous event data
    let mut data = DataFrame::new();
    data.add_column("onset", vec![0.0, 5.0, 10.0]).unwrap();
    data.add_column("duration", vec![0.5, 0.5, 0.5]).unwrap();
    data.add_column("amplitude", vec![1.0, 2.0, 3.0]).unwrap();

    // Create model with gamma HRF
    let model = event_model(
        "hrf(amplitude, hrf='gamma')",
        &data,
        fmridesign::sampling::SamplingFrame::from_duration(20.0, 1.0),
        1.0,
    )
    .unwrap();
    let dm = model.generate_design_matrix().unwrap();

    // Check shape
    assert_eq!(dm.shape(), &[20, 1]); // 20 timepoints, 1 continuous predictor

    // Check that amplitude modulation works
    // The response to amplitude=3 should be larger than amplitude=1
    let col = dm.column(0);

    // Find peaks for each event
    let peak_1 = col.slice(ndarray::s![0..8]).fold(0.0f64, |a, &b| a.max(b));
    let peak_2 = col.slice(ndarray::s![5..13]).fold(0.0f64, |a, &b| a.max(b));
    let peak_3 = col
        .slice(ndarray::s![10..18])
        .fold(0.0f64, |a, &b| a.max(b));

    // Check amplitude scaling (allowing for some convolution overlap)
    assert!(peak_2 > peak_1 * 1.5); // Second peak should be ~2x first
    assert!(peak_3 > peak_1 * 2.5); // Third peak should be ~3x first
}

#[test]
fn test_hrf_types() {
    let mut data = DataFrame::new();
    data.add_column("onset", vec![5.0]).unwrap();
    data.add_column("duration", vec![1.0]).unwrap();
    data.add_str_column("event", vec!["stimulus".to_string()])
        .unwrap();

    // Test different HRF types
    let hrf_types = vec!["canonical", "spm", "gamma", "fir", "fourier"];

    for hrf_type in hrf_types {
        let formula = format!("hrf(event, hrf='{}')", hrf_type);
        let model = event_model(
            &formula,
            &data,
            fmridesign::sampling::SamplingFrame::from_duration(40.0, 2.0),
            2.0,
        )
        .unwrap();
        let dm = model.generate_design_matrix().unwrap();

        // All should produce non-zero design matrices
        assert!(
            dm.iter().any(|&v| v.abs() > 1e-10),
            "HRF {} produced all zeros",
            hrf_type
        );

        // Check reasonable values
        let max_val = dm.fold(0.0f64, |a, &b| a.max(b.abs()));
        assert!(
            max_val > 0.0 && max_val < 10.0,
            "HRF {} produced unreasonable max value: {}",
            hrf_type,
            max_val
        );
    }
}

#[test]
fn test_no_hrf_for_covariates() {
    let factor = EventFactor::new(
        "condition",
        vec![0.0, 5.0, 10.0],
        vec!["A", "B", "A"],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    let term = EventTerm::new(vec![Box::new(factor)], false).unwrap();

    let motion_x =
        CovariateEvent::new("motion_x", ndarray::Array1::from_vec(vec![0.1; 20]), None).unwrap();
    let motion_y =
        CovariateEvent::new("motion_y", ndarray::Array1::from_vec(vec![0.2; 20]), None).unwrap();

    let model = EventModel::new(
        vec![Box::new(term), Box::new(motion_x), Box::new(motion_y)],
        fmridesign::sampling::SamplingFrame::from_duration(40.0, 2.0),
    )
    .unwrap();
    let dm = model.generate_design_matrix().unwrap();

    // Should have 4 columns: 2 conditions + 2 motion parameters
    assert_eq!(dm.shape(), &[20, 4]);

    // Check that motion columns are not convolved (should be constant)
    let motion_x_col = dm.column(2);
    let motion_y_col = dm.column(3);

    assert!(motion_x_col.iter().all(|&v| (v - 0.1).abs() < 1e-10));
    assert!(motion_y_col.iter().all(|&v| (v - 0.2).abs() < 1e-10));
}
