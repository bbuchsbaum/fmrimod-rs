//! Tests for serialization support.

#[cfg(feature = "serde")]
mod tests {
    use fmri_hrf::error::HrfError;
    use fmri_hrf::hrf::spm_canonical;
    use fmri_hrf::penalty::PenaltyOptions;
    use fmri_hrf::regressor::{regressor, regressor_set, Regressor, RegressorSet};
    use fmri_hrf::sampling::{SamplingFrame, SamplingFrameBuilder};
    use serde_json;

    #[test]
    fn test_sampling_frame_serialization() {
        let sf1 =
            SamplingFrame::new(vec![50, 50], vec![2.0, 2.5], Some(vec![1.0, 1.5]), 0.1).unwrap();

        // Serialize to JSON
        let json = serde_json::to_string(&sf1).unwrap();

        // Deserialize back
        let sf2: SamplingFrame = serde_json::from_str(&json).unwrap();

        // Verify round-trip
        assert_eq!(sf2.blocklens, sf1.blocklens);
        assert_eq!(sf2.tr, sf1.tr);
        assert_eq!(sf2.start_time, sf1.start_time);
        assert_eq!(sf2.precision, sf1.precision);
    }

    #[test]
    fn test_sampling_frame_builder_serialization() {
        let builder = SamplingFrameBuilder::default()
            .blocklens(vec![100])
            .tr(vec![2.0])
            .precision(0.05);

        // Serialize to JSON
        let json = serde_json::to_string(&builder).unwrap();

        // Deserialize back
        let builder2: SamplingFrameBuilder = serde_json::from_str(&json).unwrap();

        // Build and verify
        let sf1 = builder.build().unwrap();
        let sf2 = builder2.build().unwrap();

        assert_eq!(sf1.n_scans(), sf2.n_scans());
        assert_eq!(sf1.precision, sf2.precision);
    }

    #[test]
    fn test_regressor_data_serialization() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let reg1 = regressor(
            vec![10.0, 30.0, 50.0],
            Some(hrf),
            Some(vec![1.0]),
            Some(vec![2.0]),
            Some(25.0),
            Some(false),
        )
        .unwrap();

        // Convert to data
        let data1 = reg1.to_data();

        // Serialize to JSON
        let json = serde_json::to_string(&data1).unwrap();

        // Deserialize back
        let data2: fmri_hrf::regressor::RegressorData = serde_json::from_str(&json).unwrap();

        // Convert back to Regressor
        let reg2 = Regressor::from_data(data2).unwrap();

        // Verify round-trip
        assert_eq!(reg2.onsets, reg1.onsets);
        assert_eq!(reg2.durations, reg1.durations);
        assert_eq!(reg2.amplitudes, reg1.amplitudes);
        assert_eq!(reg2.span, reg1.span);
        assert_eq!(reg2.summate, reg1.summate);
        assert_eq!(reg2.filtered_all, reg1.filtered_all);
        assert_eq!(reg2.hrf.name(), reg1.hrf.name());
    }

    #[test]
    fn test_regressor_set_data_serialization() {
        let hrf = spm_canonical(None, None, None, None, None, Some(true));
        let rset1 = regressor_set(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
            ],
            Some(hrf),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        // Convert to data
        let data1 = rset1.to_data();

        // Serialize to JSON
        let json = serde_json::to_string(&data1).unwrap();

        // Deserialize back
        let data2: fmri_hrf::regressor::RegressorSetData = serde_json::from_str(&json).unwrap();

        // Convert back to RegressorSet
        let rset2 = RegressorSet::from_data(data2).unwrap();

        // Verify round-trip
        assert_eq!(rset2.levels, rset1.levels);
        assert_eq!(rset2.regressors.len(), rset1.regressors.len());

        for (r1, r2) in rset1.regressors.iter().zip(rset2.regressors.iter()) {
            assert_eq!(r2.onsets, r1.onsets);
            assert_eq!(r2.hrf.name(), r1.hrf.name());
        }
    }

    #[test]
    fn test_penalty_options_serialization() {
        let opts1 = PenaltyOptions { shrink_deriv: 3.5 };

        // Serialize to JSON
        let json = serde_json::to_string(&opts1).unwrap();

        // Deserialize back
        let opts2: PenaltyOptions = serde_json::from_str(&json).unwrap();

        // Verify round-trip
        assert_eq!(opts2.shrink_deriv, opts1.shrink_deriv);
    }

    #[test]
    fn test_error_serialization() {
        let err1 = HrfError::InvalidInput("test error".to_string());

        // Serialize to JSON
        let json = serde_json::to_string(&err1).unwrap();

        // Deserialize back
        let err2: HrfError = serde_json::from_str(&json).unwrap();

        // Verify the error message is preserved
        assert_eq!(err1.to_string(), err2.to_string());
    }
}
