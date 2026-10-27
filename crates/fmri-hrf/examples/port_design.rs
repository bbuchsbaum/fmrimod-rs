fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmri_hrf::{hrf::spm_hrf_with_derivatives, regressor::ConditionEvents, SamplingFrame};
    let frame = SamplingFrame::new(vec![100, 80], vec![2.0], None, 0.05)?;
    let conditions = vec!["A".to_string(), "B".to_string()];
    let hrf = spm_hrf_with_derivatives();
    let design = ConditionEvents::new(&[0.13, 4.27], &conditions, &[0, 1])
        .with_durations(&[0.0, 0.15])
        .evaluate(&frame, hrf.as_ref())?;
    assert_eq!(design.data.dim(), (180, 6));
    Ok(())
}
