fn main() -> Result<(), Box<dyn std::error::Error>> {
    use fmri_design::{formula::event_model, sampling::SamplingFrame, DataFrame};

    let mut data = DataFrame::new();
    data.add_column("onset", vec![0.13, 8.27])?;
    data.add_column("duration", vec![0.0, 0.15])?;
    data.add_str_column("condition", vec!["A".into(), "B".into()])?;
    let frame = SamplingFrame::new(vec![100], vec![2.0], None, 0.05)?;
    let model = event_model("onset ~ hrf(condition, basis=\"spmg3\")", &data, frame, 2.0)?;
    let matrix = model.generate_design_matrix()?;
    assert_eq!(matrix.dim(), (100, 6));
    assert_eq!(model.column_names().len(), matrix.ncols());
    Ok(())
}
