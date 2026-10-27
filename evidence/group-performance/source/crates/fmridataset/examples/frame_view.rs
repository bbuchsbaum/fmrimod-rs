use fmridataset::*;

fn main() -> Result<()> {
    let observations = Axis::new(
        (1..=6)
            .map(|i| ObservationId::new(format!("volume-{i}")))
            .collect::<Result<_>>()?,
        vec![(
            "run".into(),
            Column::Text((0..6).map(|i| Some(format!("run-{}", i / 3 + 1))).collect()),
        )],
    )?;
    let space = VolumeSpace::new(
        [2, 2, 1],
        [
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ],
        None,
        Some("toy".into()),
    )?;
    // R's matrix(1:24, nrow=6), represented explicitly in row-major order.
    let values = (0..6)
        .flat_map(|r| (0..4).map(move |c| (1 + r + 6 * c) as f64))
        .collect();
    let frame = FmriFrame::from_source(
        "signal",
        SourceSpec::memory(NumericBlock::new([6, 4], Values::F64(values))?),
        observations,
        FeatureAxis::from_space(FeatureSpace::Volume(space))?,
    )?;
    let view = frame.view_ids(
        &[
            ObservationId::new("volume-6")?,
            ObservationId::new("volume-1")?,
        ],
        &[FeatureId::new("voxel-4")?, FeatureId::new("voxel-2")?],
    )?;
    let context = ExecutionContext::new(MemoryBudget::new(32));
    let block = view.collect("signal", &context)?;
    assert_eq!(block.values(), &Values::F64(vec![24., 12., 19., 7.]));
    println!("{:?}: {:?}", block.shape(), block.values());
    Ok(())
}
