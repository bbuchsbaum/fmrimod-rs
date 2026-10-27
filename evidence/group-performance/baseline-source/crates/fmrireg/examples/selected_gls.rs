//! Known AR(2), original-grid censoring, ID-checked lazy selection and block GLS.
use fmri_noise::{ArStart, GapPolicy, ObservationSelection, SelectedArPlan};
use fmridataset::{
    Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
};
use fmrireg::{
    dataset::{BlockOptions, BoundModel},
    selection::select_assay,
    GlsModel,
};
use ndarray::{array, Array2};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let runs = vec![40, 32];
    let n = 72;
    let features = 7;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => {
            if i < 40 {
                1.
            } else {
                0.
            }
        }
        1 => {
            if i >= 40 {
                1.
            } else {
                0.
            }
        }
        _ => ((i as f64) * 0.3).sin(),
    });
    let y = Array2::from_shape_fn((n, features), |(i, j)| {
        2. + x[(i, 2)] * (0.5 + j as f64 / 10.) + 0.2 * ((i * 3 + j) as f64).cos()
    });
    let retained = (0..n)
        .filter(|i| ![0, 5, 6, 39, 40, 43, 71].contains(i))
        .collect();
    let selection = ObservationSelection::new(runs, retained)?;
    // Preserve uses the original run covariance across gaps. Restart would declare
    // independent segments at each gap; neither choice is implied by censoring.
    let noise = SelectedArPlan::new(
        selection.clone(),
        vec![vec![0.45, -0.12], vec![0.3, -0.08]],
        ArStart::Stationary,
        GapPolicy::Preserve,
    )?;
    let selected_x = selection.select(x.view())?;
    let model = GlsModel::new(selected_x.view(), noise, 1e-12)?;
    let ids: Vec<_> = (0..n)
        .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
        .collect();
    let feature_ids = (0..features)
        .map(|j| FeatureId::new(format!("voxel-{j}")).unwrap())
        .collect();
    let assay = Assay::new(
        "bold",
        SourceSpec::memory(NumericBlock::new(
            [n, features],
            Values::F64(y.iter().copied().collect()),
        )?),
        ids.clone(),
        feature_ids,
        AssayAnnotations::default(),
    )?;
    let selected_assay = select_assay(&assay, &ids, &selection)?;
    let bound = BoundModel::new(&model, selected_assay.observation_ids().to_vec())?;
    bound.fit_assay_with_options(
        &selected_assay,
        BlockOptions {
            block_features: 3,
            workers: 2,
            working_bytes: 1_000_000,
        },
        |ids, fit| {
            let contrast = fit.t_contrast(array![0., 0., 1.].view())?;
            println!(
                "{} features; residual df {}; task t {:?}",
                ids.len(),
                fit.residual_df(),
                contrast.statistic.to_vec()
            );
            Ok(())
        },
    )?;
    println!(
        "Retained {} of {n} original scans",
        selection.retained_indices().len()
    );
    Ok(())
}
