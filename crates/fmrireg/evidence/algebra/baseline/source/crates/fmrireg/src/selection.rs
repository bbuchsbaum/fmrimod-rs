//! Lazy assay selection bound to the original design's declared observation IDs.
use crate::{FitError, Result};
use fmri_noise::ObservationSelection;
use fmridataset::{Assay, ObservationId, ReadSelection, Selection};
/// Validate the complete original ID order before constructing a lazy selected
/// assay. Bind the resulting IDs to a GLS model prepared on the same selection.
/// This checks declared identity, not the correctness of acquisition metadata.
pub fn select_assay(
    assay: &Assay,
    original_design_ids: &[ObservationId],
    selection: &ObservationSelection,
) -> Result<Assay> {
    if original_design_ids.len() != selection.original_observations()
        || assay.observation_ids() != original_design_ids
    {
        return Err(FitError::Invalid(
            "original design/assay observation IDs or ordering differ",
        ));
    }
    let rows = selection.retained_indices();
    let read = ReadSelection::new(
        assay.source().shape(),
        Selection::Indices(rows.to_vec()),
        Selection::All,
    )
    .map_err(|_| FitError::Invalid("selected assay rows"))?;
    Assay::new(
        assay.name(),
        assay
            .source()
            .view(read)
            .map_err(|_| FitError::Invalid("selected assay source"))?,
        rows.iter()
            .map(|&i| original_design_ids[i].clone())
            .collect(),
        assay.feature_ids().to_vec(),
        assay.annotations().clone(),
    )
    .map_err(|_| FitError::Invalid("selected assay identity"))
}
