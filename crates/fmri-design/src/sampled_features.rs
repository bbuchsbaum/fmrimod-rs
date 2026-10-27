//! Named design matrices from prepared continuous features.
use crate::{design::DesignMatrix, types::ValidationError};
use fmri_hrf::{
    regressor::{ConvolutionMethod, SampledFeatures},
    Hrf, SamplingFrame,
};
use ndarray::Array1;
use std::collections::HashSet;

/// Convolve prepared features and attach column names and acquisition times.
///
/// Names must be nonblank and unique and are preserved verbatim. A single HRF
/// basis preserves feature names; multiple bases append `.b1`, `.b2`, etc.
/// Columns are basis-major: `a.b1, b.b1, a.b2, b.b2`. No intercept is added.
/// Row times follow [`SamplingFrame::samples`], including cumulative run offsets.
/// Run-local convolution and preprocessing are provided by [`SampledFeatures`].
/// The numeric output is moved into the design matrix without an extra copy.
///
/// Use exact output names when constructing named contrasts. Before combining
/// with other designs, ensure their column names are distinct as well.
pub fn sampled_feature_design(
    features: &SampledFeatures,
    feature_names: &[&str],
    frame: &SamplingFrame,
    hrf: &dyn Hrf,
    method: ConvolutionMethod,
) -> Result<DesignMatrix, ValidationError> {
    let expected = features.amplitudes().ncols();
    if feature_names.len() != expected {
        return Err(ValidationError::LengthMismatch {
            expected,
            actual: feature_names.len(),
        });
    }
    let mut unique = HashSet::with_capacity(expected);
    if feature_names
        .iter()
        .any(|name| name.trim().is_empty() || !unique.insert(*name))
    {
        return Err(ValidationError::InvalidParameter(
            "feature names must be nonblank and unique".into(),
        ));
    }
    let bases = hrf.nbasis();
    if bases == 0 {
        return Err(ValidationError::InvalidParameter(
            "HRF must have at least one basis column".into(),
        ));
    }
    // Normalize broadcast fields again: SamplingFrame fields are publicly mutable.
    let frame = SamplingFrame::new(
        frame.blocklens.clone(),
        frame.tr.clone(),
        Some(frame.start_time.clone()),
        frame.precision,
    )
    .map_err(|e| ValidationError::InvalidParameter(e.to_string()))?;
    let data = features
        .evaluate(&frame, hrf, method)
        .map_err(|e| ValidationError::ComputationError(e.to_string()))?;
    let times = Array1::from_vec(frame.samples());
    if data.iter().chain(times.iter()).any(|x| !x.is_finite()) {
        return Err(ValidationError::NonFiniteValues);
    }
    let names = (0..bases)
        .flat_map(|b| {
            feature_names.iter().map(move |name| {
                if bases == 1 {
                    (*name).to_owned()
                } else {
                    format!("{name}.b{}", b + 1)
                }
            })
        })
        .collect();
    Ok(DesignMatrix::with_times(data, names, times))
}
