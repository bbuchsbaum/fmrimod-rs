//! Estimate shared AR(1) from representative retained OLS residuals, then stream GLS.
use fmri_noise::{GapPolicy, ObservationSelection, Pooling};
use fmridataset::{
    Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
};
use fmrireg::{
    dataset::{BlockOptions, BoundModel},
    selected_estimation::{estimate_selected_ar1, SelectedAr1Options},
    selection::select_assay,
};
use ndarray::{array, Array2};
struct Rng(u64);
impl Rng {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / 9007199254740992.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.uniform().ln()).sqrt() * (2. * std::f64::consts::PI * self.uniform()).cos()
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 150;
    let features = 24;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
        0 => {
            if i < 80 {
                1.
            } else {
                0.
            }
        }
        1 => {
            if i >= 80 {
                1.
            } else {
                0.
            }
        }
        _ => ((i as f64) * 0.2).sin(),
    });
    let mut y = Array2::zeros((n, features));
    let mut rng = Rng(92371);
    for j in 0..features {
        let mut previous = 0.;
        for i in 0..n {
            previous = if i == 0 || i == 80 {
                rng.normal() / (1.0_f64 - 0.4 * 0.4).sqrt()
            } else {
                0.4 * previous + rng.normal()
            };
            y[(i, j)] = 2. + 0.8 * x[(i, 2)] + previous;
        }
    }
    let selection =
        ObservationSelection::new(vec![80, 70], (0..n).filter(|i| i % 11 != 0).collect())?;
    let estimated = estimate_selected_ar1(
        x.view(),
        y.view(),
        SelectedAr1Options {
            selection: selection.clone(),
            gaps: GapPolicy::Preserve,
            pooling: Pooling::Global,
            max_lag: 8,
            relative_tolerance: 1e-12,
        },
    )?;
    println!(
        "AR coefficients {:?}; correction condition {:.3}; lag budget {}",
        estimated.diagnostics.coefficients,
        estimated.diagnostics.correction_condition,
        estimated.diagnostics.correction_lags
    );
    let ids: Vec<_> = (0..n)
        .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
        .collect();
    let fids = (0..features)
        .map(|j| FeatureId::new(format!("voxel-{j}")).unwrap())
        .collect();
    let assay = Assay::new(
        "bold",
        SourceSpec::memory(NumericBlock::new(
            [n, features],
            Values::F64(y.iter().copied().collect()),
        )?),
        ids.clone(),
        fids,
        AssayAnnotations::default(),
    )?;
    let view = select_assay(&assay, &ids, &selection)?;
    let bound = BoundModel::new(&estimated.model, view.observation_ids().to_vec())?;
    bound.fit_assay_with_options(
        &view,
        BlockOptions {
            block_features: 7,
            workers: 2,
            working_bytes: 1_000_000,
        },
        |ids, fit| {
            let task = fit.t_contrast(array![0., 0., 1.].view())?;
            println!(
                "{} features; df {}; task effects {:?}",
                ids.len(),
                fit.residual_df(),
                task.effect.to_vec()
            );
            Ok(())
        },
    )?;
    Ok(())
}
