use fmrimod::design::{contrast::Contrast, types::ContrastSpec};
use fmrimod::{
    dataset::{
        Assay, AssayAnnotations, FeatureId, NumericBlock, ObservationId, SourceSpec, Values,
    },
    noise::{GapPolicy, ObservationSelection, Pooling},
    regression::{
        dataset::{BlockOptions, BoundModel},
        selected_estimation::{estimate_selected_ar, SelectedArOptions},
        selection::select_assay,
    },
};
// Complete first-level design, selected AR(2) estimation and streamed contrasts.
use fmrimod::design::{
    baseline::InterceptType,
    design::{hstack_design_matrices, DesignMatrix},
    nuisance::*,
    sampled_features::sampled_feature_design,
};
use fmrimod::hrf::{hrf::SpmG1Hrf, regressor::*, SamplingFrame};
use ndarray::{array, Array1, Array2};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let frame = SamplingFrame::new(vec![80, 70], vec![0.8, 1.], None, 0.025)?;
    let hrf = SpmG1Hrf::new(5., 15., 1. / 120.);
    let amps = array![[1.], [1.], [1.], [1.]];
    let events = EventTable {
        onsets: &[2., 12., 3., 15.],
        durations: &[0.5],
        amplitudes: amps.view(),
        run_ids: &[0, 0, 1, 1],
    };
    let task = DesignMatrix::with_times(
        events.evaluate(&frame, &hrf, ConvolutionMethod::Auto)?,
        vec!["task".into()],
        Array1::from_vec(frame.samples()),
    );
    let f1 = Array2::from_shape_fn((140, 1), |(i, _)| (i as f64 * 0.4).sin());
    let f2 = Array2::from_shape_fn((140, 1), |(i, _)| (i as f64 * 0.3).cos());
    let features = SampledFeatures::new(
        &[
            FeatureRun {
                values: f1.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 0.5,
                },
                mask: None,
            },
            FeatureRun {
                values: f2.view(),
                grid: FeatureGrid::Uniform {
                    start: 0.,
                    step: 0.5,
                },
                mask: None,
            },
        ],
        FeatureOptions::default(),
    )?;
    let continuous =
        sampled_feature_design(&features, &["sound"], &frame, &hrf, ConvolutionMethod::Auto)?;
    let drift = drift_design(
        &frame,
        DriftBasis::Dct {
            cutoff_seconds: 32.,
        },
        InterceptType::Runwise,
    )?;
    let motion1 = Array2::from_shape_fn((80, 1), |(i, _)| 0.1 * (i as f64 * 0.7).sin());
    let motion2 = Array2::from_shape_fn((70, 1), |(i, _)| 0.1 * (i as f64 * 0.9).sin());
    let motion = nuisance_design(
        &frame,
        &[
            NuisanceRun {
                values: motion1.view(),
                names: &["motion_x"],
            },
            NuisanceRun {
                values: motion2.view(),
                names: &["motion_x"],
            },
        ],
    )?;
    let design = hstack_design_matrices(&[task, continuous, drift, motion])?;

    let mut mask = vec![false; 150];
    for i in [0, 5, 6, 43, 79, 80, 113, 149] {
        mask[i] = true;
    }
    let censor = CensorPlan::new(&frame, &mask)?;
    let selection =
        ObservationSelection::new(frame.blocklens.clone(), censor.retained_indices().to_vec())?;
    let selected_design = censor.select_design(&design)?;
    assert_eq!(selection.select(design.data.view())?, selected_design.data);
    let features = 32;
    let mut response = Array2::zeros((150, features));
    let mut rng = Rng(612731);
    for j in 0..features {
        let mut state = [0.; 2];
        for i in 0..150 {
            if i == 0 || i == 80 {
                state = [0.; 2];
                for _ in 0..256 {
                    rng.step(&mut state);
                }
            }
            response[(i, j)] =
                2. + 0.8 * design.data[(i, 0)] + 0.3 * design.data[(i, 1)] + rng.step(&mut state);
        }
    }
    let estimated = estimate_selected_ar(
        design.data.view(),
        response.view(),
        SelectedArOptions {
            selection: selection.clone(),
            gaps: GapPolicy::Preserve,
            pooling: Pooling::Global,
            order: 2,
            max_lag: 10,
            relative_tolerance: 1e-12,
        },
    )?;
    let ids: Vec<_> = (0..150)
        .map(|i| {
            ObservationId::new(format!(
                "subject-demo/run-{}/scan-{}",
                if i < 80 { 1 } else { 2 },
                if i < 80 { i } else { i - 80 }
            ))
            .unwrap()
        })
        .collect();
    let fids: Vec<_> = (0..features)
        .map(|i| FeatureId::new(format!("voxel-{i}")).unwrap())
        .collect();
    let assay = Assay::new(
        "bold",
        SourceSpec::memory(NumericBlock::new(
            [150, features],
            Values::F64(response.iter().copied().collect()),
        )?),
        ids.clone(),
        fids.clone(),
        AssayAnnotations::default(),
    )?;
    let view = select_assay(&assay, &ids, &selection)?;
    let bound = BoundModel::new(&estimated.model, view.observation_ids().to_vec())?;
    let contrast = Contrast::simple("task effect", "task").compute_weights(&design.column_names)?;
    let selected_response = selection.select(response.view())?;
    let dense = estimated.model.fit(selected_response.view())?;
    let reference = dense.t_contrast(contrast.view())?;
    let mut delivered = 0;
    bound.fit_assay_with_options(
        &view,
        BlockOptions {
            block_features: 7,
            workers: 2,
            working_bytes: 2_000_000,
        },
        |ids, fit| {
            assert_eq!(ids, &fids[delivered..delivered + ids.len()]);
            let result = fit.t_contrast(contrast.view())?;
            for (j, &t) in result.statistic.iter().enumerate() {
                assert!((t - reference.statistic[delivered + j]).abs() < 1e-9);
            }
            delivered += ids.len();
            Ok(())
        },
    )?;
    assert_eq!(delivered, features);
    println!(
        "{}",
        serde_json::json!({"original_scans":150,"retained_scans":selection.retained_indices().len(),"column_names":design.column_names,"rank":estimated.diagnostics.design_rank,"residual_df":dense.residual_df(),"estimated_ar":estimated.diagnostics.coefficients,"delivered_features":delivered,"task_effects":reference.effect.to_vec(),"task_t":reference.statistic.to_vec(),"streamed_equals_dense":true})
    );
    Ok(())
}
struct Rng(u64);
impl Rng {
    fn normal(&mut self) -> f64 {
        let mut u = [0.; 2];
        for v in &mut u {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((self.0 >> 11) as f64 + 0.5) / 9007199254740992.;
        }
        (-2. * u[0].ln()).sqrt() * (std::f64::consts::TAU * u[1]).cos()
    }
    fn step(&mut self, state: &mut [f64; 2]) -> f64 {
        let next = 0.4 * state[0] - 0.15 * state[1] + self.normal();
        state[1] = state[0];
        state[0] = next;
        next
    }
}
