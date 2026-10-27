//! Reproducible release performance and adverse statistical scenarios.
use fmri_data::{ExecutionContext, MemoryBudget, NumericBlock, SourceSpec, Values};
use fmri_noise::{GapPolicy, ObservationSelection, Pooling};
use fmri_reg::{
    dataset::{fit_blocks_with_options, BlockOptions},
    selected_estimation::{estimate_selected_ar, SelectedArOptions},
};
use ndarray::{array, s, Array2};
use std::time::Instant;
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 + 0.5) / 9007199254740992.
    }
    fn normal(&mut self) -> f64 {
        (-2. * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
    }
}
fn simulate(n: usize, features: usize, phi: &[f64], seed: u64) -> Array2<f64> {
    let mut rng = Rng(seed);
    let mut y = Array2::zeros((n, features));
    for j in 0..features {
        let mut state = vec![0.; phi.len()];
        for i in 0..n {
            if i == 0 || i == n / 2 {
                state.fill(0.);
                for _ in 0..2048 {
                    let next =
                        state.iter().zip(phi).map(|(a, b)| a * b).sum::<f64>() + rng.normal();
                    state.rotate_right(1);
                    state[0] = next;
                }
            }
            let value = state.iter().zip(phi).map(|(a, b)| a * b).sum::<f64>() + rng.normal();
            state.rotate_right(1);
            state[0] = value;
            y[(i, j)] = value;
        }
    }
    y
}
fn design(n: usize, p: usize) -> Array2<f64> {
    Array2::from_shape_fn((n, p), |(i, j)| match j {
        0 => {
            if i < n / 2 {
                1.
            } else {
                0.
            }
        }
        1 => {
            if i >= n / 2 {
                1.
            } else {
                0.
            }
        }
        _ => ((i as f64) * 0.071 * (j - 1) as f64).sin(),
    })
}
fn coefficients(order: usize) -> Vec<f64> {
    match order {
        1 => vec![0.4],
        2 => vec![0.45, -0.18],
        3 => vec![0.4, -0.16, 0.08],
        _ => vec![0.4, -0.15, 0.08, -0.04],
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "stress") {
        return stress();
    }
    let order: usize = args.get(1).unwrap().parse()?;
    let n: usize = args.get(2).unwrap().parse()?;
    let features: usize = args.get(3).unwrap().parse()?;
    let block: usize = args.get(4).unwrap().parse()?;
    let workers: usize = args.get(5).unwrap().parse()?;
    let x = design(n, 8);
    let phi = coefficients(order);
    let selection = ObservationSelection::new(
        vec![n / 2, n - n / 2],
        (0..n).filter(|i| i % 11 != 0).collect(),
    )?;
    let noise = simulate(n, 32, &phi, 80317);
    let begin = Instant::now();
    let estimated = estimate_selected_ar(
        x.view(),
        noise.view(),
        SelectedArOptions {
            selection: selection.clone(),
            gaps: GapPolicy::Preserve,
            pooling: Pooling::Global,
            order,
            max_lag: 12,
            relative_tolerance: 1e-12,
        },
    )?;
    let preparation = begin.elapsed().as_secs_f64();
    let y =
        simulate(n, features, &phi, 79311) + x.column(2).to_owned().insert_axis(ndarray::Axis(1));
    let selected = selection.select(y.view())?;
    let start = Instant::now();
    let expected = estimated.model.fit(selected.view())?;
    let first_dense = start.elapsed().as_secs_f64();
    let source = SourceSpec::memory(NumericBlock::new(
        [selected.nrows(), features],
        Values::F64(selected.iter().copied().collect()),
    )?);
    let context = ExecutionContext::new(MemoryBudget::new(usize::MAX));
    let mut reader = context.open(&source);
    let mut timings = Vec::new();
    let mut max_error = 0.0_f64;
    let mut outputs = 0;
    for _ in 0..4 {
        let begin = Instant::now();
        let mut delivered = 0;
        fit_blocks_with_options(
            &estimated.model,
            &mut reader,
            BlockOptions {
                block_features: block,
                workers,
                working_bytes: 256 * 1024 * 1024,
            },
            |range, fit| {
                for (a, b) in fit
                    .coefficients()
                    .iter()
                    .zip(expected.coefficients().slice(s![.., range.clone()]).iter())
                {
                    max_error = max_error.max((a - b).abs() / (1. + b.abs()));
                }
                for (a, b) in fit
                    .residual_variance()
                    .iter()
                    .zip(expected.residual_variance().slice(s![range.clone()]).iter())
                {
                    max_error = max_error.max((a - b).abs() / (1. + b.abs()));
                }
                delivered += range.len();
                Ok(())
            },
        )?;
        assert_eq!(delivered, features);
        outputs = delivered;
        timings.push(begin.elapsed().as_secs_f64());
    }
    assert!(max_error < 2e-10);
    let warm = &timings[1..];
    let mut sorted = warm.to_vec();
    sorted.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({"order":order,"original_rows":n,"retained_rows":selected.nrows(),"regressors":8,"features":features,"block":block,"workers":workers,"prepare_seconds":preparation,"first_dense_seconds":first_dense,"first_stream_seconds":timings[0],"warm_seconds":warm,"median_warm_seconds":sorted[1],"features_per_second":features as f64/sorted[1],"max_relative_error":max_error,"delivered":outputs,"coefficients":estimated.diagnostics.coefficients,"working_budget_bytes":256*1024*1024})
    );
    Ok(())
}
fn stress() -> Result<(), Box<dyn std::error::Error>> {
    for order in 1..=4 {
        for scenario in [
            "baseline",
            "persistent",
            "heavy_censor",
            "short_runs",
            "few_features",
        ] {
            let n = if scenario == "short_runs" { 40 } else { 180 };
            let features = if scenario == "few_features" { 1 } else { 16 };
            let phi = if scenario == "persistent" {
                let mut p = vec![0.; order];
                p[0] = 0.95;
                p
            } else {
                coefficients(order)
            };
            let keep: Vec<_> = (0..n)
                .filter(|i| {
                    if scenario == "heavy_censor" {
                        i % 5 < 2
                    } else {
                        i % 11 != 0
                    }
                })
                .collect();
            let selection = ObservationSelection::new(vec![n / 2, n - n / 2], keep)?;
            let x = design(n, 3);
            let mut reasons = std::collections::BTreeMap::new();
            let mut successes = 0;
            let mut covered = 0;
            let mut bias = vec![0.; order];
            let mut residual_lag = 0.;
            for rep in 0..60 {
                let errors = simulate(n, features + 1, &phi, 451231 + order as u64 * 1000 + rep);
                let y = errors.slice(s![.., ..features]);
                let estimate = estimate_selected_ar(
                    x.view(),
                    y,
                    SelectedArOptions {
                        selection: selection.clone(),
                        gaps: GapPolicy::Preserve,
                        pooling: Pooling::Global,
                        order,
                        max_lag: 12,
                        relative_tolerance: 1e-12,
                    },
                );
                let estimated = match estimate {
                    Ok(e) => e,
                    Err(e) => {
                        *reasons.entry(e.to_string()).or_insert(0) += 1;
                        continue;
                    }
                };
                let target = errors.slice(s![.., features..]).to_owned()
                    + x.column(2).to_owned().insert_axis(ndarray::Axis(1)) * 0.7;
                let target = selection.select(target.view())?;
                let xs = selection.select(x.view())?;
                let fit = estimated.model.fit(target.view())?;
                let t = fit.t_contrast(array![0., 0., 1.].view())?;
                let critical = statrs::distribution::ContinuousCDF::inverse_cdf(
                    &statrs::distribution::StudentsT::new(0., 1., t.df as f64)?,
                    0.975,
                );
                if (t.effect[0] - 0.7).abs() <= critical * t.standard_error[0] {
                    covered += 1;
                }
                use fmri_noise::WhiteningPlan;
                let residual = estimated
                    .model
                    .noise_plan()
                    .apply((&target - &xs.dot(fit.coefficients())).view())?;
                let denom = residual.iter().map(|v| v * v).sum::<f64>();
                residual_lag += (1..residual.nrows())
                    .map(|i| residual[(i, 0)] * residual[(i - 1, 0)])
                    .sum::<f64>()
                    / denom;
                for k in 0..order {
                    bias[k] += estimated.diagnostics.coefficients[0][k] - phi[k];
                }
                successes += 1;
            }
            let coverage = if successes > 0 {
                Some(covered as f64 / successes as f64)
            } else {
                None
            };
            let mean_bias: Vec<_> = bias
                .into_iter()
                .map(|b| {
                    if successes > 0 {
                        Some(b / successes as f64)
                    } else {
                        None
                    }
                })
                .collect();
            println!(
                "{}",
                serde_json::json!({"order":order,"scenario":scenario,"original_rows":n,"retained_rows":selection.retained_indices().len(),"noise_features":features,"reps":60,"successes":successes,"failures":reasons,"conditional_coverage":coverage,"mean_coefficient_bias":mean_bias,"mean_residual_lag1":if successes>0{Some(residual_lag/successes as f64)}else{None},"true_coefficients":phi})
            );
        }
    }
    Ok(())
}
