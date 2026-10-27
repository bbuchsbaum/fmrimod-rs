//! Reproducible null calibration and end-to-end timing; synthetic inputs only.
#[allow(dead_code, unused_imports)]
#[path = "../tests/repeated_numerics.rs"]
mod fixtures;
use fmri_group::*;
use ndarray::{array, Array2};
use std::{collections::BTreeMap, time::Instant};
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
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(200);
    for ns in [8usize, 30] {
        for (scenario_id, scenario) in [
            "ri_boundary",
            "ri_interior",
            "slope_boundary",
            "slope_interior",
        ]
        .iter()
        .enumerate()
        {
            let slope = scenario.starts_with("slope");
            let boundary = scenario.ends_with("boundary");
            let n = ns * 3;
            let p = if slope { 2 } else { 1 };
            let t = (0..n).map(|i| (i % 3) as f64 - 1.).collect::<Vec<_>>();
            let x = Array2::from_shape_fn((n, p), |(i, j)| if j == 0 { 1. } else { t[i] });
            let sampling = array![[0.3, 0.05, 0.05], [0.05, 0.3, 0.05], [0.05, 0.05, 0.3]];
            let model = RepeatedModel {
                random_effects: if slope {
                    RandomEffects::InterceptSlope
                } else {
                    RandomEffects::Intercept
                },
                residual_variance: if slope {
                    ResidualVariance::Estimate
                } else {
                    ResidualVariance::FixedZero
                },
                time: if slope { Some(t.clone()) } else { None },
                max_evaluations: 4096,
                ..Default::default()
            };
            let seed = 93217 + ns as u64 * 100 + scenario_id as u64;
            let mut rng = Rng(seed);
            let mut successes = 0usize;
            let mut covered = vec![0usize; p];
            let mut failures = BTreeMap::new();
            let mut samples = vec![];
            let mut prepare_ms = 0.;
            let mut fit_ms = 0.;
            let mut total_evaluations = 0usize;
            for rep in 0..reps {
                let mut y = vec![0.; n];
                for subject in 0..ns {
                    let a = rng.normal();
                    let b = rng.normal();
                    let common = rng.normal() * 0.05_f64.sqrt();
                    let b0 = if boundary { 0. } else { 0.4_f64.sqrt() * a };
                    let b1 = if boundary || !slope {
                        0.
                    } else {
                        0.1 / 0.4_f64.sqrt() * a + (0.2_f64 - 0.1 * 0.1 / 0.4).sqrt() * b
                    };
                    let extra = if slope && !boundary { 0.1_f64 } else { 0. };
                    for visit in 0..3 {
                        y[subject * 3 + visit] = b0
                            + b1 * t[subject * 3 + visit]
                            + common
                            + (0.25 + extra).sqrt() * rng.normal();
                    }
                }
                let now = Instant::now();
                let plan =
                    fixtures::prepared(y.clone(), 3, x.clone(), sampling.clone(), model.clone())?;
                prepare_ms += now.elapsed().as_secs_f64() * 1000.;
                let now = Instant::now();
                let result = plan.fit_feature(0)?;
                fit_ms += now.elapsed().as_secs_f64() * 1000.;
                if let Some(ref d) = result.diagnostics {
                    total_evaluations += d.evaluations;
                }
                if let Some(e) = result.estimates {
                    successes += 1;
                    for (j, k) in covered.iter_mut().enumerate() {
                        if e.beta[j].abs() <= 1.959963984540054 * e.beta_covariance[(j, j)].sqrt() {
                            *k += 1;
                        }
                    }
                    if rep < 10 {
                        samples.push(serde_json::json!({"replicate":rep,"y":y,"beta":e.beta,"covariance":e.beta_covariance.iter().copied().collect::<Vec<_>>(),"g":e.g.iter().copied().collect::<Vec<_>>(),"eta":e.eta,"objective":e.objective}));
                    }
                } else {
                    *failures
                        .entry(format!("{:?}", result.status))
                        .or_insert(0usize) += 1;
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"repeated_calibration","scenario":scenario,"subjects":ns,"observations":n,"p":p,"replicates":reps,"seed":seed,"successes":successes,"covered":covered,"failures":failures,"prepare_ms":prepare_ms,"fit_ms":fit_ms,"evaluations":total_evaluations,"samples":samples,"likelihood":"REML","normal_plugin":true})
            );
        }
    }
    Ok(())
}
