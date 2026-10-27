//! Qualification over independent simulated data, never a coverage guarantee.
#[path = "../tests/common/mod.rs"]
mod common;
use fmri_group::*;
use ndarray::Array2;
use std::collections::BTreeMap;
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
        (-2. * self.uniform().ln()).sqrt() * (std::f64::consts::TAU * self.uniform()).cos()
    }
}
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(500);
    for n in [8usize, 30] {
        for scenario in [
            "known_boundary",
            "known_heterogeneous",
            "estimated_df4",
            "estimated_df20",
            "high_leverage_missing",
            "skew_errors",
        ] {
            let seed = 73101 + n as u64;
            let mut rng = Rng(seed);
            let tau = if scenario == "known_boundary" {
                0.0_f64
            } else {
                0.4_f64
            };
            let df = match scenario {
                "estimated_df4" => Some(4),
                "estimated_df20" => Some(20),
                _ => None,
            };
            let p = if scenario == "high_leverage_missing" {
                2
            } else {
                1
            };
            let x = Array2::from_shape_fn((n, p), |(i, j)| {
                if j == 0 {
                    1.
                } else if i == n - 1 {
                    10.
                } else {
                    (i as f64 / (n - 1) as f64) - 0.5
                }
            });
            let mut y = Array2::zeros((n, reps));
            let mut v = Array2::zeros((n, reps));
            for j in 0..reps {
                for i in 0..n {
                    let sampling = 0.2 + (i % 7) as f64 / 10.;
                    let error = if scenario == "skew_errors" {
                        -rng.uniform().ln() - 1.
                    } else {
                        rng.normal()
                    };
                    y[(i, j)] = sampling.sqrt() * error + tau.sqrt() * rng.normal();
                    v[(i, j)] = sampling
                        * if let Some(df) = df {
                            (0..df).map(|_| rng.normal().powi(2)).sum::<f64>() / df as f64
                        } else {
                            1.
                        };
                    if scenario == "high_leverage_missing" && rng.uniform() < 0.1 {
                        y[(i, j)] = f64::NAN;
                    }
                }
            }
            let base = common::cohort(y.clone(), Some(v.clone()));
            let c = IndependentCohort::new(
                base.frame().clone(),
                base.subjects().to_vec(),
                base.estimand().clone(),
                "effect".into(),
                Some((
                    "variance".into(),
                    vec![
                        if let Some(df) = df {
                            VarianceOrigin::Estimated {
                                df: DegreesOfFreedom::Scalar(df as f64),
                                method: "independent scaled chi-square estimate".into(),
                                approximate: false,
                            }
                        } else {
                            VarianceOrigin::Known {
                                method: "simulation parameter".into(),
                            }
                        };
                        n
                    ],
                )),
                base.fit_receipts().to_vec(),
                base.admission().clone(),
            )?;
            let columns = if p == 1 {
                vec!["intercept".into()]
            } else {
                vec!["intercept".into(), "slope".into()]
            };
            let d = GroupDesign::new(c.frame().observations().ids().to_vec(), columns, x.clone())?;
            let term = if p == 1 { "intercept" } else { "slope" };
            let outputs = MetaOutputSpec {
                contrasts: vec![GroupContrast::named("target", &d, &[(term, 1.)])?],
                uncertainty: MetaUncertainty::Marginal,
                p_values: true,
                confidence_level: Some(0.95),
            };
            let mut methods = vec![];
            for heterogeneity in [
                HeterogeneityEstimator::CommonEffect,
                HeterogeneityEstimator::DerSimonianLaird,
                HeterogeneityEstimator::PauleMandel,
            ] {
                for inference in [MetaInference::Normal, MetaInference::ModifiedKnappHartung] {
                    let mut options = MetaRegressionOptions {
                        heterogeneity,
                        inference,
                        ..Default::default()
                    };
                    options.execution.missingness = Missingness::OmitObservation;
                    let mut success = 0usize;
                    let mut covered = 0usize;
                    let mut failures = BTreeMap::new();
                    let mut estimates = vec![];
                    fit_meta_regression(&c, &d, outputs.clone(), options, |b| {
                        for f in b.features {
                            if let Some(e) = f.estimates {
                                success += 1;
                                let ci = e.confidence_intervals.as_ref().unwrap()[0];
                                if ci[0] <= 0. && ci[1] >= 0. {
                                    covered += 1;
                                }
                                estimates.push(serde_json::json!({"effect":e.effects[0],"se":e.standard_errors.unwrap()[0],"tau":f.diagnostics.unwrap().tau_squared}));
                            } else {
                                *failures.entry(format!("{:?}", f.status)).or_insert(0usize) += 1;
                                estimates.push(serde_json::Value::Null);
                            }
                        }
                        Ok(())
                    })?;
                    methods.push(serde_json::json!({"method":format!("{heterogeneity:?}"),"inference":format!("{inference:?}"),"successes":success,"covered":covered,"failures":failures,"estimates":estimates}));
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"group_calibration","scenario":scenario,"n":n,"p":p,"replicates":reps,"seed":seed,"tau_squared":tau,"sampling_df":df,"x":x.iter().copied().collect::<Vec<_>>(),"y":y.iter().map(|x|if x.is_finite(){Some(*x)}else{None}).collect::<Vec<_>>(),"v":v.iter().copied().collect::<Vec<_>>(),"methods":methods})
            );
        }
    }
    Ok(())
}
