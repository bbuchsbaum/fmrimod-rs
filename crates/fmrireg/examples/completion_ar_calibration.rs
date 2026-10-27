//! Seeded same-feature versus held-out feasible-GLS qualification.
use fmri_noise::{GapPolicy, ObservationSelection, Pooling};
use fmrireg::selected_estimation::{estimate_selected_ar, SelectedArOptions};
use ndarray::{array, Array2};
use statrs::distribution::{ContinuousCDF, StudentsT};
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
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reps: usize = std::env::args()
        .nth(1)
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(500);
    for scenario in ["moderate", "persistent", "short_runs", "heavy_censor"] {
        let n = if scenario == "short_runs" { 40 } else { 180 };
        let phi = if scenario == "persistent" { 0.95 } else { 0.4 };
        let selection = ObservationSelection::new(
            vec![n / 2, n - n / 2],
            (0..n)
                .filter(|i| {
                    if scenario == "heavy_censor" {
                        i % 5 < 2
                    } else {
                        i % 11 != 0
                    }
                })
                .collect(),
        )?;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| match j {
            0 => f64::from(i < n / 2),
            1 => f64::from(i >= n / 2),
            _ => (i as f64 * 0.071).sin(),
        });
        for same in [true, false] {
            let mut failures = BTreeMap::<String, usize>::new();
            let mut covered = 0usize;
            let mut success = 0usize;
            let mut bias = 0.;
            for rep in 0..reps {
                let mut rng = Rng(0x12345 + rep as u64);
                let mut y = Array2::zeros((n, 2));
                for j in 0..2 {
                    let mut state = 0.;
                    for i in 0..n {
                        if i == 0 || i == n / 2 {
                            state = 0.;
                            for _ in 0..2048 {
                                state = phi * state + rng.normal();
                            }
                        }
                        state = phi * state + rng.normal();
                        y[(i, j)] = state + 0.7 * x[(i, 2)];
                    }
                }
                let noise = y.slice(ndarray::s![.., 0..1]);
                let result = (|| -> Result<(f64, f64, usize), Box<dyn std::error::Error>> {
                    let estimated = estimate_selected_ar(
                        x.view(),
                        noise,
                        SelectedArOptions {
                            selection: selection.clone(),
                            gaps: GapPolicy::Preserve,
                            pooling: Pooling::Global,
                            order: 1,
                            max_lag: 12,
                            relative_tolerance: 1e-12,
                        },
                    )?;
                    let col = usize::from(!same);
                    let target = selection.select(y.slice(ndarray::s![.., col..col + 1]))?;
                    let fitted = estimated.model.fit(target.view())?;
                    let contrast = fitted.t_contrast(array![0., 0., 1.].view())?;
                    Ok((contrast.effect[0], contrast.standard_error[0], contrast.df))
                })();
                match result {
                    Ok((effect, se, df)) => {
                        success += 1;
                        bias += effect - 0.7;
                        let critical = StudentsT::new(0., 1., df as f64)?.inverse_cdf(0.975);
                        if (effect - 0.7).abs() <= critical * se {
                            covered += 1;
                        }
                    }
                    Err(e) => {
                        *failures.entry(e.to_string()).or_default() += 1;
                    }
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"ar_calibration","scenario":scenario,"same_feature":same,"seed_base":0x12345,"replicates":reps,"successes":success,"covered":covered,"failures":failures,"conditional_coverage":if success>0{Some(covered as f64/success as f64)}else{None},"mean_effect_bias":if success>0{Some(bias/success as f64)}else{None},"phi":phi,"order":1,"max_lag":12,"original_rows":n,"retained_rows":selection.retained_indices().len()})
            );
        }
    }
    Ok(())
}
