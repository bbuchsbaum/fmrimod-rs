#[allow(dead_code)]
#[path = "../tests/common/mod.rs"]
mod common;
use fmri_group::*;
use ndarray::Array2;
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
    for n in [8usize, 30] {
        for fl in [false, true] {
            let seed = 77173 + n as u64;
            let mut rng = Rng(seed);
            let mut marginal = 0usize;
            let mut family = 0usize;
            let begin = std::time::Instant::now();
            for rep in 0..reps {
                let mut y = Array2::zeros((n, 4));
                for i in 0..n {
                    let common = 0.6_f64.sqrt() * rng.normal();
                    for j in 0..4 {
                        y[(i, j)] =
                            common + 0.4_f64.sqrt() * rng.normal() + if fl { 2. } else { 0. };
                    }
                }
                let c = common::cohort(y, None);
                let d = if fl {
                    GroupDesign::new(
                        c.frame().observations().ids().to_vec(),
                        vec!["intercept".into(), "slope".into()],
                        Array2::from_shape_fn((n, 2), |(i, j)| {
                            if j == 0 {
                                1.
                            } else {
                                i as f64 / n as f64 - 0.5
                            }
                        }),
                    )?
                } else {
                    GroupDesign::intercept(&c)
                };
                let ct = GroupContrast::named(
                    "null",
                    &d,
                    &[(if fl { "slope" } else { "intercept" }, 1.)],
                )?;
                let result=permutation_max_t(&c,&d,&ct,PermutationOptions{method:if fl{PermutationMethod::FreedmanLane{blocks:vec!["all".into();n]}}else{PermutationMethod::SignFlip},schedule:PermutationSchedule::MonteCarlo{draws:999,seed:rep as u64+719},justification:"independent Gaussian subject vectors with joint feature covariance; fixed nuisance mean under FL null".into(),block_features:2,max_working_bytes:4*1024*1024,rank_tolerance:1e-12})?;
                if result.features[0].p_uncorrected <= 0.05 {
                    marginal += 1;
                }
                if result.features.iter().any(|f| f.p_max_t <= 0.05) {
                    family += 1;
                }
            }
            println!(
                "{}",
                serde_json::json!({"kind":"permutation_calibration","n":n,"freedman_lane":fl,"features":4,"replicates":reps,"draws":999,"seed":seed,"marginal_rejections":marginal,"family_rejections":family,"seconds":begin.elapsed().as_secs_f64()})
            );
        }
    }
    Ok(())
}
