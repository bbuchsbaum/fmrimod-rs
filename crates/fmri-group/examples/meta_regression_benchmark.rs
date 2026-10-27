//! Run after scripts/group_meta_reference.R --experiments. Validates before timing.
use fmri_group::*;
use ndarray::Array2;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::{fs, time::Instant};
#[path = "../tests/common/mod.rs"]
mod common;
struct CountAlloc;
static COUNT: AtomicBool = AtomicBool::new(false);
static CALLS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);
fn allocation(n: usize) {
    if COUNT.load(Ordering::Relaxed) {
        CALLS.fetch_add(1, Ordering::Relaxed);
        BYTES.fetch_add(n, Ordering::Relaxed);
    }
}
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        allocation(l.size());
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        allocation(l.size());
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        allocation(n);
        System.realloc(p, l, n)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static ALLOC: CountAlloc = CountAlloc;
const ROOT: &str = "evidence/group-meta-regression";
fn binary(path: &str) -> Vec<f64> {
    fs::read(path)
        .unwrap()
        .chunks_exact(8)
        .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
fn input(case: &Value, prefix: &str) -> (IndependentCohort, GroupDesign) {
    let n = case["n"].as_u64().unwrap() as usize;
    let nf = case["features"].as_u64().unwrap() as usize;
    let p = case["p"].as_u64().unwrap() as usize;
    let id = case["id"].as_str().unwrap();
    let c = common::cohort(
        Array2::from_shape_vec((n, nf), binary(&format!("{ROOT}/{prefix}-{id}-y.bin"))).unwrap(),
        Some(
            Array2::from_shape_vec((n, nf), binary(&format!("{ROOT}/{prefix}-{id}-v.bin")))
                .unwrap(),
        ),
    );
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        (0..p).map(|j| format!("x{j}")).collect(),
        Array2::from_shape_vec(
            (n, p),
            case["X"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect(),
        )
        .unwrap(),
    )
    .unwrap();
    (c, d)
}
fn method(s: &str) -> HeterogeneityEstimator {
    match s {
        "FE" => HeterogeneityEstimator::CommonEffect,
        "DL" => HeterogeneityEstimator::DerSimonianLaird,
        _ => HeterogeneityEstimator::PauleMandel,
    }
}
fn spec(d: &GroupDesign, profile: &str, target: Option<usize>) -> MetaOutputSpec {
    let indices = target.map_or_else(|| (0..d.columns().len()).collect::<Vec<_>>(), |j| vec![j]);
    MetaOutputSpec {
        contrasts: indices
            .iter()
            .map(|&j| {
                GroupContrast::named(format!("c{j}"), d, &[(d.columns()[j].as_str(), 1.)]).unwrap()
            })
            .collect(),
        uncertainty: if profile == "effects" {
            MetaUncertainty::None
        } else {
            MetaUncertainty::Marginal
        },
        p_values: profile == "inference",
        confidence_level: if profile == "inference" {
            Some(0.95)
        } else {
            None
        },
    }
}
fn collect(
    c: &IndependentCohort,
    plan: &mut PreparedMetaRegression,
) -> (Vec<f64>, MetaRegressionReceipt, usize, usize, usize) {
    let mut out = vec![];
    let mut newton = 0;
    let mut bisection = 0;
    let mut boundary = 0;
    let receipt = plan
        .fit(c, |block| {
            for row in block.features {
                assert_eq!(row.status, MetaStatus::Estimated, "{row:?}");
                let e = row.estimates.unwrap();
                let h = row.diagnostics.unwrap();
                out.extend(e.effects);
                if let Some(se) = e.standard_errors {
                    out.extend(se);
                }
                if let Some(stat) = e.statistics {
                    out.extend(stat);
                }
                if let Some(p) = e.p_values {
                    out.extend(p);
                }
                if let Some(ci) = e.confidence_intervals {
                    for bounds in ci {
                        out.extend(bounds);
                    }
                }
                out.extend([
                    h.q_fixed.expect("estimated/common-effect Q"),
                    h.tau_squared,
                    row.n as f64,
                    h.residual_df as f64,
                ]);
                newton += h.newton_steps;
                bisection += h.bisection_steps;
                boundary += usize::from(h.convergence == TauConvergence::Boundary);
            }
            Ok(())
        })
        .unwrap();
    (out, receipt, newton, bisection, boundary)
}
fn main() {
    let inputs: Vec<Value> =
        serde_json::from_str(&fs::read_to_string(format!("{ROOT}/benchmark-inputs.json")).unwrap())
            .unwrap();
    let mut records = vec![];
    for case in inputs {
        let id = case["id"].as_str().unwrap();
        let nf = case["features"].as_u64().unwrap() as usize;
        let p = case["p"].as_u64().unwrap() as usize;
        let (c, d) = input(&case, "input");
        for name in if nf <= 200 {
            vec!["FE", "DL", "PM"]
        } else {
            vec!["FE", "DL"]
        } {
            let reference = binary(&format!("{ROOT}/reference-{id}-{name}.bin"));
            for profile in ["marginal", "effects", "inference"] {
                let o = MetaRegressionOptions {
                    heterogeneity: method(name),
                    ..Default::default()
                };
                let start = Instant::now();
                let mut plan = PreparedMetaRegression::new(&d, spec(&d, profile, None), o).unwrap();
                let prepare = start.elapsed().as_secs_f64();
                let start = Instant::now();
                let (out, receipt, newton, bisection, boundary) = collect(&c, &mut plan);
                let first = start.elapsed().as_secs_f64();
                let width = match profile {
                    "effects" => p + 4,
                    "inference" => 6 * p + 4,
                    _ => 2 * p + 4,
                };
                assert_eq!(out.len(), nf * width);
                assert_eq!(reference.len(), nf * (2 * p + 4));
                let mut error = 0_f64;
                for (a, b) in out
                    .chunks_exact(width)
                    .zip(reference.chunks_exact(2 * p + 4))
                {
                    for j in 0..if profile == "effects" { p } else { 2 * p } {
                        let e = (a[j] - b[j]).abs() / (1. + b[j].abs());
                        assert!(e < 2e-8, "{id} {name} {profile}: {a:?} != {b:?}");
                        error = error.max(e);
                    }
                    for j in 0..4 {
                        let e = (a[width - 4 + j] - b[2 * p + j]).abs() / (1. + b[2 * p + j].abs());
                        assert!(e < 2e-8);
                        error = error.max(e);
                    }
                }
                let mut times = vec![];
                for _ in 0..5 {
                    let start = Instant::now();
                    std::hint::black_box(collect(&c, &mut plan));
                    times.push(start.elapsed().as_secs_f64());
                }
                let mut sorted = times.clone();
                sorted.sort_by(f64::total_cmp);
                CALLS.store(0, Ordering::Relaxed);
                BYTES.store(0, Ordering::Relaxed);
                COUNT.store(true, Ordering::Relaxed);
                std::hint::black_box(collect(&c, &mut plan));
                COUNT.store(false, Ordering::Relaxed);
                let record = json!({"id":id,"method":name,"profile":profile,"prepare_seconds":prepare,"first_seconds":first,"seconds":times,"median_seconds":sorted[2],"max_scaled_error":error,"allocation_calls":CALLS.load(Ordering::Relaxed),"allocated_bytes":BYTES.load(Ordering::Relaxed),"weighted_evaluations":receipt.weighted_evaluations,"newton_steps":newton,"bisection_steps":bisection,"boundary_features":boundary,"numeric_working_bytes":receipt.run.numeric_working_bytes});
                println!(
                    "{id} {name} {profile}: {:.6}s, error {error:.3e}, evaluations {}",
                    sorted[2], receipt.weighted_evaluations
                );
                records.push(record);
            }
        }
    }
    fs::write(
        format!("{ROOT}/rust-benchmark.json"),
        serde_json::to_string_pretty(&records).unwrap(),
    )
    .unwrap();
    let panels: Vec<Value> = serde_json::from_str(
        &fs::read_to_string(format!("{ROOT}/calibration-inputs.json")).unwrap(),
    )
    .unwrap();
    let mut calibration = vec![];
    for case in panels {
        let id = case["id"].as_str().unwrap();
        let nf = case["features"].as_u64().unwrap() as usize;
        let p = case["p"].as_u64().unwrap() as usize;
        let (c, d) = input(&case, "cal");
        for name in ["FE", "DL", "PM"] {
            for inference in if name == "FE" {
                vec![MetaInference::Normal]
            } else {
                vec![MetaInference::Normal, MetaInference::ModifiedKnappHartung]
            } {
                let label = if inference == MetaInference::Normal {
                    "z"
                } else {
                    "adhoc"
                };
                let reference = binary(&format!("{ROOT}/cal-reference-{id}-{name}-{label}.bin"));
                assert_eq!(reference.len(), nf * 6);
                let mut plan = PreparedMetaRegression::new(
                    &d,
                    spec(&d, "inference", Some(p - 1)),
                    MetaRegressionOptions {
                        heterogeneity: method(name),
                        inference,
                        ..Default::default()
                    },
                )
                .unwrap();
                let mut rejects = 0;
                let mut covered = 0;
                let mut error = 0_f64;
                let mut index = 0;
                let mut evaluations = 0;
                plan.fit(&c, |block| {
                    for row in block.features {
                        assert_eq!(
                            row.status,
                            MetaStatus::Estimated,
                            "{id} {name} {label} {row:?}"
                        );
                        let e = row.estimates.unwrap();
                        let h = row.diagnostics.unwrap();
                        evaluations += h.evaluations;
                        let values = [
                            e.effects[0],
                            e.standard_errors.as_ref().unwrap()[0],
                            e.p_values.as_ref().unwrap()[0],
                            e.confidence_intervals.as_ref().unwrap()[0][0],
                            e.confidence_intervals.as_ref().unwrap()[0][1],
                            h.tau_squared,
                        ];
                        for (a, b) in values.iter().zip(&reference[index * 6..index * 6 + 6]) {
                            let delta = (a - b).abs() / (1. + b.abs());
                            assert!(delta < 2e-8, "cal {id} {name} {label}: {a} != {b}");
                            error = error.max(delta);
                        }
                        rejects += usize::from(values[2] < 0.05);
                        covered += usize::from(values[3] <= 0. && values[4] >= 0.);
                        index += 1;
                    }
                    Ok(())
                })
                .unwrap();
                assert_eq!(index, nf);
                calibration.push(json!({"id":id,"method":name,"inference":label,"replicates":nf,"rejections":rejects,"covered":covered,"coverage":covered as f64/nf as f64,"max_scaled_error":error,"weighted_evaluations":evaluations}));
            }
        }
    }
    fs::write(
        format!("{ROOT}/rust-calibration.json"),
        serde_json::to_string_pretty(&calibration).unwrap(),
    )
    .unwrap();
}
