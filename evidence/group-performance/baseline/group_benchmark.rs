//! Run after benchmarks/reference.R. Validates every output before reporting timing.
use fmri_group::*;
use fmridataset::*;
use ndarray::Array2;
use serde_json::{json, Value};
use std::{fs, time::Instant};
fn cohort(n: usize, nf: usize, y: Vec<f64>, v: Vec<f64>) -> IndependentCohort {
    let rows = (0..n)
        .map(|i| Subject {
            observation: ObservationId::new(format!("o{i}")).unwrap(),
            dataset: "synthetic".into(),
            participant: format!("s{i}"),
            measurement: "visit1".into(),
        })
        .collect::<Vec<_>>();
    let ids = rows
        .iter()
        .map(|s| s.observation.clone())
        .collect::<Vec<_>>();
    let features = FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new(
            "fixture",
            (0..nf)
                .map(|j| FeatureId::new(format!("f{j}")).unwrap())
                .collect(),
        )
        .unwrap(),
    ))
    .unwrap();
    let assays = [("effect", y), ("variance", v)]
        .into_iter()
        .map(|(name, data)| {
            Assay::new(
                name,
                SourceSpec::memory(NumericBlock::new([n, nf], Values::F64(data)).unwrap()),
                ids.clone(),
                features.axis().ids().to_vec(),
                AssayAnnotations::default(),
            )
            .unwrap()
        })
        .collect();
    let frame = FmriFrame::new(
        Axis::new(ids, vec![]).unwrap(),
        features,
        assays,
        "effect",
        Metadata::new(),
    )
    .unwrap();
    IndependentCohort::new(
        frame,
        rows,
        Estimand {
            id: "effect".into(),
            definition: "synthetic response".into(),
            units: "arbitrary".into(),
        },
        "effect".into(),
        Some((
            "variance".into(),
            vec![
                VarianceOrigin::Known {
                    method: "simulation".into()
                };
                n
            ],
        )),
        vec![
            FitReceipt {
                model: "simulation".into(),
                noise: "independent".into(),
                selection: "all".into()
            };
            n
        ],
        Admission {
            target: "synthetic".into(),
            evidence: "common generated support".into(),
        },
    )
    .unwrap()
}
fn run(c: &IndependentCohort, method: &str, block: usize, p: usize) -> (Vec<f64>, RunReceipt) {
    let n = c.subjects().len();
    let names = (0..p).map(|j| format!("x{j}")).collect::<Vec<_>>();
    let d = GroupDesign::new(
        c.frame().observations().ids().to_vec(),
        names.clone(),
        Array2::from_shape_fn((n, p), |(i, j)| {
            if j == 0 {
                1.
            } else {
                (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / n as f64).cos()
            }
        }),
    )
    .unwrap();
    let ct = GroupContrast::named("mean", &d, &[(&names[p - 1], 1.)]).unwrap();
    let mut out = Vec::with_capacity(c.frame().shape()[1] * 7);
    let sink = |b: ResultBlock| {
        for row in b.features {
            assert_eq!(row.status[0], Status::Estimated);
            let a = row.estimates[0].as_ref().unwrap();
            let (q, tau) = row
                .heterogeneity
                .map(|h| (h.q, h.tau_squared))
                .unwrap_or((0., 0.));
            out.extend([
                a.effect,
                a.standard_error,
                a.statistic,
                a.p_value,
                q,
                tau,
                row.n as f64,
            ]);
        }
        Ok(())
    };
    let o = FitOptions {
        block_features: block,
        missingness: Missingness::OmitObservation,
        ..Default::default()
    };
    let receipt = if method == "ols" {
        fit_ols(c, &d, &[ct], o, sink)
    } else {
        fit_meta(
            c,
            if method == "fe" {
                MetaMethod::CommonEffect
            } else {
                MetaMethod::DerSimonianLaird
            },
            o,
            sink,
        )
    }
    .unwrap();
    (out, receipt)
}
fn main() {
    let root = "evidence/group-tranche1";
    let mut records = vec![];
    for (n, nf, missing, p) in [
        (20, 1000, false, 1),
        (100, 10000, false, 1),
        (100, 10000, true, 1),
        (100, 100000, false, 1),
        (100, 10000, false, 5),
        (100, 10000, false, 20),
    ] {
        let mut y = vec![];
        let mut v = vec![];
        for i in 1..=n {
            for j in 1..=nf {
                y.push(
                    if missing && ((i == 2 && j % 7 == 0) || (i == 5 && j % 11 == 0)) {
                        f64::NAN
                    } else {
                        0.25 + ((i * j) as f64 * 0.017).sin() + (i as f64 * 0.23).cos() * 0.3
                    },
                );
                v.push(0.2 + ((i + 3 * j) % 17) as f64 * 0.03);
            }
        }
        let start = Instant::now();
        let c = cohort(n, nf, y, v);
        let setup = start.elapsed().as_secs_f64();
        let mut id = format!("{n}-{nf}-{}", usize::from(missing));
        if p > 1 {
            id = format!("{id}-p{p}");
        }
        for method in if p == 1 {
            vec!["ols", "fe", "dl"]
        } else {
            vec!["ols"]
        } {
            let start = Instant::now();
            let (output, receipt) = run(&c, method, 256, p);
            let first = start.elapsed().as_secs_f64();
            let bytes = fs::read(format!("{root}/reference-{id}-{method}.bin")).unwrap();
            let reference = bytes
                .chunks_exact(8)
                .map(|x| f64::from_le_bytes(x.try_into().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(reference.len(), output.len());
            let mut max_error = 0_f64;
            let mut errors_by_output = [0_f64; 7];
            for (i, (a, b)) in output.iter().zip(&reference).enumerate() {
                let error = (a - b).abs() / (1. + b.abs());
                max_error = max_error.max(error);
                errors_by_output[i % 7] = errors_by_output[i % 7].max(error);
                assert!(error < 2e-9, "{id} {method} index {i}: {a} != {b}");
            }
            // Partitioning changes no statistical output, including missing patterns.
            let (alternate, _) = run(&c, method, 113, p);
            for (a, b) in output.iter().zip(&alternate) {
                assert!((a - b).abs() < 2e-10 * (1. + a.abs()));
            }
            let mut times = vec![];
            for _ in 0..5 {
                let start = Instant::now();
                let (out, _) = run(&c, method, 256, p);
                std::hint::black_box(out);
                times.push(start.elapsed().as_secs_f64());
            }
            let mut sorted = times.clone();
            sorted.sort_by(f64::total_cmp);
            println!(
                "{id} {method}: median {:.6}s, max scaled error {:.3e}",
                sorted[2], max_error
            );
            records.push(json!({"id":id,"n":n,"features":nf,"method":method,"columns":p,"block_features":256,"setup_seconds":setup,"first_seconds":first,"seconds":times,"median_seconds":sorted[2],"max_scaled_error":max_error,"errors_by_output":errors_by_output,"numeric_working_bytes":receipt.numeric_working_bytes,"factorizations":receipt.factorizations}));
        }
    }
    fs::write(
        format!("{root}/rust-benchmark.json"),
        serde_json::to_string_pretty(&records).unwrap(),
    )
    .unwrap();
    let null: Value =
        serde_json::from_str(&fs::read_to_string(format!("{root}/null-input.json")).unwrap())
            .unwrap();
    let n = null["n"].as_u64().unwrap() as usize;
    let nf = null["features"].as_u64().unwrap() as usize;
    let c = cohort(
        n,
        nf,
        null["effects"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect(),
        vec![1.; n * nf],
    );
    let mut calibration = vec![];
    for method in ["ols", "fe", "dl"] {
        let (out, _) = run(&c, method, 256, 1);
        let rejects = out.chunks_exact(7).filter(|row| row[3] < 0.05).count();
        let rate = rejects as f64 / nf as f64;
        let mcse = (rate * (1. - rate) / nf as f64).sqrt();
        assert!(rate < 0.075, "unexpected null rejection rate");
        calibration.push(json!({"method":method,"seed":2710,"n":n,"replicates":nf,"rejections":rejects,"rate":rate,"monte_carlo_se":mcse,"scope":"iid normal null, unit known sampling variance"}));
    }
    fs::write(
        format!("{root}/calibration.json"),
        serde_json::to_string_pretty(&calibration).unwrap(),
    )
    .unwrap();
}
