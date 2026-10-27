//! Local qualification protocol: newline JSON commands in, identified maps out.
//! Pair with scripts/analysis_viewer.py. Synthetic data only in the demo.
use fmrimod::{
    analysis::{
        AnalysisSubject, CohortAnalysis, ExecutionPlan, ExecutionPolicy, ExpectedUse,
        RetentionPolicy, StoragePolicy, SubjectRequest,
    },
    cohort::ScalarCohortBuilder,
    dataset::{NiftiSource, ObservationId},
    group,
    group_requirements::GroupAnalysisSpec,
    ingress::NiftiSubject,
    noise::{ArStart, Pooling},
    outputs::{
        ConsumerRequirement, DesignColumn, DesignColumns, NamedLinearEstimand, NamedWeight,
        ScientificSemantics,
    },
    regression::{
        output::{Summary, Uncertainty},
        MatrixBackend,
    },
    session::{ModelSpec, NoisePolicy, SessionError},
};
use ndarray::Array2;
use serde_json::{json, Value};
use std::{
    error::Error,
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
    time::Instant,
};
struct App {
    input: PathBuf,
    manifest: Value,
    support: Vec<usize>,
    analysis: Option<CohortAnalysis>,
    mode: String,
}
fn model(
    rows: Vec<ObservationId>,
    runs: Vec<usize>,
    features: usize,
    phase: f64,
    fixed: bool,
) -> ModelSpec {
    let n: usize = runs.iter().sum();
    let split = runs[0];
    let design = Array2::from_shape_fn((n, 6), |(i, j)| {
        let run = usize::from(i >= split);
        let t = if run == 0 { i } else { i - split };
        match j {
            0 => f64::from(run == 0),
            1 => f64::from(run == 1),
            2 => (t as f64 * 0.07 + phase).sin(),
            3 => (t as f64 * 0.11).cos(),
            4 => {
                if run == 0 {
                    t as f64 / (runs[0] - 1) as f64 - 0.5
                } else {
                    0.
                }
            }
            _ => {
                if run == 1 {
                    t as f64 / (runs[1] - 1) as f64 - 0.5
                } else {
                    0.
                }
            }
        }
    });
    let sample = features.min(64);
    ModelSpec {
        design,
        observation_ids: rows,
        columns: DesignColumns::new(
            ["run0", "run1", "task", "nuisance", "drift0", "drift1"]
                .into_iter()
                .map(|id| DesignColumn::new(id, "signal", id).unwrap())
                .collect(),
        )
        .unwrap(),
        noise: if fixed {
            NoisePolicy::FixedAr {
                run_lengths: runs,
                coefficients: vec![vec![0.4, -0.12], vec![0.2, 0.05]],
                startup: ArStart::ZeroHistory,
                resets: vec![],
            }
        } else {
            NoisePolicy::EstimateAr {
                run_lengths: runs,
                noise_valid: None,
                order: 2,
                correction_max_lag: 4,
                pooling: Pooling::Run,
                startup: ArStart::ZeroHistory,
                feature_indices: (0..sample).map(|i| i * features / sample).collect(),
            }
        },
        tolerance: 1e-12,
        backend: MatrixBackend::default(),
    }
}
fn request(
    source: &NiftiSubject,
    phase: f64,
    subtract: bool,
    variance: bool,
    fixed: bool,
    runs: Vec<usize>,
    group_spec: &GroupAnalysisSpec,
) -> Result<SubjectRequest, Box<dyn Error>> {
    let definition = if subtract {
        "task minus half nuisance"
    } else {
        "task"
    };
    let mut weights = vec![NamedWeight::new("task", 1.)?];
    if subtract {
        weights.push(NamedWeight::new("nuisance", -0.5)?);
    }
    let effect = NamedLinearEstimand::new(
        ScientificSemantics::new(
            "task",
            definition,
            "signal",
            definition,
            "none",
            "shared slope across two runs",
            "synthetic-v1",
        )?,
        weights,
    )?;
    let mut consumers = vec![
        ConsumerRequirement::new(
            "viewer",
            source.subject_id(),
            "two-run-model",
            vec![effect.clone()],
            Uncertainty::None,
        )?,
        group_spec.first_level_requirement(
            "group",
            source.subject_id(),
            "two-run-model",
            effect.clone(),
        )?,
    ];
    if variance {
        consumers.push(ConsumerRequirement::new(
            "variance-export",
            source.subject_id(),
            "two-run-model",
            vec![effect],
            Uncertainty::Marginal,
        )?);
    }
    Ok(SubjectRequest {
        model_id: "two-run-model".into(),
        model: model(
            source.observation_ids().to_vec(),
            runs,
            source.shape()[1],
            phase,
            fixed,
        ),
        consumers,
    })
}
fn plan_json(plan: &ExecutionPlan) -> Value {
    json!({"storage":format!("{:?}",plan.storage),"workers":plan.workers,"retention":format!("{:?}",plan.retention),"reserved_numeric_bytes":plan.reserved_numeric_bytes,"external_reserved_bytes":plan.external_reserved_bytes,"reason":plan.reason,"subjects":plan.subjects.iter().map(|s|json!({"id":s.subject_id,"data":format!("{:?}",s.data),"model":format!("{:?}",s.model),"fit":format!("{:?}",s.fit),"noise":format!("{:?}",s.noise),"uncertainty":format!("{:?}",s.uncertainty),"estimands":s.estimand_ids,"tolerance":s.rank_tolerance})).collect::<Vec<_>>()})
}
impl App {
    fn command(&mut self, cmd: Value) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();
        if ![Some("run"), Some("inspect")].contains(&cmd["action"].as_str()) {
            return Err("unknown action".into());
        }
        let request_id = cmd.get("request_id").cloned();
        let cancel_file = cmd["cancel_file"].as_str().map(PathBuf::from);
        let checkpoint = |phase: &str,
                          completed_subjects: usize,
                          total_subjects: usize|
         -> Result<(), SessionError> {
            if cancel_file.as_ref().is_some_and(|p| p.exists()) {
                return Err(SessionError::Invalid("analysis cancelled"));
            }
            if let Some(id) = &request_id {
                println!(
                    "{}",
                    json!({"type":"progress","request_id":id,"phase":phase,"completed_subjects":completed_subjects,"total_subjects":total_subjects})
                );
                io::stdout()
                    .flush()
                    .map_err(|_| SessionError::Invalid("progress output closed"))?;
            }
            Ok(())
        };
        checkpoint(
            "admission",
            0,
            self.manifest["subjects"].as_u64().unwrap() as usize,
        )?;
        let phase = cmd["phase"].as_f64().ok_or("phase required")?;
        if !phase.is_finite() || phase.abs() > 1. {
            return Err("phase outside [-1,1]".into());
        }
        let contrast = cmd["contrast"].as_str().ok_or("contrast required")?;
        if !["task", "difference"].contains(&contrast) {
            return Err("unknown contrast".into());
        }
        let mode = cmd["mode"].as_str().ok_or("mode required")?;
        if !["resident", "files", "detached"].contains(&mode) {
            return Err("unknown mode".into());
        }
        let fixed = cmd["noise"].as_str() == Some("fixed");
        if ![Some("fixed"), Some("estimated")].contains(&cmd["noise"].as_str()) {
            return Err("unknown noise policy".into());
        }
        let group_kind = cmd["group"].as_str().unwrap_or("ols");
        if !["ols", "meta"].contains(&group_kind) {
            return Err("unknown group provider".into());
        }
        let rows = self.manifest["files"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| ObservationId::new(f["id"].as_str().unwrap()).unwrap())
            .collect::<Vec<_>>();
        let group_design = group::GroupDesign::new(
            rows.clone(),
            vec!["intercept".into()],
            Array2::ones((rows.len(), 1)),
        )?;
        let group_contrast =
            group::GroupContrast::named("mean", &group_design, &[("intercept", 1.)])?;
        let group_spec = if group_kind == "meta" {
            let mut outputs = group::MetaOutputSpec::new(vec![group_contrast.clone()]);
            outputs.uncertainty = if cmd["group_uncertainty"].as_bool().unwrap_or(true) {
                group::MetaUncertainty::Marginal
            } else {
                group::MetaUncertainty::None
            };
            GroupAnalysisSpec::MetaRegression {
                options: group::MetaRegressionOptions {
                    heterogeneity: group::HeterogeneityEstimator::PauleMandel,
                    inference: group::MetaInference::ModifiedKnappHartung,
                    ..Default::default()
                },
                outputs,
            }
        } else {
            GroupAnalysisSpec::Ols {
                options: group::FitOptions::default(),
            }
        };
        let variance = cmd["variance"].as_bool().ok_or("variance required")?
            || group_spec.requires_first_level_variance();
        let workers = cmd["workers"].as_u64().unwrap_or(4) as usize;
        let block = cmd["block_features"].as_u64().unwrap_or(1024) as usize;
        let budget_mib = cmd["budget_mib"].as_u64().unwrap_or(match mode {
            "resident" => 3072,
            "detached" => 256,
            _ => 128,
        }) as usize;
        if workers == 0
            || workers > 32
            || block == 0
            || block > 65536
            || budget_mib == 0
            || budget_mib > 65536
        {
            return Err("execution option out of range".into());
        }
        let policy_key = format!("{mode}:{workers}:{block}:{budget_mib}");
        let subtract = contrast == "difference";
        let runs = self.manifest["run_lengths"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        if self.analysis.is_none() || self.mode != policy_key {
            let mut subjects = Vec::new();
            for file in self.manifest["files"].as_array().unwrap() {
                let source = NiftiSource::new(
                    self.input.join(file["file"].as_str().unwrap()),
                    Some(self.support.clone()),
                    Some("synthetic-mask-grid".into()),
                )?;
                let rows = (0..source.shape()[0])
                    .map(|i| {
                        ObservationId::new(format!(
                            "run-{}-frame-{}",
                            usize::from(i >= runs[0]),
                            i % runs[0]
                        ))
                        .unwrap()
                    })
                    .collect();
                let input = NiftiSubject::new(
                    source,
                    rows,
                    file["id"].as_str().unwrap(),
                    file["sha256"].as_str().unwrap(),
                )?;
                let req = request(
                    &input,
                    phase,
                    subtract,
                    variance,
                    fixed,
                    runs.clone(),
                    &group_spec,
                )?;
                subjects.push(AnalysisSubject {
                    source: input,
                    request: req,
                });
            }
            let policy = ExecutionPolicy {
                expected_use: if mode == "resident" {
                    ExpectedUse::ContrastExploration
                } else {
                    ExpectedUse::OneShot
                },
                retention: if mode != "files" {
                    RetentionPolicy::CoefficientsAndScale
                } else {
                    RetentionPolicy::None
                },
                storage: if mode == "detached" {
                    StoragePolicy::RetainedFits
                } else {
                    StoragePolicy::Auto
                },
                max_numeric_bytes: budget_mib * 1024 * 1024,
                external_reserved_bytes: subjects.len() * self.support.len() * 16,
                workers,
                block_features: block,
            };
            let analysis = CohortAnalysis::new(subjects, policy)?;
            self.analysis = Some(analysis);
            self.mode = policy_key;
        } else {
            let analysis = self.analysis.as_mut().unwrap();
            let requests = analysis
                .sources()
                .iter()
                .map(|s| {
                    request(
                        s,
                        phase,
                        subtract,
                        variance,
                        fixed,
                        runs.clone(),
                        &group_spec,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            analysis.update(requests)?;
        }
        let analysis = self.analysis.as_mut().unwrap();
        let plan = analysis.inspect()?;
        if cmd["action"].as_str() == Some("inspect") {
            return Ok(
                json!({"plan":plan_json(&plan),"server_ms":start.elapsed().as_secs_f64()*1000.,"inspection_only":true}),
            );
        }
        if cmd["action"].as_str() != Some("run") {
            return Err("unknown action".into());
        }
        let subjects = analysis
            .sources()
            .iter()
            .map(|s| group::Subject {
                observation: ObservationId::new(s.subject_id()).unwrap(),
                dataset: "generated-multirun".into(),
                participant: s.subject_id().into(),
                measurement: "two runs".into(),
            })
            .collect();
        let mut collector = ScalarCohortBuilder::new(
            subjects,
            group::Estimand {
                id: "task".into(),
                definition: contrast.into(),
                units: "signal".into(),
            },
            analysis.sources()[0].features().clone(),
            group::Admission {
                target: "synthetic-mask-grid".into(),
                evidence: "identical generated geometry and mask; not registration evidence".into(),
            },
            variance,
            analysis.policy().external_reserved_bytes,
        )?;
        let probes = [
            0,
            self.support.len() / 4,
            self.support.len() / 2,
            self.support.len() - 1,
        ];
        let mut first_level = Vec::new();
        let receipt=analysis.run_with_progress(|p|checkpoint(p.phase,p.completed_subjects,p.total_subjects), |i,out,_| {
   let mut estimates=Vec::new();let mut variances=Vec::new();
   for feature in probes {let (range,b)=out.blocks().iter().find(|(r,_)|r.contains(&feature)).unwrap();let j=feature-range.start;estimates.push(b.estimates()[(0,j)]);variances.push(match b {Summary::Estimates(_)=>None,Summary::Marginal(m)=>Some(m.variance(0,j)?),Summary::Joint(m)=>Some(m.variance(0,j)?)});}
   first_level.push(json!({"subject":i,"estimates":estimates,"variances":variances,"df":out.receipt().residual_df,"noise_coefficients":out.receipt().noise_coefficients,"estimated_noise":out.receipt().estimated_noise}));
   collector.push(i,out).map_err(|_|SessionError::Invalid("group collector"))
  })?;
        let first_level_ms = start.elapsed().as_secs_f64() * 1000.;
        let cohort = collector.finish()?;
        checkpoint("group", 0, 1)?;
        let mut effects = Vec::new();
        let mut se: Vec<Option<f64>> = Vec::new();
        let mut tau_squared: Vec<Option<f64>> = Vec::new();
        match group_spec {
            GroupAnalysisSpec::Ols { options } => {
                group::fit_ols(
                    &cohort,
                    &group_design,
                    &[group_contrast],
                    options,
                    |block| {
                        checkpoint("group", block.start, self.support.len())
                            .map_err(|_| group::Error::Invalid("analysis cancelled"))?;
                        for feature in block.features {
                            let e = feature.estimates[0]
                                .as_ref()
                                .ok_or(group::Error::Invalid("unavailable mean"))?;
                            effects.push(e.effect);
                            se.push(Some(e.standard_error));
                            tau_squared.push(None);
                        }
                        Ok(())
                    },
                )?;
            }
            GroupAnalysisSpec::MetaRegression { options, outputs } => {
                group::fit_meta_regression(&cohort, &group_design, outputs, options, |block| {
                    checkpoint("group", block.start, self.support.len())
                        .map_err(|_| group::Error::Invalid("analysis cancelled"))?;
                    for feature in block.features {
                        let e = feature
                            .estimates
                            .as_ref()
                            .ok_or(group::Error::Invalid("unavailable meta estimate"))?;
                        effects.push(e.effects[0]);
                        se.push(e.standard_errors.as_ref().map(|s| s[0]));
                        tau_squared.push(feature.diagnostics.as_ref().map(|d| d.tau_squared));
                    }
                    Ok(())
                })?;
            }
        }
        checkpoint("complete", 1, 1)?;
        Ok(
            json!({"plan":plan_json(&receipt.plan),"shape":self.manifest["shape"],"support":self.support,"effects":effects,"group_se":se,"tau_squared":tau_squared,"group":group_kind,"probe_indices":probes,"first_level":first_level,"server_ms":start.elapsed().as_secs_f64()*1000.,"through_first_level_ms":first_level_ms,"source_reads":receipt.source_reads,"model_preparations":receipt.model_preparations,"model_reuses":receipt.model_reuses,"retained_fit_builds":receipt.retained_fit_builds,"phase":phase,"contrast":cmd["contrast"],"variance":variance,"noise":cmd["noise"]}),
        )
    }
}
fn main() -> Result<(), Box<dyn Error>> {
    let input = PathBuf::from(std::env::args().nth(1).ok_or("input directory required")?);
    let manifest: Value = serde_json::from_slice(&fs::read(input.join("manifest.json"))?)?;
    let support: Vec<usize> = serde_json::from_slice(&fs::read(input.join("mask.json"))?)?;
    let mut app = App {
        input,
        manifest,
        support,
        analysis: None,
        mode: String::new(),
    };
    for line in io::stdin().lock().lines() {
        let parsed = serde_json::from_str::<Value>(&line?);
        let request_id = parsed
            .as_ref()
            .ok()
            .and_then(|v| v.get("request_id"))
            .cloned();
        let response = parsed
            .map_err(|e| -> Box<dyn Error> { Box::new(e) })
            .and_then(|cmd| app.command(cmd));
        let mut value = match response {
            Ok(v) => v,
            Err(e) => json!({"error":e.to_string()}),
        };
        if let Some(id) = request_id {
            value["request_id"] = id;
            value["type"] = json!(if value.get("error").is_some() {
                "error"
            } else {
                "result"
            });
        }
        println!("{value}");
        io::stdout().flush()?;
    }
    Ok(())
}
