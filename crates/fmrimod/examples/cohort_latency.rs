//! Reproducible cold-session / warm-design / contrast-edit latency experiment.
//! Generate inputs with scripts/generate_cohort.py; no research data is bundled.
use fmrimod::{
    cohort::{run_cohort, CohortOptions, ScalarCohortBuilder},
    dataset::{FeatureAxis, NiftiSource, ObservationId},
    group,
    ingress::{load_subjects, run_file_cohort, FileSession, LoadOptions, NiftiSubject},
    noise::{ArStart, Pooling},
    outputs::{DesignColumn, DesignColumns},
    regression::{
        output::{EstimandSet, OutputSpec, Uncertainty},
        MatrixBackend,
    },
    session::{ModelSpec, NoisePolicy, SessionError, SubjectSession},
};
use ndarray::Array2;
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};
fn budget() -> usize {
    std::env::var("COHORT_BUDGET_MIB")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()
        .expect("validated budget")
        * 1024
        * 1024
}
fn estimated_noise() -> bool {
    std::env::var("COHORT_NOISE").as_deref() == Ok("estimated")
}
fn model(session: &SubjectSession, phase: f64) -> ModelSpec {
    model_input(
        session.subject().data().nrows(),
        session.subject().data().ncols(),
        session.subject().observation_ids(),
        phase,
    )
}
fn model_input(n: usize, v: usize, rows: &[ObservationId], phase: f64) -> ModelSpec {
    ModelSpec {
        design: Array2::from_shape_fn((n, 8), |(i, j)| {
            if j == 0 {
                1.
            } else {
                ((i + 1) as f64 * (j + 1) as f64 * 0.017 + if j == 1 { phase } else { 0. }).sin()
            }
        }),
        observation_ids: rows.to_vec(),
        columns: DesignColumns::new(
            (0..8)
                .map(|j| {
                    DesignColumn::new(format!("c{j}"), "signal", format!("synthetic column{j}"))
                        .unwrap()
                })
                .collect(),
        )
        .unwrap(),
        noise: if estimated_noise() {
            let k = v.min(256);
            NoisePolicy::EstimateAr {
                run_lengths: vec![n],
                noise_valid: None,
                order: 4,
                correction_max_lag: 8,
                pooling: Pooling::Global,
                startup: ArStart::ZeroHistory,
                feature_indices: (0..k).map(|i| i * v / k).collect(),
            }
        } else {
            NoisePolicy::FixedAr {
                run_lengths: vec![n],
                coefficients: vec![vec![0.4, -0.1, 0.08, -0.03]],
                startup: ArStart::ZeroHistory,
                resets: vec![],
            }
        },
        tolerance: 1e-12,
        backend: MatrixBackend::default(),
    }
}
fn request(contrast_edit: bool) -> OutputSpec {
    let l = Array2::from_shape_fn((1, 8), |(_, j)| {
        if j == 1 {
            1.
        } else if contrast_edit && j == 2 {
            -0.5
        } else {
            0.
        }
    });
    OutputSpec::new(EstimandSet::contrasts(vec!["task".into()], l).unwrap())
        .with_uncertainty(Uncertainty::Marginal)
}
fn make_collector(
    ids: &[String],
    features: &FeatureAxis,
    contrast_edit: bool,
) -> Result<ScalarCohortBuilder, Box<dyn Error>> {
    let subjects = ids
        .iter()
        .map(|s| group::Subject {
            observation: ObservationId::new(s.as_str()).unwrap(),
            dataset: "synthetic-fixture".into(),
            participant: s.as_str().into(),
            measurement: "run-1".into(),
        })
        .collect();
    Ok(ScalarCohortBuilder::new(
        subjects,
        group::Estimand {
            id: "task".into(),
            definition: if contrast_edit {
                "c1 minus half c2"
            } else {
                "c1"
            }
            .into(),
            units: "signal".into(),
        },
        features.clone(),
        group::Admission {
            target: "synthetic-grid".into(),
            evidence: "generated identical affines and distinct synthetic subjects".into(),
        },
        true,
        budget(),
    )?)
}
fn run(
    sessions: &[SubjectSession],
    workers: usize,
    contrast_edit: bool,
) -> Result<(Vec<f64>, serde_json::Value), Box<dyn Error>> {
    let requests = vec![request(contrast_edit); sessions.len()];
    let mut collector = make_collector(
        &sessions
            .iter()
            .map(|s| s.subject_id().to_string())
            .collect::<Vec<_>>(),
        sessions[0].features(),
        contrast_edit,
    )?;
    let collector_bytes = collector.numeric_bytes();
    let start = Instant::now();
    let receipt = run_cohort(
        sessions,
        &requests,
        CohortOptions {
            workers,
            block_features: 2048,
            max_numeric_bytes: budget() - collector_bytes,
        },
        |i, out| {
            collector
                .push(i, out)
                .map_err(|_| SessionError::Invalid("scalar group bridge"))
        },
    )?;
    let fit_ms = start.elapsed().as_secs_f64() * 1000.;
    finish_run(collector, receipt, collector_bytes, fit_ms)
}
fn finish_run(
    collector: ScalarCohortBuilder,
    receipt: fmrimod::cohort::CohortReceipt,
    collector_bytes: usize,
    fit_ms: f64,
) -> Result<(Vec<f64>, serde_json::Value), Box<dyn Error>> {
    let start = Instant::now();
    let cohort = collector.finish()?;
    let design = group::GroupDesign::intercept(&cohort);
    let contrast = group::GroupContrast::named("group mean", &design, &[("intercept", 1.)])?;
    let mut effects = Vec::new();
    group::fit_ols(
        &cohort,
        &design,
        &[contrast],
        group::FitOptions {
            block_features: 1024,
            ..Default::default()
        },
        |block| {
            for feature in block.features {
                effects.push(
                    feature.estimates[0]
                        .as_ref()
                        .ok_or(group::Error::Invalid("group estimate unavailable"))?
                        .effect,
                );
            }
            Ok(())
        },
    )?;
    let group_ms = start.elapsed().as_secs_f64() * 1000.;
    Ok((
        effects,
        json!({"first_level_ms":fit_ms,"group_ms":group_ms,"resident_numeric_bytes":receipt.resident_numeric_bytes,"reserved_pipeline_numeric_bytes":receipt.reserved_numeric_bytes+collector_bytes,"workers":receipt.active_workers}),
    ))
}
fn run_files(
    sessions: &mut [FileSession],
    workers: usize,
    phase: f64,
    contrast_edit: bool,
) -> Result<(Vec<f64>, serde_json::Value), Box<dyn Error>> {
    let specs = sessions
        .iter()
        .map(|s| {
            model_input(
                s.source().shape()[0],
                s.source().shape()[1],
                s.source().observation_ids(),
                phase,
            )
        })
        .collect::<Vec<_>>();
    let requests = vec![request(contrast_edit); sessions.len()];
    let mut collector = make_collector(
        &sessions
            .iter()
            .map(|s| s.source().subject_id().to_string())
            .collect::<Vec<_>>(),
        sessions[0].source().features(),
        contrast_edit,
    )?;
    let collector_bytes = collector.numeric_bytes();
    let start = Instant::now();
    let receipt = run_file_cohort(
        sessions,
        &specs,
        &requests,
        CohortOptions {
            workers,
            block_features: 2048,
            max_numeric_bytes: budget()
                .checked_sub(collector_bytes)
                .ok_or("collector budget")?,
        },
        |i, out| {
            collector
                .push(i, out)
                .map_err(|_| SessionError::Invalid("scalar group bridge"))
        },
    )?;
    finish_run(
        collector,
        receipt,
        collector_bytes,
        start.elapsed().as_secs_f64() * 1000.,
    )
}
fn file_experiment(
    initial_start: Instant,
    sources: Vec<NiftiSubject>,
    workers: usize,
    dimensions: [usize; 3],
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    let mut sessions = sources
        .into_iter()
        .map(FileSession::new)
        .collect::<Vec<_>>();
    let mut stages = Vec::new();
    for (name, phase, contrast) in [
        ("cold", 0., false),
        ("design-edit", 0.18, false),
        ("contrast-edit", 0.18, true),
    ] {
        let start = if name == "cold" {
            initial_start
        } else {
            Instant::now()
        };
        let (values, mut receipt) = run_files(&mut sessions, workers, phase, contrast)?;
        viewer(
            &output.join(format!("{name}.html")),
            &values,
            dimensions,
            name,
        )?;
        receipt["stage"] = json!(name);
        receipt["to_html_ms"] = json!(start.elapsed().as_secs_f64() * 1000.);
        stages.push(receipt);
    }
    let receipt = json!({"mode":"file batches, prepared models retained, data released", "noise_estimated":estimated_noise(), "budget_bytes":budget(),"stages":stages,"source_reads_total":sessions.iter().map(FileSession::read_count).sum::<usize>(),"model_preparations_total":sessions.iter().map(|s|s.preparation_counts().prepared).sum::<usize>(),"model_reuses_total":sessions.iter().map(|s|s.preparation_counts().reused).sum::<usize>()});
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", receipt);
    Ok(())
}
fn viewer(
    path: &Path,
    values: &[f64],
    dims: [usize; 3],
    title: &str,
) -> Result<(), Box<dyn Error>> {
    let data = serde_json::to_string(values)?;
    let html = r#"<!doctype html><meta charset="utf-8"><title>Synthetic group contrast</title><style>body{background:#101923;color:#e8edf2;font:16px system-ui;max-width:850px;margin:40px auto}canvas{width:512px;height:512px;image-rendering:pixelated;border:1px solid #657380}input{width:420px}small{color:#b7c5d2}</style><h1>__TITLE__</h1><p><small>Synthetic qualification data · group mean effect · unthresholded · axial slices, voxel array orientation</small></p><canvas id="map"></canvas><p><label>Slice <input id="slice" type="range" min="0" max="__MAX__" value="__MID__"><span id="z"></span></label></p><p id="scale"></p><script>const values=__DATA__,nx=__NX__,ny=__NY__;const c=document.querySelector('#map'),s=document.querySelector('#slice');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),max=values.reduce((m,v)=>Math.max(m,Math.abs(v)),0)||1;document.querySelector('#scale').textContent=`Blue: ${(-max).toFixed(3)} · white: 0 · red: ${max.toFixed(3)}`;function draw(){const z=+s.value;document.querySelector('#z').textContent=z;const im=ctx.createImageData(nx,ny);for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const v=values[x+nx*(y+ny*z)]/max,k=4*(x+nx*(ny-1-y)),a=Math.min(1,Math.abs(v));im.data[k]=v>=0?255:255*(1-a);im.data[k+1]=255*(1-a);im.data[k+2]=v<=0?255:255*(1-a);im.data[k+3]=255;}ctx.putImageData(im,0,0);}s.oninput=draw;draw();</script>"#;
    fs::write(
        path,
        html.replace("__TITLE__", title)
            .replace("__DATA__", &data)
            .replace("__NX__", &dims[0].to_string())
            .replace("__NY__", &dims[1].to_string())
            .replace("__MAX__", &(dims[2] - 1).to_string())
            .replace("__MID__", &(dims[2] / 2).to_string()),
    )?;
    Ok(())
}
fn main() -> Result<(), Box<dyn Error>> {
    let mib = std::env::var("COHORT_BUDGET_MIB")
        .unwrap_or_else(|_| "1024".into())
        .parse::<usize>()?;
    if mib == 0 || mib.checked_mul(1024 * 1024).is_none() {
        return Err("invalid budget".into());
    }
    if !matches!(
        std::env::var("COHORT_NOISE").as_deref(),
        Err(_) | Ok("fixed") | Ok("estimated")
    ) {
        return Err("COHORT_NOISE must be fixed or estimated".into());
    }
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        return Err("usage: cohort_latency INPUT_DIRECTORY OUTPUT_DIRECTORY [WORKERS]".into());
    }
    let input = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    fs::create_dir_all(output)?;
    let workers = args.get(3).map(|x| x.parse()).transpose()?.unwrap_or(4);
    let manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(input.join("manifest.json"))?)?;
    let cold_start = Instant::now();
    let load_start = Instant::now();
    let load_workers = std::env::var("COHORT_LOAD_WORKERS")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    let mut sources = Vec::new();
    let mut dimensions = [0; 3];
    for file in manifest["files"].as_array().ok_or("manifest files")? {
        let name = file["file"].as_str().ok_or("file name")?;
        let id = name.trim_end_matches(".gz").trim_end_matches(".nii");
        let source = NiftiSource::new(input.join(name), None, Some("synthetic-grid".into()))?;
        dimensions = source.space().dimensions();
        let rows = (0..source.shape()[0])
            .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
            .collect();
        sources.push(NiftiSubject::new(
            source,
            rows,
            id,
            file["sha256"].as_str().ok_or("source version")?,
        )?);
    }
    if std::env::var("COHORT_STREAMING").as_deref() == Ok("1") {
        return file_experiment(cold_start, sources, workers, dimensions, output);
    }
    let (mut sessions, load_receipt) = load_subjects(
        &sources,
        LoadOptions {
            workers: load_workers,
            max_numeric_bytes: budget(),
        },
    )?;
    drop(sources);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    for i in 0..sessions.len() {
        let spec = model(&sessions[i], 0.);
        let other = sessions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .try_fold(0usize, |sum, (_, s)| {
                Ok::<_, SessionError>(sum + s.resident_numeric_bytes()?)
            })?;
        let available = budget() - other - sessions[i].data_bytes();
        sessions[i].prepare(&spec, available)?;
    }
    let prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    let (values, cold) = run(&sessions, workers, false)?;
    viewer(
        &output.join("cold.html"),
        &values,
        dimensions,
        "Cold-session group effect",
    )?;
    let cold_total_ms = cold_start.elapsed().as_secs_f64() * 1000.;
    let start = Instant::now();
    for i in 0..sessions.len() {
        let spec = model(&sessions[i], 0.18);
        let other = sessions
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .try_fold(0usize, |sum, (_, s)| {
                Ok::<_, SessionError>(sum + s.resident_numeric_bytes()?)
            })?;
        let available = budget() - other - sessions[i].data_bytes();
        sessions[i].prepare(&spec, available)?;
    }
    let edit_prepare_ms = start.elapsed().as_secs_f64() * 1000.;
    let edit_start = Instant::now();
    let (values, edit) = run(&sessions, workers, false)?;
    viewer(
        &output.join("design-edit.html"),
        &values,
        dimensions,
        "Warm design-edit group effect",
    )?;
    let edit_total_ms = edit_prepare_ms + edit_start.elapsed().as_secs_f64() * 1000.;
    let before = sessions
        .iter()
        .map(|s| s.preparation_counts().prepared)
        .sum::<usize>();
    let retention_start = Instant::now();
    if std::env::var("COHORT_RETAIN").as_deref() == Ok("1") {
        for i in 0..sessions.len() {
            let resident = sessions.iter().try_fold(0usize, |a, s| {
                Ok::<_, SessionError>(a + s.resident_numeric_bytes()?)
            })?;
            sessions[i].retain_for_edits(
                2048,
                budget().checked_sub(resident).ok_or("retention budget")?,
            )?;
        }
    }
    let retention_ms = retention_start.elapsed().as_secs_f64() * 1000.;
    let contrast_start = Instant::now();
    let (values, contrast) = run(&sessions, workers, true)?;
    viewer(
        &output.join("contrast-edit.html"),
        &values,
        dimensions,
        "Warm contrast-edit group effect",
    )?;
    let contrast_total_ms = contrast_start.elapsed().as_secs_f64() * 1000.;
    let after = sessions
        .iter()
        .map(|s| s.preparation_counts().prepared)
        .sum::<usize>();
    assert_eq!(before, after);
    let receipt = json!({"retention_build_ms":retention_ms,"retained_bytes":sessions.iter().map(SubjectSession::retained_numeric_bytes).sum::<usize>(),"load_workers":load_receipt.active_workers,"load_reserved_bytes":load_receipt.reserved_numeric_bytes,"noise_policy":if estimated_noise(){"estimated global AR4, 256 spatially spread features, correction lag8"}else{"fixed AR4"},"budget_bytes":budget(),"backend":format!("{:?}",MatrixBackend::default()),"subjects":sessions.len(),"frames":manifest["frames"],"features":manifest["features"],"load_ms":load_ms,"initial_prepare_ms":prepare_ms,"cold_session_to_html_ms":cold_total_ms,"cold":cold,"design_edit_prepare_ms":edit_prepare_ms,"design_edit_to_html_ms":edit_total_ms,"design_edit":edit,"contrast_edit_to_html_ms":contrast_total_ms,"contrast_edit":contrast,"source_reads_total":sessions.iter().map(SubjectSession::read_count).sum::<usize>(),"model_preparations_total":after,"contrast_edit_preparations":after-before,"notes":"Synthetic AR4; see noise_policy. Filesystem cache uncontrolled. HTML written, browser rendering not included. Design edits scan data; contrast scans depend on retained_bytes. Numeric reservation excludes backend/decomposition scratch, metadata and allocator overhead."});
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
