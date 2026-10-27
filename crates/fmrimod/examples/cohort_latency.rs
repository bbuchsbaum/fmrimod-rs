//! Reproducible cold-session / warm-design / contrast-edit latency experiment.
//! Generate inputs with scripts/generate_cohort.py; no research data is bundled.
use fmrimod::{
    cohort::{run_cohort, CohortOptions, ScalarCohortBuilder},
    dataset::{FeatureAxis, FeatureSpace, NiftiSource, ObservationId},
    group,
    noise::ArStart,
    outputs::{DesignColumn, DesignColumns},
    regression::{
        output::{EstimandSet, OutputSpec, Uncertainty},
        MatrixBackend,
    },
    session::{LoadedSubject, ModelSpec, NoisePolicy, SessionError, SubjectSession},
};
use ndarray::Array2;
use serde_json::json;
use std::{error::Error, fs, path::Path, time::Instant};
const BUDGET: usize = 1024 * 1024 * 1024;
fn model(session: &SubjectSession, phase: f64) -> ModelSpec {
    let n = session.subject().data().nrows();
    ModelSpec {
        design: Array2::from_shape_fn((n, 8), |(i, j)| {
            if j == 0 {
                1.
            } else {
                ((i + 1) as f64 * (j + 1) as f64 * 0.017 + if j == 1 { phase } else { 0. }).sin()
            }
        }),
        observation_ids: session.subject().observation_ids().to_vec(),
        columns: DesignColumns::new(
            (0..8)
                .map(|j| {
                    DesignColumn::new(format!("c{j}"), "signal", format!("synthetic column{j}"))
                        .unwrap()
                })
                .collect(),
        )
        .unwrap(),
        noise: NoisePolicy::FixedAr {
            run_lengths: vec![n],
            coefficients: vec![vec![0.4, -0.1, 0.08, -0.03]],
            startup: ArStart::ZeroHistory,
            resets: vec![],
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
fn run(
    sessions: &[SubjectSession],
    workers: usize,
    contrast_edit: bool,
) -> Result<(Vec<f64>, serde_json::Value), Box<dyn Error>> {
    let requests = vec![request(contrast_edit); sessions.len()];
    let subjects = sessions
        .iter()
        .map(|s| group::Subject {
            observation: ObservationId::new(s.subject_id()).unwrap(),
            dataset: "synthetic-fixture".into(),
            participant: s.subject_id().into(),
            measurement: "run-1".into(),
        })
        .collect();
    let mut collector = ScalarCohortBuilder::new(
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
        sessions[0].features().clone(),
        group::Admission {
            target: "synthetic-grid".into(),
            evidence: "generated identical affines and distinct synthetic subjects".into(),
        },
        true,
        BUDGET,
    )?;
    let collector_bytes = collector.numeric_bytes();
    let start = Instant::now();
    let receipt = run_cohort(
        sessions,
        &requests,
        CohortOptions {
            workers,
            block_features: 2048,
            max_numeric_bytes: BUDGET - collector_bytes,
        },
        |i, out| {
            collector
                .push(i, out)
                .map_err(|_| SessionError::Invalid("scalar group bridge"))
        },
    )?;
    let fit_ms = start.elapsed().as_secs_f64() * 1000.;
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
    let mut sessions = Vec::new();
    let mut loaded = 0usize;
    let mut dimensions = [0; 3];
    for file in manifest["files"].as_array().ok_or("manifest files")? {
        let name = file["file"].as_str().ok_or("file name")?;
        let id = name.trim_end_matches(".gz").trim_end_matches(".nii");
        let source = NiftiSource::new(input.join(name), None, Some("synthetic-grid".into()))?;
        dimensions = source.space().dimensions();
        let features = FeatureAxis::from_space(FeatureSpace::Volume(source.space().clone()))?;
        let rows = (0..source.shape()[0])
            .map(|i| ObservationId::new(format!("scan-{i}")).unwrap())
            .collect();
        let mut reader = source.open_reader()?;
        let subject = LoadedSubject::from_reader(
            &mut reader,
            rows,
            features,
            id,
            file["sha256"].as_str().ok_or("source version")?,
            BUDGET.checked_sub(loaded).ok_or("resident budget")?,
        )?;
        loaded += subject.data_bytes();
        sessions.push(SubjectSession::new(subject));
    }
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
        let available = BUDGET - other - sessions[i].data_bytes();
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
        let available = BUDGET - other - sessions[i].data_bytes();
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
    let receipt = json!({"backend":format!("{:?}",MatrixBackend::default()),"subjects":sessions.len(),"frames":manifest["frames"],"features":manifest["features"],"load_ms":load_ms,"initial_prepare_ms":prepare_ms,"cold_session_to_html_ms":cold_total_ms,"cold":cold,"design_edit_prepare_ms":edit_prepare_ms,"design_edit_to_html_ms":edit_total_ms,"design_edit":edit,"contrast_edit_to_html_ms":contrast_total_ms,"contrast_edit":contrast,"source_reads_total":sessions.iter().map(SubjectSession::read_count).sum::<usize>(),"model_preparations_total":after,"contrast_edit_preparations":after-before,"notes":"Synthetic fixed AR4; filesystem cache uncontrolled. HTML written, browser rendering not included. Warm requests still scan resident data. Numeric reservation excludes backend/decomposition scratch, metadata and allocator overhead."});
    fs::write(
        output.join("receipt.json"),
        serde_json::to_vec_pretty(&receipt)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
