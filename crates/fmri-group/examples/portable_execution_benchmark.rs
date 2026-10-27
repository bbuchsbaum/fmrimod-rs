//! Bounded portable-cohort generation, execution, and equivalence checks.
//!
//! Usage:
//!   cargo run -p fmri-group --example portable_execution_benchmark -- generate DIR N F P BLOCK
//!   cargo run -p fmri-group --example portable_execution_benchmark -- bench DIR OUT_JSON disk|memory general|shared P BLOCK
//!   cargo run -p fmri-group --example portable_execution_benchmark -- compare DIR1 DIR2 P

use fmri_group::*;
use fmridataset::{
    Assay, AssayAnnotations, Axis, ExecutionContext, FeatureAxis, FeatureId, FeatureSpace,
    FmriFrame, IndexSpace, MemoryBudget, Metadata, NumericBlock, ObservationId, ReadSelection,
    Selection, SourceSpec, Values,
};
use ndarray::Array2;
use serde_json::{json, Value};
use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

const METADATA_BUDGET: usize = 64 << 20;
const COMPARISON_BLOCK: usize = 257;

struct CountAlloc;
static COUNT_ALLOCATIONS: AtomicBool = AtomicBool::new(false);
static ALLOCATION_CALLS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATION_BYTES: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_allocation(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        count_allocation(size);
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

fn count_allocation(bytes: usize) {
    if COUNT_ALLOCATIONS.load(Ordering::Relaxed) {
        ALLOCATION_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOCATION_BYTES.fetch_add(bytes, Ordering::Relaxed);
    }
}

fn parse_usize(value: Option<&String>, name: &str) -> usize {
    value
        .and_then(|x| x.parse().ok())
        .filter(|x: &usize| *x > 0)
        .unwrap_or_else(|| panic!("{name} must be a positive integer"))
}

fn features(features: usize) -> FeatureAxis {
    FeatureAxis::from_space(FeatureSpace::Index(
        IndexSpace::new(
            "portable-execution",
            (0..features)
                .map(|j| FeatureId::new(format!("f{j}")).unwrap())
                .collect(),
        )
        .unwrap(),
    ))
    .unwrap()
}

fn metadata(n: usize, f: usize) -> PortableMetadata {
    PortableMetadata {
        subjects: (0..n)
            .map(|i| Subject {
                observation: ObservationId::new(format!("o{i}")).unwrap(),
                dataset: "portable-execution".into(),
                participant: format!("participant-{i}"),
                measurement: "baseline".into(),
            })
            .collect(),
        estimand: Estimand {
            id: "synthetic-effect".into(),
            definition: "deterministic portable execution fixture".into(),
            units: "arbitrary".into(),
        },
        features: features(f),
        variance_origins: Some(vec![
            VarianceOrigin::Known {
                method: "known synthetic sampling variance".into(),
            };
            n
        ]),
        fit_receipts: vec![
            FitReceipt {
                model: "deterministic generator".into(),
                noise: "known variance".into(),
                selection: "all generated features".into(),
            };
            n
        ],
        admission: Admission {
            target: "known synthetic feature support".into(),
            evidence: "same generated indexed feature axis for every subject".into(),
        },
    }
}

fn effect(i: usize, j: usize) -> f64 {
    0.25 + ((i + 1) as f64 * (j + 1) as f64 * 0.017).sin() + 0.3 * ((i + 1) as f64 * 0.23).cos()
}

fn reference_variances(n: usize) -> Vec<f64> {
    (0..n).map(|i| 0.25 + (i % 5) as f64 * 0.125).collect()
}

fn variance(reference: &[f64], i: usize, j: usize) -> f64 {
    reference[i] * 2_f64.powi(j as i32 % 5 - 2)
}

fn generate(directory: &Path, n: usize, f: usize, p: usize, block: usize) -> Result<()> {
    if p >= n || block == 0 {
        return Err(Error::Invalid(
            "generation requires n > p and nonzero block",
        ));
    }
    let block_bytes = n
        .checked_mul(block)
        .and_then(|x| x.checked_mul(8))
        .ok_or(Error::Invalid("generation block overflow"))?;
    let mut writer = EstimateBundleWriter::create(
        directory,
        metadata(n, f),
        PortableWriteOptions {
            max_block_bytes: block_bytes,
        },
    )?;
    let reference = reference_variances(n);
    for start in (0..f).step_by(block) {
        let end = (start + block).min(f);
        let effects =
            Array2::from_shape_fn((n, end - start), |(i, offset)| effect(i, start + offset));
        let variances = Array2::from_shape_fn((n, end - start), |(i, offset)| {
            variance(&reference, i, start + offset)
        });
        writer.write_effect_block(start, effects.view())?;
        writer.write_variance_block(start, variances.view())?;
    }
    writer.finish()
}

fn open(directory: &Path) -> Result<IndependentCohort> {
    read_cohort_bundle(
        directory,
        PortableReadOptions {
            max_manifest_bytes: METADATA_BUDGET,
            max_block_bytes: METADATA_BUDGET,
        },
    )
}

fn materialize(cohort: &IndependentCohort) -> Result<IndependentCohort> {
    let [n, f] = cohort.frame().shape();
    // Materialization is an explicit comparison path with a separate 256 MiB
    // per-assay read limit; the streaming fit still uses its 64 MiB budget.
    let context = ExecutionContext::new(MemoryBudget::new(256 << 20));
    let effect = context.collect(cohort.frame().assay(cohort.effect_assay())?.source())?;
    let variance_name = cohort
        .variance_assay()
        .ok_or(Error::Invalid("portable fixture lacks variance"))?;
    let variance = context.collect(cohort.frame().assay(variance_name)?.source())?;
    let ids = cohort.frame().observations().ids().to_vec();
    let feature_ids = cohort.frame().features().axis().ids().to_vec();
    let assays = vec![
        Assay::new(
            "effect",
            SourceSpec::memory(NumericBlock::new([n, f], effect.into_values())?),
            ids.clone(),
            feature_ids.clone(),
            AssayAnnotations::default(),
        )?,
        Assay::new(
            "variance",
            SourceSpec::memory(NumericBlock::new([n, f], variance.into_values())?),
            ids.clone(),
            feature_ids,
            AssayAnnotations::default(),
        )?,
    ];
    let frame = FmriFrame::new(
        Axis::new(ids, Vec::new())?,
        cohort.frame().features().clone(),
        assays,
        "effect",
        Metadata::new(),
    )?;
    IndependentCohort::new(
        frame,
        cohort.subjects().to_vec(),
        cohort.estimand().clone(),
        "effect".into(),
        Some(("variance".into(), cohort.variance_origins().to_vec())),
        cohort.fit_receipts().to_vec(),
        cohort.admission().clone(),
    )
}

fn design(cohort: &IndependentCohort, p: usize) -> Result<GroupDesign> {
    let n = cohort.subjects().len();
    if p >= n {
        return Err(Error::Invalid("design requires n > p"));
    }
    GroupDesign::new(
        cohort.frame().observations().ids().to_vec(),
        (0..p).map(|j| format!("x{j}")).collect(),
        Array2::from_shape_fn((n, p), |(i, j)| {
            (std::f64::consts::PI * (i as f64 + 0.5) * j as f64 / n as f64).cos()
        }),
    )
}

fn output_spec(design: &GroupDesign) -> MetaOutputSpec {
    MetaOutputSpec {
        contrasts: design
            .columns()
            .iter()
            .map(|name| GroupContrast::named(name.clone(), design, &[(name.as_str(), 1.)]))
            .collect::<Result<Vec<_>>>()
            .unwrap(),
        uncertainty: MetaUncertainty::Marginal,
        p_values: false,
        confidence_level: None,
    }
}

fn options(block: usize) -> MetaRegressionOptions {
    MetaRegressionOptions {
        execution: FitOptions {
            block_features: block,
            max_working_bytes: METADATA_BUDGET,
            ..Default::default()
        },
        heterogeneity: HeterogeneityEstimator::CommonEffect,
        inference: MetaInference::Normal,
        ..Default::default()
    }
}

fn prepared(
    cohort: &IndependentCohort,
    p: usize,
    block: usize,
    shared: bool,
) -> Result<PreparedMetaRegression> {
    let design = design(cohort, p)?;
    let plan = PreparedMetaRegression::new(&design, output_spec(&design), options(block))?;
    if shared {
        plan.with_shared_variances(reference_variances(cohort.subjects().len()))
    } else {
        Ok(plan)
    }
}

#[derive(Default)]
struct Sink {
    checksum: u64,
    sum: f64,
    count: usize,
}
impl Sink {
    fn push(&mut self, value: f64) {
        self.checksum ^= value.to_bits().rotate_left((self.count % 63) as u32);
        self.sum += value;
        self.count += 1;
    }
    fn consume(&mut self, block: MetaRegressionBlock) -> Result<()> {
        for feature in block.features {
            if feature.status != MetaStatus::Estimated
                || feature.estimates.is_none()
                || feature.diagnostics.is_none()
            {
                return Err(Error::Invalid("benchmark feature was not estimated"));
            }
            self.push(feature.n as f64);
            self.push(feature.rank as f64);
            if let Some(estimates) = feature.estimates {
                for value in estimates.effects {
                    self.push(value);
                }
                if let Some(values) = estimates.standard_errors {
                    for value in values {
                        self.push(value);
                    }
                }
            }
            if let Some(diagnostics) = feature.diagnostics {
                self.push(diagnostics.q_fixed.expect("estimated/common-effect Q"));
                self.push(diagnostics.q_final);
                self.push(diagnostics.tau_squared);
                self.push(diagnostics.residual_df as f64);
            }
        }
        Ok(())
    }
}

fn run(
    plan: &mut PreparedMetaRegression,
    cohort: &IndependentCohort,
) -> Result<(Sink, MetaRegressionReceipt)> {
    let mut sink = Sink::default();
    let receipt = plan.fit(cohort, |block| sink.consume(block))?;
    Ok((sink, receipt))
}

fn bench(
    directory: &Path,
    output: &Path,
    storage: &str,
    strategy: &str,
    p: usize,
    block: usize,
) -> Result<()> {
    let opened = Instant::now();
    let disk = open(directory)?;
    let open_seconds = opened.elapsed().as_secs_f64();
    let materialized = Instant::now();
    let cohort = match storage {
        "disk" => disk,
        "memory" => materialize(&disk)?,
        _ => return Err(Error::Invalid("storage must be disk or memory")),
    };
    let materialization_seconds = materialized.elapsed().as_secs_f64();
    let shared = match strategy {
        "general" => false,
        "shared" => true,
        _ => return Err(Error::Invalid("strategy must be general or shared")),
    };
    let started = Instant::now();
    let mut plan = prepared(&cohort, p, block, shared)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let first_started = Instant::now();
    let (first_sink, first_receipt) = run(&mut plan, &cohort)?;
    let first_seconds = first_started.elapsed().as_secs_f64();
    let mut warm_seconds = Vec::with_capacity(5);
    for _ in 0..5 {
        let start = Instant::now();
        let (sink, _) = run(&mut plan, &cohort)?;
        assert_eq!(sink.count, first_sink.count);
        assert_eq!(sink.checksum, first_sink.checksum);
        std::hint::black_box((sink.checksum, sink.sum, sink.count));
        warm_seconds.push(start.elapsed().as_secs_f64());
    }
    ALLOCATION_CALLS.store(0, Ordering::Relaxed);
    ALLOCATION_BYTES.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.store(true, Ordering::Relaxed);
    let (allocation_sink, _) = run(&mut plan, &cohort)?;
    COUNT_ALLOCATIONS.store(false, Ordering::Relaxed);
    // This executable is launched once per workload. getrusage observes only
    // its own process and avoids /usr/bin/time's privileged sysctl on macOS.
    let peak_rss_bytes = unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut usage) != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        let units = if cfg!(target_os = "macos") { 1 } else { 1024 };
        usage.ru_maxrss as u64 * units
    };
    let record = json!({
        "directory": directory,
        "storage": storage,
        "strategy": strategy,
        "shape": cohort.frame().shape(),
        "open_checksum_seconds": open_seconds,
        "materialization_seconds": materialization_seconds,
        "materialization_read_budget_bytes": 256usize << 20,
        "preparation_seconds": preparation_seconds,
        "first_fit_seconds": first_seconds,
        "warm_fit_seconds": warm_seconds,
        "sink_checksum": first_sink.checksum,
        "sink_sum": first_sink.sum,
        "sink_count": first_sink.count,
        "allocation_checksum": allocation_sink.checksum,
        "allocation_calls": ALLOCATION_CALLS.load(Ordering::Relaxed),
        "allocated_bytes": ALLOCATION_BYTES.load(Ordering::Relaxed),
        "shared_features": first_receipt.shared_features,
        "factorizations": first_receipt.run.factorizations,
        "numeric_working_bytes": first_receipt.run.numeric_working_bytes,
        "peak_rss_bytes": peak_rss_bytes,
    });
    std::fs::write(output, serde_json::to_vec_pretty(&record)?)?;
    Ok(())
}

fn same_float(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || a == b
}
fn near(a: f64, b: f64) -> bool {
    same_float(a, b) || (a - b).abs() <= 2e-9 * (1. + b.abs())
}

fn compare_cells(left: &IndependentCohort, right: &IndependentCohort) -> Result<()> {
    if left.subjects() != right.subjects()
        || left.estimand() != right.estimand()
        || left.frame().features().space() != right.frame().features().space()
        || left.variance_origins() != right.variance_origins()
        || left.fit_receipts() != right.fit_receipts()
        || left.admission() != right.admission()
        || left.frame().shape() != right.frame().shape()
    {
        return Err(Error::Invalid("portable metadata differs"));
    }
    let [n, f] = left.frame().shape();
    let context = ExecutionContext::new(MemoryBudget::new(METADATA_BUDGET));
    for assay in ["effect", "variance"] {
        let left_source = left.frame().assay(assay)?.source();
        let right_source = right.frame().assay(assay)?.source();
        let mut left_reader = context.open(left_source);
        let mut right_reader = context.open(right_source);
        for start in (0..f).step_by(COMPARISON_BLOCK) {
            let end = (start + COMPARISON_BLOCK).min(f);
            let selection =
                ReadSelection::new([n, f], Selection::All, Selection::Range(start..end))?;
            let l = context.read(&mut left_reader, &selection)?.into_values();
            let r = context.read(&mut right_reader, &selection)?.into_values();
            let (Values::F64(l), Values::F64(r)) = (l, r) else {
                return Err(Error::Invalid("portable values are not f64"));
            };
            if !l.iter().zip(&r).all(|(a, b)| same_float(*a, *b)) {
                return Err(Error::Invalid("portable numeric cell differs"));
            }
        }
    }
    Ok(())
}

fn capture(
    cohort: &IndependentCohort,
    p: usize,
    shared: bool,
) -> Result<Vec<MetaRegressionFeature>> {
    let mut plan = prepared(cohort, p, COMPARISON_BLOCK, shared)?;
    let mut results = Vec::new();
    plan.fit(cohort, |block| {
        results.extend(block.features);
        Ok(())
    })?;
    Ok(results)
}

fn compare_results(left: &[MetaRegressionFeature], right: &[MetaRegressionFeature]) -> Result<()> {
    if left.len() != right.len() {
        return Err(Error::Invalid("result count differs"));
    }
    for (a, b) in left.iter().zip(right) {
        if a.feature != b.feature || a.n != b.n || a.rank != b.rank || a.status != b.status {
            return Err(Error::Invalid("result status differs"));
        }
        let (Some(ae), Some(be), Some(ad), Some(bd)) =
            (&a.estimates, &b.estimates, &a.diagnostics, &b.diagnostics)
        else {
            if a.estimates.is_some() != b.estimates.is_some()
                || a.diagnostics.is_some() != b.diagnostics.is_some()
            {
                return Err(Error::Invalid("result availability differs"));
            }
            continue;
        };
        if !ae
            .effects
            .iter()
            .zip(&be.effects)
            .all(|(x, y)| near(*x, *y))
            || !ae
                .standard_errors
                .as_ref()
                .unwrap()
                .iter()
                .zip(be.standard_errors.as_ref().unwrap())
                .all(|(x, y)| near(*x, *y))
            || ![
                ad.q_fixed.expect("estimated/common-effect Q"),
                ad.q_final,
                ad.tau_squared,
                ad.residual_df as f64,
            ]
            .iter()
            .zip([
                bd.q_fixed.expect("estimated/common-effect Q"),
                bd.q_final,
                bd.tau_squared,
                bd.residual_df as f64,
            ])
            .all(|(x, y)| near(*x, y))
        {
            return Err(Error::Invalid("result value differs"));
        }
    }
    Ok(())
}

fn compare(left_dir: &Path, right_dir: &Path, p: usize) -> Result<()> {
    let left_disk = open(left_dir)?;
    let right_disk = open(right_dir)?;
    compare_cells(&left_disk, &right_disk)?;
    let left_memory = materialize(&left_disk)?;
    let right_memory = materialize(&right_disk)?;
    let baseline = capture(&left_disk, p, false)?;
    for shared in [false, true] {
        for candidate in [&left_disk, &right_disk, &left_memory, &right_memory] {
            compare_results(&baseline, &capture(candidate, p, shared)?)?;
        }
    }
    Ok(())
}

fn oracle_array<'a>(root: &'a Value, name: &str, features: usize) -> Result<&'a Vec<Value>> {
    let values = root
        .get(name)
        .and_then(Value::as_array)
        .ok_or(Error::Invalid("oracle field is not an array"))?;
    if values.len() != features {
        return Err(Error::Invalid("oracle feature count differs"));
    }
    Ok(values)
}

fn oracle_number(value: &Value) -> Result<Option<f64>> {
    match value {
        Value::Null => Ok(None),
        _ => value
            .as_f64()
            .filter(|x| x.is_finite())
            .map(Some)
            .ok_or(Error::Invalid("oracle number is invalid")),
    }
}

fn oracle_integer(value: &Value) -> Result<usize> {
    value
        .as_u64()
        .and_then(|x| usize::try_from(x).ok())
        .ok_or(Error::Invalid("oracle integer is invalid"))
}

fn validate_oracle_results(
    results: &[MetaRegressionFeature],
    root: &Value,
    p: usize,
) -> Result<()> {
    let shape = root
        .get("shape")
        .and_then(Value::as_object)
        .ok_or(Error::Invalid("oracle shape missing"))?;
    let features = oracle_integer(
        shape
            .get("features")
            .ok_or(Error::Invalid("oracle feature shape missing"))?,
    )?;
    let parameters = oracle_integer(
        shape
            .get("parameters")
            .ok_or(Error::Invalid("oracle parameter shape missing"))?,
    )?;
    if features != results.len() || parameters != p {
        return Err(Error::Invalid("oracle shape differs"));
    }
    let coefficients = oracle_array(root, "coefficients", features)?;
    let ses = oracle_array(root, "ses", features)?;
    let q_fixed = oracle_array(root, "q_fixed", features)?;
    let n = oracle_array(root, "n", features)?;
    let df = oracle_array(root, "df", features)?;
    for index in 0..features {
        let result = &results[index];
        if result.n != oracle_integer(&n[index])?
            || result
                .diagnostics
                .as_ref()
                .map(|x| x.residual_df)
                .unwrap_or(result.n.saturating_sub(p))
                != oracle_integer(&df[index])?
        {
            return Err(Error::Invalid("oracle n or df differs"));
        }
        let oracle_coefficients = coefficients[index].as_array();
        let oracle_ses = ses[index].as_array();
        match (
            &result.estimates,
            &result.diagnostics,
            oracle_coefficients,
            oracle_ses,
        ) {
            (Some(estimates), Some(diagnostics), Some(coefficients), Some(ses)) => {
                if coefficients.len() != p || ses.len() != p || estimates.effects.len() != p {
                    return Err(Error::Invalid("oracle coefficient width differs"));
                }
                let standard_errors = estimates
                    .standard_errors
                    .as_ref()
                    .ok_or(Error::Invalid("missing marginal standard errors"))?;
                if standard_errors.len() != p {
                    return Err(Error::Invalid("standard error width differs"));
                }
                for ((coefficient, standard_error), (oracle_coefficient, oracle_se)) in estimates
                    .effects
                    .iter()
                    .zip(standard_errors)
                    .zip(coefficients.iter().zip(ses))
                {
                    let Some(oracle_coefficient) = oracle_number(oracle_coefficient)? else {
                        return Err(Error::Invalid("estimated oracle coefficient is null"));
                    };
                    let Some(oracle_se) = oracle_number(oracle_se)? else {
                        return Err(Error::Invalid("estimated oracle standard error is null"));
                    };
                    if !near(*coefficient, oracle_coefficient) || !near(*standard_error, oracle_se)
                    {
                        return Err(Error::Invalid(
                            "oracle coefficient or standard error differs",
                        ));
                    }
                }
                let Some(oracle_q) = oracle_number(&q_fixed[index])? else {
                    return Err(Error::Invalid("estimated oracle Q is null"));
                };
                if !near(
                    diagnostics.q_fixed.expect("estimated/common-effect Q"),
                    oracle_q,
                ) {
                    return Err(Error::Invalid("oracle Q differs"));
                }
            }
            (None, None, Some(coefficients), Some(ses)) => {
                if coefficients.len() != p || ses.len() != p {
                    return Err(Error::Invalid("unestimated oracle width differs"));
                }
                if coefficients
                    .iter()
                    .chain(ses)
                    .try_fold(false, |any, value| {
                        Ok::<bool, Error>(any || oracle_number(value)?.is_some())
                    })?
                    || oracle_number(&q_fixed[index])?.is_some()
                {
                    return Err(Error::Invalid("unestimated oracle Q is present"));
                }
            }
            _ => return Err(Error::Invalid("oracle result availability differs")),
        }
    }
    Ok(())
}

fn oracle(directory: &Path, oracle_json: &Path, p: usize) -> Result<()> {
    let root: Value = serde_json::from_slice(&std::fs::read(oracle_json)?)?;
    if root.get("method").and_then(Value::as_str) != Some("FE")
        || root.get("test").and_then(Value::as_str) != Some("z")
    {
        return Err(Error::Invalid("oracle is not FE z inference"));
    }
    let disk = open(directory)?;
    let memory = materialize(&disk)?;
    for cohort in [&disk, &memory] {
        for shared in [false, true] {
            validate_oracle_results(&capture(cohort, p, shared)?, &root, p)?;
        }
    }
    Ok(())
}

fn usage() -> ! {
    panic!("usage: generate DIR N F P BLOCK | bench DIR OUT_JSON disk|memory general|shared P BLOCK | compare DIR1 DIR2 P | oracle DIR ORACLE_JSON P")
}

fn main() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.first().map(String::as_str) {
        Some("generate") if args.len() == 6 => generate(
            Path::new(&args[1]),
            parse_usize(args.get(2), "N"),
            parse_usize(args.get(3), "F"),
            parse_usize(args.get(4), "P"),
            parse_usize(args.get(5), "BLOCK"),
        ),
        Some("bench") if args.len() == 7 => bench(
            Path::new(&args[1]),
            Path::new(&args[2]),
            &args[3],
            &args[4],
            parse_usize(args.get(5), "P"),
            parse_usize(args.get(6), "BLOCK"),
        ),
        Some("compare") if args.len() == 4 => compare(
            Path::new(&args[1]),
            Path::new(&args[2]),
            parse_usize(args.get(3), "P"),
        ),
        Some("oracle") if args.len() == 4 => oracle(
            Path::new(&args[1]),
            Path::new(&args[2]),
            parse_usize(args.get(3), "P"),
        ),
        _ => usage(),
    }
}
