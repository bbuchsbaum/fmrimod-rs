# Resident first-level sessions and cohort latency

The first runnable path now connects 4D NIfTI files to requested first-level
summaries, an existing group reducer, and a browsable synthetic group map.
This is an in-process resident session, not a durable cache service.

## Boundaries

- `fmri_data::NiftiSource` owns NIfTI-1 header validation, streamed decoding,
  conversion to f64 and exact spatial support. Enable `nifti` in fmrimod.
- `LoadedSubject` reads once and owns immutable data, observation IDs, feature
  geometry and a caller-supplied source version. Construction through a reader
  transfers its numerical buffer without a second full-data copy.
- `SubjectSession` retains one prepared OLS/GLS model. Output requests use the
  existing regression `OutputSpec`; no separate regression solver is introduced.
- `run_cohort` schedules bounded batches and delivers identified summaries in
  subject order. Each subject uses one regression worker.
- `ScalarCohortBuilder` copies one requested effect and optional variance into
  the existing `fmri-group` provider. It never manufactures t/p arrays or variance
  for an estimates-only result. The example uses group OLS.

## Session lifecycle

```rust,ignore
let source = NiftiSource::new(path, Some(mask_positions), Some(template_id))?;
let features = FeatureAxis::from_space(FeatureSpace::Volume(source.space().clone()))?;
let mut reader = source.open_reader()?;
let loaded = LoadedSubject::from_reader(
    &mut reader, observation_ids, features, subject_id, source_version, load_budget,
)?;
let mut session = SubjectSession::new(loaded);
session.prepare(&model_spec, preparation_budget)?;
session.run_blocks(&output_spec, block_features, execution_budget, |range, summary| {
    // Consume identified requested estimates/uncertainty; copy only if retaining.
    Ok(())
})?;
```

See the executable [cohort_latency example](examples/cohort_latency.rs) for full
imports, model specification, budgeting, group assembly and map generation.
`ModelSpec` contains the actual design, exact observation IDs, column semantics,
rank tolerance, matrix backend and complete `NoisePolicy`.

| Change | Data loading | Model preparation | Noise handling |
|---|---|---|---|
| Requested contrast/uncertainty | Reuse resident data | Reuse model | Reuse fitted plan |
| Identical ModelSpec | Reuse | Returns `Reused` | Reuse |
| Design, columns, tolerance or backend | Reuse | Rebuild | Re-estimate if policy is estimated AR |
| Fixed AR coefficients/startup/resets | Reuse | Rebuild | Use explicitly declared fixed plan |
| Estimated AR settings or feature sample/order | Reuse | Rebuild | Re-estimate |
| Data, observation selection/order or geometry | Construct a new LoadedSubject | New model | New policy evaluation |

Exact structural comparison is session-local; it does not infer equality from
user labels or hashes. Failed preparation retains the previous valid model and
its receipt. Callers must handle the error rather than treating that old model
as the failed requested design. Generation, rank, df, noise coefficients and
estimation diagnostics accompany results.

`EstimateAr` is currently restricted to tolerance exactly `1e-12`, matching the
existing correction estimator's rank threshold. `noise_valid` affects noise
estimation and filtering resets; it does **not** censor regression observations.
Selected-observation GLS remains available through the existing regression APIs,
but the new session policy does not yet wrap that specialized path. Custom
mutable whitening is deliberately outside this first session API.

`conditional_noise` describes all GLS uncertainty conditional on its plan.
`estimated_noise` separately records whether that plan was estimated; the group
bridge uses the latter for approximation provenance. No requested AR model is
replaced by IID errors.

By default, contrast edits scan resident time series and recompute needed
residual scales. `retain_for_edits` explicitly opts into coefficients plus scales
for subsequent contrast edits; requested outputs remain controlled by OutputSpec.
No temporal projection or squared-norm residual shortcut is used.

## Memory and scheduling

Budgets bound documented numerical allocations, not OS RSS. Counts include
loaded f64 data, cached model/spec matrices, AR startup/receipt arrays, known
preparation temporaries, contrast validation/geometry, workspaces and owned
summaries awaiting delivery. Model replacement reserves old and new state.
Logical-only clones prevent sliced arrays from retaining hidden backing elements.

Allocator excess capacity, semantic/index metadata, thread stacks, decomposer
private scratch and backend packing remain excluded. Reader estimates include
output plus two raw 3D frames; compressed decoder/header metadata is excluded,
with NIfTI extension offset explicitly capped at 1 MiB above the header.
`from_f64` canonicalizes a caller-supplied array outside the reader budget; the
caller must allow that input-construction copy.

`prepare` receives a budget excluding loaded subject data but including old model
overlap. `run_required_bytes` reports additional execution/retention reservation;
`resident_numeric_bytes` reports persistent subject/model storage. `run_cohort`
adds all subjects' resident reservations and the largest active batch's job
reservations. A fixed batch holds at most `workers` owned results, even when an
early subject is slow. An error joins active jobs and stops later callbacks/batches;
previously delivered outputs are not rolled back.

The scalar collector has a separate `numeric_bytes()` reservation. Subtract it
from the pipeline limit supplied to `run_cohort`, as the example does. Arbitrary
copies made by external sinks are outside the library budget. All loaded subjects
must fit the configured resident budget for `run_cohort`. Use `run_file_cohort`
when that is inappropriate: it loads bounded batches and releases their data
after fitting, retaining prepared models. There is no automatic mmap or durable
spill-to-disk cache.

For Accelerate, set `VECLIB_MAXIMUM_THREADS=1` before process start when running
multiple subject workers. The library never changes global thread settings.
OLS views with padded row strides currently use the portable matrix fallback;
GLS transforms into contiguous workspace. This is a remaining optimization
opportunity, not a changed numerical policy.

## NIfTI admission

Ingress supports single-file `.nii`/`.nii.gz`, scalar supported datatypes,
explicit millimeter units, four dimensions and valid sform or qform geometry.
It does not infer MNI registration, resample images, or invent unknown affine
metadata. Arbitrary unique masked voxel support retains x-fastest spatial
identity and requested order. Nonstandard spatial units and paired/NIfTI-2 files
are rejected in this initial path.

Header-only inspection bounds the extension gap before streamed opening; source
metadata is rechecked when opening and reading. Relevant float header fields
compare by bits, so unchanged NaNs in unused fields do not invalidate a file.
Qform behavior follows the [NIfTI reference implementation](https://github.com/NIFTI-Imaging/nifti_clib/blob/master/niftilib/nifti1_io.c).
NIfTI sources are not content-hashed by the library: source versions and scientific
registration/estimand admission remain caller assertions. Immutable loaded data
is the session snapshot; modifying a file does not mutate an existing session.

Each independent reader request reopens a streamed decoder. Load a subject once
for warm edits; repeated small feature reads of `.nii.gz` would repeat decoding.
Persistent decompression caches are a later optimization.

## Reproduction and evidence

```sh
OPENBLAS_NUM_THREADS=1 python3 crates/fmrimod/scripts/generate_cohort.py /tmp/cohort
CARGO_TARGET_DIR=/tmp/fmrimod-session-target cargo build -p fmrimod \
  --release --features nifti,accelerate --example cohort_latency
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/fmrimod-session-target/release/examples/cohort_latency /tmp/cohort /tmp/results 4
```

The generator writes synthetic float32 NIfTI files; no research data is included.
The example writes `cold.html`, `design-edit.html`, `contrast-edit.html` and a
latency receipt. Browser rendering time is excluded. The static page has a slice
slider and is not a clinical viewer or a registration-quality check.

Qualification artifacts are in `evidence/regression/cohort/`. An independent
NumPy least-squares calculation verifies selected group-map coordinates across
all 25 subjects and all three edit scenarios. Worker-count comparisons cover
all 8,192 output values. Unit/integration tests cover file geometry/scaling,
reuse and stale-noise avoidance, no-copy loading, checked budgets and ordered
cohort/group inference.

Temporal projection is deferred: the current synthetic warm path is already
roughly 0.1 seconds for 25 subjects, while disk ingress dominates a new session.
The next speed work should measure realistic data/masks and compressed ingress
before adding a new algebraic cache with additional rank/df/error obligations.

## Explicit loading and retention policies

`ingress::NiftiSubject` binds a source, exact observation IDs, subject ID and
immutable source version. `load_subjects(&sources, LoadOptions { workers,
max_numeric_bytes })` preflights all batches, then loads in parallel and returns
resident `SubjectSession`s in input order. Its reservation includes previously
loaded data plus every active reader's peak. It joins active jobs on failure and
returns no partial cohort. Float32 ingress uses the NIfTI provider's typed frame
decoder; scaling still occurs in f64 with provider-equivalent zero-slope behavior.
Integer and other supported scalar inputs retain the existing accessor path.

For cohorts larger than RAM, construct one `FileSession` per `NiftiSubject`, then
call `run_file_cohort(&mut sessions, &model_specs, &output_specs, options, sink)`.
It reduces the requested worker count until active data and numerical work fit
the budget. If one subject cannot fit, it fails before voxel reads. The scalar
group collector remains a separate reservation, as in the resident executor.
The preflight includes old and replacement models when a model changes; reused
models are counted once. It is not an
allocator-exact RSS bound. Caller-owned source metadata and input model specs are
outside that reservation.

File sessions release time-series arrays after each job but retain prepared
models, including estimated noise. A contrast edit therefore reloads data but
reuses the model. A design edit rebuilds it and re-estimates noise when requested.
These are immutable file snapshots by contract: source versions are caller
assertions. To change voxel values, rows or geometry, construct a new file
session. Header changes are checked; voxel content is not automatically hashed.
On failure, completed subjects can have updated preparation state; later batches
and callbacks stop. Source reload failures preserve prior preparation state.

Lower-level callers can `detach()` a resident session to release its time series
and later `attach(loaded_subject)`. If coefficients/scales were retained,
`DetachedSession::run_blocks` and `run_owned` execute output requests directly
without reattachment or time-series reads. They reject a detached session without
a retained fit. Their budget covers output preparation, summary work and owned
results; it excludes the already resident detached cache. Attachment checks subject/source version,
observation order, dimensions and feature geometry. A mismatched attachment is
rejected and consumes the detached state; construct a new session for new data.

For repeated contrast exploration on a fixed design, use:

```rust,ignore
let required = session.retention_required_bytes(block_features)?;
session.retain_for_edits(block_features, required)?;
// Subsequent run_owned/run_blocks calls use coefficients and residual scales.
// OutputSpec still determines which effects and uncertainty leave the session.
session.clear_retained(); // Return to scanning time series.
```

The caller must reserve `required` in addition to current resident data/model
storage. Retention costs `(p + 1) * features * 8` bytes plus temporary fitting
workspace while building it. It intentionally computes residual scales even if
the current output request needs only estimates. Repeating retention is a no-op.
Successful model replacement clears retained fits; failed preparation preserves
the old model and cache. Default sessions retain neither coefficients nor scales.
The regression-level `RetainedFit` carries an opaque exact-model identity, so
`PreparedOutputs::summarize_retained` cannot combine it with another model. This
is an in-process cache, not a serialized or cross-session compatibility promise.

## Larger workload qualification

The example accepts `COHORT_BUDGET_MIB` (default 1024), `COHORT_LOAD_WORKERS`
(default 1), `COHORT_NOISE=fixed|estimated`, `COHORT_RETAIN=1`, and
`COHORT_STREAMING=1`. Streaming runs keep models and release data; retention is
only used in the resident experiment. Estimated AR uses an explicit global AR(4)
policy on up to 256 spatially spread features with correction lag 8. This is a
qualified example policy, not an assertion that 256 features suffice for every
dataset. The generator accepts `--shape X Y Z`, `--frames N`, and `--gzip`.

```sh
OPENBLAS_NUM_THREADS=1 python3 crates/fmrimod/scripts/generate_cohort.py \
  /tmp/cohort-large --shape 64 64 32
VECLIB_MAXIMUM_THREADS=1 COHORT_BUDGET_MIB=12288 COHORT_LOAD_WORKERS=4 \
  COHORT_RETAIN=1 /usr/bin/time -l /tmp/fmrimod-session-target/release/examples/cohort_latency \
  /tmp/cohort-large /tmp/results-large 4
# For a 1 GiB numerical budget, use COHORT_STREAMING=1 COHORT_BUDGET_MIB=1024.
```

See `evidence/regression/whole-brain/` for source hashes, independent NumPy
checks, complete map comparisons, timing variability, and measured peak memory.
The input has a whole-brain-sized feature count, but is synthetic, has no brain
mask or verified MNI registration, and cannot establish real-data throughput.
A durable decompressed/masked cache remains a separate decision: warm resident
requests already avoid disk, and file sessions explicitly trade repeated loading
for a smaller footprint. Temporal projection remains deferred.


The higher-level [analysis planner](ANALYSIS.md) supports `StoragePolicy::RetainedFits`
for cohorts. It fits/reloads bounded subject batches, releases time series, and
reuses detached fits for contrast edits. `run_with_progress` permits cooperative
cancellation between batches and subject preparations. Group consumer adapters
keep first-level input uncertainty separate from requested group-output uncertainty.
