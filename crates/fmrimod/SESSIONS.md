# Resident first-level sessions and cohort latency

The first runnable path now connects 4D NIfTI files to requested first-level
summaries, an existing group reducer, and a browsable synthetic group map.
This is an in-process resident session, not a durable cache service.

## Boundaries

- `fmridataset::NiftiSource` owns NIfTI-1 header validation, streamed decoding,
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

Contrast edits still scan resident time series and recompute needed residual
scales. Avoiding that work requires retained coefficients/scales or a qualified
projection cache; neither is silently introduced here.

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
must fit the configured resident budget; there is no automatic eviction, mmap
cache or spill-to-disk policy yet.

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

Qualification artifacts are in `crates/fmrireg/evidence/cohort/`. An independent
NumPy least-squares calculation verifies selected group-map coordinates across
all 25 subjects and all three edit scenarios. Worker-count comparisons cover
all 8,192 output values. Unit/integration tests cover file geometry/scaling,
reuse and stale-noise avoidance, no-copy loading, checked budgets and ordered
cohort/group inference.

Temporal projection is deferred: the current synthetic warm path is already
roughly 0.1 seconds for 25 subjects, while disk ingress dominates a new session.
The next speed work should measure realistic data/masks and compressed ingress
before adding a new algebraic cache with additional rank/df/error obligations.
