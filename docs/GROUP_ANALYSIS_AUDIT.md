# Group analysis in fmrimod: source audit and integration proposal

Audit date: 2026-10-27. Status: **assessment complete; tranche 1 implemented**.
See [GROUP_ANALYSIS.md](GROUP_ANALYSIS.md) for the delivered contract and limits,
and [GROUP_BENCHMARK.md](GROUP_BENCHMARK.md) for measured comparisons.
Mote: `bd-01M5YK74R39H0EF9BMFSX58GND`.

## Decision

Add a separate **`fmri-group`** crate, exposed through `fmrimod::group`, using
`fmridataset` for aligned axes and bounded reads and `fmrireg` for appropriate
linear algebra. Do not turn `fmrireg` into a group workflow package. Do not copy
the R three-dimensional object model or its mutable string-keyed reducer registry.
The useful scientific abstraction is **identified estimates with uncertainty,
assembled into independent or repeated observations**, evaluated over feature blocks.

The immediate priority is the first-level-to-group contract, followed by core
OLS and inverse-variance meta-analysis. Most risks occur before the numerical
reduction: joining the wrong estimands, treating runs as independent participants,
losing covariance, mismatching spatial support, or silently changing the tested
family when blocking. A fast reducer cannot repair those mistakes.

This extends the programme beyond the original first-level scope in `PLAN.md`;
it does not authorize an implementation of every fmrigds capability. No Rust
numerical kernels or R sources were changed in this audit.

## Evidence and boundaries

Reference checkout: `~/code/fmrigds`, HEAD
`273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`, version 0.1.0.9000.
Its untracked `UPSTREAM-ISSUES-from-fmrireg.md` was preserved. The source manifest
records file hashes, Git state, Rust interface hashes, and selected Scala references.

A private source copy was freshly compiled and installed using R 4.5.1. The
17 executable witnesses in `scripts/audit_group_reference.R` exercise public
constructors/plans and current compiled kernels. Results, session details and
build log are in `evidence/group-audit/`. Comparisons use relative tolerance
1e-10 for small, well-conditioned double-precision fixtures. Seeds are fixed.
Source-matched native/R agreement is distinguished from independent analytic,
base-R QR, and dense-covariance oracles.

This is not an exhaustive R package certification, an LMM optimizer calibration,
or a benchmark. Adapters, advanced diagnostics, LDR, and post-hoc methods received
source/contract review, not complete runtime qualification. No Rust speedup is
claimed. Existing whole-workspace tests were not rerun for this documentation
and standalone R witness change.

## What fmrigds actually provides

| Capability | Actual source contract | Rust disposition |
|---|---|---|
| Group data | `new_gds`: sample × subject × contrast arrays, named assays, subject/sample metadata, space and provenance | Reuse dataset axes; add estimate/observation semantics, not a mandatory cube |
| Sources and lazy plans | Memory, table, NIfTI, HDF5 and other adapters; inspect/open/read/close; plan transformations | Preserve probe/read separation and owned lifetimes; format adapters remain outside numerical core |
| OLS | Per-feature finite-effect deletion; unweighted model; estimated residual variance; t inference | First tranche; reuse prepared SVD models per missingness pattern |
| Common-effect meta-analysis | Genuine positive sampling variance; inverse variance; normal inference; Q/I² | First tranche, explicit known-as-input variance interpretation |
| Random-effects meta-analysis | DL heterogeneity; reweight by v+tau²; plug-in normal inference | First tranche with explicit heterogeneity/inference policies |
| Meta-regression | FE and DL regression with feature-dependent weights; full-rank checks; normal inference | Next tranche, stable weighted QR/SVD; do not reuse a single unweighted factorization |
| Repeated measures | Random intercept, or intercept plus one slope; one grouping factor; shared balanced layout | Dedicated prepared repeated-model API, later tranche |
| Known-variance repeated models | diag(v) + residual heterogeneity I + ZGZ'; voxelwise covariance estimation | Preserve separately from unweighted LMM; later add block sampling covariance |
| Evidence combination | Stouffer, Fisher, Lancaster | Separate evidence API; cannot manufacture effect size/variance |
| Permutations | One-sample sign flips; two-sample label permutations; max-T within supplied feature family | Later, with explicit exchangeability and family scope |
| Spatial transformations | Subject maps, linear aggregation and diagonal/full-provider uncertainty paths | Sparse support-aware transforms; explicit uncertainty policy |
| Multiplicity | BH/BY and spatial Simes + size-weighted group BH; optional spatial integrations | Global terminal operation, declared family and target of error control |
| Examination | Model-conditioned surprise, leverage/deletion, exact retained-subject refits, conclusion sensitivity, optional Huber sensitivity | Valuable second wave; typed diagnostics tied to fitted model |
| Experimental cancellation / LDR | Genuine paired-split uncertainty, voxel geometry, held-out subject cross-fitting, displacement integration and calibration | Separate experimental layer after core qualification; not generic group GLM |
| Provenance/export | Versioned entities/activities/edges, portable plan/design/reduction receipts, source verification | Preserve scientific identities and explicit provenance; choose wire contract separately |

Important source locations: `R/gds-class.R`, `adapter-contract.R`, `compute.R`,
`scan-exec.R`, `reducers-core.R`, `reduce-exec.R`, `reducers-lmm.R`,
`reducers-lmm-knownvar.R`, `reducer-lmm.R`, `src/reducers_core.cpp`,
`src/lmm_core.cpp`, `examine-group.R`, `examination-compile.R`,
`examination-robust.R`, `local-displacement-rescue.R`, `ldr-inference.R`.

The R implementation already contains useful low-rank repeated-model algebra:
subject blocks and Woodbury-style updates avoid full observation covariance
inversion. Preserve that structure. Conversely, `ols_voxelwise_cpp` is presently
an **R function**, despite its name, using normal-equation solves and per-feature
loops. The FE-regression registry function is also R. Audit execution paths,
not names, before making performance comparisons.

## Scientific distinctions the Rust API must retain

1. **OLS is not common-effect meta-analysis with all variances set to one.** The
   coefficient can coincide, but OLS estimates residual scale and uses residual
   degrees of freedom; common-effect inference treats input variance as fixed.
   On effects [0,2,4,6] with variances [1,1,4,4], OLS gives 3 and meta-analysis
   gives 1.8 with variance 0.4. Share algebra, not an undifferentiated weight option.
2. **First-level estimated variance is not literally known.** Carry its origin,
   scalar or featurewise df, noise estimator and approximation status. A meta
   method may intentionally treat it as fixed, but that is an inference policy,
   not permission to erase its origin. First-level df do not automatically
   become group df; do not invent an adjustment formula.
3. **Within-subject covariance is independent of between-subject heterogeneity.**
   Known-variance LMM uses v + additional residual heterogeneity + random effects.
   Correlated contrasts need their joint sampling covariance. Two unit-variance
   contrasts with correlation 0.8 have difference variance 0.4, not 2.
4. **Independent subject rows differ from repeated observations.** Multiple runs,
   visits, contrasts and trials from one participant cannot silently increase
   the group sample size. Compile an independent cohort or a repeated layout.
5. **Missingness is method-specific.** R OLS/meta methods delete individual rows
   per feature; current LMM rejects an entire feature if any repeated observation
   is missing. Keep these policies explicit. A missing-data bitmap is not proof
   that deletion is unbiased; the scientific missingness assumption remains.
6. **Evidence, effects and decisions are different outputs.** A combined p-value
   is not an effect; group-level spatial q-values broadcast to voxels do not
   become voxel-level FDR guarantees; robust sensitivity is not an outlier
   probability or an automatic participant exclusion rule.

## Reproduced hazards and deliberate limitations

These witnesses document observed behavior at the pinned source. They are not
silent requests to modify fmrigds, and not all are bugs.

| Witness | Observed result | Integration requirement |
|---|---|---|
| `map_zero_times_na` | Public map [1,0] applied to [2,NA] returns NA | Iterate nonzero support; missing values outside support must not affect output |
| `mapping_covariance` | Equal average variance 0.5 under independence, 0.9 with correlation 0.8 | Diagonal propagation is an explicit assumption, not generic variance preservation |
| `provider_scope` | Provider receives feature indices only; same returned covariance used across subjects/contrasts, irrespective of their `var` | Provider must receive identified observation/estimand/support, or declare shared covariance |
| `affine_compatibility` | Default space comparison accepts a 10-unit affine translation | Exact geometry/support checks plus explicit registration/admission evidence |
| `contrast_collision` | Public harmonization permits two contrasts to become `same` | Validate resulting IDs; renaming cannot establish estimand equivalence |
| `absolute_floor` | Rescaling effect by 1e-8 and variance by 1e-16 changes z from 3.4641 to 0.034641 | Avoid hidden dimensional variance floors; scale arithmetic and report unsupported extremes |
| `effect_only_constructor` | `new_gds` rejects beta without var/se although OLS does not consume variance | Genuine effects-only type; no fabricated variance to satisfy a container |
| `pooled_scope` | Pooled RI lambda ≈608.85 jointly, ≈2213.91 and 0 when features fitted separately | Fit pooled shape over a declared set, then apply it; block size must not change the model |
| `lmm_missing` | One missing repeated value makes the feature unestimable | Document complete-layout initial support; ragged fitting is a separate extension |
| `max_t_family` | Six-feature max-T p-values exceed singleton-family results | Reduce permutation maxima across the entire declared family before calculating FWER |
| `valid_zero` | Zero retained by default; explicit background policy masks all assays together | Preserve valid zero, separate availability from values |

`propagate_variance_independent` and mapping execution also convert sparse
operators to dense matrices. A 100,000 × 100,000 f64 dense operator alone is
80 GB (decimal). Rust should consume sparse rows or an operator, with bounded
support reads; that is an allocation argument, not a measured speedup.

`compute()` currently reads requested arrays then applies plan nodes. It does
not itself implement a general blockwise group engine. The examination scanner
has direct-block and staged/target-support strategies. Keep this distinction:
lazy construction is not a bounded-memory execution guarantee. The pooled/max-T
witnesses demonstrate what a naive streaming translation would change, not a
claim that current full-array `compute()` has that specific block bug.

Current LMM inference is plug-in covariance with t(n-p), explicitly without
Satterthwaite/Kenward–Roger corrections. The known-variance optimizer uses bounded
log-scale components; it does not represent exact zero components in the same
way as RI's explicit zero-lambda branch. Rust needs boundary/convergence status
and calibration before promoting those approximate p-values.

## Rust representation and crate boundaries

### Reuse the existing foundation

`fmri_data::FmriFrame` already binds observation × feature assays to exact ordered
IDs. `FeatureAxis` binds support and geometry; `FeatureSpace::assert_compatible`
compares semantic values, not dimensions alone. Reuse this instead of introducing
an unrelated `GroupSpace`. Index/volume spaces exist; full surface/mesh semantics
are a future dataset capability, not something label-only arrays can assert.

`fmri_reg::LinearModel` already has rank-aware SVD, reusable block workspaces,
contrast estimability and unit covariance. Reuse it for complete-case group OLS
and cache preparations for repeated validity patterns. Set a cache budget;
worst-case every feature has a distinct pattern. Never fill missing effects with
zero. A feature-local failure should produce a status and missing result, not
abort unrelated features; malformed axes/design are whole-request errors.

`GlsModel<P>` handles a prepared observation covariance. Feature-dependent
sampling variance generally changes X'WX at each feature; it cannot be handled
by one common whitening plan. Weighted meta-regression needs its own feature
loop and bounded factorization workspace. Do not alter AR/OLS hot paths to make
this integration happen.

### Proposed nouns (design sketches, not existing public symbols)

- `EstimateSet`: identified first-level scalar estimands, feature axis, observation
  keys, effects, optional uncertainty, validity and fit receipt. Start in-memory;
  design a versioned portable envelope only after the contract is tested.
- `ObservationKey`: dataset-qualified participant plus visit/run/measurement ID.
  `IndependentCohort` validates one admitted row per participant; `RepeatedCohort`
  owns grouping and observation ordering explicitly.
- `EstimandDefinition`: stable ID, display label, effect units, scientific target,
  contrast definition and model provenance. Different first-level parameterizations
  may represent the same scientific target: use an explicit audited equivalence
  mapping rather than requiring identical design hashes or accepting equal labels.
- `EffectBlock<'a>` and `VarianceBlock<'a>`: borrowed observation × feature views
  with aligned validity. Constructors validate schema; readers validate each block.
  Effects-only OLS and variance-required meta methods have different inputs.
- `SamplingUncertainty`: absent, marginal variance, or within-observation joint
  covariance, accompanied by `Known`/`Estimated`/`Unknown` origin and df metadata.
  Do not allocate dense feature × feature covariance. Distinguish covariance
  across contrasts at one feature from spatial covariance across features.
- `GroupDesign`: bound observation IDs, named columns and estimable contrasts;
  compiled once. Formula support is an optional front end to an explicit matrix.
- `PreparedGroupModel`: validated model, missingness policy, requested estimands,
  inference policy and execution scope. Start with concrete estimator structs;
  a small closed enum can drive orchestration. No mutable global registry.
- `GroupResultBlock`: identified feature range, requested effect/SE/statistic,
  distribution/df, n/rank, optional Q/I²/tau² and typed per-feature status.
  No artificial subject named `meta`, no coefficients encoded in assay-name strings.

Metadata binding precedes numeric reads. Never use string labels in inner loops.
Borrow ndarray views, reuse owned scratch, and emit through a sink. Dynamic
traits are appropriate at source/sink boundaries; numerical loops should not
need cell-level trait objects. Preserve license boundaries: dataset/facade are
GPL-3.0-or-later today; existing numerical crates have their own licenses. The
MIT fmrigds reference does not change licenses of dependencies.

### First-level handoff

For scalar t contrasts, current `TContrast` has effect, SE, statistic, p and df;
effect and SE² provide the marginal group inputs. The new bridge must attach
feature identity, participant/measurement identity, estimand definition, units,
fit/noise/censoring provenance and valid/failed status. An F map supplies no
signed scalar effect and must not be admitted as though it did.

For several contrasts from one fit, preserve C Cov(beta) C'. OLS/common-known-GLS
can exploit shared covariance shape times each feature's residual scale. For
feature-varying estimated noise, retain covariance from the actual fit path;
do not attach a shared approximation without naming it. Retain only requested
contrast pairs or a justified factorization; avoid quadratic storage in every
model coefficient. Marginals suffice for separate univariate group analyses,
not a joint repeated-contrast likelihood.

Identical affines/support do not prove anatomical registration. Require admission
metadata identifying the common target and transformation provenance; automated
validation can enforce consistency but cannot infer scientific alignment.

## ScalaFIM comparison

Read current source at `ac2ec8b706fce08596a943d79f71a140aad05948`, notably
`GroupResponse.scala`, `GroupWeighting.scala`, `EstimateGroup.scala`, and the
estimate metadata encoder (see manifest). This is source review, not a Scala run.

Use its effects-only/variance-bearing distinction, explicit statistic distribution,
and estimate admission boundary. Its estimate bridge validates matching effect
and uncertainty axes/pooling, pinned units, participant uniqueness, common
catalog/units, geometry admission, and scalar or sample-dependent uncertainty df.
It opens one source at a time and counts effects/variances/df against a block
budget. These are stronger integration precedents than an R cube translation.

Its current weighting algebra also separates DL/Paule–Mandel heterogeneity from
normal/modified-Knapp–Hartung inference. Reserve that separation in Rust now;
implement and qualify the extra estimators later. Do not copy the whole Scala
framework: its current bridge is strict complete-case and rereads integrity data
per block. Rust should support explicit per-feature omission where statistically
intended and pin/verify immutable source handles without redundant whole-file
scans. Those are design objectives, not implemented guarantees.

## Execution architecture and performance acceptance

Keep kernel fitting, global fitting, and terminal inference distinct:

- Local OLS/FE/DL: stream features; OLS prepares once per validity pattern;
  scalar FE/DL uses stable passes over subjects. With B features, N observations,
  P terms, effects/variances require O(NB) working values; general weighted fits
  need O(NP + P²) scratch per worker in addition to the block.
- Requested contrasts: compute from factor solves and discard full coefficient
  covariance unless the caller requests later contrasts. Retaining packed
  per-feature covariance costs O(BP²) and must enter the budget explicitly.
- Pooled LMM: declare fit features and objective scope; accumulated objective
  evaluations or staged data may require repeated passes. Prepare the common
  fitted shape once, then apply it. A per-block optimum is a different model.
- Permutations: generate a deterministic shared schedule keyed by participant,
  preserve it across blocks, accumulate one maximum per permutation across the
  family, then finalize. Nuisance handling/repeated exchangeability require their
  own algorithms; they are not arbitrary group GLM options.
- BH/BY: global sorting/selection over a declared set, with an explicit missing-p
  policy. Use in-memory collection or an external/staged algorithm; never apply
  separately to execution blocks and call the result whole-map correction.
- Sparse mapping: bounded support gathers, operator identity and explicit missing
  support policy (reject, propagate missing, or documented renormalization).
  Renormalization changes the estimand; it cannot be an invisible optimization.

Benchmark equivalent outputs against the freshly built native R paths, not just
R wrappers or allocations. Proposed workloads: N=20/100/500, features=1k/100k,
P=1/5/20, complete and patterned missingness, f32 input/f64 accumulation, a
1-feature tail block, one and multiple workers. Report setup, warm fit, I/O,
peak RSS, all retained buffers and threads; preserve correctness receipts. There
is no speed target justified by measurements yet. Success includes block-size
invariance and memory scaling with B rather than total feature count.

## Ordered implementation plan and acceptance gates

Delivery update: tranche 2 is implemented. The repeated-model core from tranche 3
and permutation/exact-deletion pieces from tranche 4 are now implemented and
qualified within the explicit limits in [COMPLETION_REPORT.md](COMPLETION_REPORT.md).
Sparse spatial covariance transforms, pooled covariance fits, general predictive
diagnostics and robust sensitivity workflows remain separate extensions.

### Tranche 1: useful independent-subject analysis (five deliverables)

1. **Estimate/cohort contract and first-level bridge.** In-memory effects-only and
   marginal-variance inputs, unique participant/estimand keys, units/geometry
   admission, fit/uncertainty receipts and explicit missingness. Reject duplicate
   participants, permuted variance axes, incompatible space, label-only joins
   and F-as-effect. Preserve valid zeros. Keep the bridge outside AR/OLS kernels.
2. **Bounded group OLS.** Intercept, two-group and covariate models, named linear
   contrasts; pattern-cached SVD with bounded fallback. Verify base-R QR,
   rank/estimability, per-feature df, row permutations, singleton blocks and
   independent missingness masks. No fake variance required.
3. **Scalar common-effect and DL meta-analysis.** Stable weights, genuine variance
   validation, effect/SE/z, Q/I²/tau² and n/status. Verify analytic two-study cases,
   fresh native R outputs, zero heterogeneity, unit scaling and invalid variances.
   Report plug-in normal inference explicitly.
4. **Typed result sinks and BH/BY terminal inference.** Preserve family identity,
   per-feature status, source/fit receipt and requested outputs. Verify p-values
   and q-values with base R, all-missing and tied values, whole-family block
   invariance. No file-format proliferation in this tranche.
5. **End-to-end numerical/performance qualification.** Synthetic first-level fits
   through cohort assembly to group contrast; independent references, deliberate
   alignment failures, bounded source/sink tests and equivalent-output benchmarks.
   Include normal-null simulation as measured calibration, not a proof. Update
   capability/qualification docs with actual supported limits.

Do not make full covariance, arbitrary mixed models, NIfTI/HDF5 writers, or a
query planner prerequisites to delivering this tranche.

### Tranche 2: covariates, richer inference, and portable estimates

FE/DL meta-regression with stable weighted factorizations; Paule–Mandel and
modified Knapp–Hartung as independent explicit policies with fresh metafor
references and finite-sample simulations. Carry group coefficient/contrast
covariance only on demand. Define a versioned estimate interchange profile,
reader validation and one practical source/sink; interoperating with Scala/R
requires schema fixtures, not a claim that their internal serialization matches.

### Tranche 3: repeated measures and covariance-aware transforms

Known marginal and within-participant block sampling covariance; RI then one
slope; explicit ML/REML and pooled/featurewise scope. Dense likelihood oracles,
boundary/singularity diagnostics, subject-block permutation/order invariants,
missing-layout acceptance/rejection and inference calibration. Implement sparse
linear maps with support and covariance providers that identify observation and
estimand. Exact zero variance components and optimizer convergence need witnesses.

### Tranche 4: examination and resampling

Model-bound predictive/influence diagnostics, exact deletion checks, fixed-versus-
refitted tau² semantics, robust sensitivity without silently changing the primary
fit; exchangeability-aware permutations and whole-family max-T. Keep subject
review and automated exclusion separate. Spatial group-level procedures must name
their inferential target and have tests beyond comparison to their own formula.

### Tranche 5: experimental spatial methods, only when needed

Paired-split LDR/cancellation, cross-fitting and calibration, topology-aware
operations, reports and additional adapters. Preserve split independence,
training-only selection and held-out scoring. These deserve dedicated scientific
qualification and should remain outside the minimal group engine.

## Upstream reports and benchmark contract (2026-10-27 follow-up)

Four public-API reproductions were rerun in a fresh `Rscript --vanilla` process
and reported with explicit user authorization after checking existing issues:

- [#36](https://github.com/bbuchsbaum/fmrigds/issues/36): missing source values outside map support contaminate targets.
- [#37](https://github.com/bbuchsbaum/fmrigds/issues/37): default compatibility accepts different voxel affines.
- [#38](https://github.com/bbuchsbaum/fmrigds/issues/38): contrast renaming creates duplicate IDs.
- [#39](https://github.com/bbuchsbaum/fmrigds/issues/39): absolute variance floor changes inference under unit conversion.

Standalone reproductions, output, posted bodies and issue receipts are retained
under `evidence/group-audit/github/`. These reproduce the pinned local build;
current GitHub source was inspected, not rebuilt. The default branch differs
from the local reference: notably its covariance-provider path now derives
correlation from the provider and applies each subject/contrast's marginal
variance. Consequently the earlier `provider_scope` witness describes the local
SHA only, not current default-branch behavior. That limitation was not filed as
a new defect. The four filed defects remain in the inspected relevant source.

Before numerical implementation qualification, pin the selected upstream revision
and freshly build it again; preserve this audit baseline separately. Do not
update the reference or import changed numerical behavior silently.

The comparison suite will distinguish correctness oracles from speed baselines:

| Operation | Primary independent correctness oracle | Performance baseline |
|---|---|---|
| Group OLS / contrasts | Base R `lm.fit` QR and analytic cases; explicit retained rows and rank policy | fmrigds public OLS, including its actual R implementation; Rust prepared and end-to-end paths separately |
| Common effect, DL, FE/DL meta-regression | Fresh `metafor` fits with matched method, intercept/design, inference, variance and missingness policy; analytic cases | Fresh fmrigds public/native paths, reporting which path runs |
| BH/BY | Base R `p.adjust` on the identical declared family and missing-value policy | Whole-family fmrigds and Rust terminal operations |
| Known-covariance GLS / LMM algebra | Dense Cholesky/solve and log-determinant calculations | fmrigds compiled low-rank kernels at identical fixed covariance parameters |
| Estimated unweighted repeated LMM | `lme4` where the likelihood/design match, plus dense objective checks | fmrigds with matching ML/REML, parameter bounds and pooled/featurewise scope |
| Known-sampling-variance LMM | Dense likelihood oracle; `metafor::rma.mv` only after matching the exact covariance parameterization | fmrigds known-variance path; optimizer work and fit precision reported |

An oracle need not be the fastest baseline. Equal results must be established
before timing; where a reference has a reproduced defect, retain a named
expected disagreement and use the independent oracle rather than copying the
defect. For approximate inference, numerical agreement does not establish
calibration: use separate null/coverage simulations with Monte Carlo uncertainty.
Record versions/source hashes, data seed, output sets, threading, warm-up,
repetitions, timing distributions and peak memory. Measure preparation, repeated
fit and end-to-end read/fit/write separately. No timings were collected in this
reporting follow-up.

## Reproduction commands

Build from an isolated copy, excluding prebuilt native objects, so the reference
checkout is not modified:

```sh
mkdir -p evidence/group-audit/r-library
copy_dir=$(mktemp -d /tmp/fmrigds-audit.XXXXXX)
rsync -a --exclude=.git --exclude='*.o' --exclude='*.so' --exclude='*.dylib' \
  "$HOME/code/fmrigds/" "$copy_dir/"
LC_ALL=C R CMD INSTALL --preclean --no-multiarch \
  --library="$PWD/evidence/group-audit/r-library" "$copy_dir" \
  > evidence/group-audit/install.log 2>&1
LC_ALL=C Rscript scripts/audit_group_reference.R \
  > evidence/group-audit/witness.log 2>&1
```

Confirm source hashes against `evidence/group-audit/source-manifest.json` before
calling a rerun a reproduction. The witness script intentionally asserts several
current adverse behaviors; a repaired upstream reference should make those
assertions fail and trigger review, not be silently accepted as the same reference.
The unit-variance arrays used to satisfy the R constructor in OLS/LMM fixtures
are test plumbing; the unweighted likelihood does not consume them.
