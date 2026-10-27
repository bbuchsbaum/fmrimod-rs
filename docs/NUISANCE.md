# Nuisance design and censoring

The numeric API in `fmri_design::nuisance` constructs nuisance columns on the
complete original acquisition grid. Build event and continuous-feature designs
on that same grid, compose them, then choose an explicit censoring operation.

```sh
cargo run -p fmri-design --example nuisance_design
```

The example combines events, sampled features, DCT drift, run intercepts and
motion covariates. It demonstrates both spike columns and matching row selection
for a design and response matrix. It does not fit an AR model.

## Drift and intercepts

`drift_design(frame, basis, intercept)` requires both choices explicitly:

- `DriftBasis::None`: no drift columns.
- `Polynomial { degree }`: the existing R-checked orthonormal polynomial basis
  evaluated at scan indices 1..N separately in each run, excluding the constant.
  Degree must be less than every run's scan count; zero is allowed.
- `Dct { cutoff_seconds }`: orthonormal DCT-II columns k=1 through
  floor(2*N*TR/cutoff), separately per run. The exact cutoff endpoint is included.
  Component zero is excluded, and a short run may produce zero columns. A cutoff
  at or below twice any run's TR is rejected rather than silently truncated.

`InterceptType::{None, Global, Runwise}` is independent of drift. A runwise
intercept is one column of ones inside its run and zeros outside. Drift columns
come first in run order; intercept columns follow. Column names identify run,
basis and component. Row times retain the frame's acquisition offsets. DCT
values use original scan indices, not an absolute acquisition-time cosine.
These are regression columns, not a filter applied to the response alone.

## Covariates

`nuisance_design(frame, runs)` takes borrowed `NuisanceRun` matrices and names,
one per run. Runs may have different column counts, including zero. Values are
placed in one block-diagonal output allocation without intermediate matrix
copies. Names are `nuisance.run<one-based run>.<supplied name>`; supplied names
must be nonblank and unique within a run. All input values must be finite.

The API preserves units and constant or collinear columns. It performs no hidden
centering, imputation or column removal. R's newer nuisance warning/drop/NA-repair
policies are **not** reproduced here. Check the complete design's numerical rank
and estimability before inference; use rank rather than the nominal column count
for residual degrees of freedom. Match covariate names and acquisition identity
at the caller boundary; matching dimensions alone cannot establish provenance.

## Two explicit censor choices

`CensorPlan::new(frame, censored)` accepts an original-grid Boolean mask where
**true means censored**. At least one observation must remain. An all-false mask
is valid. This convention belongs to this API; do not assume another API's mask
has the same polarity.

1. `spikes()` adds one unit column at each censored row and retains the complete
   design/response grid. These columns are not HRF-convolved or centered.
2. `select_design()` and `select_rows()` remove the same original rows from a
   preconstructed design and its responses. Retained values must be finite;
   excluded response values may be nonfinite. `retained_indices()` gives the
   original zero-based indices for selecting observation IDs and carrying run/gap
   metadata forward. Acquisition times are preserved, not renumbered.

Choose one of these operations for a particular fit. Do not recompute drift or
convolution on the selected grid, or append all-zero spike columns after selection.
The plan checks design time values and dimensions, but cannot prove subject or
run identity from those numbers. It deliberately does not manufacture a compressed
regular SamplingFrame or automatically adapt AR estimation/whitening.

For IID OLS, adding unrestricted spike coefficients is algebraically equivalent
to excluding those observations for the identifiable original coefficients and
residual sum of squares. The full design gains the spike rank while retaining
more rows, giving the same residual degrees of freedom. If an entire run is
excluded, its intercept becomes unidentifiable; simply counting columns gives
incorrect degrees of freedom. The tests exercise this case explicitly.

For correlated errors, an inference workflow must specify its covariance and
boundary policy. Noise-estimation censor masks, which exclude lag products and
restart filtering, are distinct from exclusion from the regression objective.
Do not apply a pre-existing regular-grid AR plan to compressed selected rows or
infer estimated-AR equivalence from the IID tests in this tranche.

## Audit and evidence

The audit inspected R fmridesign's `baseline_model.R`, including its newer
nuisance diagnostics and NA-repair behavior; Rust's existing baseline and
polynomial implementations; ScalaFIM's `DctDrift.scala`; and its
`ArCensorBoundaryGlsScenarioSuite.scala`. ScalaFIM provided the explicit DCT cutoff
policy and original-grid semantics. Its fit scenario reinforces that selected
observations require explicit covariance/run-gap handling. This tranche does not
port or qualify that Scala fit implementation.

Fresh fixtures come from an isolated install of the current local R fmridesign
source. Polynomial/intercept and covariate matrices use its public design
extraction (the nuisance term constructor is internal). R's direct DCT formula
is a mathematical reference, not a claimed R fmridesign DCT API. `lm.fit` supplies
an independent oracle for spike/row-selection coefficients, rank, RSS and df.

Tests use 2e-11*(1+abs(reference)) for matrices and coefficients, 1e-12 for RSS,
and exact rank/df checks. Independent DCT checks cover orthonormality, exclusion
of the constant, run isolation, cutoff endpoints and short runs. Invalid-input
checks cover nonfinite values, dimensions, names, timing and censor masks.
The complete example and existing design contracts are also run. Source hashes,
commands, exit codes and limitations are in `evidence/nuisance/verification.json`.
No regression/noise kernel, source repository or external publication is changed.
