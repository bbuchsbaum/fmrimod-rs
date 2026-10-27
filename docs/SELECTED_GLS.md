# AR-GLS with excluded observations

`fmri_noise::SelectedArPlan` provides known-parameter AR(p) whitening for an
explicit subset of observations on the **original** run grid. It implements the
existing `WhiteningPlan` interface, so `GlsModel` applies the same transform to
the selected design and response. The optimized fitting kernels are unchanged.

```sh
cargo run -p fmri-reg --features dataset --example selected_gls
```

This example constructs a two-run design with separate intercepts and a task
column, selects original scans, preserves known AR(2) covariance across gaps,
checks assay identities and runs bounded multi-worker feature-block fits with a
task contrast. Its synthetic responses demonstrate the API, not estimator quality.

## Selection and identity

Construct `ObservationSelection::new(original_run_lengths, retained_indices)`
using strictly increasing, unique, zero-based original indices. Empty selection,
zero-length original runs, out-of-range indices and reordered/duplicate rows are
errors. An entire original run may be excluded. The selection retains the full
original run layout, including excluded runs; coefficient vectors still correspond
to that original layout. Select X and Y with `selection.select(...)` for dense fits.
Excluded matrix values are never inspected; retained values must be finite.

A design-layer `CensorPlan` supplies compatible retained indices:

```text
ObservationSelection::new(original_frame.blocklens.clone(),
                          censor.retained_indices().to_vec())
```

Build/convolve drift, event and feature columns on the full grid first. Do not
recompute them after exclusion or invent a compressed regularly spaced frame.
Keep original acquisition metadata alongside the selection; the AR model uses
integer original-scan lags, assuming a constant acquisition interval within each
run. It does not model arbitrary irregular acquisition times.

For datasets, `fmri_reg::selection::select_assay(assay, original_design_ids,
selection)` first requires exact equality of the complete original observation ID
sequence. It then creates a lazy source view and selected IDs, retaining feature
IDs and annotations. Bind those IDs with `BoundModel::new` and use the existing
`fit_assay_with_options` execution path. Matching IDs checks declared alignment,
not the truth of externally supplied timing metadata. The low-level array APIs
cannot infer identity from equal dimensions.

## Covariance policy is explicit

`SelectedArPlan::new(selection, coefficients_per_original_run, startup, gaps)`
requires both startup and gap policy. Each run has stable finite AR coefficients;
empty coefficients explicitly request IID, never an automatic fallback.

- `GapPolicy::Preserve`: marginalize the missing observations. For stationary
  startup, the selected covariance is the original stationary run covariance
  submatrix. For zero-history startup, it is the submatrix of the finite-run
  zero-prehistory covariance, including propagation through leading exclusions.
- `GapPolicy::Restart`: declare independent segments at the first retained scan
  and after each gap. Apply the chosen startup policy afresh to every segment.
  Adjacent retained scans from distinct runs are always independent.

`ArStart::Stationary` uses a stationary state covariance and `ZeroHistory` starts
with zero unobserved history and unit innovation variance. Both policies retain
short segments even when their length is below p. Missing observations are not
zero-valued measurements. Preserve and Restart are different covariance models;
neither is selected implicitly by a mask.

Only retained rows enter the regression objective and its residual degrees of
freedom. Numerical rank and contrast estimability are handled by the existing
GLS/linear-model preparation. A fully excluded run may leave an all-zero run
intercept, so rank rather than column count determines df. This API does not
estimate AR coefficients from censored data. Noise-estimation masks, AR parameter
uncertainty and censored feasible-GLS bias correction remain separate concerns.
Supplying externally estimated coefficients gives conditional plug-in inference,
not uncertainty accounting for those estimates.

## Computation and limits

Preparation propagates a p-dimensional state covariance over the original scan
grid and precomputes retained-row transitions, innovation scales and gains. Each
feature is then whitened using only its retained values and O(p) temporary state.
No dense N-by-N covariance is stored or inverted by this production path.
For maximum order p, preparation is O(N*p^3), storage O(M*p^2), and application
O(M*p^2*features), where N is original scans and M retained scans; zero order uses
a one-dimensional state. Completely excluded runs and trailing excluded scans
are skipped after coefficient validation. This general selected-row path does
not replace the specialized contiguous AR kernels.

There are two reusable state vectors per transform call (about 16*p bytes of
numeric scratch), independent of feature-block width. As with other custom
transforms, this scratch and resident plan storage are outside the existing
reader/fitting working-buffer estimate; allocator/backend overhead is also
excluded. No new throughput or total-RSS guarantee is claimed.

## Evidence

`scripts/generate_selected_gls.R` generates 24 independent dense reference cases:
AR orders 0–4 plus negative AR(1), two startup policies, and two gap policies.
Stationary references use R `stats::ARMAacf` with unit-innovation scaling;
zero-history references invert the finite lower-triangular AR filter. They then
subset the covariance or construct independent segment blocks and solve dense
GLS. Tests compare:

- W Sigma W' to identity, coefficient unit covariance, coefficients and variance;
- t/F statistics, p-values, and residual df;
- allocating and reusable GLS execution, and dense versus lazy assay fits with
  one and three workers, tail blocks and feature IDs;
- no-gap equivalence to the established contiguous AR kernels, strided arrays,
  leading/interior/trailing gaps, short runs, and a fully excluded run;
- invalid indices, coefficients, nonfinite retained inputs, ID mismatches and
  insufficient working budgets.

Comparison tolerance is 2e-9*(1+abs(reference)); rank/df and IDs are exact. Existing
noise/regression tests, minimal selected-plan tests, strict Clippy and the example
are recorded in `evidence/selected-gls/verification.json`. These are local known-
parameter results, not qualification of censored AR estimation or real-data
inference, and not a whole-workspace/hosted-CI claim.

For stationary AR(1) coefficients estimated from retained OLS residuals, see
[Selected-observation estimation](SELECTED_ESTIMATION.md). That separate API
provides finite-lag correction and diagnostics with narrower simulation limits;
it does not qualify censored AR(p) or zero-history estimation.
