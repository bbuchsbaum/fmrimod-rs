# Requested first-level outputs

`LinearModel::prepare_outputs` and `GlsModel::prepare_outputs` compile an
`OutputSpec` alongside the existing full-fit API. Request scientific results,
then let the plan allocate only the required working buffers. The default is
estimates only; uncertainty is explicit.

| Request | Intended consumer | Retained numeric values per feature | Work skipped |
|---|---|---|---|
| `None` | Group methods consuming subject effects only | q estimates | Predictions, residuals, RSS |
| `Marginal` | Separate variance-aware or mixed-effects analyses | q estimates + one residual scale | Unrequested exported beta maps and joint geometry |
| `Joint` | Coupled outcomes or correlated contrasts | q estimates + one residual scale | Unrequested exported beta maps |

Here q is the number of requested estimands. A coefficient is a selector row of
the same linear estimand matrix as a contrast. An all-beta request is supported
by selecting every coefficient, provided each is estimable. The existing `Fit`
still exposes minimum-norm coefficients in rank-deficient designs.

## Basic use

```rust
use fmri_reg::{LinearModel, output::{EstimandSet, OutputSpec, SummaryView, Uncertainty}};
use ndarray::array;

fn main() -> Result<(), fmri_reg::FitError> {
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.]];
    let y = array![[1., 4.], [2., 2.], [2., 3.], [4., 1.]];
    let model = LinearModel::new(x.view(), 1e-12)?;
    let spec = OutputSpec::new(EstimandSet::contrasts(
        vec!["task-minus-baseline".into()], array![[0., 1.]],
    )?).with_uncertainty(Uncertainty::Marginal);
    let plan = model.prepare_outputs(&spec)?;
    let mut workspace = plan.workspace(2)?;
    let result = plan.run_into(y.view(), &mut workspace)?;
    if let SummaryView::Marginal(ref summary) = result {
        let effect = summary.estimates()[(0, 0)];
        let se = summary.standard_error(0, 0)?;
        println!("{effect} +/- {se}");
    }
    let saved = result.to_owned(); // owns estimates; shares immutable context
    // `saved` remains valid after the plan, model and workspace are dropped.
    Ok(())
}
```

For selected coefficients use
`EstimandSet::coefficients(ids, column_indices, design_columns)`. Mixed
coefficient/contrast requests are simply rows in `EstimandSet::contrasts`.
No output variant fabricates absent variance. `variance`, `standard_error` and
`covariance` accessors check indices and finite multiplication. Variance is
available through Marginal and Joint; cross-covariance only through Joint.
Zero variance and dependent or zero estimands are valid summaries. A subsequent
t/F test or group estimator can have stricter requirements.

## Computation and uncertainty

For a shared design X and fixed whitening W, D=(WX)^+, e=L D WY. The first
implementation uses coefficient scratch D WY and then L beta. Estimates-only
omits the prediction and residual-scale pass. Marginal and Joint compute the
full-model explicit residuals, then sigma²=RSS/(n-rank). They do not omit
nuisance columns from the fit. The per-feature covariance is sigma² G, with
G=(L D)(L D)ᵀ. This Gram form avoids expanding and cancelling large entries of
the coefficient covariance when a contrast itself has small variance.

Marginal stores only diag(G); Joint stores G once per plan, shared by owned
results. Joint refers to covariance between estimands **at the same feature**,
not spatial covariance or between-subject heterogeneity. These models share
one design, row selection and whitening plan across features. Voxel-specific
geometry requires separate plans; this encoding cannot silently represent it.

GLS summaries explicitly report `conditional_noise() == true`: uncertainty is
conditional on the supplied whitening plan, excluding uncertainty in estimating
AR parameters. A feasible-GLS noise-estimation stage may still require residual
work even when the final request is estimates-only. No IID substitution, f32
conversion, residual norm subtraction, or hidden variance floor is introduced.
Legacy t/F calculations are unchanged; the new stable geometry does not claim
to repair legacy near-collinear covariance calculations.

## Memory and blocks

Inspect a plan through `output_count`, `requires_residual_scale`,
`workspace_bytes(capacity)` and `retained_numeric_bytes(features)`.
For n observations, p regressors and block capacity b, explicit workspace bytes
are:

| Model | Estimates only | Marginal / Joint |
|---|---:|---:|
| OLS | 8b(p+q) | 8b(p+q+n+1) |
| GLS | 8b(p+q+n) | 8b(p+q+2n+1) |

These are checked numeric allocations, not measured process RSS. Model storage,
immutable output specification/geometry, preparation temporaries, allocator
metadata, backend packing and custom-whitening scratch are excluded. An owned
summary retains 8bq bytes, or 8b(q+1) with uncertainty, plus shared context.
Requesting uncertainty reduces retained maps but does not eliminate full-model
beta scratch or residual computation; it can slightly increase workspace over
an ordinary full fit by adding the requested-estimate buffer.

With the `dataset` feature, `output_dataset::run_blocks_with_options` uses the
same private executor as existing full fits. It preflights the aggregate reader
and workspace budget, keeps the reader on the caller thread, and delivers
ordered borrowed summaries. `BoundOutputs::new(plan, observation_ids).run_assay`
checks exact observation identity/order before opening the source and supplies
feature IDs to the sink. Use `summary.to_owned()` only for results that must
survive reuse. A cloned plan shares its workspace identity; independently
prepared plans intentionally reject each other's workspaces.

## Upstream compilation and reuse

`fmrimod::outputs` resolves `NamedLinearEstimand` weights against `DesignColumns`
and unions `ConsumerRequirement`s for a single subject/model. It preserves each
consumer's requested order, checks semantic conflicts under the same estimand
ID, and requests the strongest required uncertainty once. Scientific semantics
include units, coding, normalization, run aggregation and a version; display
labels alone do not establish equivalence. This compiler checks supplied metadata
and weights, not the scientific truth of those declarations. Callers must bind
the column order and model ID to the actual design used for fitting.

`ReuseIdentity` includes data/source versions, original and selected observation
order, feature/spatial identity, design/rank policy, whitening, numerical backend
provenance, and structural output binding. `OutputCapability` checks exact
identity plus sufficient uncertainty: Joint can satisfy Marginal/None, and
Marginal can satisfy None for the **same estimand binding**. It does not derive
new combinations or match arbitrary subsets. Session-only/unversioned custom
whitening never passes this reuse check. Deterministic custom whitening needs
an explicit version contract.

This is a pure compatibility API, not a cache store or automatic attachment of
provenance to an owned result. Immutable version strings remain caller assertions;
paths, Rust types or labels alone are insufficient content fingerprints. Keep
compiled semantics and the result together in the orchestration layer. Group
estimators remain separate from regression.

## Qualification and remaining work

See `evidence/outputs/` for executed checks, source/reference hashes and warm
benchmarks. Tests include independent orthogonal-design and stationary AR(1)
precision solutions, existing R-fixture agreement, near-collinear contrast
variance, rank deficiency, zero covariance, dependent estimands, strided/tail
blocks, plan identity, custom whitening, ordered workers and budget boundaries.

`examples/output_benchmark.rs` compares full fits plus requested estimates with
the new paths. It separately probes K=L D applied to explicitly whitened data;
that arithmetic reassociation remains outside production dispatch until its
accuracy boundary is supported. It never composes W into K.

NIfTI ingestion and resident cohort sessions now live in `fmridataset` and
`fmrimod`; see [the session interface](../fmrimod/SESSIONS.md) and
[25-subject latency evidence](../../evidence/regression/cohort/README.md). These reuse the
requested-output regression path and existing group reducer. Temporal dictionary
projections and durable on-disk caches remain deferred. Future projection work
must preserve original selected observation count, residual df and complement
energy.
