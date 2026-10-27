# fmrireg

Prepared mass-univariate OLS and AR-GLS in f64. Factor the design once, reuse it
across feature blocks, and compute coefficients, explicit residual variances,
and estimable t/F contrasts. Rank-deficient designs retain the existing SVD rank
policy; requested AR models are never replaced with IID noise.

## Fast fitting

On macOS, enable the `accelerate` Cargo feature to use the system Accelerate
matrix backend. This becomes the default for prepared fits when enabled. Other
platforms and builds without the feature use the portable ndarray backend.
Select `MatrixBackend::Portable` with `with_backend` for comparisons or portable
execution in the same binary. Inputs that cannot be passed directly as contiguous
row-major matrices fall back to the portable kernel without changing the API.
No f32 conversion or squared-norm shortcut is used.

```rust
use fmrireg::LinearModel;
use ndarray::{array, s};

fn main() -> Result<(), fmrireg::FitError> {
    let x = array![[1., 0.], [1., 1.], [1., 2.], [1., 3.]];
    let y = array![[1., 4.], [2., 2.], [2., 3.], [4., 1.]];
    let model = LinearModel::new(x.view(), 1e-12)?;
    let mut workspace = model.workspace(2)?;
    let fit = model.fit_into(y.view(), &mut workspace)?;
    let contrast = fit.t_contrast(array![0., 1.].view())?;
    // Copy compact results only if they must survive the next fit.
    let saved = fit.to_owned();
    let tail = model.fit_into(y.slice(s![.., 0..1]), &mut workspace)?;
    Ok(())
}
```

`GlsModel` supports the same `workspace`, `fit_into`, `with_backend`, and
`backend` methods. Its workspace also holds transformed data. Built-in AR(1)
and AR(p) plans expose `apply_into` for reuse outside the fitter.

The allocating `fit` methods and existing dataset adapter remain available.
`fit_into` reuses all explicit numeric fitting buffers; matrix-backend packing
may allocate scratch. A workspace has a fixed maximum feature count and returns
an error for incompatible dimensions or excess capacity. Smaller tail blocks
reuse its storage. A failed fit may overwrite workspace buffers; only a
successful result provides access to output.

## Requested outputs

Use `model.prepare_outputs(&OutputSpec)` when a downstream analysis needs only
selected coefficients or contrasts. `Uncertainty::None` skips predictions and
RSS; `Marginal` supplies variances/standard errors; `Joint` also supplies
within-feature covariance among requested effects. Owned summaries retain only
requested effects and, when needed, residual scales. Full-fit APIs remain
available. See [OUTPUTS.md](OUTPUTS.md) for examples, memory accounting, upstream
`fmrimod::outputs` compilation, numerical qualification and remaining work.

## Ordered parallel blocks

Enable `dataset` for `dataset::fit_blocks_with_options` and
`BoundModel::fit_assay_with_options`. `BlockOptions` specifies `block_features`,
`workers`, and the aggregate `working_bytes` budget. The one-worker path also
reuses buffers. Multiple workers are persistent for the duration of a call.

Reads and sink calls run on the caller thread; the reader need not be `Send`.
Results arrive in feature order, with at most one block per worker in flight.
The sink can retain `fit.to_owned()` results when needed. On a returned error,
new reads stop and running fits finish before the call returns. Previously
delivered results are not rolled back. Panics in custom code propagate.

The conservative numeric reservation is:

```
active_workers * (workspace_bytes(max_block) + maximum_reader_peak_bytes)
workspace OLS = 8 * block_features * (timepoints + regressors + 1)
workspace GLS = 8 * block_features * (2*timepoints + regressors + 1)
```

The executor checks reader estimates and the aggregate budget before reading
or allocating fitting buffers. Resident source/model data, backend/custom-plan
scratch, allocation overhead, thread stacks, and outputs retained by the sink
are outside this bound. The caller must supply honest reader estimates.

Avoid nested parallelism: when using multiple block workers with Accelerate,
set `VECLIB_MAXIMUM_THREADS=1` before starting the process. The library does not
change process-wide thread settings. Benchmark the actual reader and sink;
serial I/O can limit scaling. `BoundModel` enforces observation-ID order and
passes feature IDs to the sink just as its serial counterpart does.

## Qualification

See [ALGEBRA.md](ALGEBRA.md) for the subsequent GLS/QR experiments and numerical
counterexamples, [HILL_CLIMB.md](HILL_CLIMB.md) for the retained speed improvements,
and [PERFORMANCE.md](PERFORMANCE.md) for the initial optimization results, source fingerprints,
reproduction commands, and limitations. Existing R/dense covariance fixtures and
new analytic, memory, layout, and concurrency regressions cover both backends.
