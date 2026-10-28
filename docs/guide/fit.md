# Fit subjects with AR noise

A `SubjectSession` holds a loaded subject and its prepared model. Preparation
validates row identity and model metadata, estimates the requested noise model,
and prepares reusable fitting state. Output requests can then reuse that state.

Inside the synthetic subject loop, the example prepares:

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:prepare}}
```

The two declared design columns follow the matrix order exactly. Signal units
and coding describe the model the caller actually built; metadata does not
supply a missing design or silently repair a mismatch.

The noise request uses one run, AR order 1, a design-correction lag limit of 12,
and all four features as its representative noise sample. It assumes one shared
run-level AR plan for these features. The sample data were generated with that
shared structure. For heterogeneous noise, choose a representative set and a
modelling strategy whose assumptions fit your data.

`EstimateAr` uses design-corrected residual noise estimation. It is a feasible
GLS procedure. Inference remains conditional on the resulting whitening plan;
noise-parameter estimation uncertainty is not added to the reported t/F intervals.
`ZeroHistory` specifies the filter's initial condition, matching this simulation.
Known coefficients can instead be supplied through `FixedAr` with an explicit
startup/reset policy.

All scans are retained here. A noise-validity mask controls noise estimation;
it does not itself remove scans from the regression objective. The
[complete selected-row example](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/crates/fmrimod/examples/complete_first_level.rs)
shows explicit censor selection and gap handling.

Unsupported lag pairs, nonstationary estimates, invalid ranks and insufficient
budgets produce errors. Inspect those causes when adapting the example. The
requested AR model does not silently become an IID fit.

Next, [request the outputs the group model needs](outputs.md). See the
[session API](api/fmrimod/session/index.html) for reuse and detachment.
