# Retain named estimates

The group model consumes one scalar task effect and its marginal sampling
variance per subject and feature. Retain that information directly instead of
collecting every coefficient, residual and fitted time series.

First bind the named contrast to the matrix columns:

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:request}}
```

The weights `[0, 1]` refer to `[intercept, task]`; adding or reordering columns
requires updating that binding. For larger applications,
[`compile_requirements`](api/fmrimod/outputs/fn.compile_requirements.html) resolves
named requirements against declared design columns.

`Marginal` retains the variance needed by weighted meta-regression. An effects-only
request would be sufficient for some group OLS work, but cannot supply genuine
meta-regression weights. `Joint` additionally retains cross-estimand covariance
at each feature when a downstream consumer needs it.

The collector declares the participants, estimand and common feature space before execution:

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:collect}}
```

Run the prepared subjects with bounded active work:

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:execute}}
```

Results arrive in subject order. The collector verifies subject IDs, feature
space/order, requested estimand and degrees of freedom before copying the selected
scalar outputs. Scientific admission still comes from the caller: it must be
the same quantity, units, coding and run aggregation across subjects.

This tutorial keeps all 12 subject arrays resident. The cohort has at most two
active workers and two features per block. Its 16 MiB numeric budget excludes
allocator/backend overhead and additional sink retention; the collector has its
own separate 1 MiB budget. These are explicit numeric budgets, not a measured
bound on total process RSS.

The session receipt marks estimated-noise uncertainty as approximate when the
collector records variance provenance. Keep that provenance with the estimates.

Next, [fit the population model](group.md). The
[output API](api/fmri_reg/output/index.html) describes borrowed versus owned results.
