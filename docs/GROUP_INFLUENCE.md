# Exact group influence diagnostics

`meta_influence` takes the same identified independent-subject cohort, row-bound
design, named outputs and FE/DL/PM options as weighted regression. For each feature
it returns the full fit and an exact refit omitting each available participant.
No diagnostic automatically excludes a participant or changes the primary model.

`DeletionHeterogeneity::Refit` estimates heterogeneity again after every deletion.
`FixedAtFullFit` holds that feature's full-fit tau² constant while refitting the
coefficients. If mKH inference was requested, its Q/df adjustment still uses the
deleted sample. These answer different sensitivity questions and are recorded
in the receipt. Neither mode pretends to propagate uncertainty in supplied
sampling variances or in the estimated heterogeneity.

The requested fixed covariance is fitted directly. A zero-heterogeneity fit can
be numerically unusable even when the positive fixed-tau fit is well conditioned;
it must not gate that deletion. `MetaDiagnostics::q_fixed` is therefore optional
and absent for an externally fixed positive tau², rather than a fabricated value.

Changes are deleted minus full contrast effects. When standard errors were
requested, the API also returns changes divided by the full-fit standard error.
This is a descriptive sensitivity measure, not a new p-value or a Cook-distance
claim. Rank loss, insufficient observations and nonconvergence remain statuses.
Deletion of an already missing observation has no refit. If the full fit fails,
the feature has no interpretable deletion comparison.

The implementation reuses the qualified weighted solver and refits exactly.
Its cost is O(subjects) fits per feature; analytic deletion updates are a future
optimization requiring equivalence checks. Results arrive in feature blocks, with
a numeric preflight budget including deletion outputs. Sink-retained results,
resident source data and metadata are outside that budget.

`scripts/influence_reference.R` generates fresh metafor FE/DL/PM references for
full fits and both deletion modes. Tests also cover loss of design rank, missing
subjects, insufficient retained rows, and sink failure.
