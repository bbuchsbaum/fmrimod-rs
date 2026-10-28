# Estimate a population effect

Each row in this cohort is a distinct simulated participant with one visit.
The population model is an intercept: the average task effect across independent
participants. Paule–Mandel estimates between-participant heterogeneity, and the
modified Knapp–Hartung policy supplies the requested group uncertainty.

```rust,ignore
{{#include ../../crates/fmrimod/examples/guide_workflow.rs:group}}
```

For the fixed synthetic seed, the portable run reports approximately:

```text
region-0: effect=0.7122, CI=[0.3982,1.0263], p=0.00041
region-1: effect=0.7551, CI=[0.4014,1.1089], p=0.00065
region-2: effect=0.9481, CI=[0.6058,1.2904], p=0.00008
region-3: effect=1.1253, CI=[0.8946,1.3561], p=0.00000
PASS guide: 12 participants, 4 group estimates; conditional first-level uncertainty
```

Rounded `0.00000` is display rounding, not an exact zero probability. The intervals
and p-values treat the supplied first-level sampling variances as fixed. The
approximate origin is preserved in the cohort; it does not automatically integrate
noise/variance estimation uncertainty into the group model.

These p-values are unadjusted featurewise results. Define a test family explicitly
before applying BH/BY or permutation max-T. Multiplicity adjustment does not
correct a misspecified model or poorly calibrated input probabilities.

Repeated visits belong in a repeated-measures model with participant-bound
within-person sampling covariance. The separate
[`RepeatedCohort`](api/fmri_group/struct.RepeatedCohort.html) API represents that
contract. Arbitrary correlated measurements are outside the scalar independent
cohort profile used here.

Next, [interpret the results and diagnose failures](interpret.md). For covariates,
contrasts and heterogeneity policies, consult the
[group API](api/fmri_group/index.html).
