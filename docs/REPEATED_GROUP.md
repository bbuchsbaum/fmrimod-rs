# Repeated-measures group model

`RepeatedCohort` is separate from `IndependentCohort`. Each row carries a
dataset-qualified participant and measurement identity. A feature supplies one
SPD sampling-covariance matrix for each subject block, explicitly bound to its
ordered observation IDs and feature ID. Subject blocks follow first occurrence
in the row list; within each block rows retain their original order. Duplicate
participant/measurement keys are rejected. No covariance is inferred from
marginal variances. Response units and scientific admission are explicit; upstream
estimated covariance retains `Estimated` provenance while fitting treats it as fixed.

The fitted model is `y_i = X_i beta`, with
`V_i = S_i + eta I + Z_i G Z_i'`. `Z=1` gives a random intercept; `Z=[1,t]`
gives a correlated random intercept/slope. `G` is projected onto the PSD cone
and `eta` is nonnegative. `FixedZero` and `Estimate` are distinct residual
variance choices. ML/REML use normal plug-in covariance for fixed effects;
there are no residual-df, Satterthwaite, or KR claims.

The objective is half of log determinant plus the directly computed whitened
residual sum of squares; REML additionally includes `log|X' V^-1 X|/2`.
The returned `objective` excludes the recorded Gaussian normalizing constant:
log likelihood is minus their sum. This REML convention contains a design-only
constant and must not be compared across different fixed-effect spaces.

Each small covariance block is Cholesky-whitened; a column-scaled SVD solves the
tall fixed-effect problem. Scores for random covariance use whitened products,
avoiding cancellation in `Z' V^-1 Z` under very high heterogeneity. A fixed
per-feature response normalization and centered/scaled slope coordinates improve
conditioning; covariance and coefficients are returned in original units.

Optimization uses deterministic multistart projected-gradient steps over the PSD
cone, Armijo line search, dimensionless projected stationarity and local feasible
probes. Zero variance components and rank-one random covariance are legitimate
outcomes, consistent with [lme4's discussion of singular fits](https://lme4.github.io/lme4/reference/isSingular.html).
This is a local optimizer, with no global-optimum guarantee. Evaluation-limit and
nonconvergence statuses contain diagnostics but no usable estimates. All objective
evaluations, including line search and probes, count against the declared budget.

The constructor rejects row/feature/covariance binding errors, non-SPD known
sampling covariances, constant slope bases, and structurally unidentified
covariance bases. REML also checks covariance directions after projecting out the
fixed effects. In particular, a common two-visit full random intercept/slope model
cannot also estimate an unrestricted extra observation variance. Per-feature nonfinite effects are reported as status, rather
than changing the declared repeated layout. `evaluate_fixed` is supplied for
independent dense-GLS oracle checks.

Unequal declared subject sizes are supported. Feature-specific missing-layout
selection, shared/pooled covariance fitting, and arbitrary random-effect formulas
are not provided. The current cohort holds covariance matrices in memory;
feature reads and fitting workspace are bounded separately. `max_working_bytes`
excludes resident inputs, metadata, allocator overhead and sink-retained outputs.

Run `cargo run -p fmri-group --example repeated_group` for a complete public-API
example. `scripts/repeated_reference.R` supplies independent dense QR/Cholesky
and optimizer witnesses; `repeated_calibration` and `scripts/repeated_calibration.R`
record null coverage, all failures, preparation/fitting time and independent
conditional covariance checks. See [current results](COMPLETION_REPORT.md).
