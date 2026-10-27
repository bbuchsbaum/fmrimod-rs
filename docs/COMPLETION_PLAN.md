# Core capability completion

Mote: `bd-01M5ZDMBSFQ69V0TCT480QV3YT`. User authorized all five outstanding areas
on 2026-10-27. This document records the executable scope and evidence gates;
implementation status is recorded below as checks complete.

## Outcome and constraints

Extend the existing first-level and independent-subject engine to useful repeated
group models, resampling and influence analysis, and the remaining core noise
capabilities. Preserve all existing solver optimizations and scientific contracts.
Keep restricted data out of fixtures. The user selected synthetic data for this tranche; real-data qualification is deferred. No remote publication or retirement of source repos.
Reuse the workspace build cache; disk space is limited.

## Five work packages

1. **Repeated measures.** Separate identified subject-block inputs with known
   positive-definite within-person sampling covariance. Random intercept and one
   random slope; explicit ML/REML, covariance parameter boundaries, convergence
   and conditional plug-in uncertainty. Accept unequal subject block sizes;
   reject nonfinite data and unidentifiable models rather than guessing missing
   layouts. Independent dense likelihood and R references, exact zero-boundary
   cases, ordering and scaling invariants. Bounded feature processing and explicit
   covariance scope; no arbitrary random-effects formula language.
2. **Scientific qualification.** Reproducible simulations covering small cohorts,
   estimated sampling variance, leverage/missingness and repeated dependence;
   extend AR stress checks to same-feature estimation. Report attempted and failed
   fits, Monte Carlo uncertainty and adverse results. Numerical oracle agreement
   is separate from interval calibration. Real-data acceptance requires actual
   approved inputs and is not replaced by synthetic browser qualification.
3. **Permutation and influence.** Explicit independent-subject sign flips and
   exchangeable residual permutations for a prespecified contrast/nuisance model;
   one synchronized schedule and max statistic across the declared feature family.
   Exact small-sample enumeration plus seeded Monte Carlo with the appropriate
   finite-sample counting rule. Exact deletion diagnostics for weighted models with
   explicit fixed versus refitted heterogeneity. Diagnostics never exclude data.
4. **Noise capabilities.** Comparable AIC/BIC AR candidate selection, parcel-bound
   pooled ACVF with explicit shrinkage, known-parameter ARMA whitening and
   Hannan–Rissanen estimation, plus per-run innovation scales. Preserve matched
   design/response transforms and run/reset boundaries; do not turn parcel-specific
   noise into a shared-design transform. Strict numerical errors and diagnostics;
   no hidden IID fallback or automatic stability repair.
5. **Delivery.** Reconcile the capability matrix and usage docs, add portable
   platform CI, run integrated default/all-feature and affected lint checks,
   preserve input/source/check receipts, and make a coherent local Git checkpoint
   of authorized project sources. Hosted CI and remote publication remain separate
   from local qualification. Exclude build products, local tracker authority,
   private state and regenerable large inputs from Git.

## Evidence and acceptance

New numerical APIs need analytic/independent dense or fresh R references, invalid
input and degeneracy tests, identity/order checks, and meaningful limiting cases.
Use source-bound scripts under `scripts/` and receipts under `evidence/completion/`.
Tests and oracle comparisons must pass before measuring equivalent workloads.
Document startup, covariance/variance assumptions, inference distribution and
unsupported scopes in each public API. A nonconverged optimizer must never return
an apparently usable inferential result. Do not change model assumptions to make
a simulation look calibrated.

Run focused package tests during development, then integrated workspace checks.
Record exact commands, exit statuses and source hashes. Linux/Windows CI files
are configuration until those hosted jobs actually execute.

## Status

Implementation and numerical acceptance are complete for all five work
packages. Final all-feature and calibration receipts are recorded in
[the completion report](COMPLETION_REPORT.md). The user explicitly selected
synthetic data; real-data qualification is deferred.

Repeated models include unequal visit counts, fixed and fitted covariance,
zero/singular boundaries, independent R and analytic likelihood witnesses, and
explicit failure statuses. Resampling and influence include exhaustive R and
fresh metafor comparisons. Noise extensions include full-precision independent R
fixtures and strict failure behavior. Existing numerical optimizations are retained.

The default workspace passed 1,024 tests; all-feature checks passed 1,054 plus
58 example builds. Formatting, numerical Clippy, 48,000 attempted synthetic fits,
800 permutation experiments and both independent R audits are complete.
An eager all-feature build exhausted
shared disk; its failed receipt is retained. Qualification now runs targets one
at a time, removing only its newly produced executable artifacts after recording
hashes. Source files, data, original repositories and concurrent work are preserved.

The separately completed cohort/viewer work is checkpointed at `0febf8c`.
This tranche adds the consolidated provider/root workspace checkpoint with final
checks and source receipts. Hosted CI, remote publication and real-data acceptance remain outside
this local completion claim.
