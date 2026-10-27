# Group permutation inference

`permutation_max_t` tests one named group contrast over every feature in the supplied
independent-subject cohort. Select the scientific family before calling it. It uses
two-sided absolute t statistics and returns both marginal permutation probabilities
and probabilities corrected by the maximum across that entire family. A block is
only an execution unit; every block receives the same transformation schedule.

Two transformation laws are explicit:

- `SignFlip`: intercept-only zero-mean tests, requiring independent subjects whose
  entire error vector over the tested feature family is jointly sign symmetric
  under the null. Marginal symmetry of each feature alone is insufficient.
- `FreedmanLane`: fit the contrast-null nuisance space, permute its residuals within
  row-bound exchangeability blocks, add the nuisance fit and refit the full model.
  The null space is constructed from the actual contrast, including non-axis
  contrasts. With nuisance covariates this is generally an approximate test;
  exhaustive enumeration does not make the nuisance procedure exact. Exchangeability
  labels alone do not establish valid exchangeability or support repeated measures.

Design columns and contrast coordinates are scaled together before factorization.
The nuisance projector is `UU' - h h'`, where `U` spans the full design and `h`
is the normalized estimable contrast direction. This avoids an overflow-prone
normalization of the original coefficient weights and a second factorization.

The caller records a nonempty scientific justification. These assumptions follow
the [FSL randomise formulation](https://fsl.fmrib.ox.ac.uk/fsl/docs/statistics/randomise.html);
arbitrary heteroscedastic or correlated rows are not admitted by relabeling them.
The separate repeated-model API does not imply repeated-measures permutations.

Exhaustive enumeration includes identity and computes exceedances divided by all
transformations. Seeded Monte Carlo samples uniformly with replacement and uses
`(1 + exceedances)/(1 + draws)`. Ties receive a 64-epsilon relative guard. Undefined
observed t statistics, missing observations or invalid designs reject the whole
family; no feature is silently discarded. A nonzero effect with zero resampled
residual variance contributes an infinite maximum, conservatively.

The implementation retains a block of data, one maximum per transformation, and
family statistics/counts. It regenerates transformations deterministically per
block rather than storing a transformations-by-subject matrix. Its numeric budget
includes these arrays and conservative factorization scratch, but excludes caller
input and metadata; it is not a total process memory guarantee.

`scripts/resampling_reference.R` independently enumerates 64 sign flips and 36
restricted Freedman–Lane permutations using base-R QR. Tests compare statistics,
uncorrected/max-T probabilities, and full maximum distributions, including across
feature block sizes. Enumeration is capped before fitting.
