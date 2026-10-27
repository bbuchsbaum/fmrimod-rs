# Numerical counterexamples retained during qualification

These are development observations, not additional claims about fmrigds or
metafor. The defective paths were in the new Rust implementation before repair.

For rows X=[(1,0),(1,0),(0,1),(0,1)], y=[1,3,2,8] and
v=[epsilon,1,epsilon,1], exact arithmetic gives
Q(t)=40/(1+epsilon+2t), -Q'(t)=80/(1+epsilon+2t)^2,
tr(P(0))=4/(1+epsilon), and tau²_DL=tau²_PM=9.5-epsilon/2.

At epsilon=1e-30, the initial explicit-residual implementation produced
Q=40.19721522630525. A QR-tail-only alternative did not solve row-order sensitivity:
an independent SciPy/LAPACK review found Q spanning 37.32298 to 40.93038 over the
24 permutations. Projecting explicit residuals repaired Q, but using raw QR
complement energy for the DL denominator still produced tau²=9.3843292119 in one
ordering. Ordinary coefficient agreement was therefore an inadequate witness.

The final path retains the original weighted design, applies two residual-based
coefficient refinements, projects explicit residuals into residual coordinates,
and refines complement basis vectors when needed. Tests now require the analytic
Q and tau for every row permutation at epsilon=1e-30. Additional analytic tests
use unequal pair variances, exchanged columns, column factors 1e-8 and -1e8,
and several tau values to verify Q, its derivative and the DL denominator.

At epsilon=1e-310, nalgebra complement calculations still produced grossly wrong
tau values in some orders despite seemingly plausible norm checks. The supported
contract now rejects subnormal normalized weights with `NumericalFailure` and
no estimates. The regression tests require this rejection across permutations;
they do not weaken numerical tolerances or treat these cases as successful fits.

Other guards reject unstable complement-energy refinement and disagreement
between reconstructed residual norms and Q. An uncertain derivative falls back
to bisection. These finite witnesses and guards do not certify arbitrary
ill-conditioned problems. No high-precision or alternative-backend fallback is
silently substituted.

The test that separates near-zero and genuinely large contrast statistics also
checks a confidence level of 1-f64::EPSILON. Lower-tail critical values avoid
rounding (1+level)/2 to exactly one. No arbitrary effect threshold replaces a
statistical test.
