> Imported historical document. Current workspace scope and evidence are in
> [the workspace README](../../README.md) and [plan](../../docs/PLAN.md).
> Current R SPMG3 uses a dispersion basis; old parity statements below are historical.

# Core R-to-Rust port audit — 2026-09-05

The core event-to-design path is now checked against current R source and
independent mathematical cases. API names or historical feature percentages were
not sufficient: the starting code failed basic block integration, dropped HRF
bases, substituted custom HRFs, and had unsafe cache identities.

This is a contract-based core audit, not certification of every R export or of
statistical inference built on these matrices. Both Rust worktrees already had
extensive changes. Work was applied over those changes; nothing was committed or
published. The R source trees were read and copied for validation, not edited.

## Final gates

- fmridesign: **496 passed, 0 failed, 20 ignored** with all features enabled.
- fmrihrf: **283 passed, 0 failed, 1 ignored** with all features enabled.
- fmrihrf library: all-features Clippy passed with `-D warnings`.
- fmridesign library: Clippy completed with **45 remaining legacy warnings**;
  repository-wide lint modernization is not complete.
- Both documented quickstarts executed successfully; both `git diff --check`
  checks passed. Existing test/example warnings and ignored doctests remain.

The totals include unit, integration, and executed documentation tests. The R
fixtures contain multiple numerical cases per contract test. No hosted CI or
publication was performed.

## Source and fixture provenance

| Repository | HEAD at audit start |
| --- | --- |
| R fmridesign | `6907d0b43cb0390c331406ff6e8d6608b541d69d` |
| R fmrihrf | `b7b3a74057d71a3d135084a15ea9adab8bd31ba3` |
| Rust fmridesign | `544b160128a6c30d617dfc5ffb841ea2d1da9caf` |
| Rust fmrihrf | `e01ad338b087f98ffd86488a377d71e500e567e9` |

The R HRF checkout also contained uncommitted feature-regressor work. HEAD alone
therefore does not identify its complete source. The JSON fixtures record source
file MD5 hashes, R version, package version, and source HEAD. R 4.5.1 was used with
a freshly compiled fmrihrf 0.4.0 installed from an isolated source copy. R
fmridesign 0.6.0 was loaded from source using `pkgload`, without recompilation.

- `tests/fixtures/r_core_contract.json`: three SPMG event designs; signed
  modulation and factor interactions; six runwise baseline designs; eight
  B-spline/natural-spline fits and extrapolation cases.
- `../fmrihrf-rs/tests/fixtures/convolution_contract.json`: 64 convolution cases
  crossing SPMG1/3, summation/duration-average behavior, 0.33/0.1 precision, and
  compiled/reference-loop methods; five fixed normalization modes.
- HRF CSV fixtures were regenerated with explicit precision. Previously,
  `regressor_design` fixtures used R's default precision despite specifying a
  different sampling-frame precision. The `/tmp`-only SPM reference is now a
  checked-in fixture. Tent tolerance tightened from 1.5 to 1e-12 with no skipped
  boundary rows.

## Core contracts and evidence

| Capability | Current disposition and evidence |
| --- | --- |
| SPMG1/2/3 | Raw R scaling and analytic first/second temporal derivatives; correct basis counts and registry aliases; R fixtures |
| Other common HRFs | Gamma, Gaussian, Mexican hat, six-parameter half-cosine, lag/block decorators, tent basis: R CSV checks; causal and partition-invariant evaluation checks |
| Normalization | Typed `None`, `Spm`, `UnitPeak`, `UnitIntegral`, `UnitPeakPerBasis`; fixed reference factors; R mode fixtures and singleton-vs-batch invariance |
| Impulses and blocks | Fractional hat projection; positive-duration integrals even below precision; exact quadrature endpoint; signed amplitudes; additive overlaps; direct/FFT agreement and analytic ramp integrals |
| Run boundaries | Zero-based borrowed event tables, run-local onsets, zero rows for empty runs, all requested predictor columns retained, tails isolated; multi-TR/start-time R fixtures |
| Regressor sets | Share the actual supplied HRF with `Arc`; retain custom parameters/bases; analytic custom-HRF regression |
| Formula event designs | Categorical, continuous, matrix/basis amplitude protocol and interactions feed one convolution path; frame precision and full HRF basis retained; complete R matrix checks |
| Continuous amplitudes | Raw values by default; explicit centering/scaling; signed-modulator R oracle |
| Polynomial drift | QR includes the constant during orthogonalization; output omits it when requested; R degree-1/3 runwise matrices |
| B-spline / natural-spline drift | Correct R degree/df semantics, quantile knots, fixed fitted knots, boundary polynomial/linear extrapolation, natural second-derivative constraints; R matrices and eight fitted/prediction cases (sixteen matrices) |
| Contrasts | Explicit weights and bounded linear formulas are evaluated; unknown/ambiguous names and unsupported syntax error; exact independent weight vectors. Existing pair/oneway/F tests pass separately; broad R formula parity remains unproven |
| Caching | Exact float hashes; immutable HRF identity opt-in; arbitrary custom HRFs bypass kernel caching; unsafe global model caches bypassed; cross-amplitude/timing/HRF regressions |
| Portability | Existing nalgebra handles PSD validation; mandatory OpenBLAS/Fortran dependency removed; numerical and invalid-matrix checks |
| Optional plotting | Validate finite values, dimensions, supported formats, and nondegenerate time grids before rendering; single-point/NaN/empty comparison regressions |

`port-evidence/convolution-before.log` preserves the original five failing
analytic regressions after changing only the linear-algebra backend enough to
build. The fixes are tested against analytic integrals rather than against the
same convolution implementation.

## Deliberate differences and upstream discrepancies

1. Current R fmridesign writes multi-basis values in condition-major order but
   names them in basis-major order (`R/event_vector.R`, shared-HRF scatter and
   subsequent `make_column_names`). Rust uses basis-major values and names.
   Fixtures retain R's original matrix and an explicit column permutation; the
   generator independently checks every reordered condition against a separate
   amplitude model. It does not silently relabel the raw R oracle.
2. R event_model recodes noncontiguous block labels, which can collapse an absent
   middle run. Rust preserves the supplied sampling frame and explicit run IDs.
   The ordinary R fixtures populate every run; an independent Rust test covers
   empty middle runs and absent condition levels.
3. Rust borrowed tables use zero-based runs. Formula data use one-based numeric
   `run`, or `block` for multi-run frames. In a single-run frame `block` remains
   an experimental factor. Column names omit R's redundant repeated term prefix.
4. Explicit Rust spans are respected. R Reg may prioritize span attached to the
   HRF; fixtures record the effective R span.
5. The legacy `spm_canonical(a1,a2,b1,b2,c,normalize)` signature still ignores its
   first five parameters. Use `SpmG1Hrf`, `SpmG2Hrf`, or `SpmG3Hrf` for P1/P2/A1.
   Raw amplitudes are now the default. Legacy named derivative aliases remain,
   but SPMG3's third column is the second temporal derivative.
6. `EventVariable::new` now preserves amplitudes. B-spline and natural-spline
   intercepts default to false; `df` counts returned columns after constraints.
   Existing tests requesting partition of unity now request the intercept
   explicitly. Invalid polynomial degrees are rejected instead of producing
   deficient columns. These changes can alter existing analyses and are
   intentional corrections, documented in CHANGELOG.md.

## Performance

`../fmrihrf-rs/examples/port_benchmark.rs` measures 600 scans, 240 signed events,
three SPMG bases, impulses and mixed block durations, and precisions 0.1/0.02 s.
The release profile uses optimization level 3, LTO, and one codegen unit.
[convolution-benchmark.json](port-evidence/convolution-benchmark.json) records
medians and configuration on this macOS arm64 host with rustc 1.91.1.

At 0.02-second precision, the measured mixed-block case took approximately
13.5 ms with automatic selection versus 190 ms with forced direct convolution.
Sparse impulses select direct convolution. Direct and FFT outputs agreed to
within 2.4e-14 across these cases. The default estimates convolution work from
nonzero projected samples and kernel length; methods can still be forced.

These are local measurements under concurrent development load, not a portable
speed guarantee, an R speed comparison, or a speedup relative to the incorrect
starting implementation. The reference loop has different finite-precision
semantics and is not an equal-output performance baseline. Peak memory and
end-to-end formula/dataframe throughput were not benchmarked.

## Verification and reproducibility

See `port-evidence/verification.json` for final commands, exit codes, test totals,
fixture hashes, and limitations. Fixtures are consumed by ordinary Rust tests;
R is needed only to regenerate them.

To regenerate, install a copied current R fmrihrf source into a temporary R
library, set `R_LIBS` to that library and `LC_ALL=en_US.UTF-8`, then run:

```sh
# From fmrihrf-rs:
Rscript --vanilla scripts/generate_convolution_contract.R
Rscript --vanilla scripts/generate_r_fixtures.R
# From fmridesign-rs:
Rscript --vanilla scripts/generate_r_core_contract.R
```

The generators default to `~/code/fmrihrf` and `~/code/fmridesign`; the JSON
contract generators also accept `FMRIHRF_R_SOURCE` and `FMRIDESIGN_R_SOURCE`.
Do not confuse a preinstalled older R package with the freshly compiled oracle.

## Remaining work outside this certification

- Parameter-preserving HRF/regressor serialization: the legacy serde format
  stores HRF names, which cannot reconstruct arbitrary custom HRFs or parameters.
- Full R expression evaluation, factor-cell metadata, general factorial
  interaction contrasts, nuisance formulas, and automatic multi-basis contrast
  expansion need their own end-to-end R oracle. The four-cell interaction helper
  remains a restricted ordered 2x2 operation; unsupported cell counts now error.
- Advanced feature-regressor models, design optimization, empirical interpolation
  derivatives, and broad reconstruction/penalty-family equivalence have existing
  code or upstream APIs but were not certified by this core pass.
- Sampled-design convolution is a separate scalar-kernel API; it rejects
  multi-basis HRFs rather than silently dropping columns. Use event convolution
  for multiple bases. Irregular or inconsistent sampled grids are rejected; they need an explicit
  resampling/integration contract.
- Cross-platform hosted CI, package publication, warning-clean/clippy-wide
  modernization, and end-to-end memory/throughput budgets remain unverified.
  Existing test/example warnings and ignored doctests are reported, not treated
  as clean release evidence.
