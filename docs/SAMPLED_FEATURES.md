# Continuous sampled features

`fmri_hrf::regressor::SampledFeatures` prepares multiple numeric feature columns
and evaluates them on an acquisition frame. Run the complete example with:

```sh
cargo run -p fmri-hrf --example sampled_features
```

Each sample describes a zero-order-hold bin: its contribution is amplitude times
an integral of the HRF over the bin. It is not a unit-mass impulse. Resampling a
constant signal therefore preserves its integrated response rather than scaling
it with the number of samples.

`FeatureRun` borrows a samples-by-features ndarray view (including strided views),
a timing grid and an optional row mask. Uniform grids specify start and interval;
explicit times must increase strictly, and the final bin repeats the last gap.
Uniform grids permit one sample; explicit times require at least two.
All values, including masked values, must be finite.

`SampledFeatures::new` owns the prepared amplitudes and timing. Centering and
optional sample-SD scaling happen independently per run and column, using only
active rows. Off-mask amplitudes become zero. SD uses n−1 and requires at least
two active samples with nonzero variance. On irregular grids these statistics
are sample-weighted, not duration-weighted. `statistics()` exposes the means,
divisors and active counts. Default preprocessing centers without scaling.

`evaluate` reuses the existing run-aware event convolution implementation, with
Auto, Direct, FFT or exact-lag Loop selection. No HRF tail crosses a run boundary.
Output rows follow acquisition order; columns are basis-major, then feature-major.
The number of feature runs must match the acquisition frame. Precision must not
exceed the shortest bin; smaller precision improves the numerical approximation.
The acquisition frame defaults to a first sample at TR/2; set its start explicitly
when zero or another offset is required.

No companion intercept or presence column is added automatically. For masked
features, preconvolution centering obeys H(x − mean·mask) = H(x) − mean·H(mask).
Subtracting a constant from the convolved output is generally a different model.
`as_events()` borrows the prepared duration-bearing event table without copying.
The returned numeric matrix can be composed with other design columns while
preserving this column order and the acquisition frame.

## Reference evidence and limits

The fixture generator `scripts/generate_sampled_features.R` uses the locally
source-built R fmrihrf package and records its source identity. Sixteen cases
cross uniform/irregular timing, centering, SD scaling and masks, with two runs,
two features and SPMG2's two basis columns. The fixture explicitly sets R's span
to 24 seconds. Rust uses the HRF's own span; it does not inherit R's hidden
40-second feature-regressor default. Rust additionally rejects integration
precision exceeding the shortest bin and supports multiple columns together.

Tests compare prepared amplitudes at relative-plus-absolute tolerance 2e−12 and
convolved results at 2e−9. Independent tests check the analytic ramp response,
resampling-invariant mass, masked-centering linearity, run isolation, strided
input, invalid grids and scaling at 1e−200 and 1e200. See
`evidence/sampled-features/verification.json` for executed checks and source hashes.
This tranche makes no new speedup claim and does not change AR/OLS kernels.

## Named designs and contrasts

`fmri_design::sampled_features::sampled_feature_design` accepts prepared features,
a slice of feature names, the acquisition frame, HRF and convolution method. It
moves the convolved matrix into a `DesignMatrix`, attaching acquisition times
and names without another copy of the numeric matrix. It is also in the prelude.

Names must be nonblank, unique and match the feature count. A single basis keeps
the supplied names. Multiple bases append `.b1`, `.b2`, etc., so two features
`sound` and `motion` with SPMG2 produce:

```text
sound.b1, motion.b1, sound.b2, motion.b2
```

Use these exact names for basis-specific contrasts. The adapter does not sanitize
names or add intercepts. When combining with event or nuisance designs, use
`hstack_design_matrices` and ensure distinct column names across all components.
That function checks identical acquisition times. Row times use the frame's
cumulative run offsets; convolution still uses run-local times and isolates tails.
Matching numeric times alone does not prove matching subject/run identity.

```sh
cargo run -p fmri-design --example sampled_feature_design
```

The example composes sampled features and an intercept, then selects the
canonical-basis sound-minus-motion contrast. Tests check all 16 existing R
fixtures through the named adapter, independent cumulative acquisition times,
and selected contrast values after composition. Additional cases reject ambiguous
names and invalid public frame fields and check broadcast frame normalization.
Evidence is recorded in `evidence/sampled-feature-design/verification.json`.
