# Noise extensions

`fmri-noise` now exposes bounded additions without changing the established AR
kernels. `select_ar_order_conditional` uses mean-centred, Gaussian conditional
likelihood on the same trailing `n - max_order` samples for every candidate.
It estimates one common mean before scoring and omits that shared mean from the
criterion penalty; this is a conditional criterion, not a full likelihood.
Its diagnostics include p=0 explicitly; a failed non-IID candidate is never
silently converted into IID.

`ArmaPlan` is for known stable AR and invertible MA coefficients. It uses the
zero-prehistory innovation recursion independently for every run and explicit
reset. Use `apply_pair(X, Y)` when transforming a model: both matrices must
have the same row count and receive the identical transform.

`estimate_hannan_rissanen` accepts exactly one contiguous finite series. It
performs a high-order AR initialization and SVD least-squares lag regression,
then rejects
rank deficiency, nonstationarity, or noninvertibility. It does not project an
invalid estimate back into the stable/invertible region.

`estimate_parcel_ar` requires a user supplied `global_weight` in `[0, 1]`.
For every parcel and run it pools centred lag products per feature (never an
ACVF of the parcel mean), then combines ACVFs as
`(1-global_weight) * parcel + global_weight * global`; labels, parcel IDs, and
run-specific coefficients remain attached to the plan. The global ACVF is
weighted by feature count, rather than giving equally sized and unequal parcels
the same weight. The resulting Toeplitz system must already be strictly SPD and
the AR result stationary; this API rejects instead of repairing either one. It intentionally has no
single-design `apply`: `apply_matched` returns one matched whitened design per
parcel plus response columns whitened by that parcel's coefficients.

`InnovationScaled` applies caller-known per-run innovation SDs after any
`WhiteningPlan`. `estimate_residual_innovation_scale` returns residual RMS
plug-in values only. Those estimates have the same variance convention as the
whitened residuals but carry no uncertainty claim or degrees-of-freedom
correction.

The accompanying R witness (`scripts/noise_extensions_reference.R`) uses base
R QR and an independent scalar recursion. It regenerates complete inputs and
17-digit expected order scores, HR coefficients/variance, pooled ACVFs and filtered
values. Rust tests compare all of these outputs. Consulted fmriAR source hashes
and its checkout revision are recorded in the fixture. This is independent
numerical evidence, not a claim to execute fmriAR's compiled HR estimator.

These are explicit lower-level capabilities. The existing design-corrected,
selected-row AR estimator does not automatically select its order or switch to
ARMA/parcel fitting. HR accepts a single contiguous series; parcel estimation
currently requires finite complete residuals, with labels bound to original
column positions. Stationary ARMA startup and censored automatic selection remain
outside these APIs. Per-run scale estimation is residual RMS without a leverage
or degrees-of-freedom correction; existing FGLS inference remains conditional.
