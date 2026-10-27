# Group estimate bundle, version 1

This is the explicit interchange profile for one scalar first-level estimand per independent subject, not a general `FmriFrame` export. Axis annotations, arbitrary assays, metadata, and source descriptors are intentionally excluded.

A new bundle directory has fixed names: `effects.f64`, optional `variances.f64`, and `manifest.json` written last. Missing manifest means incomplete output. The manifest is `format: "fmri-group-estimate"`, `version: 1`, shape `[subjects, features]`, fixed payload checksums, subject identity, estimand, exact Index or Volume space/order/support, fit receipts, variance origin/df/approximation, and admission. Unknown versions and fields are rejected.

Payloads are IEEE-754 f64, little-endian, feature-major: each feature holds all subjects contiguously. Their logical shape remains `[subjects, features]`. Values preserve bit patterns: zero is real; unavailable/nonfinite cells are unchanged. The profile does not invent variances or reinterpret fitting validity.

Writers take contiguous `[subjects, block_features]` blocks and reject gaps, reordering, shape errors, omitted uncertainty, and blocks over budget. They flush and sync payloads, then write/sync a temporary manifest and publish it with a no-clobber hard link only after all features. Readers use an opened regular-file descriptor, cap manifest input, and verify exact payload length and SHA-256 using one reusable hash chunk no larger than 64 KiB, then use validated constructors. Input files must remain immutable while file-backed cohort sources are used.
