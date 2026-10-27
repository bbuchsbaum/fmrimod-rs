# fmrimod cohort execution checkpoint

This standalone source checkpoint contains the complete Cargo workspace needed
for the qualified cohort path: NIfTI admission, explicit OLS/AR models, named
consumer outputs, resident/file/detached execution, independent-subject OLS and
weighted meta-regression, and the local interactive example.

See crates/fmrimod/ANALYSIS.md for API and lifecycle contracts, and
crates/fmrimod/SESSIONS.md for the lower-level session interface.

The input baseline is source-hash-bound to original workspace commit 5606a58
and its retained analysis-plan evidence. Three provider files were restored to
those exact previously qualified hashes in this isolated copy, excluding
concurrent unfinished noise/repeated-model/resampling/influence extensions.
No live shared source was reverted. Original package license declarations and
available license files are preserved. This checkpoint is local, not published.

Reproduce the qualified path with Rust/Cargo and the locked dependency set:

```sh
cargo test --locked -p fmridataset -p fmrireg -p fmrimod --features fmrimod/nifti,fmrimod/accelerate
cargo clippy --locked -p fmrimod --features nifti,accelerate --all-targets --no-deps -- -D warnings
cargo check --locked -p fmrimod --no-default-features
```

`accelerate` is macOS-only; omit it for the portable implementation. Cached
external dependencies were reused for clean-checkout qualification. No claim
of an empty-cache or cross-platform build is made by that evidence.

Run the viewer and synthetic generator as documented in ANALYSIS.md. It needs
NumPy; browser automation additionally needs installed Playwright/Chromium.
The synthetic generator checks disk headroom before each subject. Use a fresh
input directory; it never overwrites an existing fixture directory.
