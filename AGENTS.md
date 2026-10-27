# fmrimod-rs

One Cargo workspace for idiomatic Rust fMRI modelling capabilities. R sources
in ~/code/fmrihrf, ~/code/fmridesign, ~/code/fmridataset, ~/code/fmrireg and
~/code/fmriAR define scientific reference behavior, not an API translation target.
See docs/REPOSITORY.md for structure, docs/CAPABILITIES.md for scope and
evidence/source-manifest.json for original provenance. Use Mote for work state.
Preserve original sibling repositories and their dirty work. The user authorized
creation and pushes to bbuchsbaum/fmrimod-rs on 2026-10-27. Removal of originals
and crates.io publication remain outside that scope. Retain each crate's license.

Regression scope: core fast OLS/GLS estimation, AR modelling, contrasts,
uncertainty and bounded-memory data processing. Reduced rank and peripheral
workflows are deferred. Do not substitute IID errors for requested AR models.
For numerical changes add independent analytic or freshly built R reference
checks, bind fixtures to source hashes, and record tolerances. Benchmarks must
compare equivalent outputs and report workload, threads and memory limitations.

Use hyphenated Cargo package names and underscore Rust imports. The facade is
`fmrimod`. Keep raw experiments under evidence/ and preserve historical receipts.
Run scripts/check_repository.py after changing manifests, layout or public guides.
