# fmri-data

Typed observation and feature axes, spatial identities, annotated assays,
row/feature selections, and bounded readers. Rust imports use `fmri_data`.

`FmriFrame` binds identities and assays; `NumericBlock` carries numeric data at
reader boundaries. This foundational crate does not depend on a fitting engine.
Group and first-level adapters perform their own model-specific admission checks.

Default builds support the in-memory/file-source data contracts. Enable `nifti`
for NIfTI ingress. See the [capability matrix](../../docs/CAPABILITIES.md) and
[session/ingress guide](../fmrimod/SESSIONS.md).

```sh
cargo test -p fmri-data --all-features
```

License: GPL-3.0-or-later; the license text is included here.
