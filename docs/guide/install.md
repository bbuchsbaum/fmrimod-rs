# Install the developer preview

Use Rust 1.91.1 or newer. The preview is distributed through GitHub; crates.io
publication is disabled. For a separate application, pin the preview tag:

```toml
[dependencies]
fmrimod = { git = "https://github.com/bbuchsbaum/fmrimod-rs.git", tag = "v0.1.0-alpha.1" }
ndarray = "0.16"
```

Keep the resulting Cargo lockfile. For a research workflow, retain the resolved
commit SHA with the analysis: Cargo records it in the Git source entry. Use a
full `rev` instead of `tag` when you already have an approved immutable revision.

To run the complete companion example:

```sh
git clone https://github.com/bbuchsbaum/fmrimod-rs.git
cd fmrimod-rs
git checkout v0.1.0-alpha.1
cargo run -p fmrimod --example guide_workflow --locked
```

The default facade includes HRF, typed data, noise, regression and group APIs.
Optional features add specific work:

| Feature | Use it for |
| --- | --- |
| `formula` | The Polars-based formula/design frontend |
| `nifti` | NIfTI ingress and the analysis controller |
| `accelerate` | The optional macOS system matrix backend |

The companion uses borrowed event arrays and the portable backend, so it needs
none of these optional features. Its only data are generated synthetic values.
A larger formula/censoring example is available with:

```sh
cargo run -p fmrimod --features formula --example complete_first_level --locked
```

The [license map](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/LICENSE.md)
applies to Git consumers too. The facade, data and group crates declare
GPL-3.0-or-later; other components retain their individual declarations.

Next, [construct the event design](design.md).
