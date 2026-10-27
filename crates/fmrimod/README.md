# fmrimod

The user-facing entry point for the fmrimod workspace. It connects experimental
events, typed data, first-level fits, named outputs and group analysis.

Components are reexported as `hrf`, `dataset`, `noise`, `regression`, and `group`.
The `design` reexport requires `formula`. The component crates remain usable
independently for smaller applications.

| Feature | Adds |
| --- | --- |
| `formula` | The Polars-based design/formula frontend |
| `nifti` | NIfTI ingress and the interactive analysis controller |
| `accelerate` | Optional macOS system backend in the regression engine |

No optional feature is enabled by default. Start with the
[workspace examples](../../README.md), then read [sessions](SESSIONS.md),
[interactive analysis](ANALYSIS.md), and [group analysis](../../docs/GROUP_ANALYSIS.md).

```sh
cargo run -p fmrimod --example first_level
cargo run -p fmrimod --features formula --example complete_first_level
```

License: GPL-3.0-or-later; the license text is included here.
