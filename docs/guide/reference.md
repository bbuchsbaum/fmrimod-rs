# Choose a component or a deeper reference

Use the facade for a composed analysis or depend on a component for a smaller
application. Cargo package names use hyphens; Rust imports use underscores.

| Task | Generated API |
| --- | --- |
| Compose sessions and named output requirements | [fmrimod](api/fmrimod/index.html) |
| Bind observations, features, assays and readers | [fmri-data](api/fmri_data/index.html) |
| Evaluate HRFs, convolve events and reconstruct signals | [fmri-hrf](api/fmri_hrf/index.html) |
| Build formula/dataframe designs and nuisance terms | [fmri-design](api/fmri_design/index.html) |
| Estimate or apply AR/ARMA transforms | [fmri-noise](api/fmri_noise/index.html) |
| Reuse numeric OLS/GLS and contrast preparation | [fmri-reg](api/fmri_reg/index.html) |
| Fit group/repeated models, resampling and influence | [fmri-group](api/fmri_group/index.html) |
| Work directly with spline bases | [fmri-splines](api/fmri_splines/index.html) |

Detailed contracts remain in the source repository:

- [Current capabilities](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/docs/CAPABILITIES.md)
- [Sessions and data ingress](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/crates/fmrimod/SESSIONS.md)
- [Named outputs and uncertainty](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/crates/fmri-reg/OUTPUTS.md)
- [Repeated models](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/docs/REPEATED_GROUP.md)
- [Resampling](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/docs/GROUP_RESAMPLING.md)
- [Noise extensions](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/docs/NOISE_EXTENSIONS.md)
- [Scientific evidence](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/evidence/README.md)

The complete companion source is
[`guide_workflow.rs`](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/crates/fmrimod/examples/guide_workflow.rs).
Its excerpts are included during the book build and the whole application runs
in CI, including from an external Git-dependent project.
