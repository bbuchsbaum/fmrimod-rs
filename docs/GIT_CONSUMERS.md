# Using the published Git source

The repository is [bbuchsbaum/fmrimod-rs](https://github.com/bbuchsbaum/fmrimod-rs).
Packages currently use Git dependencies; registry publication is disabled.
Pin a reviewed full commit SHA for reproducible builds. For example, the first
published checkpoint can be used with:

```toml
[dependencies]
fmrimod = { git = "https://github.com/bbuchsbaum/fmrimod-rs.git", rev = "796f2bbe4b26e718716fd5f2a05a69ddcdf61680" }
ndarray = "0.16"
```

For a direct numerical component:

```toml
[dependencies]
fmri-reg = { git = "https://github.com/bbuchsbaum/fmrimod-rs.git", rev = "796f2bbe4b26e718716fd5f2a05a69ddcdf61680" }
```

Cargo package names use hyphens; imports use underscores. The facade exposes
`fmrimod::regression`, `hrf`, `dataset`, `noise`, and `group`. Enable `formula`
for `design`, `nifti` for ingress/controller support, and `accelerate` for the
optional macOS backend. Default facade dependencies omit Polars and NIfTI.
The [license map](../LICENSE.md) still applies to Git consumers.

## External-project qualification

Run the harness from this checkout, passing a full SHA that is already pushed:

```sh
python3 scripts/verify_git_consumer.py --rev 796f2bbe4b26e718716fd5f2a05a69ddcdf61680
```

It creates separate temporary Cargo projects outside the workspace, resolves the
public Git dependency, verifies the actual source revision, and runs:

- A default facade application with analytic OLS checks and HRF evaluation.
- Direct component use with covariance-whitening and spline checks.
- The identified first-level-to-group pipeline and repeated-model example.
- The formula/nuisance/censoring/estimated-AR first-level example, with its explicit
  `serde_json` application dependency.

Every fMRI package must resolve to the requested Git revision. The facade case
also checks its resolved dependency graph for the absence of Polars and NIfTI.
The component case enables all three facade features. These are consumer and
integration checks; scientific qualification remains in the numerical test suite
and [calibration report](COMPLETION_REPORT.md).

The harness records commands, exit codes, output hashes, generated manifests,
lockfiles and harness source hashes under `evidence/git-consumer`. Temporary
projects are removed automatically. CI runs the same harness against its own
pushed commit on all three platforms.
