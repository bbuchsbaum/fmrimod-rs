# Developing fmrimod-rs

Use Rust 1.91.1 or newer. One root Cargo lockfile pins the workspace dependency
resolution; there are no independently resolved member lockfiles. Python tools
use Python 3.11 or newer. Rust tests consume checked-in fixtures and do not need R.

```sh
cargo fmt --all --check
cargo test --workspace --locked
cargo test --workspace --all-features --locked
cargo clippy -p fmri-noise -p fmri-reg -p fmri-group --lib --all-features --locked -- -D warnings
python3 scripts/check_repository.py
```

For limited disk space, `python3 scripts/qualify_completion.py --checks` runs
one target at a time and removes only its newly generated executables after
recording hashes. Run `--calibration` separately for the seeded synthetic studies;
these R audits require `jsonlite` and `metafor`. After regenerating a shared R
fixture, run `python3 scripts/sync_reference_fixtures.py`; this keeps each package
test suite self-contained without introducing a cross-package file dependency. See the
[qualification report](docs/COMPLETION_REPORT.md) for assumptions and adverse cases.

Keep algorithms and data contracts in their component crate. The `fmrimod` crate
owns composition and identified sessions; numerical kernels do not depend on it.
Preserve the optional formula/plotting boundaries. Scientific behavior changes
need independent analytic or reference evidence, not only implementation-shaped tests.

Write temporary logs, benchmarks and data under `evidence/`; avoid adding generated
artifacts to source directories. Commit compact reviewed receipts and necessary
fixtures deliberately. See [the evidence policy](evidence/README.md). Do not edit
historical receipts to make old source hashes or paths look current.

The source repository can be shared on GitHub independently of a crates.io
release. All packages currently inherit `publish = false`. A registry release
requires its own version, package-content and dependency-order qualification.

Build and check the published guide/API artifact with:

```sh
python3 scripts/build_docs.py
```

This installs a checksum-pinned mdBook binary into `target/docs-tools`, executes
the guide companion, builds all-feature Rustdoc, and validates the generated
local links/assets under `target/site`. Serve that directory with a local HTTP
server for inspection. The optional `scripts/check_docs_browser.cjs` check uses
an installed Playwright Chromium; follow the repository's browser lifecycle policy.

Documentation deployment follows a successful `main` Rust workflow and checks
out that workflow's exact qualified SHA. Developer-preview tags point to the
qualified code commit. Registry publication remains a separate authorization.
