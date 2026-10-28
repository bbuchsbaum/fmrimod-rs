# Public source and hosted qualification

Source repository: [bbuchsbaum/fmrimod-rs](https://github.com/bbuchsbaum/fmrimod-rs).

CI-qualified code revision: `a9f601f415748a5dcf8c66d93ea4fea9135a67a2`.
[Hosted run](https://github.com/bbuchsbaum/fmrimod-rs/actions/runs/40607394554) passed on all three platforms.

| Platform | Rust tests passed | Failed | Ignored | External consumer commands |
| --- | --- | --- | --- | --- |
| linux | 2079 | 0 | 36 | 7 |
| macos | 2079 | 0 | 36 | 7 |
| windows | 2079 | 0 | 36 | 7 |

Test counts combine the default and all-feature suites. Every job also passed
formatting, package/guide checks and numerical-library Clippy. Consumer commands
resolve the public Git revision and execute analytic OLS/HRF, whitening/spline,
group, repeated-model, and formula first-level applications in separate projects.

Local Git consumers also passed on Rust 1.91.1 and 1.99 at the preceding code
revision `f09cac2`; the final code revision adds Windows path normalization to the
Python guide checker. API documentation built for all workspace features.

Hosted checks exposed and the published code repairs:

- A benchmark using Unix-only RSS APIs on Windows. Unsupported RSS is now `null`.
- Rust 1.99 Clippy requiring fixed-size byte chunks; typed arrays preserve the decoder.
- Two parallel tests clearing the same global cache. Those tests share a lock;
  exact counters and suite parallelism remain enabled. Ten repeated checks passed.
- Windows backslashes being compared against Git forward-slash paths in the guide checker.

The initial failed runs and the cancelled superseded run remain in GitHub Actions.
[The publication receipt](../evidence/github/publication.json) binds the platform
results to log hashes; accepted logs are stored as lossless gzip archives. [Git consumer instructions](GIT_CONSUMERS.md) describe reproduction.

A subsequent documentation/evidence-only receipt commit may skip CI; the exact
tested code revision is given above. Scientific calibration and its limitations
remain documented in [the completion report](COMPLETION_REPORT.md). Hosted
portability checks use synthetic data and do not establish real-data acceptance.
Registry publication and a tagged developer preview are separate release steps.
