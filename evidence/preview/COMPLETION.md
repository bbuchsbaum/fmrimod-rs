# Developer preview publication

Published tag: `v0.1.0-alpha.1`, annotated tag target
`fc1895f19cb2fdd0303457be72a93832e19fe26f`.

- [Guide](https://bbuchsbaum.github.io/fmrimod-rs/)
- [API](https://bbuchsbaum.github.io/fmrimod-rs/api/fmrimod/index.html)
- [Prerelease and downloadable assets](https://github.com/bbuchsbaum/fmrimod-rs/releases/tag/v0.1.0-alpha.1)
- [Core CI](https://github.com/bbuchsbaum/fmrimod-rs/actions/runs/40613346174)
- [Documentation build/deployment](https://github.com/bbuchsbaum/fmrimod-rs/actions/runs/40614700297)

Both hosted workflows succeeded at the tag target. Each platform's default and
all-feature test invocations totalled 2,079 passes, zero failures and 36 explicit
ignored cases; these totals include repeated tests across feature configurations.
Formatting, numerical Clippy, repository checks and external Git consumer
applications passed. A separate external consumer resolved the published tag and
ran its analytic checks successfully (`tag-consumer.json`).

The executable guide covers event design, estimated AR noise, conditional GLS,
named estimates, bounded cohort collection and independent-subject group
estimation for 12 synthetic participants and four regions. Its source snippets
are included directly in the guide. Local fresh-Chromium checks covered desktop
and mobile layout, navigation, guide search and API search; receipts and inspected
screenshots are under `../docs/browser/`. The task browser and server were closed.

The deployed Linux documentation artifact contains 1,216 HTML pages. Its 133,809
local links passed validation. Seven live URLs returned HTTP 200, and the deployed
build metadata matched the tag version and exact source revision (`live-site.json`).
The locally built macOS documentation was checked separately (`../docs/build.json`);
platform-dependent API links account for its different link total.

Release assets include that exact deployed documentation artifact, the benchmark
receipt and four compared numeric output streams, `release-evidence.json`, and
`SHA256SUMS`. Every uploaded asset's GitHub-reported SHA-256 matched the local
file before and after publication (`published-release.json`). The benchmark
receipt retains its original precommit Git head; its recorded source hashes were
verified against the tagged source. It has not been rewritten to imply a later
measurement. Reproduction commands and timing boundaries are in the release notes.

All qualification data are synthetic. Scientific limitations, conditional
uncertainty, measured calibration shortfalls and explicit failure cases remain
documented in the guide and release notes. APIs remain unstable. Installation is
through pinned Git dependencies; crates.io publication remains disabled.

The final evidence-only commit follows the qualified tag without changing its
source, version, workflows or benchmark measurements. Compressed CI, tag-consumer
and local documentation logs preserve the underlying command output.
