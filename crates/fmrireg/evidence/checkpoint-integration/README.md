# Qualified checkpoint integration

The local merge joins the standalone cohort checkpoint
`9d8fe7168d4250c581f83ed9d3d58cdd51096fc2` to consolidated main
`5e0c8a9b7dd36259f122f8c8d29872e0caeaac9e`.

The consolidation landed during integration preparation and already contained
all checkpoint source files. Of the checkpoint's 429 files, 417 were identical,
11 had reviewed newer versions in main, and only `source-manifest.json` was
missing. The merge preserves every existing main file, adds that manifest, and
records both histories. It makes no numerical, dependency or API changes.

The root `source-manifest.json` is the **historical cohort checkpoint manifest**,
not a manifest of the merged workspace. Its 428 hashes describe the checkpoint
parent. It differs from `evidence/source-manifest.json`, which records the
original sibling repository provenance. The original standalone checkpoint and
its qualification remain available through the merge parent and the bundle in
`../cohort-completion/`.

## Reconciliation

`verification.json` records every difference and both Git blob identities.
Main's group deletion/refit support, optional `q_fixed` diagnostic and updated
callers, group/noise exports, workspace README, ignore protections, design
metadata cleanup and formatting all remain intact. Newer repeated-model,
resampling, influence and noise implementation files remain intact as well.
No existing working file was overwritten.

## Verification and limits

- All 428 checkpoint manifest entries were verified against its Git tree.
- All 380 source hashes in the final integrated qualification receipt match
  both the main parent and the working files.
- Fresh locked offline Cargo metadata resolved eight workspace members; every
  declared target source is tracked. Workspace formatting and seven Python
  controller tests passed; commands, statuses and log hashes are recorded here.
- Rust runtime qualification is inherited from the unchanged source tree:
  `evidence/completion/qualification-summary.json` records 1,024 default and
  1,054 all-feature tests, with source transitions and exclusions. It was not
  rerun for this metadata-only merge. The numerical and synthetic-data limits
  in `docs/COMPLETION_REPORT.md` continue to apply.

No remote publication or hosted CI run is part of this integration.
