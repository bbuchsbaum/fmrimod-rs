# Cohort execution completion

User-authorized items 1–5: detached-fit execution, larger synthetic qualification,
interactive request handling, group-derived input contracts, and a reproducible
local source checkpoint. No remote publication or original-repository removal.

## Delivered behavior

`DetachedSession::run_blocks/run_owned` answers outputs directly from retained
coefficients/scales. `StoragePolicy::RetainedFits` loads bounded subject batches,
prepares and retains fits, then releases time series. Contrast edits do not read
data; changed designs reload and rebuild affected fits. Failed replacement keeps
the old valid cache. Auto storage may select retained-fit batches when explicit
retention cannot fit with resident time series. Retention is never silently
removed, and noise/rank/estimand policies are preserved.

`run_with_progress` provides fallible checkpoints before work and between load,
preparation and output batches. Cancellation stops further dispatch and joins
active jobs; completed state is resumable. External sinks are not rolled back.
Cancellation does not interrupt an in-progress BLAS operation or gzip decode.

`GroupAnalysisSpec` derives first-level inputs from the actual group provider:
OLS needs effects, while weighted meta-regression requires genuine marginal
sampling variances even for effects-only group output. The existing scalar
portable-bundle contract is exposed by the same adapter. The viewer now executes
the existing Paule–Mandel/modified Knapp–Hartung provider as well as group OLS.
No new numerical estimator was introduced in this tranche.

The localhost controller owns one active backend request and one latest queued
request, with a bounded terminal history. It implements request IDs, progress,
cooperative cancellation, error recovery and stale-result suppression. Serialized
submission acknowledgements plus browser generations protect latest-request
ordering. Terminal completion wakes long-polling clients immediately; the initial
100 ms polling regression is retained in `polling-delay-baseline.log`.
This remains a local integration example, not a publicly authenticated service.

## Verification

- 108 affected Rust tests passed in an isolated snapshot and a fresh Git checkout.
  New coverage includes no-source detached execution, exact budget rejection,
  automatic storage fallback, changed-model reload, failed replacement recovery,
  and cancellation/resume with one and two active workers.
- An analytic straight-line example checks detached effects and marginal variance
  at absolute tolerance 1e-12. Strict scoped Clippy and portable builds passed.
- Seven Python controller tests cover bounded queuing/history, cancellation,
  backend errors, request identity, recovery and immediate completion notification.
- Actual installed Playwright Chromium exercised nine main browser cases plus
  rapid supersession, a deliberately delayed acknowledgement, cancellation and
  recovery. An additional protocol request checked effects-only weighted group
  output with required first-level variance retained. No page errors; slice
  endpoints work. Browser/server/backend processes were closed afterward.
- `verify_browser.py` independently reconstructs conditional GLS with NumPy SVD,
  then solves scalar PM by bisection and computes modified-KH uncertainty.
  Twelve successful browser/protocol results passed at four probes per subject.
  Maximum first-level effect discrepancy: 5.76e-12; variance: 7.43e-16;
  group heterogeneity: 1.67e-14, against a 2e-9 absolute tolerance.
  This conditions on exported AR coefficients; it is not an independent
  re-estimation of the noise model or an inferential calibration study.

`browser/labels-reviewed.png` is a separate visual check using a previously
qualified result after shortening/correcting control labels. The other screenshots
and browser receipt bind the executed integration run; no display-latency timing
is inferred from this later label check.

## Larger workload measurements

Apple M3 Max, macOS 14.3, release Rust/Accelerate; requested four cohort workers,
one BLAS/OMP thread per worker. Each mode has three fresh-process repetitions of
first request, contrast edit, repeat, design edit and estimates-only output.
Times include the real Rust backend, group OLS, JSON serialization/transport and
Python parsing. They exclude the browser. `wait4` measures peak RSS for each
individual Rust child. OS page-cache state is uncontrolled; first-process and
warm-request measurements do not establish cold physical-disk performance.

The whole-brain-size fixture has 25 subjects, 120,935 masked voxels on an
80×80×64 grid, two 120-frame runs, six columns and estimated per-run AR(2) using
64 sampled features. It occupies 2,116,371,506 compressed bytes. The grid and
signals are synthetic; no anatomical registration or real-data acceptance claim.

| Mode | First result median | Contrast edit median | Design edit median | Maximum peak RSS |
|---|---:|---:|---:|---:|
| Resident + retained fit | 11.239 s | 0.238 s | 2.640 s | 6.47 GiB |
| Detached retained fit | 9.017 s | 0.228 s | 11.207 s | 1.88 GiB |
| File batches | 7.491 s | 6.155 s | 6.217 s | 2.34 GiB |

These compare complete execution policies with equivalent outputs. Resident
execution had a 16 GiB numerical budget; detached/files had 1 GiB. Actual worker
counts were four for resident/files, three for initial detached fitting, two for
detached design replacement, and four for cached detached contrasts. The planner
accounts for old and replacement caches conservatively. This is not an isolated
same-worker kernel comparison. Requested estimates-only results still use the
explicitly retained fit where selected.

Numerical reservations are not RSS caps. The detached initial reservation was
912 MiB and its cached contrast reservation 216 MiB, while measured process peak
was 1.88 GiB. Source/request metadata, allocator capacity and backend scratch are
outside the reservation. Actual peak RSS is reported rather than inferred from
the budget.

`wholebrain/` contains all runs, min/median/max summaries, input hashes and the
independent conditional-GLS witness. Complete effect maps match across all modes,
repeats and equivalent output requests at absolute tolerance 2e-9. Each workload
also checks first-level effects/variance and group mean/SE independently at four
spatial probes per subject.

The bounded raw/gzip comparison uses the same two subjects and identical voxel
values at this grid/length. All resulting map hashes match bit-for-bit. Median
file-batch contrast time was 1.048 s for gzip and 0.448 s for raw NIfTI. This
isolates a useful local codec/storage difference without claiming a 25-subject
all-raw or cold-storage benchmark. `codec-gzip/` and `codec-raw/` retain details.

The longer-acquisition fixture has 25 subjects, 58,075 masked voxels on a
64×64×48 grid and two 240-frame runs (480 total), occupying 2,056,380,960
compressed bytes. It uses the same six-column/per-run AR(2) specification:

| Mode | First result median | Contrast edit median | Design edit median | Maximum peak RSS |
|---|---:|---:|---:|---:|
| Resident + retained fit | 10.208 s | 0.113 s | 2.710 s | 5.75 GiB |
| Detached retained fit | 8.094 s | 0.113 s | 7.442 s | 1.29 GiB |
| File batches | 6.508 s | 5.942 s | 5.999 s | 1.61 GiB |

`longer-acquisition/` records all repetitions and actual worker counts. Across
the two larger workloads and two codec variants there were 36 fresh child
processes and 180 successful measured requests. Equivalent complete maps matched
exactly; the largest independent conditional-GLS effect error was 6.18e-12.
These are bounded local synthetic qualifications, not real-data acceptance or
measurements of network storage, an emptied OS cache or physical display latency.

## Source checkpoint and reproduction

`source.bundle` is a complete local Git source checkpoint, including the Cargo
workspace, lockfile, required package resources, fixtures and the new code.
`source-checkpoint.json` identifies its revision and clean-checkout check.
`source-manifest.json` binds the checkpoint files. Clone it with:

```sh
git clone source.bundle cohort-source
cd cohort-source
cargo test --locked -p fmridataset -p fmrireg -p fmrimod \
  --features fmrimod/nifti,fmrimod/accelerate
```

The three concurrently edited provider files were restored **only in the isolated
copy** to exact hashes from `../analysis-plan/source-hashes.json`:
`fmri-noise/src/lib.rs`, `fmri-group/src/lib.rs`, and
`fmri-group/src/meta_regression.rs`. Other baseline provider hashes matched.
The shared working tree was not reverted or broadly staged. Concurrent new noise,
repeated-model, resampling and influence work is outside this checkpoint.

The clean-checkout run used `--locked --offline` and the existing dependency
cache/target, not an empty dependency cache. All 359 initially manifested files
were checked after cloning. This establishes self-contained source inputs for
the qualified path on this machine; it is not hosted CI or cross-platform proof.

Reproduce fixture and timing runs using `scripts/analysis_viewer.py generate`
under `fmrimod` and `scripts/cohort_qualification.py` under `fmrireg`.
NumPy 2.4.3, seeds, generator/input hashes, actual worker counts, binary hash,
timing distributions and tolerances are retained in the workload receipts.
Generated data is removed after verification; source, recipes and receipts remain.
Shared build artifacts are preserved for the other active workspace task.
