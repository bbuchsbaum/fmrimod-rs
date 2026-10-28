# Interpret results and recover from errors

A successful fit says that the declared model was numerically usable. Scientific
acceptance also depends on the data, experimental design and inference assumptions.
The guide's single synthetic realization is a usage example, not a coverage study.

The retained studies show specific limits. Plug-in AR intervals under-cover with
persistent noise and short runs. Some repeated-model normal intervals also
under-cover, and nonconvergence is an explicit result. Small-cohort group
intervals are sensitive to estimated sampling variance; modified Knapp–Hartung
helps in checked cases without providing universal nominal coverage.

Read the
[calibration report](https://github.com/bbuchsbaum/fmrimod-rs/blob/main/docs/COMPLETION_REPORT.md)
for attempted fits, failed fits, conditional coverage, Monte Carlo intervals
and the exact oracle cases. Synthetic qualification is the current data boundary.
No real-data acceptance is claimed for this preview.

| Symptom | Inspect |
| --- | --- |
| Observation/feature binding error | IDs, row order, region/voxel order and actual spatial alignment |
| Rank or contrast error | Design scaling, dependencies among columns, and estimability |
| Noise estimation error | Available lag pairs, censor/run boundaries, sample size and stationarity |
| Budget error | Resident inputs, active workers, block size, collector storage and backend overhead |
| Group missingness or nonconvergence | Per-feature status and retained subjects; preserve failures in summaries |
| Unexpected effect scale | HRF normalization, event amplitude, signal units and run aggregation |

The cohort example's callback reports a group-admission error before mapping it
to the session callback's error type. That callback currently requires
`SessionError`; applications should retain the original diagnostic when bridging
providers with different error types.

For a reproducible analysis, retain the resolved Git SHA and lockfile, design
and contrast definitions, units and identities, row selection, noise policy and
estimated parameters, requested uncertainty, numeric budgets and every failure
status. Performance comparisons also need equivalent requested outputs, thread
settings, startup/warm timing and memory scope.

You can now adapt the workflow with explicit assumptions. Use the
[reference map](reference.md) for the next specific task.
