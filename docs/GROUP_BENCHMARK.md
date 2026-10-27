# Group tranche 1: measured fmrigds comparison

Measured 2026-10-27 on macOS 14.3, arm64, Rust 1.91.1 (release, portable backend),
R 4.5.1 and a source-built fmrigds 0.1.0.9000 at
`273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Thread environment is set to one
for both runners. The sandbox prevents the CPU-model sysctl query; no CPU model
is asserted here. This is a shared workstation, not a controlled benchmark host.

Each entry is the median of five warm runs. Raw trials, setup and first-run
measurements are in `evidence/group-tranche1/{r,rust}-benchmark.json`.

| Subjects | Features | Design columns | Availability | Method | fmrigds ms | Rust ms | R / Rust |
|---:|---:|---:|---|---|---:|---:|---:|
| 20 | 1,000 | 1 | complete | OLS | 18.000 | 0.344 | 52.38× |
| 20 | 1,000 | 1 | complete | FE | 7.000 | 0.476 | 14.71× |
| 20 | 1,000 | 1 | complete | DL | 8.000 | 0.494 | 16.20× |
| 100 | 10,000 | 1 | complete | OLS | 122.000 | 9.406 | 12.97× |
| 100 | 10,000 | 1 | complete | FE | 31.000 | 18.643 | 1.66× |
| 100 | 10,000 | 1 | complete | DL | 33.000 | 18.993 | 1.74× |
| 100 | 10,000 | 1 | patterned | OLS | 156.000 | 9.460 | 16.49× |
| 100 | 10,000 | 1 | patterned | FE | 34.000 | 18.590 | 1.83× |
| 100 | 10,000 | 1 | patterned | DL | 34.000 | 19.218 | 1.77× |
| 100 | 100,000 | 1 | complete | OLS | 1223.000 | 102.523 | 11.93× |
| 100 | 100,000 | 1 | complete | FE | 245.000 | 193.956 | 1.26× |
| 100 | 100,000 | 1 | complete | DL | 261.000 | 189.197 | 1.38× |
| 100 | 10,000 | 5 | complete | OLS | 134.000 | 8.805 | 15.22× |
| 100 | 10,000 | 20 | complete | OLS | 196.000 | 12.312 | 15.92× |

The OLS result is a requested coefficient (intercept for P=1, last covariate for
P=5/20); meta-analysis is scalar. The requested outputs are effect, SE, statistic,
p, n, and Q/tau² where applicable. The R public pipeline additionally computes
its normal full set of coefficient/statistic assays and provenance. These are
application-path comparisons, not equal-count kernel FLOP comparisons. R OLS is
currently an R implementation; the meta reducers use compiled C++.

Timed fits include numeric reads, model preparation, estimation and collection of
requested outputs. Data generation, cohort construction, reference I/O, and BH/BY
are excluded. R plans are built before timing; Rust's small design/contrast wrapper
is constructed in the timed call. Both runners use resident in-memory input.
No disk-read or file-format throughput claim follows from these timings.

Every requested output was compared against fmrigds before timing. Maximum
scaled error `abs(Rust-R)/(1+abs(R))` across all cases was
**6.65e-10**, below the 2e-9 benchmark tolerance.
Per-output errors are retained in Rust records in the order effect, SE, statistic,
p, Q, tau², n. The largest discrepancies are p-value CDF evaluation; estimated
coefficients/SEs agree much more closely. Independent base-R QR and metafor
FE/DL fixtures also pass at the smaller-case 2e-10 scaled tolerance.
Block sizes 256 and 113 agree within 2e-10 for all requested outputs.

The whole R benchmark process peaked at **2192.2 MiB**;
the whole Rust process at **477.2 MiB**. These include
resident inputs, retained results, reference payloads and runtime allocation.
They are not isolated kernel-memory measurements or a universal memory ratio.
The Rust per-fit conservative numeric buffer estimate is separately recorded;
for the 100-subject, 100,000-feature intercept fit at B=256 it is
1,716,640 bytes,
excluding resident sources, metadata, sink-retained outputs and backend scratch.

## Limited null calibration

R seed 2710; 2,000 independent normal-null features; 20 independent observations;
unit known sampling variances; nominal pointwise alpha=0.05. This is one stated
simulation setting, not evidence for arbitrary first-level variance estimation,
heterogeneity, dependence, missingness, or repeated-measures models.

| Method | Rejections / 2,000 | Rate | Monte Carlo SE |
|---|---:|---:|---:|
| OLS | 112 | 0.0560 | 0.0051 |
| FE | 118 | 0.0590 | 0.0053 |
| DL | 95 | 0.0475 | 0.0048 |

For context, approximately 1.96 Monte Carlo SE gives a roughly one-percentage-point
uncertainty radius. This check does not establish exact finite-sample coverage.
See [GROUP_ANALYSIS.md](GROUP_ANALYSIS.md) for commands, contracts and limitations.
