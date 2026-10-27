"""Build a report from retained measurements, never hard-coded speedup claims."""
import json,pathlib
root=pathlib.Path('evidence/group-tranche1')
r=json.loads((root/'r-benchmark.json').read_text()); rust=json.loads((root/'rust-benchmark.json').read_text())
rows=[]
for a,b in zip(r,rust,strict=True):
    assert (a['id'],a['method'])==(b['id'],b['method'])
    rows.append(f"| {a['n']} | {a['features']:,} | {a['columns']} | {'patterned' if a['missing'] else 'complete'} | {a['method'].upper()} | {a['median_seconds']*1000:.3f} | {b['median_seconds']*1000:.3f} | {a['median_seconds']/b['median_seconds']:.2f}× |")
rr=json.loads((root/'r-benchmark-resource.json').read_text());sr=json.loads((root/'rust-benchmark-resource.json').read_text());cal=json.loads((root/'calibration.json').read_text())
assert rr['exit']==sr['exit']==0
text='''# Group tranche 1: measured fmrigds comparison

Measured 2026-10-27 on macOS 14.3, arm64, Rust 1.91.1 (release, portable backend),
R 4.5.1 and a source-built fmrigds 0.1.0.9000 at
`273b029bbe9c6c9024bfc12fe4ff6cdb98e18639`. Thread environment is set to one
for both runners. The sandbox prevents the CPU-model sysctl query; no CPU model
is asserted here. This is a shared workstation, not a controlled benchmark host.

Each entry is the median of five warm runs. Raw trials, setup and first-run
measurements are in `evidence/group-tranche1/{r,rust}-benchmark.json`.

| Subjects | Features | Design columns | Availability | Method | fmrigds ms | Rust ms | R / Rust |
|---:|---:|---:|---|---|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

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
**{max(x['max_scaled_error'] for x in rust):.3g}**, below the 2e-9 benchmark tolerance.
Per-output errors are retained in Rust records in the order effect, SE, statistic,
p, Q, tau², n. The largest discrepancies are p-value CDF evaluation; estimated
coefficients/SEs agree much more closely. Independent base-R QR and metafor
FE/DL fixtures also pass at the smaller-case 2e-10 scaled tolerance.
Block sizes 256 and 113 agree within 2e-10 for all requested outputs.

The whole R benchmark process peaked at **{rr['max_rss_bytes']/1024**2:.1f} MiB**;
the whole Rust process at **{sr['max_rss_bytes']/1024**2:.1f} MiB**. These include
resident inputs, retained results, reference payloads and runtime allocation.
They are not isolated kernel-memory measurements or a universal memory ratio.
The Rust per-fit conservative numeric buffer estimate is separately recorded;
for the 100-subject, 100,000-feature intercept fit at B=256 it is
{next(x['numeric_working_bytes'] for x in rust if x['id']=='100-100000-0' and x['method']=='ols'):,} bytes,
excluding resident sources, metadata, sink-retained outputs and backend scratch.

## Limited null calibration

R seed 2710; 2,000 independent normal-null features; 20 independent observations;
unit known sampling variances; nominal pointwise alpha=0.05. This is one stated
simulation setting, not evidence for arbitrary first-level variance estimation,
heterogeneity, dependence, missingness, or repeated-measures models.

| Method | Rejections / 2,000 | Rate | Monte Carlo SE |
|---|---:|---:|---:|
'''+ '\n'.join(f"| {x['method'].upper()} | {x['rejections']} | {x['rate']:.4f} | {x['monte_carlo_se']:.4f} |" for x in cal)+'''

For context, approximately 1.96 Monte Carlo SE gives a roughly one-percentage-point
uncertainty radius. This check does not establish exact finite-sample coverage.
See [GROUP_ANALYSIS.md](GROUP_ANALYSIS.md) for commands, contracts and limitations.
'''
pathlib.Path('docs/GROUP_BENCHMARK.md').write_text(text)
print('Wrote 14-case benchmark report; max error',max(x['max_scaled_error'] for x in rust))
