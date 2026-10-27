#!/usr/bin/env python3
"""Independent conditional GLS witness; requires NumPy, fixture and HTTP receipts.

Usage: python verify.py INPUT RESPONSES OUTPUT.json
Checks the reported AR whitening; it does not independently estimate AR noise.
"""
import gzip
import hashlib
import json
import pathlib
import sys

import numpy as np

root, receipts, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((root / 'manifest.json').read_text())
cases = [(p, json.loads(p.read_text())) for p in sorted(receipts.glob('*-run.json'))]
first = cases[0][1]['response']
coordinates = np.array(first['support'])[first['probe_indices']]
data = []
for f in manifest['files']:
    path = root / f['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == f['sha256']
    raw = gzip.decompress(path.read_bytes())
    data.append(np.frombuffer(raw, dtype='<f4', offset=352).reshape(240, -1)[:, coordinates].astype(float))

def whiten(a, coefficients):
    result = a.copy()
    for run, phi in enumerate(coefficients):
        start, end = 120 * run, 120 * (run + 1)
        for lag, weight in enumerate(phi, 1):
            result[start + lag:end] -= weight * a[start:end - lag]
    return result

checks = []
for path, receipt in cases:
    r = receipt['response']
    t = np.arange(240) % 120
    run = np.arange(240) // 120
    x = np.column_stack([run == 0, run == 1, np.sin(t * .07 + r['phase']),
                         np.cos(t * .11), (t / 119 - .5) * (run == 0),
                         (t / 119 - .5) * (run == 1)])
    c = np.array([0., 0., 1., -.5 if r['contrast'] == 'difference' else 0., 0., 0.])
    effects, errors, variance_errors = [], [], []
    for s in r['first_level']:
        xw = whiten(x, s['noise_coefficients'])
        yw = whiten(data[s['subject']], s['noise_coefficients'])
        beta, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=1e-12)
        assert rank == 6 and s['df'] == 234
        residual = yw - xw @ beta
        effect = c @ beta
        factor = c @ np.linalg.pinv(xw, rcond=1e-12)
        variance = (residual ** 2).sum(axis=0) / 234 * (factor @ factor)
        effects.append(effect)
        errors.append(float(np.max(np.abs(effect - s['estimates']))))
        if r['variance']:
            variance_errors.append(float(np.max(np.abs(variance - s['variances']))))
        else:
            assert s['variances'] == [None] * 4
    effects = np.array(effects)
    mean = effects.mean(axis=0)
    se = effects.std(axis=0, ddof=1) / np.sqrt(25)
    group_error = float(np.max(np.abs(mean - np.array(r['effects'])[r['probe_indices']])) )
    se_error = float(np.max(np.abs(se - np.array(r['group_se'])[r['probe_indices']])) )
    check = dict(case=path.name, effect_max_abs=max(errors),
                 variance_max_abs=max(variance_errors, default=0),
                 group_effect_max_abs=group_error, group_se_max_abs=se_error)
    assert max(check[k] for k in check if k != 'case') < 2e-9, check
    checks.append(check)

# Same estimand/model across resident repeat, estimates-only and file execution.
responses = [c[1]['response'] for c in cases]
map_errors = {}
for i in [3, 4, 5]:
    error = float(np.max(np.abs(np.array(responses[2]['effects']) - responses[i]['effects'])))
    assert error < 2e-9
    map_errors[cases[i][0].name] = error
noise_change = float(np.max(np.abs(
    np.array([s['noise_coefficients'] for s in responses[1]['first_level']]) -
    np.array([s['noise_coefficients'] for s in responses[2]['first_level']]))))
assert noise_change > 1e-6
report = dict(numpy=np.__version__, tolerance_absolute=2e-9,
              scope='Independent conditional GLS given reported AR coefficients; four spatial probes per subject, all eight runs. Not an independent AR estimator oracle.',
              checks=checks, full_map_equivalence_max_abs=map_errors,
              design_edit_max_noise_coefficient_change=noise_change,
              receipt_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in cases})
output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
