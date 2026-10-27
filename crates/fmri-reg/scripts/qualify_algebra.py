#!/usr/bin/env python3
"""Independent Decimal GLS oracle for algebra-reassociation stress cases.

The oracle solves the small full-rank GLS problem at 90 decimal digits. Its
normal equations are an independent high-precision reference, never a proposed
production solver. Input floats are converted exactly; the AR(1) transform is
constructed independently, including resets and stationary startup.
"""
import hashlib
import itertools
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np


def solve(a, b):
    a = [row[:] + [v] for row, v in zip(a, b)]
    for k in range(len(b)):
        pivot = max(range(k, len(b)), key=lambda i: abs(a[i][k]))
        a[k], a[pivot] = a[pivot], a[k]
        scale = a[k][k]
        a[k] = [v / scale for v in a[k]]
        for i in range(len(b)):
            if i != k:
                scale = a[i][k]
                a[i] = [x - scale * y for x, y in zip(a[i], a[k])]
    return [row[-1] for row in a]


def oracle(x, y, phi, resets, stationary):
    with localcontext() as ctx:
        ctx.prec = 90
        xd = [[Decimal.from_float(float(v)) for v in row] for row in x]
        yd = [Decimal.from_float(float(v)) for v in y]
        ph = Decimal.from_float(phi)
        first = (1 - ph * ph).sqrt() if stationary else Decimal(1)
        wx, wy = [], []
        for i in range(len(yd)):
            wx.append([v * first for v in xd[i]] if i in resets else
                      [v - ph * prev for v, prev in zip(xd[i], xd[i - 1])])
            wy.append(yd[i] * first if i in resets else yd[i] - ph * yd[i - 1])
        p = len(xd[0])
        gram = [[sum(row[i] * row[j] for row in wx) for j in range(p)] for i in range(p)]
        rhs = [sum(row[i] * v for row, v in zip(wx, wy)) for i in range(p)]
        beta = solve(gram, rhs)
        variance = sum((v - sum(a * b for a, b in zip(row, beta))) ** 2
                       for row, v in zip(wx, wy)) / (len(y) - p)
        return np.array([float(v) for v in beta]), float(variance)


def main():
    rows = []
    n = 32
    resets = {0, 1, 8, 16, 19}
    for exponent, offset_exp, phi, stationary in itertools.product(
            [10, 20, 30, 35, 38, 40], [0, 20, 32], [0.4, -0.9, 0.999999], [False, True]):
        t = np.array([(-1.) ** i for i in range(n)])
        z = np.array([(-1.) ** (i // 2) for i in range(n)])
        noise = np.array([(-1.) ** (i // 4) for i in range(n)])
        x = np.column_stack([np.ones(n), t, t + 2. ** -exponent * z])
        y = 2. ** offset_exp - 3 * t + .5 * z + .125 * noise
        w = np.eye(n)
        for i in range(n):
            if i in resets:
                w[i, i] = np.sqrt(1 - phi * phi) if stationary else 1.
            else:
                w[i, i - 1] = -phi
        xw, yw = w @ x, w @ y
        u, s, vt = np.linalg.svd(xw, full_matrices=False)
        retained = s > s[0] * 1e-12
        row = dict(delta_exponent=exponent, offset_exponent=offset_exp, phi=phi,
                   stationary=stationary, rank=int(retained.sum()), singular_values=s.tolist())
        if not retained.all():
            row['qualification'] = 'rank-truncated; excluded from full-rank oracle comparison'
            rows.append(row)
            continue
        d = (vt.T / s) @ u.T
        old_beta = d @ yw
        new_beta = (d @ w) @ y
        reference_beta, reference_var = oracle(x, y, phi, resets, stationary)
        row['reference_variance'] = reference_var
        row['reference_beta'] = reference_beta.tolist()
        for name, beta, residual in [
                ('baseline', old_beta, yw - xw @ old_beta),
                ('composed', new_beta, w @ (y - x @ new_beta)),
                ('composed_coefficients_only', new_beta, yw - xw @ new_beta),
                ('raw_residual_only', old_beta, w @ (y - x @ old_beta))]:
            variance = float(residual @ residual / (n - 3))
            row[name] = dict(variance=variance,
                             relative_variance_error=abs(variance / reference_var - 1),
                             max_scaled_beta_error=float(np.max(abs(beta - reference_beta) / (1 + abs(reference_beta)))))
        rows.append(row)
    qualified = [r for r in rows if r['rank'] == 3]
    regressions = [r for r in qualified if r['composed']['relative_variance_error'] >
                   max(2e-8, 4 * r['baseline']['relative_variance_error'])]
    out = Path(__file__).resolve().parents[3] / 'evidence/regression/algebra/reassociation-stress.json'
    native = {}
    keys = ['delta_exponent', 'offset_exponent', 'phi', 'stationary']
    references = {tuple(row[k] for k in keys): row for row in rows}
    for backend in ['accelerate', 'portable']:
        path = out.parent / f'composition-{backend}.json'
        if not path.exists():
            continue
        records = json.loads(path.read_text())
        valid = []
        for row in records:
            ref = references[tuple(row[k] for k in keys)]
            if row['rank'] != 3 or ref['rank'] != 3:
                continue
            row['reference_variance'] = ref['reference_variance']
            for mode in ['baseline', 'composed']:
                row[mode + '_relative_variance_error'] = abs(row[mode + '_variance'] / ref['reference_variance'] - 1)
                row[mode + '_max_scaled_beta_error'] = float(np.max(abs(np.asarray(row[mode + '_beta']) - ref['reference_beta']) / (1 + abs(np.asarray(ref['reference_beta'])))))
            valid.append(row)
        clean_regressions = [r for r in valid if r['baseline_relative_variance_error'] < 2e-8 < r['composed_relative_variance_error']]
        native[backend] = dict(source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), cases=valid,
                               baseline_pass_composed_fail=len(clean_regressions),
                               decisive_case=max(clean_regressions, key=lambda r: r['composed_relative_variance_error']) if clean_regressions else None)
    report = dict(decimal_precision=90, numpy=np.__version__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  assumptions='full rank oracle only; exact float inputs; independently constructed AR(1), five segments including length one',
                  cases=rows, full_rank_cases=len(qualified),
                  material_composed_regressions=len(regressions),
                  worst_composed_case=max(qualified, key=lambda r: r['composed']['relative_variance_error']), native=native)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{len(rows)} cases; {len(qualified)} full-rank; {len(regressions)} composed regressions over 4x baseline and 2e-8')
    print(json.dumps(report['worst_composed_case'], indent=2))
    for backend, result in native.items():
        print(backend, 'baseline passes and composed fails:', result['baseline_pass_composed_fail'])
        print(json.dumps(result['decisive_case'], indent=2))


if __name__ == '__main__':
    main()
