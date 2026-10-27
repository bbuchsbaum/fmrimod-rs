#!/usr/bin/env python3
"""Independent decimal oracle for group reducer qualification (standard library).

Inputs are converted from their exact binary64 values. The 256-digit calculations
use direct reciprocal weights and centered quadratic forms; no Rust scaling,
compensated summation, tiling or factorization code is shared.
"""
import hashlib
import json
import os
import statistics
import subprocess
import sys
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def d(x):
    return Decimal.from_float(float(x))

def main():
    ols, meta = [], []
    with localcontext() as ctx:
        ctx.prec = 256
        cases = {
            'ordinary': [-3., 1., 2., 4., 8.],
            'large_offset': [2.**50 + x for x in [0., .25, .5, 1., 2.]],
            'cancelling_mean': [1e100, -1e100, 2e100, -2e100, 0.],
            'small_units': [x*1e-100 for x in [-3., 1., 2., 4., 8.]],
            'large_units': [x*1e100 for x in [-3., 1., 2., 4., 8.]],
        }
        for name, y in cases.items():
            yy = list(map(d, y))
            n = Decimal(len(yy))
            mean = sum(yy)/n
            se = (sum((x-mean)**2 for x in yy)/(n*(n-1))).sqrt()
            ols.append(dict(name=name, y=y, mean=float(mean), se=float(se), t=float(mean/se)))
        for name, y, v in [
            ('ordinary', [-3., 1., 2., 4., 8.], [.1, 1., 2., 4., 8.]),
            ('zero_tau', [0., .1, -.1, .2, -.2], [1., 2., 1., 2., 1.]),
            ('extreme_weights', [0., 1., 2., 4., 8.], [1e-30, 1., 1e10, 4., 1e30]),
            ('cancelling_effect', [-1e8, 1e8, 2., -2., 0.], [1., 1., 2., 2., 8.]),
        ]:
            for scale in [1., 1e-8, 1e8]:
                ys = [x*scale for x in y]
                vs = [x*scale*scale for x in v]
                yy, vv = list(map(d,ys)), list(map(d,vs))
                w = [1/x for x in vv]
                sw = sum(w)
                mu = sum(a*b for a,b in zip(w,yy))/sw
                q = sum(a*(b-mu)**2 for a,b in zip(w,yy))
                denom = sw-sum(a*a for a in w)/sw
                tau = max(Decimal(0),(q-(len(yy)-1))/denom)
                for method in ['fe','dl']:
                    t = tau if method == 'dl' else Decimal(0)
                    weights = [1/(a+t) for a in vv]
                    effect = sum(a*b for a,b in zip(weights,yy))/sum(weights)
                    se = (1/sum(weights)).sqrt()
                    meta.append(dict(name=name, scale=scale, method=method,y=ys,v=vs,effect=float(effect),se=float(se),t=float(effect/se),q=float(q),tau=float(t)))
    payload = dict(oracle='Python decimal; exact binary64 inputs; precision=256', generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), ols=ols, meta=meta)
    path = ROOT/'crates/fmri-group/tests/performance_reference.json'
    path.write_text(json.dumps(payload,indent=2)+'\n')
    print(path)

def compare():
    """Sequential ABBA comparison of frozen release executables; no builds overlap."""
    out = ROOT/'evidence/group-performance'
    assert (out/'baseline_benchmark').read_bytes() != (out/'candidate_benchmark').read_bytes(), 'refuse identical baseline/candidate executables'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    cases = json.loads((out/'baseline/rust-benchmark.json').read_text())
    comparisons = []
    for case in cases:
        key = f"{case['id']}:{case['method']}"
        env['FMRI_GROUP_BENCH_CASE'] = key
        records = {'baseline': [], 'candidate': []}
        for i, version in enumerate(['baseline', 'candidate', 'candidate', 'baseline']):
            destination = out/'paired'/key.replace(':','-')/f'round-{i+1}-{version}'
            destination.mkdir(parents=True, exist_ok=True)
            env['FMRI_GROUP_BENCH_OUTPUT'] = str(destination)
            with (destination/'run.log').open('w') as log:
                proc = subprocess.run([str(out/f'{version}_benchmark')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            (destination/'exit.json').write_text(json.dumps({'exit_code':proc.returncode})+'\n')
            proc.check_returncode()
            payload = json.loads((destination/'rust-benchmark.json').read_text())
            assert len(payload) == 1
            record = payload[0]
            assert (record['id'],record['method']) == (case['id'],case['method'])
            if record['method']=='ols' and record['columns']==1:
                assert (record['factorizations']==0) == (version=='candidate'), 'wrong executable: intercept factorization witness'
            records[version].append(record)
        row = {key:case[key] for key in ['id','method','n','features','columns']}
        for version, rs in records.items():
            times = sum((r['seconds'] for r in rs), [])
            row[version] = dict(seconds=times, median_seconds=statistics.median(times),
                allocation_calls=rs[0]['allocation_calls'], allocated_bytes=rs[0]['allocated_bytes'],
                max_scaled_error=max(r['max_scaled_error'] for r in rs),
                standalone_read_seconds=statistics.median(r['standalone_read_seconds'] for r in rs),
                standalone_cdf_seconds=statistics.median(r['standalone_cdf_seconds'] for r in rs))
            assert all(r['allocation_calls']==rs[0]['allocation_calls'] for r in rs)
        row['speedup']=row['baseline']['median_seconds']/row['candidate']['median_seconds']
        comparisons.append(row)
        print(f"{key}: {row['speedup']:.2f}x", flush=True)
    (out/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    for version in ['baseline', 'candidate']:
        (out/f'{version}-binary-sha256.txt').write_text(hashlib.sha256((out/f'{version}_benchmark').read_bytes()).hexdigest()+'\n')
        directory=out/('source' if version=='candidate' else 'baseline-source')
        hashes={str(p.relative_to(directory)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.rglob('*')) if p.is_file()}
        (out/f'{version}-source-sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for row in comparisons:
        print(f"{row['id']} {row['method']}: {row['speedup']:.2f}x")

if __name__ == '__main__':
    if sys.argv[1:] == ['--compare']:
        compare()
    elif not sys.argv[1:]:
        main()
    else:
        raise SystemExit('usage: group_performance.py [--compare]')
