#!/usr/bin/env python3
"""Repeated real-backend latency/RSS qualification; input must be synthetic.

Each mode/repetition owns a fresh Rust process. First request is process-cold,
not OS-cache-cold. ru_maxrss comes from wait4 for that exact child (macOS bytes).
"""
import argparse
import gzip
import hashlib
import json
import os
import pathlib
import platform
import statistics
import struct
import subprocess
import time

import numpy as np


def run(args):
    root, out = args.input, args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / 'manifest.json').read_text())
    assert 'synthetic' in manifest['provenance'].lower()
    expected = {}
    probes = []
    records = []
    sequence = [('first', 0., 'task', True), ('contrast', 0., 'difference', True),
                ('repeat', 0., 'difference', True), ('design', .18, 'difference', True),
                ('estimates-only', .18, 'difference', False)]
    for mode in ['resident', 'detached', 'files']:
        for repeat in range(args.repeats):
            env = dict(os.environ, VECLIB_MAXIMUM_THREADS='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
            proc = subprocess.Popen([str(args.binary), str(root)], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            measurements = []
            start = time.perf_counter()
            try:
                for case, phase, contrast, variance in sequence:
                    cmd = dict(action='run', phase=phase, contrast=contrast, variance=variance,
                               noise='estimated', mode=mode, group='ols', request_id=case,
                               budget_mib=args.resident_budget if mode == 'resident' else args.batch_budget,
                               workers=args.workers)
                    before = time.perf_counter()
                    proc.stdin.write(json.dumps(cmd) + '\n'); proc.stdin.flush()
                    progress = 0
                    while True:
                        line = proc.stdout.readline()
                        if not line:
                            raise RuntimeError(proc.stderr.read())
                        result = json.loads(line)
                        if result.get('type') == 'progress':
                            progress += 1
                            continue
                        if 'error' in result:
                            raise RuntimeError(result['error'])
                        break
                    wall = (time.perf_counter() - before) * 1000
                    values = np.asarray(result['effects'], dtype='<f8')
                    assert np.isfinite(values).all()
                    key = (phase, contrast)
                    if key not in expected:
                        expected[key] = values.copy()
                    error = float(np.max(np.abs(values - expected[key])))
                    assert error < 2e-9, (mode, case, error)
                    if case in ['contrast', 'repeat', 'estimates-only']:
                        assert result['model_preparations'] == 0
                        assert result['source_reads'] == (manifest['subjects'] if mode == 'files' else 0)
                    else:
                        assert result['model_preparations'] == manifest['subjects']
                    measurements.append(dict(case=case, wall_ms=wall, server_ms=result['server_ms'],
                                             first_level_ms=result['through_first_level_ms'],
                                             reads=result['source_reads'], preparations=result['model_preparations'],
                                             reservation_bytes=result['plan']['reserved_numeric_bytes'],
                                             workers=result['plan']['workers'], progress_events=progress,
                                             map_max_abs_error=error,
                                             map_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
                    if mode == 'resident' and repeat == 0 and case != 'repeat':
                        probes.append({k: result[k] for k in ['phase', 'contrast', 'variance', 'first_level',
                                      'probe_indices', 'support', 'effects', 'group_se']})
                proc.stdin.close()
                _, status, usage = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                assert proc.returncode == 0, proc.stderr.read()
                rss = usage.ru_maxrss * (1 if platform.system() == 'Darwin' else 1024)
                record = dict(mode=mode, repeat=repeat, peak_rss_bytes=rss,
                              process_wall_ms=(time.perf_counter() - start) * 1000, cases=measurements)
                records.append(record)
                (out / 'runs.json').write_text(json.dumps(records, indent=2) + '\n')
                print(json.dumps(dict(mode=mode, repeat=repeat, rss_mib=rss/2**20,
                                      first_ms=measurements[0]['wall_ms'], contrast_ms=measurements[1]['wall_ms'])), flush=True)
            finally:
                if proc.returncode is None:
                    proc.terminate(); proc.wait(timeout=30)
                proc.stdout.close(); proc.stderr.close()
    reference = verify(root, manifest, probes)
    (out / 'reference.json').write_text(json.dumps(reference, indent=2) + '\n')
    (out / 'fixture.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = []
    for mode in ['resident', 'detached', 'files']:
        rs = [r for r in records if r['mode'] == mode]
        summary.append(dict(mode=mode, peak_rss_max_bytes=max(r['peak_rss_bytes'] for r in rs),
                            cases={c: dict(median_ms=statistics.median(r['cases'][i]['wall_ms'] for r in rs),
                                           min_ms=min(r['cases'][i]['wall_ms'] for r in rs),
                                           max_ms=max(r['cases'][i]['wall_ms'] for r in rs))
                                   for i, (c, *_) in enumerate(sequence)}))
    (out / 'summary.json').write_text(json.dumps(dict(summary=summary, repeats=args.repeats,
        binary_sha256=hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        timing='Submission to complete JSON response parsed; no browser. Fresh process each repeat, OS cache uncontrolled.',
        threads=dict(requested_workers=args.workers, blas=1), numpy=np.__version__), indent=2) + '\n')


def verify(root, manifest, probes):
    """Independent SVD conditional GLS; bounded frame decoding, four probes."""
    runs = manifest['run_lengths']; n = sum(runs)
    coordinates = np.array(probes[0]['support'])[probes[0]['probe_indices']]
    errors = dict(effect=0., variance=0., group_effect=0., group_se=0.)
    effects = [[] for _ in probes]
    for subject, entry in enumerate(manifest['files']):
        path = root / entry['file']
        with path.open('rb') as f:
            assert hashlib.file_digest(f, 'sha256').hexdigest() == entry['sha256']
        with (gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')) as f:
            header = f.read(352); assert struct.unpack_from('<f', header, 108)[0] == 352
            frame_bytes = int(np.prod(manifest['shape'])) * 4
            y = np.array([np.frombuffer(f.read(frame_bytes), dtype='<f4')[coordinates] for _ in range(n)], dtype=float)
        for k, result in enumerate(probes):
            x = np.zeros((n, 6)); start = 0
            for run, length in enumerate(runs):
                t = np.arange(length)
                x[start:start+length, run] = 1
                x[start:start+length, 2] = np.sin(t*.07+result['phase'])
                x[start:start+length, 3] = np.cos(t*.11)
                x[start:start+length, 4+run] = t/(length-1)-.5
                start += length
            receipt = result['first_level'][subject]
            xw, yw = x.copy(), y.copy(); start = 0
            for length, coeff in zip(runs, receipt['noise_coefficients']):
                end = start+length
                for lag, a in enumerate(coeff, 1):
                    xw[start+lag:end] -= a*x[start:end-lag]
                    yw[start+lag:end] -= a*y[start:end-lag]
                start = end
            b, _, rank, _ = np.linalg.lstsq(xw, yw, rcond=1e-12)
            assert receipt['df'] == n-rank
            c = np.array([0,0,1,-.5 if result['contrast']=='difference' else 0,0,0])
            effect=c@b; sigma=((yw-xw@b)**2).sum(axis=0)/(n-rank)
            g=c@np.linalg.pinv(xw,rcond=1e-12); variance=sigma*(g@g)
            errors['effect']=max(errors['effect'],float(np.max(np.abs(effect-receipt['estimates']))))
            if result['variance']:
                errors['variance']=max(errors['variance'],float(np.max(np.abs(variance-receipt['variances']))))
            effects[k].append(effect)
    for result, e in zip(probes,effects):
        e=np.array(e); idx=result['probe_indices']
        errors['group_effect']=max(errors['group_effect'],float(np.max(np.abs(e.mean(axis=0)-np.array(result['effects'])[idx]))))
        errors['group_se']=max(errors['group_se'],float(np.max(np.abs(e.std(axis=0,ddof=1)/np.sqrt(len(e))-np.array(result['group_se'])[idx]))))
    assert max(errors.values()) < 2e-9, errors
    return dict(max_absolute_errors=errors,tolerance=2e-9,
                scope='Four probes per subject across changed contrasts/design and estimates-only; conditional on exported AR coefficients, not independent noise estimation.')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('input',type=pathlib.Path);p.add_argument('output',type=pathlib.Path)
    p.add_argument('--binary',type=pathlib.Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--resident-budget',type=int,default=16384);p.add_argument('--batch-budget',type=int,default=1024)
    run(p.parse_args())
