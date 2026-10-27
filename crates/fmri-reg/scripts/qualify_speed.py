#!/usr/bin/env python3
"""Compare all outputs with source-bound R receipts and measure bounded native fits.
Run serially; requires NumPy, macOS time -l, and a built gls_benchmark/block_throughput.
"""
import argparse, hashlib, itertools, json, os, platform, re, statistics, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--target',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--executor-only',action='store_true');args=p.parse_args()
crate=Path(__file__).resolve().parents[1];root=crate.parents[1];out=root/'evidence/regression/speed'/args.label
out.mkdir(exist_ok=False)
env=os.environ.copy();env.update(VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
provenance=json.loads((root/'crates/fmri-noise/tests/fixtures/arp.json').read_text())['provenance']
for rel,digest in provenance['source_md5'].items():assert hashlib.md5((Path(provenance['source'])/rel).read_bytes()).hexdigest()==digest
for rel,digest in provenance['binary_md5'].items():assert hashlib.md5((root/rel).read_bytes()).hexdigest()==digest
import numpy as np
records=[]
def run(name,cmd,backend='accelerate',reference=None):
    e=env.copy();e['FMRIREG_BACKEND']=backend
    stdout=out/(name+'.json');stderr=out/(name+'.log')
    with stdout.open('w') as so,stderr.open('w') as se:
        started=time.time();r=subprocess.run(['/usr/bin/time','-l']+list(map(str,cmd)),env=e,stdout=so,stderr=se)
    assert r.returncode==0,(name,r.returncode,stderr.read_text())
    row=json.loads(stdout.read_text());row.update(name=name,command=list(map(str,cmd)),process_seconds=time.time()-started,peak_rss_bytes=int(re.search(r'(\d+)\s+maximum resident set size',stderr.read_text())[1]))
    binary=out/(name+'.bin')
    if binary.exists():
        row['output_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest()
        if reference is not None:
            a=np.fromfile(binary,dtype='<f8');b=np.fromfile(reference,dtype='<f8');assert a.shape==b.shape
            err=float(np.max(np.abs(a-b)/(1+np.abs(b))))
            assert np.isfinite(a).all() and err<2e-8,(name,err)
            row['comparison']={'reference':str(reference),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'values':a.size,'max_scaled_error':err,'tolerance':2e-8}
    records.append(row);(out/'results.json').write_text(json.dumps(records,indent=2))
    phases=row['warm'];wall=statistics.median([x['wall'] for x in phases]) if isinstance(phases[0],dict) else statistics.median(phases)
    print(name,round(wall*1000,2),'ms',round(row['peak_rss_bytes']/2**20,1),'MiB',flush=True)
    return row
old=root/'evidence/regression/speed/baseline/gls_benchmark';new=args.target/'release/examples/gls_benchmark';blocks=args.target/'release/examples/block_throughput'
for n,p,v,order in ([] if args.executor_only else itertools.product([300,1200],[8,32],[1000,50000],[1,4])):
    key=f'{n}-{p}-{v}-ar{order}'
    reference=root/'evidence/gls-baseline'/f'{key}-b1000-w1-r.bin';assert reference.exists()
    settings=[n,p,v,order,1000,1,5]
    if v==50000 and order==4:
        for mode,exe,backend in [('baseline',old,'portable'),('portable',new,'portable'),('accelerate',new,'accelerate')]:
            name=key+'-'+mode;run(name,[exe,*settings,out/(name+'.bin')],backend,reference)
    name=key+'-reuse';run(name,[new,*settings,out/(name+'.bin'),'reuse'],reference=reference)
# Public production executor: identical serial generated reader and sink in every case.
canonical=None
for b,w in [(1000,1),(1000,2),(1000,4),(128,1),(128,4),(4096,4)]:
    name=f'executor-1200-32-50000-b{b}-w{w}';binary=out/(name+'.bin')
    run(name,[blocks,1200,32,50000,b,w,binary],reference=canonical)
    if canonical is None:canonical=binary
# Exact same outputs with legacy allocating serial adapter.
name='executor-legacy';run(name,[blocks,1200,32,50000,1000,1,out/(name+'.bin'),'legacy'],reference=canonical)
for v in [1000,200000]:
    name=f'executor-memory-{v}';run(name,[blocks,1200,32,v,1000,4,'-'])
files=list((crate/'src').rglob('*.rs'))+list((crate/'tests').rglob('*.rs'))+list((crate/'examples').rglob('*.rs'))+list((root/'crates/fmri-noise/src').rglob('*.rs'))+[crate/'Cargo.toml',root/'Cargo.lock',Path(__file__)]
metadata={'platform':platform.platform(),'machine':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'threads':{k:env[k] for k in ['VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS']},'sources':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'binaries':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [old,new,blocks]},'r_provenance':provenance,'limitations':['unisolated load/frequency','warm medians, five repetitions; setup and first pass recorded separately','GLS phase benchmark uses synthetic generation; public executor uses a cheaper serial synthetic reader','numeric memory budget excludes backend/custom-transform scratch, allocator, stacks and retained outputs']}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
print('PASS',len(records),'processes',flush=True)
