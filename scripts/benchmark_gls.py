#!/usr/bin/env python3
"""Run isolated processes serially, retain phase timings/RSS, compare ALL outputs."""
import argparse,hashlib,itertools,json,os,platform,re,subprocess,time
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];os.chdir(root)
a=argparse.ArgumentParser();a.add_argument('--label',default='baseline');a.add_argument('--smoke',action='store_true');a.add_argument('--rust-only',action='store_true');a.add_argument('--memory-only',action='store_true');args=a.parse_args()
out=root/'evidence'/('gls-'+args.label);out.mkdir(exist_ok=True)
env=os.environ.copy();env.update(LC_ALL='en_US.UTF-8',R_LIBS=str(root/'evidence/r-library'),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
cases=[(n,p,v,o,1000,1) for n,p,v,o in itertools.product([300,1200],[8,32],[1000,50000],[1,4])]
if args.smoke:cases=[(300,8,1000,4,1000,1)]
else:cases+= [(1200,32,50000,4,b,w) for b,w in itertools.product([128,1000,4096],[1,2,4]) if (b,w)!=(1000,1)]
if args.memory_only:cases=[(1200,32,v,4,1000,1) for v in [1000,50000,200000]]
records=[]
for n,p,v,o,b,w in cases:
 key=f'{n}-{p}-{v}-ar{o}-b{b}-w{w}'
 engines=['rust'] if args.rust_only or args.memory_only or w!=1 or b!=1000 else ['rust','r']
 for engine in engines:
  prefix=out/(key+'-'+engine);binary=prefix.with_suffix('.bin');stdout=prefix.with_suffix('.json');stderr=prefix.with_suffix('.log')
  cmd=([str(root/'target/release/examples/gls_benchmark')] if engine=='rust' else ['Rscript','--vanilla','scripts/benchmark_gls.R'])+[str(x) for x in (n,p,v,o,b,w,3)]+[str(binary)]
  with stdout.open('w') as so,stderr.open('w') as se:
   started=time.time();result=subprocess.run(['/usr/bin/time','-l']+cmd,env=env,stdout=so,stderr=se)
  if result.returncode:raise RuntimeError(f'{key} {engine}: {stderr.read_text()}')
  record=json.loads(stdout.read_text());record.update(engine=engine,key=key,exit=result.returncode,process_seconds=time.time()-started,command=cmd,peak_rss_bytes=int(re.search(r'(\d+)\s+maximum resident set size',stderr.read_text())[1]))
  record['output_sha256']=hashlib.sha256(binary.read_bytes()).hexdigest();records.append(record)
  print(engine,key,'median',np.median([x['wall'] for x in record['warm']]),'MiB',round(record['peak_rss_bytes']/2**20,1),flush=True)
 if 'r' in engines:
  x=np.fromfile(out/(key+'-rust.bin'),dtype='<f8').reshape(v,p+5);y=np.fromfile(out/(key+'-r.bin'),dtype='<f8').reshape(v,p+5)
  # O(1)-scale synthetic design with full rank. Same fixed 2e-8 bound as GLS contract.
  delta=np.abs(x-y);scaled=delta/(1+np.abs(y));comparison={'max_abs':float(delta.max()),'max_scaled':float(scaled.max()),'values':int(x.size),'tolerance':2e-8}
  comparison['passed']=bool(np.isfinite(x).all() and np.isfinite(y).all() and np.max(scaled)<2e-8)
  records[-1]['comparison']=comparison
  if not comparison['passed']:
   (out/'results.json').write_text(json.dumps(records,indent=2));raise AssertionError(comparison)
 elif (root/'evidence/gls-baseline'/(key+'-r.bin')).exists():
  x=np.fromfile(out/(key+'-rust.bin'),dtype='<f8');y=np.fromfile(root/'evidence/gls-baseline'/(key+'-r.bin'),dtype='<f8')
  err=float(np.max(np.abs(x-y)/(1+np.abs(y))));records[-1]['comparison']={'max_scaled':err,'values':int(x.size),'tolerance':2e-8,'passed':bool(np.isfinite(x).all() and err<2e-8)};assert records[-1]['comparison']['passed']
 elif not args.smoke and (out/'1200-32-50000-ar4-b1000-w1-rust.bin').exists():
  x=np.fromfile(out/(key+'-rust.bin'),dtype='<f8');y=np.fromfile(out/'1200-32-50000-ar4-b1000-w1-rust.bin',dtype='<f8')
  if x.shape==y.shape:
   err=float(np.max(np.abs(x-y)/(1+np.abs(y))));records[-1]['block_worker_max_scaled']=err;assert err<2e-8
 (out/'results.json').write_text(json.dumps(records,indent=2))
files=[p for c in ['fmri-noise','fmri-reg'] for p in Path('crates',c).rglob('*.rs')]+[Path('scripts/benchmark_gls.R'),Path('scripts/benchmark_gls.py'),Path('Cargo.lock')]
metadata={'platform':platform.platform(),'machine':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'threads':{k:env[k] for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},'hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'binary_sha256':hashlib.sha256(Path('target/release/examples/gls_benchmark').read_bytes()).hexdigest(),'r_source_commit':subprocess.check_output(['git','-C',str(Path.home()/'code/fmriAR'),'rev-parse','HEAD'],text=True).strip()}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2))
