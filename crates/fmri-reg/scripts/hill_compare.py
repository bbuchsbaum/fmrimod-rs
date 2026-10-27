#!/usr/bin/env python3
"""Serial ABBA kernel comparison; timings exclude generation/preparation."""
import argparse,hashlib,json,os,shutil,statistics,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--label',required=True);p.add_argument('--base',type=Path);p.add_argument('--order',default=4,type=int);args=p.parse_args()
crate=Path(__file__).resolve().parents[1];root=crate.parents[1];out=root/'evidence/regression/hill-climb'/args.label;out.mkdir(exist_ok=False)
base=args.base or root/'evidence/regression/hill-climb/baseline/hill_kernel'
current=Path('/private/tmp/fmrimod-fit-speed-target/release/examples/hill_kernel')
shutil.copy2(current,out/'hill_kernel')
env=os.environ.copy();env.update(VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
results={'base':[],'candidate':[]}
for i,which in enumerate(['base','candidate','candidate','base']):
 exe=base if which=='base' else out/'hill_kernel';cmd=[str(exe),str(args.order)]
 result=subprocess.run(cmd,env=env,capture_output=True,text=True,check=True)
 (out/f'{i}-{which}.json').write_text(result.stdout);results[which].append(json.loads(result.stdout))
 print(args.label,i,which,'complete',flush=True)
summary=[]
for index in range(len(results['base'][0])):
 b=[r[index] for r in results['base']];c=[r[index] for r in results['candidate']]
 for x in c:
  for k in ['beta_sum','variance_sum']:assert abs(x[k]-b[0][k])<1e-10*(1+abs(b[0][k])),(index,k,x[k],b[0][k])
 row={k:b[0][k] for k in ['n','p','block','order','backend']}
 for k in ['whiten','fit','full']:
  bm=statistics.median([v for r in b for v in r[k]]);cm=statistics.median([v for r in c for v in r[k]])
  row[k]={'base_ms':bm*1000,'candidate_ms':cm*1000,'speedup':bm/cm,'base_range_ms':[min(v for r in b for v in r[k])*1000,max(v for r in b for v in r[k])*1000],'candidate_range_ms':[min(v for r in c for v in r[k])*1000,max(v for r in c for v in r[k])*1000]}
 summary.append(row);print((row['n'],row['p'],row['block']),'full speedup',round(row['full']['speedup'],3),flush=True)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
files=[*Path(crate/'src').glob('*.rs'),*Path(root/'crates/fmri-noise/src').glob('*.rs'),crate/'examples/hill_kernel.rs']
(out/'metadata.json').write_text(json.dumps({'base':str(base),'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'candidate_sha256':hashlib.sha256((out/'hill_kernel').read_bytes()).hexdigest(),'order':args.order,'sources':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},'threads':{k:env[k] for k in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},'method':'ABBA fresh processes; seven warmed batches per stage per process; checksum guard only, independent full-output qualification follows'},indent=2))
for f in files:
 dest=out/'source'/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
