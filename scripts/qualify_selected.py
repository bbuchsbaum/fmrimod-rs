"""Run release workloads serially; preserve every process receipt and RSS."""
from pathlib import Path
import subprocess,json,platform,re,hashlib,os
from types import SimpleNamespace
def measured(command,stdout,stderr,env):
 proc=subprocess.Popen(command,cwd=root,env=env,stdout=stdout,stderr=stderr)
 _,status,usage=os.wait4(proc.pid,0)
 proc.returncode=os.waitstatus_to_exitcode(status)
 return SimpleNamespace(returncode=proc.returncode,rss=usage.ru_maxrss)
root=Path(__file__).resolve().parents[1];out=root/'evidence/consolidation';out.mkdir(exist_ok=True)
binary=root/'target/release/examples/selected_qualification'
env=__import__('os').environ.copy()
for key in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']:env[key]='1'
cases=[(p,n,v,b,w) for p in range(1,5) for n,v in [(300,1000),(1200,1000)] for b,w in [(64,1),(256,1),(256,4)]]
cases += [(p,300,10000,256,1) for p in range(1,5)]
records=[]
for index,args in enumerate(cases):
 label='bench-'+'-'.join(map(str,args));log=out/(label+'.log');rss=out/(label+'.time')
 with log.open('w') as stdout,rss.open('w') as stderr:
  result=measured([str(binary),*map(str,args)],stdout,stderr,env)
 row={'label':label,'arguments':args,'exit':result.returncode,'log':str(log.relative_to(root)),'rss_log':str(rss.relative_to(root))}
 if result.returncode==0:
  row.update(json.loads(log.read_text().splitlines()[-1]));row['peak_process_rss_bytes']=result.rss
 records.append(row);(out/'benchmarks.json').write_text(json.dumps(records,indent=2)+'\n');print(label,'exit',result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)
with (out/'stress.log').open('w') as stdout,(out/'stress.time').open('w') as stderr:
 result=measured([str(binary),'stress'],stdout,stderr,env)
(out/'stress.exit').write_text(str(result.returncode)+'\n')
if result.returncode:raise SystemExit(result.returncode)
rows=[json.loads(line) for line in (out/'stress.log').read_text().splitlines()]
(out/'stress.json').write_text(json.dumps(rows,indent=2)+'\n')
(out/'runtime.json').write_text(json.dumps({'platform':platform.platform(),'machine':platform.machine(),'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'thread_env':{k:env[k] for k in ['VECLIB_MAXIMUM_THREADS','OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},'build':'cargo build --release -p fmrireg --features dataset --example selected_qualification','memory_note':'wait4 rusage process peak includes resident original/selected/dense reference copies; not streaming-only RSS; macOS bytes','timing_note':'one cold preparation per fresh process; first stream then three warm repetitions; execution serial; no competing benchmark jobs started by this script'},indent=2)+'\n')
print('completed',len(records),'workloads and',len(rows),'stress scenarios',flush=True)
