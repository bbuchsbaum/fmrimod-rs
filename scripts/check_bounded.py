"""Execute the all-feature workspace suite without retaining every test executable.
Only executables Cargo reports for successful commands in this workspace's own
target directory are removed. Sources, dependencies and failed artifacts remain.
"""
from pathlib import Path
import json, subprocess, sys, re, hashlib, argparse
parser=argparse.ArgumentParser()
parser.add_argument("--label",default="bounded-checks")
parser.add_argument("--default-features",action="store_true",help="Check workspace defaults rather than all features")
parser.add_argument("--resume",action="store_true",help="Reuse passed targets only when edits affect failed or unrun targets")
args=parser.parse_args()
if not re.fullmatch(r"[a-zA-Z0-9_-]+",args.label):parser.error("invalid receipt label")
receipt_label=args.label
default_features=args.default_features
root=Path(__file__).resolve().parents[1]
out=root/'evidence'/receipt_label;out.mkdir(exist_ok=True)
metadata=json.loads(subprocess.check_output(['cargo','metadata','--no-deps','--format-version','1'],cwd=root,text=True))
commands=[('libraries',['--lib'])]
for kind,flag in [('test','--test'),('example','--example')]:
 names=sorted({t['name'] for p in metadata['packages'] for t in p['targets'] if kind in t['kind']})
 commands += [(kind+'-'+name,[flag,name]) for name in names]
commands.append(('doctests',['--doc']))
previous={}
receipt=root/'evidence'/(receipt_label+'.json')
if args.resume and receipt.exists():
 previous={r['label']:r for r in json.loads(receipt.read_text()) if r['exit_code']==0}
records=[]
for label,args in commands:
 if label in previous:
  records.append(previous[label]);print(f'{label}: retained successful target',flush=True);continue
 cmd=['cargo','test','--workspace']+([] if default_features else ['--all-features'])+['-j','1']+args
 if label!='doctests':cmd+=['--message-format=json']
 log=out/(label+'.log')
 with log.open('w') as f:result=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT)
 text=log.read_text();totals=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
 row={'label':label,'command':cmd,'exit_code':result.returncode,'log':str(log.relative_to(root)),'sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
      'passed':sum(int(t[1]) for t in totals),'failed':sum(int(t[2]) for t in totals),'ignored':sum(int(t[3]) for t in totals)}
 records.append(row)
 (root/'evidence'/(receipt_label+'.json')).write_text(json.dumps(records,indent=2)+'\n')
 print(f'{label}: exit={result.returncode}, passed={row["passed"]}, ignored={row["ignored"]}',flush=True)
 if result.returncode:
  print('\n'.join(text.splitlines()[-30:]),flush=True);sys.exit(result.returncode)
 for line in text.splitlines():
  try:msg=json.loads(line)
  except (ValueError,TypeError):continue
  if msg.get('reason')!='compiler-artifact' or not msg.get('executable'):continue
  p=Path(msg['executable'])
  if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root/'target'):
   p.unlink()
print(f'All {len(records)} bounded workspace checks passed.',flush=True)
