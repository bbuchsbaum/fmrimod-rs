"""Check workspace portability and summarize executable test receipts."""
from pathlib import Path
import json,subprocess,re,hashlib,platform
root=Path(__file__).resolve().parents[1]
metadata=json.loads(subprocess.check_output(['cargo','metadata','--no-deps','--format-version','1'],cwd=root,text=True))
for package in metadata['packages']:
 for dep in package['dependencies']:
  if 'path' in dep:
   assert Path(dep['path']).resolve().is_relative_to(root),dep
records={}
for log in ['workspace-final.log','kernel-clippy-final.log','rust-throughput.log','r-throughput.log','final-contrast-guards.log']:
 p=root/'evidence'/log
 if not p.exists():continue
 text=p.read_text()
 totals=re.findall(r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored;',text)
 records[log]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'test_summaries':len(totals),'passed':sum(int(t[1]) for t in totals),'failed':sum(int(t[2]) for t in totals),'ignored':sum(int(t[3]) for t in totals)}
record={'platform':platform.platform(),'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),'workspace_members':[p['name'] for p in metadata['packages']], 'all_path_dependencies_internal':True,'logs':records}
bounded=root/'evidence'/'bounded-checks.json'
if bounded.exists():
 checks=json.loads(bounded.read_text())
 record['bounded_checks']=checks
 record['bounded_suite_exit']=int((root/'evidence'/'bounded-checks.exit').read_text()) if (root/'evidence'/'bounded-checks.exit').exists() else None
 record['bounded_totals']={k:sum(c[k] for c in checks) for k in ['passed','failed','ignored']}
source_files=sorted(p for p in (root/'crates').rglob('*') if p.is_file() and p.suffix in ['.rs','.toml','.json','.csv'])
source_files += [root/'Cargo.toml',root/'Cargo.lock']
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
(root/'evidence'/'qualified-source.json').write_text(json.dumps(hashes,indent=2)+'\n')
record['post_suite_checks']={name:int((root/'evidence'/(name+'.exit')).read_text()) for name in ['final-contrast-guards','first-level','kernel-clippy-final'] if (root/'evidence'/(name+'.exit')).exists()}
record['source_manifest_sha256']=hashlib.sha256((root/'evidence'/'qualified-source.json').read_bytes()).hexdigest()
(root/'evidence'/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
