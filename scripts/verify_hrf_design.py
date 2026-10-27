"""Bind the HRF/design qualification receipts to local sources and references."""
from pathlib import Path
import hashlib,json,subprocess,re
root=Path(__file__).resolve().parents[1]
suite=json.loads((root/'evidence/hrf-design-final.json').read_text())
assert all(x['exit_code']==0 for x in suite)
assert suite[-1]['label']=='doctests'
for x in suite:assert hashlib.sha256((root/x['log']).read_bytes()).hexdigest()==x['sha256']
f=json.loads((root/'crates/fmrihrf/tests/fixtures/hrf_audit.json').read_text())
for p,h in f['provenance']['source_md5'].items():assert hashlib.md5((Path(f['provenance']['source'])/p).read_bytes()).hexdigest()==h
p=json.loads((root/'crates/fmrimod/tests/fixtures/pipeline.json').read_text())
assert hashlib.md5((root/p['input_fixture']).read_bytes()).hexdigest()==p['input_md5']
s=json.loads((root/'evidence/scala-design-reference.json').read_text())
for p,h in s['files'].items():assert hashlib.sha256((Path(s['source'])/p).read_bytes()).hexdigest()==h
checks={}
for name in ['hrf-design-post','hrf-design-minimal','hrf-design-clippy','hrf-design-format']:
 log=root/'evidence'/(name+'.log');code=int((root/'evidence'/(name+'.exit')).read_text());assert code==0,(name,code)
 counts=re.findall(r'test result: \w+\. (\d+) passed; (\d+) failed; (\d+) ignored;',log.read_text())
 checks[name]={'exit':code,'passed':sum(int(c[0]) for c in counts),'failed':sum(int(c[1]) for c in counts),'ignored':sum(int(c[2]) for c in counts),'log_sha256':hashlib.sha256(log.read_bytes()).hexdigest()}
files=[p for p in (root/'crates').rglob('*') if p.is_file() and p.suffix in ['.rs','.json','.toml','.csv']]+[root/'Cargo.toml',root/'Cargo.lock']
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(root/'evidence/hrf-design-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
record={'full_workspace':{'batches':len(suite),'passed':sum(x['passed'] for x in suite),'failed':sum(x['failed'] for x in suite),'ignored':sum(x['ignored'] for x in suite),'receipt':'evidence/hrf-design-final.json'},'post_checks':checks,'source_manifest_sha256':hashlib.sha256((root/'evidence/hrf-design-source.json').read_bytes()).hexdigest(),'scope':'All-feature workspace campaign plus affected-crate rechecks after final missing-pair guard. R/Scala reference files verified unchanged. Legacy Clippy warnings are not claimed clean.','rustc':subprocess.check_output(['rustc','--version'],text=True).strip()}
(root/'evidence/hrf-design-verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
