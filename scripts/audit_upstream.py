"""Detect local R reference drift; no fetch, source mutation or automatic acceptance."""
from pathlib import Path
import hashlib,json,subprocess
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'evidence/source-manifest.json').read_text())
drift=False
for label,record in manifest.items():
 if not label.startswith('R:'):continue
 source=Path(record['source']);current={}
 files=[source/'DESCRIPTION',source/'NAMESPACE']
 for folder in ['R','src','tests/testthat']:
  files.extend(p for p in (source/folder).rglob('*') if p.is_file() and p.suffix in ['.R','.cpp','.h','.hpp'])
 for p in files:
  if p.is_file():current[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
 changed=sorted(k for k in set(record['files'])|set(current) if current.get(k)!=record['files'].get(k))
 head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
 print(f'{label[2:]}: HEAD {head}; {len(changed)} changed reference files')
 for name in changed:print('  '+name)
 if head!=record['git_head']:print('  Git HEAD differs from recorded reference; inspect non-source changes too.')
 drift |= bool(changed) or head!=record['git_head']
raise SystemExit(1 if drift else 0)
