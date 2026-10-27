"""Verify that original imported working files have not changed since capture."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'evidence/source-manifest.json').read_text())
errors=[]; checked=0
for label,record in manifest.items():
 for name,digest in record['files'].items():
  source=Path(record['source'])/name
  if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=digest:
   errors.append(f'{label}: {source}')
  checked+=1
print(f'Checked {checked} original source files; changes={len(errors)}')
for error in errors: print(error)
raise SystemExit(bool(errors))
