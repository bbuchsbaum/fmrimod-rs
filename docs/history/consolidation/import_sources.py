"""One-time, non-destructive import of explicitly selected local source content."""
from pathlib import Path
import hashlib, json, shutil, subprocess
root = Path(__file__).resolve().parents[1]
base = root.parent
sources = {'fmrihrf': base/'fmrihrf-rs', 'fmridesign': base/'fmridesign-rs',
           'fmridataset': base/'fmridataset-rs'/'crates'/'fmridataset', 'splines': base/'splines'}
manifest = {}
for name, src in sources.items():
    dest = root/'crates'/name
    if dest.exists(): raise RuntimeError(f'Refusing to overwrite {dest}')
    chosen = []
    for folder in ['src','tests','examples','benches','scripts']:
        p = src/folder
        if p.exists(): chosen.extend(x for x in p.rglob('*') if x.is_file() and not any(s.startswith('.') for s in x.relative_to(src).parts))
    chosen.extend(p for p in src.iterdir() if p.is_file() and (p.suffix in ['.md', '.rs'] or p.name.startswith('LICENSE') or p.name in ['Cargo.toml','Cargo.lock']))
    record = {'source':str(src),'files':{}}
    result = subprocess.run(['git','-C',str(src),'rev-parse','HEAD'],capture_output=True,text=True)
    record['git_head'] = result.stdout.strip() if result.returncode == 0 else None
    if record['git_head']:
        record['git_status'] = subprocess.check_output(['git','-C',str(src),'status','--porcelain'],text=True)
        subprocess.run(['git','-C',str(src),'bundle','create',str(root/'evidence'/f'{name}-history.bundle'),'--all'],check=True)
    for p in sorted(set(chosen)):
        rel = p.relative_to(src)
        data = p.read_bytes()
        record['files'][str(rel)] = hashlib.sha256(data).hexdigest()
        target = dest/rel; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(data)
    manifest[name] = record
for name in ['fmrihrf','fmridesign','fmridataset','fmrireg','fmriAR']:
    src = Path.home()/'code'/name
    files = [src/'DESCRIPTION',src/'NAMESPACE']
    for folder in ['R','src','tests/testthat']:
        files += [p for p in (src/folder).rglob('*') if p.is_file() and p.suffix in ['.R','.cpp','.h','.hpp']]
    manifest['R:'+name] = {'source':str(src),'git_head':subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),
        'files':{str(p.relative_to(src)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
(root/'evidence'/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Imported:',', '.join(sources),'; original sources unchanged')
