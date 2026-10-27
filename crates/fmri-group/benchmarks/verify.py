"""Bind local numerical/test receipts to source hashes and the isolated R build."""
import datetime, hashlib, json, pathlib, re, subprocess
base = pathlib.Path('evidence/group-tranche1')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def counts(name):
    text = (base/name).read_text()
    assert 'FAILED' not in text
    matches = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored', text)
    assert matches
    return dict(zip(('passed','failed','ignored'),(sum(int(x[i]) for x in matches) for i in range(3))), batches=len(matches))
manifest = json.loads(pathlib.Path('evidence/group-audit/source-manifest.json').read_text())
copy = pathlib.Path(manifest['reference']['isolated_build_copy'])
assert all(sha(copy/p)==h for p,h in manifest['reference']['sha256'].items())
paths = list(pathlib.Path('crates/fmri-group').rglob('*.rs')) + list(pathlib.Path('crates/fmri-group/benchmarks').glob('*'))
paths += [pathlib.Path(p) for p in ['crates/fmri-group/Cargo.toml','crates/fmri-group/tests/reference.json','scripts/group_reference.R','crates/fmrimod/src/lib.rs','crates/fmrimod/Cargo.toml','crates/fmrimod/tests/group.rs','Cargo.toml','Cargo.lock']]
producer = list(pathlib.Path('crates/fmri-reg/src').glob('*.rs')) + list(pathlib.Path('crates/fmri-data/src').glob('*.rs'))
for label in ['r-benchmark','rust-benchmark']:
    run = json.loads((base/f'{label}-resource.json').read_text())
    assert run['exit']==0
    assert all(sha(pathlib.Path(p))==h for p,h in run['command_file_sha256'].items())
reference = json.loads((base/'r-benchmark.json').read_text())
rust = json.loads((base/'rust-benchmark.json').read_text())
assert len(rust)==len(reference)==14
assert all((a['id'],a['method'])==(b['id'],b['method']) for a,b in zip(reference,rust))
assert max(x['max_scaled_error'] for x in rust)<2e-9
record = {
 'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'reference_head':manifest['reference']['head'],
 'isolated_reference_source_files_verified':len(manifest['reference']['sha256']),
 'live_reference_drift':json.loads((base/'reference-drift.json').read_text()),
 'rustc':subprocess.check_output(['rustc','--version'],text=True).strip(),
 'source_sha256':{str(p):sha(p) for p in paths if p.is_file()},
 'dependency_source_sha256':{str(p):sha(p) for p in producer},
 'installed_reference_library_sha256':{str(p):sha(p) for p in pathlib.Path('evidence/group-audit/r-library/fmrigds').rglob('*') if p.is_file()},
 'tests':{'whole_workspace_all_features_before_final_facade_and_status_additions':counts('workspace-tests.log'),'final_affected_all_features':counts('final-affected-tests.log')},
 'clippy':'passed: fmri-group --all-targets --no-deps -D warnings',
 'reference':'base-R QR, metafor FE/DL, source-built fmrigds FE/DL, base-R p.adjust',
 'benchmarks':{'cases':14,'max_scaled_error':max(x['max_scaled_error'] for x in rust),'block_sizes_compared':[256,113],'receipts':['r-benchmark-resource.json','rust-benchmark-resource.json']},
 'limits':['shared workstation timings','in-memory sources only','numeric budget excludes resident sources, metadata, allocator/backend scratch, sink-retained outputs','iid normal null calibration only','no hosted CI or publication'],
 'changes_exclude_first_level_kernels':True,
}
(base/'verification.json').write_text(json.dumps(record,indent=2)+'\n')
print('Verified:',record['tests'],record['benchmarks'])
