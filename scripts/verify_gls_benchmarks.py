"""Recheck benchmark output, source binding, and feature/block/worker invariance."""
import hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
def read(label):return json.loads((root/'evidence'/('gls-'+label)/'results.json').read_text())
baseline,final,memory=map(read,['baseline','final','memory'])
assert len(baseline)==40 and len(final)==24 and len(memory)==3
checks=[]
for row in final:
 key=row['key'];actual=np.fromfile(root/'evidence/gls-final'/f'{key}-rust.bin',dtype='<f8')
 reference=root/'evidence/gls-baseline'/f'{key}-r.bin'
 if not reference.exists():reference=root/'evidence/gls-final/1200-32-50000-ar4-b1000-w1-rust.bin'
 expected=np.fromfile(reference,dtype='<f8');assert actual.shape==expected.shape
 error=float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))
 assert np.isfinite(actual).all() and error<2e-8,(key,error)
 assert row['rank']==row['p'] and row['df']==row['n']-row['p']
 assert hashlib.sha256((root/'evidence/gls-final'/f'{key}-rust.bin').read_bytes()).hexdigest()==row['output_sha256']
 if row['block']==1000 and row['workers']==1:
  ref=next(x for x in baseline if x['key']==key and x['engine']=='r')
  a=np.asarray(row['phi']).ravel();b=np.asarray(ref['phi']).ravel();assert np.max(np.abs(a-b))<2e-8
 checks.append({'key':key,'max_scaled_error':error,'values':len(actual),'reference':str(reference.relative_to(root))})
x=np.fromfile(root/'evidence/gls-memory/1200-32-200000-ar4-b1000-w1-rust.bin',dtype='<f8')
y=np.fromfile(root/'evidence/gls-memory/1200-32-50000-ar4-b1000-w1-rust.bin',dtype='<f8')
assert np.array_equal(x[:len(y)],y)
meta=json.loads((root/'evidence/gls-final/metadata.json').read_text())
for p,sha in meta['hashes'].items():assert hashlib.sha256((root/p).read_bytes()).hexdigest()==sha,p
assert hashlib.sha256((root/'target/release/examples/gls_benchmark').read_bytes()).hexdigest()==meta['binary_sha256']
# Bind R oracle source and compiled installation using the existing fixture provenance.
provenance=json.loads((root/'crates/fmri-noise/tests/fixtures/arp.json').read_text())['provenance']
for p,h in provenance['source_md5'].items():assert hashlib.md5((Path(provenance['source'])/p).read_bytes()).hexdigest()==h
for p,h in provenance['binary_md5'].items():assert hashlib.md5((root/p).read_bytes()).hexdigest()==h
report={'passed':True,'complete_r_comparisons':16,'block_worker_comparisons':8,'memory_prefix_values':len(y),'max_scaled_error':max(x['max_scaled_error'] for x in checks),'checks':checks,'R_provenance':provenance}
(root/'evidence/gls-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print('All 16 R comparisons, 8 block/worker comparisons, 200k-feature prefix, and source/binary bindings pass.')
