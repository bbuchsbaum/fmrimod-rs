#!/usr/bin/env python3
"""Independent conditional GLS and intercept-only PM/mKH browser witness."""
import gzip
import hashlib
import json
import pathlib
import sys

import numpy as np

root, receipts, output = map(pathlib.Path, sys.argv[1:])
manifest = json.loads((root/'manifest.json').read_text())
cases = [(p, json.loads(p.read_text())) for p in sorted(receipts.glob('*.json'))]
cases = [(p,r['response']) for p,r in cases if r.get('status')=='completed' and r.get('response',{}).get('effects')]
assert cases
first=cases[0][1]; coordinates=np.array(first['support'])[first['probe_indices']]
data=[]
for entry in manifest['files']:
    p=root/entry['file']
    with p.open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
    with gzip.open(p,'rb') as f:
        f.read(352); size=int(np.prod(manifest['shape']))*4
        data.append(np.array([np.frombuffer(f.read(size),dtype='<f4')[coordinates] for _ in range(sum(manifest['run_lengths']))],dtype=float))

def pm(y,v):
    def solve(tau):
        w=1/(v+tau); mean=np.sum(w*y)/sum(w); q=np.sum(w*(y-mean)**2)
        return mean,q,1/sum(w)
    df=len(y)-1
    if solve(0)[1]<=df: tau=0.
    else:
        lo=0.; hi=max(float(np.var(y,ddof=1)),float(np.max(v)))
        while solve(hi)[1]>df:hi*=2
        for _ in range(120):
            mid=(lo+hi)/2
            if solve(mid)[1]>df:lo=mid
            else:hi=mid
        tau=(lo+hi)/2
    mean,q,var=solve(tau)
    return mean,np.sqrt(var*max(1,q/df)),tau

checks=[]
for p,r in cases:
    errors=dict(effect=0.,variance=0.,group_effect=0.,group_se=0.,tau_squared=0.)
    n=sum(manifest['run_lengths']);x=np.zeros((n,6));start=0
    for run,length in enumerate(manifest['run_lengths']):
        t=np.arange(length);x[start:start+length,run]=1
        x[start:start+length,2]=np.sin(t*.07+r['phase']);x[start:start+length,3]=np.cos(t*.11)
        x[start:start+length,4+run]=t/(length-1)-.5;start+=length
    c=np.array([0,0,1,-.5 if r['contrast']=='difference' else 0,0,0]);effects=[];variances=[]
    for s in r['first_level']:
        y=data[s['subject']];xw=x.copy();yw=y.copy();start=0
        for length,phi in zip(manifest['run_lengths'],s['noise_coefficients']):
            for lag,a in enumerate(phi,1):
                xw[start+lag:start+length]-=a*x[start:start+length-lag]
                yw[start+lag:start+length]-=a*y[start:start+length-lag]
            start+=length
        b,_,rank,_=np.linalg.lstsq(xw,yw,rcond=1e-12);assert s['df']==n-rank
        effect=c@b;g=c@np.linalg.pinv(xw,rcond=1e-12)
        variance=np.sum((yw-xw@b)**2,axis=0)/(n-rank)*(g@g)
        effects.append(effect);variances.append(variance)
        errors['effect']=max(errors['effect'],float(np.max(np.abs(effect-s['estimates']))))
        if r['variance']: errors['variance']=max(errors['variance'],float(np.max(np.abs(variance-s['variances']))))
    effects=np.array(effects);variances=np.array(variances)
    for j,k in enumerate(r['probe_indices']):
        if r['group']=='meta':mean,se,tau=pm(effects[:,j],variances[:,j])
        else:mean=effects[:,j].mean();se=effects[:,j].std(ddof=1)/5;tau=None
        errors['group_effect']=max(errors['group_effect'],abs(mean-r['effects'][k]))
        if r['group_se'][k] is not None:errors['group_se']=max(errors['group_se'],abs(se-r['group_se'][k]))
        if tau is not None:errors['tau_squared']=max(errors['tau_squared'],abs(tau-r['tau_squared'][k]))
    assert max(errors.values())<2e-9,(p.name,errors)
    checks.append(dict(case=p.name,group=r['group'],errors=errors,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
assert any(c['group']=='meta' for c in checks)
output.write_text(json.dumps(dict(checks=checks,tolerance=2e-9,numpy=np.__version__,scope='Conditional GLS at four probes per subject, followed by independent bisection PM and modified KH group SE. AR estimation itself is not independently re-estimated.'),indent=2)+'\n')
print('Passed',len(checks),'successful browser/protocol results including weighted group inference.')
