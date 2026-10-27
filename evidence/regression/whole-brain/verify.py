"""Independent NumPy GLS/group-mean witness for generated cohort fixtures."""
import json, pathlib, re, sys
import numpy as np
source=pathlib.Path(sys.argv[1]); results=pathlib.Path(sys.argv[2])
m=json.loads((source/'manifest.json').read_text()); n=m['frames'];v=m['features']
indices=np.array([0,1,63,64,4095,32768,65536,v-1]);x=np.loadtxt(source/'design.csv',delimiter=',')
phi=[.4,-.1,.08,-.03]
def whiten(y):
 out=y.copy()
 for lag,p in enumerate(phi,1):out[lag:]-=p*y[:-lag]
 return out
reports=[]
for stage,phase,contrast in [('cold',0.,False),('design-edit',.18,False),('contrast-edit',.18,True)]:
 design=x.copy()
 if phase:design[:,1]=np.sin((np.arange(n)+1)*2*.017+phase)
 wx=whiten(design); rows=[]
 for f in m['files']:
  raw=np.memmap(source/f['file'],dtype='<f4',mode='r',offset=352,shape=(n,v))
  beta=np.linalg.lstsq(wx,whiten(np.asarray(raw[:,indices],dtype=np.float64)),rcond=1e-12)[0]
  rows.append(beta[1]-(.5*beta[2] if contrast else 0.))
 expected=np.mean(rows,axis=0)
 html=(results/(stage+'.html')).read_text(); match=re.search(r'const values=(\[.*?\]),nx=',html)
 actual=np.array(json.loads(match.group(1)))[indices]
 err=float(np.max(np.abs(expected-actual)))
 assert err<2e-10,(stage,err)
 reports.append({'stage':stage,'features':indices.tolist(),'max_abs_error':err,'tolerance':2e-10})
print(json.dumps({'numpy':np.__version__,'subjects':len(m['files']),'checks':reports},indent=2))
