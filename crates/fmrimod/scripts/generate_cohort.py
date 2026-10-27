#!/usr/bin/env python3
"""Deterministic synthetic 4D NIfTI-1 cohort; no clinical/research data.

Writes x-fastest float32 frames, explicit millimeter geometry and separate
reference design/effects. Generation is outside measured cold-session latency.
"""
import argparse, gzip, hashlib, json, pathlib, struct
import numpy as np
p=argparse.ArgumentParser();p.add_argument('directory',type=pathlib.Path);p.add_argument('--subjects',type=int,default=25);p.add_argument('--frames',type=int,default=300);p.add_argument('--gzip',action='store_true');args=p.parse_args()
args.directory.mkdir(parents=True,exist_ok=True)
n=args.frames;nx,ny,nz=32,32,8;v=nx*ny*nz
x=np.empty((n,8))
for i in range(n):
 for j in range(8):x[i,j]=1. if j==0 else np.sin((i+1)*(j+1)*0.017)
np.savetxt(args.directory/'design.csv',x,delimiter=',',fmt='%.17g')
voxel=np.arange(v);xx=voxel%nx;yy=voxel//nx%ny;zz=voxel//(nx*ny)
blob=np.exp(-((xx-16.)**2+(yy-16.)**2+4*(zz-3.5)**2)/80.)
files=[]
for subject in range(args.subjects):
 rng=np.random.default_rng(70591+subject)
 noise=rng.normal(0.,1.,(n,v))
 for i in range(n):
  for k,phi in enumerate([.4,-.1,.08,-.03]):
   if i>k:noise[i]+=phi*noise[i-k-1]
 beta=np.zeros((8,v));beta[0]=1000.;beta[1]=(1.+subject*.02)*blob;beta[2]=.15;beta[3:]=.04
 values=(x@beta+noise).astype('<f4')
 h=bytearray(352);struct.pack_into('<i',h,0,348);struct.pack_into('<8h',h,40,4,nx,ny,nz,n,1,1,1);struct.pack_into('<2h',h,70,16,32);struct.pack_into('<8f',h,76,1.,2.,2.,2.,1.,1.,1.,1.);struct.pack_into('<3f',h,108,352.,1.,0.);h[123]=10;struct.pack_into('<h',h,254,1);struct.pack_into('<12f',h,280,2.,0.,0.,-32.,0.,2.,0.,-32.,0.,0.,2.,-8.);h[344:348]=b'n+1\0'
 name=f'sub-{subject:02d}.nii'+('.gz' if args.gzip else '')
 path=args.directory/name
 with (gzip.open(path,'wb') if args.gzip else path.open('wb')) as f:f.write(h);f.write(values.tobytes(order='C'))
 files.append({'file':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
manifest={'subjects':args.subjects,'frames':n,'dimensions':[nx,ny,nz],'features':v,'design_columns':8,'dtype':'float32','noise':'fixed zero-history AR4 [.4,-.1,.08,-.03], one run','seed_base':70591,'spatial_identity':'synthetic-grid, not registration evidence','numpy':np.__version__,'files':files}
(args.directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:manifest[k] for k in ['subjects','frames','features','dtype']}))
