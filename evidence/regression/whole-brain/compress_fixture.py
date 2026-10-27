import gzip,hashlib,json,pathlib,shutil
root=pathlib.Path('/private/tmp/fmrimod-whole-input');manifest=json.loads((root/'manifest.json').read_text())
def sha(path,opener=open):
 h=hashlib.sha256()
 with opener(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
for i,entry in enumerate(manifest['files']):
 source=root/entry['file'];target=source.with_suffix('.nii.gz')
 assert sha(source)==entry['sha256']
 with source.open('rb') as src,target.open('wb') as dst:
  with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as gz:shutil.copyfileobj(src,gz,1024*1024)
 assert sha(target,gzip.open)==entry['sha256']
 entry['uncompressed_sha256']=entry['sha256'];entry['uncompressed_bytes']=entry['bytes'];entry['file']=target.name;entry['sha256']=sha(target);entry['bytes']=target.stat().st_size
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 source.unlink() # Only this task's generated fixture, verified round-trip first.
 if i%5==4:print('compressed and verified',i+1,flush=True)
print('compressed bytes',sum(f['bytes'] for f in manifest['files']),flush=True)
