#!/usr/bin/env python3
"""Synthetic masked/two-run fixture and localhost qualification viewer.

No production data, external assets, telemetry or public listener. Serve launches
only the supplied Rust example and closes it when this server exits.
"""
import argparse, gzip, hashlib, json, pathlib, signal, struct, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Cohort analysis qualification</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101820;color:#e8eef2;font:16px system-ui,sans-serif}main{max-width:1120px;margin:32px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}.muted{color:#aabcc8;line-height:1.5}.layout{display:grid;grid-template-columns:300px 1fr;gap:28px;margin-top:28px}.panel{background:#1b2833;border:1px solid #3b5263;border-radius:10px;padding:20px}label{display:block;margin:0 0 16px;line-height:1.7}select,input[type=number]{width:100%;padding:9px;background:#0e1b25;color:#fff;border:1px solid #6b8495;border-radius:4px;font:inherit}button{font:inherit;border:0;border-radius:4px;padding:11px 16px;background:#88d8c0;color:#0a2822;cursor:pointer;font-weight:650;margin:0 8px 10px 0}button.secondary{background:#d5e4ed;color:#152c3a}button:disabled{opacity:.5;cursor:wait}canvas{display:block;width:100%;max-width:512px;aspect-ratio:1;margin:auto;background:#06111a;image-rendering:pixelated}input[type=range]{width:100%;accent-color:#88d8c0}.facts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}.fact{padding:12px;background:#101e28;border-radius:5px}.fact strong{display:block;font-size:20px;color:#a6e6d3}.small{font-size:13px;line-height:1.5}.error{color:#ffb7a8}#status{min-height:24px}h2{font-size:18px;margin:20px 0 12px}@media(max-width:750px){.layout{grid-template-columns:1fr}}
</style><main><h1>Design → group contrast</h1><div class="muted">Synthetic qualification · 25 subjects · two runs · explicit spatial mask</div><div class="layout"><section class="panel" aria-label="Analysis specification">
<label>Task phase shift<input id="phase" type="number" min="-1" max="1" step="0.01" value="0"></label>
<label>Contrast<select id="contrast"><option value="task">Task effect</option><option value="difference">Task − ½ nuisance</option></select></label>
<label>Noise model<select id="noise"><option value="estimated">Estimated AR(2), per run</option><option value="fixed">Fixed AR(2), per run</option></select></label>
<label>Execution<select id="mode"><option value="resident">Resident · repeated contrast edits</option><option value="files">File batches · bounded memory</option></select></label>
<label><input id="variance" type="checkbox" checked> Supply first-level variances</label>
<button id="preview" class="secondary">Inspect plan</button><button id="run">Run analysis</button>
<p id="status" role="status">Ready to inspect or run.</p><p id="error" class="error" role="alert"></p><h2>Execution plan</h2><div id="plan" class="small muted">Inspection reads no voxel values. Retention and requested outputs are separate choices.</div></section>
<section class="panel" aria-label="Group contrast viewer"><canvas id="map" aria-label="Axial group contrast"></canvas><label>Axial slice <span id="sliceValue">—</span><input id="slice" type="range" min="0" max="31" value="16"></label><p id="scale" class="small muted">No result loaded.</p><div class="facts"><div class="fact">Map ready<strong id="latency">—</strong></div><div class="fact">This request<strong id="reuse">—</strong></div></div><p class="small muted">Group mean effect, unthresholded. Black is outside the mask. Voxel array orientation; no MNI registration claim. Timing includes the local request and two animation frames, not physical display latency.</p></section></div></main>
<script>
const $=id=>document.getElementById(id);let result=null,volume=null,maximum=1;window.renderSequence=0;window.drawCount=0;
function draw(){if(!result)return;const [nx,ny]=result.shape,z=+$('slice').value,c=$('map');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),im=ctx.createImageData(nx,ny);for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const index=x+nx*(y+ny*z),value=volume[index],k=4*(x+nx*(ny-1-y));if(Number.isNaN(value)){im.data[k]=6;im.data[k+1]=17;im.data[k+2]=26;}else{const v=value/maximum,a=Math.min(1,Math.abs(v));im.data[k]=v>=0?255:255*(1-a);im.data[k+1]=255*(1-a);im.data[k+2]=v<=0?255:255*(1-a);}im.data[k+3]=255;}ctx.putImageData(im,0,0);$('sliceValue').textContent=z;window.drawCount++;}
function showPlan(p){const count=key=>p.subjects.filter(s=>s.model===key).length;const first=p.subjects[0];$('plan').textContent=`${p.storage}; ${p.workers} workers; ${(p.reserved_numeric_bytes/1048576).toFixed(1)} MiB numerical reservation. Retention: ${p.retention}. Outputs: ${first.uncertainty}. Models: ${count('Reuse')} reuse, ${count('EstimateNoise')} estimate noise, ${count('Prepare')} prepare. ${p.reason}. Metadata and backend scratch are outside this budget.`;}
async function submit(action){$('run').disabled=$('preview').disabled=true;$('error').textContent='';$('status').textContent=action==='inspect'?'Inspecting…':'Running analysis…';const start=performance.now();window.requestStart=start;const command={action,phase:+$('phase').value,contrast:$('contrast').value,noise:$('noise').value,mode:$('mode').value,variance:$('variance').checked};try{const response=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command)});const data=await response.json();if(data.error)throw Error(data.error);showPlan(data.plan);if(action==='inspect'){window.lastInspection=data;$('status').textContent='Plan inspected; no voxel values read.';return;}result=data;volume=new Float64Array(data.shape.reduce((a,b)=>a*b,1)).fill(NaN);data.support.forEach((p,i)=>volume[p]=data.effects[i]);maximum=data.effects.reduce((m,v)=>Math.max(m,Math.abs(v)),0)||1;$('slice').max=data.shape[2]-1;draw();$('scale').textContent=`Blue ${(-maximum).toFixed(3)} · white 0 · red ${maximum.toFixed(3)} signal units`;$('reuse').textContent=`${data.source_reads} reads / ${data.model_preparations} prepares`;await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));window.lastResult=data;window.lastRender={sequence:++window.renderSequence,command,request_start_ms:start,render_opportunity_ms:performance.now()-start,server_ms:data.server_ms};$('latency').textContent=(window.lastRender.render_opportunity_ms/1000).toFixed(3)+' s';$('status').textContent='Map ready.';}catch(e){$('error').textContent=e.message;$('status').textContent='Analysis failed.';}finally{$('run').disabled=$('preview').disabled=false;}}
$('run').onclick=()=>submit('run');$('preview').onclick=()=>submit('inspect');$('slice').oninput=draw;
</script></html>'''

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def generate(directory):
    import numpy as np
    directory.mkdir(parents=True,exist_ok=False)
    shape=(48,48,32);v=int(np.prod(shape));runs=[120,120];n=sum(runs)
    pos=np.arange(v);x=pos%shape[0];y=pos//shape[0]%shape[1];z=pos//(shape[0]*shape[1])
    mask=((x-23.5)/20)**2+((y-23.5)/20)**2+((z-15.5)/13)**2<1
    support=np.flatnonzero(mask);(directory/'mask.json').write_text(json.dumps(support.tolist()))
    blob=np.exp(-((x-25)**2+(y-22)**2+2*(z-16)**2)/70)
    design=np.zeros((n,6))
    for t in range(n):
        run=t//120;local=t%120;design[t,run]=1;design[t,2]=np.sin(local*.07);design[t,3]=np.cos(local*.11);design[t,4+run]=local/119-.5
    np.savetxt(directory/'design.csv',design,delimiter=',',fmt='%.17g')
    files=[]
    for subject in range(25):
        rng=np.random.default_rng(84521+subject);noise=rng.normal(size=(n,v))
        for run,phi in enumerate([[.4,-.12],[.2,.05]]):
            for t in range(run*120,(run+1)*120):
                for lag,p in enumerate(phi,1):
                    if t-run*120>=lag:noise[t]+=p*noise[t-lag]
        beta=np.zeros((6,v));beta[0]=1000;beta[1]=1050;beta[2]=(1+subject*.02)*blob;beta[3]=.2;beta[4]=.3;beta[5]=-.2
        values=(design@beta+noise).astype('<f4');values[:,~mask]=0
        h=bytearray(352);struct.pack_into('<i',h,0,348);struct.pack_into('<8h',h,40,4,*shape,n,1,1,1);struct.pack_into('<2h',h,70,16,32);struct.pack_into('<8f',h,76,1,3,3,3,1,1,1,1);struct.pack_into('<3f',h,108,352,1,0);h[123]=10;struct.pack_into('<h',h,254,1);struct.pack_into('<12f',h,280,3,0,0,-72,0,3,0,-72,0,0,3,-48);h[344:348]=b'n+1\0'
        path=directory/f'sub-{subject:02d}.nii.gz'
        with path.open('wb') as dst:
            with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as gz:gz.write(h);gz.write(values.tobytes())
        files.append({'id':f'sub-{subject:02d}','file':path.name,'sha256':digest(path),'bytes':path.stat().st_size})
    manifest={'shape':shape,'run_lengths':runs,'masked_features':len(support),'subjects':25,'seed_base':84521,'numpy':np.__version__,'mask_sha256':digest(directory/'mask.json'),'generator_sha256':digest(pathlib.Path(__file__)),'noise':[[.4,-.12],[.2,.05]],'files':files,'provenance':'Generated synthetic ellipsoid mask, two runs, nuisance and run intercepts; not research data or MNI registration evidence'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'features':len(support),'bytes':sum(f['bytes'] for f in files)}))

def serve(args):
    import os
    receipts=args.receipts;receipts.mkdir(parents=True,exist_ok=True)
    backend=subprocess.Popen([str(args.binary),str(args.directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
    class Handler(BaseHTTPRequestHandler):
        sequence=0
        def log_message(self,*args):pass
        def send(self,body,status=200,mime='application/json'):
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            if self.path=='/':self.send(HTML.encode(),mime='text/html; charset=utf-8')
            elif self.path=='/health':self.send(b'{"ready":true}')
            else:self.send(b'{}',404)
        def do_POST(self):
            if self.path!='/api':return self.send(b'{}',404)
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<=1024:raise ValueError('invalid request size')
                command=json.loads(self.rfile.read(length));start=time.perf_counter()
                backend.stdin.write(json.dumps(command)+'\n');backend.stdin.flush();line=backend.stdout.readline()
                if not line:raise RuntimeError('analysis process exited')
                response=json.loads(line);Handler.sequence+=1
                receipt={'command':command,'response':response,'http_backend_ms':(time.perf_counter()-start)*1000}
                (receipts/f'{Handler.sequence:02d}-{command.get("action","unknown")}.json').write_text(json.dumps(receipt)+'\n')
                self.send(json.dumps(response).encode(),500 if 'error' in response else 200)
            except Exception as error:self.send(json.dumps({'error':str(error)}).encode(),500)
    server=HTTPServer(('127.0.0.1',args.port),Handler)
    args.info.write_text(json.dumps({'server_pid':os.getpid(),'backend_pid':backend.pid,'port':server.server_port,'owner':'analysis-plan qualification','binary':str(args.binary)})+'\n')
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();backend.stdin.close()
        try:backend.wait(timeout=5)
        except subprocess.TimeoutExpired:backend.terminate();backend.wait(timeout=5)

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    gen=sub.add_parser('generate');gen.add_argument('directory',type=pathlib.Path)
    web=sub.add_parser('serve');web.add_argument('directory',type=pathlib.Path);web.add_argument('--binary',type=pathlib.Path,required=True);web.add_argument('--port',type=int,default=0);web.add_argument('--info',type=pathlib.Path,required=True);web.add_argument('--receipts',type=pathlib.Path,required=True)
    args=p.parse_args()
    if args.action=='generate':generate(args.directory)
    else:serve(args)
