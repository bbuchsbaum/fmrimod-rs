#!/usr/bin/env python3
"""Synthetic masked/two-run fixture and localhost qualification viewer.

No production data, external assets, telemetry or public listener. Serve launches
only the supplied Rust example and closes it when this server exits.
"""
import argparse, collections, gzip, hashlib, json, os, pathlib, signal, struct, subprocess, sys, tempfile, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Cohort analysis qualification</title>
<style>*{box-sizing:border-box}body{margin:0;background:#101820;color:#e8eef2;font:16px system-ui,sans-serif}main{max-width:1120px;margin:32px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}.muted{color:#aabcc8;line-height:1.5}.layout{display:grid;grid-template-columns:300px 1fr;gap:28px;margin-top:28px}.panel{background:#1b2833;border:1px solid #3b5263;border-radius:10px;padding:20px}label{display:block;margin:0 0 16px;line-height:1.7}select,input[type=number]{width:100%;padding:9px;background:#0e1b25;color:#fff;border:1px solid #6b8495;border-radius:4px;font:inherit}button{font:inherit;border:0;border-radius:4px;padding:11px 16px;background:#88d8c0;color:#0a2822;cursor:pointer;font-weight:650;margin:0 8px 10px 0}button.secondary{background:#d5e4ed;color:#152c3a}button:disabled{opacity:.5;cursor:wait}canvas{display:block;width:100%;max-width:512px;aspect-ratio:1;margin:auto;background:#06111a;image-rendering:pixelated}input[type=range]{width:100%;accent-color:#88d8c0}.facts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:18px}.fact{padding:12px;background:#101e28;border-radius:5px}.fact strong{display:block;font-size:20px;color:#a6e6d3}.small{font-size:13px;line-height:1.5}.error{color:#ffb7a8}#status{min-height:24px}h2{font-size:18px;margin:20px 0 12px}@media(max-width:750px){.layout{grid-template-columns:1fr}}
</style><main><h1>Design → group contrast</h1><div class="muted">Synthetic qualification · 25 subjects · two runs · explicit spatial mask</div><div class="layout"><section class="panel" aria-label="Analysis specification">
<label>Task phase shift<input id="phase" type="number" min="-1" max="1" step="0.01" value="0"></label>
<label>Contrast<select id="contrast"><option value="task">Task effect</option><option value="difference">Task − ½ nuisance</option></select></label>
<label>Noise model<select id="noise"><option value="estimated">Estimated AR(2), per run</option><option value="fixed">Fixed AR(2), per run</option></select></label>
<label>Group estimator<select id="group"><option value="ols">OLS group mean</option><option value="meta">PM + modified KH</option></select></label>
<label>Execution<select id="mode"><option value="resident">Keep time series</option><option value="files">Reload each request</option><option value="detached">Keep fitted coefficients</option></select></label>
<label><input id="variance" type="checkbox" checked> Supply first-level variances</label>
<button id="preview" class="secondary">Inspect plan</button><button id="run">Run analysis</button><button id="cancel" class="secondary" disabled>Cancel</button>
<p id="status" role="status">Ready to inspect or run.</p><p id="error" class="error" role="alert"></p><h2>Execution plan</h2><div id="plan" class="small muted">Inspection reads no voxel values. Retention and requested outputs are separate choices.</div></section>
<section class="panel" aria-label="Group contrast viewer"><canvas id="map" aria-label="Axial group contrast"></canvas><label>Axial slice <span id="sliceValue">—</span><input id="slice" type="range" min="0" max="31" value="16"></label><p id="scale" class="small muted">No result loaded.</p><div class="facts"><div class="fact">Map ready<strong id="latency">—</strong></div><div class="fact">This request<strong id="reuse">—</strong></div></div><p class="small muted">Group mean effect, unthresholded. Black is outside the mask. Voxel array orientation; no MNI registration claim. Timing includes the local request and two animation frames, not physical display latency.</p></section></div></main>
<script>
const $=id=>document.getElementById(id);let result=null,volume=null,maximum=1,currentRequest=null,pollTimer=null,submission=Promise.resolve(),generation=0;window.renderSequence=0;window.drawCount=0;
function draw(){if(!result)return;const [nx,ny]=result.shape,z=+$('slice').value,c=$('map');c.width=nx;c.height=ny;const ctx=c.getContext('2d'),im=ctx.createImageData(nx,ny);for(let y=0;y<ny;y++)for(let x=0;x<nx;x++){const index=x+nx*(y+ny*z),value=volume[index],k=4*(x+nx*(ny-1-y));if(Number.isNaN(value)){im.data[k]=6;im.data[k+1]=17;im.data[k+2]=26;}else{const v=value/maximum,a=Math.min(1,Math.abs(v));im.data[k]=v>=0?255:255*(1-a);im.data[k+1]=255*(1-a);im.data[k+2]=v<=0?255:255*(1-a);}im.data[k+3]=255;}ctx.putImageData(im,0,0);$('sliceValue').textContent=z;window.drawCount++;}
function showPlan(p){const count=key=>p.subjects.filter(s=>s.model===key).length;const first=p.subjects[0];$('plan').textContent=`${p.storage}; ${p.workers} workers; ${(p.reserved_numeric_bytes/1048576).toFixed(1)} MiB numerical reservation. Retention: ${p.retention}. Outputs: ${first.uncertainty}. Models: ${count('Reuse')} reuse, ${count('EstimateNoise')} estimate noise, ${count('Prepare')} prepare. ${p.reason}. Metadata and backend scratch are outside this budget.`;}
function updateVariance(){const required=$('group').value==='meta';$('variance').disabled=required;if(required)$('variance').checked=true;}
function setBusy(busy){$('run').disabled=$('preview').disabled=false;$('cancel').disabled=!busy;}
function render(data,command,start,ownGeneration){if(ownGeneration!==generation)return;showPlan(data.plan);if(command.action==='inspect'){window.lastInspection=data;$('status').textContent='Plan inspected; no voxel values read.';return;}result=data;volume=new Float64Array(data.shape.reduce((a,b)=>a*b,1)).fill(NaN);data.support.forEach((p,i)=>volume[p]=data.effects[i]);maximum=data.effects.reduce((m,v)=>Math.max(m,Math.abs(v)),0)||1;$('slice').max=data.shape[2]-1;draw();$('scale').textContent=`Blue ${(-maximum).toFixed(3)} · white 0 · red ${maximum.toFixed(3)} signal units`;$('reuse').textContent=`${data.source_reads} reads / ${data.model_preparations} prepares`;requestAnimationFrame(()=>requestAnimationFrame(()=>{if(ownGeneration!==generation)return;window.lastResult=data;window.lastRender={sequence:++window.renderSequence,command,request_start_ms:start,render_opportunity_ms:performance.now()-start,server_ms:data.server_ms};$('latency').textContent=(window.lastRender.render_opportunity_ms/1000).toFixed(3)+' s';$('status').textContent='Map ready.';}));}
async function poll(requestId,command,start,ownGeneration){try{const response=await fetch('/api/status?request_id='+encodeURIComponent(requestId));const data=await response.json();if(ownGeneration!==generation||requestId!==currentRequest)return;if(data.status==='queued'||data.status==='running'){$('status').textContent=data.status==='queued'?'Queued; latest request will run next.':`Running${data.completed_subjects!==undefined?` · ${data.completed_subjects}/${data.total_subjects} subjects`:''}…`;pollTimer=setTimeout(()=>poll(requestId,command,start,ownGeneration),0);return;}currentRequest=null;setBusy(false);if(data.status==='completed')render(data.response,command,start,ownGeneration);else {$('error').textContent=data.error||'Analysis cancelled.';const retained=result?' Previous map retained.':'';$('status').textContent=data.status==='cancelled'?'Analysis cancelled;'+retained:'Analysis failed;'+retained;}}catch(e){if(ownGeneration===generation&&requestId===currentRequest){currentRequest=null;setBusy(false);$('error').textContent=e.message;$('status').textContent='Analysis failed;'+(result?' Previous map retained.':'');}}}
function submit(action){const ownGeneration=++generation;currentRequest=null;updateVariance();$('error').textContent='';const start=performance.now();window.requestStart=start;const command={action,phase:+$('phase').value,contrast:$('contrast').value,noise:$('noise').value,group:$('group').value,mode:$('mode').value,variance:$('variance').checked};setBusy(true);$('status').textContent='Submitting latest request…';submission=submission.catch(()=>{}).then(async()=>{const response=await fetch('/api',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(command)});const data=await response.json();if(data.error)throw Error(data.error);if(ownGeneration!==generation)return;currentRequest=data.request_id;$('status').textContent=data.status==='queued'?'Queued; latest request will run next.':'Running analysis…';clearTimeout(pollTimer);poll(data.request_id,command,start,ownGeneration);}).catch(e=>{if(ownGeneration===generation){$('error').textContent=e.message;$('status').textContent='Analysis failed.';setBusy(false);}});return submission;}
async function cancel(){const requestId=currentRequest,ownGeneration=generation;if(!requestId)return;await fetch('/api/cancel',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({request_id:requestId})});if(ownGeneration!==generation||requestId!==currentRequest)return;$('status').textContent='Cancellation requested; waiting for current subject boundary…';$('cancel').disabled=true;}
$('run').onclick=()=>submit('run');$('preview').onclick=()=>submit('inspect');$('cancel').onclick=cancel;$('slice').oninput=draw;$('group').onchange=updateVariance;updateVariance();
</script></html>'''

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def generate(directory, shape=(48,48,32), frames_per_run=120, subjects=25):
    import numpy as np
    directory.mkdir(parents=True,exist_ok=False)
    shape=tuple(shape)
    if len(shape)!=3 or any(d<=0 for d in shape) or frames_per_run<=2 or subjects<=0:raise ValueError('shape dimensions and subject/frame counts must be positive')
    v=int(np.prod(shape));runs=[frames_per_run,frames_per_run];n=sum(runs)
    pos=np.arange(v);x=pos%shape[0];y=pos//shape[0]%shape[1];z=pos//(shape[0]*shape[1])
    sx,sy,sz=shape[0]/48,shape[1]/48,shape[2]/32
    mask=((x-(23.5*sx))/(20*sx))**2+((y-(23.5*sy))/(20*sy))**2+((z-(15.5*sz))/(13*sz))**2<1
    support=np.flatnonzero(mask);(directory/'mask.json').write_text(json.dumps(support.tolist()))
    blob=np.exp(-((x-25*sx)**2+(y-22*sy)**2+2*(z-16*sz)**2)/(70*sx*sx))
    design=np.zeros((n,6))
    for t in range(n):
        run=t//frames_per_run;local=t%frames_per_run;design[t,run]=1;design[t,2]=np.sin(local*.07);design[t,3]=np.cos(local*.11);design[t,4+run]=local/(frames_per_run-1)-.5
    np.savetxt(directory/'design.csv',design,delimiter=',',fmt='%.17g')
    files=[]
    for subject in range(subjects):
        import shutil
        # Leave headroom for a worst-case uncompressed subject and other local work.
        if shutil.disk_usage(directory).free < n*v*4 + 1024**3:
            raise OSError('insufficient free space for next synthetic subject; partial fixture is regenerable')
        rng=np.random.default_rng(84521+subject);noise=rng.normal(size=(n,v))
        for run,phi in enumerate([[.4,-.12],[.2,.05]]):
            for t in range(run*frames_per_run,(run+1)*frames_per_run):
                for lag,p in enumerate(phi,1):
                    if t-run*frames_per_run>=lag:noise[t]+=p*noise[t-lag]
        beta=np.zeros((6,v));beta[0]=1000;beta[1]=1050;beta[2]=(1+subject*.02)*blob;beta[3]=.2;beta[4]=.3;beta[5]=-.2
        values=(design@beta+noise).astype('<f4');values[:,~mask]=0
        h=bytearray(352);struct.pack_into('<i',h,0,348);struct.pack_into('<8h',h,40,4,*shape,n,1,1,1);struct.pack_into('<2h',h,70,16,32);struct.pack_into('<8f',h,76,1,3,3,3,1,1,1,1);struct.pack_into('<3f',h,108,352,1,0);h[123]=10;struct.pack_into('<h',h,254,1);struct.pack_into('<12f',h,280,3,0,0,-72,0,3,0,-72,0,0,3,-48);h[344:348]=b'n+1\0'
        path=directory/f'sub-{subject:02d}.nii.gz'
        with path.open('wb') as dst:
            with gzip.GzipFile(filename='',mode='wb',fileobj=dst,compresslevel=1,mtime=0) as gz:gz.write(h);gz.write(values.tobytes())
        files.append({'id':f'sub-{subject:02d}','file':path.name,'sha256':digest(path),'bytes':path.stat().st_size})
    manifest={'shape':shape,'run_lengths':runs,'masked_features':len(support),'subjects':subjects,'seed_base':84521,'numpy':np.__version__,'mask_sha256':digest(directory/'mask.json'),'generator_sha256':digest(pathlib.Path(__file__)),'noise':[[.4,-.12],[.2,.05]],'files':files,'provenance':'Generated synthetic ellipsoid mask, two runs, nuisance and run intercepts; not research data or MNI registration evidence'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'features':len(support),'bytes':sum(f['bytes'] for f in files)}))

class JobController:
    """The only owner of backend stdin/stdout; HTTP threads only touch job state."""
    def __init__(self, binary, directory, receipts):
        self.receipts=receipts;self.receipts.mkdir(parents=True,exist_ok=True)
        self.cancel_directory=pathlib.Path(tempfile.mkdtemp(prefix='analysis-viewer-cancel-'))
        self.backend=subprocess.Popen([str(binary),str(directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'))
        self.lock=threading.RLock();self.wake=threading.Condition(self.lock);self.jobs={};self.history=collections.deque();self.history_limit=8;self.active=None;self.queued=None;self.closed=False;self.sequence=0
        self.thread=threading.Thread(target=self._run,name='analysis-backend',daemon=True);self.thread.start()

    def submit(self, command):
        request_id=uuid.uuid4().hex;cancel_file=self.cancel_directory/f'{request_id}.cancel'
        job={'request_id':request_id,'command':command,'cancel_file':cancel_file,'status':'queued','completed_subjects':0,'total_subjects':None}
        with self.wake:
            if self.closed:raise RuntimeError('analysis server is shutting down')
            if self.queued:self._finish(self.queued,'superseded',error='superseded by a newer request')
            if self.active:self._signal_cancel(self.active)
            self.jobs[request_id]=job;self.queued=job;self._prune();self.wake.notify()
        return {'request_id':request_id,'status':'queued' if self.active else 'running'}

    def status(self, request_id, wait=False, timeout=0.1):
        """Optionally wait for a terminal notification; ordinary callers snapshot immediately."""
        with self.wake:
            job=self.jobs.get(request_id)
            if not job:return None
            deadline=time.monotonic()+timeout
            while wait and job['status'] in {'queued','running'}:
                remaining=deadline-time.monotonic()
                if remaining<=0:break
                self.wake.wait(remaining)
            return {key:value for key,value in job.items() if key not in {'command','cancel_file'}}

    def cancel(self, request_id):
        with self.wake:
            job=self.jobs.get(request_id)
            if not job:return None
            if job is self.queued:
                self.queued=None;self._finish(job,'cancelled',error='cancelled before execution')
            elif job is self.active:self._signal_cancel(job)
            return self.status(request_id)

    def _signal_cancel(self, job):
        if job['status'] in {'completed','failed','cancelled','superseded'}:return
        job['cancel_requested']=True;job['cancel_file'].touch(exist_ok=True)

    def _finish(self, job, status, response=None, error=None):
        job['status']=status
        if response is not None:job['response']=response
        if error is not None:job['error']=str(error)
        self.sequence+=1
        receipt={'request_id':job['request_id'],'command':job['command'],'status':status,'response':response,'error':job.get('error'),'cancel_requested':job.get('cancel_requested',False)}
        (self.receipts/f'{self.sequence:02d}-{job["command"].get("action","unknown")}.json').write_text(json.dumps(receipt)+'\n')
        try:job['cancel_file'].unlink()
        except FileNotFoundError:pass
        self.history.append(job['request_id']);self._prune()
        self.wake.notify_all()

    def _prune(self):
        while len(self.history)>self.history_limit:
            request_id=self.history[0];job=self.jobs.get(request_id)
            if job is self.active or job is self.queued:return
            self.history.popleft();self.jobs.pop(request_id,None)

    def _run(self):
        while True:
            with self.wake:
                while not self.closed and self.queued is None:self.wake.wait()
                if self.closed:return
                job=self.queued;self.queued=None;self.active=job;job['status']='running'
            started=time.perf_counter()
            try:
                command=dict(job['command'],request_id=job['request_id'],cancel_file=str(job['cancel_file']))
                self.backend.stdin.write(json.dumps(command)+'\n');self.backend.stdin.flush()
                while True:
                    line=self.backend.stdout.readline()
                    if not line:raise RuntimeError('analysis process exited')
                    message=json.loads(line)
                    kind=message.get('type')
                    if message.get('request_id') is not None and message['request_id'] != job['request_id']:
                        raise RuntimeError('backend request_id mismatch')
                    if kind=='progress':
                        with self.lock:
                            job['completed_subjects']=message.get('completed_subjects',0);job['total_subjects']=message.get('total_subjects');job['phase']=message.get('phase')
                        continue
                    if kind=='result':message=message.get('result',message.get('response',message))
                    if kind=='error' or 'error' in message:raise RuntimeError(message.get('error','analysis failed'))
                    with self.lock:self._finish(job,'cancelled' if job.get('cancel_requested') else 'completed',response=message)
                    break
            except Exception as error:
                with self.lock:self._finish(job,'cancelled' if job.get('cancel_requested') else 'failed',error=error)
            finally:
                with self.lock:
                    job['backend_ms']=(time.perf_counter()-started)*1000;self.active=None;self._prune();self.wake.notify()

    def close(self):
        with self.wake:
            self.closed=True
            if self.active:self._signal_cancel(self.active)
            if self.queued:self._finish(self.queued,'cancelled',error='server shutdown');self.queued=None
            self.wake.notify_all()
        self.thread.join(timeout=5)
        if self.backend.poll() is None:
            self.backend.terminate()
            try:self.backend.wait(timeout=5)
            except subprocess.TimeoutExpired:self.backend.kill();self.backend.wait()
        self.thread.join(timeout=1)
        if self.backend.stdin:self.backend.stdin.close()
        if self.backend.stdout:self.backend.stdout.close()
        try:self.cancel_directory.rmdir()
        except OSError:pass

def serve(args):
    controller=JobController(args.binary,args.directory,args.receipts)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,body,status=200,mime='application/json'):
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def read_json(self):
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('invalid request size')
            return json.loads(self.rfile.read(length))
        def do_GET(self):
            if self.path=='/':self.send(HTML.encode(),mime='text/html; charset=utf-8')
            elif self.path=='/health':self.send(b'{"ready":true}')
            elif self.path.startswith('/api/status?'):
                from urllib.parse import parse_qs,urlparse
                request_id=parse_qs(urlparse(self.path).query).get('request_id',[''])[0];body=controller.status(request_id,wait=True)
                self.send(json.dumps(body or {'error':'unknown request'}).encode(),200 if body else 404)
            else:self.send(b'{}',404)
        def do_POST(self):
            try:
                command=self.read_json()
                if self.path=='/api':self.send(json.dumps(controller.submit(command)).encode(),202)
                elif self.path=='/api/cancel':
                    body=controller.cancel(command.get('request_id'));self.send(json.dumps(body or {'error':'unknown request'}).encode(),200 if body else 404)
                else:self.send(b'{}',404)
            except Exception as error:self.send(json.dumps({'error':str(error)}).encode(),500)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    args.info.write_text(json.dumps({'server_pid':os.getpid(),'backend_pid':controller.backend.pid,'port':server.server_port,'owner':'analysis-plan qualification','binary':str(args.binary)})+'\n')
    def stop(*_):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();controller.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    gen=sub.add_parser('generate');gen.add_argument('directory',type=pathlib.Path);gen.add_argument('--shape',type=int,nargs=3,metavar=('NX','NY','NZ'),default=(48,48,32));gen.add_argument('--frames-per-run',type=int,default=120);gen.add_argument('--subjects',type=int,default=25)
    web=sub.add_parser('serve');web.add_argument('directory',type=pathlib.Path);web.add_argument('--binary',type=pathlib.Path,required=True);web.add_argument('--port',type=int,default=0);web.add_argument('--info',type=pathlib.Path,required=True);web.add_argument('--receipts',type=pathlib.Path,required=True)
    args=p.parse_args()
    if args.action=='generate':generate(args.directory,args.shape,args.frames_per_run,args.subjects)
    else:serve(args)
