#!/usr/bin/env python3
"""Deterministic controller tests; no browser or Rust binary is required."""
import importlib.util, json, pathlib, tempfile, threading, time, unittest

HERE=pathlib.Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('analysis_viewer',HERE/'analysis_viewer.py')
VIEWER=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(VIEWER)

FAKE='''#!/usr/bin/env python3
import json, pathlib, sys, time
for line in sys.stdin:
  command=json.loads(line); request_id=command['request_id']; cancel=pathlib.Path(command['cancel_file'])
  if command.get('kind') == 'failure':
    print(json.dumps({'type':'error','request_id':request_id,'error':'intentional failure'}),flush=True); continue
  if command.get('kind') == 'mismatch':
    print(json.dumps({'type':'result','request_id':'wrong-request','result':{'plan':{'subjects':[]}}}),flush=True); continue
  steps=20 if command.get('kind') == 'slow' else 2
  for i in range(steps):
    print(json.dumps({'type':'progress','request_id':request_id,'phase':'fitting','completed_subjects':i,'total_subjects':steps}),flush=True)
    time.sleep(.01)
    if cancel.exists():
      print(json.dumps({'type':'error','request_id':request_id,'error':'cancelled'}),flush=True); break
  else:
    print(json.dumps({'type':'result','request_id':request_id,'result':{'token':command.get('token'),'plan':{'subjects':[]}}}),flush=True)
'''

class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();root=pathlib.Path(self.temp.name)
        self.fake=root/'fake.py';self.fake.write_text(FAKE);self.fake.chmod(0o755)
        self.controller=VIEWER.JobController(self.fake,root,root/'receipts')

    def tearDown(self):
        self.controller.close();self.temp.cleanup()

    def wait(self, request_id, timeout=3):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            state=self.controller.status(request_id)
            if state['status'] not in {'queued','running'}:return state
            time.sleep(.01)
        self.fail(f'timeout: {self.controller.status(request_id)}')

    def test_progress_cancel_and_recovery(self):
        slow=self.controller.submit({'action':'run','kind':'slow','token':'old'})['request_id']
        end=time.monotonic()+1
        while time.monotonic()<end and not self.controller.status(slow).get('phase'):time.sleep(.01)
        self.assertEqual(self.controller.status(slow)['phase'],'fitting')
        self.controller.cancel(slow)
        self.assertEqual(self.wait(slow)['status'],'cancelled')
        fresh=self.controller.submit({'action':'run','token':'fresh'})['request_id']
        done=self.wait(fresh);self.assertEqual(done['status'],'completed');self.assertEqual(done['response']['token'],'fresh')

    def test_latest_queued_replaces_prior_queue(self):
        active=self.controller.submit({'action':'run','kind':'slow','token':'active'})['request_id'];time.sleep(.03)
        old_queue=self.controller.submit({'action':'run','token':'discard'})['request_id']
        latest=self.controller.submit({'action':'run','token':'latest'})['request_id']
        self.assertEqual(self.wait(old_queue)['status'],'superseded')
        self.assertEqual(self.wait(active)['status'],'cancelled')
        done=self.wait(latest);self.assertEqual(done['status'],'completed');self.assertEqual(done['response']['token'],'latest')

    def test_failure_does_not_poison_session(self):
        failed=self.controller.submit({'action':'run','kind':'failure'})['request_id']
        self.assertEqual(self.wait(failed)['status'],'failed')
        next_id=self.controller.submit({'action':'run','token':'after-error'})['request_id']
        self.assertEqual(self.wait(next_id)['response']['token'],'after-error')

    def test_mismatched_backend_request_id_is_rejected(self):
        mismatched=self.controller.submit({'action':'run','kind':'mismatch'})['request_id']
        self.assertEqual(self.wait(mismatched)['status'],'failed')
        fresh=self.controller.submit({'action':'run','token':'after-mismatch'})['request_id']
        self.assertEqual(self.wait(fresh)['response']['token'],'after-mismatch')

    def test_long_poll_wakes_on_terminal_notification(self):
        request_id=self.controller.submit({'action':'run','token':'wake'})['request_id']
        started=time.monotonic();state=self.controller.status(request_id,wait=True,timeout=1)
        self.assertEqual(state['status'],'completed')
        self.assertLess(time.monotonic()-started,.5)

    def test_concurrent_submissions_never_run_in_parallel(self):
        gate=threading.Barrier(7);ids=[];lock=threading.Lock()
        def submit(n):
            gate.wait();request_id=self.controller.submit({'action':'run','kind':'slow','token':n})['request_id']
            with lock:ids.append(request_id)
        threads=[threading.Thread(target=submit,args=(n,)) for n in range(6)]
        for thread in threads:thread.start()
        gate.wait()
        for thread in threads:thread.join()
        states=[self.wait(request_id) for request_id in ids]
        self.assertEqual(sum(s['status']=='completed' for s in states),1)
        self.assertTrue(all(s['status'] in {'completed','cancelled','superseded'} for s in states))

    def test_html_rejects_stale_request_completion(self):
        # The guard is intentionally adjacent to polling: old completions must not repaint the map.
        self.assertIn('if(ownGeneration!==generation||requestId!==currentRequest)return',VIEWER.HTML)

if __name__=='__main__':unittest.main()
