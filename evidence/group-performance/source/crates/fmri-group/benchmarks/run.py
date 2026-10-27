"""Run a single benchmark subprocess, preserving exit status and child max RSS."""
import hashlib, json, os, pathlib, resource, subprocess, sys, time
label, *command = sys.argv[1:]
root = pathlib.Path('evidence/group-tranche1')
env = dict(os.environ, LC_ALL='C', FMRIGDS_THREADS='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
start = time.monotonic()
with (root / f'{label}.log').open('w') as out, (root / f'{label}-stderr.log').open('w') as err:
    code = subprocess.run(command, env=env, stdout=out, stderr=err).returncode
usage = resource.getrusage(resource.RUSAGE_CHILDREN)
(root / f'{label}-resource.json').write_text(json.dumps({
    'command':command, 'command_file_sha256':{x:hashlib.sha256(pathlib.Path(x).read_bytes()).hexdigest() for x in command if pathlib.Path(x).is_file()}, 'exit':code, 'elapsed_seconds':time.monotonic()-start,
    'max_rss_bytes': usage.ru_maxrss * (1 if sys.platform=='darwin' else 1024),
    'scope':'whole benchmark process including resident inputs, oracle payloads and retained outputs',
    'platform':sys.platform,'thread_environment':{k:env[k] for k in ['FMRIGDS_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']}
},indent=2)+'\n')
print(label, 'exit',code, 'RSS',usage.ru_maxrss)
sys.exit(code)
