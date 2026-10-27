"""Generate a compact measured-performance table from recorded processes."""
import json,statistics
from pathlib import Path
base=Path(__file__).resolve().parents[1]
r=json.loads((base/'evidence/gls-baseline/results.json').read_text())
f=json.loads((base/'evidence/gls-final/results.json').read_text())
def med(x,keys):return statistics.median(sum(t[k] for k in keys) for t in x['warm'])
def find(rs,key,engine):return next(x for x in rs if x['key']==key and x['engine']==engine)
lines=['# AR-GLS measured results','', 'All times are warm medians over three repetitions. Compute includes whitening, fitting and a t contrast; synthetic generation is excluded. Setup is measured separately.','', '| T | P | Features | AR | Rust compute ms | R compute ms | Ratio R/Rust | Rust/R RSS MiB |','|---:|---:|---:|---:|---:|---:|---:|---:|']
for x in f:
 if x['workers']!=1 or x['block']!=1000 or x['engine']!='rust':continue
 y=find(r,x['key'],'r');a=med(x,['whiten','fit','contrast']);b=med(y,['whiten','fit','contrast'])
 lines.append(f"| {x['n']} | {x['p']} | {x['features']} | {x['order']} | {1000*a:.2f} | {1000*b:.2f} | {b/a:.2f} | {x['peak_rss_bytes']/2**20:.1f} / {y['peak_rss_bytes']/2**20:.1f} |")
lines+=['','## Block and worker sweep','','Stage timers sum time across workers; they are not parallel wall time. Wall time includes generated input and thread launch.','', '| Block | Workers | Wall median ms | Wall min–max ms | RSS MiB |','|---:|---:|---:|---:|---:|']
for x in f:
 if x['n']!=1200 or x['p']!=32 or x['features']!=50000 or x['order']!=4 or x['engine']!='rust':continue
 t=[y['wall']*1000 for y in x['warm']]
 lines.append(f"| {x['block']} | {x['workers']} | {statistics.median(t):.1f} | {min(t):.1f}–{max(t):.1f} | {x['peak_rss_bytes']/2**20:.1f} |")
lines+=['','## Optimization comparison','','Same benchmark harness; baseline vs contiguous residual-variance accumulation. Single-worker, 50,000 features, AR(4).','', '| T | P | Baseline fit ms | Final fit ms | Fit ratio |','|---:|---:|---:|---:|---:|']
for x in f:
 if x['workers']!=1 or x['block']!=1000 or x['features']!=50000 or x['order']!=4 or x['engine']!='rust':continue
 y=find(r,x['key'],'rust');a=med(y,['fit']);b=med(x,['fit']);lines.append(f"| {x['n']} | {x['p']} | {1000*a:.2f} | {1000*b:.2f} | {a/b:.2f} |")
(base/'docs/GLS_RESULTS.md').write_text('\n'.join(lines)+'\n')
lines=['','## Largest-case phases','','1,200 timepoints × 32 regressors × 50,000 features, AR(4), one worker, block 1,000. Setup is a single cold observation per process; block phases are warm medians. Noise estimation uses 32 representative features in both implementations.','', '| Phase | Rust ms | R ms |','|---|---:|---:|']
x=find(f,'1200-32-50000-ar4-b1000-w1','rust');y=find(r,x['key'],'r')
for k in x['setup']:
 lines.append(f"| Setup: {k} | {x['setup'][k]*1000:.3f} | {y['setup'][k]*1000:.3f} |")
for k in ['generate','whiten','fit','contrast','wall']:
 lines.append(f"| Warm: {k} | {med(x,[k])*1000:.3f} | {med(y,[k])*1000:.3f} |")
lines.append(f"\nFirst block pass wall time: Rust {x['cold']['wall']*1000:.1f} ms; R {y['cold']['wall']*1000:.1f} ms. R elapsed timer resolution is about 1 ms; a zero setup entry means below timer resolution.")
m=json.loads((base/'evidence/gls-memory/results.json').read_text())
lines+=['','## Fixed-block memory growth','','Same 1,200 × 32 AR(4) model, block 1,000, one worker; fresh process per row. RSS includes setup, warm-up, repetitions and streamed output validation.','', '| Features | Wall ms | RSS MiB |','|---:|---:|---:|']
for x in m:lines.append(f"| {x['features']} | {med(x,['wall'])*1000:.1f} | {x['peak_rss_bytes']/2**20:.1f} |")
with (base/'docs/GLS_RESULTS.md').open('a') as out:out.write('\n'.join(lines)+'\n')
