"""Offline: per plan slot of vitest/testem, sanitizer actions under 'all' vs 'non_additive' and the class:
  same_as_all   : no effective kept-additive file -> refined tree == 'all' tree
  same_as_old   : every surviving test-path edit is kept -> refined tree == old-grader tree
  mixed         : some surviving edits kept, others restored/deleted -> a tree neither grader produced
Also joins the v1 audit: recorded / all outcome / flip."""
import importlib.util, json, sys, glob, os
from pathlib import Path
S = Path('/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth')
spec = importlib.util.spec_from_file_location('g2', '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/verifier/grader.py')
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
plan = json.load(open(S / 'repair/plan.json'))
runs = sys.argv[1:] or ['official1-vitest', 'official1-testem']

def v1_audit(run, tid):
    p = S / f'runs/{run}/audit/grading_repair/{tid}.json'
    return json.load(open(p)) if p.exists() else None

def v1_slot(audit, slot):
    """(recorded, all_outcome, flipped) from the v1 audit (two formats: vitest, testem)."""
    if audit is None: return (None, None, None)
    for r in audit.get('regraded_slots', []):
        if r['slot'] == slot:
            rec = r['recorded']['status'] if isinstance(r.get('recorded'), dict) else r.get('recorded')
            new = r['new']['status'] if isinstance(r.get('new'), dict) else r.get('new')
            flips = [f if isinstance(f, str) else f.get('slot') for f in audit.get('flips', [])]
            return (rec, new, slot in flips)
    return (audit.get('statuses_before', {}).get(slot), None, False)

result = []
for t in plan['tasks']:
    if t['run'] not in runs: continue
    audit = v1_audit(t['run'], t['task_id'])
    test_reset = set(t['test_patch_paths'])
    for s in t['scheduled'] + t['listed']:
        text = Path(s['model_patch']).read_bytes().decode('utf-8', 'surrogateescape')
        hits = [p for p in G.touched_paths(text) if any(G.glob_match(g, p) for g in t['globs'])]
        add = G.additive_paths(text)
        surv = [p for p in hits if p not in test_reset]
        kept = [p for p in surv if p in add]
        other = [p for p in surv if p not in add]
        cls = 'same_as_all' if not kept else ('same_as_old' if not other else 'mixed')
        rec, allo, flipped = v1_slot(audit, s['slot'])
        result.append(dict(run=t['run'], task=t['task_id'], tstatus=t['status'], slot=s['slot'], model=s['model'],
                           status=s['status'], sched=s in t['scheduled'], cls=cls, kept=kept, other=other,
                           other_kinds={p: (s.get('surviving_kinds') or {}).get(p) for p in other},
                           v1_recorded=rec, v1_all=allo, v1_flip=flipped))
json.dump(result, open('/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/repair/non-additive-20261020/classify2.json', 'w'), indent=1)
from collections import Counter
print(Counter((r['run'], r['cls']) for r in result))
print('flipped slots:')
for r in result:
    if r['v1_flip']:
        print(f"  {r['run'][10:]:8s} {r['task'][9:]:28s} {r['slot']:8s} rec {r['v1_recorded']:5s} all {r['v1_all']:5s} cls {r['cls']:12s} kept {[p.split('/')[-1] for p in r['kept']]} other {[(p.split('/')[-1], k) for p, k in r['other_kinds'].items()]}")
print('mixed, not flipped:')
for r in result:
    if r['cls'] == 'mixed' and not r['v1_flip']:
        print(f"  {r['run'][10:]:8s} {r['tstatus']:8s} {r['task'][9:]:28s} {r['slot']:8s} {r['status']:5s} sched {r['sched']!s:5s} v1all {r['v1_all']!s:5s} kept {[p.split('/')[-1] for p in r['kept']]} other {[(p.split('/')[-1], k) for p, k in r['other_kinds'].items()]}")
