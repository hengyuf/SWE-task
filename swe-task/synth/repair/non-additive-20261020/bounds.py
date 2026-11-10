"""Offline bounds: which slots' refined outcome is already determined, and best/worst case per task."""
import json, re
from pathlib import Path
from collections import defaultdict
S = Path('/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth')
plan = json.load(open(S / 'repair/plan.json'))
cls = {(r['task'], r['slot']): r for r in json.load(open('/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/repair/non-additive-20261020/classify2.json'))}

def md_status(scored, slot):
    p = Path(scored) / 'rollouts' / f'{slot}.md'
    if not p.exists(): return None
    m = re.match(r'^# (\S+) — (\S+)$', p.open(errors='replace').readline().strip())
    return m.group(2) if m else None

def in_range(s, k): return s is not None and k is not None and not (s > 0.2 or k < 0.6)

out = {}
for t in plan['tasks']:
    if t['run'] not in ('official1-vitest', 'official1-testem'): continue
    scored = Path(t['scored_dir'])
    audit = json.load(open(scored / 'teacher_audit.json')) if (scored / 'teacher_audit.json').exists() else {}
    slots = sorted(p.stem for p in (scored / 'rollouts').glob('*.md') if re.fullmatch(r'(sc14|kimi)-r\d+', p.stem))
    rows = []
    for slot in slots:
        st = md_status(scored, slot)
        model = slot.split('-')[0]
        c = cls.get((t['task_id'], slot))
        cheated = (audit.get(slot) or {}).get('cheated') if isinstance(audit.get(slot), dict) else None
        if c is None: known, why = st, 'unaffected'
        elif c['v1_flip']: known, why = None, 'flip(regrade)'
        elif c['cls'] == 'same_as_old': known, why = st, 'same_as_old'
        elif c['cls'] == 'same_as_all': known, why = (c['v1_all'] or st), 'same_as_all'
        else: known, why = None, 'mixed'
        rows.append(dict(slot=slot, model=model, status=st, cheated=cheated, known=known, why=why))
    def rate(fill_s, fill_k):
        s = [r for r in rows if r['model'] == 'sc14' and r['status'] in ('pass', 'fail')]
        k = [r for r in rows if r['model'] == 'kimi' and r['status'] in ('pass', 'fail')]
        sp = sum((r['known'] if r['known'] else fill_s) == 'pass' for r in s)
        kp = sum((r['known'] if r['known'] else fill_k) == 'pass' and r['cheated'] is not True for r in k)
        return (sp / len(s) if s else None, kp / len(k) if k else None)
    best, worst = rate('fail', 'pass'), rate('pass', 'fail')
    unknown = [r['slot'] + ':' + r['status'] + ':' + r['why'] for r in rows if r['known'] is None]
    print(f"{t['run'][10:]:7s} {t['status']:8s} {t['task_id'][9:]:26s} best {best} in={in_range(*best)} worst {worst} in={in_range(*worst)} unknown {len(unknown)}")
    out[t['task_id']] = dict(rows=rows, best=best, worst=worst)
json.dump(out, open('/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/repair/non-additive-20261020/bounds.json', 'w'), indent=1)
