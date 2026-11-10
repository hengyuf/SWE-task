'use strict';
/* data/explorer.json (CONTRACT "v1.7") built from task / run documents by the contract's rules, for the frontend tests:
frontend.test.js makes its Explorer fixture from the narwhals / bandit / batch fixtures with it, and frontend_real.test.js
checks the collector's real explorer.json against the real task and run files with it (a third implementation of the
rules, next to collect.py's and validate_contract.py's).

    const {explorerOf} = require('./explorer_fixture.js');
    explorerOf([{source: 'hengyuf', detail: 'task/<task>.json', doc}, {source: 'baiheh', detail: 'run/...', doc}], stamp,
               [{source: 'hengyuf', name: 'hengyuf', label: '...', audited: true}, ...])
v1.8: every official run, an older one as {source: 'hengyuf', detail: 'run/hengyuf/<run>.json', doc, latest: false}
(linked with run=); runs[].latest is true for an official entry without it, null for a batch run's. legacy: true omits
runs[].latest (what a v1.7 collector wrote).
*/
const TASK_COLUMNS = ['run', 'node', 'task_id', 'round', 'branch', 'kind', 'title', 'decision', 'accepted', 'withdrawn',
                      'in_batch', 'similarity', 'student_rate', 'teacher_clean_rate', 'teacher_rate', 'valid', 'audit',
                      'reproduced_yes', 'reproduced_partial', 'f2p', 'p2p', 'scored', 'rates_differ'];
const TRAJECTORY_COLUMNS = ['task', 'slot', 'model', 'status', 'reward', 'f2p', 'p2p', 'tool_calls', 'patch_chars',
                            'agent_error', 'verifier_error', 'clean'];
const VERDICTS = ['strong', 'partial', 'weak', 'none'];
const OFFICIAL = {source: 'hengyuf', name: 'hengyuf', label: 'hengyuf · official runs', audited: true};

const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const finite = (v) => (typeof v === 'number' && Number.isFinite(v) ? v : null);
const loose = (s) => (typeof s === 'string' ? s.split(/\s+/).filter(Boolean).join(' ').toLowerCase() : null);

function explorerOf(entries, generatedAt, sources, legacy) {
  const srcs = sources || [OFFICIAL];
  const audited = new Map(srcs.map((s) => [s.source, s.audited === true]));
  const totals = {};
  const total = (s) => (totals[s] = totals[s] || {runs: 0, tasks: 0, trajectories: 0, accepted: 0, audited: 0, unmatched_patterns: 0,
                                                   rates_differ: 0});
  for (const s of srcs) total(s.source);
  const runs = [], targets = new Map(), tasks = [], trajectories = [];
  for (const {source, detail, doc, latest} of entries) {
    const batch = source !== OFFICIAL.source, older = !batch && latest === false;
    const run = {run: doc.run, source, deepswe_task: doc.task_id, build_root: doc.build_root, max_rounds: doc.max_rounds,
                 detail, link: `#task=${encodeURIComponent(doc.task_id)}${batch || older ? `&run=${encodeURIComponent(doc.run)}` : ''}`};
    if (!legacy) run.latest = batch ? null : !older;
    runs.push(run);
    total(source).runs += 1;
    const info = targets.get(doc.task_id) || {deepswe_task: doc.task_id, runs: [], patterns: [], exact: new Map(), loose: new Map()};
    targets.set(doc.task_id, info);
    info.runs.push(runs.length - 1);
    const fa = isObj(doc.failure_analysis) ? doc.failure_analysis : {};
    for (const p of Array.isArray(fa.patterns) ? fa.patterns : []) {
      const name = isObj(p) && typeof p.name === 'string' ? p.name : null;
      if (info.runs.length > 1 && (name === null || info.exact.has(name))) continue;
      if (name !== null) {
        if (!info.exact.has(name)) info.exact.set(name, info.patterns.length);
        if (!info.loose.has(loose(name))) info.loose.set(loose(name), info.patterns.length);
      }
      info.patterns.push({name, category: isObj(p) && typeof p.category === 'string' ? p.category : null});
    }
    for (const node of doc.nodes || []) {
      const build = isObj(node.build) ? node.build : {};
      if (!build.task_id) continue;
      const score = isObj(node.score) ? node.score : null, probe = isObj(node.probe) ? node.probe : {};
      const fm = node.failure_match;
      let audit;
      if (!(audited.has(source) ? audited.get(source) : !batch) || (isObj(node.audit) && node.audit.status === 'not_audited')) audit = 'not_audited';
      else if (node.audit_eligible !== true) audit = 'not_eligible';
      else audit = isObj(fm) && VERDICTS.includes(fm.verdict) ? fm.verdict : 'not_audited';
      let yes = null, partial = null;
      if (VERDICTS.includes(audit)) {
        const y = new Set(), p = new Set();
        for (const x of Array.isArray(fm.patterns) ? fm.patterns : []) {
          const name = isObj(x) ? x.name : null;
          const k = typeof name === 'string' ? (info.exact.has(name) ? info.exact.get(name) : info.loose.get(loose(name))) : undefined;
          if (k === undefined) total(source).unmatched_patterns += 1;
          else if (x.reproduced === 'yes') y.add(k);
          else if (x.reproduced === 'partial') p.add(k);
        }
        yes = [...y].sort((a, b) => a - b);
        partial = [...p].filter((k) => !y.has(k)).sort((a, b) => a - b);
      }
      const valid = score && Array.isArray(score.valid) && score.valid.length === 2 && score.valid.every((v) => finite(v) !== null)
        ? score.valid.slice() : null;
      const accepted = node.decision === 'accepted' || node.decision === 'accepted_evolve';
      const details = isObj(node.details) ? node.details : {};
      const cheats = new Map((Array.isArray(details.teacher_audit) ? details.teacher_audit : []).filter(isObj).map((e) => [e.slot, e.cheated]));
      // task_builder score()'s rates from the rollout rows; metrics.json's differ when it was re-scored afterwards
      const rows = Array.isArray(probe.rollouts) ? probe.rollouts.filter(isObj) : [];
      const graded = (m) => rows.filter((r) => r.model === m && (r.status === 'pass' || r.status === 'fail'));
      const st = graded('sc14'), te = graded('kimi');
      const rowsRates = [st.length ? st.filter((r) => r.status === 'pass').length / st.length : null,
                         te.length ? te.filter((r) => r.status === 'pass' && cheats.get(r.slot) !== true).length / te.length : null];
      const same = (a, b) => (a === null || b === null ? a === b : Math.abs(a - b) < 1e-9);
      const differ = score !== null && (!same(finite(score.student_rate), rowsRates[0]) || !same(finite(score.teacher_clean_rate), rowsRates[1]) ||
                                        (valid !== null && (valid[0] !== st.length || valid[1] !== te.length)));
      tasks.push([runs.length - 1, node.id, build.task_id, node.round, node.branch, node.kind, node.title, node.decision, accepted,
                  !!node.withdrawn, node.in_batch === true, score ? finite(score.similarity) : null,
                  score ? finite(score.student_rate) : null, score ? finite(score.teacher_clean_rate) : null,
                  score ? finite(isObj(probe.teacher) ? probe.teacher.rate : null) : null, valid, audit, yes, partial,
                  finite(build.f2p), finite(build.p2p), score !== null, differ]);
      const t = total(source);
      t.tasks += 1;
      t.accepted += accepted ? 1 : 0;
      t.audited += VERDICTS.includes(audit) ? 1 : 0;
      t.rates_differ += differ ? 1 : 0;
      for (const r of Array.isArray(probe.rollouts) ? probe.rollouts : []) {
        if (!isObj(r)) continue;
        const cheated = cheats.get(r.slot);
        const clean = r.model === 'kimi' && r.status === 'pass' && typeof cheated === 'boolean' ? !cheated : null;
        trajectories.push([tasks.length - 1, r.slot, r.model, r.status, finite(r.reward), r.f2p, r.p2p, finite(r.tool_calls),
                           finite(r.patch_chars), !!r.agent_error, !!r.verifier_error, clean]);
        t.trajectories += 1;
      }
    }
  }
  return {schema: 1, generated_at: generatedAt, sources: srcs, runs,
          targets: [...targets.values()].sort((a, b) => (a.deepswe_task < b.deepswe_task ? -1 : a.deepswe_task > b.deepswe_task ? 1 : 0))
            .map(({deepswe_task, runs: r, patterns}) => ({deepswe_task, runs: r, patterns})),
          task_columns: TASK_COLUMNS, tasks, trajectory_columns: TRAJECTORY_COLUMNS, trajectories, totals};
}

module.exports = {explorerOf, TASK_COLUMNS, TRAJECTORY_COLUMNS, OFFICIAL};
