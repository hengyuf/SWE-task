'use strict';
/* Regenerates tests/fixtures/*.json: realistic CONTRACT.md data for the frontend tests. Deterministic (seeded PRNG,
fixed clock), no input files. Titles, F2P/P2P counts, failure patterns and rates are modeled on official1-narwhals and
official1-bandit as of 2026-10-17; the narwhals-like tree is pushed to round 3 so that it holds every decision kind,
a live probe (one task with all of its own slots done), a superseded probe (abandoned nodes) and invalid seeds; the
bandit-like tree has a finished round 1 and a round 2 whose tasks are being built.
Collector shapes: narwhals, fleet.json and api.json use the current ones (NODE.probe.partial / own_done,
NODE.focus_pattern, the teacher_audit list, totals.busy_workers / prompt_tps_incl_cache, router.scope, burn windows,
api.scan); bandit keeps the older ones (no partial / focus_pattern, teacher_audit as a JSON-dumped string), so the page
is tested on both.

    node tests/make_fixtures.js
*/
const fs = require('node:fs');
const path = require('node:path');

const OUT = path.join(__dirname, 'fixtures');
const NOW = 1792274400.0;                              // 2026-10-17T15:00:00-07:00
const S = '/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task';

let seed = 20261017;
function rand() {                                      // mulberry32
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
}
const randint = (a, b) => a + Math.floor(rand() * (b - a + 1));
const round = (x, d) => Number(x.toFixed(d));
function iso(t) {
  return new Date((t - 7 * 3600) * 1000).toISOString().replace(/\.\d{3}Z$/, '-07:00');
}

// ------------------------------------------------------------------------------------------------ index.json

const MODELS = {
  kimi: {label: 'Kimi K3', role: 'teacher', source: "Baihe's DeepSWE export (three_way_per_task.csv), 5 rollouts per task"},
  super_vl: {label: 'Nemotron 3.5 Super VL', role: "Baihe's student",
             source: "Baihe's DeepSWE export (three_way_per_task.csv), 5 rollouts per task"},
  sc14: {label: 'Local Super ckpt sc14 (swonlysc-step0014, 131k cap)', role: 'student',
         source: 'hengyuf DeepSWE eval of swonlysc-step0014 (131k per-call cap), 5 rollouts per task'},
};

/** Accuracy card from per-rollout F2P passes (null = unscored/infra); a rollout passes when it passes every F2P. sc14
    rollouts carry slot and status like hengyuf's eval (an infra slot reads "0/0": the verifier never ran). */
function card(model, f2pTotal, p2pTotal, f2ps, extra) {
  const rollouts = f2ps.map((f, i) => (model === 'sc14'
    ? {slot: `sc14-r${i}`, f2p: f === null ? '0/0' : `${f}/${f2pTotal}`, p2p: f === null ? '0/0' : `${p2pTotal}/${p2pTotal}`,
       score: f === null ? null : f === f2pTotal ? 1 : 0, status: f === null ? 'infra' : f === f2pTotal ? 'pass' : 'fail'}
    : {f2p: f === null ? null : `${f}/${f2pTotal}`, p2p: f === null ? null : `${p2pTotal}/${p2pTotal}`,
       score: f === null ? null : f === f2pTotal ? 1 : 0}));
  const scored = f2ps.filter((f) => f !== null);
  const out = {passed: scored.filter((f) => f === f2pTotal).length, total: f2ps.length, rollouts};
  if (model === 'sc14') {
    out.infra = f2ps.length - scored.length;
    out.mean_f2p_ratio = scored.length ? round(scored.reduce((s, f) => s + f / f2pTotal, 0) / scored.length, 4) : null;
  } else {
    out.unscored = f2ps.length - scored.length;
  }
  return Object.assign(out, extra || {});
}

function indexTask(id, i, f2p, p2p, acc, run) {
  const accuracy = {};
  for (const m of ['kimi', 'super_vl', 'sc14']) accuracy[m] = acc[m] ? card(m, f2p, p2p, acc[m]) : null;
  const sc = accuracy.sc14, k = accuracy.kimi;
  return {task_id: id, task_index: i, f2p_total: f2p, accuracy,
          eligible: !!(sc && k && sc.passed <= 1 && k.passed === 5), queue_rank: null, run: run || null};
}

const TASKS = [
  indexTask('abs-module-cache-flags', 0, 20, 3, {kimi: [20, 20, 20, 20, 20], super_vl: [20, 17, 20, 18, 16], sc14: [20, 20, 19, 20, 20]}),
  indexTask('abs-stepped-slices', 1, 14, 212, {kimi: [14, 14, 13, 14, 14], super_vl: [11, 14, 9, 12, 10], sc14: [12, 14, 11, 13, 10]}),
  indexTask('bandit-incremental-cache-control', 10, 24, 412, {kimi: [24, 24, 24, 24, 24], super_vl: [19, 18, 22, 20, 17], sc14: [20, 19, 21, 20, 20]}, 'official1-bandit'),
  indexTask('expr-try-catch-errors', 27, 31, 1180, {kimi: [31, 31, null, 31, 30], sc14: [27, null, 29, null, 25]}),
  indexTask('fastapi-implicit-head-options', 29, 9, 2510, {sc14: [9, 7, 8, 6, 8]}),
  indexTask('fd-deterministic-multi-key-sorting', 30, 38, 911, {kimi: [38, 38, 38, 38, 38], super_vl: [31, 38, 29, 33, 30], sc14: [27, 38, 25, 26, 19]}, 'official2-fd'),
  indexTask('go-git-worktree-merge-conflicts', 34, 47, 3120, {kimi: [47, 47, 47, 47, 47], sc14: [30, 33, 28, 29, 31]}),
  indexTask('happy-dom-abort-pending-body-reads', 37, 12, 640, {kimi: [12, 12, 12, 12, 12], super_vl: [3, 5, 4, 6, 2], sc14: [12, 2, 4, 3, 4]}),
  indexTask('helm-array-merge-strategies', 39, 26, 1455, {kimi: [26, 26, 26, 26, 26], super_vl: [26, 22, 26, 26, 19], sc14: [26, 24, 26, 21, 23]}),
  indexTask('narwhals-rolling-window-suite', 57, 103, 10093, {kimi: [103, 103, 103, 103, 103], super_vl: [99, 103, 97, 100, 98], sc14: [102, 100, 101, 101, 103]}, 'official1-narwhals'),
];
TASKS.filter((t) => t.eligible)
  .sort((a, b) => (b.accuracy.sc14.mean_f2p_ratio || -1) - (a.accuracy.sc14.mean_f2p_ratio || -1))
  .forEach((t, i) => { t.queue_rank = i + 1; });

// -------------------------------------------------------------------------------------------- tree helpers

const STEP = /-r(\d+)b(\d+)$/;

/** Rollout rows for a probe: `s` = [pass, fail, infra] student, `t` = [pass, fail, infra] teacher. */
function rollouts(f2p, p2p, s, t, opts) {
  const o = opts || {};
  const rows = [];
  const add = (model, counts) => {
    let r = 0;
    for (const [status, n] of [['pass', counts[0]], ['fail', counts[1]], ['infra', counts[2]]]) {
      for (let i = 0; i < n; i += 1, r += 1) {
        const fail = status === 'fail';
        const missing = fail ? randint(1, Math.min(6, Math.max(1, f2p - 1))) : 0;
        rows.push({slot: `${model}-r${r}`, model, status, reward: status === 'infra' ? null : status === 'pass' ? 1.0 : 0.0,
                   f2p: status === 'infra' ? null : `${f2p - missing}/${f2p}`,
                   p2p: status === 'infra' ? null : `${p2p - (fail && rand() < 0.25 ? 1 : 0)}/${p2p}`,
                   tool_calls: o.live ? null : randint(model === 'sc14' ? 180 : 110, model === 'sc14' ? 640 : 260),
                   patch_chars: o.live ? null : randint(4000, 26000),
                   agent_error: status === 'infra' && !o.verifier
                     ? 'AgentTimeoutError: agent exceeded 10800 s\nTraceback (most recent call last):\n' +
                       '  File "/app/harness/run_agent.py", line 412, in run_slot\n    result = agent.wait(timeout=budget)\n' +
                       `  File "/app/harness/agent.py", line 188, in wait\n    raise AgentTimeoutError(f"agent exceeded {budget} s")` : '',
                   verifier_error: status === 'infra' && o.verifier ? 'test.patch failed to apply: patch does not apply to ibis/backends/tests/test_window.py' : ''});
      }
    }
  };
  add('sc14', s);
  add('kimi', t);
  return rows;
}

function testNames(prefix, n, cap) {
  const names = [];
  const backends = ['duckdb', 'datafusion', 'polars', 'sqlite', 'pyarrow', 'pandas'];
  for (let i = 0; i < Math.min(n, cap); i += 1) names.push(`${prefix}[${backends[i % backends.length]}-case${i}]`);
  return names;
}

function similarityDetail(score) {
  return JSON.stringify({
    matched_patterns: [
      {synthetic: 'Extending a backend restriction from rolling quantile to rolling median', target: 'Extending a narrow backend exception to a supported operation'},
      {synthetic: 'Skipping its own failing median tests on SQLite', target: 'Making tests conform to the unsupported assumption'},
    ],
    missing_patterns: score >= 6 ? ['Certifying interpolation arithmetic without testing the changed backend'] :
      ['Certifying interpolation arithmetic without testing the changed backend',
       'High-prevalence exception broadening: the synthetic scope error appears in 1/5 slots instead of 4/5'],
    rationale: score >= 6
      ? 'The synthetic task reproduces the core requirement-scope weakness at the root-cause level and with similar prevalence; the interpolation pattern is not exercised.'
      : 'The student fails, but mostly through unrelated numeric slips; the scope-broadening mechanism appears in only one slot.',
    score,
  }, null, 1);
}

/** teacher_audit of the teacher's passing slots: the collector's list ({slot, cheated, categories, summary}), or with
    asString the older JSON-dumped teacher_audit.json. The first `cheated` slots are disqualified. */
function audit(passSlots, cheated, asString) {
  const flag = (i) => i < (cheated || 0);
  if (asString) {
    const out = {};
    passSlots.forEach((slot, i) => {
      out[slot] = {categories: flag(i) ? ['answer_retrieval'] : [], cheated: flag(i),
                   evidence: [`${slot} implemented the windows from the task text and probed DuckDB natively before committing.`],
                   notes: 'No disqualifier met; upstream retrieval was attempted but blocked and never used.'};
    });
    return JSON.stringify(out, null, 1);
  }
  return passSlots.map((slot, i) => (flag(i)
    ? {slot, cheated: true, categories: ['answer_retrieval', 'test_tampering'],
       summary: `${slot} fetched the upstream implementation with pip download, copied its rolling_quantile branch and ` +
                'then relaxed a failing assertion in tests/test_window.py instead of fixing the median dispatch.'}
    : {slot, cheated: false, categories: [],
       summary: `${slot} implemented the windows from the task text, probed DuckDB natively before committing; upstream ` +
                'retrieval was attempted but blocked and never used.'}));
}

const REASON = {
  range: (s, t) => `pass rates out of range (student ${s.toFixed(2)} > 0.2 or clean teacher ${t.toFixed(2)} < 0.6)`,
  sim: (x) => `similarity ${x}/10`,
};

/**
 * One NODE. spec: {title, f2p, p2p, kind of outcome}. Outcomes:
 *   s/t: [pass, fail, infra] -> scored; sim: similarity (in-range rates); accept: bool; invalid: seeds.json reason
 *   cheated: teacher passes disqualified by the audit; fail: build problems (stage ok=false)
 *   live: {sc14: [d, t], kimi: [d, t]} probing (liveInfra: {model: n} of the done slots are infra; ownDone: all of the
 *   node's own slots are done); status: pending build status
 * ctx: {probeDir, patterns: names for NODE.focus_pattern (current shape), auditString: older teacher_audit shape}
 */
function makeNode(id, spec, ctx) {
  const steps = [];
  let rest = id;
  for (let m = rest.match(STEP); m; m = rest.match(STEP)) { steps.unshift([+m[1], +m[2]]); rest = rest.slice(0, m.index); }
  const k = steps.length;
  const parent = k ? id.replace(STEP, '') : null;
  const built = !spec.fail && (!spec.status || spec.status === 'built');
  const n = {
    id, parent, lineage: rest, round: k, branch: k ? steps[k - 1][1] : null, kind: k ? 'evolve' : 'mask',
    title: spec.title,
    summary: spec.summary || `${spec.title}: ${k ? 'an evolved High-IID feature on top of image A aimed at one target failure pattern' : 'restore a masked feature family'} across the ibis SQL and DataFrame backends, with null handling, ordering and backend capability rules spelled out in the instruction.`,
    status: null,
    build: {ok: spec.fail ? false : built ? true : null, attempts: spec.attempts || (spec.fail ? 2 : built ? 1 : spec.status === 'planning' ? 0 : 1),
            f2p: spec.fail ? (spec.failF2p === undefined ? null : spec.failF2p) : built ? spec.f2p : null,
            p2p: spec.fail ? null : built ? spec.p2p : null,
            f2p_tests: built ? testNames(`ibis/backends/tests/test_window.py::test_${id.replace(/-/g, '_')}`, spec.f2p, spec.manyTests ? 60 : 8) : [],
            p2p_tests_sample: built ? testNames('ibis/backends/tests/test_generic.py::test_existing', spec.p2p, 6) : [],
            problems: spec.fail || null,
            task_id: built ? `syn-ibis-${id}-${(0x10000000 + Math.floor(rand() * 0xefffffff)).toString(16)}` : null},
    probe: null, score: null, decision: null, decision_reason: null, children: [],
    details: {instruction_md: null, evolve_plan_md: null, synthetic_failure_analysis_md: null, rubric_md: null, teacher_audit: null},
  };
  // task_builder.evolve aims branch j at patterns[j % len(patterns)] of the target's analysis ("overall" without any).
  if (k && ctx.patterns) n.focus_pattern = ctx.patterns.length ? ctx.patterns[n.branch % ctx.patterns.length] : 'overall';
  if (built) {
    n.details.instruction_md = `# ${spec.title}\n\nImplement ${spec.title.toLowerCase()} for ibis expressions.\n\n` +
      '## Requirements\n\n- Work on a new branch and commit your final changes.\n- Nulls are excluded from every statistic and from the valid-observation count.\n' +
      '- Backends that cannot support an operation must raise `NotImplementedError` naming the backend and the operation; do not broaden this to other operations.\n' +
      (spec.longText ? '\n## Details\n\n' + 'The window frame follows the existing rolling methods (trailing or centered, `min_periods` defaulting to the window size). '.repeat(40) : '');
    n.details.rubric_md = '# Rubric\n\n1. Behavior matches the instruction on every listed backend.\n2. Restrictions are not broadened beyond the instruction.\n3. Tests exercise the changed backend, not a delegated one.\n';
    if (k) n.details.evolve_plan_md = `# Evolve plan\n\nFocus pattern: ${spec.focus || 'Extending a narrow backend exception to a supported operation'}.\n\nNew feature: ${spec.title}. Keep what made the parent work: an explicit, narrow backend restriction next to a supported sibling operation.\n`;
  } else if (k && spec.status !== 'planning') {
    n.details.evolve_plan_md = `# Evolve plan\n\nNew feature: ${spec.title}.\n`;
  }
  if (spec.s) {                                                   // scored
    const live = false;
    const rows = rollouts(spec.f2p, spec.p2p, spec.s, spec.t, {live, verifier: !!spec.verifier});
    const sValid = spec.s[0] + spec.s[1], tValid = spec.t[0] + spec.t[1], cheated = spec.cheated || 0;
    const sRate = sValid ? spec.s[0] / sValid : null, tRate = tValid ? (spec.t[0] - cheated) / tValid : null;
    n.probe = {dir: ctx.probeDir, live, progress: {sc14: [5, 5], kimi: [5, 5]},
               student: {pass: spec.s[0], fail: spec.s[1], infra: spec.s[2], rate: sRate},
               teacher: {pass: spec.t[0], fail: spec.t[1], infra: spec.t[2], clean_rate: tRate, cheated},
               rollouts: rows};
    const inRange = sValid >= 3 && tValid >= 3 && sRate <= 0.2 && tRate >= 0.6;
    const accept = inRange && (spec.sim || 0) >= 6;
    const reason = sValid < 3 || tValid < 3 ? 'inconclusive: fewer than 3 valid rollouts per model'
      : inRange ? REASON.sim(spec.sim || 0) : REASON.range(sRate, tRate);
    n.score = {accept, reason, similarity: inRange ? spec.sim || 0 : null,
               similarity_detail: inRange ? similarityDetail(spec.sim || 0) : null, valid: [sValid, tValid],
               grading_failures: spec.verifier ? rows.filter((r) => r.status === 'infra').map((r) => r.slot) : [],
               answer_retrieval: cheated > 0, target_student_rate: 0.2, target_teacher_rate: 1.0,
               student_rate: sRate, teacher_clean_rate: tRate, cheated};
    n.status = 'scored';
    n.details.teacher_audit = audit(rows.filter((r) => r.model === 'kimi' && r.status === 'pass').map((r) => r.slot), cheated,
                                    !!ctx.auditString);
    if (inRange) {
      n.details.synthetic_failure_analysis_md = `# Failure analysis: ${n.build.task_id}\n\nThe student passed ${spec.s[0]} of ${sValid} graded rollouts; ` +
        `the teacher passed ${spec.t[0]} of ${tValid}. The failing student rollouts reject the supported operation on SQL backends and skip their own failing tests.\n`;
    }
  } else if (spec.live) {
    const rows = [];
    for (const [model, [done]] of Object.entries(spec.live)) {
      const infra = (spec.liveInfra || {})[model] || 0;
      for (let r = 0; r < done; r += 1) {
        if (r >= done - infra) {
          rows.push({slot: `${model}-r${r}`, model, status: 'infra', reward: 0.0, f2p: '0/0', p2p: '0/0', tool_calls: null,
                     patch_chars: null, agent_error: '', verifier_error: 'Verifier did not produce /logs/verifier/reward.json'});
          continue;
        }
        const pass = model === 'kimi' ? rand() < 0.8 : rand() < 0.25;
        rows.push({slot: `${model}-r${r}`, model, status: pass ? 'pass' : 'fail', reward: pass ? 1.0 : 0.0,
                   f2p: `${pass ? spec.f2p : spec.f2p - randint(1, 4)}/${spec.f2p}`, p2p: `${spec.p2p}/${spec.p2p}`,
                   tool_calls: null, patch_chars: null, agent_error: '', verifier_error: ''});
      }
    }
    const count = (m, st) => rows.filter((r) => r.model === m && r.status === st).length;
    const so = {pass: count('sc14', 'pass'), fail: count('sc14', 'fail'), infra: count('sc14', 'infra')};
    const to = {pass: count('kimi', 'pass'), fail: count('kimi', 'fail'), infra: count('kimi', 'infra')};
    n.probe = {dir: ctx.probeDir, live: true, progress: spec.live,
               student: Object.assign({}, so, {rate: so.pass + so.fail ? so.pass / (so.pass + so.fail) : null}),
               teacher: Object.assign({}, to, {rate: to.pass + to.fail ? to.pass / (to.pass + to.fail) : null, clean_rate: null, cheated: null}),
               rollouts: rows};
    if (ctx.patterns) {                                           // current collector: graded-slot partials
      n.probe.partial = {};
      for (const [model, [done, total]] of Object.entries(spec.live)) {
        n.probe.partial[model] = Object.assign({done, total}, model === 'sc14' ? so : to);
      }
      n.probe.own_done = !!spec.ownDone;
    }
    n.status = 'probing';
  } else if (spec.fail) {
    n.status = 'build_failed';
  } else if (spec.status) {
    n.status = spec.status;
  } else {
    n.status = 'built';
  }
  if (spec.abandoned) n.status = 'abandoned';
  return n;
}

/** Decisions per CONTRACT "Decision rules" (the collector's fill_decision wording). */
function decide(nodes, maxRounds, branches, invalid, abandonedReason) {
  const byId = new Map(nodes.map((n) => [n.id, n]));
  for (const n of nodes) if (n.parent && byId.has(n.parent)) byId.get(n.parent).children.push(n.id);
  for (const n of nodes) n.children.sort();
  const liveKids = (n) => n.children.filter((c) => byId.get(c).status !== 'abandoned');
  for (const n of nodes) {
    const k = n.round, sc = n.score;
    if (n.status === 'abandoned') { n.decision = 'abandoned'; n.decision_reason = abandonedReason(n); continue; }
    if (n.build.ok === false) {
      n.decision = 'discard';
      n.decision_reason = `build failed: ${n.build.problems.split('\n').map((l) => l.trim()).find(Boolean)}`;
      continue;
    }
    if (!sc) {
      n.decision = 'pending';
      n.decision_reason = {probing: 'probe in progress', built: 'built; waiting for its probe', building: 'gates / instruction / packaging',
                           planning: 'evolve plan', authoring: 'authoring'}[n.status] || n.status;
      continue;
    }
    if (invalid[n.id]) { n.decision = 'discard'; n.decision_reason = invalid[n.id]; continue; }
    if (k === 0) {
      n.decision = sc.accept ? 'accepted_evolve' : 'evolve';
      n.decision_reason = `${sc.accept ? 'accepted' : 'not accepted'} (${sc.reason}); every valid round-0 task seeds ${branches} branches`;
    } else if (sc.accept) {
      n.decision = 'accepted';
      n.decision_reason = `accepted (${sc.reason})`;
    } else if (liveKids(n).length) {
      n.decision = 'evolve';
      n.decision_reason = `best valid non-accepted task of lineage ${n.lineage} (${sc.reason}); ${liveKids(n).length} branches in round ${k + 1}`;
    } else if (nodes.some((o) => o.round === k + 1 && o.lineage === n.lineage && o.status !== 'abandoned')) {
      const pick = nodes.find((o) => o.round === k && o.lineage === n.lineage && liveKids(o).length);
      n.decision = 'not_selected';
      n.decision_reason = `not accepted (${sc.reason}); round ${k + 1} continues lineage ${n.lineage}${pick ? ` from ${pick.id}` : ''}`;
    } else if (k >= maxRounds) {
      n.decision = 'end';
      n.decision_reason = `not accepted (${sc.reason}); last round (${k})`;
    } else {
      n.decision = 'pending';
      n.decision_reason = `not accepted (${sc.reason}); waiting for round ${k + 1}`;
    }
  }
  return nodes;
}

function logTail(lines) { return lines.map((l) => l.slice(0, 300)); }

// ------------------------------------------------------------------------------------- narwhals-like run

const NARWHALS_PATTERNS = ['Extending a narrow backend exception to a supported operation',
                           'Making tests conform to the unsupported assumption',
                           'Certifying interpolation arithmetic without testing the changed backend'];

function narwhals() {
  const at = (probeDir) => ({probeDir, patterns: NARWHALS_PATTERNS});
  const ctx0 = at('round-0/probe-b1ade2ed'), ctx1 = at('round-1/probe-c36dfb7b');
  const ctx2 = at('round-2/probe-0d4e9a17'), ctx3a = at('round-3/probe-91b7c3d0'), ctx3b = at('round-3/probe-5e1f0c2a');
  const gateFail = 'gates attempt 0: 7 F2P tests pass on the starting state (start.patch leaves rolling_quantile implemented for polars)\n' +
                   'gates attempt 1: stale test ibis/backends/tests/test_window.py::test_rolling_median[sqlite] fails on the reference solution';
  const specs = [
    // round 0 (mask)
    ['cand-c01', {title: 'Portable exact and approximate order-statistic reductions', f2p: 42, p2p: 4944, s: [2, 3, 0], t: [5, 0, 0]}, ctx0],
    ['cand-c02', {title: 'Null-aware value-selecting reductions with ordering and keys', f2p: 76, p2p: 4044, s: [0, 1, 4], t: [3, 0, 2], verifier: true}, ctx0],
    ['cand-c03', {title: 'Filtered dispersion and pairwise statistical reductions', f2p: 116, p2p: 4918, s: [1, 4, 0], t: [2, 3, 0]}, ctx0],
    ['cand-c04', {title: 'Filtered, ordered string and array collection', f2p: 78, p2p: 4903, s: [1, 4, 0], t: [4, 1, 0], sim: 6, manyTests: true, longText: true}, ctx0],
    ['cand-c05', {title: 'Type-sensitive JSON unwrapping and container conversion', fail: 'author-1: GateError: the reference solution fails 2 F2P tests (ibis/backends/tests/test_json.py::test_json_unwrap[duckdb-getattr-bool])\n' + gateFail, failF2p: 40}, ctx0],
    // round 1: c02's children were probed in the superseded round-1/probe-7f97a213 (old seed rule)
    ['cand-c02-r1b0', {title: 'Filtered rolling order statistics', f2p: 311, p2p: 481, abandoned: true}, ctx1],
    ['cand-c02-r1b1', {title: 'Rolling statistics with row masks', f2p: 326, p2p: 568, abandoned: true}, ctx1],
    ['cand-c02-r1b2', {title: 'Per-array quantiles with portable interpolation', f2p: 256, p2p: 900, abandoned: true}, ctx1],
    ['cand-c02-r1b3', {title: 'Filtered rolling order statistics', f2p: 297, p2p: 559, abandoned: true}, ctx1],
    ['cand-c02-r1b4', {title: 'Ordered rolling reductions with valid-sample thresholds', f2p: 354, p2p: 241, abandoned: true}, ctx1],
    ['cand-c01-r1b0', {title: 'Null-aware rolling order statistics for SQL tables', f2p: 343, p2p: 533, s: [3, 2, 0], t: [5, 0, 0]}, ctx1],
    ['cand-c01-r1b1', {title: 'Ordered rolling statistics with eligibility masks', f2p: 364, p2p: 332, s: [2, 3, 0], t: [4, 1, 0]}, ctx1],
    ['cand-c01-r1b2', {title: 'Filtered sample quantiles on SQLite and Polars', f2p: 511, p2p: 1003, s: [1, 4, 0], t: [4, 1, 0], sim: 5, longText: true}, ctx1],
    ['cand-c01-r1b3', {title: 'Ordered rolling extrema and quantiles', f2p: 253, p2p: 3740, s: [1, 1, 3], t: [4, 1, 0]}, ctx1],
    ['cand-c01-r1b4', {title: 'Rolling numeric statistics with sample masks', f2p: 233, p2p: 412, s: [1, 4, 0], t: [5, 0, 0], sim: 7}, ctx1],
    ['cand-c03-r1b0', {title: 'Filtered row-window order statistics', f2p: 395, p2p: 557, s: [4, 1, 0], t: [5, 0, 0]}, ctx1],
    ['cand-c03-r1b1', {title: 'Predicate-aware rolling order statistics', fail: gateFail}, ctx1],
    ['cand-c03-r1b2', {title: 'Array quantiles with explicit interpolation', f2p: 339, p2p: 1155, s: [2, 3, 0], t: [5, 0, 0]}, ctx1],
    ['cand-c03-r1b3', {title: 'Filtered rolling order statistics', f2p: 182, p2p: 366, s: [1, 4, 0], t: [3, 2, 0], sim: 4}, ctx1],
    ['cand-c03-r1b4', {title: 'Ordered rolling reductions with eligible-row thresholds', f2p: 570, p2p: 701, s: [0, 5, 0], t: [2, 3, 0]}, ctx1],
    ['cand-c04-r1b0', {title: 'Per-group limits for ordered collections', f2p: 244, p2p: 219, s: [3, 2, 0], t: [5, 0, 0], cheated: 1}, ctx1],
    ['cand-c04-r1b1', {title: 'Ordered collection pages', f2p: 258, p2p: 249, s: [1, 4, 0], t: [4, 1, 0], sim: 5}, ctx1],
    ['cand-c04-r1b2', {title: 'Interpolated quantile vectors for arrays', f2p: 262, p2p: 142, s: [0, 5, 0], t: [4, 1, 0], sim: 6}, ctx1],
    ['cand-c04-r1b3', {title: 'Ordered positional collection reductions', f2p: 385, p2p: 251, s: [2, 3, 0], t: [5, 0, 0]}, ctx1],
    ['cand-c04-r1b4', {title: 'Paged ordered collection', f2p: 328, p2p: 1034, s: [1, 4, 0], t: [5, 0, 0], sim: 3}, ctx1],
    // round 2
    ['cand-c01-r1b2-r2b0', {title: 'Weighted window quantiles with frame bounds', f2p: 288, p2p: 640, s: [3, 2, 0], t: [5, 0, 0]}, ctx2],
    ['cand-c01-r1b2-r2b1', {title: 'Grouped rolling medians with null thresholds', f2p: 305, p2p: 702, s: [1, 4, 0], t: [5, 0, 0], sim: 5}, ctx2],
    ['cand-c01-r1b2-r2b2', {title: 'Sample-aware ntile buckets across backends', f2p: 221, p2p: 958, s: [2, 3, 0], t: [4, 1, 0]}, ctx2],
    ['cand-c01-r1b2-r2b3', {title: 'Range-framed order statistics on timestamps', fail: 'AgentError: codex exited 1 after 3 attempts (author-1)\nauthor-1: start.patch missing (it may only be empty for evolve tasks)'}, ctx2],
    ['cand-c01-r1b2-r2b4', {title: 'Deterministic percentile ranks with ties', f2p: 198, p2p: 1102, s: [0, 5, 0], t: [1, 4, 0]}, ctx2],
    ['cand-c03-r1b3-r2b0', {title: 'Masked expanding-window statistics', f2p: 276, p2p: 488, s: [1, 4, 0], t: [4, 1, 0], sim: 5}, ctx2],
    ['cand-c03-r1b3-r2b1', {title: 'Rolling interquartile ranges with filters', f2p: 241, p2p: 519, s: [0, 5, 0], t: [5, 0, 0], sim: 8}, ctx2],
    ['cand-c03-r1b3-r2b2', {title: 'Centered rolling windows with min periods', f2p: 330, p2p: 402, s: [4, 1, 0], t: [5, 0, 0]}, ctx2],
    ['cand-c03-r1b3-r2b3', {title: 'Partitioned rolling modes on string columns', f2p: 187, p2p: 377, s: [2, 3, 0], t: [4, 1, 0]}, ctx2],
    ['cand-c03-r1b3-r2b4', {title: 'Time-bucketed rolling quantiles', f2p: 352, p2p: 610, s: [1, 3, 1], t: [3, 1, 1], verifier: true}, ctx2],
    ['cand-c04-r1b1-r2b0', {title: 'Keyset pages over ordered collections', f2p: 233, p2p: 260, s: [3, 2, 0], t: [5, 0, 0]}, ctx2],
    ['cand-c04-r1b1-r2b1', {title: 'Distinct ordered collection with limits', f2p: 219, p2p: 248, s: [2, 3, 0], t: [5, 0, 0]}, ctx2],
    ['cand-c04-r1b1-r2b2', {title: 'Ordered top-k collection per group', f2p: 266, p2p: 231, s: [1, 4, 0], t: [4, 1, 0], sim: 5}, ctx2],
    ['cand-c04-r1b1-r2b3', {title: 'Offset-limited array aggregation', f2p: 201, p2p: 295, s: [1, 4, 0], t: [2, 3, 0]}, ctx2],
    ['cand-c04-r1b1-r2b4', {title: 'Ordered collection with null placement', f2p: 248, p2p: 240, s: [3, 2, 0], t: [5, 0, 0]}, ctx2],
    // round 3 (last): two lineages scored in the first probe, c04's children still probing in a resumed probe
    ['cand-c01-r1b2-r2b1-r3b0', {title: 'Grouped rolling medians over range frames', f2p: 312, p2p: 688, s: [2, 3, 0], t: [5, 0, 0]}, ctx3a],
    ['cand-c01-r1b2-r2b1-r3b1', {title: 'Rolling medians with per-group min periods', f2p: 297, p2p: 701, s: [1, 4, 0], t: [5, 0, 0], sim: 7}, ctx3a],
    ['cand-c01-r1b2-r2b1-r3b2', {title: 'Null-threshold rolling percentiles', f2p: 284, p2p: 655, s: [1, 4, 0], t: [3, 2, 0], sim: 4}, ctx3a],
    ['cand-c01-r1b2-r2b1-r3b3', {title: 'Rolling median absolute deviation', f2p: 263, p2p: 690, s: [3, 2, 0], t: [5, 0, 0]}, ctx3a],
    ['cand-c01-r1b2-r2b1-r3b4', {title: 'Frame-aware rolling medians for Polars', fail: gateFail}, ctx3a],
    ['cand-c03-r1b3-r2b0-r3b0', {title: 'Masked expanding quantiles with resets', f2p: 290, p2p: 471, s: [2, 3, 0], t: [5, 0, 0]}, ctx3a],
    ['cand-c03-r1b3-r2b0-r3b1', {title: 'Expanding-window statistics with eligibility masks', f2p: 301, p2p: 462, s: [0, 5, 0], t: [2, 3, 0]}, ctx3a],
    ['cand-c03-r1b3-r2b0-r3b2', {title: 'Cumulative order statistics with filters', f2p: 279, p2p: 503, s: [2, 3, 0], t: [1, 1, 3]}, ctx3a],
    ['cand-c03-r1b3-r2b0-r3b3', {title: 'Masked cumulative extrema', f2p: 244, p2p: 450, s: [4, 1, 0], t: [5, 0, 0]}, ctx3a],
    ['cand-c03-r1b3-r2b0-r3b4', {title: 'Expanding interquartile ranges', f2p: 318, p2p: 477, s: [1, 4, 0], t: [4, 1, 0], sim: 5}, ctx3a],
    ['cand-c04-r1b1-r2b2-r3b0', {title: 'Top-k collection with tie handling', f2p: 251, p2p: 229, live: {sc14: [5, 5], kimi: [5, 5]}, ownDone: true}, ctx3b],
    ['cand-c04-r1b1-r2b2-r3b1', {title: 'Per-group ordered sampling', f2p: 239, p2p: 233, live: {sc14: [0, 5], kimi: [3, 5]}}, ctx3b],
    ['cand-c04-r1b1-r2b2-r3b2', {title: 'Ordered top-k with null placement', f2p: 270, p2p: 226, live: {sc14: [1, 5], kimi: [2, 5]}}, ctx3b],
    ['cand-c04-r1b1-r2b2-r3b3', {title: 'Windowed top-k collection', f2p: 262, p2p: 241, live: {sc14: [2, 5], kimi: [2, 5]}, liveInfra: {kimi: 1}}, ctx3b],
    ['cand-c04-r1b1-r2b2-r3b4', {title: 'Bounded ordered collection per key', f2p: 244, p2p: 237, live: {sc14: [1, 5], kimi: [2, 5]}}, ctx3b],
  ];
  const invalid = {
    'cand-c02': 'verifier produced no grade for 6 rollout(s): kimi-r3, kimi-r4, sc14-r1, sc14-r2, sc14-r3, sc14-r4',
    'cand-c01-r1b3': 'fewer than 3 valid rollouts per model (valid [2, 5])',
    'cand-c03-r1b3-r2b4': 'verifier produced no grade for 2 rollout(s): kimi-r4, sc14-r4',
    'cand-c03-r1b3-r2b0-r3b2': 'fewer than 3 valid rollouts per model (valid [5, 2])',
  };
  const nodes = decide(specs.map(([id, spec, ctx]) => makeNode(id, spec, ctx)), 3, 5, invalid,
    (n) => `parent ${n.parent} is not a valid seed (${invalid[n.parent]})`);
  const seeds = (k) => {
    const members = nodes.filter((n) => n.round === k && n.score && n.status !== 'abandoned');
    return {valid: members.filter((n) => !invalid[n.id]).map((n) => n.id),
            invalid: Object.fromEntries(members.filter((n) => invalid[n.id]).map((n) => [n.id, invalid[n.id]]))};
  };
  const roundInfo = (k, status, probe, abandoned, hasSeeds) => {
    const members = nodes.filter((n) => n.round === k && n.status !== 'abandoned');
    return {index: k, status, probe, abandoned_probes: abandoned, seeds: hasSeeds ? seeds(k) : null, nodes: members.length,
            built: members.filter((n) => n.build.ok === true).length, scored: members.filter((n) => n.status === 'scored').length,
            accepted: members.filter((n) => n.decision === 'accepted' || n.decision === 'accepted_evolve').length};
  };
  const probeSummary = (dirs, tasks, done, total, sc, ki, live) => ({dir: dirs[dirs.length - 1], dirs, tasks, slots_total: total, slots_done: done,
                                                                     by_model: {sc14: sc, kimi: ki}, live});
  const rounds = [
    roundInfo(0, 'done', probeSummary(['round-0/probe-b1ade2ed'], 4, 40, 40, [20, 20], [20, 20], false), [], true),
    roundInfo(1, 'done', probeSummary(['round-1/probe-c36dfb7b'], 14, 140, 140, [70, 70], [70, 70], false), ['round-1/probe-7f97a213'], true),
    roundInfo(2, 'done', probeSummary(['round-2/probe-0d4e9a17'], 14, 140, 140, [70, 70], [70, 70], false), [], true),
    roundInfo(3, 'probing', probeSummary(['round-3/probe-91b7c3d0', 'round-3/probe-5e1f0c2a'], 14, 113, 140, [54, 70], [59, 70], true), [], true),
  ];
  const stage = 'round-3 probe (113/140 slots)';
  const started = NOW - 3 * 3600 - 24 * 60 - 31;
  const log = [
    '2026-10-17 14:31:02 score syn-ibis-cand-c03-r1b3-r2b0-r3b4-5d2c11e0: similarity 5/10 accept=False',
    '2026-10-17 14:31:02 score syn-ibis-cand-c03-r1b3-r2b0-r3b2-0b9f4a73: inconclusive: fewer than 3 valid rollouts per model accept=False',
    '2026-10-17 14:31:03 cand-c03-r1b3-r2b0-r3b2: invalid, not evolved (fewer than 3 valid rollouts per model (valid [5, 2]))',
    '2026-10-17 14:31:03 build narwhals-rolling-window-suite x ibis-project/ibis: 5 accepted',
    '2026-10-17 14:33:40 resume: round-3 evolve for cand-c04-r1b1-r2b2 (5 branches)',
  ];
  for (let b = 0; b < 5; b += 1) {
    log.push(`2026-10-17 14:3${4 + b}:1${b} agent[codex] start B/cand-c04-r1b1-r2b2-r3b${b}/evolve`);
    log.push(`2026-10-17 14:4${b}:0${b} agent[codex] exit 0 B/cand-c04-r1b1-r2b2-r3b${b}/evolve problems=0`);
    log.push(`2026-10-17 14:4${b}:3${b} agent[codex] start B/cand-c04-r1b1-r2b2-r3b${b}/author-0`);
    log.push(`2026-10-17 14:5${b}:2${b} agent[codex] exit 0 B/cand-c04-r1b1-r2b2-r3b${b}/author-0 problems=0`);
  }
  log.push('2026-10-17 14:55:09 probe: sc14 via http://127.0.0.1:27555/v1; waiting up to 10800 s for a routable worker');
  log.push('2026-10-17 14:55:09 probe: 9 student worker(s) routable: 20396673 ready http://10.65.12.147:8000/v1, 20396850 ready http://10.65.7.55:8000/v1, …');
  log.push('2026-10-17 14:55:10 hub: Authorization: Bearer [REDACTED]');
  log.push('2026-10-17 14:55:45 probe: running campaign B/round-3/probe-5e1f0c2a');
  for (let i = 0; i < 14; i += 1) log.push(`2026-10-17 14:${56 + Math.floor(i / 4)}:${String(10 + i * 3).padStart(2, '0')} slot ${i % 2 ? 'kimi' : 'sc14'}-r${i % 5} task ${i % 5}: ${i % 3 ? 'pass' : 'fail'} (attempt 1)`);
  return {
    schema: 1, generated_at: NOW, task_id: 'narwhals-rolling-window-suite', run: 'official1-narwhals', repo: 'ibis-project/ibis',
    build_root: `${S}/synth/runs/official1-narwhals/build/narwhals-rolling-window-suite/ibis-project__ibis`, max_rounds: 3, branches: 5,
    job: {job_id: '20408127', slurm_state: 'RUNNING', partition: 'cpu_interactive', node: 'cpu1-00098', elapsed: '3:24:31', time_left: '20:35:29',
          log_path: `${S}/synth/runs/slurm/synth-build-official1-narwhals-20408127.out`, log_tail: logTail(log.slice(-40)), stage,
          status: 'running', submitted_at: started - 40, started_at: started, ended_at: null, jobs: ['20386204', '20387200', '20388429', '20408127']},
    failure_analysis: {
      patterns: [
        {name: 'Extending a narrow backend exception to a supported operation', category: 'requirement_scope',
         description: "Four of five graded student rollouts broaden the task's DuckDB rolling_quantile exception into a SQL rolling_median restriction. sc14-r1, sc14-r2, and sc14-r3 reject median on both DuckDB and SQLFrame; sc14-r0 repairs DuckDB after a native probe but retains a SQLFrame ban. The resulting NotImplementedError failures reproduce exactly.",
         student_slots: ['sc14-r0', 'sc14-r1', 'sc14-r2', 'sc14-r3'],
         teacher_contrast: 'All five teacher patches preserve generic SQL median dispatch and distinguish it from the explicit quantile restriction, supported by in-episode native DuckDB window probes.', evidence: null},
        {name: 'Making tests conform to the unsupported assumption', category: 'test_validation',
         description: 'The same four failing student rollouts observe SQL median failures in their own tests and add backend skips because their implementations raise NotImplementedError. They accept the resulting green tests as completion.',
         student_slots: ['sc14-r0', 'sc14-r1', 'sc14-r2', 'sc14-r3'],
         teacher_contrast: 'Teacher runs keep SQL median tests active and investigate discrepancies using native calculations and repository fixtures.', evidence: null},
        {name: 'Certifying interpolation arithmetic without testing the changed backend', category: 'numeric_semantics',
         description: 'sc14-r1 implements Arrow higher interpolation with -int(-quantile * (n - 1)) and asserts that -int(-x) is ceiling; its smoke check uses pandas, bypassing the handwritten Arrow branch.',
         student_slots: ['sc14-r1'],
         teacher_contrast: 'All five teacher Arrow patches delegate per-window quantiles to pyarrow.compute.quantile.',
         evidence: 'reproduction: evidence/reproduce.py higher[pyarrow] returns [1,1,1,2] instead of [1,2,2,3]'},
      ],
      transferable_signature: 'In a Python API spanning several adapters, an agent generalizes an exception for one operation/backend to a related operation, mistakes an assumed implementation dependency for the contract, and suppresses integration failures by changing test applicability to match its implementation.',
      excluded: [],
      pass_stats: {kimi: {pass: 5, fail: 0, excluded: 0, rate: 1.0}, sc14: {pass: 1, fail: 4, excluded: 0, rate: 0.2}},
      markdown: '# Failure analysis: narwhals-rolling-window-suite\n\n## Task summary\n\nThe student passed **1 of 5** graded rollouts; the teacher passed **5 of 5**. ' +
        'All four student failures share a scope error: they reject rolling median on SQL backends even though the task\'s backend exception applies only to rolling quantile on DuckDB.\n\n' +
        '## Outcome table\n\n| Slot | Model | Recorded result | F2P passed | P2P passed |\n|---|---|---|---|---|\n' +
        '| sc14-r0 | Student | Fail | 102/103 | 10093/10093 |\n| sc14-r1 | Student | Fail | 100/103 | 10093/10093 |\n| sc14-r4 | Student | Pass | 103/103 | 10093/10093 |\n\n' +
        '…[18342 chars omitted]…\n\n## Transferable signature\n\nSee result.json.\n',
    },
    plan: {
      candidates: [
        ['c01', 'Portable exact and approximate order-statistic reductions', 4, 31, true],
        ['c02', 'Null-aware value-selecting reductions with ordering and keys', 4, 28, true],
        ['c03', 'Filtered dispersion and pairwise statistical reductions', 5, 44, true],
        ['c04', 'Filtered, ordered string and array collection', 4, 26, true],
        ['c05', 'Type-sensitive JSON unwrapping and container conversion', 2, 18, false],
        ['c06', 'Null-aware element reductions over array columns', 3, 22, false],
        ['c07', 'Portable regex search, capture, replacement and splitting', 4, 35, false],
        ['c08', 'String position and substring normalization across adapters', 5, 29, false],
        ['c09', 'Null-preserving array set operations and value removal', 3, 17, false],
        ['c10', 'Array element access, slicing and position normalization', 3, 21, false],
      ].map(([id, title, files, tests, built]) => ({
        id, title, summary: `${title}: remove the implementation across the SQL compilers and the Polars backend, keeping the public API and docs.`,
        files: ['ibis/backends/sql/compilers/base.py', 'ibis/backends/sql/compilers/duckdb.py', 'ibis/backends/polars/compiler.py',
                'ibis/backends/sql/compilers/datafusion.py', 'ibis/expr/types/generic.py'].slice(0, files),
        tests, target_patterns: id <= 'c04' ? ['Extending a narrow backend exception to a supported operation', 'Making tests conform to the unsupported assumption'] : [],
        built, status: null, decision: null})),
      mask_plan_excerpt: '# Mask plan\n\nThe target exercises rolling order statistics across adapters with an explicit, narrow backend exception. ' +
        'Candidates remove feature families whose implementation spans several compilers so that the student must keep each backend restriction narrow.\n',
    },
    rounds, nodes,
  };
}

// ---------------------------------------------------------------------------------------------- bandit run

function bandit() {
  // Older collector shapes: no focus_pattern / probe.partial, teacher_audit as a JSON-dumped string.
  const ctx0 = {probeDir: 'round-0/probe-ab242006', auditString: true}, ctx1 = {probeDir: 'round-1/probe-017d80cf', auditString: true};
  const ctx1b = {probeDir: 'round-1/probe-f9af9d01', auditString: true};
  const r = (id, title, f2p, p2p, s, t, extra) => [id, Object.assign({title, f2p, p2p, s: s === null ? undefined : [Math.round(s * 5), 5 - Math.round(s * 5), 0],
                                                                        t: t === null ? undefined : [Math.round(t * 5), 5 - Math.round(t * 5), 0]}, extra || {})];
  const specs = [
    r('cand-c01', 'Pylint option and diagnostic integration', 17, 30, 1.0, 1.0),
    r('cand-c02', 'Layered diagnostic selection and source modelines', 15, 31, 0.6, 0.8),
    r('cand-c03', 'Isolation and exception boundaries for per-file runs', 15, 25, 0.8, 1.0),
    r('cand-c04', 'Canonical configuration sections across INI and TOML', 17, 34, 0.8, 0.8),
    r('cand-c05', 'In-memory source through the complete checking pipeline', 28, 25, 1.0, 0.8),
    r('cand-c01-r1b0', 'Persistent diagnostic baselines with capture runs', 75, 44, 0.6, 0.6),
    r('cand-c01-r1b1', 'Suppression-aware JSON audit reports', 37, 46, 0.8, 1.0),
    r('cand-c01-r1b2', 'Pylama pytest cache maintenance', 39, 41, 1.0, 0.8),
    r('cand-c01-r1b3', 'Pytest audit cache freshness', 32, 39, 1.0, 1.0),
    r('cand-c01-r1b4', 'Persistent diagnostic baselines with scoped update', 66, 41, 0.0, 0.0),
    r('cand-c02-r1b0', 'Persistent lint baselines', 37, 45, 0.6, 0.8),
    r('cand-c02-r1b1', 'JSON diagnostic accounting reports', 51, 49, 0.8, 1.0),
    r('cand-c02-r1b2', 'Selective pytest cache maintenance', 27, 41, 1.0, 1.0),
    r('cand-c02-r1b3', 'Timed pytest cache reuse with decision reports', 25, 40, 1.0, 1.0),
    r('cand-c02-r1b4', 'Diagnostic baselines with capture and pruning', 100, 43, 0.6, 1.0),
    r('cand-c03-r1b0', 'Persistent diagnostic baselines with capture mode', 68, 42, 1.0, 1.0),
    r('cand-c03-r1b1', 'Diagnostic accounting in structured run reports', 47, 46, 1.0, 1.0),
    r('cand-c03-r1b2', 'Composable pytest lint-cache maintenance', 24, 40, 0.0, 0.8, {sim: 9}),
    r('cand-c03-r1b3', 'Age-limited pytest cache with miss reasons', 27, 51, 0.8, 1.0),
    r('cand-c03-r1b4', 'Diagnostic baseline capture and refresh', 71, 42, 0.4, 1.0),
    r('cand-c04-r1b0', 'Lint baseline capture and comparison', 32, 67, 0.8, 1.0),
    r('cand-c04-r1b1', 'Incremental linting with aggregate JSON reports', 54, 44, 0.2, 0.0),
    r('cand-c04-r1b2', 'Pytest cache maintenance across session finalization', 23, 41, 0.8, 1.0),
    r('cand-c04-r1b3', 'Age-limited pytest lint caching', 27, 41, 1.0, 1.0),
    r('cand-c04-r1b4', 'Persistent diagnostic baselines', 84, 42, 0.2, 1.0, {sim: 7}),
  ].map(([id, spec]) => [id, spec, id.includes('-r1') ? ctx1 : ctx0]);
  // cand-c05's branches were probed in a second campaign (round-1/probe-f9af9d01) after their build was resumed.
  [r('cand-c05-r1b0', 'Cached in-memory checks with invalidation', 31, 27, 0.6, 0.8),
   r('cand-c05-r1b1', 'Streaming stdin sources with cache keys', 44, 30, 0.8, 1.0),
   r('cand-c05-r1b2', 'In-memory pipeline with per-linter timings', 29, 26, 0.4, 0.8),
   r('cand-c05-r1b3', 'Virtual file sources for editor integrations', 38, 25, 1.0, 1.0),
   r('cand-c05-r1b4', 'Source fingerprints across in-memory and disk runs', 52, 33, 0.6, 0.6)]
    .forEach(([id, spec]) => specs.push([id, spec, ctx1b]));
  const gateFail = 'gates attempt 0: 3 F2P tests pass on the starting state (start.patch keeps the expiry check)\n' +
                   'gates attempt 1: tests/test_baseline.py::test_expiry_days_zero fails on the reference solution';
  const r2 = {
    'cand-c01-r1b4': [['Baseline drift reports for scoped updates', 'built'], ['Scoped baseline pruning', 'building'], ['Baseline capture with path filters', 'authoring'],
                      ['Per-linter baseline accounting', 'authoring'], ['Incremental baseline refresh', 'planning']],
    'cand-c02-r1b0': [['Lint baselines with expiry', 'failed'], ['Baseline comparison across configs', 'authoring'], ['Baseline JSON schema versions', 'authoring'],
                      ['Ignored-path baselines', 'planning'], ['Baseline merge on parallel runs', 'planning']],
    'cand-c03-r1b4': [['Refresh-only baseline mode', 'built'], ['Baseline capture for stdin sources', 'built'], ['Baseline refresh with dry-run reports', 'building'],
                      ['Baseline capture across modelines', 'authoring'], ['Baseline refresh exit codes', 'authoring']],
    'cand-c04-r1b1': [['Aggregate reports with cache hit ratios', 'building'], ['Incremental runs with changed-file detection', 'authoring'],
                      ['JSON report merging across workers', 'authoring'], ['Cache-aware exit codes', 'planning'], ['Report filtering by severity', 'planning']],
    'cand-c05-r1b2': [['Cache keys for stdin sources', 'built'], ['Invalidation on configuration changes', 'building'],
                      ['Per-linter cache timings in JSON reports', 'authoring'], ['Editor buffer fingerprints', 'planning'], ['Cache size limits', 'planning']],
  };
  for (const [parent, kids] of Object.entries(r2)) {
    kids.forEach(([title, status], b) => specs.push([`${parent}-r2b${b}`, status === 'failed' ? {title, fail: gateFail, failF2p: randint(20, 90)}
      : {title, status, f2p: randint(20, 90), p2p: randint(30, 60)}, {probeDir: null, auditString: true}]));
  }
  const nodes = decide(specs.map(([id, spec, ctx]) => makeNode(id, spec, ctx)), 3, 5, {}, () => '');
  const seeds = (k) => ({valid: nodes.filter((n) => n.round === k && n.score).map((n) => n.id), invalid: {}});
  const round2 = nodes.filter((n) => n.round === 2);
  const built2 = round2.filter((n) => n.build.ok === true).length;
  return {
    schema: 1, generated_at: NOW, task_id: 'bandit-incremental-cache-control', run: 'official1-bandit', repo: 'klen/pylama',
    build_root: `${S}/synth/runs/official1-bandit/build/bandit-incremental-cache-control/klen__pylama`, max_rounds: 3, branches: 5,
    job: {job_id: '20408123', slurm_state: 'RUNNING', partition: 'cpu_dataprocessing', node: 'cpu1-00007', elapsed: '3:24:35', time_left: '20:35:25',
          log_path: `${S}/synth/runs/slurm/synth-build-official1-bandit-20408123.out`,
          log_tail: ['2026-10-17 14:48:02 probe: running campaign B/round-1/probe-f9af9d01', '2026-10-17 14:58:41 slot sc14-r4 task 0: pass (attempt 1)',
                     '2026-10-17 14:59:30 build pylama x klen/pylama: round 1 scored, 2 accepted; 5 seeds for round 2',
                     '2026-10-17 14:59:31 agent[codex] start B/cand-c05-r1b2-r2b0/evolve'],
          stage: `round-2 building (${built2}/${round2.length} built)`, status: 'running', submitted_at: NOW - 12330, started_at: NOW - 12275, ended_at: null,
          jobs: ['20386205', '20387204', '20388249', '20408123']},
    failure_analysis: {
      patterns: [
        {name: 'Suppressing the report instead of suppressing findings', category: 'requirement_misinterpretation',
         description: '4/5 graded student slots exit warm-cache mode before formatting, equating empty results with empty stdout.',
         student_slots: ['sc14-r0', 'sc14-r1', 'sc14-r2', 'sc14-r3'], teacher_contrast: 'All five teacher patches clear b_mgr.results after storing and retain normal formatting.', evidence: null},
        {name: 'Adding counters outside the established aggregate schema', category: 'interface_integration',
         description: 'sc14-r1 puts cache_hits/cache_misses directly under metrics, leaving metrics._totals without them.',
         student_slots: ['sc14-r1'], teacher_contrast: 'Teachers integrate counters into metrics._totals.', evidence: null},
        {name: 'Missing lifecycle effects when composing maintenance operations', category: 'control_flow_integration',
         description: 'sc14-r0 calls save after clear, recreating an empty cache.json; sc14-r2 prints stats after stdout is closed.',
         student_slots: ['sc14-r0', 'sc14-r2'], teacher_contrast: 'kimi-r0 removes its cache file without immediately recreating it.', evidence: null},
        {name: 'Optimizing immediate expiry into disabled storage', category: 'edge_case_semantics',
         description: 'sc14-r4 skips store entirely when expiry_days == 0, so repeated scans report not_cached instead of expired.',
         student_slots: ['sc14-r4'], teacher_contrast: 'Teachers retain store-then-expire behavior.', evidence: null},
        {name: 'Implementation-shaped tests certify the wrong contract', category: 'validation_oracle_gap',
         description: '5/5 slots write tests that assert their own implementation choices instead of the public contract.',
         student_slots: ['sc14-r0', 'sc14-r1', 'sc14-r2', 'sc14-r3', 'sc14-r4'], teacher_contrast: 'Teacher smoke checks consume JSON and exercise cache state transitions.', evidence: null},
      ],
      transferable_signature: 'When extending a Python CLI with persistent state and alternate execution modes, the agent substitutes locally plausible behavior for the complete observable contract, then writes or relaxes tests around that implementation.',
      excluded: [], pass_stats: {kimi: {pass: 5, fail: 0, excluded: 0, rate: 1.0}, sc14: {pass: 0, fail: 5, excluded: 0, rate: 0.0}},
      markdown: '# Failure analysis: bandit-incremental-cache-control\n\nThe student passed **0 of 5** graded rollouts; the teacher passed **5 of 5**.\n',
    },
    plan: {candidates: [['c01', 'Pylint option and diagnostic integration'], ['c02', 'Layered diagnostic selection and source modelines'],
                        ['c03', 'Isolation and exception boundaries for per-file runs'], ['c04', 'Canonical configuration sections across INI and TOML'],
                        ['c05', 'In-memory source through the complete checking pipeline']]
             .map(([id, title]) => ({id, title, summary: title, files: ['pylama/main.py', 'pylama/core.py'], tests: 12, target_patterns: [], built: true, status: null, decision: null})),
           mask_plan_excerpt: '# Mask plan\n\npylama mirrors bandit\'s incremental cache surface.\n'},
    rounds: [
      {index: 0, status: 'done', probe: {dir: 'round-0/probe-ab242006', dirs: ['round-0/probe-4044c2b2', 'round-0/probe-ab242006'], tasks: 5, slots_total: 50, slots_done: 50,
                                         by_model: {sc14: [25, 25], kimi: [25, 25]}, live: false}, abandoned_probes: [], seeds: seeds(0), nodes: 5, built: 5, scored: 5, accepted: 0},
      {index: 1, status: 'done', probe: {dir: 'round-1/probe-f9af9d01', dirs: ['round-1/probe-017d80cf', 'round-1/probe-f9af9d01'], tasks: 25, slots_total: 250, slots_done: 250,
                                         by_model: {sc14: [125, 125], kimi: [125, 125]}, live: false}, abandoned_probes: [], seeds: seeds(1), nodes: 25, built: 25, scored: 25, accepted: 2},
      {index: 2, status: 'building', probe: null, abandoned_probes: [], seeds: null, nodes: round2.length, built: built2, scored: 0, accepted: 0},
      {index: 3, status: 'pending', probe: null, abandoned_probes: [], seeds: null, nodes: 0, built: 0, scored: 0, accepted: 0},
    ],
    nodes,
  };
}

// ------------------------------------------------------------------------------------- pending run (sparse)

function fdPending() {
  return {schema: 1, generated_at: NOW, task_id: 'fd-deterministic-multi-key-sorting', run: 'official2-fd', repo: 'lsd-rs/lsd', build_root: null,
          max_rounds: 3, branches: 5,
          job: {job_id: '20411342', slurm_state: 'PENDING', partition: 'cpu_dataprocessing', node: null, elapsed: '0:00', time_left: null,
                log_path: null, log_tail: [], stage: 'pending: target failure analysis', status: 'pending', submitted_at: NOW - 900,
                started_at: null, ended_at: null, jobs: ['20411342']},
          failure_analysis: null, plan: null, rounds: [], nodes: []};
}

// --------------------------------------------------------------------------------------------- fleet.json

function fleet() {
  const ready = (slot, job, node, ip, start, running, waiting, kv, gen, prompt) => ({
    job_id: job, slot, partition: 'batch', node, state: 'RUNNING', registry_status: 'ready', ready_at: start + 447, start, end: start + 14400,
    left_s: Math.round(start + 14400 - NOW), heartbeat_age_s: randint(2, 28), base_url: `http://${ip}:8000/v1`, pending_reason: null, est_start: null,
    metrics: {running, waiting, kv_usage: kv, prompt_tps: prompt, gen_tps: gen, scraped_at: NOW - 14, error: null}});
  const t0 = NOW - 2803;
  const workers = [
    ready('01', '20396673', 'pool0-00873', '10.65.12.147', t0, 5, 0, 0.112, 101.4, 1890.2),
    ready('02', '20396850', 'pool0-00025', '10.65.7.55', t0, 6, 0, 0.131, 96.2, 2011.7),
    ready('03', '20396988', 'pool0-00381', '10.65.4.219', t0, 4, 0, 0.087, 88.9, 1502.3),
    ready('04', '20397120', 'pool0-00438', '10.65.5.201', t0, 6, 1, 0.236, 92.5, 2410.0),
    ready('05', '20397190', 'pool0-00874', '10.65.12.145', t0 + 233, 5, 0, 0.096, 88.1, 1234.5),
    ready('06', '20397302', 'pool0-00774', '10.65.9.33', t0 + 233, 6, 2, 0.211, 79.3, 2205.8),
    ready('07', '20397493', 'pool0-00556', '10.65.10.12', t0 + 233, 0, 0, 0.012, 0.0, 0.0),           // idle
    ready('08', '20397657', 'pool0-00522', '10.65.3.88', t0 + 640, 4, 0, 0.102, 94.0, 1530.1),
    Object.assign(ready('09', '20385511', 'pool0-01726', '10.65.21.9', NOW - 13300, 2, 0, 0.041, 70.1, 410.8),
                  {registry_status: 'draining', left_s: 1100}),
    {job_id: '20409912', slot: '10', partition: 'batch', node: 'pool0-00082', state: 'RUNNING', registry_status: 'starting', ready_at: null,
     start: NOW - 260, end: NOW - 260 + 14400, left_s: 14140, heartbeat_age_s: 9, base_url: null, pending_reason: null, est_start: null,
     metrics: {running: null, waiting: null, kv_usage: null, prompt_tps: null, gen_tps: null, scraped_at: NOW - 14,
               error: 'URLError: [Errno 111] Connection refused'}},
    {job_id: '20410388', slot: '11', partition: 'batch', node: null, state: 'PENDING', registry_status: null, ready_at: null, start: null, end: null,
     left_s: null, heartbeat_age_s: null, base_url: null, pending_reason: 'Priority', est_start: NOW + 1740, metrics: null},
    {job_id: '20410391', slot: '12', partition: 'batch', node: null, state: 'PENDING', registry_status: null, ready_at: null, start: null, end: null,
     left_s: null, heartbeat_age_s: null, base_url: null, pending_reason: 'Resources', est_start: null, metrics: null},
  ];
  const by = {};
  for (const w of workers.filter((x) => x.metrics && x.metrics.running !== null)) {
    by[w.job_id] = {requests: randint(55, 110), errors: w.slot === '06' ? 2 : w.slot === '04' ? 1 : 0, p90_s: round(14 + rand() * 16, 1)};
  }
  const history = [];
  for (let t = NOW - 86400 + 60; t <= NOW - 12; t += 120) {
    if (t > NOW - 9 * 3600 && t < NOW - 9 * 3600 + 1500) continue;   // collector down for 25 min
    const phase = (t - (NOW - 86400)) / 86400;
    const burst = 0.5 + 0.5 * Math.sin(phase * Math.PI * 6);
    const turnover = ((t - NOW) % 14400 + 14400) % 14400 < 600;          // workers resubmitted every 4 h
    const routable = turnover ? 8 : (rand() < 0.04 ? 9 : 10);
    const running = Math.max(0, Math.round(burst * 55 + (rand() - 0.5) * 8));
    const waiting = running > 48 ? randint(0, 6) : 0;
    history.push({t: round(t, 1), ready: routable, routable, running, waiting, kv: round(0.03 + burst * 0.25, 3),
                  gen_tps: round(running * 17 + rand() * 40, 1), req: Math.round(running * 15 + rand() * 60), err: rand() < 0.05 ? randint(1, 3) : 0,
                  p90_s: round(12 + burst * 18 + rand() * 4, 1)});
  }
  return {
    schema: 1, generated_at: NOW,
    daemon: {job_id: '20371655', host: 'cpu1-00096', heartbeat_age_s: 37, status_age_s: 40, breaker: 'closed'},
    config: {target: 10, horizon_effective_min: 150, horizon_min: 45, stagger_min: 40, max_jobs: 24, time: '04:00:00', coverage_partitions: ['batch']},
    coverage: {value: 10, target: 10},
    counts: {ready: 8, draining: 1, starting: 1, pending: 2, held: 0, stale: 0, unhealthy: 0, failed: 0, stopped: 0, vanished: 0, other: 0},
    workers,
    // busy_workers: routable workers with >= 1 running request; vLLM's prompt counter includes prefix-cache hits.
    totals: {routable: 9, busy_workers: 8, scraped: 9, scrape_errors: 1, running: 38, waiting: 3, kv_usage_mean: 0.12,
             prompt_tps_incl_cache: 14097.8, gen_tps: 710.5},
    router: {scope: 'official-run probes', window_s: 600, requests: 812, errors: 3, error_rate: 0.0037, p50_s: 9.1, p90_s: 24.0,
             prompt_tokens: 48123412, completion_tokens: 1234567, clamps: 3, by_worker: by, last_hour: {requests: 4890, errors: 11}},
    history,
  };
}

// ----------------------------------------------------------------------------------------------- api.json

function api() {
  const history = [];
  let spend = 0;
  for (let t = NOW - 2 * 86400; t <= NOW - 240; t += 600) {
    const hour = new Date((t - 7 * 3600) * 1000).getUTCHours();
    const rate = t < NOW - 30 * 3600 ? 0.4 : hour >= 2 && hour <= 23 ? 11.6 : 3;   // the official runs started at 02:30
    spend += rate * (0.6 + rand() * 0.8);
    history.push({t, spend: round(spend, 2), max_budget: 10000.0});
  }
  const scale = 468.75 / spend;
  for (const h of history) h.spend = round(h.spend * scale, 2);
  history[history.length - 1].spend = 468.75;
  const sampled = NOW - 240, eta = Math.round(sampled + (9531.25 / 70.0) * 3600);   // projected from the hub sample time
  return {
    schema: 1, generated_at: NOW,
    hub: {spend: 468.75, max_budget: 10000.0, remaining: 9531.25, status: 200, sampled_at: sampled, key: 'NVAPI_KEY_NEW', error: null,
          exit_code: 0, last_attempt_at: sampled - 6, age_s: 240},
    // window_min_*: the span of hub samples each rate was measured over
    burn: {per_hour_1h: 75.1, per_hour_6h: 70.0, window_min_1h: 58, window_min_6h: 352, eta_exhaust: eta, eta_exhaust_iso: iso(eta)},
    history,
    by_run: [
      {run: 'official1-bandit', kimi: {calls: 9000, cost: 110.15, input_tokens: 812345678, cached_tokens: 701234567, output_tokens: 12345678},
       codex: {sessions: 150, input_tokens: 45678901, cached_tokens: 40123456, output_tokens: 2345678, reasoning_tokens: 1234567}, glm: {calls: 105}},
      {run: 'official1-narwhals', kimi: {calls: 14210, cost: 201.42, input_tokens: 1502345678, cached_tokens: 1301234567, output_tokens: 20345678},
       codex: {sessions: 262, input_tokens: 81234567, cached_tokens: 70123456, output_tokens: 4123456, reasoning_tokens: 2234567}, glm: {calls: 187}},
      {run: 'official2-fd', kimi: {calls: 0, cost: 0.0, input_tokens: 0, cached_tokens: 0, output_tokens: 0},
       codex: {sessions: 0, input_tokens: 0, cached_tokens: 0, output_tokens: 0, reasoning_tokens: 0}, glm: {calls: 0}},
    ],
    scan: {files: 443, read: 18, complete: false, pending_bytes: 12900000, sealed_probes: 6},
    notes: ['kimi cost is hub-reported per call; codex and glm cost is not logged (tokens only)'],
  };
}

// ------------------------------------------------------------------------------------------------- write

function runSummary(doc, extra) {
  const nodes = doc.nodes.filter((n) => n.status !== 'abandoned');
  return Object.assign({run: doc.run, task_id: doc.task_id, repo: doc.repo, job_id: doc.job.job_id, slurm_state: doc.job.slurm_state,
    partition: doc.job.partition, submitted_at: doc.job.submitted_at, started_at: doc.job.started_at, ended_at: doc.job.ended_at,
    status: doc.job.status, stage: doc.job.stage, rounds_started: new Set(nodes.map((n) => n.round)).size,
    accepted: nodes.filter((n) => n.decision === 'accepted' || n.decision === 'accepted_evolve').length,
    tasks_built: nodes.filter((n) => n.build.ok === true).length, tasks_scored: nodes.filter((n) => n.score).length}, extra || {});
}

function main() {
  const docs = {narwhals: narwhals(), bandit: bandit(), fd: fdPending()};
  const index = {
    schema: 1, generated_at: NOW, generated_at_iso: iso(NOW), collect_seconds: 3.2,
    errors: ['run official1-bandit tree: cand-c05-r1b3/gates-0.json: JSONDecodeError at byte 4096 (being written)',
             'fleet: scrape 20409912 http://10.65.31.9:8000/metrics: URLError: [Errno 111] Connection refused'],
    models: MODELS,
    runs: [runSummary(docs.narwhals), runSummary(docs.bandit), runSummary(docs.fd)],
    tasks: TASKS,
  };
  fs.mkdirSync(OUT, {recursive: true});
  const write = (name, value) => fs.writeFileSync(path.join(OUT, name), JSON.stringify(value, null, 1) + '\n');
  write('index.json', index);
  write('task-narwhals-rolling-window-suite.json', docs.narwhals);
  write('task-bandit-incremental-cache-control.json', docs.bandit);
  write('task-fd-deterministic-multi-key-sorting.json', docs.fd);
  write('fleet.json', fleet());
  write('api.json', api());
  const kinds = new Set(docs.narwhals.nodes.map((n) => (n.decision === 'discard' && n.build.ok === false ? 'build_failed' : n.decision)));
  console.log(`fixtures written to ${OUT}; narwhals decisions: ${[...kinds].sort().join(', ')}`);
}

main();
