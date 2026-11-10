'use strict';
/* Frontend on the REAL collector output (D/data, read-only; node:test, no dependencies).

    node --test tests/frontend_real.test.js
    DASHBOARD_DATA=<dir> node --test tests/frontend_real.test.js       (another collector output, e.g. a staged cycle's)

Every pure renderer of static/app.js runs on the live files: index.json (every task's view), each official run's task
file (the tree, the round-0 table, every node card and every node's detail panel), fleet.json and api.json. No renderer
may throw or fall back to its inline error, key elements must be present and agree with the data, the markup may only
contain tags the app itself writes (escaping), the tree layout must hold its invariants, and the decisions of known
nodes of the official runs must show the right badges. The collector's newer fields (NODE.probe.partial / own_done,
NODE.focus_pattern, the teacher_audit list, totals.busy_workers, router.scope, burn windows, the failure-match audit and
its scope NODE.audit_eligible, the v1.5 seed notes NODE.seed_invalid) are checked when the live files have them and
their fallbacks when they do not; runs launched minutes ago (no plan, no candidate yet) render too. A last test drives
createApp() with a fake DOM whose fetch serves D/data at exactly the URLs the page requests. Skips when
D/data/index.json does not exist (no collector cycle yet). v1.6: every run of a batch (Baihe's) renders from its own
document data/run/<source>/<run>.json, never audited, with its in_batch tasks, the run selector of a task with two runs,
and the batch's own totals; the page fetches those documents by their served paths too. v1.7: the live explorer.json is
exactly what the live task and run files give (tests/explorer_fixture.js), every deep link opens its node, the Explorer
renders sane markup and sound numbers fast for many filter combinations, and #explorer fetches it by its served path.
v1.8: every official run's document (an older one's data/run/hengyuf/<run>.json) renders its own view with the run
selector, index.json's official runs agree (one latest per task, official_runs), the Explorer covers every official run,
and the Monitor's synthesis queue (index.json queues) and the picker's "queued in" notes render on the live index.
*/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = require('../static/app.js');

const DATA = process.env.DASHBOARD_DATA ? path.resolve(process.env.DASHBOARD_DATA) : path.join(__dirname, '..', 'data');
const HAVE = fs.existsSync(path.join(DATA, 'index.json'));
const read = (rel) => JSON.parse(fs.readFileSync(path.join(DATA, rel), 'utf8'));
const R = HAVE ? load() : null;
const NOW = HAVE ? R.index.generated_at + 30 : 0;
const opts = {skip: HAVE ? false : 'no collector output in data/ yet'};

function load() {
  const index = read('index.json');
  const details = {};
  for (const t of index.tasks) {
    if (t.run && fs.existsSync(path.join(DATA, 'task', `${t.task_id}.json`))) details[t.task_id] = read(`task/${t.task_id}.json`);
  }
  const batchDocs = {};                                   // v1.6: runs[].detail -> a batch run's document
  for (const run of index.runs.filter((r) => r.source)) batchDocs[run.detail] = read(run.detail);
  const officialDocs = {};                                // v1.8: runs[].detail -> an older official run's document
  for (const run of index.runs.filter((r) => !r.source && r.latest === false)) officialDocs[run.detail] = read(run.detail);
  // v1.7: the Explorer's tables, when the collector names them
  const explorer = index.explorer && fs.existsSync(path.join(DATA, 'explorer.json')) ? read('explorer.json') : null;
  return {index, details, batchDocs, officialDocs, explorer, fleet: read('fleet.json'), api: read('api.json')};
}

const count = (s, needle) => s.split(needle).length - 1;
const esc = app.esc;
// Tags app.js writes; anything else in the output would be data that escaped esc().
const TAGS = new Set(['div', 'span', 'section', 'article', 'aside', 'h1', 'h2', 'h3', 'h4', 'p', 'b', 'code', 'pre',
  'details', 'summary', 'table', 'thead', 'tbody', 'tr', 'th', 'td', 'ul', 'li', 'dl', 'dt', 'dd', 'svg', 'path', 'line',
  'text', 'circle', 'rect', 'g', 'title', 'button', 'figure', 'figcaption',
  'a']);   // v1.8: the queue table links a launched run's view (href="#task=...")

// v1.7: the Explorer's filter bar is made of form controls (never inside a <form>: CSP form-action 'none')
const EX_TAGS = new Set([...TAGS, 'input', 'select', 'option', 'label', 'fieldset', 'legend']);

/** Structural checks shared by every rendered string (tags: the tags allowed, TAGS by default). */
function sane(html, where, tags) {
  assert.equal(typeof html, 'string', `${where}: not a string`);
  assert.ok(!html.includes('render-error'), `${where}: a section fell back to its error box: ` +
    (html.match(/<div class="render-error">[^<]*/) || [''])[0]);
  assert.ok(!html.includes('[object Object]'), `${where}: [object Object]`);
  assert.doesNotMatch(html, /\b(?:style|width|height|x|y|x1|x2|y1|y2|cx|cy|r|d|viewBox|aria-valuenow|aria-valuemax)="[^"]*(?:NaN|undefined|Infinity)/,
    `${where}: NaN/undefined in a layout attribute`);
  for (const tag of html.match(/<[a-zA-Z][^>]*>/g) || []) {
    const name = tag.match(/^<([a-zA-Z0-9]+)/)[1].toLowerCase();
    assert.ok((tags || TAGS).has(name), `${where}: unexpected tag ${tag.slice(0, 80)}`);
    const attrs = tag.replace(/"[^"]*"/g, '""');
    assert.doesNotMatch(attrs, /\son[a-z]+\s*=/i, `${where}: event handler attribute in ${tag.slice(0, 80)}`);
    assert.doesNotMatch(tag, /javascript:/i, `${where}: javascript: URL`);
  }
}

/** The data with every free-text string (whitespace, or longer than 60 chars) replaced by 'text' (or by
    replace(string)): ids, enums, fractions, slots and ISO times stay, so every code path is unchanged, but no document
    can contain "NaNs", and a payload reaches every text field. */
function neutral(value, replace) {
  const swap = replace || (() => 'text');
  if (typeof value === 'string') return /\s/.test(value) || value.length > 60 ? swap(value) : value;
  if (Array.isArray(value)) return value.map((v) => neutral(v, swap));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, neutral(v, swap)]));
  return value;
}

/** Numbers formatted by the renderers (on neutral() data) never read NaN / undefined / Infinity. */
function numbersOk(html, where) {
  assert.doesNotMatch(html, /\$NaN|NaN%|\bNaN\b|\bundefined\b|\bInfinity\b/, `${where}: NaN/undefined/Infinity in a value: ` +
    ((html.match(/.{0,60}(?:\bNaN\b|\bundefined\b|\bInfinity\b).{0,20}/) || [''])[0]));
}

function decisionLabel(node) {
  if (node.decision === 'discard' && ((node.build || {}).ok === false || node.status === 'build_failed')) return 'BUILD FAILED';
  return app.DECISIONS[node.decision].label;
}

// ----------------------------------------------------------------------------------------- header, picker

test('header status, banner and the task picker over the real index', opts, () => {
  const {index} = R;
  const status = app.renderStatus(index, NOW);
  sane(status, 'status');
  assert.match(status, /updated \d\d:\d\d:\d\d<span class="age"> · 30s ago/);
  assert.doesNotMatch(status, /stale/);
  assert.match(app.renderStatus(index, index.generated_at + 600), /stale/);
  const banner = app.renderBanner(index, NOW, {errors: {}});
  sane(banner, 'banner');
  for (const error of index.errors) assert.ok(banner.includes(esc(error)), `collector error not shown: ${error}`);

  const converting = index.tasks.filter((t) => t.run);
  const eligible = index.tasks.filter((t) => t.eligible);
  const batched = index.tasks.filter((t) => (t.batch_runs || []).length);   // v1.6: one group per batch
  const menu = app.renderTaskMenu(index, '', null, 0);
  sane(menu, 'menu');
  assert.equal(count(menu, 'data-opt="'), converting.length + batched.length + eligible.length + index.tasks.length);
  assert.match(menu, new RegExp(`Converting<span>${converting.length}</span>`));
  assert.match(menu, new RegExp(`Eligible queue<span>${eligible.length}</span>`));
  assert.match(menu, new RegExp(`All tasks<span>${index.tasks.length}</span>`));
  for (const b of index.batches || []) {
    const mine = batched.filter((t) => t.batch_runs.some((r) => r.source === b.source));
    assert.ok(menu.includes(`>${esc(b.name)} batch · not audited<span>${mine.length}</span>`), `${b.source}: its picker group`);
  }
  const groups = app.taskGroups(index, '');
  const conv = groups.find((g) => g.key === 'converting'), queue = groups.find((g) => g.key === 'eligible');
  assert.equal(groups[0], conv, 'Converting (the official runs) first');
  assert.deepEqual(queue.items.map((t) => t.queue_rank), eligible.map((t) => t.queue_rank).sort((a, b) => a - b));
  assert.deepEqual(new Set(conv.items.map((t) => t.task_id)), new Set(converting.map((t) => t.task_id)));
  assert.ok(converting.some((t) => t.task_id === app.defaultTask(index)), 'the default task is a converting one');
  for (const t of converting) {
    const hits = app.renderTaskMenu(index, t.run, null, 0);
    assert.ok(hits.includes(`data-task="${esc(t.task_id)}"`), `searching ${t.run} finds ${t.task_id}`);
  }
});

// -------------------------------------------------------------------------------------------- task views

test('every task view: accuracy cards, eligibility, and the conversion state', opts, () => {
  const {index, details} = R;
  for (const t of index.tasks) {
    const html = app.renderTaskView(index, t.task_id, details[t.task_id] || null, {now: NOW, ui: {}});
    sane(html, t.task_id);
    assert.ok(html.includes(`<h1 class="task-id">${esc(t.task_id)}</h1>`), `${t.task_id}: title`);
    assert.equal(count(html, 'class="acc-card'), 3, `${t.task_id}: three accuracy cards`);
    for (const [key, model] of Object.entries(index.models)) {
      assert.ok(html.includes(esc(model.label)), `${t.task_id}: ${key} label`);
      const a = t.accuracy[key];
      if (a) {
        assert.ok(html.includes(`${a.passed}<span class="acc-den">/${a.total}</span>`), `${t.task_id}: ${key} ${a.passed}/${a.total}`);
      }
    }
    const rollouts = ['kimi', 'super_vl', 'sc14'].flatMap((k) => (t.accuracy[k] || {}).rollouts || []);
    assert.equal(count(html, 'class="chip chip-'), rollouts.length, `${t.task_id}: one chip per rollout`);
    const ungraded = rollouts.filter((r) => r.score === null || r.status === 'infra').length;
    assert.equal(count(app.renderAccuracy(index, t), '">–</span>'), ungraded, `${t.task_id}: unscored / infra chips read "–"`);
    const sc14 = t.accuracy.sc14;
    if (sc14 && sc14.mean_f2p_ratio !== null) assert.match(html, /mean F2P ratio <b>/);
    if (t.eligible) {
      assert.match(html, /ELIGIBLE/);
      assert.ok(html.includes(`queue #${t.queue_rank}`), `${t.task_id}: queue rank`);
    } else {
      assert.match(html, /not eligible/);
    }
    if (!t.run && !(t.batch_runs || []).length) {
      assert.match(html, /Not converted yet/);
      assert.doesNotMatch(html, /Generation tree/);
    }
  }
});

test('converting tasks: run bar, failure patterns, plan and generation tree in order', opts, () => {
  const {index, details} = R;
  assert.ok(Object.keys(details).length >= 1, 'at least one official run has a task file');
  for (const [id, d] of Object.entries(details)) {
    const run = index.runs.find((r) => r.run === d.run);
    for (const ui of [{}, {showAbandoned: true, treeWidth: 1900}, {treeWidth: 1100}]) {
      const html = app.renderTaskView(index, id, d, {now: NOW, ui});
      sane(html, `${id} view`);
      const at = (s) => html.indexOf(s);
      // a run launched minutes ago has no plan (and no candidate) yet: its sections are left out, the order stays
      const order = ['panel-task', 'panel-run', 'Target failure patterns', d.plan ? 'panel-plan' : null, 'panel-tree'].filter(Boolean);
      assert.ok(order.every((s, i) => at(s) >= 0 && (i === 0 || at(s) > at(order[i - 1]))), `${id}: section order`);
      assert.ok(html.includes(`<b>${esc(run.run)}</b>`) && html.includes(`<code>${esc(run.repo)}</code>`), `${id}: run bar`);
      assert.ok(html.includes(esc(run.stage)), `${id}: stage`);
      assert.ok(html.includes(`Log tail <span class="muted small">${d.job.log_tail.length} line`), `${id}: log tail`);
      for (const p of (d.failure_analysis || {}).patterns || []) assert.ok(html.includes(esc(p.name)), `${id}: pattern ${p.name}`);
      if (d.plan) {
        const built = d.plan.candidates.filter((c) => c.built).length;
        assert.ok(html.includes(`${d.plan.candidates.length} candidates · ${built} built`), `${id}: plan summary`);
      }
      if (d.nodes.length) {
        for (let k = 0; k <= d.max_rounds; k += 1) assert.ok(html.includes(`Round ${k} <span`), `${id}: round ${k} column`);
      } else {
        assert.ok(html.includes('No candidate tasks yet'), `${id}: an empty tree says so`);
      }
      for (const r of d.rounds) {
        if (r.seeds) {
          const invalid = Object.keys(r.seeds.invalid || {}).length;
          assert.ok(html.includes(`seeds: ${r.seeds.valid.length} valid · `) && (invalid ? html.includes(`>${invalid} invalid</span>`) : html.includes('· 0 invalid')),
                    `${id}: round ${r.index} seeds with the invalid count`);
        }
        // "built x / y" while a round's tasks are built and none is probed yet (with a probe: its slots)
        if (r.status === 'building' && r.nodes && !r.probe) assert.ok(html.includes(`built ${r.built} / ${r.nodes}`), `${id}: round ${r.index} built x / y`);
      }
      if (d.nodes.some((n) => n.round === 0)) {
        assert.ok(at(d.plan ? 'panel-plan' : 'Target failure patterns') < at('Round 0 at a glance') &&
                  at('Round 0 at a glance') < at('panel-tree'), `${id}: round-0 table above the tree`);
      }
      if (html.includes('panel-fa')) {
        assert.ok(at('Round 0 at a glance') < at('panel-fa') && at('panel-fa') < at('panel-tree'), `${id}: audit table above the tree`);
      }
      const abandoned = d.nodes.filter((n) => n.decision === 'abandoned');
      for (const n of d.nodes) {
        const shown = html.includes(`data-node="${esc(n.id)}"`);
        assert.equal(shown, ui.showAbandoned || n.decision !== 'abandoned', `${id}: ${n.id} shown=${shown}`);
      }
      if (abandoned.length) {
        assert.ok(html.includes(`${ui.showAbandoned ? 'Hide' : 'Show'} abandoned (${abandoned.length})`), `${id}: abandoned toggle`);
      }
    }
  }
});

test('tree layout invariants on the real trees', opts, () => {
  for (const [id, d] of Object.entries(R.details)) {
    const byId = new Map(d.nodes.map((n) => [n.id, n]));
    for (const showAbandoned of [false, true]) {
      const L = app.treeLayout(d, {showAbandoned, treeWidth: 1400});
      const roots = d.nodes.filter((n) => n.round === 0);
      assert.equal(L.lineages.length, roots.length, `${id}: one lineage block per round-0 task`);
      assert.equal(L.cols, d.max_rounds + 1, `${id}: one column per round`);
      const placed = [];
      for (const lin of L.lineages) {
        const seen = new Set();
        for (const it of lin.items) {
          assert.equal(it.col, it.node.round, `${id}: ${it.id} sits in its round's column`);
          assert.equal(it.node.lineage, lin.root, `${id}: ${it.id} is in lineage ${lin.root}`);
          const cell = `${it.col}:${it.y}`;
          assert.ok(!seen.has(cell), `${id}: two cards overlap at ${cell}`);
          seen.add(cell);
          placed.push(it.id);
        }
        const pos = new Map(lin.items.map((it) => [it.id, it]));
        for (const e of lin.edges) {
          assert.equal(pos.get(e.to).col, pos.get(e.from).col + 1, `${id}: ${e.to} one column right of ${e.from}`);
          assert.equal(byId.get(e.to).parent, e.from, `${id}: edge ${e.from} -> ${e.to} follows parent`);
        }
        assert.equal(lin.edges.length, lin.items.length - 1, `${id}: ${lin.root} block is a tree`);
      }
      const expect = d.nodes.filter((n) => showAbandoned || n.decision !== 'abandoned').map((n) => n.id).sort();
      assert.deepEqual(placed.sort(), expect, `${id}: every visible node placed exactly once`);
    }
  }
});

test('node cards and node detail panels for every node', opts, () => {
  for (const [id, d] of Object.entries(R.details)) {
    const L = app.treeLayout(d, {showAbandoned: true});
    for (const lin of L.lineages) {
      for (const it of lin.items) {
        const n = it.node, card = app.renderNodeCard(it, {}, L.geom);
        sane(card, `${id} card ${n.id}`);
        assert.ok(card.includes(`data-node="${esc(n.id)}"`), `${n.id}: card id`);
        assert.ok(card.includes(`>${decisionLabel(n)}</span>`), `${n.id}: badge ${decisionLabel(n)}`);
        const b = n.build || {};
        if (b.ok !== false && (b.f2p !== null || b.p2p !== null)) {
          assert.ok(card.includes(`F2P ${app.fmtInt(b.f2p)} / P2P ${app.fmtInt(b.p2p)}`), `${n.id}: F2P/P2P line`);
        }
        const live = app.liveProbe(n);
        if (n.status === 'probing') assert.ok(live, `${n.id}: a probing node has partial results`);
        if (live && live.ownDone) {
          assert.ok(card.includes('probe done, scoring when the round&#39;s probe ends'), `${n.id}: own slots done`);
        } else if (live) {
          for (const r of live.rows.slice(0, 2)) assert.ok(card.includes(esc(app.partialText(r))), `${n.id}: partial ${app.partialText(r)}`);
          assert.doesNotMatch(card, /probing \d+\/\d+/, `${n.id}: no bare progress line`);
        }
        if (live) assert.ok(card.includes('node-pbar') && card.includes(`aria-label="probe progress ${live.done}/${live.total}"`), `${n.id}: slim progress bar`);
        if (n.status === 'scored') {
          const s = app.probeStats(n.probe, 'student'), t = app.probeStats(n.probe, 'teacher');
          assert.ok(card.includes(`sc14 ${s.pass}/${s.valid}${s.infra ? ` +${s.infra} infra` : ''}`) &&
                    card.includes(`kimi ${t.clean}/${t.valid} clean${t.infra ? ` +${t.infra} infra` : ''}`), `${n.id}: rollout accuracy and infra of both models`);
        }
        if (n.focus_pattern) assert.ok(card.includes(`>${esc(n.focus_pattern)}</span>`), `${n.id}: target pattern tag`);
        else assert.ok(!card.includes('tag-focus'), `${n.id}: no target pattern recorded, no tag`);
      }
    }
    for (const n of d.nodes) {
      const html = app.renderNodeDetail(n, d, {}, NOW);
      sane(html, `${id} detail ${n.id}`);
      assert.ok(html.includes(`<div class="drawer-id">${esc(n.id)}</div>`), `${n.id}: header`);
      assert.ok(html.includes(`<b>${decisionLabel(n)}</b>`), `${n.id}: decision`);
      assert.ok(html.includes(esc(n.decision_reason)), `${n.id}: decision reason`);
      for (const h of ['Build &amp; gates', '>Probe<', '>Score<', '>Documents<']) assert.ok(html.includes(h), `${n.id}: ${h}`);
      const rollouts = (n.probe || {}).rollouts || [];
      for (const r of rollouts) assert.ok(html.includes(`<td><code>${esc(r.slot)}</code></td>`), `${n.id}: rollout ${r.slot}`);
      if (n.score) {
        assert.ok(html.includes('<dt>Accepted</dt>'), `${n.id}: score table`);
        if (n.score.similarity_detail) assert.match(html, /Similarity detail/);
      } else {
        assert.match(html, /Not scored yet/);
      }
      for (const [key, value] of Object.entries(n.details || {})) {
        if (value) assert.ok(html.includes(`data-key="nd:${esc(n.id)}:${key}"`), `${n.id}: document ${key}`);
      }
      const audit = (n.details || {}).teacher_audit;
      if (Array.isArray(audit)) {
        assert.ok(html.includes('<th>slot</th><th>cheated</th><th>categories</th><th>summary</th>') || !audit.length, `${n.id}: audit table`);
        for (const a of audit) assert.ok(html.includes(`<tr><td><code>${esc(a.slot)}</code></td>`), `${n.id}: audit row ${a.slot}`);
      } else if (audit) {
        assert.ok(html.includes(`data-scroll="nd:${esc(n.id)}:teacher_audit"`), `${n.id}: audit as text`);
      }
      for (const r of rollouts) {
        const err = [r.agent_error && `agent: ${r.agent_error}`, r.verifier_error && `verifier: ${r.verifier_error}`].filter(Boolean).join('\n');
        if (err) assert.ok(html.includes(esc(err)), `${n.id}: the full error of ${r.slot}`);
      }
      for (const child of n.children) assert.ok(html.includes(`data-node="${esc(child)}"`), `${n.id}: child link ${child}`);
    }
    // Round 0 at a glance: one row per round-0 task, each opening its drawer.
    const table = app.renderRoundZero(d, {});
    sane(table, `${id} round 0`);
    const roots = d.nodes.filter((n) => n.round === 0);
    assert.equal(count(table, '<tr class="r0-row'), roots.length, `${id}: one row per round-0 task`);
    for (const n of roots) {
      const start = table.indexOf(`<tr class="r0-row" data-node="${esc(n.id)}">`);
      assert.ok(start >= 0, `${id}: row ${n.id}`);
      // the decision badge, then (v1.5) a seed-invalid badge for an accepted invalid seed (2026-10-22: scriggo
      // cand-c01), then the reason
      const row = table.slice(start, table.indexOf('</tr>', start));
      const decision = row.match(/<td><span class="badge[^"]*"[^>]*><span class="dot" aria-hidden="true"><\/span>([^<]*)<\/span>/);
      assert.equal(decision && decision[1], decisionLabel(n), `${id}: ${n.id} decision`);
      assert.ok(row.includes(`<td class="r0-reason">${esc(n.decision_reason)}</td>`), `${id}: ${n.id} decision reason`);
      assert.equal(row.includes('>seed-invalid</span>'), !!(n.seed_invalid && n.decision !== 'discard'), `${id}: ${n.id} seed note`);
    }
  }
});

test('formatted values never read NaN, undefined or Infinity (data text neutralized)', opts, () => {
  const index = neutral(R.index), fleet = neutral(R.fleet), api = neutral(R.api);
  numbersOk(app.renderStatus(index, NOW) + app.renderTaskMenu(index, '', null, 0), 'header');
  for (const t of index.tasks) {
    const d = R.details[t.task_id] ? neutral(R.details[t.task_id]) : null;
    numbersOk(app.renderTaskView(index, t.task_id, d, {now: NOW, ui: {showAbandoned: true}}), t.task_id);
    for (const n of (d || {}).nodes || []) numbersOk(app.renderNodeDetail(n, d, {}, NOW), `${t.task_id} ${n.id}`);
  }
  numbersOk(app.renderMonitor(fleet, api, {now: NOW, ui: {}, index}), 'monitor');
  numbersOk(app.renderApi(api, NOW, {spendScale: 'budget'}), 'api');
});

test('escaping: a markup payload in every free-text string of the real data never becomes markup', opts, () => {
  const payload = (v) => `${v} <img src=x onerror="alert(1)"><script>alert(2)</script>' "&`;
  const index = neutral(R.index, payload), fleet = neutral(R.fleet, payload), api = neutral(R.api, payload);
  sane(app.renderStatus(index, NOW) + app.renderBanner(index, NOW, {}) + app.renderTaskMenu(index, '', null, 0), 'header');
  for (const t of index.tasks) {
    const d = R.details[t.task_id] ? neutral(R.details[t.task_id], payload) : null;
    sane(app.renderTaskView(index, t.task_id, d, {now: NOW, ui: {showAbandoned: true}}), `payload ${t.task_id}`);
    for (const n of (d || {}).nodes || []) sane(app.renderNodeDetail(n, d, {}, NOW), `payload ${t.task_id} ${n.id}`);
  }
  sane(app.renderMonitor(fleet, api, {now: NOW, ui: {}, index}), 'payload monitor');
});

// ------------------------------------------------------------------- known facts of the official runs (step 5)

test('official runs: the known decisions show the right badges', opts, (t) => {
  const nar = R.details['narwhals-rolling-window-suite'], ban = R.details['bandit-incremental-cache-control'];
  if (!nar && !ban) { t.skip('neither official run is in data/'); return; }
  const cardOf = (d, nodeId) => {
    const L = app.treeLayout(d, {showAbandoned: true});
    for (const lin of L.lineages) for (const it of lin.items) if (it.id === nodeId) return app.renderNodeCard(it, {}, L.geom);
    return null;
  };
  if (nar) {
    const by = new Map(nar.nodes.map((n) => [n.id, n]));
    assert.equal(by.get('cand-c02').decision, 'discard');
    assert.match(cardOf(nar, 'cand-c02'), />DISCARD</);
    assert.equal(by.get('cand-c04').decision, 'accepted_evolve');
    assert.match(cardOf(nar, 'cand-c04'), />ACCEPTED\+EVOLVE</);
    const c02kids = nar.nodes.filter((n) => n.parent === 'cand-c02');
    assert.equal(c02kids.length, 5);
    assert.ok(c02kids.every((n) => n.decision === 'abandoned'), 'cand-c02 children abandoned');
    const c04kids = nar.nodes.filter((n) => n.parent === 'cand-c04');
    assert.equal(c04kids.length, 5);
    assert.ok(c04kids.every((n) => n.decision !== 'abandoned'), 'cand-c04 children present');
    const html = app.renderTaskView(R.index, 'narwhals-rolling-window-suite', nar, {now: NOW, ui: {}});
    assert.ok(!html.includes('data-node="cand-c02-r1b0"') && html.includes('data-node="cand-c04-r1b0"'));
    assert.match(html, /1 invalid/);                         // round-0 seeds: cand-c02
  }
  if (ban) {
    const by = new Map(ban.nodes.map((n) => [n.id, n]));
    for (const id of ['cand-c03-r1b2', 'cand-c04-r1b4']) {
      assert.equal(by.get(id).decision, 'accepted', id);
      assert.match(cardOf(ban, id), />ACCEPTED</);
    }
    assert.ok(ban.nodes.some((n) => n.round === 2), 'round-2 nodes present');
    const c05 = ban.nodes.filter((n) => n.lineage === 'cand-c05');
    assert.ok(c05.some((n) => n.round === 1) && c05.some((n) => n.round === 2), 'cand-c05 lineage spans rounds 1 and 2');
    const seeds = ban.nodes.filter((n) => n.round === 1 && n.decision === 'evolve').map((n) => n.id).sort();
    const parents = [...new Set(ban.nodes.filter((n) => n.round === 2).map((n) => n.parent))].sort();
    assert.deepEqual(seeds, parents, 'round-1 EVOLVE nodes are exactly the round-2 parents');
  }
});

test('failure-match audit: the scope, the cards, the table and the run bar agree with the real data', opts, (t) => {
  const docs = Object.entries(R.details).filter(([, d]) => d.nodes.some((n) => 'failure_match' in n));
  if (!docs.length) { t.skip('no failure-match audit in data/ (a collector before v1.3)'); return; }
  const accepted = (n) => n.decision === 'accepted' || n.decision === 'accepted_evolve';
  for (const [id, d] of docs) {
    const run = R.index.runs.find((r) => r.run === d.run) || {};
    const v14 = d.nodes.some((n) => 'audit_eligible' in n);          // v1.4: the eligible tasks; v1.3: the accepted ones
    const scope = d.nodes.filter((n) => (v14 ? n.audit_eligible === true : accepted(n)));
    const table = app.renderAuditTable(d, {});
    sane(table, `${id} audit table`);
    const rows = [...table.matchAll(/<tr class="fa-row[^"]*" data-node="([^"]+)">/g)].map((m) => m[1]);
    assert.deepEqual([...rows].sort(), scope.map((n) => esc(n.id)).sort(), `${id}: one row per task in the audit's scope`);
    const first = rows.findIndex((r) => !accepted(d.nodes.find((n) => esc(n.id) === r)));
    assert.ok(first === -1 || rows.slice(first).every((r) => !accepted(d.nodes.find((n) => esc(n.id) === r))), `${id}: accepted rows first`);
    if (scope.length) {
      const audited = scope.filter((n) => n.failure_match).length;
      assert.ok(table.includes(`${app.fmtInt(scope.length)} ${v14 ? 'eligible' : 'accepted'} task${scope.length === 1 ? '' : 's'} · ` +
                               `audit ${app.fmtInt(audited)}/${app.fmtInt(scope.length)}`), `${id}: table head`);
    } else {
      assert.equal(table, '', `${id}: no table without a task in the scope`);
    }
    const L = app.treeLayout(d, {showAbandoned: true});
    for (const lin of L.lineages) {
      for (const it of lin.items) {
        const want = v14 ? it.node.audit_eligible === true : accepted(it.node);
        assert.equal(app.renderNodeCard(it, {}, L.geom).includes('badge-fa'), want, `${id}: ${it.node.id} FA badge`);
      }
    }
    const bar = app.renderRunBar(run, d, NOW, {});
    sane(bar, `${id} run bar`);
    const a = run.audit || {};
    if (v14 && 'eligible' in a) {
      assert.equal(a.eligible, scope.length, `${id}: index runs[].audit.eligible = the eligible nodes`);
      assert.ok(bar.includes(`audit ${app.fmtInt(a.audited_eligible)}/${app.fmtInt(a.eligible)}</b>`), `${id}: run bar, eligible`);
      assert.ok(bar.includes(`accepted: audit ${app.fmtInt(a.audited)}/${app.fmtInt(a.accepted)}`), `${id}: run bar, accepted`);
    }
    for (const n of scope.filter((x) => !x.failure_match).slice(0, 3)) {
      assert.match(app.renderNodeDetail(n, d, {}, NOW), /Not audited yet: no failure-match audit of this (?:eligible|accepted) task/, `${id}: ${n.id}`);
    }
    for (const n of d.nodes.filter((x) => x.withdrawn)) {
      assert.ok(!accepted(n) && n.audit_eligible !== true, `${id}: ${n.id} withdrawn, so discarded`);
      assert.match(app.renderNodeDetail(n, d, {}, NOW), /<div class="callout-k">Withdrawn<\/div>/, `${id}: ${n.id} withdrawn callout`);
    }
  }
});

test('seed notes (v1.5): accepted tasks seeds.json lists invalid read ACCEPTED plus "seed-invalid"; wasmi cand-c03-r1b2', opts, (t) => {
  const docs = Object.entries(R.details).filter(([, d]) => d.nodes.some((n) => 'seed_invalid' in n));
  if (!docs.length) { t.skip('no seed_invalid in data/ (a collector before v1.5)'); return; }
  const accepted = (n) => n.decision === 'accepted' || n.decision === 'accepted_evolve';
  const oneLine = (s) => String(s).split(/\s+/).filter(Boolean).join(' ');
  for (const [id, d] of docs) {
    const L = app.treeLayout(d, {showAbandoned: true});
    for (const lin of L.lineages) {
      for (const it of lin.items) {
        const n = it.node, note = app.seedNote(n);
        const card = app.renderNodeCard(it, {}, L.geom, app.treeContext(d));
        assert.equal(card.includes('badge-seed'), !!note, `${id}: ${n.id} seed-invalid badge`);
        if (!note) continue;
        assert.ok(n.decision === 'accepted' || n.decision === 'abandoned', `${id}: ${n.id} a seed note with ${n.decision}`);
        const drawer = app.renderNodeDetail(n, d, {}, NOW);
        sane(drawer, `${id} ${n.id} drawer`);
        assert.ok(drawer.includes(`<div class="callout-k">Seed-invalid</div><p>${esc(note)}</p>`), `${id}: ${n.id} Seed-invalid callout`);
      }
    }
    for (const r of d.rounds) {                     // an accepted task its round's seeds.json lists invalid: the reason is its note
      for (const [cand, why] of Object.entries((r.seeds || {}).invalid || {})) {
        const n = d.nodes.find((x) => x.id === cand);
        if (n && accepted(n)) assert.equal(app.seedNote(n), oneLine(why), `${id}: ${cand} accepted, its seeds.json reason as the note`);
      }
    }
  }
  const was = R.details['wasmi-trap-coredumps'];
  const node = was && was.nodes.find((n) => n.id === 'cand-c03-r1b2');
  if (node && 'seed_invalid' in node && !node.withdrawn) {      // official1-wasmi, 2026-10-21: the case that made v1.5
    assert.equal(node.decision, 'accepted');
    assert.equal(node.seed_invalid, 'verifier produced no grade for 1 rollout(s): kimi-r3');
    const L = app.treeLayout(was, {showAbandoned: true});
    const it = L.lineages.flatMap((lin) => lin.items).find((x) => x.id === 'cand-c03-r1b2');
    assert.match(app.renderNodeCard(it, {}, L.geom, app.treeContext(was)),
                 /<span class="node-badges"><span class="badge tone-pending badge-xs badge-seed" title="seed-invalid: verifier produced no grade for 1 rollout\(s\): kimi-r3\n[^"]*">[^]*?seed-invalid<\/span><span class="badge tone-good badge-sm"><span class="dot" aria-hidden="true"><\/span>ACCEPTED<\/span><\/span>/);
    const run = R.index.runs.find((r) => r.run === was.run) || {};
    const shown = was.nodes.filter((n) => accepted(n) && n.status !== 'abandoned').length;
    assert.equal(run.accepted, shown, 'official1-wasmi: runs[].accepted counts the accepted nodes');
    if (run.status === 'finished') assert.equal(run.stage, `finished (${shown} accepted)`);
  }
});

// ------------------------------------------------------------------------------------------------ monitor

test('batch runs (v1.6): each renders from its own document, never audited, with its batch tasks', opts, (t) => {
  const {index, batchDocs} = R;
  const runs = index.runs.filter((r) => r.source);
  if (!runs.length) { t.skip('no batch in this collector output'); return; }
  const payload = (v) => `${v} <img src=x onerror="alert(1)"><script>alert(2)</script>' "&`;
  for (const b of index.batches) {                          // the batch's own totals, apart from the official runs'
    const mine = runs.filter((r) => r.source === b.source);
    assert.equal(b.runs, mine.length, `${b.source}: runs`);
    const flagged = mine.reduce((sum, r) => sum + batchDocs[r.detail].nodes.filter((n) => n.in_batch).length, 0);
    assert.equal(b.in_batch, flagged, `${b.source}: in_batch nodes`);
    assert.deepEqual(b.unmatched, [], `${b.source}: every task of the batch is an accepted node`);
  }
  // a data string that quotes "[object Object]" itself (q21's synthetic failure analysis quotes a test log) would trip
  // sane()'s check for objects printed as text: such strings are replaced first
  const literal = (v) => (typeof v === 'string' ? (v.includes('[object Object]') ? 'text' : v)
    : Array.isArray(v) ? v.map(literal) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, literal(x)])) : v);
  for (const run of runs) {
    const doc = literal(batchDocs[run.detail]), name = app.batchName(index, run), where = `${run.source} ${run.run}`;
    const task = index.tasks.find((x) => x.task_id === run.task_id);
    const views = app.viewsOf(index, task);
    const inBatch = doc.nodes.filter((n) => n.in_batch);
    assert.equal(inBatch.length, run.batch, `${where}: in_batch nodes = runs[].batch`);
    assert.ok(inBatch.every((n) => n.decision === 'accepted' || n.decision === 'accepted_evolve'), `${where}: batch tasks accepted`);
    for (const ui of [{}, {showAbandoned: true, treeWidth: 1900}, {treeWidth: 1100}]) {
      const html = app.renderTaskView(index, run.task_id, doc, {now: NOW, ui, run: run.run});
      sane(html, `${where} view`);
      assert.ok(html.includes(`<b>${esc(run.run)}</b>`) && html.includes(esc(run.stage)), `${where}: run bar`);
      assert.ok(html.includes(`${esc(name)} batch · not audited</span>`), `${where}: the batch badge`);
      assert.match(html, /its jobs are not polled/);
      assert.match(html, /Log not read/);
      assert.doesNotMatch(html, /FA match|Failure-match audit<\/h2>|Not audited yet/, `${where}: never audited`);
      assert.equal(count(html, 'data-run-select="'), views.length > 1 ? views.length : 0, `${where}: run selector`);
      if (views.length > 1) assert.ok(html.includes(`class="run-tab active" data-run-select="${esc(run.run)}"`), `${where}: selected`);
      assert.equal(count(html, 'class="fa-row'), doc.nodes.filter((n) => n.in_batch || ((n.decision === 'accepted' || n.decision === 'accepted_evolve') && n.decision !== 'abandoned')).length,
                   `${where}: one batch-table row per batch task`);
      const shownBatch = inBatch.filter((n) => ui.showAbandoned || n.decision !== 'abandoned').length;
      assert.equal(count(html, '>batch · not audited</span>'), shownBatch, `${where}: the cards of its batch tasks`);
      for (const r of doc.rounds) {
        if ((r.unfinished_probes || []).length) assert.ok(html.includes(`${r.unfinished_probes.length} unfinished probe`), `${where}: round ${r.index}`);
      }
    }
    for (const n of doc.nodes) {
      const drawer = app.renderNodeDetail(n, doc, {}, NOW);
      sane(drawer, `${where} drawer ${n.id}`);
      assert.doesNotMatch(drawer, /FA match|Not audited yet/);
      assert.equal(drawer.includes(`<div class="callout-k">${esc(name)} batch</div>`), !!n.in_batch, `${where} ${n.id}: batch callout`);
      if (n.score) assert.ok(drawer.includes(`Not audited: ${esc(name)}&#39;s batch is not audited`), `${where} ${n.id}: not audited`);
    }
    // (data strings that name such words themselves, e.g. pilot9's file remark-lint-no-undefined-references, go too)
    const words = (v) => (typeof v === 'string' ? (/\b(?:NaN|undefined|Infinity)\b/.test(v) ? 'text' : v)
      : Array.isArray(v) ? v.map(words) : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, words(x)])) : v);
    const quiet = words(neutral(doc)), loud = neutral(doc, payload), ix = words(neutral(index)), loudIx = neutral(index, payload);
    numbersOk(app.renderTaskView(ix, run.task_id, quiet, {now: NOW, ui: {showAbandoned: true}, run: run.run}), `${where} numbers`);
    for (const n of quiet.nodes) numbersOk(app.renderNodeDetail(n, quiet, {}, NOW), `${where} ${n.id} numbers`);
    sane(app.renderTaskView(loudIx, run.task_id, loud, {now: NOW, ui: {showAbandoned: true}, run: run.run}), `${where} payload`);
    for (const n of loud.nodes) sane(app.renderNodeDetail(n, loud, {}, NOW), `${where} ${n.id} payload`);
  }
  // official tasks never show a batch's markup; (v1.8) one with two official runs has its run selector
  for (const [id, d] of Object.entries(R.details)) {
    const task = index.tasks.find((x) => x.task_id === id);
    if ((task.batch_runs || []).length) continue;
    const html = app.renderTaskView(index, id, d, {now: NOW, ui: {}});
    assert.doesNotMatch(html, /batch · not audited|batch-strip/, id);
    const views = app.viewsOf(index, task).length;
    assert.equal(count(html, 'data-run-select="'), views > 1 ? views : 0, id);
  }
});

test("Baihe's batch (v1.6, 2026-10-22): its known runs and totals", opts, (t) => {
  const baihe = (R.index.batches || []).find((b) => b.source === 'baiheh');
  if (!baihe) { t.skip("no Baihe batch in this collector output"); return; }
  assert.equal(baihe.name, 'Baihe');
  assert.ok(baihe.tasks >= 97 && baihe.in_batch === baihe.tasks && baihe.accepted >= baihe.in_batch, JSON.stringify(baihe));
  const tengo = R.index.tasks.find((x) => x.task_id === 'tengo-callable-instance-isolation');
  assert.deepEqual(tengo.batch_runs.map((b) => b.run).sort(), ['pilot10-tengo-golua-inline', 'pilot7-tengo-callable-isolation']);
  const html = app.renderTaskView(R.index, tengo.task_id, null, {now: NOW, ui: {}});
  assert.equal(count(html, 'data-run-select="'), 2);
  const q21 = R.index.runs.find((r) => r.run === 'q21-boa-hierarchical-evaluation-cancellation');
  if (q21) assert.equal(q21.status, 'stopped');
  const menu = app.renderTaskMenu(R.index, '', null, 0);
  assert.match(menu, /Baihe batch · not audited<span>\d+<\/span>/);
});

test('monitor: fleet tiles, worker table, 24 h charts and the API panel', opts, () => {
  const {fleet, api, index} = R;
  const html = app.renderMonitor(fleet, api, {now: NOW, ui: {}});
  sane(html, 'monitor');
  for (const h of ['Inference workers', 'Workers', 'Fleet, last 24 h', 'Hub API usage', 'Usage by run']) {
    assert.ok(html.includes(h), `monitor: ${h}`);
  }
  assert.ok(html.includes(`${app.fmtInt(fleet.totals.routable)}<span class="tile-den">/${app.fmtInt(fleet.config.target)}</span>`), 'routable/target tile');
  // Utilization: busy / routable workers (the collector's busy_workers, else counted from the workers).
  const util = app.busyWorkers(fleet);
  if (typeof fleet.totals.busy_workers === 'number') assert.equal(util.busy, fleet.totals.busy_workers);
  else {
    const serving = fleet.workers.filter((w) => w.routing === 'ready' || w.routing === 'draining');
    const scraped = serving.filter((w) => w.metrics && typeof w.metrics.running === 'number');
    assert.equal(util.busy, scraped.length ? scraped.filter((w) => w.metrics.running >= 1).length : null, 'busy workers from workers[]');
  }
  assert.ok(html.includes(`Busy workers</div><div class="tile-value">${app.fmtInt(util.busy)}<span class="tile-den">/${app.fmtInt(util.routable)}</span>`), 'busy workers tile');
  assert.ok(html.includes('tok/s (incl. prefix-cache hits)'), 'prompt tok/s says it counts cache hits');
  assert.ok(html.includes(`${esc(fleet.router.scope || 'official-run probes')} · `), 'router stats name their scope');
  if (api.scan) {
    assert.ok(html.includes(api.scan.complete ? 'usage scan complete' : 'usage scan catching up'), 'scan health');
  }
  for (const [w, label] of [['window_min_1h', 'Burn (1 h)'], ['window_min_6h', 'Burn (6 h)']]) {
    assert.ok(html.includes(label), label);
    if (typeof api.burn[w] === 'number') assert.ok(html.includes(`over the last ${app.fmtDur(api.burn[w] * 60)} of hub samples`), `${w} shown`);
  }
  if (typeof api.hub.sampled_at === 'number') assert.ok(html.includes(`hub sample ${app.fmtDur(NOW - api.hub.sampled_at)} old`), 'hub sample age');
  const rows = app.renderWorkers(fleet, NOW).split('<tr>').slice(2);       // [0] before the table, [1] the header
  for (const w of fleet.workers) {
    assert.ok(html.includes(`>${esc(w.job_id)}</code>`), `worker ${w.job_id}`);
    // The daemon's bucket (student/fleet.py): a ready/draining record is its routing, or stale when not routable.
    const serving = ['ready', 'draining'].includes(w.registry_status);
    const expect = serving ? (w.routing || 'stale') : (w.registry_status || (w.state === 'PENDING' ? 'pending' : String(w.state).toLowerCase()));
    const row = rows.find((r) => r.includes(`>${esc(w.job_id)}</code>`));
    assert.ok(row && row.includes(`</span>${expect}</span>`), `worker ${w.job_id} shows ${expect}`);
  }
  assert.equal(count(html, 'class="chart-svg"'), 5, 'four fleet charts and the spend chart');
  assert.ok(html.includes(app.fmtMoney(api.hub.spend)), 'hub spend');
  for (const r of api.by_run) assert.ok(html.includes(`<b>${esc(r.run)}</b>`), `usage row ${r.run}`);
  if (api.by_run.length > 1) assert.match(html, /row-total/);
  for (const n of api.notes) assert.ok(html.includes(esc(n)), 'note');
  if (api.burn.eta_exhaust_iso) {                          // collect.py projects with the higher of the two rates
    const used = (api.burn.per_hour_1h || 0) >= (api.burn.per_hour_6h || 0) ? '1 h' : '6 h';
    assert.ok(html.includes(`at the ${used} burn`), 'ETA names the rate it used');
  }
  const runs = new Set(index.runs.map((r) => r.run));
  assert.ok(api.by_run.every((r) => runs.has(r.run)), 'usage only for index runs');
  sane(app.renderApi(api, NOW, {spendScale: 'budget'}), 'api budget scale');
  sane(app.renderMonitor(fleet, api, {now: NOW + 3 * 3600, ui: {}}), 'monitor 3 h later');
});

// ------------------------------------------------------------------------------- wiring over the real files

class ClassList {
  constructor() { this.set = new Set(); }
  add(c) { this.set.add(c); }
  remove(c) { this.set.delete(c); }
  contains(c) { return this.set.has(c); }
  toggle(c, force) { const on = force === undefined ? !this.set.has(c) : !!force; if (on) this.set.add(c); else this.set.delete(c); return on; }
}

class Element {
  constructor(doc, id, tag, parent) {
    Object.assign(this, {doc, id, tagName: tag || 'DIV', parent: parent || null, innerHTML: '', value: '', hidden: false,
                         dataset: {}, attrs: {}, listeners: {}, scrollTop: 0, scrollLeft: 0, clientWidth: 1500});
    this.classList = new ClassList();
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  matches(sel) {
    const attr = sel.match(/^\[data-([a-z-]+)\]$/);
    if (attr) return attr[1] in this.dataset;
    if (sel.startsWith('.')) return this.classList.contains(sel.slice(1));
    return this.tagName.toLowerCase() === sel.toLowerCase();
  }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
  focus() { this.doc.activeElement = this; }
  blur() { if (this.doc.activeElement === this) this.doc.activeElement = this.doc.body; }
  querySelector() { return null; }
  scrollIntoView() {}
}

/** A page whose fetch serves D/data under the relative URLs app.js uses (anything else: 404). */
function page() {
  const doc = {listeners: {}, title: '', visibilityState: 'visible', elements: {}};
  doc.body = new Element(doc, 'body', 'BODY');
  doc.activeElement = doc.body;
  doc.getElementById = (id) => doc.elements[id] || null;
  doc.addEventListener = (type, fn) => { (doc.listeners[type] = doc.listeners[type] || []).push(fn); };
  doc.dispatch = (type, ev) => { for (const fn of doc.listeners[type] || []) fn(Object.assign({preventDefault() {}}, ev)); };
  for (const id of ['task-input', 'task-menu', 'tab-task', 'tab-monitor', 'tab-explorer', 'status', 'refresh', 'banner', 'main',
                    'drawer', 'ex-results', 'ex-tlist', 'ex-tip', 'ex-head', 'ex-st-nj', 'ex-st-s', 'ex-st-n', 'ex-st-k', 'ex-st-targets']) {
    doc.elements[id] = new Element(doc, id, id === 'task-input' ? 'INPUT' : 'DIV', doc.body);
  }
  const win = {listeners: {}};
  let hash = '';
  win.location = {pathname: '/', search: '', get hash() { return hash; },
                  set hash(v) { hash = v.startsWith('#') ? v : `#${v}`; for (const fn of win.listeners.hashchange || []) fn({}); }};
  win.history = {replaceState: (s, t, url) => { const m = String(url).match(/(#.*)$/); hash = m ? m[1] : ''; }};
  win.addEventListener = (type, fn) => { (win.listeners[type] = win.listeners[type] || []).push(fn); };
  win.setInterval = () => 0; win.setTimeout = () => 0; win.clearInterval = () => {}; win.clearTimeout = () => {};
  const calls = [];
  const fetch = async (url, init) => {
    calls.push(url);
    assert.equal(init.credentials, 'same-origin');
    assert.equal(init.cache, 'no-cache');
    const rel = String(url).replace(/^data\//, '');
    const file = path.join(DATA, decodeURIComponent(rel));
    if (!String(url).startsWith('data/') || !file.startsWith(DATA + path.sep) || !fs.existsSync(file)) {
      return {status: 404, ok: false, json: async () => ({})};
    }
    const body = fs.readFileSync(file, 'utf8');
    return {status: 200, ok: true, json: async () => JSON.parse(body)};
  };
  return {env: {window: win, document: doc, fetch, now: () => NOW}, win, doc, calls, el: (id) => doc.elements[id]};
}

async function settle(a) { for (let i = 0; i < 6; i += 1) { await a.idle(); await new Promise((r) => setImmediate(r)); } }

test('wiring: the page fetches the real files by their served paths and renders every view', opts, async () => {
  const p = page();
  const a = app.createApp(p.env);
  await a.start();
  await settle(a);
  const first = app.defaultTask(R.index);
  assert.equal(p.win.location.hash, app.taskHash(first));
  assert.ok(p.calls.includes('data/index.json') && p.calls.includes(`data/task/${encodeURIComponent(first)}.json`), p.calls.join(', '));
  sane(p.el('main').innerHTML, 'main (default task)');
  assert.match(p.el('main').innerHTML, /Generation tree/);
  sane(p.el('status').innerHTML, 'header status');

  // Each converting task, then a node's drawer.
  for (const id of Object.keys(R.details)) {
    p.win.location.hash = app.taskHash(id);
    await settle(a);
    assert.ok(p.calls.includes(`data/task/${encodeURIComponent(id)}.json`), `fetched ${id}`);
    assert.ok(p.el('main').innerHTML.includes(`<h1 class="task-id">${esc(id)}</h1>`), `shows ${id}`);
    const node = R.details[id].nodes.find((n) => n.decision !== 'abandoned');
    if (!node) continue;                                      // a run without candidates yet: no drawer to open
    a.selectNode(node.id);
    assert.equal(p.el('drawer').hidden, false);
    assert.ok(p.el('drawer').innerHTML.includes(esc(node.id)), `drawer for ${node.id}`);
    sane(p.el('drawer').innerHTML, `drawer ${node.id}`);
    a.selectNode(node.id);                                    // toggles closed
  }
  // v1.6: a task of the batch: its first run's document; the hash's run= shows another one
  for (const t of R.index.tasks.filter((x) => (x.batch_runs || []).length).slice(0, 3)) {
    const runs = t.batch_runs.map((b) => R.index.runs.find((r) => r.source === b.source && r.run === b.run));
    for (const run of runs) {
      // v1.8: run= unless this run is the task's default view (a task with an official run opens on it)
      p.win.location.hash = app.taskHash(t.task_id, null, app.viewsOf(R.index, t)[0].run.run === run.run ? null : run.run);
      await settle(a);
      assert.ok(p.calls.includes(`data/${run.detail}`), `fetched ${run.detail}`);
      const html = p.el('main').innerHTML;
      sane(html, `batch view ${run.run}`);
      assert.ok(html.includes(`<b>${esc(run.run)}</b>`) && /Generation tree/.test(html), `shows ${run.run}`);
      const node = R.batchDocs[run.detail].nodes.find((n) => n.in_batch);
      a.selectNode(node.id);
      assert.ok(p.el('drawer').innerHTML.includes(`batch</div>`), `${run.run} ${node.id}: the batch callout`);
      sane(p.el('drawer').innerHTML, `drawer ${run.run} ${node.id}`);
      a.selectNode(node.id);
    }
  }
  // A task without a run: no task file is requested.
  const plain = R.index.tasks.find((t) => !t.run && !(t.batch_runs || []).length);
  const before = p.calls.length;
  p.win.location.hash = app.taskHash(plain.task_id);
  await settle(a);
  assert.ok(p.calls.slice(before).every((u) => !u.startsWith('data/task/')), 'no task file for a task without a run');
  assert.match(p.el('main').innerHTML, /Not converted yet/);

  // The monitor tab.
  p.win.location.hash = '#monitor';
  await settle(a);
  assert.ok(p.calls.includes('data/fleet.json') && p.calls.includes('data/api.json'));
  sane(p.el('main').innerHTML, 'main (monitor)');
  assert.match(p.el('main').innerHTML, /Hub API usage/);
  assert.equal(p.el('banner').innerHTML.includes('Refresh failed'), false, 'every fetch succeeded');
  a.stop();
});

// ------------------------------------------------------------------------------------------- explorer (v1.7)

const {explorerOf} = require('./explorer_fixture.js');
const exOpts = {skip: !HAVE ? 'no collector output in data/ yet' : !(R && R.explorer) ? 'no explorer.json (a collector before v1.7)' : false};

/** The documents the Explorer covers, in its runs[] order: each official run's (v1.8: the latest's task file, an older
    one's data/run/hengyuf/<run>.json), then each batch run's. */
function explorerEntries() {
  const official = R.index.runs.filter((r) => !r.source).map((r) => (r.latest === false
    ? (R.officialDocs[r.detail] ? {source: 'hengyuf', detail: r.detail, doc: R.officialDocs[r.detail], latest: false} : null)
    : (R.details[r.task_id] && R.details[r.task_id].run === r.run ? {source: 'hengyuf', detail: `task/${r.task_id}.json`,
                                                                    doc: R.details[r.task_id]} : null))).filter(Boolean);
  const batch = R.index.runs.filter((r) => r.source && R.batchDocs[r.detail])
    .map((r) => ({source: r.source, detail: r.detail, doc: R.batchDocs[r.detail]}));
  return official.concat(batch);
}

test('explorer (v1.7): explorer.json holds exactly the rows the task and run files give (every field, recounted)', exOpts, () => {
  const ex = R.explorer, built = explorerOf(explorerEntries(), ex.generated_at, ex.sources);
  for (const key of ['runs', 'targets', 'task_columns', 'trajectory_columns', 'totals']) assert.deepEqual(ex[key], built[key], key);
  assert.equal(ex.tasks.length, built.tasks.length, 'task rows');
  assert.equal(ex.trajectories.length, built.trajectories.length, 'trajectory rows');
  for (const [i, row] of ex.tasks.entries()) assert.deepEqual(row, built.tasks[i], `tasks[${i}]`);
  for (const [i, row] of ex.trajectories.entries()) assert.deepEqual(row, built.trajectories[i], `trajectories[${i}]`);
  assert.deepEqual([R.index.explorer.tasks, R.index.explorer.trajectories, R.index.explorer.generated_at],
                   [ex.tasks.length, ex.trajectories.length, ex.generated_at]);
  assert.ok(ex.generated_at <= R.index.generated_at);
  const sum = (k) => Object.values(ex.totals).reduce((s, t) => s + t[k], 0);
  assert.deepEqual([sum('tasks'), sum('trajectories'), sum('runs')], [ex.tasks.length, ex.trajectories.length, ex.runs.length]);
});

test('explorer (v1.7): every deep link opens its node in its task view; the batch is never audited', exOpts, () => {
  const m = app.explorerModel(R.explorer), docs = new Map(explorerEntries().map((e) => [e.detail, e.doc]));
  assert.equal(m.tasks.length, R.explorer.tasks.length);
  assert.equal(m.trajectories.length, R.explorer.trajectories.length);
  for (const t of m.tasks) {
    const h = app.parseHash(t.link), doc = docs.get(t.run.detail);
    assert.equal(h.view, 'task', t.link);
    assert.equal(h.task, doc.task_id, t.link);
    // the latest official run's view needs no run=; (v1.8) an older official run's and a batch run's do
    assert.equal(h.run, t.run.source.audited && t.run.latest !== false ? null : doc.run, t.link);
    const task = R.index.tasks.find((x) => x.task_id === h.task);
    const view = app.selectedView(app.viewsOf(R.index, task), h.run);
    assert.equal(view.run.run, doc.run, `${t.link} opens ${doc.run}`);
    const node = doc.nodes.find((n) => n.id === h.node);
    assert.ok(node && node.build.task_id === t.taskId && node.decision === t.decision, t.link);
    assert.equal(t.dir, `${doc.build_root}/${node.id}`);
    for (const r of t.traj) {
      assert.equal(r.transcript, node.score ? `${doc.build_root}/round-${node.round}/${node.id}/rollouts/${r.slot}.md` : '');
    }
  }
  const batch = m.tasks.filter((t) => !t.run.source.audited);
  assert.ok(batch.every((t) => t.audit === 'not_audited' && t.yes === null && t.partial === null), 'a batch task with an audit');
  for (const s of m.sources.filter((x) => !x.audited)) assert.equal(R.explorer.totals[s.source].audited, 0, s.source);
  const st = app.explorerStats(m, app.filterExplorer(m, {}), {});
  assert.ok(st.patterns.every((g) => m.tasks.some((t) => t.target === g.id && t.run.source.audited)), 'pattern coverage: official targets only');
  assert.equal(st.excluded.notAuditedBatch, batch.length);
});

test('explorer (v1.7): the view on the real data for many filters: sane markup, sound numbers, counts that agree, fast', exOpts, () => {
  const m = app.explorerModel(R.explorer), doc = R.explorer;
  const combos = [{}, {round: [0]}, {round: [1]}, {round: [2]}, {dec: 'acc'}, {dec: 'not'}, {smin: 6}, {smin: 6, nj: true},
                  {smin: 8}, {smin: 8, nj: true}, {smin: 8, smax: 6}, {nmin: 0.6, nmax: 0.2}, {rd: 'only'}, {rd: 'none'},
                  {nmax: 0.2}, {kmin: 0.6}, {kmin: 0.6, kb: 'raw'}, {audit: ['strong', 'partial']}, {audit: ['not_audited']},
                  {audit: ['not_eligible']}, {tm: ['sc14'], ts: ['fail'], tw: true}, {tm: ['kimi'], ts: ['pass'], tc: true},
                  {ts: ['infra'], tw: true}, {cov: 'traj'}, {sort: 'sim', desc: true, p: 2}, {sort: 'traj', tp: 9999, p: 9999},
                  {round: [1], nmax: 0.2, kmin: 0.6, smin: 6, dec: 'acc', audit: ['strong', 'partial', 'weak', 'none']}];
  for (const s of m.sources) combos.push({src: [s.source]});
  for (const id of m.targetIds.slice(0, 5)) combos.push({t: [id]});
  let slowest = 0;
  for (const over of combos) {
    const f = Object.assign(app.explorerDefaults(), over), where = `explorer ${app.explorerHash(f)}`;
    const t0 = process.hrtime.bigint();
    const r = app.filterExplorer(m, f);
    const results = app.renderExplorerResults(doc, R.index, {now: NOW, ui: {}, filters: f});
    slowest = Math.max(slowest, Number(process.hrtime.bigint() - t0) / 1e6);
    sane(results, where, EX_TAGS);
    numbersOk(results, where);
    assert.ok(results.includes(`<div class="tile-value">${app.fmtInt(r.tasks.length)}<span class="tile-den"> / ${app.fmtInt(m.tasks.length)}</span>`), where);
    assert.ok(results.includes(`<div class="tile-value">${app.fmtInt(r.trajectories.length)}<span class="tile-den"> / ${app.fmtInt(m.trajectories.length)}</span>`), where);
    const rows = Math.min(app.EX_PAGE, r.tasks.length);
    if (!f.p || f.p === 1) assert.equal(count(results, '<tr class="ex-row" data-exlink="'), rows, where);
    const dists = results.slice(results.indexOf('<h2>Distributions</h2>'), results.indexOf('<h2>Synthetic tasks</h2>'));
    assert.ok(!dists.includes('vbar-col s1'), `${where}: no histogram in the official source's color`);
    assert.ok(count(results, 'tabindex="0"') <= 3, `${where}: one tab stop per chart panel`);
    if (r.tasks.length) {
      const st = app.explorerStats(m, r, f);
      assert.equal(st.dist.audits.reduce((a, b) => a + b, 0), r.tasks.length, where);
      assert.equal(st.coverage.reduce((s, c) => s + c.tasks.reduce((a, b) => a + b, 0), 0), r.tasks.length, `${where}: coverage sums to the tasks`);
      assert.equal(st.coverage.reduce((s, c) => s + c.traj.reduce((a, b) => a + b, 0), 0), r.trajectories.length, `${where}: and the trajectories`);
    }
  }
  sane(app.renderExplorer(doc, R.index, {now: NOW, ui: {}, filters: {}, query: 'tengo'}), 'explorer page', EX_TAGS);
  assert.ok(slowest < 500, `filtering and rendering take ${slowest.toFixed(0)} ms at most`);
  for (const w of ['tasks.csv', 'tasks.json', 'trajectories.csv', 'trajectories.json']) {
    const e = app.explorerExport(doc, {round: [1], tm: ['kimi']}, w);
    assert.ok(e && e.text.length > 100, w);
  }
});

test('explorer (v1.7): escaping and NaN guards with every free-text string of the real explorer.json neutralized', exOpts, () => {
  const PAY = '"\'><script>X1</script><img src=x onerror=X2>';
  const payload = neutral(R.explorer, () => PAY);
  for (const f of [{}, {cov: 'traj', p: 2, tp: 3}, {sort: 'krate', kb: 'raw'}]) {
    const html = app.renderExplorer(payload, R.index, {now: NOW, ui: {}, filters: f, query: PAY});
    assert.ok(!html.includes('<script>X1') && !html.includes('<img src=x'), JSON.stringify(f));
    sane(html, `escaped ${JSON.stringify(f)}`, EX_TAGS);
    numbersOk(app.renderExplorer(neutral(R.explorer), R.index, {now: NOW, ui: {}, filters: f}), `neutral ${JSON.stringify(f)}`);
  }
});

test('wiring (v1.7): #explorer fetches data/explorer.json by its served path; a filter edit and a row open as on the page', exOpts, async () => {
  const p = page();
  p.win.location.hash = '#explorer';
  const a = app.createApp(p.env);
  await a.start();
  await settle(a);
  assert.ok(p.calls.includes('data/explorer.json'), p.calls.join(', '));
  assert.ok(!p.calls.some((u) => u.startsWith('data/task/')));
  const main = p.el('main').innerHTML;
  sane(main, 'main (explorer)', EX_TAGS);
  assert.match(main, /<h2>Coverage per DeepSWE task<\/h2>/);
  const control = Object.assign(new Element(p.doc, null, 'INPUT', p.el('main')), {value: '1', checked: false});
  control.dataset.exf = 'round';
  p.doc.dispatch('change', {target: control});
  const m = app.explorerModel(R.explorer);
  assert.equal(p.win.location.hash, app.explorerHash(Object.assign(app.explorerDefaults(), {round: m.rounds.filter((k) => k !== 1)})));
  sane(p.el('ex-results').innerHTML, 'results after a filter edit', EX_TAGS);
  const row = new Element(p.doc, null, 'TR', p.el('main'));
  const t = m.tasks.find((x) => x.round !== 1);
  row.dataset.exlink = t.link;
  p.doc.dispatch('click', {target: row});
  await settle(a);
  assert.equal(p.win.location.hash, t.link);
  assert.ok(p.calls.includes(`data/${t.run.detail}`), `fetched ${t.run.detail}`);
  assert.equal(p.el('drawer').hidden, false);
  assert.ok(p.el('drawer').innerHTML.includes(esc(t.node)));
  assert.equal(p.el('banner').innerHTML.includes('Refresh failed'), false, 'every fetch succeeded');
  a.stop();
});

test('explorer (v1.7 review): the real data: a similarity bound, the KPI scope, the re-scored rows, exact rate columns', exOpts, () => {
  const m = app.explorerModel(R.explorer), F = (q) => app.parseExplorerParams(q);
  const never = m.tasks.filter((t) => t.sim === null).length, judged8 = m.tasks.filter((t) => t.sim !== null && t.sim >= 8).length;
  assert.ok(never > 0 && judged8 > 0);
  const r8 = app.filterExplorer(m, F('smin=8'));
  assert.deepEqual([r8.tasks.length, r8.neverJudged, r8.simBounded], [judged8, never, true]);
  assert.ok(r8.tasks.every((t) => t.sim !== null && t.sim >= 8));
  assert.equal(app.filterExplorer(m, F('smin=8&nj=1')).tasks.length, judged8 + never);
  assert.equal(app.filterExplorer(m, F('smin=8&sun=1')).tasks.length, judged8 + never, 'a link of the first page that kept them');
  assert.equal(app.filterExplorer(m, F('sun=0')).tasks.length, m.tasks.length - never, 'one that left them out');
  const bar = app.renderExplorerFilters(m, F('smin=8'), {});
  assert.ok(bar.includes(`>+${app.fmtInt(never)} never judged</span>`), 'the count next to "include never judged"');
  const results = app.renderExplorerResults(R.explorer, R.index, {filters: F('smin=8')});
  assert.ok(results.includes(`<div class="tile-value">${app.fmtInt(judged8)}<span class="tile-den"> / ${app.fmtInt(m.tasks.length)}</span>`));
  assert.match(app.renderExplorerResults(R.explorer, R.index, {filters: F('smin=8&smax=6')}), /A range is inverted:/);
  // the KPI: covered of what the chosen sources converted, of index.json's DeepSWE tasks
  for (const src of m.sources) {
    const mine = new Set(m.tasks.filter((t) => t.source === src.source).map((t) => t.target)).size;
    const html = app.renderExplorerResults(R.explorer, R.index, {filters: {src: [src.source]}});
    assert.ok(html.includes(`<div class="tile-value">${mine}<span class="tile-den"> / ${mine}</span></div><div class="tile-sub">with ≥ 1 ` +
                            `filtered task, of the ${mine} converted by the chosen sources (of ${R.index.tasks.length} DeepSWE tasks)</div>`), src.source);
  }
  // the re-scored rows: as many as the totals say, each one's metrics.json rates apart from its rollout rows'
  const rescored = m.tasks.filter((t) => t.rescored);
  assert.equal(rescored.length, Object.values(R.explorer.totals).reduce((a, t) => a + t.rates_differ, 0));
  for (const t of rescored) {
    const same = (a, b) => (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) < 1e-9);
    assert.ok(!same(t.sRate, t.rowsRates.s) || !same(t.kClean, t.rowsRates.k) ||
              JSON.stringify(t.valid) !== JSON.stringify(t.rowsRates.valid), `${t.target}/${t.node}`);
  }
  const table = app.renderExplorerTaskTable(m, app.filterExplorer(m, F('rd=only')), F('rd=only'));
  assert.ok(count(table, 'class="ex-mark"') >= rescored.length);
  assert.equal(count(table, '<tr class="ex-row" data-exlink="'), Math.min(rescored.length, app.EX_PAGE));
  // exact rate columns: every value its own column; no column holds two values
  for (const k of ['s', 'clean', 'raw']) {
    const cols = m.rateColumns[k], key = k === 's' ? 'sRate' : k === 'clean' ? 'kClean' : 'kRaw';
    const values = new Set(m.tasks.map((t) => app.exPct(t[key])).filter((v) => v !== null));
    if (values.size <= 16) assert.deepEqual(cols.map((c) => c.lo), [...values].sort((a, b) => a - b), k);
  }
  // the CSV: a BOM, numbers for F2P / P2P
  const csv = app.explorerExport(R.explorer, {}, 'trajectories.csv').text;
  assert.equal(csv.charCodeAt(0), 0xfeff);
  assert.doesNotMatch(csv, /,\d+\/\d+,/);
});

test('wiring (v1.7 review): on the real data, Escape after a node visit leaves the Explorer\'s URL alone', exOpts, async () => {
  const p = page();
  const start = '#explorer&round=1&audit=strong';
  p.win.location.hash = start;
  const a = app.createApp(p.env);
  await a.start();
  await settle(a);
  const m = app.explorerModel(R.explorer);
  const t = m.tasks.find((x) => x.round === 1 && x.audit === 'strong');
  assert.ok(t, 'a round-1 task with a strong verdict');
  const row = new Element(p.doc, null, 'TR', p.el('main'));
  row.dataset.exlink = t.link;
  p.doc.dispatch('click', {target: row});
  await settle(a);
  assert.equal(p.win.location.hash, t.link);
  assert.equal(p.el('drawer').hidden, false);
  p.win.location.hash = start;                                    // Back
  await settle(a);
  p.doc.dispatch('keydown', {key: 'Escape', target: p.doc.body});
  assert.equal(p.win.location.hash, start);
  const box = new Element(p.doc, null, 'INPUT', p.el('main'));
  box.dataset.exf = 'tq';
  p.doc.dispatch('keydown', {key: 'Escape', target: box});
  assert.equal(p.win.location.hash, start);
  assert.match(p.el('main').innerHTML, /<h2>Coverage per DeepSWE task<\/h2>/);
  a.stop();
});

// ------------------------------------------------------------------------------------ every official run (v1.8)

test('official runs (v1.8): one latest per task (its task file), the older ones each with their own document and view', opts, () => {
  const official = R.index.runs.filter((r) => !r.source);
  if (!official.some((r) => 'latest' in r)) return;                // a collector before v1.8
  const byTask = new Map();
  for (const r of official) byTask.set(r.task_id, (byTask.get(r.task_id) || []).concat([r]));
  for (const [task, runs] of byTask) {
    const t = R.index.tasks.find((x) => x.task_id === task);
    const latest = runs.filter((r) => r.latest === true);
    assert.equal(latest.length, 1, task);
    assert.equal(t.run, latest[0].run, task);
    assert.deepEqual(t.official_runs.map((o) => o.run), app.officialRunsOf(R.index, t).map((r) => r.run), task);
    assert.equal(R.details[task].run, latest[0].run, task);
    for (const r of runs.filter((x) => x.latest === false)) {
      const doc = R.officialDocs[r.detail];
      assert.ok(doc && doc.run === r.run && doc.task_id === task, r.detail);
      assert.equal(doc.generated_at, r.detail_generated_at, r.detail);
      const html = app.renderTaskView(R.index, task, doc, {now: NOW, run: r.run, ui: {showAbandoned: true}});
      sane(html, `older ${r.run}`);
      assert.ok(html.includes('older-run-note') && html.includes(`data-run-select="${esc(r.run)}" aria-pressed="true"`), r.run);
      for (const n of doc.nodes) sane(app.renderNodeDetail(n, doc, {}, NOW), `older ${r.run} ${n.id}`);
    }
    if (runs.length > 1) {
      const html = app.renderTaskView(R.index, task, R.details[task], {now: NOW});
      assert.equal(count(html, 'data-run-select="'), app.viewsOf(R.index, t).length, task);
      assert.match(html, /class="run-tab active" data-run-select=""/);
    }
  }
});

test('queue (v1.8): the Monitor\'s synthesis queues and the picker\'s "queued in" notes on the live index', opts, () => {
  const queues = R.index.queues;
  if (!Array.isArray(queues)) return;                              // a collector before v1.8 (or no queue configured)
  const html = app.renderMonitor(R.fleet, R.api, {now: NOW, ui: {}, index: R.index});
  sane(html, 'monitor with queues');
  numbersOk(html, 'monitor with queues');
  for (const q of queues) {
    const at = html.indexOf(`<h2>Synthesis queue ${esc(q.name)}</h2>`);
    assert.ok(at >= 0, q.name);
    const part = html.slice(at, html.indexOf('</section>', at));
    assert.equal(count(part, '<tr><td>'), q.runs.length, `${q.name}: a row per run`);
    assert.equal(count(part, '<a href="#task='), q.runs.filter((r) => R.index.runs.some((x) => !x.source && x.run === r.run)).length,
                 `${q.name}: the launched runs link their view`);
    assert.equal(q.runs.length, new Set(q.runs.map((r) => r.run)).size, `${q.name}: each run once`);
    assert.ok(q.runs.every((r) => R.index.tasks.some((t) => t.task_id === r.task)), `${q.name}: DeepSWE tasks`);
    assert.ok(!q.budget || Object.keys(q.budget).join() === 'key,remaining,at', `${q.name}: the budget is a key name and a number`);
  }
  const menu = app.renderTaskMenu(R.index, '', null, 0);
  sane(menu, 'picker');
  const queued = R.index.tasks.filter((t) => app.queuedIn(R.index, t.task_id).length);
  for (const t of queued) assert.ok(menu.includes(`data-task="${esc(t.task_id)}"`) && menu.includes('queued in '), t.task_id);
  assert.equal(queued.length > 0, queues.some((q) => q.runs.some((r) => r.status === 'queued' && !R.index.runs.some((x) => x.run === r.run))));
});
