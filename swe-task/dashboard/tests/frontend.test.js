'use strict';
/* Frontend tests (node:test, no dependencies). Renders every view of static/app.js from the fixtures and from
sparse/broken copies; checks key text and badges, escaping of injected markup, the tree layout, the older collector
shapes next to the current ones, the failure-match audits (card badges, the drawer section, the audit table, "audit n/m"
in the run bar; on copies of the narwhals fixture with audits: v1.3, the accepted tasks; v1.4, every eligible task with
NODE.audit_eligible, a withdrawn task and an audit of a discarded one), the v1.5 seed notes (NODE.seed_invalid: an
accepted task seeds.json lists invalid reads ACCEPTED plus a "seed-invalid" badge and a drawer callout), the pure refresh
helpers (skip / hold / render, scroll and open-state capture and restore), and the DOM wiring (createApp) against a
small fake DOM that parses the data-scroll / data-key / data-fkey / data-node elements out of the markup it is given.
v1.6 (a batch from another tree, never audited; synthetic documents made from the narwhals fixture): the picker group
"<name> batch · not audited", the run selector of a task with two batch runs, the batch run bar and table, the cards'
"batch · not audited" badges, the drawers, escaping, the wiring of #task=..&run=.. (the run's document fetched by its
served path, refetched only for a newer detail_generated_at), and official views, drawers, menu and monitor unchanged
(byte-identical to the v1.5 page, static/app.js.pre-baihe-batch, when that backup is there).
v1.7 (the Explorer; its explorer.json made by tests/explorer_fixture.js from the fixtures, the batch's included): the hash
round trip of every filter, each filter and their combination, the statistics (KPIs, coverage by source, failure-pattern
coverage, distributions), the view's sections and chart marks, sorting and pages, exports, escaping, broken documents,
the wiring of #explorer (the file, filter edits re-rendering the results alone, rows opening their task view, exports,
refreshes), and the task / monitor views, drawers, menu and header unchanged (byte-identical to static/app.js.pre-explorer).

    node --test tests/frontend.test.js        (fixtures: node tests/make_fixtures.js)
*/
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const app = require('../static/app.js');

const FIX = path.join(__dirname, 'fixtures');
const STATIC = path.join(__dirname, '..', 'static');
const load = (name) => JSON.parse(fs.readFileSync(path.join(FIX, name), 'utf8'));
const clone = (v) => JSON.parse(JSON.stringify(v));
const fixtures = () => ({
  index: load('index.json'),
  narwhals: load('task-narwhals-rolling-window-suite.json'),
  bandit: load('task-bandit-incremental-cache-control.json'),
  fd: load('task-fd-deterministic-multi-key-sorting.json'),
  fleet: load('fleet.json'),
  api: load('api.json'),
});
const F = fixtures();
const NOW = F.index.generated_at + 30;
const NAR = 'narwhals-rolling-window-suite';
const BAN = 'bandit-incremental-cache-control';
const count = (s, needle) => s.split(needle).length - 1;
const nodeOf = (doc, id) => doc.nodes.find((n) => n.id === id);
/** The card of one node as the tree renders it (with the tree's context: "P<n>" focus tags). */
function cardOf(doc, id, ui) {
  const L = app.treeLayout(doc, Object.assign({showAbandoned: true}, ui));
  for (const lin of L.lineages) for (const it of lin.items) if (it.id === id) return app.renderNodeCard(it, ui || {}, L.geom, app.treeContext(doc));
  return null;
}
/** Collector output from before 2026-10-17 18:00: no probe.partial / own_done, no focus_pattern, string audits. */
function olderShapes(doc) {
  const d = clone(doc);
  for (const n of d.nodes) {
    delete n.focus_pattern;
    if (n.probe) { delete n.probe.partial; delete n.probe.own_done; }
    if (Array.isArray(n.details.teacher_audit)) n.details.teacher_audit = JSON.stringify(n.details.teacher_audit);
  }
  return d;
}

/** Every view of the page for one data set (each must return a string and never throw). */
function renderAll(d, ui) {
  const u = ui || {};
  const out = {
    status: app.renderStatus(d.index, NOW), statusHeld: app.renderStatus(d.index, NOW, {held: true, shown: NOW - 400}),
    banner: app.renderBanner(d.index, NOW, {errors: {index: 'x'}}),
    menu: app.renderTaskMenu(d.index, '', NAR, 0), menuQuery: app.renderTaskMenu(d.index, 'nar', NAR, 1),
    narwhals: app.renderTaskView(d.index, NAR, d.narwhals, {now: NOW, ui: u}),
    narwhalsAll: app.renderTaskView(d.index, NAR, d.narwhals, {now: NOW, ui: Object.assign({showAbandoned: true, selected: 'cand-c04-r1b1', treeWidth: 1700}, u)}),
    bandit: app.renderTaskView(d.index, BAN, d.bandit, {now: NOW, ui: u}),
    fd: app.renderTaskView(d.index, 'fd-deterministic-multi-key-sorting', d.fd, {now: NOW, ui: u}),
    noRun: app.renderTaskView(d.index, 'go-git-worktree-merge-conflicts', null, {now: NOW, ui: u}),
    loading: app.renderTaskView(d.index, NAR, null, {now: NOW}),
    detailError: app.renderTaskView(d.index, NAR, null, {now: NOW, detailError: 'HTTP 404'}),
    unknown: app.renderTaskView(d.index, 'no-such-task', null, {now: NOW}),
    monitor: app.renderMonitor(d.fleet, d.api, {now: NOW, ui: u}),
    monitorMissing: app.renderMonitor(null, null, {now: NOW, fleetError: 'HTTP 500', apiError: 'HTTP 500'}),
    auth: app.renderAuthError(),
  };
  const nodes = (doc) => (Array.isArray(doc && doc.nodes) ? doc.nodes : []);
  out.drawers = nodes(d.narwhals).map((n) => app.renderNodeDetail(n, d.narwhals, u, NOW)).join('\n');
  out.drawersBandit = nodes(d.bandit).map((n) => app.renderNodeDetail(n, d.bandit, u, NOW)).join('\n');
  out.drawerMissing = app.renderNodeDetail(null, d.narwhals, u, NOW);
  // Section renderers called directly (outside safe()), so broken data reaches each of them.
  const pick = (v, k) => (v && typeof v === 'object' ? v[k] : undefined);
  const tasks = Array.isArray(pick(d.index, 'tasks')) ? d.index.tasks : [];
  const runs = pick(d.index, 'runs');
  out.accuracy = tasks.map((t) => app.renderAccuracy(d.index, t) + app.renderNoRun(t)).join('\n');
  out.runBar = app.renderRunBar(Array.isArray(runs) ? runs[0] : runs, d.narwhals, NOW, u);
  out.failure = app.renderFailureAnalysis(pick(d.narwhals, 'failure_analysis'), u, pick(d.narwhals, 'nodes')) +
                app.renderFailureAnalysis(pick(d.bandit, 'failure_analysis'), u, pick(d.bandit, 'nodes'));
  out.plan = app.renderPlan(pick(d.narwhals, 'plan'), u) + app.renderPlan(pick(d.bandit, 'plan'), u);
  out.tree = app.renderTree(d.narwhals, u) + app.renderTree(d.bandit, Object.assign({showAbandoned: true}, u)) + app.renderTree(d.fd, u);
  out.roundZero = app.renderRoundZero(d.narwhals, u) + app.renderRoundZero(d.bandit, u) + app.renderRoundZero(d.fd, u);
  out.cards = nodes(d.narwhals).map((n) => app.renderNodeCard({node: n, x: 0, y: 0}, u, null, app.treeContext(d.narwhals))).join('\n');
  out.fleetParts = app.renderFleetSummary(d.fleet, NOW) + app.renderWorkers(d.fleet, NOW) + app.renderFleetCharts(d.fleet, NOW, u);
  out.api = app.renderApi(d.api, NOW, u);
  return out;
}

// --------------------------------------------------------------------------------------------- helpers

test('esc() escapes every HTML-special character and never throws', () => {
  assert.equal(app.esc('<script>alert("x")</script>'), '&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt;');
  assert.equal(app.esc(`a&b'c\`d`), 'a&amp;b&#39;c&#96;d');
  assert.equal(app.esc(null), '');
  assert.equal(app.esc(undefined), '');
  assert.equal(app.esc(0), '0');
  assert.equal(app.esc(false), 'false');
  assert.equal(app.esc({a: '<b>'}), '{&quot;a&quot;:&quot;&lt;b&gt;&quot;}');
  const cyclic = {};
  cyclic.self = cyclic;
  assert.equal(app.esc(cyclic), '[unprintable]');
  assert.equal(app.esc('&amp;'), '&amp;amp;');            // no double-decoding shortcuts
});

test('formatters handle edge values', () => {
  assert.equal(app.fmtInt(4903), '4,903');
  assert.equal(app.fmtInt(null), '–');
  assert.equal(app.fmtInt('12'), '–');
  assert.equal(app.fmtCompact(12345), '12.3K');
  assert.equal(app.fmtCompact(10000), '10K');
  assert.equal(app.fmtCompact(2.5e6), '2.5M');
  assert.equal(app.fmtCompact(88.14), '88.1');
  assert.equal(app.fmtPct(0.096, 1), '9.6%');
  assert.equal(app.fmtMoney(9531.25), '$9,531.25');
  assert.equal(app.fmtMoney(-3), '-$3.00');
  assert.equal(app.fmtDur(45), '45s');
  assert.equal(app.fmtDur(11593), '3h 13m');
  assert.equal(app.fmtDur(2 * 86400 + 4 * 3600), '2d 4h');
  assert.equal(app.fmtAgo(100, 137), '37s ago');
  assert.equal(app.fmtAgo(200, 100), 'in 1m');
  assert.equal(app.fmtDate(1e308), '–');                        // outside the Date range: a dash, not NaN
  assert.equal(app.fmtDate('2026'), '–');
  assert.equal(app.fmtBytes(0), '0 B');
  assert.equal(app.fmtBytes(1536), '1.5 KB');
  assert.equal(app.fmtBytes(12900000), '12 MB');
  assert.equal(app.fmtBytes(null), '–');
  assert.deepEqual(app.niceTicks(10, 5, true).ticks, [0, 2, 4, 6, 8, 10]);
  assert.equal(app.niceTicks(0, 4, false).top, 1);
  assert.equal(app.niceTicks(468.75, 4, false).top, 500);
});

test('hash routing round-trips task, node and (v1.6) batch run ids', () => {
  assert.deepEqual(app.parseHash('#monitor'), {view: 'monitor', task: null, node: null, run: null});
  assert.deepEqual(app.parseHash(''), {view: 'task', task: null, node: null, run: null});
  const h = app.taskHash('a b&c=d', 'cand-c01-r1b2');
  assert.equal(h, '#task=a%20b%26c%3Dd&node=cand-c01-r1b2');           // official: no run=, as before v1.6
  assert.deepEqual(app.parseHash(h), {view: 'task', task: 'a b&c=d', node: 'cand-c01-r1b2', run: null});
  assert.equal(app.parseHash('#task=narwhals-rolling-window-suite').task, NAR);
  assert.deepEqual(app.parseHash(null), {view: 'task', task: null, node: null, run: null});
  const b = app.taskHash('t-1', 'cand-c01', 'pilot7&x');                // a batch run's view
  assert.equal(b, '#task=t-1&run=pilot7%26x&node=cand-c01');
  assert.deepEqual(app.parseHash(b), {view: 'task', task: 't-1', node: 'cand-c01', run: 'pilot7&x'});
});

test('static files load nothing external, carry no inline handlers, and make room for the drawer', () => {
  const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
  const css = fs.readFileSync(path.join(STATIC, 'style.css'), 'utf8');
  const js = fs.readFileSync(path.join(STATIC, 'app.js'), 'utf8');
  for (const [name, s] of [['index.html', html], ['style.css', css]]) {
    assert.doesNotMatch(s, /(?:src|href)\s*=\s*["']?(?:https?:)?\/\//i, `${name} references an external URL`);
    assert.doesNotMatch(s, /@import|url\(\s*["']?(?:https?:)?\/\//i, `${name} imports external CSS`);
  }
  assert.match(html, /<script src="static\/app\.js" defer><\/script>/);
  assert.equal(count(html, '<script'), 1, 'one external script, no inline one');
  assert.doesNotMatch(html, /<style|\son[a-z]+\s*=|javascript:/i, 'no <style>, on*= handlers or javascript: URLs');
  assert.match(html, /Content-Security-Policy/);
  for (const id of ['task-input', 'task-menu', 'tab-task', 'tab-monitor', 'status', 'refresh', 'banner', 'main', 'drawer', 'drawer-grip']) {
    assert.match(html, new RegExp(`id="${id}"`), `index.html lacks #${id}`);
  }
  assert.match(html, /id="drawer-grip"[^>]*role="separator"[^>]*tabindex="0"/);
  assert.doesNotMatch(js, /\beval\(|new Function\(/);
  assert.match(js, /cache: 'no-cache', credentials: 'same-origin'/);
  // F3: from 1100 px the open drawer pads the page by its (adjustable) width; the grip resizes it.
  assert.match(css, /@media \(min-width: 1100px\) \{\s*body\.drawer-open \.main, body\.drawer-open \.banner \{\s*padding-right: calc\(var\(--drawer-w-eff\) \+ 24px\)/);
  assert.match(css, /\.drawer \{[^}]*width: var\(--drawer-w-eff\)/);
  assert.match(css, /\.drawer-grip \{[^}]*cursor: col-resize/);
  assert.match(css, /\.node \{[^}]*scroll-margin:/);
});

// ------------------------------------------------------------------------------------------ fixtures

test('fixtures follow the contract shape, current and older collector shapes', () => {
  assert.equal(F.index.schema, 1);
  assert.deepEqual(Object.keys(F.index.models), ['kimi', 'super_vl', 'sc14']);
  for (const t of F.index.tasks) {
    for (const k of ['task_id', 'task_index', 'f2p_total', 'accuracy', 'eligible', 'queue_rank', 'run']) assert.ok(k in t, `task.${k}`);
  }
  const nodeKeys = ['id', 'parent', 'lineage', 'round', 'branch', 'kind', 'title', 'summary', 'status', 'build', 'probe', 'score',
                    'decision', 'decision_reason', 'children', 'details'];
  for (const doc of [F.narwhals, F.bandit]) {
    for (const n of doc.nodes) for (const k of nodeKeys) assert.ok(k in n, `${n.id}.${k}`);
  }
  const kinds = new Set(F.narwhals.nodes.map((n) => app.decisionOf(n).key));
  for (const k of ['evolve', 'discard', 'build_failed', 'accepted', 'accepted_evolve', 'not_selected', 'end', 'pending', 'abandoned']) {
    assert.ok(kinds.has(k), `narwhals fixture lacks decision ${k}`);
  }
  assert.ok(F.index.tasks.some((t) => !t.run), 'a task without a run');
  const states = new Set(F.fleet.workers.map((w) => w.registry_status || (w.state === 'PENDING' ? 'pending' : w.state)));
  for (const s of ['ready', 'draining', 'starting', 'pending']) assert.ok(states.has(s), `fleet lacks a ${s} worker`);
  // Current shapes (narwhals): partial results while probing, one task whose own slots are done, focus patterns, audit lists.
  const probing = F.narwhals.nodes.filter((n) => n.status === 'probing');
  assert.equal(probing.length, 5);
  assert.ok(probing.every((n) => n.probe.partial && n.probe.partial.sc14 && n.probe.partial.kimi));
  assert.equal(probing.filter((n) => n.probe.own_done === true).length, 1);
  assert.ok(F.narwhals.nodes.filter((n) => n.round > 0).every((n) => typeof n.focus_pattern === 'string'));
  assert.ok(F.narwhals.nodes.some((n) => Array.isArray(n.details.teacher_audit) && n.details.teacher_audit.some((a) => a.cheated)));
  // Older shapes (bandit): none of that; a building round instead of a probe.
  assert.ok(F.bandit.nodes.every((n) => !('focus_pattern' in n) && !(n.probe && 'partial' in n.probe)));
  assert.ok(F.bandit.nodes.some((n) => typeof n.details.teacher_audit === 'string'));
  assert.equal(F.bandit.rounds[2].status, 'building');
  assert.equal(F.fleet.totals.busy_workers, 8);
  assert.equal(F.fleet.router.scope, 'official-run probes');
  assert.ok('prompt_tps_incl_cache' in F.fleet.totals && !('prompt_tps' in F.fleet.totals));
  assert.equal(F.api.scan.complete, false);
  assert.equal(F.api.burn.window_min_6h, 352);
});

// ------------------------------------------------------------------------------------------ task tab

test('every view renders from the fixtures without fallbacks', () => {
  const out = renderAll(F);
  for (const [name, html] of Object.entries(out)) {
    assert.equal(typeof html, 'string', name);
    assert.ok(!html.includes('render-error'), `${name} fell back to a render error`);
    assert.doesNotMatch(html, /undefined|NaN|\[object Object\]/, `${name} leaks a JS artefact`);
  }
});

test('header status: "updated hh:mm:ss" with its age, held refreshes, staleness warning', () => {
  const fresh = app.renderStatus(F.index, F.index.generated_at + 30);
  assert.ok(fresh.includes(`updated ${app.fmtClock(F.index.generated_at, true)}<span class="age"> · 30s ago</span>`), fresh);
  assert.doesNotMatch(fresh, /stale|held/);
  // A newer cycle waits while text is selected: the indicator keeps the time of the data on screen.
  const held = app.renderStatus(F.index, NOW, {held: true, shown: F.index.generated_at - 120});
  assert.ok(held.includes(`updated ${app.fmtClock(F.index.generated_at - 120, true)}`));
  assert.match(held, /newer data held/);
  assert.match(held, /text is selected on the page/);
  assert.doesNotMatch(app.renderStatus(F.index, NOW, {held: false, shown: F.index.generated_at - 120}), /held/);
  const stale = app.renderStatus(F.index, F.index.generated_at + 12 * 60);
  assert.match(stale, /stale 12m/);
  assert.match(app.renderBanner(F.index, F.index.generated_at + 12 * 60, {}), /Stale data/);
  assert.doesNotMatch(app.renderBanner(F.index, F.index.generated_at + 60, {}), /Stale data/);
  assert.match(app.renderBanner(F.index, NOW, {}), /2 collector warnings/);
  assert.match(app.renderBanner(F.index, NOW, {errors: {index: 'data/index.json: HTTP 502'}}), /Refresh failed \(index\)/);
  assert.match(app.renderStatus(null, NOW), /no data yet/);
  assert.match(app.renderAuthError(), /Missing or expired token: open the URL with \?token=\.\.\./);
});

test('task picker groups: Converting, Eligible queue (by rank), All tasks; search filters', () => {
  const groups = app.taskGroups(F.index, '');
  assert.deepEqual(groups.map((g) => g.label), ['Converting', 'Eligible queue', 'All tasks']);
  assert.deepEqual(groups[0].items.map((t) => t.task_id).slice(0, 2), [NAR, BAN]);
  assert.deepEqual(groups[1].items.map((t) => t.queue_rank), [1, 2, 3, 4, 5]);
  assert.equal(groups[2].items.length, F.index.tasks.length);
  const menu = app.renderTaskMenu(F.index, '', NAR, 0);
  assert.match(menu, /Converting/);
  assert.match(menu, /Eligible queue/);
  assert.match(menu, /All tasks/);
  assert.match(menu, /class="opt selected active"[^>]*data-task="narwhals-rolling-window-suite"/);
  assert.equal(app.menuCount(F.index, 'bandit'), 3);
  assert.equal(app.menuOption(F.index, 'bandit', 2), BAN);
  assert.equal(app.menuOption(F.index, 'bandit', 3), null);
  assert.match(app.renderTaskMenu(F.index, 'zzz', NAR, 0), /No task matches “zzz”/);
  assert.equal(app.defaultTask(F.index), NAR);
});

test('accuracy cards: kimi / super_vl / sc14 with chips, ratio and eligibility; unscored and infra chips show a dash', () => {
  const task = app.findTask(F.index, NAR);
  const html = app.renderAccuracy(F.index, task);
  assert.ok(html.indexOf('Kimi K3') < html.indexOf('Nemotron 3.5 Super VL'));
  assert.ok(html.indexOf('Nemotron 3.5 Super VL') < html.indexOf('Local Super ckpt sc14'));
  assert.equal(count(html, 'class="acc-card'), 3);
  assert.equal(count(html, 'class="chip chip-'), 15);
  assert.match(html, /<span class="chip chip-pass" title="rollout 5: pass · F2P 103\/103 · P2P 10093\/10093">103\/103<\/span>/);
  assert.match(html, /<span class="chip chip-pass" title="sc14-r4: pass · F2P 103\/103 · P2P 10093\/10093">103\/103<\/span>/);
  assert.match(html, /chip-fail[^>]*>102\/103</);
  assert.match(html, /mean F2P ratio <b>0\.98<\/b>/);
  assert.match(html, /ELIGIBLE/);
  assert.match(html, /queue #1/);
  const sparse = app.renderAccuracy(F.index, app.findTask(F.index, 'expr-try-catch-errors'));
  assert.match(sparse, /no data/);                               // super_vl: null
  assert.match(sparse, /2 infra/);
  assert.match(sparse, /1 unscored/);
  assert.match(sparse, /not eligible/);
  // F9: an infra sc14 slot ("0/0" from a verifier that never ran) and an unscored kimi rollout (no fraction) read "–".
  assert.ok(sparse.includes('<span class="chip chip-unscored" title="sc14-r1: infra · F2P 0/0 · P2P 0/0">–</span>'), sparse);
  assert.ok(sparse.includes('<span class="chip chip-unscored" title="rollout 3: unscored">–</span>'));
  assert.equal(count(sparse, 'chip-unscored'), 3);
  assert.doesNotMatch(sparse, />0\/0</);
  assert.doesNotMatch(sparse, />\?</);
  // A graded rollout keeps its fraction, even a real 0/0.
  assert.match(app.renderAccuracy(F.index, {task_id: 'x', accuracy: {kimi: {passed: 0, total: 1, rollouts: [{f2p: '0/0', p2p: '0/0', score: 0}]}}}),
               /<span class="chip chip-fail" title="rollout 1: fail · F2P 0\/0 · P2P 0\/0">0\/0<\/span>/);
});

test('task without a run: accuracy plus "not converted yet" with the queue rank', () => {
  const html = app.renderTaskView(F.index, 'go-git-worktree-merge-conflicts', null, {now: NOW});
  assert.match(html, /Not converted yet/);
  assert.match(html, /rank #4/);
  assert.match(html, /Kimi K3/);
  assert.doesNotMatch(html, /Generation tree|Round 0 at a glance/);
  const notEligible = app.renderTaskView(F.index, 'helm-array-merge-strategies', null, {now: NOW});
  assert.match(notEligible, /not in the eligible queue/);
});

test('converting task: run bar, failure patterns, plan, round 0 at a glance and the tree, in that order', () => {
  const html = app.renderTaskView(F.index, NAR, F.narwhals, {now: NOW});
  const order = ['Kimi K3', 'official1-narwhals', 'Target failure patterns', 'Plan', 'Round 0 at a glance', 'Generation tree'].map((s) => html.indexOf(s));
  assert.ok(order.every((v, i) => v >= 0 && (i === 0 || v > order[i - 1])), `section order ${order}`);
  assert.match(html, /ibis-project\/ibis/);
  assert.match(html, /round-3 probe \(113\/140 slots\)/);
  assert.match(html, /cpu1-00098/);
  assert.match(html, /20:35:29 left/);
  assert.match(html, /Log tail/);
  assert.match(html, /Authorization: Bearer \[REDACTED\]/);
  for (const name of ['Extending a narrow backend exception to a supported operation', 'Making tests conform to the unsupported assumption',
                      'Certifying interpolation arithmetic without testing the changed backend']) assert.ok(html.includes(name), name);
  assert.match(html, /Transferable signature/);
  assert.match(html, /Teacher contrast/);
  assert.match(html, /Full failure_analysis\.md/);
  assert.equal(count(html, 'BUILT</span>'), 4);                   // plan: c01-c04 built, c05 failed
  assert.match(html, /10 candidates · 4 built/);
});

test('generation tree: every decision badge, partial results while probing, both models\' infra, abandoned behind a toggle', () => {
  const html = app.renderTree(F.narwhals, {});
  for (const label of ['EVOLVE', 'ACCEPTED+EVOLVE', 'ACCEPTED', 'DISCARD', 'BUILD FAILED', 'NOT SELECTED', 'END', 'PENDING']) {
    assert.ok(html.includes(`<span class="dot" aria-hidden="true"></span>${label}</span></div>`), `card badge ${label}`);
  }
  assert.match(html, /Show abandoned \(5\)/);
  assert.equal(count(html, 'class="node tone-abandoned'), 0);
  assert.equal(count(html, 'data-node="'), 50);                  // 55 nodes, 5 abandoned hidden
  assert.equal(count(html, 'class="node tone-pending'), 5);
  assert.equal(count(html, 'is-probing'), 5);
  // F1: a probing card shows each model's partial results (graded slots), over a slim progress bar.
  assert.match(html, /<div class="node-meta">sc14 0 pass \/ 1 of 5 done<\/div><div class="node-meta">kimi 2 pass \/ 2 of 5 done<\/div>/);
  assert.match(html, /kimi 1 pass, 1 infra \/ 2 of 5 done/);      // cand-c04-r1b1-r2b2-r3b3
  assert.match(html, /sc14 0 pass \/ 0 of 5 done/);
  assert.equal(count(html, 'node-pbar'), 5);
  assert.match(html, /class="bar tone-pending node-pbar"[^>]*aria-label="probe progress 3\/10"/);
  assert.doesNotMatch(html, /probing \d+\/\d+/);                   // the old progress line is gone
  // ... and once its own slots are done, the results so far and why there is no score yet.
  assert.match(html, /<div class="node-meta">sc14 0\/5 · kimi 2\/5<\/div><div class="node-meta"><span class="muted">probe done, scoring when the round&#39;s probe ends<\/span><\/div>/);
  assert.match(html, /class="bar tone-good node-pbar"[^>]*aria-label="probe progress 10\/10"/);
  // F5: scored cards, one line per model, each with its infra count.
  assert.match(html, /<div class="node-meta">sc14 1\/5 · sim 6<\/div><div class="node-meta">kimi 4\/5 clean<\/div>/);
  assert.match(html, /<div class="node-meta">sc14 0\/1 \+4 infra<\/div><div class="node-meta">kimi 3\/3 clean \+2 infra<\/div>/);
  assert.match(html, /F2P 78 \/ P2P 4,903/);
  assert.match(html, /build failed after 2 attempts/);
  // F10: evolve cards carry their target pattern (numbered like the failure-pattern cards).
  assert.equal(count(html, 'class="tag tag-focus"'), 45);          // every shown evolve card
  assert.match(html, /<span class="tag tag-focus" title="target failure pattern: Making tests conform to the unsupported assumption">P2 · Making tests/);
  // Round headers (F6).
  assert.match(html, /113\/140 slots · 14 tasks/);
  assert.match(html, /1 abandoned probe/);
  assert.match(html, /seeds: 3 valid · <span class="txt-bad"[^>]*>1 invalid/);
  assert.equal(count(html, '<details class="lineage"'), 5);
  const shown = app.renderTree(F.narwhals, {showAbandoned: true});
  assert.match(shown, /Hide abandoned \(5\)/);
  assert.equal(count(shown, 'class="node tone-abandoned'), 5);
  assert.equal(count(shown, 'link-abandoned'), 5);
  assert.equal(count(shown, 'data-node="'), 55);
  const collapsed = app.renderTree(F.narwhals, {openState: {'lin:cand-c01': false}});
  assert.match(collapsed, /<details class="lineage" data-key="lin:cand-c01"><summary>/);
  assert.match(collapsed, /<details class="lineage" data-key="lin:cand-c03" open>/);
});

test('liveProbe: NODE.probe.partial and own_done, the older progress shape, nothing once scored', () => {
  const r3b2 = nodeOf(F.narwhals, 'cand-c04-r1b1-r2b2-r3b2');
  assert.deepEqual(app.liveProbe(r3b2), {rows: [{model: 'sc14', done: 1, total: 5, pass: 0, fail: 1, infra: 0},
                                                {model: 'kimi', done: 2, total: 5, pass: 2, fail: 0, infra: 0}],
                                         done: 3, total: 10, ownDone: false});
  assert.equal(app.liveProbe(nodeOf(F.narwhals, 'cand-c04-r1b1-r2b2-r3b0')).ownDone, true);
  assert.equal(app.partialText({model: 'sc14', done: 3, total: 5, pass: 1, fail: 1, infra: 1}), 'sc14 1 pass, 1 infra / 3 of 5 done');
  assert.equal(app.partialText({model: 'sc14', done: 3, total: 5, pass: 1, fail: 2, infra: 0}), 'sc14 1 pass / 3 of 5 done');
  // Older collectors: probe.progress with the so-far counts give the same rows; all slots done counts as own_done.
  const old = olderShapes(F.narwhals);
  for (const n of F.narwhals.nodes.filter((x) => x.status === 'probing')) {
    assert.deepEqual(app.liveProbe(nodeOf(old, n.id)), app.liveProbe(n), n.id);
  }
  const partly = clone(r3b2);
  partly.probe.own_done = false;
  partly.probe.partial = {sc14: partly.probe.partial.sc14};      // kimi from probe.progress + the teacher counts
  assert.deepEqual(app.liveProbe(partly).rows[1], {model: 'kimi', done: 2, total: 5, pass: 2, fail: 0, infra: 0});
  const explicit = clone(nodeOf(old, 'cand-c04-r1b1-r2b2-r3b0'));
  explicit.probe.own_done = false;                                // an explicit flag wins over the inference
  assert.equal(app.liveProbe(explicit).ownDone, false);
  for (const id of ['cand-c04', 'cand-c05', 'cand-c02-r1b0']) assert.equal(app.liveProbe(nodeOf(F.narwhals, id)), null, id);
  assert.equal(app.liveProbe(nodeOf(F.bandit, 'cand-c01-r1b4-r2b1')), null);   // being built: no probe
  for (const weird of [null, {}, {probe: {}}, {probe: {progress: {sc14: 'x'}, partial: {sc14: {done: 'a'}}}}]) {
    assert.equal(app.liveProbe(weird), null);
  }
});

test('probing cards and drawers read the older collector shapes the same way', () => {
  const old = olderShapes(F.narwhals);
  for (const n of F.narwhals.nodes.filter((x) => x.status === 'probing')) {
    const now = cardOf(F.narwhals, n.id), before = cardOf(old, n.id);
    const lines = (h) => h.match(/<div class="node-meta">.*?<\/div>(?=<div|<span class="bar)/g).slice(1);
    assert.deepEqual(lines(before), lines(now), n.id);
    assert.doesNotMatch(before, /tag-focus/);                      // no focus pattern recorded
  }
});

test('tree layout: columns are rounds, children one column right, parents centered, no overlaps', () => {
  for (const ui of [{}, {showAbandoned: true}]) {
    const L = app.treeLayout(F.narwhals, ui);
    assert.equal(L.cols, 4);
    assert.deepEqual(L.lineages.map((l) => l.root), ['cand-c01', 'cand-c02', 'cand-c03', 'cand-c04', 'cand-c05']);
    const pitch = L.geom.cardH + L.geom.rowGap;
    for (const lin of L.lineages) {
      const pos = new Map(lin.items.map((it) => [it.id, it]));
      for (const it of lin.items) assert.equal(it.col, it.node.round, `${it.id} column = round`);
      const kids = new Map();
      for (const e of lin.edges) {
        assert.equal(pos.get(e.to).col, pos.get(e.from).col + 1);
        kids.set(e.from, (kids.get(e.from) || []).concat(pos.get(e.to).y));
      }
      for (const [parent, ys] of kids) {
        assert.equal(pos.get(parent).y, (Math.min(...ys) + Math.max(...ys)) / 2, `${parent} centered on its children`);
      }
      const byCol = {};
      for (const it of lin.items) (byCol[it.col] = byCol[it.col] || []).push(it.y);
      for (const ys of Object.values(byCol)) {
        ys.sort((a, b) => a - b);
        for (let i = 1; i < ys.length; i += 1) assert.ok(ys[i] - ys[i - 1] >= pitch - 1e-9, 'cards overlap');
      }
    }
  }
  assert.equal(app.TREE.cardH, 96, 'five 16 px lines');
  assert.equal(app.treeLayout(F.narwhals, {treeWidth: 1900}).geom.cardW, 360);
  assert.equal(app.treeLayout(F.narwhals, {treeWidth: 1110}).geom.cardW, 247);
  assert.equal(app.treeLayout(F.narwhals, {treeWidth: 700}).geom.cardW, 220);
  // Inconsistent data: a cycle, a dangling parent, duplicate ids and a self-parent still place every node once.
  const broken = {max_rounds: 1, nodes: [{id: 'a', parent: 'b', children: ['b']}, {id: 'b', parent: 'a', children: ['a']},
    {id: 'c', parent: 'missing', round: 2}, {id: 'c', title: 'dup'}, {id: 'd', parent: 'd'}, {id: 'e', round: 'x', children: ['zz', 5]}]};
  const L = app.treeLayout(broken, {});
  const ids = L.lineages.flatMap((l) => l.items.map((it) => it.id)).sort();
  assert.deepEqual(ids, ['a', 'b', 'c', 'd', 'e']);
});

test('data strings never resolve to inherited properties', () => {
  const node = {id: 'cand-x', title: 't', status: 'constructor', decision: 'toString', build: {}, probe: {progress: {}, partial: {constructor: {}}},
                focus_pattern: 'toString'};
  const html = app.renderNodeCard({node, x: 0, y: 0}, {}, null, {focus: new Map()}) + app.renderNodeDetail(node, {nodes: [node]}, {}, NOW) +
    app.renderTree({nodes: [node], rounds: [{index: 0, status: 'hasOwnProperty'}]}, {}) + app.renderRoundZero({nodes: [Object.assign({round: 0}, node)]}, {}) +
    app.renderTaskMenu({tasks: [{task_id: 'a', run: 'official-z'}], runs: [{run: 'official-z', task_id: 'a', status: '__proto__'}]}, '', 'a', 0) +
    app.renderWorkers({workers: [{slot: '01', registry_status: 'valueOf'}]}, NOW);
  assert.doesNotMatch(html, /function|native code|\[object/);
  assert.match(html, /TOSTRING/);                                  // unknown decision shown as-is, neutral
});

test('node detail drawer shows every NODE field', () => {
  const byId = (id) => nodeOf(F.narwhals, id);
  const acc = app.renderNodeDetail(byId('cand-c04'), F.narwhals, {}, NOW);
  assert.match(acc, /cand-c04/);
  assert.match(acc, /ACCEPTED\+EVOLVE/);
  assert.match(acc, /accepted \(similarity 6\/10\); every valid round-0 task seeds 5 branches/);
  assert.match(acc, /Build &amp; gates/);
  assert.match(acc, /F2P tests <span class="muted small">60 of 78/);
  assert.match(acc, /P2P tests \(sample\)/);
  assert.equal(count(acc, '<td class="err-cell">'), 10);          // 10 rollouts
  assert.match(acc, /<td><code>sc14-r0<\/code><\/td>/);
  assert.match(acc, /kimi-r4/);
  assert.match(acc, /6\/10 \(accept ≥ 6\)/);
  assert.match(acc, /Similarity detail · score 6\/10/);
  assert.match(acc, /Missing target patterns/);
  for (const label of ['Instruction (instruction.md)', 'Rubric (rubric.md)', 'Teacher audit (teacher_audit.json)', 'Synthetic failure analysis']) {
    assert.ok(acc.includes(`<summary>${app.esc(label)}`), label);
  }
  assert.match(acc, /Evolve plan \(evolve_plan\.md\): not available/);
  assert.match(acc, /children<\/span><button type="button" class="node-link" data-node="cand-c04-r1b0" data-fkey="link:cand-c04-r1b0">/);
  assert.match(acc, /<details class="fold fold-inline" data-key="nd:cand-c04:instruction_md">/);   // collapsed by default
  assert.doesNotMatch(acc, /class="rel-k">targets/);              // a mask task targets no single pattern

  const failed = app.renderNodeDetail(byId('cand-c05'), F.narwhals, {}, NOW);
  assert.match(failed, /BUILD FAILED/);
  assert.match(failed, /Gate \/ author problems/);
  assert.match(failed, /data-key="nd:cand-c05:problems" open/);
  assert.match(failed, /Never probed: the build failed/);
  assert.match(failed, /Not scored yet/);

  const probing = app.renderNodeDetail(byId('cand-c04-r1b1-r2b2-r3b2'), F.narwhals, {}, NOW);
  assert.match(probing, /PENDING/);
  assert.match(probing, /probe in progress/);
  assert.match(probing, /so far/);
  assert.match(probing, /not audited yet/);
  assert.match(probing, /round-3\/probe-5e1f0c2a \(running\)/);
  assert.match(probing, /<span class="prog-v">1\/5 done · 0 pass · 1 fail<\/span>/);
  assert.match(probing, /parent<\/span><button type="button" class="node-link" data-node="cand-c04-r1b1-r2b2"/);
  assert.doesNotMatch(probing, /Probe done for this task/);
  // F1: own slots done, no score yet.
  const waiting = app.renderNodeDetail(byId('cand-c04-r1b1-r2b2-r3b0'), F.narwhals, {}, NOW);
  assert.match(waiting, /Probe done for this task&#39;s own slots: it is scored when the round&#39;s probe ends\./);
  assert.match(waiting, /<span class="prog-v">5\/5 done · 0 pass · 5 fail<\/span>/);
  // F10: the evolve branch's target pattern.
  assert.match(waiting, /targets<\/span><span class="tag tag-focus tag-wrap"[^>]*>P1 · Extending a narrow backend exception to a supported operation<\/span>/);

  const abandoned = app.renderNodeDetail(byId('cand-c02-r1b0'), F.narwhals, {}, NOW);
  assert.match(abandoned, /ABANDONED/);
  assert.match(abandoned, /parent cand-c02 is not a valid seed/);
  assert.match(app.renderNodeDetail(null, F.narwhals, {}, NOW), /node not found/);
});

test('teacher audit: the list as a table (open when a slot cheated), the older string as text', () => {
  const cheat = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c04-r1b0'), F.narwhals, {}, NOW);
  assert.match(cheat, /<details class="fold fold-inline" data-key="nd:cand-c04-r1b0:teacher_audit" open><summary>Teacher audit \(teacher_audit\.json\) <span class="muted small">5 slots · 1 cheated<\/span>/);
  assert.match(cheat, /<th>slot<\/th><th>cheated<\/th><th>categories<\/th><th>summary<\/th>/);
  assert.match(cheat, /<tr><td><code>kimi-r0<\/code><\/td><td><span class="badge tone-bad badge-sm"><span class="dot" aria-hidden="true"><\/span>yes<\/span><\/td><td><span class="tag">answer_retrieval<\/span> <span class="tag">test_tampering<\/span><\/td><td>kimi-r0 fetched the upstream/);
  assert.match(cheat, /<tr><td><code>kimi-r1<\/code><\/td><td>no<\/td><td>–<\/td>/);
  assert.match(cheat, /Accepted<\/dt><dd><span class="badge tone-muted">/);
  const clean = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c01'), F.narwhals, {}, NOW);
  assert.match(clean, /data-key="nd:cand-c01:teacher_audit"><summary>Teacher audit \(teacher_audit\.json\) <span class="muted small">5 slots · 0 cheated/);
  // Older collectors: the JSON-dumped teacher_audit.json, pretty-printed.
  const old = app.renderNodeDetail(nodeOf(F.bandit, 'cand-c01'), F.bandit, {}, NOW);
  assert.match(old, /<pre class="pre pre-md" data-scroll="nd:cand-c01:teacher_audit">\{\n  &quot;kimi-r0&quot;: \{/);
  assert.match(old, /Teacher audit \(teacher_audit\.json\) <span class="muted small">[\d,]+ chars/);
  assert.doesNotMatch(old, /<th>cheated<\/th>/);
  // Odd entries: no slots, categories as a string, truncated JSON, nothing.
  assert.match(app.renderAudit([], 'k', {}), /0 slots · 0 cheated.*No teacher pass was audited/);
  assert.match(app.renderAudit([{slot: 'kimi-r2', categories: 'oddity', cheated: null}], 'k', {}), /<td>–<\/td><td>oddity<\/td><td>–<\/td><\/tr>/);
  // The collector also passes the judge's rubric score: a column when any slot has one.
  const scored = app.renderAudit([{slot: 'kimi-r0', cheated: false, categories: [], summary: 's', rubric_score: 85}, {slot: 'kimi-r1', cheated: false}], 'k', {});
  assert.match(scored, /<th class="num" title="the judge&#39;s process rubric score">rubric<\/th>/);
  assert.match(scored, /<td>s<\/td><td class="num">85<\/td><\/tr>/);
  assert.match(scored, /<td>–<\/td><td class="num">–<\/td><\/tr>/);
  assert.doesNotMatch(cheat, />rubric<\/th>/);                     // no rubric scores in the fixture: no column
  assert.match(app.renderAudit('{"kimi-r0": {"chea', 'k', {}), /<pre[^>]*>\{&quot;kimi-r0&quot;: \{&quot;chea<\/pre>/);
  assert.match(app.renderAudit(null, 'k', {}), /Teacher audit \(teacher_audit\.json\): not available/);
});

test('rollouts table: long errors behind a <details> with the full text, short ones inline', () => {
  const infra = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c01-r1b3'), F.narwhals, {}, NOW);
  assert.match(infra, /<table class="tbl tbl-compact tbl-roll">/);
  assert.match(infra, /<details class="fold fold-inline fold-err" data-key="nd:cand-c01-r1b3:rollouts:err:sc14-r2"><summary><span class="err">agent: AgentTimeoutError: agent exceeded 10800 s<\/span><\/summary>/);
  assert.match(infra, /<pre class="pre pre-err" data-scroll="nd:cand-c01-r1b3:rollouts:err:sc14-r2">agent: AgentTimeoutError: agent exceeded 10800 s\nTraceback \(most recent call last\):\n  File &quot;\/app\/harness\/run_agent\.py&quot;, line 412/);
  const verifier = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c02'), F.narwhals, {}, NOW);
  const v = 'verifier: test.patch failed to apply: patch does not apply to ibis/backends/tests/test_window.py';
  assert.ok(verifier.includes(`<summary><span class="err">${v.slice(0, 71)}…</span></summary>`), 'summary cut, not the text');
  assert.ok(verifier.includes(`>${v}</pre>`), 'the full error in the <details>');
  const live = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c04-r1b1-r2b2-r3b3'), F.narwhals, {}, NOW);
  assert.match(live, /<td class="err-cell"><span class="err">verifier: Verifier did not produce \/logs\/verifier\/reward\.json<\/span><\/td>/);
  // Open state follows ui.openState like every collapsible.
  const opened = app.renderRollouts(nodeOf(F.narwhals, 'cand-c02').probe.rollouts, {openState: {'k:err:kimi-r3': true}}, 'k');
  assert.match(opened, /data-key="k:err:kimi-r3" open>/);
  assert.match(app.renderRollouts([], {}, 'k'), /No rollouts recorded/);
});

test('round 0 at a glance: one row per round-0 task, partial results while probing, rows open the drawer', () => {
  const html = app.renderRoundZero(F.narwhals, {selected: 'cand-c03'});
  assert.match(html, /<h2>Round 0 at a glance<\/h2><span class="muted small">5 tasks · 3 seeds · 1 accepted · 2 discarded · click a row for its details<\/span>/);
  assert.equal(count(html, '<tr class="r0-row'), 5);
  assert.match(html, /<tr class="r0-row selected" data-node="cand-c03">/);
  assert.match(html, /<tr class="r0-row" data-node="cand-c02"><td><button type="button" class="node-link" data-node="cand-c02" data-fkey="r0:cand-c02">c02<\/button><\/td>/);
  // name, title, F2P/P2P, sc14 x/5, clean kimi x/5, infra (sc14 · kimi), decision, reason
  assert.match(html, /c02<\/button><\/td><td class="r0-title">Null-aware value-selecting reductions with ordering and keys<\/td><td class="num">76 \/ 4,044<\/td><td class="num">0\/1<\/td><td class="num">3\/3<\/td><td class="num">4 · 2<\/td><td><span class="badge tone-bad badge-sm"[^>]*><span class="dot" aria-hidden="true"><\/span>DISCARD<\/span><\/td><td class="r0-reason">verifier produced no grade for 6 rollout\(s\)/);
  assert.match(html, /c04<\/button><\/td><td class="r0-title">[^<]*<\/td><td class="num">78 \/ 4,903<\/td><td class="num">1\/5<\/td><td class="num">4\/5<\/td><td class="num">0 · 0<\/td><td>[^]*?ACCEPTED\+EVOLVE/);
  assert.match(html, /c05<\/button><\/td><td class="r0-title">[^<]*<\/td><td class="num"><span class="txt-bad">build failed<\/span><\/td><td class="num">–<\/td><td class="num">–<\/td><td class="num">–<\/td>/);
  // A round-0 task still probing (early in a run): its partial results.
  const early = clone(F.narwhals);
  const c01 = nodeOf(early, 'cand-c01');
  Object.assign(c01, {score: null, status: 'probing', decision: 'pending', decision_reason: 'probe in progress'});
  c01.probe = {live: true, progress: {sc14: [3, 5], kimi: [2, 5]}, rollouts: [],
               partial: {sc14: {done: 3, total: 5, pass: 1, fail: 1, infra: 1}, kimi: {done: 2, total: 5, pass: 2, fail: 0, infra: 0}}, own_done: false};
  const live = app.renderRoundZero(early, {});
  assert.match(live, /<td class="num">1 pass \/ 3 of 5<\/td><td class="num">2 pass \/ 2 of 5 <span class="muted small">not audited<\/span><\/td><td class="num">1 · 0<\/td>/);
  assert.equal(app.renderRoundZero(F.fd, {}), '');                 // no candidates yet: no table
  assert.equal(app.renderRoundZero(null, {}), '');
});

test('failure patterns: how many evolve branches targeted each pattern and how many were accepted', () => {
  const tally = app.focusTally(F.narwhals.nodes);
  assert.deepEqual(tally.get('Making tests conform to the unsupported assumption'), {targeted: 18, accepted: 3, open: 2});
  assert.deepEqual([...tally.keys()].sort(), [...F.narwhals.failure_analysis.patterns.map((p) => p.name)].sort());
  const html = app.renderFailureAnalysis(F.narwhals.failure_analysis, {}, F.narwhals.nodes);
  assert.match(html, /<div class="pattern-targets"><span class="tag tag-focus">P2<\/span>18 branches targeted it · 3 accepted · 2 in progress<\/div>/);
  assert.equal(count(html, 'class="pattern-targets'), 3);
  // A target that is no pattern of the analysis ("overall": result.json without patterns) is listed apart.
  const nodes = clone(F.narwhals.nodes);
  nodeOf({nodes}, 'cand-c01-r1b0').focus_pattern = 'overall';
  const other = app.renderFailureAnalysis(F.narwhals.failure_analysis, {}, nodes);
  assert.match(other, /other evolve targets<\/span><ul class="plain"><li><span class="tag tag-focus">overall<\/span> 1 branch targeted it · 0 accepted<\/li>/);
  // Older collectors (no focus_pattern): no counts at all.
  assert.doesNotMatch(app.renderFailureAnalysis(F.bandit.failure_analysis, {}, F.bandit.nodes), /pattern-targets|targeted/);
  assert.doesNotMatch(app.renderFailureAnalysis(F.narwhals.failure_analysis, {}), /pattern-targets/);
  const none = app.renderFailureAnalysis(F.narwhals.failure_analysis, {}, nodes.filter((n) => n.focus_pattern !== 'Certifying interpolation arithmetic without testing the changed backend'));
  assert.match(none, /P3<\/span>no evolve branch targeted it yet/);
});

test('pending run without a tree (sparse but valid)', () => {
  const html = app.renderTaskView(F.index, 'fd-deterministic-multi-key-sorting', F.fd, {now: NOW});
  assert.match(html, /official2-fd/);
  assert.match(html, /PENDING/);
  assert.match(html, /pending: target failure analysis/);
  assert.match(html, /not started/);
  assert.match(html, /target failure analysis is not available yet/);
  assert.match(html, /No candidate tasks yet/);
  assert.match(html, /No log lines yet/);
});

test('bandit run (older collector shapes): round 2 being built, seeds with their invalid count', () => {
  const html = app.renderTaskView(F.index, BAN, F.bandit, {now: NOW});
  assert.match(html, /authoring…/);
  assert.match(html, /planning…/);
  assert.match(html, /built · waiting for probe/);
  assert.match(html, /building…/);
  assert.match(html, /round-2 building \(4\/25 built\)/);
  // F6: "seeds: N valid · 0 invalid", and a building round shows "built x / y".
  assert.match(html, /seeds: 5 valid · 0 invalid/);
  assert.match(html, /seeds: 25 valid · 0 invalid/);
  assert.match(html, /<div class="round-probe"><span class="bar tone-pending"[^>]*aria-valuemax="25" aria-valuenow="4"[^>]*>.*?<\/span><span>built 4 \/ 25<\/span><\/div>/);
  assert.match(html, /250\/250 slots · 25 tasks/);
  assert.match(html, /Suppressing the report instead of suppressing findings/);
  assert.match(html, /gates attempt 0: 3 F2P tests pass on the starting state/);   // the round-2 build failure
  assert.equal(count(html, 'is-probing'), 0);
  assert.doesNotMatch(html, /tag-focus|pattern-targets/);           // no focus_pattern in this shape
  assert.match(html, /Round 0 at a glance/);
  // rounds[] without nodes/built (older collectors): counted from the tree.
  const old = clone(F.bandit);
  for (const r of old.rounds) { delete r.nodes; delete r.built; }
  assert.match(app.renderTree(old, {}), /built 4 \/ 25/);
  const noSeeds = clone(F.bandit);
  noSeeds.rounds[0].seeds = {valid: ['cand-c01'], invalid: {}};
  assert.match(app.renderTree(noSeeds, {}), /seeds: 1 valid · 0 invalid/);
});

// ---------------------------------------------------------------------- failure-match audit (v1.3, v1.4)

/** One audit as collect.py attaches it (NODE.failure_match): does sc14 repeat the target's failure patterns? */
function audit(verdict) {
  return {
    verdict, summary: `sc14 ${verdict}: it extends the narrow exception in 3 of 4 failed rollouts.`,
    patterns: [
      {name: 'Extending a narrow backend exception to a supported operation', reproduced: 'yes',
       slots: ['sc14-r1', 'sc14-r2', 'sc14-r4'], evidence: 'sc14-r1 widened the NotImplementedError guard to every frame.'},
      {name: 'Making tests conform to the unsupported assumption', reproduced: 'partial', slots: ['sc14-r2'],
       evidence: 'sc14-r2 edited test_window.py to expect the error.'},
      {name: 'Certifying interpolation arithmetic without testing the changed backend', reproduced: 'no', slots: [], evidence: null},
      {name: 'A pattern the target never had', reproduced: null, slots: [], evidence: null},
    ],
    student_rollouts: [
      {slot: 'sc14-r0', status: 'pass', failing_tests: [], cause: null, pattern: null},
      {slot: 'sc14-r1', status: 'fail', cause: 'raised NotImplementedError for every frame',
       failing_tests: ['ibis/backends/tests/test_window.py::test_rolling[duckdb]', 'ibis/backends/tests/test_window.py::test_rolling[polars]'],
       pattern: 'Extending a narrow backend exception to a supported operation'},
      {slot: 'sc14-r3', status: 'infra', failing_tests: [], cause: 'the sandbox was lost', pattern: null},
    ],
    other_causes: 'sc14-r3: the sandbox died mid-run.',
    similarity_judge: {score: 7, agrees: true, comment: 'the judge saw the same exception widening'},
    audited_at: '2026-10-18T20:15:00-07:00',
  };
}
// Four of the five accepted narwhals tasks are audited, one per verdict; cand-c01-r1b2-r2b1-r3b1 is not (yet).
const AUDITED = {'cand-c04': 'strong', 'cand-c01-r1b4': 'partial', 'cand-c04-r1b2': 'weak', 'cand-c03-r1b3-r2b1': 'none'};
/** A task file as a v1.3 collector writes it: failure_match on every node (the audit, or null). */
function withAudits(doc) {
  const d = clone(doc);
  for (const n of d.nodes) n.failure_match = AUDITED[n.id] ? audit(AUDITED[n.id]) : null;
  return d;
}
const A = withAudits(F.narwhals);
const AI = clone(F.index);                                         // ... and its index: runs[].audit
AI.runs.find((r) => r.run === 'official1-narwhals').audit = {accepted: 5, audited: 4, strong: 1, partial: 1, weak: 1, none: 1};
const FA_TONES = {strong: 'good', partial: 'pending', weak: 'serious', none: 'bad'};

/** v1.4, the audit's scope: a task file as a v1.4 collector writes it. NODE.audit_eligible as collect.py computes it from
    what the fixture shows (scored, a graded sc14 rollout, not discarded: 37 tasks), NODE.withdrawn; on top of A's audits,
    those of three eligible tasks that were not accepted (rounds 0, 1, 3), of a discarded task (cand-c02), and of
    cand-c03-r1b0, which main withdrew after its audit (36 eligible tasks left, 7 of them audited). */
const MORE = {'cand-c01': 'weak', 'cand-c04-r1b1': 'strong', 'cand-c03-r1b3-r2b0-r3b0': 'none'};
const WITHDRAWN = {at: '2026-10-18T23:03:14-0700', by: 'main',
                   why: 'grading artifact: sc14-r2 only failed a P2P test that test.patch never resets'};
function withScope(doc) {
  const d = withAudits(doc);
  for (const n of d.nodes) {
    const graded = ((n.probe || {}).rollouts || []).some((r) => r.model === 'sc14' && (r.status === 'pass' || r.status === 'fail'));
    n.audit_eligible = !!n.score && n.decision !== 'discard' && graded;
    n.withdrawn = null;
    if (MORE[n.id]) n.failure_match = audit(MORE[n.id]);
  }
  Object.assign(nodeOf(d, 'cand-c03-r1b0'), {audit_eligible: false, withdrawn: clone(WITHDRAWN), failure_match: audit('partial')});
  nodeOf(d, 'cand-c02').failure_match = audit('weak');
  return d;
}
const E = withScope(F.narwhals);
const EI = clone(AI);                                              // ... and its index: the eligible counts too
Object.assign(EI.runs.find((r) => r.run === 'official1-narwhals').audit,
              {eligible: 36, audited_eligible: 7, eligible_verdicts: {strong: 2, partial: 1, weak: 2, none: 2}});
const ELIGIBLE = E.nodes.filter((n) => n.audit_eligible).map((n) => n.id);

/** Every view that shows failure-match audits, for one task document: the task view, the table, the run bar (with
    the index's counts and counted from the nodes), cards and drawers (focus: only of the nodes an audit can show on,
    accepted, eligible or carrying one, which keeps the broken-data loops fast). */
function renderAudits(d, index, focus) {
  const ix = index === undefined ? AI : index;
  const all = d && typeof d === 'object' && Array.isArray(d.nodes) ? d.nodes : [];
  const nodes = !focus ? all : all.filter((n) => n && typeof n === 'object' &&
                                                (n.failure_match || n.audit_eligible === true || ['accepted', 'accepted_evolve'].includes(n.decision)));
  const run = ix && typeof ix === 'object' && Array.isArray(ix.runs) ? ix.runs[0] : null;
  return {
    view: app.renderTaskView(ix, NAR, d, {now: NOW, ui: {showAbandoned: true, selected: 'cand-c04'}}),
    table: app.renderAuditTable(d, {selected: 'cand-c04'}),
    bar: app.renderRunBar(run, d, NOW, {}) + app.renderRunBar(null, d, NOW, {}),
    cards: nodes.map((n) => app.renderNodeCard({node: n, x: 0, y: 0}, {}, null, app.treeContext(d))).join('\n'),
    drawers: nodes.map((n) => app.renderNodeDetail(n, d, {}, NOW)).join('\n'),
  };
}

test('failure-match audit (v1.3 collectors): accepted cards end with an "FA match" badge in the verdict\'s color, grey when not audited', () => {
  for (const [id, verdict] of Object.entries(AUDITED)) {
    const card = cardOf(A, id);
    assert.ok(card.includes(`<span class="badge tone-${FA_TONES[verdict]} badge-xs badge-fa" title="`), `${id}: ${verdict} tone`);
    assert.ok(card.includes(`<span class="dot" aria-hidden="true"></span>FA match: ${verdict}</span></div>`), `${id}: badge label`);
    assert.match(card, /<div class="node-meta node-l5"><span class="node-l5-text">kimi \d\/\d clean<\/span><span class="badge /);
    assert.ok(card.includes(`FA match: ${verdict} · sc14 ${verdict}: it extends the narrow exception`), `${id}: tooltip summary`);
  }
  assert.ok(cardOf(A, 'cand-c01-r1b2-r2b1-r3b1').includes('<span class="badge tone-muted badge-xs badge-fa" title="no failure-match ' +
    'audit of this task yet"><span class="dot" aria-hidden="true"></span>FA match: not audited</span></div>'));
  assert.equal(count(app.renderTree(A, {showAbandoned: true}), 'badge-fa'), 5);   // the five accepted tasks only
  // The other cards are unchanged; a collector that predates the audit (no failure_match) gives no badge at all.
  for (const n of F.narwhals.nodes.filter((x) => !AUDITED[x.id] && x.id !== 'cand-c01-r1b2-r2b1-r3b1')) {
    assert.equal(cardOf(A, n.id), cardOf(F.narwhals, n.id), n.id);
  }
  assert.equal(count(app.renderTree(F.narwhals, {showAbandoned: true}), 'badge-fa'), 0);
  // An audit of a task that was not accepted shows in its drawer, never as a card badge.
  const d = clone(A);
  nodeOf(d, 'cand-c01').failure_match = audit('weak');
  assert.doesNotMatch(cardOf(d, 'cand-c01'), /badge-fa|FA match/);
  assert.match(app.renderNodeDetail(nodeOf(d, 'cand-c01'), d, {}, NOW), /Failure-pattern match \(audit\)[^]*FA match: weak/);
  assert.equal(app.faVerdict(null).label, 'not audited');
  assert.equal(app.faVerdict({verdict: ' Strong '}).key, 'strong');
  assert.equal(app.faVerdict({verdict: 'constructor'}).key, 'unknown');           // never an inherited property
  assert.equal(app.failureMatch({failure_match: []}), null);
  assert.equal(app.failureMatch({}), undefined);
});

test('failure-match audit: the drawer section (verdict, summary, per-pattern and per-rollout tables, other causes, judge)', () => {
  const html = app.renderNodeDetail(nodeOf(A, 'cand-c04'), A, {}, NOW);
  assert.match(html, /<h3 class="drawer-h">Failure-pattern match \(audit\)<\/h3><div class="callout tone-good"><div class="callout-k">Verdict<\/div><p><span class="badge tone-good"[^>]*><span class="dot" aria-hidden="true"><\/span>FA match: strong<\/span>/);
  assert.ok(html.includes('<p>sc14 strong: it extends the narrow exception in 3 of 4 failed rollouts.</p></div>'));
  const at = (s) => html.indexOf(s);
  assert.ok(at('>Score<') < at('Failure-pattern match (audit)') && at('Failure-pattern match (audit)') < at('>Documents<'), 'after the score');
  // One row per target pattern: numbered like the pattern cards, reproduced, the sc14 slots, the evidence.
  assert.match(html, /<th>target pattern<\/th><th>reproduced<\/th><th>sc14 slots<\/th><th>evidence<\/th>/);
  assert.ok(html.includes('<tr><td><span class="tag tag-fa tag-wrap">P1 · Extending a narrow backend exception to a supported operation</span></td>' +
    '<td><span class="badge tone-good badge-sm"><span class="dot" aria-hidden="true"></span>yes</span></td>' +
    '<td><span class="slot">sc14-r1</span> <span class="slot">sc14-r2</span> <span class="slot">sc14-r4</span></td>' +
    '<td>sc14-r1 widened the NotImplementedError guard to every frame.</td></tr>'));
  assert.match(html, /P2 · Making tests conform to the unsupported assumption<\/span><\/td><td><span class="badge tone-pending badge-sm"><span class="dot" aria-hidden="true"><\/span>partial<\/span>/);
  assert.match(html, /P3 · Certifying[^<]*<\/span><\/td><td><span class="badge tone-muted badge-sm"><span class="dot" aria-hidden="true"><\/span>no<\/span><\/td><td>–<\/td><td>–<\/td><\/tr>/);
  assert.match(html, /<td><span class="tag tag-fa tag-wrap">A pattern the target never had<\/span><\/td><td>–<\/td>/);   // no P number
  // One row per sc14 rollout: status, failing tests, cause, matched pattern.
  assert.match(html, /<th>slot<\/th><th>status<\/th><th>failing tests<\/th><th>cause<\/th><th>matched pattern<\/th>/);
  assert.ok(html.includes('<tr><td><code>sc14-r1</code></td><td><span class="badge tone-bad badge-sm"><span class="dot" aria-hidden="true"></span>fail</span></td>' +
    '<td><ul class="fa-tests"><li><code>ibis/backends/tests/test_window.py::test_rolling[duckdb]</code></li>' +
    '<li><code>ibis/backends/tests/test_window.py::test_rolling[polars]</code></li></ul></td>' +
    '<td>raised NotImplementedError for every frame</td><td><span class="tag tag-fa tag-wrap">P1 · Extending a narrow'));
  assert.ok(html.includes('<tr><td><code>sc14-r0</code></td><td><span class="badge tone-good badge-sm"><span class="dot" aria-hidden="true"></span>pass</span></td><td>–</td><td>–</td><td>–</td></tr>'));
  assert.match(html, /<code>sc14-r3<\/code><\/td><td><span class="badge tone-muted badge-sm"><span class="dot" aria-hidden="true"><\/span>infra<\/span>/);
  // The similarity judge's score and whether the auditor agrees; other causes; when.
  assert.ok(html.includes('<dt>Similarity judge</dt><dd>7/10 <span class="badge tone-good badge-sm" title="the auditor agrees with the similarity judge">' +
    '<span class="dot" aria-hidden="true"></span>agrees</span><div class="muted small">the judge saw the same exception widening</div></dd>'));
  assert.ok(html.includes('<dt>Other causes</dt><dd>sc14-r3: the sandbox died mid-run.</dd>'));
  assert.ok(html.includes(`<dt>Audited</dt><dd>${app.fmtDate(Date.parse('2026-10-18T20:15:00-07:00') / 1000)} <span class="muted small">(`));
  for (const k of ['nd:cand-c04:failure_match:patterns', 'nd:cand-c04:failure_match:rollouts']) assert.ok(html.includes(`data-scroll="${k}"`), k);
  // Other verdicts, a judge the auditor disagrees with.
  const d = clone(A);
  Object.assign(nodeOf(d, 'cand-c04-r1b2').failure_match.similarity_judge, {agrees: false, score: 6.5});
  const weak = app.renderNodeDetail(nodeOf(d, 'cand-c04-r1b2'), d, {}, NOW);
  assert.match(weak, /<div class="callout tone-serious"><div class="callout-k">Verdict<\/div><p><span class="badge tone-serious"[^]*?FA match: weak/);
  assert.match(weak, /6\.5\/10 <span class="badge tone-bad badge-sm"[^>]*><span class="dot" aria-hidden="true"><\/span>disagrees<\/span>/);
  assert.match(app.renderNodeDetail(nodeOf(A, 'cand-c01-r1b4'), A, {}, NOW), /callout tone-pending[^]*?FA match: partial/);
  assert.match(app.renderNodeDetail(nodeOf(A, 'cand-c03-r1b3-r2b1'), A, {}, NOW), /callout tone-bad[^]*?FA match: none/);
  // An accepted task without an audit says so; other nodes, and collectors that predate the audit, have no section.
  assert.ok(app.renderNodeDetail(nodeOf(A, 'cand-c01-r1b2-r2b1-r3b1'), A, {}, NOW).includes('Failure-pattern match (audit)</h3><p class="note">' +
    'Not audited yet: no failure-match audit of this accepted task (runs/&lt;run&gt;/audit/failure_match/&lt;task_id&gt;.json).</p>'));
  assert.doesNotMatch(app.renderNodeDetail(nodeOf(A, 'cand-c01'), A, {}, NOW), /Failure-pattern match/);
  assert.doesNotMatch(app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c04'), F.narwhals, {}, NOW), /Failure-pattern match/);
  // A bare audit: dashes and notes instead of empty tables; an unknown verdict is neutral.
  const bare = clone(A);
  nodeOf(bare, 'cand-c04').failure_match = {verdict: 'strong'};
  const b = app.renderNodeDetail(nodeOf(bare, 'cand-c04'), bare, {}, NOW);
  assert.match(b, /No per-pattern findings recorded[^]*No per-rollout findings recorded/);
  assert.ok(b.includes('<dt>Similarity judge</dt><dd>–</dd><dt>Other causes</dt><dd><span class="muted">none recorded</span></dd><dt>Audited</dt><dd>–</dd>'));
  nodeOf(bare, 'cand-c04').failure_match = {verdict: 'moderate', audited_at: 'yesterday'};
  assert.match(app.renderNodeDetail(nodeOf(bare, 'cand-c04'), bare, {}, NOW), /callout tone-neutral[^]*?FA match: unknown[^]*<dt>Audited<\/dt><dd>yesterday<\/dd>/);
});

test('failure-match audit (v1.3 collectors): the table of the accepted tasks above the tree, and "audit n/m" in the run bar', () => {
  const view = app.renderTaskView(AI, NAR, A, {now: NOW, ui: {}});
  const at = (s) => view.indexOf(s);
  assert.ok(at('Round 0 at a glance') < at('<h2>Failure-match audit</h2>') && at('<h2>Failure-match audit</h2>') < at('Generation tree'), 'above the tree');
  const table = app.renderAuditTable(A, {selected: 'cand-c04-r1b2'});
  assert.ok(table.includes('<h2>Failure-match audit</h2><span class="muted small">5 accepted tasks · audit 4/5 · 1 strong · ' +
    '1 partial · 1 weak · 1 none · 1 not audited · click a row for its details</span>'));
  // One group (the accepted tasks) without eligibility from the collector, one row per accepted task, by round.
  assert.equal(count(table, '<tr class="fa-group">'), 1);
  assert.ok(table.includes('<tr class="fa-group"><th colspan="8" scope="rowgroup">Accepted <span class="muted small">5 tasks · audit 4/5</span></th></tr>'));
  assert.deepEqual([...table.matchAll(/<tr class="fa-row[^"]*" data-node="([^"]+)">/g)].map((m) => m[1]),
                   ['cand-c04', 'cand-c01-r1b4', 'cand-c04-r1b2', 'cand-c03-r1b3-r2b1', 'cand-c01-r1b2-r2b1-r3b1']);
  assert.match(table, /<tr class="fa-row selected" data-node="cand-c04-r1b2">/);
  // task (opens the drawer), round, decision, sc14, kimi (clean), similarity, the verdict, the summary's first line.
  assert.match(table, /<thead><tr><th>task<\/th><th class="num">round<\/th><th>decision<\/th><th class="num">sc14 pass<\/th><th class="num">kimi clean<\/th><th class="num">similarity<\/th><th title="[^"]+">audit verdict<\/th><th>summary<\/th><\/tr><\/thead>/);
  assert.match(table, /<tr class="fa-row" data-node="cand-c04"><td class="fa-task"><button type="button" class="node-link" data-node="cand-c04" data-fkey="audit:cand-c04">c04<\/button><div class="fa-title muted small" title="[^"]+">[^<]+<\/div><\/td><td class="num">0<\/td><td class="fa-decision"><span class="badge tone-accepted-evolve badge-sm" title="accepted \(similarity 6\/10\); every valid round-0 task seeds 5 branches"><span class="dot" aria-hidden="true"><\/span>ACCEPTED\+EVOLVE<\/span><\/td><td class="num">1\/5<\/td><td class="num">4\/5<\/td><td class="num">6\/10<\/td><td><span class="badge tone-good badge-sm badge-fa"[^>]*><span class="dot" aria-hidden="true"><\/span>FA match: strong<\/span><\/td><td class="fa-summary"><span title="sc14 strong: it extends[^"]*">sc14 strong: it extends the narrow exception in 3 of 4 failed rollouts\.<\/span><\/td><\/tr>/);
  for (const [id, verdict] of Object.entries(AUDITED)) {
    assert.ok(table.includes(`data-node="${id}">`) && table.includes(`badge tone-${FA_TONES[verdict]} badge-sm badge-fa`), `${id}: ${verdict}`);
  }
  assert.match(table, /FA match: not audited<\/span><\/td><td class="fa-summary"><span class="muted">not audited yet<\/span><\/td><\/tr>/);
  const long = clone(A);
  nodeOf(long, 'cand-c04').failure_match.summary = `${'word '.repeat(80)}\nsecond line`;
  const cut = app.renderAuditTable(long, {});
  assert.match(cut, /<td class="fa-summary"><span title="(?:word ){80}\nsecond line">(?:word ){47}word…<\/span>/);   // its first line, cut to 240 chars
  // Run bar: audit n/m with the verdicts (index runs[].audit, else counted from the nodes).
  const run = AI.runs.find((r) => r.run === 'official1-narwhals');
  assert.ok(app.renderRunBar(run, A, NOW, {}).includes('<div class="runcell-k">Failure-match audit</div><div class="runcell-v"><b title="accepted tasks with ' +
    'a failure-match audit / accepted tasks">audit 4/5</b><div class="muted small">1 strong · 1 partial · 1 weak · 1 none · 1 not audited</div></div>'));
  assert.match(app.renderRunBar(F.index.runs[0], A, NOW, {}), /audit 4\/5/);
  const noneYet = clone(run);
  noneYet.audit = {accepted: 5, audited: 0, strong: 0, partial: 0, weak: 0, none: 0};
  assert.match(app.renderRunBar(noneYet, F.narwhals, NOW, {}), /audit 0\/5<\/b><div class="muted small">5 not audited<\/div>/);
  assert.deepEqual(app.auditCounts(null, A), {accepted: 5, audited: 4, strong: 1, partial: 1, weak: 1, none: 1});
  // Collectors that predate the audit: no table, no run-bar cell; a run without accepted tasks: no table.
  assert.equal(app.auditCounts(F.index.runs[0], F.narwhals), null);
  assert.doesNotMatch(app.renderTaskView(F.index, NAR, F.narwhals, {now: NOW}), /failure-match audit|FA match/i);
  assert.equal(app.renderAuditTable(F.narwhals, {}), '');
  const noAccepted = clone(A);
  for (const n of noAccepted.nodes) if (n.decision === 'accepted' || n.decision === 'accepted_evolve') n.decision = 'end';
  assert.equal(app.renderAuditTable(noAccepted, {}), '');
  assert.equal(app.renderAuditTable(null, {}), '');
  // The full set of views renders cleanly.
  for (const [name, html] of Object.entries(renderAudits(A))) {
    assert.ok(!html.includes('render-error'), `${name} fell back to a render error`);
    assert.doesNotMatch(html, /undefined|NaN|\[object Object\]/, `${name} leaks a JS artefact`);
  }
});

test('failure-match audit (v1.4): every eligible card ends with its "FA match" badge; discarded and withdrawn ones have none', () => {
  assert.equal(ELIGIBLE.length, 36);
  assert.equal(count(app.renderTree(E, {showAbandoned: true}), 'badge-fa'), 36);
  for (const n of E.nodes) assert.equal(cardOf(E, n.id).includes('badge-fa'), n.audit_eligible === true, n.id);
  // Tasks that were not accepted: audited (an evolve task of round 1) and not (an evolve task of round 0).
  assert.match(cardOf(E, 'cand-c04-r1b1'), /<div class="node-meta node-l5"><span class="node-l5-text">kimi [^<]*clean[^<]*<\/span><span class="badge tone-good badge-xs badge-fa" title="[^"]*"><span class="dot" aria-hidden="true"><\/span>FA match: strong<\/span><\/div>/);
  assert.ok(cardOf(E, 'cand-c03').includes('title="no failure-match audit of this task yet"><span class="dot" aria-hidden="true"></span>FA match: not audited</span></div>'));
  // Withdrawn (audited before main withdrew it) and discarded (audited too): no badge; the tooltip names the withdrawal.
  assert.doesNotMatch(cardOf(E, 'cand-c03-r1b0'), /badge-fa|FA match/);
  assert.match(cardOf(E, 'cand-c03-r1b0'), /title="[^"]*\nwithdrawn by main · [^"]*· grading artifact: sc14-r2 only failed/);
  assert.doesNotMatch(cardOf(E, 'cand-c02'), /badge-fa|FA match/);
  // The collector's scope wins over the decision: an accepted task it leaves out (an abandoned-* root) has no badge.
  const d = clone(E);
  nodeOf(d, 'cand-c04').audit_eligible = false;
  assert.doesNotMatch(cardOf(d, 'cand-c04'), /badge-fa/);
  assert.equal(app.inAuditScope({decision: 'accepted'}), true);                   // v1.3 collectors: the accepted ones
  assert.equal(app.inAuditScope({decision: 'end'}), false);
  assert.equal(app.inAuditScope({decision: 'accepted', audit_eligible: false}), false);
  assert.equal(app.inAuditScope({decision: 'end', audit_eligible: true}), true);
  assert.equal(app.inAuditScope({decision: 'end', audit_eligible: 'yes'}), false);
  assert.equal(app.inAuditScope(null), false);
});

test('failure-match audit (v1.4): the table lists every eligible task, the accepted first, then the others round by round', () => {
  const view = app.renderTaskView(EI, NAR, E, {now: NOW, ui: {}});
  const at = (s) => view.indexOf(s);
  assert.ok(at('Round 0 at a glance') < at('<h2>Failure-match audit</h2>') && at('<h2>Failure-match audit</h2>') < at('Generation tree'), 'above the tree');
  const table = app.renderAuditTable(E, {selected: 'cand-c04-r1b1'});
  assert.ok(table.includes('<h2>Failure-match audit</h2><span class="muted small">36 eligible tasks · audit 7/36 · 2 strong · 1 partial · ' +
    '2 weak · 2 none · 29 not audited · click a row for its details</span>'));
  const groups = [...table.matchAll(/<tr class="fa-group"><th colspan="8" scope="rowgroup">([^<]+) <span class="muted small">([^<]+)<\/span><\/th><\/tr>/g)]
    .map((m) => `${m[1]}: ${m[2]}`);
  assert.deepEqual(groups, ['Accepted: 5 tasks · audit 4/5', 'Round 0 · not accepted: 2 tasks · audit 1/2',
                            'Round 1 · not accepted: 10 tasks · audit 1/10', 'Round 2 · not accepted: 12 tasks · audit 0/12',
                            'Round 3 · not accepted: 7 tasks · audit 1/7']);
  const rows = [...table.matchAll(/<tr class="fa-row[^"]*" data-node="([^"]+)">/g)].map((m) => m[1]);
  assert.equal(rows.length, 36);
  assert.deepEqual([...rows].sort(), [...ELIGIBLE].sort());
  assert.deepEqual(rows.slice(0, 8), ['cand-c04', 'cand-c01-r1b4', 'cand-c04-r1b2', 'cand-c03-r1b3-r2b1', 'cand-c01-r1b2-r2b1-r3b1',
                                      'cand-c01', 'cand-c03', 'cand-c01-r1b0']);
  for (const id of ['cand-c03-r1b0', 'cand-c02', 'cand-c05', 'cand-c02-r1b0', 'cand-c01-r1b3', 'cand-c04-r1b1-r2b2-r3b0']) {
    assert.ok(!table.includes(`data-node="${id}"`), `${id} is not eligible`);
  }
  assert.equal(count(table, 'data-fkey="audit:'), 36);
  // The decision column (with the reason as its tooltip), next to each task's audit.
  assert.match(table, /<tr class="fa-row selected" data-node="cand-c04-r1b1"><td class="fa-task">[^]*?<\/td><td class="num">1<\/td><td class="fa-decision"><span class="badge tone-evolve badge-sm" title="best valid non-accepted task of lineage cand-c04 \(similarity 5\/10\); 5 branches in round 2"><span class="dot" aria-hidden="true"><\/span>EVOLVE<\/span><\/td>[^]*?FA match: strong[^]*?<\/tr>/);
  assert.match(table, /data-node="cand-c03-r1b3-r2b0-r3b0">(?:(?!<\/tr>)[^])*>END<\/span><\/td>(?:(?!<\/tr>)[^])*FA match: none/);
  assert.match(table, /data-node="cand-c01-r1b0">(?:(?!<\/tr>)[^])*>NOT SELECTED<\/span><\/td>(?:(?!<\/tr>)[^])*FA match: not audited<\/span><\/td><td class="fa-summary"><span class="muted">not audited yet<\/span>/);
  // No eligible task: no table (the accepted ones do not stand in once the collector marks the scope).
  const none = clone(E);
  for (const n of none.nodes) n.audit_eligible = false;
  assert.equal(app.renderAuditTable(none, {}), '');
  assert.equal(app.renderAuditTable({nodes: [{id: 'cand-x', round: 'x', decision: 'end', audit_eligible: true, failure_match: null}]}, {})
    .includes('Round ? · not accepted <span class="muted small">1 task · audit 0/1</span>'), true);
  for (const [name, html] of Object.entries(renderAudits(E, EI))) {
    assert.ok(!html.includes('render-error'), `${name} fell back to a render error`);
    assert.doesNotMatch(html, /undefined|NaN|\[object Object\]/, `${name} leaks a JS artefact`);
  }
});

test('failure-match audit (v1.4): the run bar reads "audit n/m" over the eligible tasks, then over the accepted ones', () => {
  const run = EI.runs.find((r) => r.run === 'official1-narwhals');
  const cell = '<div class="runcell-k">Failure-match audit</div><div class="runcell-v"><b title="eligible tasks with a failure-match audit / ' +
    'eligible tasks (built and probed, a graded sc14 rollout, not discarded or withdrawn)">audit 7/36</b>' +
    '<div class="muted small">2 strong · 1 partial · 2 weak · 2 none · 29 not audited</div>' +
    '<div class="muted small">accepted: audit 4/5 · 1 strong · 1 partial · 1 weak · 1 none · 1 not audited</div></div>';
  assert.ok(app.renderRunBar(run, E, NOW, {}).includes(cell));
  assert.ok(app.renderRunBar(F.index.runs[0], E, NOW, {}).includes(cell), 'counted from the nodes without runs[].audit');
  assert.deepEqual(app.eligibleCounts(run, null), {eligible: 36, audited: 7, strong: 2, partial: 1, weak: 2, none: 2});
  assert.deepEqual(app.eligibleCounts(null, E), {eligible: 36, audited: 7, strong: 2, partial: 1, weak: 2, none: 2});
  assert.equal(app.eligibleCounts(AI.runs[0], A), null);                       // a v1.3 collector
  // The index's counts (the cycle the page shows) win over the nodes'.
  const later = clone(run);
  Object.assign(later.audit, {eligible: 40, audited_eligible: 9, eligible_verdicts: {strong: 9}});
  assert.match(app.renderRunBar(later, E, NOW, {}), /audit 9\/40<\/b><div class="muted small">9 strong · 31 not audited<\/div>/);
  for (const v of [null, {}, {audit: null}, {audit: {eligible: 'x'}}, {audit: {eligible: -3, audited_eligible: 2, eligible_verdicts: 'x'}},
                   {audit: {eligible: 3, audited_eligible: 1, eligible_verdicts: {strong: 'a', none: -1, constructor: 4}}}]) {
    assert.doesNotThrow(() => app.renderRunBar(v, E, NOW, {}));
    assert.doesNotThrow(() => app.auditBreakdown(app.eligibleCounts(v, null)));
  }
});

test('failure-match audit (v1.4): drawers of eligible, withdrawn and discarded tasks', () => {
  // An eligible task that was not accepted: its audit, nothing about the scope.
  const evolve = app.renderNodeDetail(nodeOf(E, 'cand-c04-r1b1'), E, {}, NOW);
  assert.match(evolve, /Failure-pattern match \(audit\)<\/h3><div class="callout tone-good"><div class="callout-k">Verdict/);
  assert.doesNotMatch(evolve, /Not counted|callout-k">Withdrawn/);
  // An eligible task without an audit says so.
  assert.ok(app.renderNodeDetail(nodeOf(E, 'cand-c03'), E, {}, NOW).includes('Failure-pattern match (audit)</h3><p class="note">Not audited yet: ' +
    'no failure-match audit of this eligible task (runs/&lt;run&gt;/audit/failure_match/&lt;task_id&gt;.json).</p>'));
  // Withdrawn by main: a callout under the decision; its earlier audit is shown, not counted.
  const w = app.renderNodeDetail(nodeOf(E, 'cand-c03-r1b0'), E, {}, NOW);
  const when = app.fmtDate(Date.parse(WITHDRAWN.at) / 1000);
  const facts = `by main · ${when} · grading artifact: sc14-r2 only failed a P2P test that test.patch never resets`;
  assert.ok(w.includes(`<div class="callout tone-bad"><div class="callout-k">Withdrawn</div><p>${facts}</p></div>`));
  assert.ok(w.indexOf('callout-k">Decision') < w.indexOf('callout-k">Withdrawn'));
  assert.ok(w.includes(`Failure-pattern match (audit)</h3><p class="note note-wait">Not counted: this task is outside the audit&#39;s scope, ` +
    `withdrawn (${facts}).</p><div class="callout tone-pending">`));
  // Discarded (invalid in seeds.json), with an audit: shown, not counted.
  assert.match(app.renderNodeDetail(nodeOf(E, 'cand-c02'), E, {}, NOW),
               /<p class="note note-wait">Not counted: this task is outside the audit&#39;s scope, discarded: verifier produced no grade for 6 rollout\(s\)[^<]*<\/p><div class="callout tone-serious">/);
  // Outside the scope without an audit: no section.
  for (const id of ['cand-c05', 'cand-c02-r1b0', 'cand-c01-r1b3', 'cand-c04-r1b1-r2b2-r3b0']) {
    assert.doesNotMatch(app.renderNodeDetail(nodeOf(E, id), E, {}, NOW), /Failure-pattern match/, id);
  }
  // A withdrawal known only from the moved task dir.
  const d = clone(E);
  nodeOf(d, 'cand-c03-r1b0').withdrawn = {at: null, by: null, why: null};
  assert.ok(app.renderNodeDetail(nodeOf(d, 'cand-c03-r1b0'), d, {}, NOW)
    .includes('<div class="callout-k">Withdrawn</div><p>its task dir was moved to runs/&lt;run&gt;/withdrawn/</p></div>'));
  assert.deepEqual(app.withdrawnParts({at: 'soon', by: '', why: null}), ['soon']);
  assert.deepEqual(app.withdrawnParts([]), []);
  assert.equal(app.outOfScopeWhy({score: null, decision: 'pending'}), 'not scored yet');
  assert.equal(app.outOfScopeWhy({score: {}, decision: 'end'}), 'no graded sc14 rollout, or an abandoned build');
  assert.equal(app.outOfScopeWhy({decision: 'discard'}), 'discarded');
});

/** v1.5 (official1-wasmi cand-c03-r1b2, 2026-10-21): a task file as a v1.5 collector writes it, NODE.seed_invalid on
    every node (each scored task's seed verdict: the seeds.json reason, null for a valid seed or an unscored task), with
    two accepted tasks seeds.json lists invalid: round-1 cand-c01-r1b4 (one kimi rollout never graded) and round-0
    cand-c02, accepted (an invalid seed, so not evolved: its children stay abandoned). */
const KIMI_R3 = 'verifier produced no grade for 1 rollout(s): kimi-r3';
function withSeedNotes(doc) {
  const d = clone(doc);
  const seedsOf = (k) => (d.rounds.find((r) => r.index === k) || {}).seeds || null;
  const r1 = seedsOf(1);
  r1.valid = r1.valid.filter((id) => id !== 'cand-c01-r1b4');
  r1.invalid['cand-c01-r1b4'] = KIMI_R3;
  for (const n of d.nodes) {
    const seeds = seedsOf(n.round);
    n.seed_invalid = n.score && seeds && Object.prototype.hasOwnProperty.call(seeds.invalid, n.id) ? seeds.invalid[n.id] : null;
  }
  const c02 = nodeOf(d, 'cand-c02');
  Object.assign(c02, {decision: 'accepted', decision_reason: 'accepted (similarity 6/10); an invalid seed, so not evolved',
                      audit_eligible: true});
  Object.assign(c02.score, {accept: true, similarity: 6, reason: 'similarity 6/10'});
  return d;
}
const V = withSeedNotes(E);
const rowOf = (html, id) => (html.match(new RegExp(`<tr class="[^"]*" data-node="${id}">(?:(?!</tr>)[^])*</tr>`)) || [''])[0];

test('seed note (v1.5): an accepted invalid seed reads ACCEPTED plus a "seed-invalid" badge, and a "Seed-invalid" callout', () => {
  const help = app.SEED_NOTE.help, dot = '<span class="dot" aria-hidden="true"></span>';
  const seedBadge = (why) => `<span class="badge tone-pending badge-xs badge-seed" title="seed-invalid: ${why}\n${help}">${dot}seed-invalid</span>`;
  assert.equal(nodeOf(V, 'cand-c01-r1b4').seed_invalid, KIMI_R3);
  // Card: the decision badge stays at the right edge, the note's badge before it; the tooltip has the note.
  const card = cardOf(V, 'cand-c01-r1b4');
  assert.ok(card.includes(`<div class="node-top"><span class="node-id">r1b4</span><span class="node-badges">${seedBadge(KIMI_R3)}` +
                          `<span class="badge tone-good badge-sm">${dot}ACCEPTED</span></span></div>`), card);
  assert.ok(card.includes(`\nACCEPTED: accepted (similarity 7/10)\nseed-invalid: ${KIMI_R3}\n`), 'card tooltip');
  assert.ok(card.includes('badge-fa'), 'still eligible: its FA match badge');
  // Drawer: a callout under the decision, with where the note comes from.
  const drawer = app.renderNodeDetail(nodeOf(V, 'cand-c01-r1b4'), V, {}, NOW);
  assert.ok(drawer.includes('<div class="callout-k">Decision</div><p><b>ACCEPTED</b> · accepted (similarity 7/10)</p></div>' +
                            `<div class="callout tone-pending"><div class="callout-k">Seed-invalid</div><p>${KIMI_R3}</p>` +
                            `<p class="muted small">round-1/seeds.json · ${help}</p></div>`), drawer);
  // Round 0: ACCEPTED (not ACCEPTED+EVOLVE) with the note, in the card and the round-0 table; its children stay abandoned.
  const why = V.rounds[0].seeds.invalid['cand-c02'];
  assert.ok(cardOf(V, 'cand-c02').includes(`<span class="node-badges">${seedBadge(why)}<span class="badge tone-good badge-sm">${dot}ACCEPTED</span></span>`));
  const r0 = rowOf(app.renderRoundZero(V, {}), 'cand-c02');
  assert.ok(r0.includes(`${dot}ACCEPTED</span>${seedBadge(why)}</td><td class="r0-reason">accepted (similarity 6/10); an invalid seed, so not evolved</td>`), r0);
  assert.ok(app.renderNodeDetail(nodeOf(V, 'cand-c02'), V, {}, NOW).includes(`<p class="muted small">round-0/seeds.json · ${help}</p>`));
  // The failure-match table: both in the "Accepted" group, the note after the decision badge.
  const table = app.renderAuditTable(V, {});
  for (const [id, note] of [['cand-c01-r1b4', KIMI_R3], ['cand-c02', why]]) {
    assert.match(rowOf(table, id), new RegExp(`<td class="fa-decision"><span class="badge tone-good badge-sm" title="accepted [^"]*">${dot}ACCEPTED</span>`));
    assert.ok(rowOf(table, id).includes(`ACCEPTED</span>${seedBadge(note)}</td>`), id);
  }
  assert.ok(table.includes('Accepted <span class="muted small">6 tasks · audit 5/6</span>'));   // cand-c02's audit counts now
  // Only these two carry the badge: a discard's note is its decision reason, a valid seed has none.
  assert.equal(count(app.renderTree(V, {showAbandoned: true}), 'badge-seed'), 2);
  const discards = V.nodes.filter((n) => n.decision === 'discard' && n.seed_invalid);
  assert.deepEqual(discards.map((n) => n.id), ['cand-c01-r1b3', 'cand-c03-r1b3-r2b4', 'cand-c03-r1b3-r2b0-r3b2'],
                   'discarded invalid seeds carry their reason as the note too');
  for (const n of discards) {
    assert.equal(n.seed_invalid, n.decision_reason);
    assert.doesNotMatch(cardOf(V, n.id), /badge-seed|seed-invalid/, n.id);
    assert.doesNotMatch(app.renderNodeDetail(n, V, {}, NOW), /callout-k">Seed-invalid/, n.id);
  }
  // A note from the recomputed seed_check() (no seeds.json verdict yet), and a collector before v1.5: no badge at all.
  const d = clone(V);
  d.rounds.find((r) => r.index === 1).seeds = null;
  assert.ok(app.renderNodeDetail(nodeOf(d, 'cand-c01-r1b4'), d, {}, NOW)
    .includes(`<p class="muted small">seed_check() recomputed (no seeds.json verdict yet) · ${help}</p>`));
  assert.equal(count(app.renderTaskView(EI, NAR, E, {now: NOW, ui: {showAbandoned: true}}), 'seed-invalid'), 0);
  // Odd values never throw nor count; inherited names are not seeds.json entries.
  assert.equal(app.seedNote({decision: 'accepted', seed_invalid: '  x  '}), 'x');
  assert.equal(app.seedNote({decision: 'discard', seed_invalid: 'x'}), '');
  for (const v of [null, undefined, '', '   ', 3, ['x'], {x: 1}]) assert.equal(app.seedNote({decision: 'accepted', seed_invalid: v}), '');
  assert.equal(app.seedNote(null), '');
  assert.equal(app.seedBadge({decision: 'end'}), '');
  assert.equal(app.seedSource({id: 'constructor', round: 0}, {rounds: [{index: 0, seeds: {invalid: {}}}]}),
               'seed_check() recomputed (no seeds.json verdict yet)');
  assert.equal(app.seedSource(null, null), 'seed_check() recomputed (no seeds.json verdict yet)');
  // Escaping: markup in the note never reaches the markup.
  const x = clone(V);
  nodeOf(x, 'cand-c01-r1b4').seed_invalid = '"><img src=x onerror=S1>';
  for (const html of [cardOf(x, 'cand-c01-r1b4'), app.renderNodeDetail(nodeOf(x, 'cand-c01-r1b4'), x, {}, NOW), app.renderAuditTable(x, {}),
                      app.renderTaskView(EI, NAR, x, {now: NOW, ui: {showAbandoned: true}})]) {
    assert.ok(!html.includes('<img') && html.includes('&quot;&gt;&lt;img src=x onerror=S1&gt;'));
    assert.ok(!html.includes('render-error'));
  }
});

test('escaping: markup in every string of the audits never reaches the markup', () => {
  const inject = (v) => (typeof v === 'string' ? PAYLOAD + v : Array.isArray(v) ? v.map(inject)
    : v && typeof v === 'object' ? Object.fromEntries(Object.entries(v).map(([k, x]) => [k, inject(x)])) : v);
  for (const [doc, index] of [[A, AI], [E, EI]]) {
    const d = clone(doc);
    for (const n of d.nodes) {
      if (n.decision === 'accepted' || n.decision === 'accepted_evolve' || n.audit_eligible) {
        n.title = PAYLOAD + n.title;
        n.decision_reason = PAYLOAD + n.decision_reason;
      }
      if (n.withdrawn) n.withdrawn = inject(n.withdrawn);
      if (!n.failure_match) continue;
      const verdict = n.failure_match.verdict;
      n.failure_match = inject(n.failure_match);
      if (n.id !== 'cand-c04') n.failure_match.verdict = verdict;  // known verdicts too, not only "unknown"
    }
    const out = renderAudits(d, index);
    for (const [name, html] of Object.entries(out)) assertEscaped(html, name);
    const shown = '&quot;&#39;&gt;&lt;script&gt;X1&lt;/script&gt;&lt;img src=x onerror=X2&gt;';
    assert.ok(out.drawers.includes(`<p>${shown}sc14 weak: it extends`), 'summary shown as text');
    assert.ok(out.drawers.includes(`<span class="slot">${shown}sc14-r1</span>`), 'slots');
    assert.ok(out.drawers.includes(`<li><code>${shown}ibis/backends/`), 'failing tests');
    assert.ok(out.table.includes(`<span title="${shown}sc14 partial:`), 'summary tooltip');
    assert.ok(out.table.includes(`badge-sm" title="${shown}`), 'decision reason tooltip');
    assert.ok(out.cards.includes(`FA match: unknown</span></div>`), 'an injected verdict is unknown, not markup');
    assert.ok(out.view.includes('<h2>Failure-match audit</h2>'));
    if (doc === E) {
      assert.ok(out.drawers.includes(`<p>by ${shown}main · ${shown}2026-10-18T23:03:14-0700 · ${shown}grading artifact`), 'the withdrawal');
    }
    // Only tags app.js writes itself.
    const tags = new Set(['div', 'span', 'section', 'article', 'h1', 'h2', 'h3', 'h4', 'p', 'b', 'code', 'pre', 'details', 'summary', 'table',
      'thead', 'tbody', 'tr', 'th', 'td', 'ul', 'li', 'dl', 'dt', 'dd', 'svg', 'path', 'line', 'text', 'circle', 'rect', 'g', 'title', 'button']);
    for (const [name, html] of Object.entries(out)) {
      for (const m of html.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)/g)) assert.ok(tags.has(m[1].toLowerCase()), `${name}: <${m[1]}>`);
    }
  }
});

test('failure-match audit: sparse and broken audits render without exceptions or fallbacks', () => {
  const variants = [['every leaf null', nullAll(clone(A))], ['null document', null], ['no nodes', {nodes: null}],
                    ['v1.4, every leaf null', nullAll(clone(E))]];
  for (const w of WEIRD) {
    const d = clone(A);
    for (const n of d.nodes) if (n.failure_match !== null) n.failure_match = clone(w);
    variants.push([`failure_match ${JSON.stringify(w)}`, d]);
    const field = clone(A);
    for (const n of field.nodes) if (n.failure_match) for (const k of Object.keys(n.failure_match)) n.failure_match[k] = clone(w);
    variants.push([`every audit field ${JSON.stringify(w)}`, field]);
    const scope = clone(E);
    for (const n of scope.nodes) { n.audit_eligible = clone(w); n.withdrawn = clone(w); }
    variants.push([`audit_eligible and withdrawn ${JSON.stringify(w)}`, scope]);
  }
  for (let seed = 1; seed <= 50; seed += 1) {
    const rnd = prng(1000 + seed);
    const d = clone(A);
    for (const n of d.nodes) if (n.failure_match) n.failure_match = mutate(n.failure_match, rnd, 1);
    variants.push([`audit mutation ${seed}`, d]);
    variants.push([`document mutation ${seed}`, mutate(clone(A), prng(2000 + seed), 0)]);
    if (seed <= 20) variants.push([`v1.4 document mutation ${seed}`, mutate(clone(E), prng(3000 + seed), 0)]);
  }
  for (const [label, d] of variants) {
    for (const ix of [AI, EI, null]) {
      let out;
      assert.doesNotThrow(() => { out = renderAudits(d, ix, true); }, label);
      for (const [name, html] of Object.entries(out)) {
        assert.equal(typeof html, 'string', `${label} / ${name}`);
        assert.ok(!html.includes('render-error'), `${label} / ${name} needed the safe() fallback`);
      }
    }
  }
  for (const v of [null, {}, {audit: null}, {audit: {accepted: 'x'}}, {audit: {accepted: 3, audited: -1, strong: 'a'}}]) {
    assert.doesNotThrow(() => app.renderRunBar(v, A, NOW, {}));
    assert.doesNotThrow(() => app.auditBreakdown(app.auditCounts(v, null)));
  }
});

// -------------------------------------------------------------------------------------------- monitor

test('monitor: summary tiles, worker table, 24 h charts, API panel', () => {
  const html = app.renderMonitor(F.fleet, F.api, {now: NOW});
  for (const label of ['Routable workers', 'Coverage', 'Busy workers', 'Running requests', 'KV cache (mean)', 'Generation',
                       'Router requests (10m)', 'Latency p50 / p90', 'Fleet daemon']) assert.ok(html.includes(label), label);
  assert.match(html, /9<span class="tile-den">\/10<\/span>/);
  assert.match(html, /ready 8 · draining 1 · starting 1 · pending 2/);
  // F7: utilization as busy / routable workers; gen tok/s is the throughput; prompt tok/s says it counts cache hits.
  assert.match(html, /Busy workers<\/div><div class="tile-value">8<span class="tile-den">\/9<\/span><\/div><div class="tile-sub">utilization 89% · routable workers with ≥ 1 running request/);
  assert.match(html, /Generation<\/div><div class="tile-value">711<span class="tile-den"> tok\/s<\/span><\/div><div class="tile-sub">prompt 14\.1K tok\/s \(incl\. prefix-cache hits\)/);
  assert.match(html, /official-run probes · 3 errors \(0\.4%\) · 3 clamps/);
  assert.match(html, /official-run probes · last hour 4,890 requests, 11 errors/);
  assert.match(html, /router columns: official-run probes/);
  assert.match(html, /9\.1<span class="tile-den"> \/ 24\.0 s/);
  assert.match(html, /breaker closed · job 20371655 on cpu1-00096/);
  for (const st of ['ready', 'draining', 'starting', 'pending']) assert.ok(html.includes(`</span>${st}</span>`), `worker status ${st}`);
  assert.match(html, /est\. start/);
  assert.match(html, /Priority/);
  assert.match(html, /scrape error/);
  assert.equal(count(html, '<svg class="chart-svg"'), 5);         // 4 fleet charts + spend
  assert.ok(count(html, '<title>') > 100, 'hover tooltips');
  assert.ok(count(html, 'class="tick"') > 20, 'axis ticks');
  assert.match(html, /target 10/);
  assert.match(html, /Router requests per cycle \(10m window, official-run probes\)/);
  assert.match(html, /<div class="chart-legend">.*running.*waiting/);
  // API
  assert.match(html, /\$468\.75/);
  assert.match(html, /\$9,531\.25/);
  assert.match(html, /\$10,000\.00/);
  assert.match(html, /Burn \(1 h\)<\/div><div class="tile-value">\$75\.10<span class="tile-den">\/h<\/span><\/div><div class="tile-sub">over the last 58m of hub samples/);
  assert.match(html, /Burn \(6 h\)<\/div><div class="tile-value">\$70\.00<span class="tile-den">\/h<\/span><\/div><div class="tile-sub">over the last 5h 52m of hub samples/);
  assert.match(html, /in 5d 16h at the [16] h burn/);
  assert.match(html, /budget \$10K ↑ off scale/);
  assert.match(html, /data-action="spend-scale" aria-pressed="false">Show budget line/);
  const withBudget = app.renderApi(F.api, NOW, {spendScale: 'budget'});
  assert.match(withBudget, /<line class="ref"/);
  assert.match(withBudget, /Fit to spend/);
  assert.match(withBudget, /\$10K<\/text>/);                        // the y axis now reaches the budget
  assert.doesNotMatch(html, /meter-bad/);
  assert.match(html, /official1-bandit.*\$110\.15/s);
  assert.match(html, /row-total/);
  assert.match(html, /kimi cost is hub-reported per call/);
  // the 25 min collector outage breaks the lines
  const routable = html.match(/<path class="line s1" d="[^"]*"/g);
  assert.ok(routable.length >= 2, 'line broken at the data gap');
});

test('monitor on an older fleet.json: busy workers counted from the workers, prompt_tps, default router scope', () => {
  const fleet = clone(F.fleet);
  delete fleet.totals.busy_workers;
  delete fleet.router.scope;
  fleet.totals.prompt_tps = fleet.totals.prompt_tps_incl_cache;
  delete fleet.totals.prompt_tps_incl_cache;
  assert.deepEqual(app.busyWorkers(fleet), {busy: 8, routable: 9});   // slot 07 is idle, 09 drains with 2 requests
  const html = app.renderFleetSummary(fleet, NOW);
  assert.match(html, /Busy workers<\/div><div class="tile-value">8<span class="tile-den">\/9<\/span>/);
  assert.match(html, /prompt 14\.1K tok\/s \(incl\. prefix-cache hits\)/);
  assert.match(html, /official-run probes · 3 errors/);
  // routable from the workers when totals lack it; no scraped routable worker: unknown, not 0.
  delete fleet.totals.routable;
  assert.deepEqual(app.busyWorkers(fleet), {busy: 8, routable: 9});
  for (const w of fleet.workers) w.metrics = null;
  assert.deepEqual(app.busyWorkers(fleet), {busy: null, routable: 9});
  assert.match(app.renderFleetSummary(fleet, NOW), /Busy workers<\/div><div class="tile-value">–<span class="tile-den">\/9/);
  assert.deepEqual(app.busyWorkers(null), {busy: null, routable: 0});
  // A routing field decides who is routable, as in the daemon: a stale ready record does not count.
  assert.deepEqual(app.busyWorkers({workers: [{registry_status: 'ready', routing: null, metrics: {running: 3}},
                                              {registry_status: 'ready', routing: 'ready', metrics: {running: 1}}]}), {busy: 1, routable: 1});
  assert.equal(app.routerScope({router: {scope: 'all requests'}}), 'all requests');
});

test('API panel turns red below $500 remaining and handles no projection', () => {
  const api = clone(F.api);
  Object.assign(api.hub, {spend: 9600, remaining: 400});
  api.burn.eta_exhaust_iso = null;
  api.burn.eta_exhaust = null;
  const html = app.renderApi(api, NOW, {});
  assert.match(html, /meter meter-bad/);
  assert.match(html, /below \$500/);
  assert.match(html, /not projected/);
  assert.match(html, /tile tone-bad/);
  api.hub.error = 'hub_budget.py: HTTP 401';
  assert.match(app.renderApi(api, NOW, {}), /hub sample error: hub_budget\.py: HTTP 401/);
});

test('API panel: scan health, hub sample age, burn windows; older api.json without them', () => {
  const html = app.renderApi(F.api, NOW, {});
  assert.match(html, /Spend<\/div><div class="tile-value">\$468\.75<\/div><div class="tile-sub">4\.7% of the budget · sampled 4m ago<\/div>/);
  assert.ok(html.includes(`hub sample 4m old (${app.fmtClock(NOW - 270, true)}) · HTTP 200 · sampled at most every 10 min`));
  assert.match(html, /<h3 class="sub-h">Usage by run <span class="badge tone-pending"[^>]*><span class="dot" aria-hidden="true"><\/span>usage scan catching up: 12 MB to read<\/span><\/h3>/);
  const done = clone(F.api);
  done.scan = {complete: true, pending_bytes: 0, files: 443};
  assert.match(app.renderApi(done, NOW, {}), /Usage by run <span class="muted small">usage scan complete · 443 logs<\/span>/);
  // The ETA comes from burn.eta_exhaust (projected from the hub sample time), else the ISO string.
  const iso = clone(F.api);
  iso.burn.eta_exhaust = iso.burn.eta_exhaust + 86400;
  assert.match(app.renderApi(iso, NOW, {}), /in 6d 16h at the [16] h burn/);
  // A sample older than 30 min stands out.
  const old = clone(F.api);
  old.hub.sampled_at = NOW - 3600;
  assert.match(app.renderApi(old, NOW, {}), /<div class="tile tone-pending"><div class="tile-label">Spend[^]*sampled 1h ago/);
  assert.match(app.renderApi(old, NOW, {}), /<div class="muted small txt-bad">hub sample 1h old/);
  // Older api.json: no windows, no scan, no epoch ETA, only age_s.
  const older = clone(F.api);
  delete older.scan;
  delete older.burn.window_min_1h;
  delete older.burn.window_min_6h;
  delete older.burn.eta_exhaust;
  delete older.hub.sampled_at;
  const h = app.renderApi(older, NOW, {});
  assert.match(h, /Burn \(1 h\)<\/div><div class="tile-value">\$75\.10<span class="tile-den">\/h<\/span><\/div><\/div>/);
  assert.match(h, /in 5d 16h at the [16] h burn/);
  assert.match(h, /sampled 4m ago/);                                // age_s 240 at the cycle, 30 s later now
  assert.match(h, /<h3 class="sub-h">Usage by run <\/h3>/);
  delete older.burn.per_hour_6h;
  assert.match(app.renderApi(older, NOW, {}), /Burn \(6 h\)<\/div><div class="tile-value">–<\/div><div class="tile-sub">not enough hub samples in the last 6 h/);
});

test('line chart edge cases', () => {
  const base = {title: 't', t0: NOW - 86400, t1: NOW};
  const empty = app.renderLineChart(Object.assign({}, base, {series: [{label: 'x', points: []}], empty: 'nothing here'}));
  assert.match(empty, /nothing here/);
  const one = app.renderLineChart(Object.assign({}, base, {series: [{label: 'x', points: [[NOW - 10, 3]]}]}));
  assert.match(one, /dot-solo/);
  const zeros = app.renderLineChart(Object.assign({}, base, {series: [{label: 'x', points: [[NOW - 100, 0], [NOW - 50, 0]]}]}));
  assert.doesNotMatch(zeros, /NaN|Infinity/);
  const weird = app.renderLineChart(Object.assign({}, base, {series: [{label: 'x', points: [[NOW - 100, -5], [NaN, 1], ['a', 2], null,
    {t: NOW - 20, v: 1e12}, [NOW + 999, 4], [NOW - 30]]}]}));
  assert.doesNotMatch(weird, /NaN|Infinity/);
  assert.doesNotMatch(app.renderLineChart({}), /NaN|Infinity/);
  assert.doesNotMatch(app.renderLineChart(null), /NaN|Infinity/);
});

// ------------------------------------------------------------------------------------- escaping

const PAYLOAD = `"'><script>X1</script><img src=x onerror=X2>`;

function assertEscaped(html, label) {
  assert.ok(!html.includes('<script>X1'), `${label}: raw <script> injected`);
  assert.ok(!html.includes('<img src=x'), `${label}: raw <img> injected`);
}

test('escaping: <script> injected into titles and names never reaches the markup', () => {
  const d = clone(F);
  for (const n of d.narwhals.nodes) {
    n.title = PAYLOAD + n.title; n.decision_reason = PAYLOAD; n.summary = PAYLOAD;
    if (n.focus_pattern) n.focus_pattern = PAYLOAD;
    if (Array.isArray(n.details.teacher_audit)) for (const a of n.details.teacher_audit) { a.summary = PAYLOAD; a.categories = [PAYLOAD]; a.slot = PAYLOAD; }
    if (n.probe) for (const r of n.probe.rollouts) r.agent_error = `${PAYLOAD}\n${PAYLOAD}`;
  }
  for (const p of d.narwhals.failure_analysis.patterns) { p.name = PAYLOAD; p.category = PAYLOAD; p.student_slots = [PAYLOAD]; }
  for (const c of d.narwhals.plan.candidates) c.title = PAYLOAD;
  d.narwhals.job.log_tail = [PAYLOAD];
  d.index.models.kimi.label = PAYLOAD;
  d.index.runs[0].stage = PAYLOAD;
  d.index.errors = [PAYLOAD];
  for (const w of d.fleet.workers) { w.node = PAYLOAD; w.pending_reason = PAYLOAD; }
  d.fleet.router.scope = PAYLOAD;
  d.api.notes = [PAYLOAD];
  d.api.by_run[0].run = PAYLOAD;
  const out = renderAll(d, {showAbandoned: true});
  for (const [name, html] of Object.entries(out)) assertEscaped(html, name);
  assert.ok(out.narwhals.includes('&quot;&#39;&gt;&lt;script&gt;X1&lt;/script&gt;&lt;img src=x onerror=X2&gt;'), 'payload shown as text');
  assert.ok(out.drawers.includes('&lt;script&gt;X1'));
  assert.ok(out.monitor.includes('&lt;script&gt;X1'));
  assert.ok(out.banner.includes('&lt;script&gt;X1'));
  assert.ok(out.roundZero.includes('&lt;script&gt;X1'));
  assert.ok(out.cards.includes('tag-focus" title="target failure pattern: &quot;&#39;&gt;&lt;script&gt;X1'));
});

test('escaping: every string field of every fixture carries the payload', () => {
  const inject = (v) => {
    if (typeof v === 'string') return PAYLOAD + v;
    if (Array.isArray(v)) return v.map(inject);
    if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, inject(x)]));
    return v;
  };
  const d = inject(clone(F));
  const out = renderAll(d, {showAbandoned: true});
  out.menuAll = app.renderTaskMenu(d.index, '', PAYLOAD + NAR, 0);
  out.narwhalsInjected = app.renderTaskView(d.index, PAYLOAD + NAR, d.narwhals, {now: NOW, ui: {showAbandoned: true}});
  // Injected run names fail the official* filter, so render the run sections directly as well.
  out.tree = app.renderTree(d.narwhals, {showAbandoned: true, selected: PAYLOAD + 'cand-c04-r1b1'});
  out.treeBandit = app.renderTree(d.bandit, {});
  out.roundZero = app.renderRoundZero(d.narwhals, {selected: PAYLOAD + 'cand-c01'});
  out.runBar = app.renderRunBar(d.index.runs[0], d.narwhals, NOW, {});
  out.failure = app.renderFailureAnalysis(d.narwhals.failure_analysis, {}, d.narwhals.nodes);
  out.plan = app.renderPlan(d.narwhals.plan, {});
  for (const [name, html] of Object.entries(out)) assertEscaped(html, name);
  assert.match(out.narwhalsInjected, /Not converted yet/);
  assert.ok(out.tree.includes('data-node="&quot;&#39;&gt;&lt;script&gt;X1'), 'ids are escaped in attributes');
  assert.ok(out.tree.includes('data-key="lin:&quot;&#39;&gt;&lt;script&gt;X1'), 'lineage keys are escaped');
  assert.ok(out.tree.includes('data-scroll="tree:&quot;&#39;&gt;&lt;script&gt;X1'), 'scroll keys are escaped');
  assert.ok(out.roundZero.includes('data-fkey="r0:&quot;&#39;&gt;&lt;script&gt;X1'), 'focus keys are escaped');
  assert.ok(out.runBar.includes('&lt;script&gt;X1'));
  assert.ok(out.plan.includes('&lt;script&gt;X1'));
  assert.ok(out.failure.includes('pattern-targets'), 'focus patterns still match their (injected) pattern names');
});

// ------------------------------------------------------------------------------- sparse / broken data

function prng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const WEIRD = [null, '', 'NaN', -1, 0, 1e308, -1e308, true, [], {}, '<b>x</b>', [1, 'a', null], {a: {b: 1}}, 'running', ['x', 'y'], 3.7];

/** A deliberately broken copy: fields dropped, replaced by wrong types, containers emptied or nulled. */
function mutate(v, rnd, depth) {
  if (Array.isArray(v)) {
    const r = rnd();
    if (depth > 0 && r < 0.04) return WEIRD[Math.floor(rnd() * WEIRD.length)];
    return v.filter(() => rnd() > 0.1).map((x) => mutate(x, rnd, depth + 1));
  }
  if (v && typeof v === 'object') {
    if (depth > 0 && rnd() < 0.04) return WEIRD[Math.floor(rnd() * WEIRD.length)];
    const out = {};
    for (const [k, x] of Object.entries(v)) {
      const q = rnd();
      if (q < 0.12) continue;
      out[k] = q < 0.2 ? WEIRD[Math.floor(rnd() * WEIRD.length)] : mutate(x, rnd, depth + 1);
    }
    return out;
  }
  return v;
}

function nullAll(v) {
  if (Array.isArray(v)) return v.map(nullAll);
  if (v && typeof v === 'object') return Object.fromEntries(Object.entries(v).map(([k, x]) => [k, x && typeof x === 'object' ? nullAll(x) : null]));
  return null;
}

test('sparse and broken copies render without exceptions or fallbacks', () => {
  const variants = [
    ['empty objects', {index: {}, narwhals: {}, bandit: {}, fd: {}, fleet: {}, api: {}}],
    ['nulls', {index: null, narwhals: null, bandit: null, fd: null, fleet: null, api: null}],
    ['arrays', {index: [], narwhals: [], bandit: [], fd: [], fleet: [], api: []}],
    ['every leaf null', nullAll(clone(F))],
    ['strings', {index: 'x', narwhals: 'x', bandit: 1, fd: true, fleet: 'x', api: 0}],
    ['older shapes', Object.assign(clone(F), {narwhals: olderShapes(F.narwhals)})],
  ];
  for (let seed = 1; seed <= 80; seed += 1) {
    const rnd = prng(seed);
    const d = {};
    for (const [k, v] of Object.entries(F)) d[k] = mutate(clone(v), rnd, 0);
    variants.push([`mutation seed ${seed}`, d]);
  }
  for (const [label, d] of variants) {
    let out;
    assert.doesNotThrow(() => { out = renderAll(d, {showAbandoned: label.length % 2 === 0, selected: 'cand-c01'}); }, label);
    for (const [name, html] of Object.entries(out)) {
      assert.equal(typeof html, 'string', `${label} / ${name}`);
      assert.ok(!html.includes('render-error'), `${label} / ${name} needed the safe() fallback`);
    }
    const ix = d.index && Array.isArray(d.index.tasks) ? d.index : null;
    if (ix) {
      for (const t of ix.tasks) {
        const id = t && typeof t === 'object' ? t.task_id : null;
        assert.doesNotThrow(() => app.renderTaskView(ix, id, d.narwhals, {now: NOW}), `${label} task ${id}`);
      }
    }
  }
});

// ------------------------------------------------------------------------------- refresh helpers (pure)

test('refreshAction: skip an unchanged view, hold an automatic refresh while text is selected, else render', () => {
  assert.equal(app.refreshAction({changed: false}), 'skip');
  assert.equal(app.refreshAction({changed: false, auto: true, selecting: true}), 'skip');
  assert.equal(app.refreshAction({changed: true}), 'render');
  assert.equal(app.refreshAction({changed: true, auto: true}), 'render');
  assert.equal(app.refreshAction({changed: true, auto: true, selecting: true}), 'hold');
  assert.equal(app.refreshAction({changed: true, auto: false, selecting: true}), 'render');   // a click re-renders anyway
  assert.equal(app.refreshAction({changed: false, force: true}), 'render');
  assert.equal(app.refreshAction(null), 'skip');
});

test('hasTextSelection: only a non-empty selection counts', () => {
  assert.equal(app.hasTextSelection({isCollapsed: false, toString: () => 'abc'}), true);
  assert.equal(app.hasTextSelection({isCollapsed: true, toString: () => ''}), false);
  assert.equal(app.hasTextSelection({isCollapsed: false, toString: () => ''}), false);
  assert.equal(app.hasTextSelection({isCollapsed: false, toString: () => { throw new Error('x'); }}), false);
  for (const v of [null, undefined, 'abc', {}, {toString: () => 'abc'}]) assert.equal(app.hasTextSelection(v), false);
});

test('captureScroll / restoreScroll / captureOpen work by stable data keys', () => {
  const el = (scroll, left, top, extra) => Object.assign({dataset: scroll === null ? {} : {scroll}, scrollLeft: left, scrollTop: top}, extra);
  const before = [el('tree:x', 300, 0), el('log:r', 0, 120), el('fa:markdown', 0, 0), el('log:r', 0, 999), el(null, 5, 5),
                  el('nd:c:instruction_md', 0, 40.5)];
  const saved = app.captureScroll(before);
  assert.deepEqual({...saved}, {'tree:x': [300, 0], 'log:r': [0, 120], 'nd:c:instruction_md': [0, 40.5]});   // unscrolled ones skipped
  const after = [el('nd:c:instruction_md', 0, 0), el('log:r', 0, 0), el('tree:y', 0, 0), el('tree:x', 0, 0), el('log:r', 0, 0)];
  assert.equal(app.restoreScroll(after, saved), 3);
  assert.deepEqual(after.map((e) => [e.dataset.scroll, e.scrollLeft, e.scrollTop]),
                   [['nd:c:instruction_md', 0, 40.5], ['log:r', 0, 120], ['tree:y', 0, 0], ['tree:x', 300, 0], ['log:r', 0, 0]]);
  const proto = app.captureScroll([el('__proto__', 1, 1)]);
  assert.deepEqual(proto.__proto__, [1, 1]);                        // a key, not the prototype
  assert.equal(app.restoreScroll(after, null), 0);
  assert.equal(app.restoreScroll(null, saved), 0);
  assert.deepEqual({...app.captureScroll([null, 3, {dataset: {scroll: 'k'}, scrollTop: NaN}])}, {});
  const open = app.captureOpen([{dataset: {key: 'a'}, open: true}, {dataset: {key: 'b'}, open: false}, {dataset: {}}, {dataset: {key: 'a'}, open: false}]);
  assert.deepEqual({...open}, {a: true, b: false});
});

test('renderers give every scrollable element a stable data-scroll key', () => {
  const view = app.renderTaskView(F.index, NAR, F.narwhals, {now: NOW, ui: {}});
  for (const key of ['tree:narwhals-rolling-window-suite', 'r0:narwhals-rolling-window-suite', 'log:official1-narwhals', 'fa:markdown',
                     'plan:excerpt', 'fa:evidence:2']) assert.ok(view.includes(`data-scroll="${key}"`), key);
  const drawer = app.renderNodeDetail(nodeOf(F.narwhals, 'cand-c04'), F.narwhals, {}, NOW);
  for (const key of ['nd:cand-c04:instruction_md', 'nd:cand-c04:f2p', 'nd:cand-c04:rollouts:table', 'nd:cand-c04:teacher_audit:table',
                     'nd:cand-c04:similarity:raw']) assert.ok(drawer.includes(`data-scroll="${key}"`), key);
  const monitor = app.renderMonitor(F.fleet, F.api, {now: NOW, ui: {}});
  for (const key of ['workers', 'chart:routable:table', 'chart:spend:table']) assert.ok(monitor.includes(`data-scroll="${key}"`), key);
});

// ------------------------------------------------------------------------------------- DOM wiring

class FakeClassList {
  constructor() { this.set = new Set(); }
  add(c) { this.set.add(c); }
  remove(c) { this.set.delete(c); }
  contains(c) { return this.set.has(c); }
  toggle(c, force) { const on = force === undefined ? !this.set.has(c) : !!force; if (on) this.set.add(c); else this.set.delete(c); return on; }
}

const unesc = (s) => s.replace(/&(amp|quot|#39|lt|gt|#96);/g, (m, k) => ({amp: '&', quot: '"', '#39': "'", lt: '<', gt: '>', '#96': '`'})[k]);
const camel = (s) => s.replace(/-([a-z])/g, (_, c) => c.toUpperCase());

/** The selectors app.js uses: tag, .class and [data-x] / [data-x="v"] parts, combined (no descendant selectors). */
function matchesSel(e, sel) {
  const m = sel.match(/^([a-zA-Z]*)((?:\.[\w-]+)*)((?:\[[\w-]+(?:="(?:[^"\\]|\\.)*")?\])*)$/);
  if (!m) return false;
  if (m[1] && e.tagName.toLowerCase() !== m[1].toLowerCase()) return false;
  for (const c of m[2].match(/\.[\w-]+/g) || []) if (!e.classList.contains(c.slice(1))) return false;
  for (const a of m[3].match(/\[[\w-]+(?:="(?:[^"\\]|\\.)*")?\]/g) || []) {
    const am = a.match(/^\[data-([\w-]+)(?:="((?:[^"\\]|\\.)*)")?\]$/);
    if (!am || !(camel(am[1]) in e.dataset)) return false;
    if (am[2] !== undefined && e.dataset[camel(am[1])] !== am[2].replace(/\\(.)/g, '$1')) return false;
  }
  return true;
}

class FakeElement {
  constructor(doc, id, tag, parent) {
    Object.assign(this, {doc, id, tagName: tag || 'DIV', parent: parent || null, value: '', hidden: false, open: false,
                         dataset: {}, attrs: {}, listeners: {}, scrollTop: 0, scrollLeft: 0, clientWidth: 1500, offsetWidth: 0,
                         kids: [], writes: 0, onWrite: null, html: ''});
    this.classList = new FakeClassList();
  }
  get innerHTML() { return this.html; }
  /** Like a browser: the markup is replaced, so are the elements it holds (fresh, unscrolled ones). */
  set innerHTML(v) {
    this.html = String(v);
    this.writes += 1;
    this.kids = parseKids(this, this.html);
    if (this.onWrite) this.onWrite();
  }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  addEventListener(type, fn) { (this.listeners[type] = this.listeners[type] || []).push(fn); }
  removeEventListener(type, fn) { this.listeners[type] = (this.listeners[type] || []).filter((f) => f !== fn); }
  dispatch(type, extra) {
    const ev = Object.assign({target: this, defaultPrevented: false}, extra || {});
    ev.preventDefault = () => { ev.defaultPrevented = true; };
    for (const fn of (this.listeners[type] || []).slice()) fn(ev);
    return ev;
  }
  matches(sel) { return matchesSel(this, sel); }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
  focus() { this.doc.activeElement = this; this.dispatch('focus'); }
  blur() { if (this.doc.activeElement === this) this.doc.activeElement = this.doc.body; this.dispatch('blur'); }
  select() {}
  querySelectorAll(sel) { return this.kids.filter((e) => e.matches(sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  setPointerCapture(id) { this.captured = id; }
  scrollIntoView(opts) { this.doc.revealed.push({el: this, opts}); }
}

/** The elements app.js looks up after a render: those carrying data-scroll, data-key, data-fkey or data-node. */
function parseKids(parent, html) {
  const kids = [];
  for (const m of html.matchAll(/<([a-zA-Z][a-zA-Z0-9]*)\b([^>]*)>/g)) {
    const attrs = m[2];
    if (!/\sdata-(?:scroll|key|fkey|node)="/.test(attrs)) continue;
    const e = new FakeElement(parent.doc, null, m[1].toUpperCase(), parent);
    for (const [, name, raw] of attrs.matchAll(/\s([a-z-]+)="([^"]*)"/g)) {
      const v = unesc(raw);
      if (name.startsWith('data-')) e.dataset[camel(name.slice(5))] = v;
      else if (name === 'class') v.split(/\s+/).filter(Boolean).forEach((c) => e.classList.add(c));
      else e.attrs[name] = v;
    }
    e.open = /\sopen(?=\s|$)/.test(attrs.replace(/"[^"]*"/g, '""'));
    kids.push(e);
  }
  return kids;
}

function fakeEnv(opts) {
  const o = opts || {};
  const doc = {listeners: {}, title: '', visibilityState: 'visible', elements: {}, revealed: []};
  doc.body = new FakeElement(doc, 'body', 'BODY');
  doc.activeElement = doc.body;
  doc.documentElement = {style: {props: {}, setProperty(k, v) { this.props[k] = v; }}};
  doc.getElementById = (id) => doc.elements[id] || null;
  doc.addEventListener = (type, fn) => { (doc.listeners[type] = doc.listeners[type] || []).push(fn); };
  doc.dispatch = (type, ev) => {
    const e = Object.assign({defaultPrevented: false}, ev);
    e.preventDefault = () => { e.defaultPrevented = true; };
    for (const fn of doc.listeners[type] || []) fn(e);
    return e;
  };
  for (const id of ['task-input', 'task-menu', 'tab-task', 'tab-monitor', 'status', 'refresh', 'banner', 'main', 'drawer', 'drawer-grip']) {
    doc.elements[id] = new FakeElement(doc, id, id === 'task-input' ? 'INPUT' : 'DIV', doc.body);
  }
  doc.elements['task-input'].parent = new FakeElement(doc, 'picker', 'DIV', doc.body);
  doc.elements['task-input'].parent.classList.add('picker');
  doc.elements['task-menu'].parent = doc.elements['task-input'].parent;
  doc.elements.drawer.offsetWidth = 600;
  const win = {listeners: {}, timers: [], replaced: [], scrollX: 0, scrollY: 0, scrolls: [], innerWidth: 1500, selection: null,
               storage: Object.assign({}, o.storage)};
  let hash = o.hash || '';
  win.location = {
    pathname: '/', search: o.search || '',
    get hash() { return hash; },
    set hash(v) { hash = v.startsWith('#') ? v : `#${v}`; win.dispatch('hashchange'); },
  };
  win.history = {replaceState: (state, title, url) => {
    win.replaced.push(url);
    const m = String(url).match(/^([^?#]*)(\?[^#]*)?(#.*)?$/);
    if (m[1]) win.location.pathname = m[1];
    win.location.search = m[2] || '';
    hash = m[3] || '';
  }};
  win.addEventListener = (type, fn) => { (win.listeners[type] = win.listeners[type] || []).push(fn); };
  win.dispatch = (type) => { for (const fn of win.listeners[type] || []) fn({type}); };
  win.setInterval = (fn, ms) => { win.timers.push({fn, ms, interval: true}); return win.timers.length; };
  win.setTimeout = (fn, ms) => { win.timers.push({fn, ms}); return win.timers.length; };
  win.clearInterval = () => {};
  win.clearTimeout = () => {};
  win.scrollTo = (x, y) => { win.scrollX = x; win.scrollY = y; win.scrolls.push([x, y]); };
  win.getSelection = () => win.selection || {isCollapsed: true, toString: () => ''};
  win.localStorage = {getItem: (k) => (k in win.storage ? win.storage[k] : null), setItem: (k, v) => { win.storage[k] = String(v); }};
  const files = Object.assign({
    'data/index.json': F.index, [`data/task/${NAR}.json`]: F.narwhals,
    [`data/task/${BAN}.json`]: F.bandit, 'data/fleet.json': F.fleet, 'data/api.json': F.api,
  }, o.files || {});
  const calls = [];
  const ctl = {fail: null, notModified: null};                    // RegExps: matching fetches fail / answer 304
  const fetch = async (url, init) => {
    calls.push({url, init});
    if (o.status) return {status: o.status, ok: false, json: async () => ({})};
    if (ctl.fail && ctl.fail.test(url)) throw new TypeError('NetworkError when attempting to fetch resource.');
    if (ctl.notModified && ctl.notModified.test(url)) return {status: 304, ok: false, json: async () => { throw new Error('no body'); }};
    if (!(url in files) || (o.missing || []).includes(url)) return {status: 404, ok: false, json: async () => ({})};
    return {status: 200, ok: true, json: async () => clone(files[url])};
  };
  /** The collector writes a new cycle: every file gets the same newer stamp. */
  const newCycle = (dt) => {
    for (const k of Object.keys(files)) files[k] = Object.assign(clone(files[k]), {generated_at: files[k].generated_at + (dt || 120)});
  };
  const tick = (ms) => win.timers.find((t) => t.interval && t.ms === ms).fn();
  return {env: {window: win, document: doc, fetch, now: () => NOW}, win, doc, calls, ctl, files, newCycle, tick,
          el: (id) => doc.elements[id]};
}

async function settle(a) { for (let i = 0; i < 5; i += 1) { await a.idle(); await new Promise((r) => setImmediate(r)); } }

test('wiring: default task, node drawer, abandoned toggle, Escape, monitor tab', async () => {
  const f = fakeEnv({search: '?token=s3cr3t-token'});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.ok(f.win.replaced.includes('/'), 'token removed from the address bar');
  assert.equal(f.win.location.search, '');
  assert.equal(f.win.location.hash, `#task=${NAR}`);
  assert.ok(f.calls.every((c) => c.init.credentials === 'same-origin' && c.init.cache === 'no-cache'), 'F13: revalidating fetches');
  assert.ok(f.calls.some((c) => c.url === `data/task/${NAR}.json`));
  const main = f.el('main');
  assert.match(main.innerHTML, /Generation tree/);
  assert.match(main.innerHTML, /Round 0 at a glance/);
  assert.match(main.innerHTML, /Kimi K3/);
  assert.match(f.el('status').innerHTML, /updated \d\d:\d\d:\d\d/);
  assert.equal(f.el('task-input').value, NAR);
  assert.ok(f.el('tab-task').classList.contains('active'));
  assert.equal(a.state.ui.treeWidth, 1410);                      // measured: clientWidth 1500 - 90
  assert.equal(f.el('drawer-grip').hidden, true);                 // no drawer, no grip

  // Click a node card: the drawer opens with its detail, the hash records it, the card comes into view.
  const card = new FakeElement(f.doc, null, 'DIV', main);
  card.dataset.node = 'cand-c04';
  card.classList.add('node');
  f.doc.dispatch('click', {target: card});
  assert.equal(f.el('drawer').hidden, false);
  assert.equal(f.el('drawer-grip').hidden, false);
  assert.match(f.el('drawer').innerHTML, /ACCEPTED\+EVOLVE/);
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04`);
  assert.ok(f.doc.body.classList.contains('drawer-open'));
  assert.match(main.innerHTML, /class="node tone-accepted-evolve selected"/);
  assert.match(main.innerHTML, /<tr class="r0-row selected" data-node="cand-c04">/);
  const shown = f.doc.revealed[f.doc.revealed.length - 1];
  assert.ok(shown.el.classList.contains('node') && shown.el.dataset.node === 'cand-c04', 'the tree card was scrolled into view');
  assert.deepEqual(shown.opts, {block: 'nearest', inline: 'nearest'});
  assert.equal(f.doc.activeElement.dataset.fkey, 'card:cand-c04');

  // Toggle state of a <details> survives re-renders.
  const details = new FakeElement(f.doc, null, 'DETAILS', main);
  details.dataset.key = 'fa:markdown';
  details.open = true;
  f.doc.dispatch('toggle', {target: details});
  const html = main.innerHTML;
  a.render();                                                    // nothing changed: the DOM is left alone
  assert.equal(main.innerHTML, html);
  a.render(true);                                                // a real re-render keeps the block open
  assert.match(main.innerHTML, /data-key="fa:markdown" open/);

  // Abandoned toggle and collapse-all.
  const toggle = new FakeElement(f.doc, null, 'BUTTON', main);
  toggle.dataset.action = 'toggle-abandoned';
  f.doc.dispatch('click', {target: toggle});
  assert.match(main.innerHTML, /Hide abandoned \(5\)/);
  const collapse = new FakeElement(f.doc, null, 'BUTTON', main);
  collapse.dataset.action = 'collapse-lineages';
  f.doc.dispatch('click', {target: collapse});
  assert.equal(count(main.innerHTML, '<details class="lineage" data-key="lin:cand-c0'), 5);
  assert.doesNotMatch(main.innerHTML, /<details class="lineage"[^>]* open>/);

  // Escape closes the drawer.
  f.doc.dispatch('keydown', {key: 'Escape', target: f.doc.body});
  assert.equal(f.el('drawer').hidden, true);
  assert.equal(f.el('drawer-grip').hidden, true);
  assert.equal(f.win.location.hash, `#task=${NAR}`);

  // Keyboard activation of a card.
  const card2 = new FakeElement(f.doc, null, 'DIV', main);
  card2.dataset.node = 'cand-c05';
  card2.setAttribute('role', 'button');
  f.doc.dispatch('keydown', {key: 'Enter', target: card2});
  assert.match(f.el('drawer').innerHTML, /BUILD FAILED/);

  // Monitor tab.
  f.win.location.hash = '#monitor';
  await settle(a);
  assert.match(main.innerHTML, /Inference workers/);
  assert.match(main.innerHTML, /Hub API usage/);
  assert.equal(f.el('drawer').hidden, true);
  assert.ok(f.el('tab-monitor').classList.contains('active'));
  assert.equal(f.el('tab-task').getAttribute('href'), `#task=${NAR}`);

  // Auto-refresh every 60 s: index.json always; fleet/api only once a new collector cycle has landed.
  const urls = (from) => f.calls.slice(from).map((c) => c.url).sort();
  let before = f.calls.length;
  f.tick(60000);
  await settle(a);
  assert.deepEqual(urls(before), ['data/index.json'], 'same cycle: only index.json');
  f.newCycle();
  before = f.calls.length;
  f.tick(60000);
  await settle(a);
  assert.deepEqual(urls(before), ['data/api.json', 'data/fleet.json', 'data/index.json'], 'new cycle: fleet and api too');
  before = f.calls.length;
  const button = new FakeElement(f.doc, null, 'BUTTON', f.doc.body);
  button.dataset.action = 'refresh';
  f.doc.dispatch('click', {target: button});
  await settle(a);
  assert.deepEqual(urls(before), ['data/api.json', 'data/fleet.json', 'data/index.json'], 'Refresh: everything');
  a.stop();
});

test('wiring: task picker search, keyboard choice and hash navigation', async () => {
  const f = fakeEnv({hash: `#task=${NAR}`});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const input = f.el('task-input'), menu = f.el('task-menu');
  input.focus();
  assert.equal(menu.hidden, false);
  assert.match(menu.innerHTML, /Converting/);
  assert.match(menu.innerHTML, /Eligible queue/);
  input.value = 'bandit';
  input.dispatch('input');
  assert.match(menu.innerHTML, /bandit-incremental-cache-control/);
  assert.doesNotMatch(menu.innerHTML, /narwhals-rolling-window-suite/);
  input.dispatch('keydown', {key: 'ArrowDown'});
  input.dispatch('keydown', {key: 'Enter'});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${BAN}`);
  assert.equal(menu.hidden, true);
  assert.match(f.el('main').innerHTML, /klen\/pylama/);
  // Clicking an option in the menu.
  input.focus();
  const opt = new FakeElement(f.doc, null, 'DIV', menu);
  opt.dataset.task = 'go-git-worktree-merge-conflicts';
  f.doc.dispatch('click', {target: opt});
  await settle(a);
  assert.match(f.el('main').innerHTML, /Not converted yet/);
  assert.ok(!f.calls.some((c) => c.url.includes('go-git')), 'no detail fetch for a task without a run');
  // Escape in the picker closes the menu.
  input.focus();
  input.dispatch('keydown', {key: 'Escape'});
  assert.equal(menu.hidden, true);
  a.stop();
});

test('wiring: 403 shows the token message; missing detail and network errors are reported; 304 keeps the copy', async () => {
  const denied = fakeEnv({status: 403});
  const a = app.createApp(denied.env);
  await a.start();
  await settle(a);
  assert.match(denied.el('main').innerHTML, /Missing or expired token: open the URL with \?token=\.\.\./);
  assert.equal(denied.el('drawer').hidden, true);

  const missing = fakeEnv({hash: `#task=${NAR}`, missing: [`data/task/${NAR}.json`]});
  const b = app.createApp(missing.env);
  await b.start();
  await settle(b);
  assert.match(missing.el('main').innerHTML, /Run detail not available: data\/task\/narwhals-rolling-window-suite\.json: HTTP 404/);
  assert.match(missing.el('main').innerHTML, /official1-narwhals/);   // the run bar still shows

  const flaky = fakeEnv({hash: `#task=${NAR}`});
  const c = app.createApp(flaky.env);
  await c.start();
  await settle(c);
  flaky.ctl.fail = /index/;
  await c.load(true);
  await settle(c);
  assert.match(flaky.el('banner').innerHTML, /Refresh failed \(index\):<\/b> NetworkError/);
  assert.match(flaky.el('main').innerHTML, /Generation tree/);         // the last data stays on screen
  flaky.ctl.fail = null;
  await c.load(true);
  await settle(c);
  assert.doesNotMatch(flaky.el('banner').innerHTML, /Refresh failed/);

  // A failed refresh of the task file keeps the older copy on screen and says so in the banner.
  flaky.ctl.fail = /data\/task\//;
  await c.load('all');
  await settle(c);
  assert.match(flaky.el('banner').innerHTML, /Refresh failed \(task\/narwhals-rolling-window-suite\)/);
  assert.match(flaky.el('main').innerHTML, /Generation tree/);
  flaky.ctl.fail = null;

  // Loads requested while one runs are coalesced into one more run, as forced as the strongest request.
  let before = flaky.calls.length;
  const running = c.load(true);
  c.load(false);
  c.load('all');
  c.load(false);
  await running;
  await settle(c);
  const urls = flaky.calls.slice(before).map((x) => x.url);
  assert.deepEqual(urls, ['data/index.json', 'data/index.json', `data/task/${NAR}.json`]);

  // HTTP 304 (the browser revalidated an unchanged file): the copy held stays, no error.
  flaky.ctl.notModified = /./;
  before = flaky.calls.length;
  await c.load('all');
  await settle(c);
  assert.deepEqual(flaky.calls.slice(before).map((x) => x.url), ['data/index.json', `data/task/${NAR}.json`]);
  assert.doesNotMatch(flaky.el('banner').innerHTML, /Refresh failed/);
  assert.equal(c.state.index.generated_at, F.index.generated_at);
  assert.equal(c.state.details[NAR].task_id, NAR);
  assert.match(flaky.el('main').innerHTML, /Generation tree/);
});

test('wiring: a refresh without a new collector cycle leaves the page alone', async () => {
  const f = fakeEnv({hash: `#task=${NAR}&node=cand-c04`});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), drawer = f.el('drawer'), banner = f.el('banner');
  assert.match(drawer.innerHTML, /cand-c04/);
  const writes = () => [main.writes, drawer.writes, banner.writes];
  const w0 = writes();
  for (let i = 0; i < 3; i += 1) { f.tick(60000); await settle(a); }
  f.tick(15000);
  assert.deepEqual(writes(), w0, 'main, drawer and banner untouched');
  assert.equal(f.calls.filter((c) => c.url === `data/task/${NAR}.json`).length, 1, 'the task file is not refetched');
  f.newCycle();
  f.tick(60000);
  await settle(a);
  assert.deepEqual(writes().slice(0, 2), [w0[0] + 1, w0[1] + 1], 'a new cycle re-renders once');
  a.stop();
});

test('wiring: a new cycle keeps window, tree, drawer and <pre> scroll positions and open <details>', async () => {
  const f = fakeEnv({hash: `#task=${NAR}&node=cand-c04`});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), drawer = f.el('drawer');
  const one = (root, sel) => { const e = root.querySelector(sel); assert.ok(e, sel); return e; };
  // The user scrolled: the page, the tree sideways, the log, the drawer and a document inside it; opened a fold
  // whose toggle event never arrived.
  f.win.scrollX = 0;
  f.win.scrollY = 1840;
  one(main, '[data-scroll="tree:narwhals-rolling-window-suite"]').scrollLeft = 310;
  one(main, '[data-scroll="log:official1-narwhals"]').scrollTop = 120;
  drawer.scrollTop = 450;
  one(drawer, '[data-scroll="nd:cand-c04:instruction_md"]').scrollTop = 66;
  one(main, 'details[data-key="plan:excerpt"]').open = true;
  main.onWrite = () => { f.win.scrollY = 0; };                    // a browser may jump while the markup is replaced
  const writes = main.writes;
  f.newCycle();
  f.tick(60000);
  await settle(a);
  assert.equal(main.writes, writes + 1);
  assert.equal(one(main, '[data-scroll="tree:narwhals-rolling-window-suite"]').scrollLeft, 310);
  assert.equal(one(main, '[data-scroll="log:official1-narwhals"]').scrollTop, 120);
  assert.equal(drawer.scrollTop, 450);
  assert.equal(one(drawer, '[data-scroll="nd:cand-c04:instruction_md"]').scrollTop, 66);
  assert.equal(f.win.scrollY, 1840);
  assert.deepEqual(f.win.scrolls[f.win.scrolls.length - 1], [0, 1840]);
  assert.match(main.innerHTML, /data-key="plan:excerpt" open>/);
  assert.equal(one(main, '[data-scroll="fa:markdown"]').scrollTop, 0, 'only what was scrolled');
  // Another node: the drawer starts at its top; the page stays where it was.
  one(drawer, '[data-scroll="nd:cand-c04:instruction_md"]').scrollTop = 80;
  a.selectNode('cand-c04-r1b0', 'link');
  assert.equal(drawer.scrollTop, 0);
  assert.equal(one(drawer, '[data-scroll="nd:cand-c04-r1b0:instruction_md"]').scrollTop, 0);
  assert.equal(f.win.scrollY, 1840);
  assert.equal(f.doc.revealed[f.doc.revealed.length - 1].el.dataset.node, 'cand-c04-r1b0', 'its tree card is revealed');
  // Another task: from the top.
  f.win.location.hash = `#task=${BAN}`;
  await settle(a);
  assert.equal(f.win.scrollY, 0);
  assert.match(main.innerHTML, /klen\/pylama/);
  a.stop();
});

test('wiring: an automatic refresh waits while text is selected; user actions do not', async () => {
  const f = fakeEnv({hash: `#task=${NAR}&node=cand-c04`});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), drawer = f.el('drawer'), status = f.el('status');
  const w0 = [main.writes, drawer.writes];
  const shownAt = app.fmtClock(F.index.generated_at, true);
  f.win.selection = {isCollapsed: false, toString: () => 'sc14-r2'};
  f.newCycle();
  f.tick(60000);
  await settle(a);
  assert.deepEqual([main.writes, drawer.writes], w0, 'held: the selection survives');
  assert.equal(a.state.index.generated_at, F.index.generated_at + 120, 'the new cycle is loaded');
  assert.match(status.innerHTML, /newer data held/);
  assert.ok(status.innerHTML.includes(`updated ${shownAt}`), 'the indicator keeps the time of what is on screen');
  f.tick(15000);                                                  // still selected: still held
  assert.deepEqual([main.writes, drawer.writes], w0);
  f.tick(60000);
  await settle(a);
  assert.deepEqual([main.writes, drawer.writes], w0, 'the next 60 s tick holds again');
  f.win.selection = null;                                         // selection cleared: the next tick renders
  f.tick(15000);
  assert.deepEqual([main.writes, drawer.writes], [w0[0] + 1, w0[1] + 1]);
  assert.doesNotMatch(status.innerHTML, /held/);
  assert.ok(status.innerHTML.includes(`updated ${app.fmtClock(F.index.generated_at + 120, true)}`));
  // A click while text is selected re-renders at once.
  f.win.selection = {isCollapsed: false, toString: () => 'x'};
  a.selectNode('cand-c05', 'card');
  assert.match(drawer.innerHTML, /BUILD FAILED/);
  assert.equal(main.writes, w0[0] + 2);
  a.stop();
});

test('wiring: round-0 rows open the drawer; the drawer is resized by its grip and keeps the width', async () => {
  const f = fakeEnv({hash: `#task=${NAR}`, storage: {'swe-task-dashboard.drawer-width': '520'}});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const props = f.doc.documentElement.style.props;
  assert.equal(props['--drawer-w'], '520px', 'the kept width is applied at start');
  // A click anywhere in a round-0 row opens that node; its row button takes the focus.
  const row = new FakeElement(f.doc, null, 'TR', f.el('main'));
  row.classList.add('r0-row');
  row.dataset.node = 'cand-c02';
  const cell = new FakeElement(f.doc, null, 'TD', row);
  f.doc.dispatch('click', {target: cell});
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c02`);
  assert.match(f.el('drawer').innerHTML, /DISCARD/);
  assert.equal(f.doc.activeElement.dataset.fkey, 'r0:cand-c02');
  // Drag the grip 100 px to the left: 100 px wider; kept on release, and the tree re-measured.
  const grip = f.el('drawer-grip');
  const writes = f.el('main').writes;
  grip.dispatch('pointerdown', {button: 0, clientX: 900, pointerId: 7});
  assert.equal(grip.captured, 7);
  assert.ok(f.doc.body.classList.contains('is-resizing'));
  grip.dispatch('pointermove', {clientX: 850});
  assert.equal(props['--drawer-w'], '650px');
  grip.dispatch('pointermove', {clientX: 800});
  assert.equal(props['--drawer-w'], '700px');
  assert.equal(f.win.storage['swe-task-dashboard.drawer-width'], '520', 'kept only on release');
  grip.dispatch('pointerup', {clientX: 800});
  assert.equal(f.win.storage['swe-task-dashboard.drawer-width'], '700');
  assert.ok(!f.doc.body.classList.contains('is-resizing'));
  assert.equal(grip.getAttribute('aria-valuenow'), '700');
  grip.dispatch('pointermove', {clientX: 100});                    // released: no more tracking
  assert.equal(props['--drawer-w'], '700px');
  assert.ok(f.el('main').writes >= writes, 'rendered after the drag');
  // Limits: at least 360 px, and the page keeps 360 px.
  grip.dispatch('pointerdown', {button: 0, clientX: 900});
  grip.dispatch('pointermove', {clientX: 2000});
  assert.equal(props['--drawer-w'], '360px');
  grip.dispatch('pointermove', {clientX: -500});
  assert.equal(props['--drawer-w'], '1140px');                      // 1500 - 360
  grip.dispatch('pointerup', {});
  grip.dispatch('pointerdown', {button: 2, clientX: 900});        // not the primary button
  assert.ok(!f.doc.body.classList.contains('is-resizing'));
  // Keyboard: Left widens, Right narrows (from the drawer's measured 600 px).
  grip.dispatch('keydown', {key: 'ArrowLeft'});
  assert.equal(props['--drawer-w'], '640px');
  grip.dispatch('keydown', {key: 'ArrowRight'});
  assert.equal(props['--drawer-w'], '560px');
  assert.equal(f.win.storage['swe-task-dashboard.drawer-width'], '560');
  assert.equal(grip.dispatch('keydown', {key: 'a'}).defaultPrevented, false);
  a.stop();
});

test('wiring: failure-match audit rows open the drawer and keep the focus', async () => {
  const f = fakeEnv({hash: `#task=${NAR}`, files: {'data/index.json': AI, [`data/task/${NAR}.json`]: A}});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main');
  assert.match(main.innerHTML, /<h2>Failure-match audit<\/h2><span class="muted small">5 accepted tasks · audit 4\/5/);
  assert.match(main.innerHTML, /<b title="[^"]*">audit 4\/5<\/b>/);
  const row = new FakeElement(f.doc, null, 'TR', main);
  row.classList.add('fa-row');
  row.dataset.node = 'cand-c04-r1b2';
  f.doc.dispatch('click', {target: new FakeElement(f.doc, null, 'TD', row)});
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04-r1b2`);
  assert.match(f.el('drawer').innerHTML, /Failure-pattern match \(audit\)[^]*FA match: weak/);
  assert.equal(f.doc.activeElement.dataset.fkey, 'audit:cand-c04-r1b2');   // the row keeps the focus
  assert.match(main.innerHTML, /<tr class="fa-row selected" data-node="cand-c04-r1b2">/);
  assert.match(main.innerHTML, /class="node tone-good selected"[^>]*data-node="cand-c04-r1b2"/);
  a.stop();
});

test('wiring (v1.4): rows of eligible tasks that were not accepted open the drawer; group headers open nothing', async () => {
  const f = fakeEnv({hash: `#task=${NAR}`, files: {'data/index.json': EI, [`data/task/${NAR}.json`]: E}});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main');
  assert.match(main.innerHTML, /<h2>Failure-match audit<\/h2><span class="muted small">36 eligible tasks · audit 7\/36/);
  assert.match(main.innerHTML, /<b title="eligible tasks [^"]*">audit 7\/36<\/b>/);
  const row = new FakeElement(f.doc, null, 'TR', main);
  row.classList.add('fa-row');
  row.dataset.node = 'cand-c04-r1b1';
  f.doc.dispatch('click', {target: new FakeElement(f.doc, null, 'TD', row)});
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04-r1b1`);
  assert.match(f.el('drawer').innerHTML, /Failure-pattern match \(audit\)[^]*FA match: strong/);
  assert.equal(f.doc.activeElement.dataset.fkey, 'audit:cand-c04-r1b1');   // the row keeps the focus
  assert.match(main.innerHTML, /<tr class="fa-row selected" data-node="cand-c04-r1b1">/);
  assert.match(main.innerHTML, /class="node tone-evolve selected"[^>]*data-node="cand-c04-r1b1"/);
  const group = new FakeElement(f.doc, null, 'TR', main);
  group.classList.add('fa-group');
  f.doc.dispatch('click', {target: new FakeElement(f.doc, null, 'TH', group)});
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04-r1b1`);   // unchanged
  a.stop();
});

test('worker status follows collect.py routing like the fleet daemon: stale / draining / ready', () => {
  const status = (w) => (app.renderWorkers({workers: [Object.assign({job_id: '7', slot: '01', state: 'RUNNING'}, w)]}, NOW)
    .match(/<\/span>([a-z]+)<\/span>/) || [])[1];
  assert.equal(status({registry_status: 'ready', routing: 'ready'}), 'ready');
  assert.equal(status({registry_status: 'ready', routing: 'draining'}), 'draining');   // past its drain time
  assert.equal(status({registry_status: 'ready', routing: null}), 'stale');            // heartbeat > 120 s or past end
  assert.equal(status({registry_status: 'draining', routing: null}), 'stale');
  assert.equal(status({registry_status: 'starting', routing: null}), 'starting');
  assert.equal(status({registry_status: 'ready'}), 'ready');                           // no routing field: as listed
  assert.equal(status({state: 'PENDING', registry_status: null, routing: null}), 'pending');
});

// ------------------------------------------------------------------------------------- batches (v1.6)
// A batch from another tree (CONTRACT "v1.6", Baihe's on the live page): index.json runs[] entries with a `source`, a
// task's batch_runs, index.json batches, and one document per batch run (data/run/<source>/<run>.json), never audited.
// Here: two batch runs on go-git-worktree-merge-conflicts (a task without an official run), their documents made from
// the narwhals fixture as a batch collector writes them (the second run stopped, its last round unfinished).
const BT = 'go-git-worktree-merge-conflicts';
const BSRC = 'testbatch';
const BRUNS = ['pilotA-go-git', 'pilotB-go-git'];
const BATCH_DOCS = BRUNS.map((run, i) => {
  const d = clone(F.narwhals);
  Object.assign(d, {task_id: BT, run, source: BSRC, source_name: 'Testy', source_label: 'Testy · fork',
                    root: `/b/synth/runs/${run}`, outputs: '/b/outputs', audit: {status: 'not_audited'},
                    generated_at: F.index.generated_at - 600 * (i + 1)});
  for (const r of d.rounds) {
    r.unfinished_probes = [];
    if (r.probe) r.probe.live = false;
  }
  for (const n of d.nodes) {
    if (n.status === 'probing') {                       // an ended run: its unscored tasks stay unscored
      Object.assign(n, {status: 'built', probe: null, decision: 'pending', decision_reason:
                        `its probe round-${n.round}/probe-x never finished: the run ${i ? 'stopped' : 'finished'} before scoring it`});
      if (i) d.rounds[n.round].unfinished_probes = [`round-${n.round}/probe-x`];
    }
    Object.assign(n, {failure_match: null, audit_eligible: false, audit: {status: 'not_audited'},
                      in_batch: n.decision === 'accepted' || n.decision === 'accepted_evolve'});
  }
  if (i) d.rounds.forEach((r) => { if (r.unfinished_probes.length) r.status = 'unfinished'; });
  d.batch = d.nodes.filter((n) => n.in_batch).length;
  const status = i ? 'stopped' : 'finished';
  d.job = {job_id: `20${i + 1}`, slurm_state: null, partition: 'cpu_dataprocessing', node: null, elapsed: null,
           time_left: null, log_path: null, log_tail: [], status, jobs: [`20${i + 1}`],
           stage: i ? 'stopped: round-3 unfinished (2/7 tasks scored, 1 probe unfinished)' : `finished (${d.batch} accepted)`};
  return d;
});
function batchIndex(docs) {
  const ix = clone(F.index);
  ix.tasks.find((t) => t.task_id === BT).batch_runs = docs.map((d) => ({source: BSRC, run: d.run}));
  for (const [i, d] of docs.entries()) {
    const shown = d.nodes.filter((n) => n.status !== 'abandoned');
    ix.runs.push({run: d.run, task_id: BT, repo: 'o/gogit', job_id: d.job.job_id, slurm_state: null,
                  partition: 'cpu_dataprocessing', submitted_at: F.index.generated_at - 86400 + i * 3600, started_at: null,
                  ended_at: null, status: d.job.status, stage: d.job.stage, rounds_started: 4,
                  accepted: shown.filter((n) => n.decision === 'accepted' || n.decision === 'accepted_evolve').length,
                  tasks_built: shown.filter((n) => n.build.ok === true).length, tasks_scored: shown.filter((n) => n.score).length,
                  audit: {status: 'not_audited'}, source: BSRC, source_name: 'Testy', source_label: 'Testy · fork',
                  root: d.root, batch: d.batch, detail: `run/${BSRC}/${d.run}.json`, detail_generated_at: d.generated_at});
  }
  const total = docs.reduce((s, d) => s + d.batch, 0);
  ix.batches = [{source: BSRC, name: 'Testy', label: 'Testy · fork', root: '/b', outputs: '/b/outputs', tasks: total,
                 runs: docs.length, targets: 1, in_batch: total, accepted: total, unmatched: [], not_in_batch: [],
                 audit: {status: 'not_audited'}}];
  return ix;
}
const BI = batchIndex(BATCH_DOCS);
const IN_BATCH = BATCH_DOCS[0].nodes.filter((n) => n.in_batch).map((n) => n.id);

test('batch (v1.6): the picker groups the batch\'s tasks under "<name> batch · not audited", apart from Converting', () => {
  const groups = app.taskGroups(BI, '');
  assert.deepEqual(groups.map((g) => g.key), ['converting', `batch:${BSRC}`, 'eligible', 'all']);
  assert.equal(groups[1].label, 'Testy batch · not audited');
  assert.deepEqual(groups[1].items.map((t) => t.task_id), [BT]);
  const plain = app.taskGroups(F.index, '');                       // the official groups are the same as without it
  assert.deepEqual(plain.map((g) => g.key), ['converting', 'eligible', 'all']);
  for (const [i, j] of [[0, 0], [2, 1], [3, 2]]) {
    assert.deepEqual(groups[i].items.map((t) => t.task_id), plain[j].items.map((t) => t.task_id));
  }
  assert.equal(app.defaultTask(BI), NAR);                          // still the first converting official task
  const menu = app.renderTaskMenu(BI, '', null, 0);
  assert.match(menu, /<div class="menu-head" title="Testy · fork: \d+ tasks \(outputs\/tasks\.jsonl\) from 2 runs on 1 DeepSWE tasks;[^"]*">Testy batch · not audited<span>1<\/span>/);
  assert.match(menu, new RegExp(`data-task="${BT}"[^>]*><span class="opt-id">${BT}</span><span class="opt-meta"><span class="badge tone-evolve badge-sm"[^>]*>` +
                                `<span class="dot" aria-hidden="true"></span>2 runs</span><span class="rank">${IN_BATCH.length * 2} in batch</span>`));
  assert.equal(app.menuCount(BI, ''), app.menuCount(F.index, '') + 1);
  for (const q of ['pilotB-go', 'o/gogit']) assert.ok(app.renderTaskMenu(BI, q, null, 0).includes(`data-task="${BT}"`), q);
  assert.equal(app.menuOption(BI, 'pilotA-go', 0), BT);
});

test('batch (v1.6): a task with two batch runs shows a run selector and the selected run, never audited', () => {
  const html = app.renderTaskView(BI, BT, BATCH_DOCS[0], {now: NOW, ui: {}});
  assert.ok(!html.includes('render-error'), html.match(/render-error[^<]*/) || '');
  // the run selector: both runs, the first (oldest launch) shown when the hash names none
  assert.equal(count(html, 'data-run-select="'), 2);
  assert.match(html, /<button type="button" class="run-tab active" data-run-select="pilotA-go-git" aria-pressed="true">/);
  assert.match(html, /<button type="button" class="run-tab" data-run-select="pilotB-go-git" aria-pressed="false">/);
  assert.match(html, /2 runs convert this task/);
  // the run bar: the batch, its own totals, launches from launch.jsonl (not polled), no log, "not audited"
  assert.match(html, /<div class="batch-strip"><span class="badge tone-muted"[^>]*><span class="dot" aria-hidden="true"><\/span>Testy batch · not audited<\/span><b class="batch-label">Testy · fork<\/b>/);
  assert.match(html, /<code class="batch-root" title="\/b\/synth\/runs\/pilotA-go-git">/);
  assert.match(html, /the batch: \d+ tasks · 2 runs · 1 DeepSWE task</);
  assert.match(html, /<div class="runcell-k">Failure-match audit<\/div><div class="runcell-v"><span class="badge tone-muted"[^>]*><span class="dot" aria-hidden="true"><\/span>not audited<\/span>/);
  assert.match(html, new RegExp(`<div class="runcell-k">In batch</div><div class="runcell-v"><span class="big-num">${IN_BATCH.length}</span>`));
  assert.match(html, /from launch\.jsonl · its jobs are not polled/);
  assert.match(html, /Log not read: Testy&#39;s Slurm logs are outside the dashboard&#39;s read roots/);
  assert.match(html, /<span class="stage-k">Stage<\/span> finished \(\d+ accepted\)/);
  // the same sections as an official run, the batch's tasks in place of the failure-match audit table
  const order = ['Target failure patterns', 'Plan', 'Round 0 at a glance', 'Testy batch · not audited</h2>', 'Generation tree'];
  const at = order.map((s) => html.indexOf(s));
  assert.ok(at.every((p, i) => p > 0 && (i === 0 || p > at[i - 1])), `section order ${at}`);
  assert.doesNotMatch(html, /FA match|Failure-match audit<\/h2>|Not audited yet/);
  assert.equal(count(html, 'class="fa-row'), IN_BATCH.length);   // one row per batch task
  for (const id of IN_BATCH) assert.match(html, new RegExp(`data-fkey="audit:${id}"`));
  assert.equal(count(html, 'badge-sm badge-fa"'), IN_BATCH.length);  // the table's "not audited" column
  // tree cards: "batch · not audited" exactly on the batch's tasks
  const L = app.treeLayout(BATCH_DOCS[0], {});
  const cards = L.lineages.flatMap((l) => l.items).map((it) => app.renderNodeCard(it, {}, L.geom, app.treeContext(BATCH_DOCS[0])));
  const marked = cards.filter((c) => c.includes('>batch · not audited</span>'));
  assert.equal(marked.length, IN_BATCH.length);
  assert.ok(marked.every((c) => /title="[^"]*in the batch \(outputs\/tasks\.jsonl\) · not audited/.test(c)));
  // the round-0 table marks the batch's round-0 tasks
  const r0 = BATCH_DOCS[0].nodes.filter((n) => n.round === 0 && n.in_batch).length;
  assert.equal(count(html.slice(html.indexOf('Round 0 at a glance'), html.indexOf('Testy batch · not audited</h2>')), '>batch</span>'), r0);
});

test('batch (v1.6): the other run, round heads of an ended run, and a document not loaded yet', () => {
  const html = app.renderTaskView(BI, BT, BATCH_DOCS[1], {now: NOW, ui: {}, run: 'pilotB-go-git'});
  assert.match(html, /<button type="button" class="run-tab active" data-run-select="pilotB-go-git" aria-pressed="true">/);
  assert.match(html, /<b>pilotB-go-git<\/b>/);
  assert.match(html, /<span class="badge tone-muted"[^>]*><span class="dot" aria-hidden="true"><\/span>stopped<\/span>/);
  assert.match(html, /<span class="badge tone-pending badge-sm"><span class="dot" aria-hidden="true"><\/span>unfinished<\/span>/);
  assert.match(html, /title="the run ended before these probes finished:\nround-3\/probe-x">1 unfinished probe</);
  assert.match(html, /never finished: the run stopped before scoring it/);
  // the wrong document for the selected run (another run's, an official one) is never shown as this run's
  for (const detail of [BATCH_DOCS[0], F.narwhals, null]) {
    const loading = app.renderTaskView(BI, BT, detail, {now: NOW, ui: {}, run: 'pilotB-go-git'});
    assert.match(loading, /Loading data\/run\/testbatch\/pilotB-go-git\.json…/);
    assert.doesNotMatch(loading, /Generation tree/);
  }
  assert.match(app.renderTaskView(BI, BT, null, {now: NOW, run: 'pilotB-go-git', detailError: 'HTTP 404'}), /Run detail not available: HTTP 404/);
  // an unknown run= falls back to the first run; a single batch run has no selector
  assert.match(app.renderTaskView(BI, BT, BATCH_DOCS[0], {now: NOW, run: 'nope'}), /class="run-tab active" data-run-select="pilotA-go-git"/);
  const one = batchIndex(BATCH_DOCS.slice(0, 1));
  const single = app.renderTaskView(one, BT, BATCH_DOCS[0], {now: NOW});
  assert.doesNotMatch(single, /data-run-select/);
  assert.match(single, /Testy batch · not audited/);
});

test('batch (v1.6): drawers say "not audited", and a batch task\'s drawer says where it is', () => {
  const doc = BATCH_DOCS[0];
  const inBatch = doc.nodes.find((n) => n.in_batch);
  const html = app.renderNodeDetail(inBatch, doc, {}, NOW);
  assert.match(html, /<div class="callout tone-muted"><div class="callout-k">Testy batch<\/div>/);
  assert.match(html, new RegExp(`<code>/b/outputs/${inBatch.build.task_id}/</code>`));
  assert.match(html, /<h3 class="drawer-h">Failure-pattern match \(audit\)<\/h3><p class="note">Not audited: Testy&#39;s batch is not audited; no failure-match audit runs on it\.<\/p>/);
  const scored = doc.nodes.find((n) => !n.in_batch && n.score && n.decision !== 'abandoned');
  const other = app.renderNodeDetail(scored, doc, {}, NOW);
  assert.match(other, /Not audited: Testy&#39;s batch is not audited/);
  assert.doesNotMatch(other, /callout-k">Testy batch/);
  const unscored = doc.nodes.find((n) => !n.score && !n.in_batch);
  assert.doesNotMatch(app.renderNodeDetail(unscored, doc, {}, NOW), /Failure-pattern match/);
  for (const n of doc.nodes) assert.doesNotMatch(app.renderNodeDetail(n, doc, {}, NOW), /render-error|Not audited yet|FA match/);
});

test('batch (v1.6): official views, drawers, menu and monitor render exactly as before', () => {
  // with the batch in index.json, an official task's view is what it was without it
  for (const [id, doc] of [[NAR, F.narwhals], [BAN, F.bandit], ['fd-deterministic-multi-key-sorting', F.fd]]) {
    for (const ui of [{}, {showAbandoned: true, selected: 'cand-c04', treeWidth: 1700}]) {
      assert.equal(app.renderTaskView(BI, id, doc, {now: NOW, ui}), app.renderTaskView(F.index, id, doc, {now: NOW, ui}), id);
    }
  }
  // and the same as the v1.5 page (static/app.js.pre-baihe-batch, kept next to app.js as the deploy's backup)
  const previous = path.join(STATIC, 'app.js.pre-baihe-batch');
  if (!fs.existsSync(previous)) return;
  const old = require(previous);
  const views = {narwhals: [NAR, F.narwhals], bandit: [BAN, F.bandit], fd: ['fd-deterministic-multi-key-sorting', F.fd],
                 noRun: [BT, null], unknown: ['no-such-task', null]};
  for (const [name, [id, doc]] of Object.entries(views)) {
    for (const ui of [{}, {showAbandoned: true, selected: 'cand-c04-r1b1', treeWidth: 1700}]) {
      assert.equal(app.renderTaskView(F.index, id, doc, {now: NOW, ui}), old.renderTaskView(F.index, id, doc, {now: NOW, ui}), name);
    }
  }
  for (const doc of [F.narwhals, F.bandit, withSeedNotes(withScope(F.narwhals))]) {
    for (const n of doc.nodes) assert.equal(app.renderNodeDetail(n, doc, {}, NOW), old.renderNodeDetail(n, doc, {}, NOW), n.id);
  }
  assert.equal(app.renderTaskMenu(F.index, '', NAR, 2), old.renderTaskMenu(F.index, '', NAR, 2));
  assert.equal(app.renderTaskMenu(F.index, 'bandit', NAR, 0), old.renderTaskMenu(F.index, 'bandit', NAR, 0));
  assert.equal(app.renderMonitor(F.fleet, F.api, {now: NOW}), old.renderMonitor(F.fleet, F.api, {now: NOW}));
  assert.equal(app.renderTaskView(EI, NAR, E, {now: NOW, ui: {}}), old.renderTaskView(EI, NAR, E, {now: NOW, ui: {}}));
});

test('escaping (v1.6): markup in the batch\'s labels, paths, stages and names never reaches the markup', () => {
  const docs = BATCH_DOCS.map((d) => {
    const x = clone(d);
    Object.assign(x, {source_label: PAYLOAD, source_name: PAYLOAD, root: PAYLOAD, outputs: PAYLOAD});
    x.job.stage = PAYLOAD;
    for (const n of x.nodes) { n.title = PAYLOAD; n.decision_reason = PAYLOAD; if (n.in_batch) n.build.task_id = PAYLOAD; }
    for (const r of x.rounds) r.unfinished_probes = [PAYLOAD];
    return x;
  });
  const ix = batchIndex(docs);
  for (const run of ix.runs.filter((r) => r.source)) Object.assign(run, {source_label: PAYLOAD, source_name: PAYLOAD, root: PAYLOAD, stage: PAYLOAD, repo: PAYLOAD});
  Object.assign(ix.batches[0], {label: PAYLOAD, name: PAYLOAD});
  assertEscaped(app.renderTaskView(ix, BT, docs[0], {now: NOW, ui: {showAbandoned: true}}), 'batch view');
  assertEscaped(app.renderTaskView(ix, BT, docs[1], {now: NOW, run: docs[1].run}), 'batch view 2');
  assertEscaped(app.renderTaskMenu(ix, '', BT, 0), 'menu');
  for (const n of docs[0].nodes) assertEscaped(app.renderNodeDetail(n, docs[0], {}, NOW), `drawer ${n.id}`);
});

test('wiring (v1.6): a batch task fetches its run\'s document, the selector switches runs, the drawer reads that run', async () => {
  const files = {'data/index.json': BI};
  for (const d of BATCH_DOCS) files[`data/run/${BSRC}/${d.run}.json`] = d;
  const f = fakeEnv({hash: `#task=${BT}`, files});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main');
  assert.ok(f.calls.some((c) => c.url === `data/run/${BSRC}/pilotA-go-git.json`));
  assert.ok(!f.calls.some((c) => c.url.startsWith('data/task/')), 'a batch task has no data/task file');
  assert.match(main.innerHTML, /class="run-tab active" data-run-select="pilotA-go-git"/);
  assert.match(main.innerHTML, /Generation tree/);
  // the selector: the hash names the run, its document is fetched, the view switches
  const tab = new FakeElement(f.doc, null, 'BUTTON', main);
  tab.dataset.runSelect = 'pilotB-go-git';
  f.doc.dispatch('click', {target: tab});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${BT}&run=pilotB-go-git`);
  assert.ok(f.calls.some((c) => c.url === `data/run/${BSRC}/pilotB-go-git.json`));
  assert.match(main.innerHTML, /class="run-tab active" data-run-select="pilotB-go-git"/);
  assert.match(main.innerHTML, /1 unfinished probe/);
  // a node of that run opens in the drawer; the hash keeps the run
  const id = BATCH_DOCS[1].nodes.find((n) => n.in_batch).id;
  const card = new FakeElement(f.doc, null, 'DIV', main);
  card.dataset.node = id;
  card.classList.add('node');
  f.doc.dispatch('click', {target: card});
  assert.equal(f.win.location.hash, `#task=${BT}&run=pilotB-go-git&node=${id}`);
  assert.match(f.el('drawer').innerHTML, /callout-k">Testy batch/);
  // refreshes: a batch run's document is fetched again only when index.json names a newer one
  const before = f.calls.length;
  f.tick(60000);
  await settle(a);
  assert.deepEqual(f.calls.slice(before).map((c) => c.url), ['data/index.json']);
  f.newCycle();                                         // a new cycle: index.json moves on, the batch docs did not change
  for (const d of BATCH_DOCS) f.files[`data/run/${BSRC}/${d.run}.json`].generated_at -= 120;   // (not rewritten)
  const later = f.calls.length;
  f.tick(60000);
  await settle(a);
  assert.deepEqual(f.calls.slice(later).map((c) => c.url), ['data/index.json']);
  // the run's document changed: rewritten with the cycle's stamp, which index.json names; the page fetches it
  const changed = f.files['data/index.json'].generated_at;
  f.files[`data/run/${BSRC}/pilotB-go-git.json`].generated_at = changed;
  f.files['data/index.json'].runs.find((r) => r.run === 'pilotB-go-git').detail_generated_at = changed;
  const last = f.calls.length;
  f.tick(60000);
  await settle(a);
  assert.deepEqual(f.calls.slice(last).map((c) => c.url), ['data/index.json', `data/run/${BSRC}/pilotB-go-git.json`]);
  // back to the first run (its official-less default: no run= in the hash)
  const first = new FakeElement(f.doc, null, 'BUTTON', main);
  first.dataset.runSelect = 'pilotA-go-git';
  f.doc.dispatch('click', {target: first});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${BT}&run=pilotA-go-git`);
  assert.equal(f.el('drawer').hidden, true);               // another run: its nodes are others
  a.stop();
});

// ------------------------------------------------------------------------------------------- explorer (v1.7)
// data/explorer.json as the contract derives it (tests/explorer_fixture.js) from the narwhals fixture with audits,
// eligibility and seed notes (V; one task re-scored: its metrics.json student_rate rewritten after the probe, as a regrade
// or a withdrawal leaves it), the bandit fixture (an older collector's shapes: no audit scope), the fd run (no tree yet)
// and the two batch documents (BATCH_DOCS: never audited). EXI: the batch index plus index.json "explorer".
const {explorerOf} = require('./explorer_fixture.js');
const FD = 'fd-deterministic-multi-key-sorting';
const RESCORED = 'cand-c01';
const VR = clone(V);
nodeOf(VR, RESCORED).score.student_rate = 0.8;                       // its rollout rows still say 2 of 5 (0.4)
const EX_SOURCES = [{source: 'hengyuf', name: 'hengyuf', label: 'hengyuf · official runs', audited: true},
                    {source: BSRC, name: 'Testy', label: 'Testy · fork', audited: false}];
const EX_ENTRIES = [{source: 'hengyuf', detail: `task/${NAR}.json`, doc: VR}, {source: 'hengyuf', detail: `task/${BAN}.json`, doc: F.bandit},
                    {source: 'hengyuf', detail: `task/${FD}.json`, doc: F.fd},
                    ...BATCH_DOCS.map((d) => ({source: BSRC, detail: `run/${BSRC}/${d.run}.json`, doc: d}))];
const EXD = explorerOf(EX_ENTRIES, F.index.generated_at - 300, EX_SOURCES);
const EXI = Object.assign(clone(BI), {explorer: {path: 'explorer.json', generated_at: EXD.generated_at,
                                                 bytes: JSON.stringify(EXD).length, tasks: EXD.tasks.length,
                                                 trajectories: EXD.trajectories.length}});
const builtOf = (doc) => (doc.nodes || []).filter((n) => n.build && n.build.task_id);
const exRun = (over) => app.filterExplorer(app.explorerModel(EXD), Object.assign(app.explorerDefaults(), over));
const VERDICT = ['strong', 'partial', 'weak', 'none'];
const EX_IDS = ['tab-explorer', 'ex-results', 'ex-tlist', 'ex-tip', 'ex-head', 'ex-st-nj', 'ex-st-s', 'ex-st-n', 'ex-st-k', 'ex-st-targets'];
/** A fake page for the Explorer: the ids the wiring looks up after a render, as elements of their own. */
function exEnv(opts) {
  const f = fakeEnv(opts);
  for (const id of EX_IDS) f.doc.elements[id] = new FakeElement(f.doc, id, 'DIV', f.doc.body);
  return f;
}
const exFiles = () => {
  const files = {'data/index.json': EXI, 'data/explorer.json': EXD, [`data/task/${NAR}.json`]: VR};
  for (const d of BATCH_DOCS) files[`data/run/${BSRC}/${d.run}.json`] = d;
  return files;
};

test('explorer (v1.7): the hash round-trips every filter; defaults are "#explorer"; bad values and old links read sensibly', () => {
  const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');      // the navigation reaches it
  assert.match(html, /<a id="tab-monitor" class="tab" href="#monitor">Monitor<\/a>\s*<a id="tab-explorer" class="tab" href="#explorer">Explorer<\/a>/);
  assert.equal(app.explorerHash(app.explorerDefaults()), '#explorer');
  assert.deepEqual(app.parseHash('#explorer'), {view: 'explorer', task: null, node: null, run: null, explorer: ''});
  assert.deepEqual(app.parseHash('#explorer&round=1'), {view: 'explorer', task: null, node: null, run: null, explorer: 'round=1'});
  assert.equal(app.parseHash('#explorers').view, 'task');
  assert.deepEqual(app.parseHash('#monitor'), {view: 'monitor', task: null, node: null, run: null});   // unchanged
  const f = Object.assign(app.explorerDefaults(), {
    src: [BSRC], round: [0, 2], kind: ['evolve'], dec: 'acc', smin: 6, smax: 9, nj: true, nmin: 0, nmax: 0.2, kmin: 0.6,
    kmax: 1, kb: 'raw', rd: 'none', audit: ['strong', 'not_audited'], t: ['a-task', 'b-task'], tm: ['sc14'], ts: ['fail', 'infra'],
    tc: true, tw: true, sort: 'rescored', desc: true, p: 3, tp: 2, cov: 'traj'});
  const h = app.explorerHash(f);
  assert.equal(h, `#explorer&src=${BSRC}&round=0,2&kind=evolve&audit=strong,not_audited&tm=sc14&ts=fail,infra&dec=acc&smin=6&smax=9` +
                  '&nj=1&nmin=0&nmax=0.2&kmin=0.6&kmax=1&kb=raw&rd=none&t=a-task,b-task&tc=1&tw=1&sort=rescored:desc&p=3&tp=2&cov=traj');
  assert.deepEqual(app.parseExplorerParams(app.parseHash(h).explorer), f);
  assert.deepEqual(app.parseExplorerParams(h), f);
  // "k=" chooses nothing; a list of mistyped values is "every value" again; bad numbers, enums, pages are ignored
  assert.equal(app.explorerHash(Object.assign(app.explorerDefaults(), {round: []})), '#explorer&round=');
  assert.deepEqual(app.parseExplorerParams('round=').round, []);
  assert.deepEqual(app.parseExplorerParams('smin=abc&nmax=7&kmin=-1&kb=x&dec=?&rd=x&p=-1&tp=x&sort=evil:desc&round=x,1&src=<b>&audit=bogus,weak&tm=gpt&cov=x&t=,'),
                   Object.assign(app.explorerDefaults(), {round: [1], audit: ['weak']}));
  for (const k of ['smin', 'smax']) assert.equal(app.parseExplorerParams(`${k}=11`)[k], null);
  assert.equal(app.parseExplorerParams('nmax=80').nmax, null);           // the hash keeps fractions: 80 is no rate
  // links of the first v1.7 page: "sun=0" (leave out the never-judged tasks) and "sun=1" (keep them)
  assert.deepEqual(app.parseExplorerParams('sun=0'), Object.assign(app.explorerDefaults(), {smin: 0}));
  assert.deepEqual(app.parseExplorerParams('smin=6&sun=0'), Object.assign(app.explorerDefaults(), {smin: 6}));
  assert.deepEqual(app.parseExplorerParams('smin=8'), Object.assign(app.explorerDefaults(), {smin: 8}));   // no longer keeps them
  assert.deepEqual(app.parseExplorerParams('smin=8&sun=1'), Object.assign(app.explorerDefaults(), {smin: 8, nj: true}));
});

test('explorer (v1.7): range inputs: similarity 0-10, pass rates in percent; out-of-range or non-numeric input is not applied', () => {
  assert.deepEqual(app.exParseBound('smin', '6'), {value: 6, bad: ''});
  assert.deepEqual(app.exParseBound('nmax', '80'), {value: 0.8, bad: ''});
  assert.deepEqual(app.exParseBound('kmin', '33.3'), {value: 0.333, bad: ''});
  assert.deepEqual(app.exParseBound('nmin', ''), {value: null, bad: ''});
  assert.deepEqual(app.exParseBound('nmin', '150'), {value: null, bad: '150 is outside 0–100 %'});
  assert.deepEqual(app.exParseBound('smax', '11'), {value: null, bad: '11 is outside 0–10'});
  assert.deepEqual(app.exParseBound('smin', '-1'), {value: null, bad: '-1 is outside 0–10'});
  assert.equal(app.exParseBound('kmax', 'abc').bad, '“abc” is not a number');
  const m = app.explorerModel(EXD), d = app.explorerDefaults();
  let f = app.updateExplorerFilter(d, 'round', {value: '1', checked: false}, m);
  assert.deepEqual(f.round, [0, 2, 3]);
  f = app.updateExplorerFilter(f, 'round', {value: '1', checked: true}, m);
  assert.equal(f.round, null);                                           // every round checked again: no filter
  assert.deepEqual(app.updateExplorerFilter(d, 'src', {value: 'hengyuf', checked: false}, m).src, [BSRC]);
  assert.deepEqual(app.updateExplorerFilter(d, 'audit', {value: 'none', checked: false}, m).audit, app.EX_AUDITS.filter((a) => a !== 'none'));
  assert.equal(app.updateExplorerFilter(d, 'smin', {value: '6'}, m).smin, 6);
  assert.equal(app.updateExplorerFilter(d, 'nmax', {value: '80'}, m).nmax, 0.8);    // percent in, fraction kept
  assert.equal(app.updateExplorerFilter(Object.assign({}, d, {nmax: 0.2}), 'nmax', {value: '120'}, m).nmax, null);
  assert.equal(app.updateExplorerFilter(Object.assign({}, d, {smin: 6}), 'smin', {value: ''}, m).smin, null);
  f = app.updateExplorerFilter(d, 't', {value: NAR, checked: true}, m);
  f = app.updateExplorerFilter(f, 't', {value: BT, checked: true}, m);
  assert.deepEqual(f.t, [BT, NAR].sort());
  assert.equal(app.updateExplorerFilter(Object.assign({}, d, {t: [NAR]}), 't', {value: NAR, checked: false}, m).t, null);
  assert.equal(app.updateExplorerFilter(d, 't', {value: '<b>', checked: true}, m).t, null);
  assert.equal(app.updateExplorerFilter(d, 'dec', {value: 'acc'}, m).dec, 'acc');
  assert.equal(app.updateExplorerFilter(d, 'dec', {value: 'bogus'}, m).dec, 'all');
  assert.equal(app.updateExplorerFilter(d, 'kb', {value: 'raw'}, m).kb, 'raw');
  assert.equal(app.updateExplorerFilter(d, 'rd', {value: 'only'}, m).rd, 'only');
  assert.equal(app.updateExplorerFilter(d, 'rd', {value: 'x'}, m).rd, 'all');
  assert.equal(app.updateExplorerFilter(d, 'nj', {checked: true}, m).nj, true);
  assert.equal(app.updateExplorerFilter(d, 'tw', {checked: true}, m).tw, true);
  assert.deepEqual(app.updateExplorerFilter(d, 'nope', {value: '1'}, m), d);
});

test('explorer (v1.7): the model reads every row; links, dirs, transcript paths and the re-scored flag are derived as the contract says', () => {
  const m = app.explorerModel(EXD);
  assert.equal(app.explorerModel(EXD), m, 'cached per document');
  assert.equal(app.explorerModel(null), null);
  const built = EX_ENTRIES.flatMap((e) => builtOf(e.doc).map((n) => [e, n]));
  assert.equal(m.tasks.length, built.length);
  assert.equal(m.tasks.length, builtOf(V).length + builtOf(F.bandit).length + 2 * builtOf(BATCH_DOCS[0]).length);
  assert.equal(m.trajectories.length, built.reduce((s, [, n]) => s + ((n.probe && n.probe.rollouts) || []).length, 0));
  for (const [i, [e, n]] of built.entries()) {
    const t = m.tasks[i];
    assert.deepEqual([t.node, t.taskId, t.title, t.decision, t.round, t.source, t.target, t.run.run],
                     [n.id, n.build.task_id, n.title, n.decision, n.round, e.source, e.doc.task_id, e.doc.run]);
    assert.equal(t.link, app.taskHash(e.doc.task_id, n.id, e.source === 'hengyuf' ? null : e.doc.run));
    const h = app.parseHash(t.link);
    assert.deepEqual([h.view, h.task, h.node, h.run], ['task', e.doc.task_id, n.id, e.source === 'hengyuf' ? null : e.doc.run]);
    assert.equal(t.dir, `${e.doc.build_root}/${n.id}`);
    assert.equal(t.traj.length, ((n.probe && n.probe.rollouts) || []).length);
    for (const r of t.traj) {
      assert.equal(r.transcript, n.score ? `${e.doc.build_root}/round-${n.round}/${n.id}/rollouts/${r.slot}.md` : '');
      assert.equal(r.patch, n.score ? `${e.doc.build_root}/round-${n.round}/${n.id}/rollouts/${r.slot}.patch` : '');
    }
  }
  assert.ok(m.tasks.some((t) => !t.scored && t.traj.length), 'a live probe: trajectories without transcripts');
  assert.deepEqual(m.sources.map((s) => [s.source, s.cls]), [['hengyuf', 's1'], [BSRC, 's2']]);
  assert.deepEqual(m.rounds, [0, 1, 2, 3]);
  assert.deepEqual(m.kinds, ['mask', 'evolve']);
  assert.deepEqual(m.targetIds, [BAN, FD, BT, NAR].sort());
  assert.ok(!m.counts.has(FD), 'a run without a tree adds no task');
  const c04 = m.tasks.find((t) => t.target === NAR && t.node === 'cand-c04');
  assert.deepEqual([c04.audit, c04.yes, c04.partial], ['strong', [0], [1]]);
  assert.ok(m.tasks.filter((t) => t.source === BSRC).every((t) => t.audit === 'not_audited' && t.yes === null));
  assert.ok(m.tasks.filter((t) => t.target === BAN).every((t) => t.audit === 'not_eligible'));
  // the re-scored task: metrics.json says 80 %, its rollout rows 2 of 5; nothing else of the fixtures differs
  const r = m.tasks.find((t) => t.target === NAR && t.node === RESCORED);
  assert.deepEqual([r.rescored, r.sRate, r.rowsRates.s, r.rowsRates.valid], [true, 0.8, 0.4, [5, 5]]);
  assert.deepEqual(m.tasks.filter((t) => t.rescored).map((t) => `${t.target}/${t.node}`), [`${NAR}/${RESCORED}`]);
  assert.equal(EXD.totals.hengyuf.rates_differ, 1);
  assert.match(app.exRescoredTitle(r), /metrics\.json: Nemotron 80% · Kimi clean \d+% · valid 5 \/ 5\nrollout rows: Nemotron 40% /);
  // the pass-rate columns: the whole percents the data holds, the same whatever the filters
  assert.deepEqual(m.rateColumns.s.map((c) => c.label), [...new Set(m.tasks.map((t) => app.exPct(t.sRate)).filter((v) => v !== null))]
    .sort((a, b) => a - b).map(String));
  assert.ok(m.rateColumns.s.every((c) => c.lo === c.hi && c.title === `${c.lo}%`));
});

test('explorer (v1.7): every task filter; a similarity bound leaves out the never-judged tasks unless "include never judged"', () => {
  const m = app.explorerModel(EXD);
  const ids = (r) => r.tasks.map((t) => t.i), want = (pred) => m.tasks.filter(pred).map((t) => t.i);
  const all = exRun({});
  assert.equal(all.tasks.length, m.tasks.length);
  assert.equal(all.trajectories.length, m.trajectories.length);
  assert.equal(all.trajActive, false);
  const cases = [
    [{round: [1]}, (t) => t.round === 1], [{round: [0, 3]}, (t) => t.round === 0 || t.round === 3], [{round: []}, () => false],
    [{src: [BSRC]}, (t) => t.source === BSRC], [{kind: ['mask']}, (t) => t.kind === 'mask'],
    [{dec: 'acc'}, (t) => t.accepted], [{dec: 'not'}, (t) => !t.accepted],
    [{smin: 6}, (t) => t.sim !== null && t.sim >= 6], [{smin: 6, smax: 7}, (t) => t.sim !== null && t.sim >= 6 && t.sim <= 7],
    [{smin: 6, nj: true}, (t) => t.sim === null || t.sim >= 6], [{smax: 5, nj: true}, (t) => t.sim === null || t.sim <= 5],
    [{smin: 0}, (t) => t.sim !== null],
    [{nmax: 0.2}, (t) => t.sRate !== null && t.sRate <= 0.2], [{nmin: 0.4, nmax: 0.6}, (t) => t.sRate !== null && t.sRate >= 0.4 && t.sRate <= 0.6],
    [{kmin: 0.6}, (t) => t.kClean !== null && t.kClean >= 0.6], [{kmin: 0.6, kb: 'raw'}, (t) => t.kRaw !== null && t.kRaw >= 0.6],
    [{audit: ['strong', 'partial']}, (t) => ['strong', 'partial'].includes(t.audit)], [{audit: ['not_audited']}, (t) => t.audit === 'not_audited'],
    [{t: [BT]}, (t) => t.target === BT], [{t: [BT, NAR]}, (t) => t.target === BT || t.target === NAR],
    [{rd: 'only'}, (t) => t.rescored], [{rd: 'none'}, (t) => !t.rescored],
    [{round: [1], dec: 'not', src: ['hengyuf']}, (t) => t.round === 1 && !t.accepted && t.source === 'hengyuf'],
  ];
  for (const [over, pred] of cases) {
    assert.deepEqual(ids(exRun(over)), want(pred), JSON.stringify(over));
    assert.ok(want(pred).length < m.tasks.length, `${JSON.stringify(over)} filters something out`);
  }
  // "include never judged" alone changes nothing (no bound); the never-judged count is over the other filters
  for (const nj of [false, true]) assert.equal(exRun({nj}).tasks.length, m.tasks.length);
  const never = m.tasks.filter((t) => t.sim === null).length;
  assert.ok(never > 0 && never < m.tasks.length);
  assert.deepEqual([exRun({}).neverJudged, exRun({smin: 6}).neverJudged, exRun({smin: 6, nj: true}).neverJudged], [never, never, never]);
  assert.equal(exRun({smin: 6, round: [1]}).neverJudged, m.tasks.filter((t) => t.sim === null && t.round === 1).length);
  assert.deepEqual([exRun({}).simBounded, exRun({smax: 9}).simBounded], [false, true]);
  assert.equal(exRun({smin: 6}).tasks.length + never, exRun({smin: 6, nj: true}).tasks.length);
  // the three ranges agree: with a bound set, a task without the value fails it
  assert.ok(exRun({nmin: 0}).tasks.every((t) => t.sRate !== null) && exRun({kmin: 0}).tasks.every((t) => t.kClean !== null));
  assert.equal(exRun({smin: 8, smax: 6}).tasks.length, 0);                // inverted: nothing passes
  // a rate bound applies to the whole percent shown: 1/3 reads 33 %, so "max 33" and "min 33" both keep it
  const third = clone(EXD), sCol = third.task_columns.indexOf('student_rate');
  third.tasks.forEach((row, i) => { if (row[sCol] !== null) row[sCol] = i % 2 ? 1 / 3 : 2 / 3; });
  const tm = app.explorerModel(third), thirds = (over) => app.filterExplorer(tm, Object.assign(app.explorerDefaults(), over)).tasks.length;
  const n13 = tm.tasks.filter((t) => t.sRate !== null && t.sRate < 0.5).length, n23 = tm.tasks.filter((t) => t.sRate > 0.5).length;
  assert.ok(n13 > 0 && n23 > 0);
  assert.deepEqual([thirds({nmax: 0.33}), thirds({nmin: 0.33, nmax: 0.33}), thirds({nmin: 0.34}), thirds({nmin: 0.67}), thirds({nmax: 0.66})],
                   [n13, n13, n23, n23, n13]);
  const odd = clone(EXD);
  odd.tasks.forEach((row) => { if (row[sCol] !== null) row[sCol] = 0.29; });
  const om = app.explorerModel(odd), scored = om.tasks.filter((t) => t.sRate !== null).length;
  for (const over of [{nmax: 0.29}, {nmin: 0.29}, {nmin: 0.29, nmax: 0.29}]) {    // 0.29 * 100 is 28.999…: still 29
    assert.equal(app.filterExplorer(om, Object.assign(app.explorerDefaults(), over)).tasks.length, scored, JSON.stringify(over));
  }
  // trajectories
  const tr = (over) => exRun(over).trajectories.map((x) => x.i), wantTr = (pred) => m.trajectories.filter(pred).map((x) => x.i);
  assert.deepEqual(tr({tm: ['sc14']}), wantTr((x) => x.model === 'sc14'));
  assert.deepEqual(tr({ts: ['pass']}), wantTr((x) => x.status === 'pass'));
  assert.deepEqual(tr({ts: ['infra']}), wantTr((x) => x.status === 'infra'));
  const clean = wantTr((x) => x.model === 'kimi' && x.status === 'pass' && x.clean === true);
  assert.deepEqual(tr({tm: ['kimi'], ts: ['pass'], tc: true}), clean);
  assert.ok(clean.length < wantTr((x) => x.model === 'kimi' && x.status === 'pass').length, 'a cheated kimi pass is dropped');
  assert.deepEqual(tr({tc: true}), wantTr((x) => !(x.model === 'kimi' && x.status === 'pass' && x.clean !== true)));
  assert.equal(exRun({tm: ['sc14']}).trajActive, true);
  assert.deepEqual(tr({round: [0], ts: ['fail']}), wantTr((x) => x.task.round === 0 && x.status === 'fail'));
  const loose = exRun({tm: ['sc14'], ts: ['pass']}), strict = exRun({tm: ['sc14'], ts: ['pass'], tw: true});
  assert.equal(loose.tasks.length, m.tasks.length);
  assert.deepEqual(ids(strict), want((t) => t.traj.some((x) => x.model === 'sc14' && x.status === 'pass')));
  assert.ok(strict.tasks.length < loose.tasks.length);
  assert.deepEqual(strict.trajectories, loose.trajectories);
  assert.equal(strict.matched.get(strict.tasks[0]), strict.tasks[0].traj.filter((x) => x.model === 'sc14' && x.status === 'pass').length);
  assert.deepEqual(app.filterExplorer(null, {}), {tasks: [], trajectories: [], matched: new Map(), trajActive: false, neverJudged: 0, simBounded: false});
});

test('explorer (v1.7): the filter bar says what the inputs do: never-judged count, inverted and rejected ranges, the task choice', () => {
  const m = app.explorerModel(EXD), never = m.tasks.filter((t) => t.sim === null).length;
  const st = (over, bad) => app.explorerFilterStatus(m, over, exRun(over), bad);
  assert.match(st({}).nj, new RegExp(`>${never} never judged, shown</span>`));
  assert.ok(st({smin: 6}).nj.includes(`>+${never} never judged</span>`));
  assert.ok(st({smin: 6, nj: true}).nj.includes(`>incl. ${never} never judged</span>`));
  assert.equal(st({}).s, '');
  assert.match(st({smin: 8, smax: 6}).s, /<span class="ex-warn" title="similarity: min 8 &gt; max 6: no task can pass">min 8 &gt; max 6: no task can pass<\/span>/);
  assert.match(st({nmin: 0.6, nmax: 0.2}).n, /min 60 % &gt; max 20 %/);
  assert.match(st({}, {kmax: '150 is outside 0–100 %'}).k, /max: 150 is outside 0–100 %/);
  assert.equal(st({t: [NAR]}).targets, '1 of 4');
  assert.equal(st({}).targets, 'all 4');
  assert.deepEqual(app.exInverted({smin: 8, smax: 6, kmin: 0.9, kmax: 0.5}), ['similarity: min 8 > max 6', 'Kimi pass rate: min 90 % > max 50 %']);
  // rendered: percent inputs (0-100), "include never judged" with its count, the slots, an invalid input as typed
  const bar = app.renderExplorerFilters(m, {smin: 6, nmax: 0.8}, {query: '', bad: {kmin: '150'}});
  // v1.8: text inputs (inputmode decimal), so that "abc" reaches the bar's own check instead of reading as empty
  assert.ok(bar.includes('<input type="text" inputmode="decimal" class="ex-num" data-exf="nmax" data-fkey="exf:nmax" value="80"'));
  assert.ok(bar.includes('<input type="text" inputmode="decimal" class="ex-num" data-exf="smin" data-fkey="exf:smin" value="6"'));
  assert.ok(bar.includes('data-exf="kmin" data-fkey="exf:kmin" value="150" placeholder="min" aria-label="Kimi pass rate minimum, %" autocomplete="off" spellcheck="false" aria-invalid="true">'));
  assert.ok(!bar.includes('type="number"'));
  assert.ok(bar.includes('<span class="ex-unit">%</span>'));
  assert.match(bar, /data-exf="nj" data-fkey="exf:nj"><span>include never judged<\/span><\/label><span id="ex-st-nj" class="ex-st" role="status"><span class="small"[^>]*>\+\d+ never judged<\/span><\/span>/);
  assert.match(bar, /<span id="ex-st-k" class="ex-st" role="status"><span class="ex-warn"[^>]*>min: 150 is outside 0–100 %<\/span><\/span>/);
  assert.ok(bar.includes('<span id="ex-st-targets" class="ex-dd-label">all 4</span>'));
  assert.match(bar, /<select data-exf="rd" data-fkey="exf:rd"[^>]*><option value="all" selected>all<\/option><option value="only">only re-scored<\/option>/);
});

test('explorer (v1.7): statistics: KPIs (in the sources\' scope), coverage, failure-pattern coverage, exact rate columns', () => {
  const m = app.explorerModel(EXD), f = app.explorerDefaults(), st = app.explorerStats(m, exRun({}), f);
  const pass = (model) => m.trajectories.filter((x) => x.model === model && x.status === 'pass').length;
  assert.deepEqual(st.kpi, {tasks: m.tasks.length, allTasks: m.tasks.length, trajectories: m.trajectories.length,
                            allTrajectories: m.trajectories.length, targets: 3, inScope: 3, converted: 3,
                            accepted: m.tasks.filter((t) => t.accepted).length, rescored: 1, scPass: pass('sc14'),
                            scTotal: m.trajectories.filter((x) => x.model === 'sc14').length, kimiPass: pass('kimi'),
                            kimiClean: m.trajectories.filter((x) => x.model === 'kimi' && x.status === 'pass' && x.clean === true).length,
                            kimiTotal: m.trajectories.filter((x) => x.model === 'kimi').length});
  // the batch alone: 1 of the 1 DeepSWE task it converted (not 1 of 3)
  const sb = app.explorerStats(m, exRun({src: [BSRC]}), {src: [BSRC]});
  assert.deepEqual([sb.kpi.targets, sb.kpi.inScope, sb.kpi.converted], [1, 1, 3]);
  assert.deepEqual(sb.coverage.map((c) => c.id), [BT]);
  const cov = new Map(st.coverage.map((c) => [c.id, c]));
  assert.deepEqual([...cov.keys()].sort(), [BAN, BT, NAR].sort());
  assert.deepEqual(cov.get(NAR).tasks, [builtOf(V).length, 0]);
  assert.deepEqual(cov.get(BT).tasks, [0, 2 * builtOf(BATCH_DOCS[0]).length]);
  assert.deepEqual(cov.get(BT).traj, [0, m.trajectories.filter((x) => x.task.target === BT).length]);
  const r1 = app.explorerStats(m, exRun({round: [3]}), {round: [3]});
  assert.deepEqual(r1.coverage.map((c) => c.id).sort(), [BAN, BT, NAR].sort());
  assert.deepEqual(r1.coverage.find((c) => c.id === BAN).tasks, [0, 0]);
  assert.deepEqual(app.explorerStats(m, exRun({t: [NAR]}), {t: [NAR]}).coverage.map((c) => c.id), [NAR]);
  const audited = m.tasks.filter((t) => VERDICT.includes(t.audit));
  assert.ok(audited.length >= 5);
  assert.equal(st.audited, audited.length);
  assert.deepEqual(st.patterns.map((g) => [g.id, g.audited]), [[NAR, audited.length]]);
  assert.deepEqual(st.patterns[0].patterns.map((p) => [p.i, p.yes, p.partial]), [[0, audited.length, 0], [1, 0, audited.length], [2, 0, 0]]);
  assert.deepEqual(st.excluded, {notAudited: m.tasks.filter((t) => t.source === 'hengyuf' && t.audit === 'not_audited').length,
                                 notAuditedBatch: m.tasks.filter((t) => t.source === BSRC).length,
                                 notEligible: m.tasks.filter((t) => t.audit === 'not_eligible').length});
  const sum = (a) => a.reduce((x, y) => x + y, 0);
  assert.equal(sum(st.dist.sim) + st.dist.simNone, m.tasks.length);
  assert.equal(sum(st.dist.sRate) + st.dist.sNone, m.tasks.length);
  assert.equal(sum(st.dist.kRate) + st.dist.kNone, m.tasks.length);
  assert.equal(sum(st.dist.audits), m.tasks.length);
  assert.deepEqual(st.dist.rounds.map((x) => [x.round, x.n]), [0, 1, 2, 3].map((k) => [k, m.tasks.filter((t) => t.round === k).length]));
  // exact columns: each column counts exactly the tasks of its whole percent
  st.dist.sCols.forEach((c, i) => assert.equal(st.dist.sRate[i], m.tasks.filter((t) => app.exPct(t.sRate) === c.lo).length, c.label));
  assert.equal(app.exRateColumn(st.dist.sCols, 0.2), st.dist.sCols.findIndex((c) => c.lo === 20));
  assert.equal(app.exRateColumn(st.dist.sCols, null), null);
  // more than 16 distinct values: closed 10-point bins ("20–29"), never ".2–.4"
  const many = clone(EXD);
  many.tasks.forEach((row, i) => { row[many.task_columns.indexOf('student_rate')] = (i % 101) / 100; });
  const cols = app.explorerModel(many).rateColumns.s;
  assert.deepEqual(cols.map((c) => c.label), ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70–79', '80–89', '90–99', '100']);
  assert.equal(app.exRateColumn(cols, 0.2), 2);
  assert.equal(app.exRateColumn(cols, 0.999), 10);                       // 99.9 % rounds to 100
});

test('explorer (v1.7): the view: scope, filters, KPI tiles, coverage, pattern coverage, distributions, both tables, in order', () => {
  const m = app.explorerModel(EXD);
  const html = app.renderExplorer(EXD, EXI, {now: NOW, ui: {}, filters: app.explorerDefaults(), query: ''});
  assert.ok(!html.includes('render-error'));
  assert.doesNotMatch(html, /undefined|NaN|\[object Object\]/);
  const order = ['<div id="ex-head"><div class="ex-head"><h1 class="ex-h1">Explorer</h1>', 'data-key="ex:scope"', 'aria-label="Explorer filters"',
                 '<div class="tile-label">Synthetic tasks</div>', 'data-exaction="skip-tables"', '<h2>Coverage per DeepSWE task</h2>',
                 '<h2>Failure-pattern coverage</h2>', '<h2>Distributions</h2>', '<h2>Synthetic tasks</h2>', '<h2>Trajectories</h2>',
                 '<div id="ex-tip" class="ex-tip" role="tooltip" hidden></div>'];
  const at = order.map((s) => html.indexOf(s));
  assert.ok(at.every((p, i) => p >= 0 && (i === 0 || p > at[i - 1])), `section order ${at}`);
  // the scope note: hengyuf's latest runs (no older one left out here), the batch's runs behind its tasks, abandoned roots
  const scope = app.explorerScope(m, EXI);
  assert.equal(scope.length, 3);
  assert.match(scope[0], /^hengyuf: every built task of each official DeepSWE target's latest run \(its data\/task\/<task>\.json\), 3 runs, no older run left out\.$/);
  assert.match(scope[1], new RegExp(`^Testy: only the 2 runs behind the ${EXI.batches[0].tasks} tasks of Testy's batch \\(outputs/tasks\\.jsonl\\); the other runs of that tree are never read`));
  assert.match(scope[2], /^Abandoned build roots \(runs\/<run>\/abandoned-<repo>, set aside by main\) are never read, so never rows; "abandoned" here is a node's decision/);
  const older = clone(EXI);
  older.runs.push(Object.assign(clone(older.runs[0]), {run: 'official0-older'}));
  assert.match(app.explorerScope(m, older)[0], /3 runs; 1 older official run of the same targets left out\./);
  // KPI tiles: covered of the converted DeepSWE tasks in scope, of all of index.json's
  assert.ok(html.includes(`<div class="tile-value">${app.fmtInt(m.tasks.length)}<span class="tile-den"> / ${app.fmtInt(m.tasks.length)}</span>`));
  assert.ok(html.includes(`<div class="tile-label">DeepSWE tasks covered</div><div class="tile-value">3<span class="tile-den"> / 3</span></div>` +
                          `<div class="tile-sub">with ≥ 1 filtered task, of the 3 converted by the chosen sources (of ${EXI.tasks.length} DeepSWE tasks)</div>`));
  const batch = app.renderExplorerResults(EXD, EXI, {filters: {src: [BSRC]}});
  assert.ok(batch.includes('<div class="tile-value">1<span class="tile-den"> / 1</span></div><div class="tile-sub">with ≥ 1 filtered task, of the 1 converted'));
  assert.ok(batch.includes(`1 of 1 converted DeepSWE tasks covered (of ${EXI.tasks.length} in all)`));
  for (const label of ['Trajectories', 'Accepted', 'Nemotron passes', 'Kimi passes']) assert.ok(html.includes(`<div class="tile-label">${label}</div>`), label);
  // the filter bar
  for (const v of ['hengyuf', BSRC]) assert.ok(html.includes(`data-exf="src" value="${v}" data-fkey="exf:src:${v}" checked`), v);
  for (const k of [0, 1, 2, 3]) assert.ok(html.includes(`data-exf="round" value="${k}"`));
  for (const a of app.EX_AUDITS) assert.ok(html.includes(`data-exf="audit" value="${a}"`), a);
  for (const k of ['smin', 'smax', 'nmin', 'nmax', 'kmin', 'kmax', 'dec', 'kb', 'rd', 'nj', 'tc', 'tw', 'tq']) assert.ok(html.includes(`data-exf="${k}"`), k);
  assert.ok(!html.includes('data-exf="sun"'));
  assert.equal(count(html, 'data-exf="t" value="'), m.targetIds.length);
  assert.ok(html.includes('<details class="ex-dd" data-key="ex:targets"><summary><span id="ex-st-targets" class="ex-dd-label">all 4</span></summary>'));
  // coverage: a row per DeepSWE task in scope, the legend (slot 1, slot 2), one tab stop for the panel
  const coverage = html.slice(html.indexOf('<h2>Coverage per DeepSWE task</h2>'), html.indexOf('<h2>Failure-pattern coverage</h2>'));
  assert.equal(count(coverage, '<div class="hbar" role="listitem"'), 3);
  assert.equal(count(coverage, 'tabindex="0"'), 1);
  assert.ok(coverage.includes('<div data-roving="coverage">'));
  assert.ok(coverage.includes('<span class="lg"><span class="swatch s1" aria-hidden="true"></span>hengyuf</span>' +
                              '<span class="lg"><span class="swatch s2" aria-hidden="true"></span>Testy</span>'));
  assert.ok(coverage.includes(`3 of 3 converted DeepSWE tasks covered (of ${EXI.tasks.length} in all)`));
  // pattern coverage
  const patterns = html.slice(html.indexOf('<h2>Failure-pattern coverage</h2>'), html.indexOf('<h2>Distributions</h2>'));
  assert.equal(count(patterns, '<div class="hbar" role="listitem"'), 3);
  assert.equal(count(patterns, '<div class="chart-legend'), 1);
  assert.match(patterns, /swatch seg-yes[^]*reproduced: yes[^]*swatch seg-partial[^]*partial/);
  assert.match(patterns, /of the \d+ filtered tasks are not counted here<\/b> <span class="muted">\(\d+ of a batch \(never audited\)/);
  assert.equal(count(patterns, 'tabindex="0"'), 1);
  // distributions: five histograms in slot 3, exact rate columns, list items, one tab stop for the panel
  const dists = html.slice(html.indexOf('<h2>Distributions</h2>'), html.indexOf('<h2>Synthetic tasks</h2>'));
  assert.equal(count(dists, '<figure class="chart ex-chart ex-hist">'), 5);
  assert.equal(count(dists, 'chart-legend'), 0);
  assert.equal(count(dists, '<div class="vbar" role="listitem"'), 11 + m.rateColumns.s.length + m.rateColumns.clean.length + 6 + 4);
  assert.equal(count(dists, 'tabindex="0"'), 1);
  assert.equal(count(dists, 'class="vbar-col s1"'), 0);
  assert.ok(count(dists, 'class="vbar-col s3"') > 10);
  assert.match(dists, /Nemotron \(sc14\) pass rate, %/);
  assert.ok(!/0–\.2|\.2–\.4/.test(dists), 'no half-open bins that read closed');
  assert.match(dists, /<span title="audit: not audited">not audited<\/span>/);
  // the task table: 50 rows a page, links, the pager, the re-scored column and marker
  const tasks = html.slice(html.indexOf('<h2>Synthetic tasks</h2>'), html.indexOf('<h2>Trajectories</h2>'));
  assert.equal(count(tasks, '<tr class="ex-row" data-exlink="'), app.EX_PAGE);
  assert.ok(tasks.includes(`page 1 of ${Math.ceil(m.tasks.length / app.EX_PAGE)} · tasks 1–50 of ${app.fmtInt(m.tasks.length)}`));
  assert.ok(tasks.includes('<div id="ex-tasks" class="tbl-wrap" data-scroll="ex:tasks">'));
  assert.match(tasks, /data-exsort="rescored"[^>]*>re-scored<\/button>/);
  assert.match(tasks, /\(1 re-scored: \* on a rate that differs\)/);
  const rescored = app.renderExplorerTaskTable(m, exRun({rd: 'only'}), {rd: 'only'});
  assert.match(rescored, /<td class="num">80%<span class="ex-mark" title="Re-scored: these rates come from metrics\.json[^"]*" aria-label="re-scored">\*<\/span><\/td>/);
  assert.match(rescored, /<span class="badge tone-pending badge-xs" title="Re-scored: [^"]*rollout rows: Nemotron 40%[^"]*">/);
  assert.match(rescored, /<div class="ex-pager muted small">1 task<\/div>/);
  assert.equal(count(rescored, 'class="ex-mark"'), 1, 'its Kimi clean rate agrees with the rows: no marker on it');
  const raw = app.renderExplorerTaskTable(m, exRun({rd: 'only', kb: 'raw'}), {rd: 'only', kb: 'raw'});
  assert.equal(count(raw, 'class="ex-mark"'), 1, 'the raw Kimi rate is the rows\' own: no marker on it');
  const both = clone(VR);                                              // the Kimi clean rate rewritten too: both marked
  nodeOf(both, RESCORED).score.teacher_clean_rate = 0.2;
  const bm = app.explorerModel(explorerOf([{source: 'hengyuf', detail: `task/${NAR}.json`, doc: both}], EXD.generated_at, EX_SOURCES.slice(0, 1)));
  const bothTable = app.renderExplorerTaskTable(bm, app.filterExplorer(bm, {rd: 'only'}), {rd: 'only'});
  assert.equal(count(bothTable, 'class="ex-mark"'), 2);
  const validOnly = clone(V);                                          // valid counts alone: the badge, no *
  nodeOf(validOnly, RESCORED).score.valid = [4, 5];
  const vm = app.explorerModel(explorerOf([{source: 'hengyuf', detail: `task/${NAR}.json`, doc: validOnly}], EXD.generated_at, EX_SOURCES.slice(0, 1)));
  const vTable = app.renderExplorerTaskTable(vm, app.filterExplorer(vm, {rd: 'only'}), {rd: 'only'});
  assert.equal(app.filterExplorer(vm, {rd: 'only'}).tasks.length, 1);
  assert.equal(count(vTable, 'class="ex-mark"'), 0);
  assert.match(vTable, /<span class="badge tone-pending badge-xs" title="Re-scored: [^"]*valid 4 \/ 5\nrollout rows: [^"]*graded 5 \/ 5">/);
  const traj = html.slice(html.indexOf('<h2>Trajectories</h2>'));
  assert.ok(traj.includes(`<code>${app.esc(m.trajectories[0].transcript)}</code>`));
  assert.ok(traj.includes(`<code>${app.esc(m.trajectories[0].patch)}</code>`));
  assert.equal(count(traj, 'data-exlink="'), app.EX_PAGE);
  // nothing matches, an inverted range, loading, an unavailable file
  const none = app.renderExplorer(EXD, EXI, {now: NOW, filters: Object.assign(app.explorerDefaults(), {round: []})});
  assert.ok(!none.includes('render-error'));
  assert.match(none, /No synthetic task matches the filters\./);
  assert.match(none, /No trajectory matches the filters\./);
  assert.match(none, /0 of 3 converted DeepSWE tasks covered/);
  const inverted = app.renderExplorerResults(EXD, EXI, {filters: {smin: 8, smax: 6}});
  assert.ok(inverted.includes('<div class="alert alert-warn ex-alert" role="status"><b>A range is inverted:</b> similarity: min 8 &gt; max 6; no task can pass it.</div>'));
  assert.match(app.renderExplorer(null, EXI, {}), /Loading data\/explorer\.json…/);
  assert.match(app.renderExplorer(null, EXI, {error: 'HTTP 404'}), /data\/explorer\.json not available: HTTP 404/);
});

test('explorer (v1.7): counts read as words: "1 task", "1 trajectory", singular tooltips and labels', () => {
  const m = app.explorerModel(EXD);
  let pick = null;                                                   // a filter that leaves exactly one task
  for (const t of m.targetIds) for (const round of m.rounds) for (const kind of m.kinds) for (const dec of ['acc', 'not']) {
    if (!pick && exRun({t: [t], round: [round], kind: [kind], dec}).tasks.length === 1) pick = {t: [t], round: [round], kind: [kind], dec};
  }
  assert.ok(pick, 'some filter leaves one task');
  const one = exRun(pick);
  const html = app.renderExplorerResults(EXD, EXI, {filters: pick});
  assert.match(html, /<span class="muted small">1 task · click a row/);
  assert.match(html, /over the 1 filtered task</);
  assert.match(html, /<div class="ex-pager muted small">1 task<\/div>/);
  const r = pick.round[0];
  assert.ok(html.includes(`aria-label="round ${r}${r === 0 ? ' (mask tasks)' : ''}: 1 task"`));
  assert.doesNotMatch(html, /\b1 tasks\b|\b1 trajectories\b|\b1 runs\b/);
  const pager = app.renderExplorerTrajTable(m, {trajectories: one.tasks[0].traj.slice(0, 1)}, {});
  assert.match(pager, /<div class="ex-pager muted small">1 trajectory<\/div>/);
  assert.equal(app.renderExplorerTargetList(m, {}, 'zzz'), '<p class="muted small">No DeepSWE task matches “zzz”.</p>');
  assert.match(app.renderExplorerTargetList(m, {}, 'fd'), />0 tasks<\/span><\/label>$/);      // a run without a tree yet
  assert.match(app.renderExplorerTargetList(m, {}, 'narwhals'), new RegExp(`>${builtOf(V).length} tasks</span>`));
  const single = clone(EXD);
  single.tasks = single.tasks.slice(0, 1);
  single.trajectories = single.trajectories.filter((r) => r[0] === 0).slice(0, 1);
  assert.match(app.renderExplorer(single, EXI, {}), /1 task, 1 trajectory from 5 runs/);
});

test('explorer (v1.7): marks follow the chart rules (thin, rounded data end, gap, entity colors, slot 3, tooltips, roles)', () => {
  const css = fs.readFileSync(path.join(STATIC, 'style.css'), 'utf8');
  assert.match(css, /\.hbar-seg \{[^}]*height: 14px/);
  assert.match(css, /\.hbar-seg\.end \{ border-radius: 0 4px 4px 0; \}/);
  assert.match(css, /\.vbar-col \{[^}]*width: min\(24px, 70%\)[^}]*border-radius: 4px 4px 0 0/);
  assert.match(css, /--ord-yes: #2a78d6;\s*--ord-partial: #86b6ef;/);                  // light: blue 450 / 250
  assert.match(css, /--ord-yes: #86b6ef;[^\n]*\n\s*--ord-partial: #256abf;/);       // dark: validated on #1a1a19
  assert.match(css, /--s1: #2a78d6;\s*--s2: #eb6834;/);
  assert.match(css, /--s3: #1baf7a;/);                                              // slot 3, light
  assert.match(css, /--s3: #199e70;/);                                              // slot 3, dark
  assert.match(css, /\.vbar-col\.s3, \.swatch\.s3, \.ex-tip-key\.s3 \{ background: var\(--s3\); \}/);
  assert.match(css, /\.vbar-x span \{[^}]*overflow-wrap: break-word/);
  const bars = app.renderHBars({key: 'k', title: 'T', series: [{label: 'a', cls: 's1'}, {label: 'b', cls: 's2'}], unit: ['task', 'tasks'],
                                rows: [{label: 'x', values: [3, 1]}, {label: 'y', values: [0, 1]}, {label: 'z', values: [0, 0]}]});
  assert.ok(bars.includes('<span class="hbar-seg s1" style="width:75.000%"></span><span class="hbar-seg s2 end" ' +
                          'style="width:calc(25.000% - 2px);margin-left:2px"></span><span class="hbar-val">4</span>'));
  assert.match(bars, /<span class="hbar-plot" style="--n:\d+"><span class="hbar-val">0<\/span>/);          // no mark, the value
  assert.equal(count(bars, 'chart-legend'), 1);
  assert.match(bars, /<div class="hbar" role="listitem" data-mark="k:0" data-fkey="mark:k:0" tabindex="0" data-tip="[^"]*" aria-label="x: 3 a, 1 b; 4 tasks in all">/);
  assert.match(bars, /data-mark="k:1" data-fkey="mark:k:1" tabindex="-1"[^>]*aria-label="y: 0 a, 1 b; 1 task in all"/);
  const tip = JSON.parse(unesc(bars.match(/data-tip="([^"]*)"/)[1]));
  assert.deepEqual(tip, {t: 'x', r: [['s1', '3', 'a'], ['s2', '1', 'b'], ['', '4', 'total']]});
  assert.ok(app.renderExplorerTip(tip).startsWith('<div class="ex-tip-title">x</div><div class="ex-tip-row"><span class="ex-tip-key s1" ' +
                                                   'aria-hidden="true"></span><b>3</b> <span>a</span></div>'));
  assert.equal(app.renderExplorerTip({t: 'q', r: [['evil" onmouseover="x', '1', 'y']]}).includes('evil'), false);  // classes: code only
  assert.equal(app.renderExplorerTip('{not json'), '');
  const later = app.renderHBars({key: 'k', title: 'T', series: [{label: 'a', cls: 's1'}], rows: [{label: 'x', values: [3]}], first: false});
  assert.equal(count(later, 'tabindex="0"'), 0);                         // not its panel's first chart
  assert.equal(count(later, 'chart-legend'), 0);                         // one series: the title names it
  assert.ok(later.includes('<details class="fold fold-inline chart-table" data-key="ex:chart:k">'));   // a table twin
  const cols = app.renderVBars({key: 'v', title: 'V', bins: [{label: '0', n: 0}, {label: '1', n: 1}, {label: '2', n: 5}]});
  assert.ok(cols.includes('<div class="vbars" role="list" aria-label="V"'));
  assert.match(cols, /<div class="vbar" role="listitem" data-mark="v:0" data-fkey="mark:v:0" tabindex="0" data-tip="[^"]*" aria-label="0: 0 tasks"><\/div>/);
  assert.match(cols, /aria-label="1: 1 task"><span class="vbar-col s3" style="height:20\.000%"><\/span>/);
  assert.ok(cols.includes('<span class="vbar-col s3" style="height:100.000%"></span><span class="vbar-val" style="bottom:100.000%">5</span>'));
  assert.deepEqual(JSON.parse(unesc(cols.match(/data-tip="([^"]*)"/)[1])), {t: '0', r: [['s3', '0', 'tasks']]});
  // color follows the source, never its rank: with only the batch shown its bars stay slot 2
  const html = app.renderExplorerResults(EXD, EXI, {filters: {src: [BSRC]}});
  const coverage = html.slice(html.indexOf('<h2>Coverage per DeepSWE task</h2>'), html.indexOf('<h2>Failure-pattern coverage</h2>'));
  assert.ok(coverage.includes('hbar-seg s2 end') && !coverage.includes('hbar-seg s1'));
  // the roving tab stop: +1 / -1 clamped, Home / End
  assert.deepEqual([app.exRovingIndex(0, 5, 1), app.exRovingIndex(4, 5, 1), app.exRovingIndex(0, 5, -1), app.exRovingIndex(3, 5, -Infinity),
                    app.exRovingIndex(1, 5, Infinity), app.exRovingIndex(-1, 5, 1), app.exRovingIndex(0, 0, 1)], [1, 4, 0, 0, 4, 0, -1]);
});

test('explorer (v1.7): sorting (nulls last, ties in row order), pages, the trajectory toggle of the coverage chart', () => {
  const m = app.explorerModel(EXD);
  const sims = app.exSortedTasks(exRun({}), {sort: 'sim', desc: true}).map((t) => t.sim);
  const scored = sims.filter((v) => v !== null);
  assert.deepEqual(scored, scored.slice().sort((a, b) => b - a));
  assert.equal(sims.indexOf(null), scored.length, 'nulls last');
  const asc = app.exSortedTasks(exRun({}), {sort: 'sim', desc: false}).map((t) => t.sim).filter((v) => v !== null);
  assert.deepEqual(asc, scored.slice().reverse());
  const byTarget = app.exSortedTasks(exRun({}), {sort: 'target'});
  assert.deepEqual(byTarget.map((t) => t.target), m.tasks.map((t) => t.target).sort((a, b) => a.localeCompare(b)));
  for (let i = 1; i < byTarget.length; i += 1) if (byTarget[i].target === byTarget[i - 1].target) assert.ok(byTarget[i].i > byTarget[i - 1].i);
  assert.deepEqual(app.exSortedTasks(exRun({}), {sort: null, desc: true}).map((t) => t.i), m.tasks.map((t) => t.i).reverse());
  assert.equal(app.exSortedTasks(exRun({}), {sort: 'rescored', desc: true})[0].node, RESCORED);
  const page2 = app.renderExplorerTaskTable(m, exRun({}), {p: 2, sort: 'srate', desc: true});
  assert.match(page2, /page 2 of \d+ · tasks 51–100 of /);
  assert.ok(page2.includes('<th class="num" scope="col" aria-sort="descending"><button type="button" class="ex-sort" data-exsort="srate" data-fkey="exs:srate"'));
  assert.match(page2, /data-exsort="srate"[^>]*>Nemotron ↓<\/button>/);
  assert.match(app.renderExplorerTaskTable(m, exRun({}), {p: 999}), /page (\d+) of \1 /);                 // clamped to the last
  assert.deepEqual(app.exPage([1, 2, 3], 1), {n: 3, page: 1, pages: 1, from: 1, to: 3, rows: [1, 2, 3]});
  const traj = app.renderExplorerResults(EXD, EXI, {filters: {cov: 'traj'}});
  assert.match(traj, /Trajectories per DeepSWE task/);
  assert.match(traj, /aria-pressed="true">Trajectories<\/button>/);
});

test('explorer (v1.7): exports: a BOM, F2P / P2P as passed and total, formula-safe cells, the re-scored columns', () => {
  const m = app.explorerModel(EXD), f = {round: [1], sort: 'sim', desc: true};
  const csv = app.explorerExport(EXD, f, 'tasks.csv');
  assert.equal(csv.type, 'text/csv;charset=utf-8');
  assert.match(csv.name, /^explorer-tasks-\d{8}-\d{4}\.csv$/);
  assert.equal(csv.text.charCodeAt(0), 0xfeff, 'a UTF-8 byte-order mark first');
  const lines = csv.text.slice(1).split('\r\n');
  assert.equal(lines[lines.length - 1], '');
  assert.equal(lines.length - 2, exRun(f).tasks.length);
  assert.match(lines[0], /^source,run,deepswe_task,node,task_id,round,branch,kind,title,decision,accepted,withdrawn,in_batch,similarity,student_rate,teacher_clean_rate,teacher_rate,valid_sc14,valid_kimi,rates_differ,rollouts_student_rate,rollouts_teacher_clean_rate,audit,reproduced_yes,reproduced_partial,f2p_tests,p2p_tests,matching_trajectories,trajectories,link,cand_dir,rollouts_dir$/);
  const first = app.exSortedTasks(exRun(f), f)[0];
  assert.ok(lines[1].startsWith(`${first.source},${first.run.run},${first.target},${first.node},${first.taskId},1,`));
  assert.ok(lines[1].includes(first.link) && lines[1].includes(first.dir));
  const rescored = app.explorerExport(EXD, {rd: 'only'}, 'tasks.csv').text.slice(1).split('\r\n')[1].split(',');
  assert.deepEqual([rescored[19], rescored[20], rescored[14]], ['true', '0.4', '0.8']);   // flagged; rows 0.4; metrics 0.8
  // the formula guard: = + - @ (also full-width, also after leading blanks), and tab / CR / LF first
  assert.equal(app.csvCell('=HYPERLINK("x")'), '"\'=HYPERLINK(""x"")"');
  assert.equal(app.csvCell('-1+2'), "'-1+2");
  assert.equal(app.csvCell('  =1+1'), "'  =1+1");
  assert.equal(app.csvCell('＝1+1'), "'＝1+1");
  assert.equal(app.csvCell('＋1'), "'＋1");
  assert.equal(app.csvCell('\n=1'), '"\'\n=1"');
  assert.equal(app.csvCell('\tx'), "'\tx");
  assert.equal(app.csvCell('a = b'), 'a = b');
  assert.equal(app.csvCell('a,b'), '"a,b"');
  assert.equal(app.csvCell('line\nbreak'), '"line\nbreak"');
  assert.deepEqual([app.csvCell(null), app.csvCell(undefined), app.csvCell(true), app.csvCell(0.25), app.csvCell(NaN)], ['', '', 'true', '0.25', '']);
  // trajectories: F2P / P2P as numbers, never "7/8" (a date to a spreadsheet)
  const tcsv = app.explorerExport(EXD, {ts: ['fail']}, 'trajectories.csv').text.slice(1).split('\r\n');
  assert.match(tcsv[0], /,reward,f2p_passed,f2p_total,p2p_passed,p2p_total,tool_calls,/);
  assert.doesNotMatch(tcsv.slice(1).join('\n'), /,\d+\/\d+,/);
  const json = JSON.parse(app.explorerExport(EXD, {ts: ['infra']}, 'trajectories.json').text);
  assert.equal(json.count, exRun({ts: ['infra']}).trajectories.length);
  assert.equal(json.view, '#explorer&ts=infra');
  assert.equal(json.explorer_generated_at, EXD.generated_at);
  assert.deepEqual(Object.keys(json.trajectories[0]), ['source', 'run', 'deepswe_task', 'node', 'task_id', 'slot', 'model', 'status',
    'reward', 'f2p_passed', 'f2p_total', 'p2p_passed', 'p2p_total', 'tool_calls', 'patch_chars', 'agent_error', 'verifier_error', 'clean',
    'transcript', 'patch', 'link']);
  const fail = JSON.parse(app.explorerExport(EXD, {ts: ['fail']}, 'trajectories.json').text).trajectories[0];
  const src = m.trajectories.find((x) => x.status === 'fail');
  assert.deepEqual([fail.f2p_passed, fail.f2p_total], src.f2p.split('/').map(Number));
  assert.ok(json.trajectories.every((x) => x.status === 'infra'));
  assert.equal(app.explorerExport(EXD, {}, '../etc/passwd'), null);
  assert.equal(app.explorerExport(null, {}, 'tasks.csv'), null);
  const tj = JSON.parse(app.explorerExport(EXD, {audit: ['strong']}, 'tasks.json').text);
  assert.deepEqual(tj.tasks.map((t) => t.reproduced_yes), tj.tasks.map(() => `P1 · ${V.failure_analysis.patterns[0].name}`));
});

test('escaping (v1.7): markup in titles, ids, run names, pattern names, labels and paths never reaches the markup', () => {
  const d = clone(EXD);
  for (const row of d.tasks) row[d.task_columns.indexOf('title')] = PAYLOAD;
  for (const r of d.runs) Object.assign(r, {run: PAYLOAD, build_root: PAYLOAD, deepswe_task: PAYLOAD});
  for (const t of d.targets) for (const p of t.patterns) Object.assign(p, {name: PAYLOAD, category: PAYLOAD});
  for (const s of d.sources) Object.assign(s, {name: PAYLOAD, label: PAYLOAD});
  for (const row of d.trajectories) row[d.trajectory_columns.indexOf('slot')] = PAYLOAD;
  const html = app.renderExplorer(d, EXI, {now: NOW, ui: {}, filters: {}, query: PAYLOAD, bad: {nmin: PAYLOAD}});
  assertEscaped(html, 'explorer');
  assertEscaped(app.renderExplorerResults(d, EXI, {filters: {cov: 'traj', p: 2, tp: 2, rd: 'only'}}), 'explorer results');
  assertEscaped(app.renderExplorerTargetList(app.explorerModel(d), {}, PAYLOAD), 'target list');
  assertEscaped(app.renderExplorerTip({t: PAYLOAD, r: [['s1', PAYLOAD, PAYLOAD]]}), 'tip');
  const st = app.explorerFilterStatus(app.explorerModel(d), {}, null, {smin: PAYLOAD, kmax: PAYLOAD});
  for (const v of Object.values(st)) assertEscaped(v, 'status');
  const ix = clone(EXI);
  ix.batches[0].name = PAYLOAD;
  assertEscaped(app.renderExplorerHead(app.explorerModel(d), ix, {error: PAYLOAD}), 'head');
  for (const w of ['tasks.json', 'trajectories.json']) assert.doesNotThrow(() => JSON.parse(app.explorerExport(d, {}, w).text));
});

test('explorer (v1.7): sparse and broken documents render without exceptions or fallbacks', () => {
  for (const doc of [{}, {tasks: 'x'}, {tasks: [[null]], task_columns: ['run']}, {runs: [null, 1], tasks: [[0]], task_columns: ['run']},
                     nullAll(clone(EXD))]) {
    const html = app.renderExplorer(doc, EXI, {now: NOW, filters: {}});
    assert.equal(typeof html, 'string');
    assert.ok(!html.includes('render-error'), JSON.stringify(doc).slice(0, 80));
  }
  assert.equal(typeof app.renderExplorer(EXD, null, {}), 'string');
  const rnd = prng(17);
  for (let i = 0; i < 12; i += 1) {
    const doc = mutate(clone(EXD), rnd, 0);
    for (const filters of [{}, {round: [1], tm: ['kimi'], tw: true, sort: 'krate', kb: 'raw'}, {cov: 'traj', p: 3, smin: 5, rd: 'only'}]) {
      const html = app.renderExplorer(doc, EXI, {now: NOW, filters});
      assert.ok(!html.includes('render-error') && !/\[object Object\]/.test(html), `mutation ${i}`);
    }
  }
});

test('wiring (v1.7): the Explorer tab: its file, filters in the hash, results and status re-rendered alone, rows, exports', async () => {
  const f = exEnv({hash: '#explorer&round=1', files: exFiles()});
  const blobs = [], clicks = [];
  f.win.Blob = function Blob(parts, opts) { this.parts = parts; this.type = opts.type; blobs.push(this); };
  f.win.URL = {createObjectURL: (b) => `blob:${blobs.indexOf(b)}`, revokeObjectURL: () => {}};
  f.doc.createElement = (tag) => ({tag, click() { clicks.push({href: this.href, download: this.download}); }, remove() {}});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), results = f.el('ex-results');
  assert.ok(f.calls.some((c) => c.url === 'data/explorer.json'));
  assert.ok(!f.calls.some((c) => c.url.startsWith('data/task/')), 'the Explorer needs no task file');
  assert.match(main.innerHTML, /<h2>Coverage per DeepSWE task<\/h2>/);
  assert.deepEqual(a.state.ex.f.round, [1]);
  assert.ok(f.el('tab-explorer').classList.contains('active') && !f.el('tab-task').classList.contains('active'));
  assert.equal(f.el('tab-explorer').getAttribute('href'), '#explorer&round=1');
  assert.equal(f.doc.title, 'Explorer · swe-task official synthesis');
  assert.equal(f.el('drawer').hidden, true);
  const m = app.explorerModel(EXD), round1 = m.tasks.filter((t) => t.round === 1).length;
  assert.ok(main.innerHTML.includes(`${app.fmtInt(round1)}<span class="tile-den"> / ${app.fmtInt(m.tasks.length)}`));
  // a chip: the hash records it (replaceState, no history entry); only the results and the status slots re-render
  const mainWrites = main.writes;
  const control = (exf, props) => Object.assign(new FakeElement(f.doc, null, 'INPUT', main), props, {dataset: {exf}});
  f.doc.dispatch('change', {target: control('round', {value: '2', checked: true})});
  assert.equal(f.win.location.hash, '#explorer&round=1,2');
  assert.equal(main.writes, mainWrites, 'the filter bar stays (focus, caret)');
  assert.equal(results.writes, 1);
  const r12 = m.tasks.filter((t) => t.round === 1 || t.round === 2);
  assert.ok(results.innerHTML.includes(`${app.fmtInt(r12.length)}<span class="tile-den"> / ${app.fmtInt(m.tasks.length)}`));
  assert.ok(f.el('ex-st-nj').innerHTML.includes(`${r12.filter((t) => t.sim === null).length} never judged, shown`));
  // a similarity bound as typed (input, then its change event: applied once): the never-judged ones leave
  f.doc.dispatch('input', {target: control('smin', {value: '6'})});
  f.doc.dispatch('change', {target: control('smin', {value: '6'})});
  assert.equal(f.win.location.hash, '#explorer&round=1,2&smin=6');
  assert.equal(results.writes, 2);
  assert.ok(f.el('ex-st-nj').innerHTML.includes(`+${r12.filter((t) => t.sim === null).length} never judged`));
  const judged = r12.filter((t) => t.sim !== null && t.sim >= 6).length;
  assert.ok(results.innerHTML.includes(`${app.fmtInt(judged)}<span class="tile-den"> / `));
  f.doc.dispatch('change', {target: control('nj', {checked: true})});
  assert.equal(f.win.location.hash, '#explorer&round=1,2&smin=6&nj=1');
  // a pass rate in percent; out of range: rejected visibly, its bound left empty
  const nmax = control('nmax', {value: '80'});
  f.doc.dispatch('input', {target: nmax});
  assert.match(f.win.location.hash, /&nmax=0\.8$/);
  assert.equal(nmax.getAttribute('aria-invalid'), 'false');
  nmax.value = '150';
  f.doc.dispatch('input', {target: nmax});
  assert.doesNotMatch(f.win.location.hash, /nmax/);
  assert.equal(nmax.getAttribute('aria-invalid'), 'true');
  assert.deepEqual(a.state.ex.bad, {nmax: '150'});
  assert.match(f.el('ex-st-n').innerHTML, /max: 150 is outside 0–100 %/);
  nmax.value = '';
  f.doc.dispatch('input', {target: nmax});
  assert.deepEqual(a.state.ex.bad, {});
  assert.equal(f.el('ex-st-n').innerHTML, '');
  // an inverted range: flagged in the bar and the results
  f.doc.dispatch('input', {target: control('smax', {value: '3'})});
  assert.match(f.el('ex-st-s').innerHTML, /min 6 &gt; max 3: no task can pass/);
  assert.match(results.innerHTML, /A range is inverted:<\/b> similarity: min 6 &gt; max 3/);
  f.doc.dispatch('input', {target: control('smax', {value: ''})});
  // the DeepSWE task search re-renders the checklist only; a task chosen there: results, checklist label (no full render)
  f.doc.dispatch('input', {target: control('tq', {value: 'narwhals'})});
  assert.equal(count(f.el('ex-tlist').innerHTML, 'data-exf="t"'), 1);
  assert.equal(a.state.ex.tq, 'narwhals');
  f.doc.dispatch('change', {target: control('t', {value: NAR, checked: true})});
  assert.equal(f.win.location.hash, `#explorer&round=1,2&smin=6&nj=1&t=${NAR}`);
  assert.equal(main.writes, mainWrites);
  assert.equal(f.el('ex-st-targets').innerHTML, '1 of 4');
  // sort, page, the coverage toggle, "Select shown" (results and checklist)
  const button = (attrs) => { const b = new FakeElement(f.doc, null, 'BUTTON', main); Object.assign(b.dataset, attrs); return b; };
  f.doc.dispatch('click', {target: button({exsort: 'sim'})});
  assert.match(f.win.location.hash, /&sort=sim:desc$/);
  f.doc.dispatch('click', {target: button({exsort: 'sim'})});
  assert.match(f.win.location.hash, /&sort=sim:asc$/);
  f.doc.dispatch('click', {target: button({exaction: 'targets-clear'})});
  assert.equal(f.el('ex-st-targets').innerHTML, 'all 4');
  f.doc.dispatch('click', {target: button({expage: 'p:2'})});
  assert.match(f.win.location.hash, /&p=2$/);
  f.doc.dispatch('click', {target: button({exaction: 'cov:traj'})});
  assert.match(f.win.location.hash, /&cov=traj$/);
  assert.match(results.innerHTML, /Trajectories per DeepSWE task/);
  assert.equal(main.writes, mainWrites, 'still no full render');
  // exports: a Blob behind a download link
  f.doc.dispatch('click', {target: button({exexport: 'tasks.csv'})});
  assert.equal(clicks.length, 1);
  assert.match(clicks[0].download, /^explorer-tasks-\d{8}-\d{4}\.csv$/);
  assert.equal(blobs[0].type, 'text/csv;charset=utf-8');
  assert.equal(blobs[0].parts[0], app.explorerExport(EXD, a.state.ex.f, 'tasks.csv').text);
  f.doc.dispatch('click', {target: button({exexport: 'evil.sh'})});
  assert.equal(clicks.length, 1);
  // Reset: every filter back to its default (the coverage toggle stays), the bar rebuilt
  f.doc.dispatch('click', {target: button({exaction: 'reset'})});
  assert.equal(f.win.location.hash, '#explorer&cov=traj');
  assert.ok(main.writes > mainWrites);
  // a task row opens its task view with the node's drawer (the deep link)
  const row = new FakeElement(f.doc, null, 'TR', main);
  const t = m.tasks.find((x) => x.target === NAR && x.node === 'cand-c04');
  row.dataset.exlink = t.link;
  f.doc.dispatch('click', {target: row});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04`);
  assert.equal(a.state.view, 'task');
  assert.equal(f.el('drawer').hidden, false);
  assert.match(f.el('drawer').innerHTML, /cand-c04/);
  assert.equal(f.el('tab-explorer').getAttribute('href'), '#explorer&cov=traj');
  f.win.location.hash = '#explorer&cov=traj&dec=acc';
  await settle(a);
  assert.equal(a.state.view, 'explorer');
  assert.equal(a.state.ex.f.dec, 'acc');
  assert.ok(main.innerHTML.includes('<option value="acc" selected>accepted</option>'), 'another hash: the bar rebuilt');
  const bad = new FakeElement(f.doc, null, 'TR', main);
  bad.dataset.exlink = 'javascript:alert(1)';
  f.doc.dispatch('click', {target: bad});
  assert.equal(f.win.location.hash, '#explorer&cov=traj&dec=acc');
  a.stop();
});

test('wiring (v1.7): Escape after a node visit never rewrites the Explorer\'s (or the monitor\'s) URL', async () => {
  const start = '#explorer&round=1&audit=strong';
  const f = exEnv({hash: start, files: exFiles()});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const m = app.explorerModel(EXD), main = f.el('main');
  const t = m.tasks.find((x) => x.round === 1 && x.audit === 'strong') || m.tasks.find((x) => x.target === NAR);
  const row = new FakeElement(f.doc, null, 'TR', main);
  row.dataset.exlink = t.link;
  f.doc.dispatch('click', {target: row});                          // a row: its node in its task view
  await settle(a);
  assert.equal(a.state.nodeId, t.node);
  f.win.location.hash = start;                                     // Back
  await settle(a);
  assert.equal(a.state.view, 'explorer');
  assert.equal(a.state.nodeId, null, 'leaving the task view closes its node');
  const mark = new FakeElement(f.doc, null, 'DIV', main);           // hover a bar: its tooltip
  mark.dataset.tip = JSON.stringify({t: 'x', r: [['s1', '1', 'a']]});
  f.doc.dispatch('pointerover', {target: mark, clientX: 10, clientY: 10});
  assert.equal(f.el('ex-tip').hidden, false);
  f.doc.dispatch('keydown', {key: 'Escape', target: f.doc.body});
  assert.equal(f.win.location.hash, start);
  assert.equal(f.el('ex-tip').hidden, true);
  const box = new FakeElement(f.doc, null, 'INPUT', main);           // Escape in the DeepSWE task search
  box.dataset.exf = 'tq';
  f.doc.dispatch('keydown', {key: 'Escape', target: box});
  assert.equal(f.win.location.hash, start);
  assert.equal(a.state.view, 'explorer');
  a.selectNode('cand-c04');                                         // no node drawer outside the task view
  assert.equal(f.win.location.hash, start);
  assert.equal(f.el('drawer').hidden, true);
  // the monitor: the same
  f.win.location.hash = `#task=${NAR}&node=cand-c04`;
  await settle(a);
  assert.equal(f.el('drawer').hidden, false);
  f.win.location.hash = '#monitor';
  await settle(a);
  f.doc.dispatch('keydown', {key: 'Escape', target: f.doc.body});
  assert.equal(f.win.location.hash, '#monitor');
  assert.equal(a.state.view, 'monitor');
  // back in the task view, Escape still closes the drawer there
  f.win.location.hash = `#task=${NAR}&node=cand-c04`;
  await settle(a);
  f.doc.dispatch('keydown', {key: 'Escape', target: f.doc.body});
  assert.equal(f.win.location.hash, `#task=${NAR}`);
  assert.equal(f.el('drawer').hidden, true);
  a.stop();
});

test('wiring (v1.7): a cycle that leaves explorer.json alone renders nothing; a new file keeps the filter bar', async () => {
  const f = exEnv({hash: '#explorer&smin=6', files: exFiles()});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), results = f.el('ex-results'), head = f.el('ex-head');
  const writes = () => [main.writes, results.writes, head.writes];
  const before = writes(), fetched = f.calls.length;
  // a new collector cycle: index.json moves on, explorer.json does not (same stamp): no fetch, no write
  f.files['data/index.json'] = Object.assign(clone(EXI), {generated_at: EXI.generated_at + 120});
  f.tick(60000);
  await settle(a);
  assert.deepEqual(f.calls.slice(fetched).map((c) => c.url), ['data/index.json']);
  assert.deepEqual(writes(), before, 'inputs keep their focus and caret');
  // a new explorer.json with the same choices: head, results, checklist and status refresh; the bar stays
  const next = clone(EXD);
  next.generated_at += 400;
  next.tasks[0][next.task_columns.indexOf('title')] = 'A new title';
  f.files['data/explorer.json'] = next;
  f.files['data/index.json'] = Object.assign(clone(EXI), {generated_at: EXI.generated_at + 240,
                                                          explorer: Object.assign({}, EXI.explorer, {generated_at: next.generated_at})});
  f.tick(60000);
  await settle(a);
  assert.ok(f.calls.some((c) => c.url === 'data/explorer.json' && f.calls.indexOf(c) >= fetched));
  assert.equal(main.writes, before[0], 'the filter bar stays');
  assert.ok(results.writes > before[1] && head.writes > before[2]);
  assert.match(head.innerHTML, /explorer\.json of /);
  // a file whose choices changed (another DeepSWE task): the whole view, bar included
  const other = clone(next);
  other.generated_at += 400;
  other.runs[0].deepswe_task = 'zz-new-task';
  other.targets.push({deepswe_task: 'zz-new-task', runs: [0], patterns: []});
  f.files['data/explorer.json'] = other;
  f.files['data/index.json'] = Object.assign(clone(EXI), {generated_at: EXI.generated_at + 360,
                                                          explorer: Object.assign({}, EXI.explorer, {generated_at: other.generated_at})});
  f.tick(60000);
  await settle(a);
  assert.ok(main.writes > before[0]);
  assert.match(main.innerHTML, /data-exf="t" value="zz-new-task"/);
  a.stop();
});

test('wiring (v1.7): keyboard: one tab stop per chart panel, arrow keys move it; "Skip the charts" reaches the task table', async () => {
  const f = exEnv({hash: '#explorer', files: exFiles()});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const panel = new FakeElement(f.doc, null, 'DIV', f.el('main'));
  panel.dataset.roving = 'dists';
  const marks = [0, 1, 2].map((i) => {
    const e = new FakeElement(f.doc, null, 'DIV', panel);
    e.dataset.mark = `v:${i}`;
    e.setAttribute('tabindex', i ? '-1' : '0');
    return e;
  });
  panel.kids = marks;
  let ev = f.doc.dispatch('keydown', {key: 'ArrowRight', target: marks[0]});
  assert.equal(ev.defaultPrevented, true);
  assert.equal(f.doc.activeElement, marks[1]);
  assert.deepEqual(marks.map((e) => e.getAttribute('tabindex')), ['-1', '0', '-1']);
  f.doc.dispatch('keydown', {key: 'End', target: marks[1]});
  assert.equal(f.doc.activeElement, marks[2]);
  f.doc.dispatch('keydown', {key: 'ArrowDown', target: marks[2]});      // the last: stays
  assert.equal(f.doc.activeElement, marks[2]);
  f.doc.dispatch('keydown', {key: 'Home', target: marks[2]});
  assert.equal(f.doc.activeElement, marks[0]);
  ev = f.doc.dispatch('keydown', {key: 'Enter', target: marks[0]});     // other keys: untouched
  assert.equal(ev.defaultPrevented, false);
  f.doc.dispatch('focusin', {target: marks[2]});                         // focused otherwise (a click, a re-render)
  assert.deepEqual(marks.map((e) => e.getAttribute('tabindex')), ['-1', '-1', '0']);
  // "Skip the charts": the first row of the task table takes the focus
  const results = f.el('ex-results');
  results.innerHTML = app.renderExplorerResults(EXD, EXI, {filters: {}});
  const skip = new FakeElement(f.doc, null, 'BUTTON', results);
  skip.dataset.exaction = 'skip-tables';
  f.doc.dispatch('click', {target: skip});
  assert.equal(f.doc.activeElement.dataset.fkey, `ex:row:${app.explorerModel(EXD).tasks[0].i}`);
  assert.equal(f.doc.revealed[f.doc.revealed.length - 1].opts.block, 'start');
  a.stop();
});

test('wiring (v1.7): explorer.json missing (an older collector) or failing: a note, the rest of the page unaffected', async () => {
  const f = fakeEnv({hash: '#explorer', missing: ['data/explorer.json']});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.match(f.el('main').innerHTML, /data\/explorer\.json not available: data\/explorer\.json: HTTP 404/);
  f.win.location.hash = `#task=${NAR}`;
  await settle(a);
  assert.match(f.el('main').innerHTML, /Generation tree/);
  a.stop();
});

test('explorer (v1.7): official views, drawers, menu, monitor, header and banner render exactly as before', () => {
  const previous = path.join(STATIC, 'app.js.pre-explorer');
  if (!fs.existsSync(previous)) return;
  const old = require(previous);
  for (const ix of [F.index, BI, EXI]) {
    const views = [[NAR, V], [NAR, F.narwhals], [BAN, F.bandit], [FD, F.fd], [BT, BATCH_DOCS[0]], [BT, null], ['no-such-task', null]];
    for (const [id, doc] of views) {
      for (const o of [{}, {ui: {showAbandoned: true, selected: 'cand-c04-r1b1', treeWidth: 1700}}, {run: BRUNS[1]}]) {
        const opts = Object.assign({now: NOW}, o);
        assert.equal(app.renderTaskView(ix, id, doc, opts), old.renderTaskView(ix, id, doc, opts), id);
      }
    }
    assert.equal(app.renderTaskMenu(ix, '', NAR, 2), old.renderTaskMenu(ix, '', NAR, 2));
    assert.equal(app.renderStatus(ix, NOW, {held: true, shown: NOW - 400}), old.renderStatus(ix, NOW, {held: true, shown: NOW - 400}));
    assert.equal(app.renderBanner(ix, NOW, {errors: {index: 'x', explorer: null}}), old.renderBanner(ix, NOW, {errors: {index: 'x', explorer: null}}));
  }
  for (const doc of [V, F.bandit, BATCH_DOCS[0]]) {
    for (const n of doc.nodes) assert.equal(app.renderNodeDetail(n, doc, {}, NOW), old.renderNodeDetail(n, doc, {}, NOW), n.id);
  }
  assert.equal(app.renderMonitor(F.fleet, F.api, {now: NOW}), old.renderMonitor(F.fleet, F.api, {now: NOW}));
  assert.equal(app.renderAuthError(), old.renderAuthError());
  for (const h of ['', '#monitor', `#task=${NAR}&node=cand-c04`, `#task=${BT}&run=${BRUNS[1]}`]) assert.deepEqual(app.parseHash(h), old.parseHash(h));
});

test('wiring (v1.7): the task and monitor tabs behave as before (same markup as the previous page after the same steps)', async () => {
  const previous = path.join(STATIC, 'app.js.pre-explorer');
  if (!fs.existsSync(previous)) return;
  const old = require(previous);
  const run = async (lib) => {
    const f = fakeEnv({hash: `#task=${NAR}`, files: {'data/index.json': EXI, [`data/task/${NAR}.json`]: V}});
    const a = lib.createApp(f.env);
    await a.start();
    await settle(a);
    const shots = [f.el('main').innerHTML];
    const card = new FakeElement(f.doc, null, 'DIV', f.el('main'));
    card.dataset.node = 'cand-c04';
    card.classList.add('node');
    f.doc.dispatch('click', {target: card});
    shots.push(f.el('main').innerHTML, f.el('drawer').innerHTML, f.win.location.hash);
    f.doc.dispatch('keydown', {key: 'Escape', target: f.doc.body});
    shots.push(f.el('drawer').hidden, f.win.location.hash);
    f.win.location.hash = '#monitor';
    await settle(a);
    shots.push(f.el('main').innerHTML, f.el('status').innerHTML, f.el('banner').innerHTML, f.doc.title);
    f.win.location.hash = `#task=${NAR}&node=cand-c05`;
    await settle(a);
    shots.push(f.el('main').innerHTML, f.el('drawer').innerHTML, f.win.location.hash);
    f.win.location.hash = '#monitor';                              // a node open, then the monitor: no drawer there
    await settle(a);
    shots.push(f.el('main').innerHTML, f.el('drawer').hidden, f.el('tab-task').getAttribute('href'));
    f.win.location.hash = `#task=${NAR}`;
    await settle(a);
    shots.push(f.el('main').innerHTML, f.el('drawer').hidden, f.win.location.hash);
    a.stop();
    return shots;
  };
  assert.deepEqual(await run(app), await run(old));
});

// ----------------------------------------------------------------------------- every official run (v1.8)
// hengyuf's official2 queue (2026-10-22) builds DeepSWE tasks again that official1 runs built: every official run keeps
// a document. Here narwhals-rolling-window-suite gets a second official run, official2-narwhals (round 0 only, launched
// later: the latest, data/task/<task>.json), and the fixture's official1-narwhals becomes an older run
// (data/run/hengyuf/official1-narwhals.json, its own stamp); index.json lists both (runs[].latest / detail,
// tasks[].official_runs) and the official2 queue (index.json queues).
const OLD1 = 'official1-narwhals', NEW2 = 'official2-narwhals';
const OLDONLY = 'cand-c04-r1b2';                                  // a round-1 node: only official1-narwhals has it
const OLDDOC = Object.assign(clone(V), {generated_at: F.index.generated_at - 900});
const NEWDOC = (() => {
  const d = Object.assign(clone(V), {run: NEW2, repo: 'other/lib', max_rounds: 1, build_root: `/s/synth/runs/${NEW2}/build/${NAR}/other__lib`});
  d.nodes = d.nodes.filter((n) => n.round === 0).map((n) => Object.assign(n, {children: [], failure_match: null}));
  d.rounds = d.rounds.slice(0, 2);
  Object.assign(d.rounds[1], {status: 'pending', probe: null, seeds: null, abandoned_probes: []});
  d.job = Object.assign({}, d.job, {job_id: '300', jobs: ['300'], stage: 'round-0 done (1 accepted)'});
  return d;
})();
const QT = F.index.tasks.find((t) => !t.run && !t.batch_runs).task_id;   // a task no run converts (yet)
/** The index of a v1.8 collector for the fixtures: runs[].latest / detail, tasks[].official_runs; `two`: narwhals has
    official2-narwhals (latest) and official1-narwhals (older); `queues`: index.json queues. */
function v18Index(two, queues) {
  const ix = clone(F.index);
  for (const r of ix.runs) Object.assign(r, {latest: true, detail: `task/${r.task_id}.json`});
  if (two) {
    const old = ix.runs.find((r) => r.run === OLD1);
    Object.assign(old, {latest: false, detail: `run/hengyuf/${OLD1}.json`, detail_generated_at: OLDDOC.generated_at});
    ix.runs.push(Object.assign(clone(old), {run: NEW2, repo: 'other/lib', job_id: '300', submitted_at: old.submitted_at + 5 * 86400,
                                            stage: NEWDOC.job.stage, rounds_started: 1, latest: true, detail: `task/${NAR}.json`}));
    delete ix.runs[ix.runs.length - 1].detail_generated_at;
  }
  for (const t of ix.tasks) {
    if (!t.run) continue;
    if (two && t.task_id === NAR) {
      t.run = NEW2;
      t.official_runs = [{run: NEW2, latest: true, detail: `task/${NAR}.json`}, {run: OLD1, latest: false, detail: `run/hengyuf/${OLD1}.json`}];
    } else {
      t.official_runs = [{run: t.run, latest: true, detail: `task/${t.task_id}.json`}];
    }
  }
  if (queues) ix.queues = queues;
  return ix;
}
const QUEUE = {name: 'official2', root: '/s/synth/queue2', read_at: NOW - 40, max_active: 5,
               runner: {heartbeat_at: NOW - 45, age_s: 45, host: 'cpu-node-7', slurm_job_id: '456', note: 'running', alive: true},
               budget: {key: 'NVAPI_KEY_OFFICIAL_1', remaining: 2345.68, at: NOW - 300}, serving: true, stop_launching: false,
               updated_at: NOW - 60, finished_at: null,
               runs: [{run: NEW2, task: NAR, repo: 'other/lib', batch: 'hengyuf', status: 'running', leg: 1, winner: '300', candidates: ['300', '301']},
                      {run: 'official2-q', task: QT, repo: 'queued/q', batch: 'baihe', status: 'queued', leg: 0, winner: null, candidates: []},
                      {run: 'official2-bandit', task: BAN, repo: 'other/bandit', batch: 'hengyuf', status: 'waiting_to_resume', leg: 2,
                       winner: '310', candidates: ['310', '311']}],
               log_tail: ['[2026-10-22T22:50:00-0700] pass 50', '[2026-10-22T22:51:00-0700] budget: NVAPI_KEY_OFFICIAL_1 remaining 2345.678']};
const MI = v18Index(true);
const QI = v18Index(true, [QUEUE]);
const v18Files = (index, extra) => Object.assign({'data/index.json': index, [`data/task/${NAR}.json`]: NEWDOC,
                                                  [`data/run/hengyuf/${OLD1}.json`]: OLDDOC}, extra || {});

test('official runs (v1.8): the views of a task, latest official run first, then its older runs, then its batch runs', () => {
  const task = MI.tasks.find((t) => t.task_id === NAR);
  assert.deepEqual(app.officialRunsOf(MI, task).map((r) => r.run), [NEW2, OLD1]);
  const views = app.viewsOf(MI, task);
  assert.deepEqual(views.map((v) => [v.kind, v.run.run, v.older === true]), [['official', NEW2, false], ['official', OLD1, true]]);
  assert.equal(app.selectedView(views, OLD1), views[1]);
  assert.equal(app.selectedView(views, NEW2), views[0]);
  assert.equal(app.selectedView(views, 'nope'), views[0]);
  assert.equal(app.selectedView(views, null), views[0]);
  assert.deepEqual(app.viewDetail(views[0], NAR), {key: NAR, path: `data/task/${NAR}.json`});
  assert.deepEqual(app.viewDetail(views[1], NAR), {key: `official:${OLD1}`, path: `data/run/hengyuf/${OLD1}.json`});
  assert.equal(app.viewDetail({kind: 'official', older: true, run: {run: OLD1, detail: '../secrets.json'}}, NAR), null);
  assert.equal(app.viewDetail({kind: 'official', older: true, run: {run: OLD1, detail: `run/baiheh/${OLD1}.json`}}, NAR), null);
  // an index of a v1.7 collector (no official_runs): the latest alone, as before
  assert.deepEqual(app.officialRunsOf(F.index, F.index.tasks.find((t) => t.task_id === NAR)).map((r) => r.run), [OLD1]);
  // a batch run after the official ones
  const both = v18Index(true);
  Object.assign(both, {runs: both.runs.concat(clone(BI.runs.filter((r) => r.source))), batches: clone(BI.batches)});
  both.tasks.find((t) => t.task_id === NAR).batch_runs = [{source: BSRC, run: BRUNS[0]}];
  both.runs.find((r) => r.run === BRUNS[0]).task_id = NAR;
  assert.deepEqual(app.viewsOf(both, both.tasks.find((t) => t.task_id === NAR)).map((v) => v.run.run), [NEW2, OLD1, BRUNS[0]]);
});

test('official runs (v1.8): the run selector and an older run\'s view (its own document, a note, the latest named)', () => {
  const latest = app.renderTaskView(MI, NAR, NEWDOC, {now: NOW});
  assert.ok(!latest.includes('render-error'));
  assert.match(latest, /<button type="button" class="run-tab active" data-run-select="" aria-pressed="true"><span class="run-tab-name">official2-narwhals<\/span>/);
  assert.match(latest, /data-run-select="official1-narwhals" aria-pressed="false"/);
  // recency is no status: "latest" in a neutral tone (the status tones good / bad stay reserved)
  assert.match(latest, /official2-narwhals<\/span><span class="run-tab-meta"><span class="badge tone-evolve badge-xs"[^>]*><span class="dot" aria-hidden="true"><\/span>official<\/span><span class="badge tone-neutral badge-xs"[^>]*>.*?latest<\/span>/);
  assert.doesNotMatch(latest, /tone-good badge-xs[^>]*>.*?latest/);
  assert.match(latest, /official1-narwhals<\/span><span class="run-tab-meta"><span class="badge tone-evolve badge-xs"[^>]*>.*?official<\/span><span class="badge tone-muted badge-xs"[^>]*>.*?older<\/span>/);
  assert.ok(!latest.includes('older-run-note'));
  assert.ok(!latest.includes(OLDONLY), 'the latest run has no round-1 node');
  const older = app.renderTaskView(MI, NAR, OLDDOC, {now: NOW, run: OLD1});
  assert.ok(!older.includes('render-error'));
  assert.match(older, /class="run-tab active" data-run-select="official1-narwhals" aria-pressed="true"/);
  assert.match(older, /<div class="older-run-note"><span class="badge tone-muted badge-sm"[^>]*>.*?older official run<\/span><span>official1-narwhals: an earlier official run of this task\. official2-narwhals is its latest official run and its default view;/);
  assert.ok(older.includes(OLDONLY) && older.includes('Generation tree'));
  // the latest run's document is no older run's: shown as loading, never mixed up
  assert.match(app.renderTaskView(MI, NAR, NEWDOC, {now: NOW, run: OLD1}), /Loading data\/run\/hengyuf\/official1-narwhals\.json…/);
  assert.match(app.renderTaskView(MI, NAR, OLDDOC, {now: NOW}), /Loading data\/task\/narwhals-rolling-window-suite\.json…/);
  assert.match(app.renderTaskView(MI, NAR, null, {now: NOW, run: OLD1, detailError: 'HTTP 404'}), /Run detail not available: HTTP 404/);
  // escaping of the run names the selector and the note show
  const ix = clone(MI);
  ix.runs.find((r) => r.run === OLD1).repo = PAYLOAD;
  assertEscaped(app.renderTaskView(ix, NAR, Object.assign(clone(OLDDOC), {repo: PAYLOAD}), {now: NOW, run: OLD1}), 'older view');
});

test('official runs (v1.8): a task with one official run renders exactly as the v1.7 page did (static/app.js.pre-v18)', () => {
  const previous = path.join(STATIC, 'app.js.pre-v18');
  if (!fs.existsSync(previous)) return;
  const old = require(previous);
  const ix = v18Index(false);
  for (const [id, doc] of [[NAR, V], [NAR, F.narwhals], [BAN, F.bandit], ['fd-deterministic-multi-key-sorting', F.fd], [QT, null]]) {
    for (const o of [{}, {ui: {showAbandoned: true, selected: 'cand-c04-r1b1'}}, {run: 'official1-narwhals'}]) {
      const opts = Object.assign({now: NOW}, o);
      assert.equal(app.renderTaskView(ix, id, doc, opts), old.renderTaskView(F.index, id, doc, opts), `${id} ${JSON.stringify(o)}`);
    }
  }
  assert.equal(app.renderTaskMenu(ix, '', NAR, 2), old.renderTaskMenu(F.index, '', NAR, 2));
  assert.equal(app.renderTaskMenu(ix, 'bandit', NAR, 0), old.renderTaskMenu(F.index, 'bandit', NAR, 0));
  assert.equal(app.renderMonitor(F.fleet, F.api, {now: NOW, index: ix}), old.renderMonitor(F.fleet, F.api, {now: NOW}));
  assert.equal(app.renderStatus(ix, NOW, {}), old.renderStatus(F.index, NOW, {}));
});

test('queue (v1.8): the picker notes the tasks queued in a synthesis queue and not launched yet', () => {
  assert.deepEqual(app.queuedIn(QI, QT), [{queue: 'official2', run: 'official2-q'}]);
  assert.deepEqual(app.queuedIn(QI, NAR), []);                    // running
  assert.deepEqual(app.queuedIn(QI, BAN), []);                    // waiting to resume: launched before
  const launched = clone(QI);                                      // its runs[] entry exists: launched, never "queued"
  launched.runs.push(Object.assign(clone(launched.runs[0]), {run: 'official2-q', task_id: QT}));
  assert.deepEqual(app.queuedIn(launched, QT), []);
  const menu = app.renderTaskMenu(QI, '', NAR, 0);
  const opt = menu.slice(menu.indexOf(`data-task="${QT}"`), menu.indexOf('</div>', menu.indexOf(`data-task="${QT}"`)) + 6);
  assert.match(opt, /<span class="opt-queued" title="official2-q: queued in official2, not launched yet">queued in official2<\/span>/);
  assert.equal(count(menu, 'opt-queued'), count(app.renderTaskMenu(QI, '', NAR, 0), `data-task="${QT}"`));
  assert.ok(app.taskGroups(QI, 'official2-q').some((g) => g.items.some((t) => t.task_id === QT)), 'search finds the queued run');
  assert.ok(app.taskGroups(QI, OLD1).find((g) => g.key === 'converting').items.some((t) => t.task_id === NAR), 'and an older run');
  const evil = clone(QI);
  Object.assign(evil.queues[0], {name: PAYLOAD});
  Object.assign(evil.queues[0].runs[1], {run: PAYLOAD});
  assertEscaped(app.renderTaskMenu(evil, '', NAR, 0), 'picker');
});

test('queue (v1.8): the Monitor\'s synthesis queue: runner, budget, flags, runs (linked once launched), log tail', () => {
  const html = app.renderMonitor(F.fleet, F.api, {now: NOW, index: QI});
  assert.ok(!html.includes('render-error'));
  assert.ok(html.indexOf('<h2>Synthesis queue official2</h2>') >= 0 && html.indexOf('<h2>Synthesis queue official2</h2>') < html.indexOf('<h2>Inference workers</h2>'));
  assert.match(html, /<span class="badge tone-good"[^>]*>.*?runner alive<\/span><span class="muted small">running · on cpu-node-7 · job 456 · heartbeat 45s ago<\/span>/);
  assert.match(html, /<span class="queue-budget">Hub budget · NVAPI_KEY_OFFICIAL_1: \$2,345\.68 remaining \(checked 5m ago\)<\/span>/);
  assert.match(html, /student serving on/);
  assert.ok(!html.includes('launching stopped'));
  assert.match(html, /3 runs, at most 5 at a time · 1 running · 1 waiting to resume · 1 queued · state 1m ago/);
  assert.ok(html.includes(`<td><a href="#task=${NAR}"><code>${NEW2}</code></a></td>`), 'a launched latest run links its task view');
  assert.ok(html.includes('<td><code>official2-q</code></td>'), 'a queued run links nothing');
  assert.match(html, /<span class="badge tone-muted badge-sm" title="not launched yet">.*?queued<\/span>/);
  assert.match(html, /<span class="badge tone-pending badge-sm"[^>]*>.*?waiting to resume<\/span>/);
  assert.ok(html.includes('<td>300</td><td class="small">300, 301</td>'));
  assert.match(html, /runner\.log, last 2 lines/);
  assert.ok(html.includes('pass 50\n[2026-10-22T22:51:00-0700] budget: NVAPI_KEY_OFFICIAL_1 remaining 2345.678</pre>'));
  // an older official run is linked with run=
  const older = clone(QI);
  older.queues[0].runs[0].run = OLD1;
  assert.ok(app.renderMonitor(F.fleet, F.api, {now: NOW, index: older}).includes(`<a href="#task=${NAR}&amp;run=${OLD1}"><code>${OLD1}</code></a>`));
  // the runner's other states, no budget yet, launching stopped
  const states = (runner) => app.renderMonitor(null, null, {now: NOW, index: Object.assign(clone(QI), {queues: [Object.assign(clone(QUEUE), {runner})]})});
  assert.match(states({heartbeat_at: null, age_s: null, host: null, slurm_job_id: null, note: null, alive: false}), /runner never ran/);
  assert.match(states({heartbeat_at: NOW - 2000, age_s: 2000, host: 'h', slurm_job_id: null, note: 'running', alive: false}), /runner stale/);
  assert.match(states({heartbeat_at: NOW - 20, age_s: 20, host: 'h', slurm_job_id: null, note: 'stopped', alive: false}), /runner stopped/);
  // the page's own clock decides, never the index's age_s / alive of when the files were read
  const late = states({heartbeat_at: NOW - 1000, age_s: 5, host: 'h', slurm_job_id: '9', note: 'running', alive: true});
  assert.match(late, /runner stale/);
  assert.match(late, /running · on h · job 9 · heartbeat 16m ago/);
  assert.match(states({heartbeat_at: NOW - 899, age_s: 950, host: 'h', slurm_job_id: null, note: 'running', alive: false}), /runner alive/);
  const none = app.renderMonitor(null, null, {now: NOW, index: Object.assign(clone(QI), {queues: [Object.assign(clone(QUEUE), {budget: null, stop_launching: true, serving: false, runs: [], log_tail: []})]})});
  assert.match(none, /Hub budget · not checked yet/);
  assert.match(none, /launching stopped/);
  assert.match(none, /student serving off/);
  assert.match(none, /The queue lists no run\./);
  assert.match(none, /No log yet\./);
  // escaping: every string of the queue
  const evil = clone(QI);
  const q = evil.queues[0];
  Object.assign(q, {name: PAYLOAD, root: PAYLOAD, log_tail: [PAYLOAD]});
  Object.assign(q.runner, {host: PAYLOAD, note: PAYLOAD, slurm_job_id: PAYLOAD});
  Object.assign(q.budget, {key: PAYLOAD});
  for (const r of q.runs) Object.assign(r, {run: PAYLOAD, task: PAYLOAD, repo: PAYLOAD, batch: PAYLOAD, winner: PAYLOAD, candidates: [PAYLOAD]});
  q.runs[0].status = PAYLOAD;
  assertEscaped(app.renderMonitor(F.fleet, F.api, {now: NOW, index: evil}), 'queue');
  for (const broken of [{}, {queues: 'x'}, {queues: [null, 1, {runs: [null, {}]}]}]) {
    assert.doesNotThrow(() => app.renderMonitor(F.fleet, F.api, {now: NOW, index: broken}));
    assert.ok(!app.renderMonitor(F.fleet, F.api, {now: NOW, index: broken}).includes('render-error'));
  }
});

test('wiring (v1.8): #task=<id>&run=<older run> opens it (its file), the selector switches back, node links keep run=', async () => {
  const f = fakeEnv({hash: `#task=${NAR}&run=${OLD1}`, files: v18Files(MI)});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main');
  assert.ok(f.calls.some((c) => c.url === `data/run/hengyuf/${OLD1}.json`));
  assert.ok(!f.calls.some((c) => c.url === `data/task/${NAR}.json`), 'the latest run\'s file is not needed');
  assert.match(main.innerHTML, /class="run-tab active" data-run-select="official1-narwhals"/);
  assert.match(main.innerHTML, /older-run-note/);
  // a node of the older run: the hash keeps run=
  const card = new FakeElement(f.doc, null, 'DIV', main);
  card.dataset.node = OLDONLY;
  card.classList.add('node');
  f.doc.dispatch('click', {target: card});
  assert.equal(f.win.location.hash, `#task=${NAR}&run=${OLD1}&node=${OLDONLY}`);
  assert.equal(f.el('drawer').hidden, false);
  // its document is fetched again only when index.json names a newer one
  const before = f.calls.length;
  f.newCycle();
  f.files[`data/run/hengyuf/${OLD1}.json`].generated_at -= 120;      // unchanged, not rewritten
  f.tick(60000);
  await settle(a);
  assert.deepEqual(f.calls.slice(before).map((c) => c.url), ['data/index.json']);
  // the latest run's tab
  const tab = new FakeElement(f.doc, null, 'BUTTON', main);
  tab.dataset.runSelect = '';
  f.doc.dispatch('click', {target: tab});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}`);
  assert.ok(f.calls.some((c) => c.url === `data/task/${NAR}.json`));
  assert.match(main.innerHTML, /class="run-tab active" data-run-select=""/);
  assert.ok(!main.innerHTML.includes('older-run-note'));
  a.stop();
});

test('wiring (v1.8): a node link without run= opens the run that has the node (an official1 link once official2 is latest)', async () => {
  // a node only the older run has: its document is read, the view and the hash name that run, the drawer shows the node
  let f = fakeEnv({hash: `#task=${NAR}&node=${OLDONLY}`, files: v18Files(MI)});
  let a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&run=${OLD1}&node=${OLDONLY}`);
  assert.deepEqual(f.calls.filter((c) => c.url.startsWith('data/task/') || c.url.startsWith('data/run/')).map((c) => c.url),
                   [`data/task/${NAR}.json`, `data/run/hengyuf/${OLD1}.json`]);
  assert.match(f.el('main').innerHTML, /class="run-tab active" data-run-select="official1-narwhals"/);
  assert.equal(f.el('drawer').hidden, false);
  assert.ok(f.el('drawer').innerHTML.includes(OLDONLY));
  a.stop();
  // a node of the default run: nothing else is read, the hash stays
  f = fakeEnv({hash: `#task=${NAR}&node=cand-c04`, files: v18Files(MI)});
  a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-c04`);
  assert.ok(!f.calls.some((c) => c.url.startsWith('data/run/')));
  assert.match(f.el('main').innerHTML, /class="run-tab active" data-run-select=""/);
  a.stop();
  // a node no run has: every run is looked at once, the hash and the view stay
  f = fakeEnv({hash: `#task=${NAR}&node=cand-nowhere`, files: v18Files(MI)});
  a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&node=cand-nowhere`);
  assert.equal(f.calls.filter((c) => c.url === `data/run/hengyuf/${OLD1}.json`).length, 1);
  f.tick(60000);
  await settle(a);
  assert.equal(f.calls.filter((c) => c.url === `data/run/hengyuf/${OLD1}.json`).length, 1, 'resolved once');
  a.stop();
  // run= named: never second-guessed; a task with a single run: never looked elsewhere
  f = fakeEnv({hash: `#task=${NAR}&run=${NEW2}&node=${OLDONLY}`, files: v18Files(MI)});
  a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&run=${NEW2}&node=${OLDONLY}`);
  assert.ok(!f.calls.some((c) => c.url.startsWith('data/run/')));
  a.stop();
  f = fakeEnv({hash: `#task=${BAN}&node=cand-nowhere`, files: {'data/index.json': v18Index(false)}});
  a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${BAN}&node=cand-nowhere`);
  a.stop();
});

test('wiring (v1.8): the Monitor shows the queue of the index.json it holds and refreshes it with every cycle', async () => {
  const f = fakeEnv({hash: '#monitor', files: v18Files(QI)});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.match(f.el('main').innerHTML, /<h2>Synthesis queue official2<\/h2>/);
  assert.match(f.el('main').innerHTML, /1 running · 1 waiting to resume · 1 queued/);
  const next = clone(QI);
  next.generated_at += 120;
  next.queues[0].runs[1].status = 'pending';
  f.files['data/index.json'] = next;
  f.tick(60000);
  await settle(a);
  assert.match(f.el('main').innerHTML, /1 running · 1 pending · 1 waiting to resume/);
  a.stop();
});

// The Explorer over every official run: official1-narwhals (older) and official2-narwhals (latest) plus the batch
const EX8_ENTRIES = [{source: 'hengyuf', detail: `run/hengyuf/${OLD1}.json`, doc: OLDDOC, latest: false},
                     {source: 'hengyuf', detail: `task/${NAR}.json`, doc: NEWDOC},
                     {source: 'hengyuf', detail: `task/${BAN}.json`, doc: F.bandit},
                     ...BATCH_DOCS.map((d) => ({source: BSRC, detail: `run/${BSRC}/${d.run}.json`, doc: d}))];
const EXD8 = explorerOf(EX8_ENTRIES, F.index.generated_at - 300, EX_SOURCES);
const EXI8 = Object.assign(v18Index(true), {batches: clone(BI.batches), runs: v18Index(true).runs.concat(clone(BI.runs.filter((r) => r.source))),
                                            explorer: {path: 'explorer.json', generated_at: EXD8.generated_at,
                                                       bytes: JSON.stringify(EXD8).length, tasks: EXD8.tasks.length,
                                                       trajectories: EXD8.trajectories.length}});
EXI8.tasks.find((t) => t.task_id === BT).batch_runs = BRUNS.map((run) => ({source: BSRC, run}));
const ex8Run = (over) => app.filterExplorer(app.explorerModel(EXD8), Object.assign(app.explorerDefaults(), over));

test('explorer (v1.8): rows of every official run; run families; an older run\'s rows link with run=', () => {
  const m = app.explorerModel(EXD8);
  assert.deepEqual(m.runs.map((r) => [r.run, r.family, r.familyLabel, r.latest]),
                   [[OLD1, 'official1', 'official1', false], [NEW2, 'official2', 'official2', true],
                    ['official1-bandit', 'official1', 'official1', true], [BRUNS[0], BSRC, 'Testy', null], [BRUNS[1], BSRC, 'Testy', null]]);
  assert.deepEqual(m.families, [{key: 'official1', label: 'official1', source: 'hengyuf'}, {key: 'official2', label: 'official2', source: 'hengyuf'},
                                {key: BSRC, label: 'Testy', source: BSRC}]);
  assert.deepEqual(m.runNames, [OLD1, NEW2, 'official1-bandit', ...BRUNS]);
  const olderRows = m.tasks.filter((t) => t.run.run === OLD1), newerRows = m.tasks.filter((t) => t.run.run === NEW2);
  assert.ok(olderRows.length > newerRows.length && newerRows.length > 0);
  for (const t of olderRows) assert.equal(t.link, app.taskHash(NAR, t.node, OLD1));
  for (const t of newerRows) assert.equal(t.link, app.taskHash(NAR, t.node, null));
  assert.equal(EXD8.totals.hengyuf.runs, 3);
  // a v1.7 file (no runs[].latest): every official run is latest, families from the names
  const legacy = app.explorerModel(explorerOf(EX_ENTRIES, F.index.generated_at - 300, EX_SOURCES, true));
  assert.ok(legacy.runs.filter((r) => r.source.audited).every((r) => r.latest === true));
  // the table marks an older run's rows, the scope note counts the older runs
  const table = app.renderExplorerTaskTable(m, ex8Run({run: [OLD1]}), {run: [OLD1]});
  assert.match(table, new RegExp(`<div class="muted small">${OLD1} · older</div>`));
  const scope = app.explorerScope(m, EXI8);
  assert.match(scope[0], /^hengyuf: every built task of every official run, 3 runs: each DeepSWE target's latest run \(data\/task\/<task>\.json\) and 1 older official run of the same targets \(data\/run\/hengyuf\/<run>\.json; their rows link with run=\)\.$/);
});

test('explorer (v1.8): the run filter (families as chips, runs as a searchable checklist) and its hash', () => {
  const m = app.explorerModel(EXD8);
  const ids = (r) => r.tasks.map((t) => t.i), want = (pred) => m.tasks.filter(pred).map((t) => t.i);
  assert.deepEqual(ids(ex8Run({fam: ['official1']})), want((t) => t.run.family === 'official1'));
  assert.deepEqual(ids(ex8Run({fam: ['official2', BSRC]})), want((t) => t.run.family !== 'official1'));
  assert.deepEqual(ids(ex8Run({run: [OLD1]})), want((t) => t.run.run === OLD1));
  assert.deepEqual(ids(ex8Run({run: [OLD1, NEW2], fam: ['official2']})), want((t) => t.run.run === NEW2));
  assert.deepEqual(ids(ex8Run({run: []})), want(() => true), 'an empty run list is every run');
  assert.ok(ex8Run({fam: ['official1']}).tasks.length < m.tasks.length);
  // the hash: fam after src, run after t; a round trip
  const f = Object.assign(app.explorerDefaults(), {src: ['hengyuf'], fam: ['official1', 'official2'], t: [NAR], run: [NEW2, OLD1].sort()});
  const h = app.explorerHash(f);
  assert.equal(h, `#explorer&src=hengyuf&fam=official1,official2&t=${NAR}&run=${OLD1},${NEW2}`);
  assert.deepEqual(app.parseExplorerParams(h), f);
  assert.deepEqual(app.parseExplorerParams('run=<b>,official1-x&fam=a b'), Object.assign(app.explorerDefaults(), {run: ['official1-x']}));
  assert.deepEqual(app.updateExplorerFilter(app.explorerDefaults(), 'run', {value: OLD1, checked: true}, m).run, [OLD1]);
  assert.equal(app.updateExplorerFilter(Object.assign(app.explorerDefaults(), {run: [OLD1]}), 'run', {value: OLD1, checked: false}, m).run, null);
  assert.deepEqual(app.updateExplorerFilter(app.explorerDefaults(), 'fam', {value: 'official1', checked: false}, m).fam, ['official2', BSRC]);
  // the bar: family chips, the run checklist (search, older runs marked, counts), its label slot
  const bar = app.renderExplorerFilters(m, {run: [OLD1]}, {});
  assert.match(bar, /<legend class="ex-f-k">Run<\/legend><div class="ex-f-v"><div class="ex-runs-fam"><label class="ex-chip" title="official runs named official1-\*"><input type="checkbox" data-exf="fam" value="official1" data-fkey="exf:fam:official1" checked><span>official1<\/span><\/label>/);
  assert.match(bar, /data-exf="fam" value="testbatch"[^>]*checked><span>Testy<\/span>/);
  assert.ok(bar.includes('<span id="ex-st-runs" class="ex-dd-label">1 of 5</span>'));
  assert.ok(bar.includes('data-exf="rq"') && bar.includes('<div id="ex-rlist" class="ex-tlist" data-scroll="ex:rlist">'));
  assert.ok(bar.includes('data-exaction="runs-shown"') && bar.includes('data-exaction="runs-clear"'));
  const list = app.renderExplorerRunList(m, {run: [OLD1]}, '');
  assert.equal(count(list, 'data-exf="run"'), 5);
  assert.match(list, new RegExp(`value="${OLD1}" data-fkey="exf:run:${OLD1}" checked><span class="ex-titem-id">${OLD1}</span><span class="muted small">older · \\d+ tasks</span>`));
  assert.equal(count(app.renderExplorerRunList(m, {}, 'testy'), 'data-exf="run"'), 2, 'the search matches the family too');
  assert.equal(app.renderExplorerRunList(m, {}, 'zzz'), '<p class="muted small">No run matches “zzz”.</p>');
  assert.equal(app.explorerFilterStatus(m, {}, null, {}).runs, 'all 5');
  assertEscaped(app.renderExplorerRunList(app.explorerModel(Object.assign(clone(EXD8), {runs: EXD8.runs.map((r) => Object.assign({}, r, {run: PAYLOAD}))})), {}, PAYLOAD), 'run list');
});

test('wiring (v1.8): the run filter re-renders the results and its label, the search its list, Reset clears it', async () => {
  const files = {'data/index.json': EXI8, 'data/explorer.json': EXD8};
  const f = exEnv({hash: '#explorer', files});
  for (const id of ['ex-rlist', 'ex-st-runs']) f.doc.elements[id] = new FakeElement(f.doc, id, 'DIV', f.doc.body);
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  const main = f.el('main'), results = f.el('ex-results'), writes = main.writes;
  const control = (exf, props) => Object.assign(new FakeElement(f.doc, null, 'INPUT', main), props, {dataset: {exf}});
  f.doc.dispatch('change', {target: control('run', {value: OLD1, checked: true})});
  assert.equal(f.win.location.hash, `#explorer&run=${OLD1}`);
  assert.equal(f.el('ex-st-runs').innerHTML, '1 of 5');
  assert.ok(f.el('ex-rlist').innerHTML.includes(`value="${OLD1}" data-fkey="exf:run:${OLD1}" checked`));
  assert.equal(main.writes, writes, 'the bar stays');
  const m = app.explorerModel(EXD8);
  assert.ok(results.innerHTML.includes(`${app.fmtInt(m.tasks.filter((t) => t.run.run === OLD1).length)}<span class="tile-den"> / `));
  f.doc.dispatch('input', {target: control('rq', {value: 'pilot'})});
  assert.equal(count(f.el('ex-rlist').innerHTML, 'data-exf="run"'), 2);
  assert.equal(a.state.ex.rq, 'pilot');
  const button = (attrs) => { const b = new FakeElement(f.doc, null, 'BUTTON', main); Object.assign(b.dataset, attrs); return b; };
  f.doc.dispatch('click', {target: button({exaction: 'runs-shown'})});
  assert.equal(f.win.location.hash, `#explorer&run=${OLD1},${BRUNS[0]},${BRUNS[1]}`);
  f.doc.dispatch('click', {target: button({exaction: 'runs-clear'})});
  assert.equal(f.win.location.hash, '#explorer');
  f.doc.dispatch('change', {target: control('fam', {value: 'official1', checked: false})});
  assert.equal(f.win.location.hash, `#explorer&fam=official2,${BSRC}`);
  f.doc.dispatch('click', {target: button({exaction: 'reset'})});
  assert.equal(f.win.location.hash, '#explorer');
  assert.equal(a.state.ex.rq, '');
  // a row of the older run opens its run's view (its file), the hash names the run
  const row = new FakeElement(f.doc, null, 'TR', main);
  row.dataset.exlink = m.tasks.find((t) => t.run.run === OLD1 && t.node === OLDONLY).link;
  f.files[`data/run/hengyuf/${OLD1}.json`] = OLDDOC;
  f.doc.dispatch('click', {target: row});
  await settle(a);
  assert.equal(f.win.location.hash, `#task=${NAR}&run=${OLD1}&node=${OLDONLY}`);
  assert.ok(f.calls.some((c) => c.url === `data/run/hengyuf/${OLD1}.json`));
  assert.ok(f.el('drawer').innerHTML.includes(OLDONLY));
  a.stop();
});

test('explorer (v1.8): range inputs are text inputs, so "abc" is rejected visibly (never a silently cleared bound)', async () => {
  const f = exEnv({hash: '#explorer&nmax=0.8', files: {'data/index.json': EXI8, 'data/explorer.json': EXD8}});
  const a = app.createApp(f.env);
  await a.start();
  await settle(a);
  assert.match(f.el('main').innerHTML, /<input type="text" inputmode="decimal" class="ex-num" data-exf="nmax" data-fkey="exf:nmax" value="80"/);
  const input = Object.assign(new FakeElement(f.doc, null, 'INPUT', f.el('main')), {value: 'abc'}, {dataset: {exf: 'nmax'}});
  f.doc.dispatch('input', {target: input});
  assert.equal(input.getAttribute('aria-invalid'), 'true');
  assert.match(f.el('ex-st-n').innerHTML, /max: “abc” is not a number/);
  assert.equal(f.win.location.hash, '#explorer', 'the bound is not applied');
  assert.deepEqual(a.state.ex.bad, {nmax: 'abc'});
  for (const [raw, why] of [['1e5', '100K is outside 0–100 %'], ['--3', '“--3” is not a number'], ['50%', '“50%” is not a number']]) {
    input.value = raw;
    f.doc.dispatch('input', {target: input});
    assert.match(f.el('ex-st-n').innerHTML, new RegExp(`max: ${why.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}`), raw);
  }
  input.value = '40';
  f.doc.dispatch('input', {target: input});
  assert.equal(input.getAttribute('aria-invalid'), 'false');
  assert.equal(f.win.location.hash, '#explorer&nmax=0.4');
  a.stop();
});

test('explorer (v1.8): a bar label keeps its full text in a title, two lines at most (no ellipsis)', () => {
  const css = fs.readFileSync(path.join(STATIC, 'style.css'), 'utf8');
  assert.match(css, /\.hbar-label \{ display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2;[^}]*overflow-wrap: anywhere/);
  assert.doesNotMatch(css.match(/\.hbar-label \{[^}]*\}/)[0], /nowrap|ellipsis/);
  assert.match(css, /\.hbar \{ align-items: center; min-height: 22px;/);
  const long = 'a-very-long-deepswe-task-name-that-would-be-cut-somewhere-in-the-middle';
  const bars = app.renderHBars({key: 'k', title: 'T', series: [{label: 'a', cls: 's1'}], rows: [{label: long, title: `${long}: 3 tasks`, values: [3]}]});
  assert.ok(bars.includes(`<span class="hbar-label" title="${long}: 3 tasks">${long}</span>`));
  assert.ok(app.renderHBars({key: 'k', title: 'T', series: [{label: 'a', cls: 's1'}], rows: [{label: 'x', values: [1]}]})
    .includes('<span class="hbar-label" title="x">x</span>'));
});
