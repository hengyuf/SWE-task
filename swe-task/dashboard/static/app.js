/* swe-task official synthesis dashboard: the single-page app (vanilla JS, no external resources).

Data (D/CONTRACT.md), same-origin JSON written by collect.py once per cycle: data/index.json,
data/task/<task_id>.json, data/fleet.json, data/api.json. The file has two halves:
  1. Pure renderers: data -> HTML/SVG string. No DOM access. They never throw on null, missing or mistyped fields
     (each page section is also wrapped in safe()), and every string taken from data goes through esc(). node loads
     this file as a module for tests/frontend.test.js.
  2. DOM wiring (createApp): fetch with the session cookie, hash routing (#task=<id>[&run=<run>][&node=<id>] |
     #monitor), the task picker, the node drawer (resizable), 60 s auto-refresh. A refresh re-renders only when a new
     collector cycle landed, waits while text is selected, and keeps the window, tree, drawer and <pre> scroll positions
     and the open/closed state of collapsibles (pure helpers: refreshAction, captureScroll, restoreScroll, captureOpen).
     window, document and fetch are injected, so the tests drive it against a small fake DOM; a browser calls boot().
Batches (CONTRACT "v1.6"): index.json runs[] entries with a `source` are the runs of a batch from another tree (Baihe's),
never audited; a task lists them in `batch_runs` and each has its own document data/run/<source>/<run>.json. Official
tasks render exactly as before; a task with batch runs gets their views (a run selector when it has two or more runs).
The Explorer (CONTRACT "v1.7", #explorer[&<filters>]): data/explorer.json, the synthetic tasks of the task and run files
and their trajectories as column-headed rows; pure filter / statistics / chart / table / export functions, and a filter
edit that re-renders the results alone. Every other view renders exactly as before.
Every official run (CONTRACT "v1.8"): a task's latest official run keeps data/task/<task_id>.json; an older official run of
the same task (official1-etree once official2-etree launched) has data/run/hengyuf/<run>.json and its own view in the run
selector (latest first, then older official runs, then batch runs; #task=<id>&run=<older run>); a #task=<id>&node=<n> link
without run= whose node is not in the default run opens the run that has it. The monitor shows the synthesis queues
(index.json queues: runs, runner heartbeat, budget, log tail) and the picker marks tasks queued but not launched yet. A
task with one official run renders exactly as before.
*/
'use strict';

// ------------------------------------------------------------------------------------------------ constants

const REFRESH_MS = 60 * 1000;
const STALE_S = 5 * 60;          // index.json older than this -> staleness warning (CONTRACT "UI")
const STUDENT = 'sc14';
const TEACHER = 'kimi';
const MODEL_ORDER = ['kimi', 'super_vl', STUDENT];            // display order (CONTRACT data/index.json)
const MODEL_FALLBACK = {kimi: {label: 'Kimi K3', role: 'teacher'},
                        super_vl: {label: 'Nemotron 3.5 Super VL', role: "Baihe's student"},
                        sc14: {label: 'Local Super ckpt sc14', role: 'student'}};
const DASH = '\u2013';
// Color classes (style.css .tone-*). Classes are only ever built from these names, never from data.
const TONES = new Set(['good', 'bad', 'evolve', 'accepted-evolve', 'pending', 'muted', 'abandoned', 'serious',
                       'neutral']);
// Decision badges (CONTRACT "Decision rules"). BUILD FAILED is shown for a discard whose build failed.
const DECISIONS = {
  evolve: {key: 'evolve', label: 'EVOLVE', tone: 'evolve', help: 'valid seed: branches into evolved tasks next round'},
  accepted_evolve: {key: 'accepted_evolve', label: 'ACCEPTED+EVOLVE', tone: 'accepted-evolve',
                    help: 'round 0, accepted; like every valid round-0 task (since 2026-10-17) it is also a seed'},
  accepted: {key: 'accepted', label: 'ACCEPTED', tone: 'good',
             help: 'accepted: copied to runs/<run>/accepted (also when seeds.json lists it as an invalid seed)'},
  discard: {key: 'discard', label: 'DISCARD', tone: 'bad',
            help: 'not accepted and invalid in seeds.json: the verifier gave no grade, or < 3 valid rollouts per model'},
  build_failed: {key: 'build_failed', label: 'BUILD FAILED', tone: 'bad',
                 help: 'authoring or gates failed (stage.json ok=false); never probed'},
  not_selected: {key: 'not_selected', label: 'NOT SELECTED', tone: 'muted',
                 help: 'valid, but another child of its lineage seeded the next round'},
  end: {key: 'end', label: 'END', tone: 'muted', help: 'scored in the last round (max_rounds)'},
  pending: {key: 'pending', label: 'PENDING', tone: 'pending', help: 'not scored yet'},
  abandoned: {key: 'abandoned', label: 'ABANDONED', tone: 'abandoned',
              help: 'its probe/round was superseded, or its parent was an invalid seed'},
};
const DECISION_ORDER = ['evolve', 'accepted_evolve', 'accepted', 'discard', 'build_failed', 'not_selected', 'end',
                        'pending', 'abandoned'];
// CONTRACT "v1.5": NODE.seed_invalid, why a scored task may not seed evolution. It decides only a task that was not
// accepted (DISCARD, its reason); on any other decision, e.g. an accepted task seeds.json lists invalid (acceptance
// beats an invalid seed), it is a secondary note: an amber "seed-invalid" badge next to the decision badge.
const SEED_NOTE = {key: 'seed_invalid', label: 'seed-invalid', tone: 'pending',
                   help: 'seed validity only decides which tasks seed the next round; it never undoes an acceptance'};
// Failure-match audit (CONTRACT "v1.3", "v1.4"): does sc14 repeat the target's failure patterns in a task? Its scope:
// the eligible tasks (v1.4: built and probed, a graded sc14 rollout, not discarded or withdrawn), the accepted ones for
// v1.3 collectors. "FA match" badges: strong green, partial amber, weak orange, none red, not audited grey.
const FA_VERDICTS = {
  strong: {key: 'strong', label: 'strong', tone: 'good',
           help: 'most failed sc14 rollouts fail the way a target failure pattern describes'},
  partial: {key: 'partial', label: 'partial', tone: 'pending',
            help: 'some failed sc14 rollouts match a target failure pattern, or the match is loose'},
  weak: {key: 'weak', label: 'weak', tone: 'serious', help: 'a match in one sc14 rollout only, or only superficially'},
  none: {key: 'none', label: 'none', tone: 'bad', help: 'the sc14 failures have other causes than the target patterns'},
};
const FA_ORDER = ['strong', 'partial', 'weak', 'none'];
const FA_NOT_AUDITED = {key: 'not_audited', label: 'not audited', tone: 'muted',
                        help: 'no failure-match audit of this task yet'};
const FA_UNKNOWN = {key: 'unknown', label: 'unknown', tone: 'neutral', help: 'the audit carries no verdict this page knows'};
const REPRODUCED_TONES = {yes: 'good', partial: 'pending', no: 'muted'};   // a target pattern, in the audited rollouts
const RUN_TONES = {running: 'evolve', pending: 'pending', finished: 'good', stopped: 'muted', failed: 'bad'};
const RUN_ORDER = {running: 0, pending: 1, stopped: 2, failed: 3, finished: 4};
// A batch run's round that never will be done (v1.6: the run ended) reads "unfinished".
const ROUND_TONES = {done: 'good', probing: 'pending', building: 'pending', pending: 'muted', unfinished: 'pending'};
// CONTRACT "v1.6": a batch from another tree (e.g. Baihe's) is never audited: "<name> batch · not audited".
const BATCH_HELP = 'not audited: no failure-match audit runs on this batch (another tree\'s tasks, shown as built there)';
const BATCH_DETAIL = /^run\/[A-Za-z0-9][A-Za-z0-9_-]*\/[A-Za-z0-9][A-Za-z0-9._-]*\.json$/;   // runs[].detail
const OFFICIAL_DETAIL = /^run\/hengyuf\/[A-Za-z0-9][A-Za-z0-9._-]*\.json$/;   // v1.8: an older official run's runs[].detail
// v1.8, a synthesis queue's run statuses (run_queue.py) and their tones; "queued": not launched yet
const QUEUE_TONES = {queued: 'muted', pending: 'pending', running: 'evolve', waiting_to_resume: 'pending', done: 'good',
                     failed: 'bad', stuck: 'serious'};
const QUEUE_ORDER = ['running', 'pending', 'waiting_to_resume', 'queued', 'done', 'failed', 'stuck'];
const QUEUE_STALE_S = 900;   // run_queue.py HEARTBEAT_STALE_S: an older heartbeat means no live runner
const WORKER_TONES = {ready: 'good', draining: 'pending', starting: 'evolve', pending: 'muted', held: 'pending',
                      stale: 'serious', unhealthy: 'bad', failed: 'bad', stopped: 'muted', vanished: 'muted'};
const NODE_STATUS_TEXT = {planning: 'planning…', authoring: 'authoring…', building: 'building…',
                          built: 'built · waiting for probe', probing: 'probing', scored: 'scored',
                          build_failed: 'build failed', abandoned: 'abandoned'};
// Generation-tree geometry (px). Cards are absolutely positioned; their width follows the space the page measured
// (ui.treeWidth) within [minW, maxW], so wide screens show whole titles and 1200 px still fits four rounds. A card
// holds five 16 px lines: name + badge, title, F2P/P2P (+ target pattern), sc14, kimi.
const TREE = {cardW: 240, minW: 220, maxW: 360, cardH: 96, colGap: 40, rowGap: 8};
// The node drawer: width set by dragging its edge (or arrow keys on the grip), kept in localStorage.
const DRAWER = {min: 360, step: 40, keep: 360, storageKey: 'swe-task-dashboard.drawer-width'};

// ------------------------------------------------------------------------- safe accessors and formatting

function isObj(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }
function obj(v) { return isObj(v) ? v : {}; }
function arr(v) { return Array.isArray(v) ? v : []; }
function num(v) { return typeof v === 'number' && Number.isFinite(v) ? v : null; }
function int(v) { const n = num(v); return n !== null && Number.isInteger(n) ? n : null; }
function hasOwn(o, k) { return Object.prototype.hasOwnProperty.call(o, k); }
/** map[key] for a key that comes from data (never an inherited property such as 'constructor'). */
function lookup(map, key) { return typeof key === 'string' && hasOwn(map, key) ? map[key] : undefined; }

/** Any value as display text: null/undefined -> '', objects -> compact JSON. */
function text(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'string') return v;
  if (typeof v === 'object') {
    try { return JSON.stringify(v) || ''; } catch (error) { return '[unprintable]'; }
  }
  return String(v);
}

const ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;', '`': '&#96;'};
/** HTML-escape any value for element content and (double-quoted) attribute values. */
function esc(v) { return text(v).replace(/[&<>"'`]/g, (c) => ESCAPES[c]); }

function truncate(s, n) { const t = text(s); return t.length > n ? t.slice(0, Math.max(0, n - 1)) + '…' : t; }
function firstLine(s) { return text(s).split('\n').map((l) => l.trim()).find(Boolean) || ''; }
function plural(n, word) { return `${fmtInt(n)} ${word}${n === 1 ? '' : 's'}`; }
function cmpStr(a, b) { return text(a).localeCompare(text(b)); }
/** A [done, total] pair of numbers, else null. */
function pair(v) { return Array.isArray(v) && num(v[0]) !== null && num(v[1]) !== null ? [v[0], v[1]] : null; }

function fmtInt(v) { const n = num(v); return n === null ? DASH : Math.round(n).toLocaleString('en-US'); }
function fmtFixed(v, digits) {
  const n = num(v);
  return n === null ? DASH : n.toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
}
/** 1234 -> 1,234; 12345 -> 12.3K; 2.5e6 -> 2.5M; 88.14 -> 88.1; 0.096 -> 0.10 */
function fmtCompact(v) {
  const n = num(v);
  if (n === null) return DASH;
  const a = Math.abs(n);
  const short = (x, digits) => x.toFixed(digits).replace(/\.0$/, '');
  if (a >= 1e9) return short(n / 1e9, a >= 1e10 ? 0 : 1) + 'B';
  if (a >= 1e6) return short(n / 1e6, a >= 1e7 ? 0 : 1) + 'M';
  if (a >= 1e4) return short(n / 1e3, a >= 1e5 ? 0 : 1) + 'K';
  if (a >= 100 || Number.isInteger(n)) return Math.round(n).toLocaleString('en-US');
  return n.toFixed(a >= 1 ? 1 : 2);
}
/** A ratio in [0, 1] as a percentage. */
function fmtPct(v, digits) { const n = num(v); return n === null ? DASH : `${(n * 100).toFixed(digits || 0)}%`; }
/** 0 -> 0 B; 1536 -> 1.5 KB; 12.3e6 -> 12 MB */
function fmtBytes(v) {
  const n = num(v);
  if (n === null) return DASH;
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let x = Math.max(0, n), i = 0;
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i += 1; }
  return `${i === 0 || x >= 10 ? Math.round(x) : x.toFixed(1)} ${units[i]}`;
}
function fmtMoney(v) {
  const n = num(v);
  if (n === null) return DASH;
  const s = Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2});
  return (n < 0 ? '-$' : '$') + s;
}
/** Seconds as a short duration: 45s, 13m, 3h 13m, 2d 4h. */
function fmtDur(v) {
  const n = num(v);
  if (n === null) return DASH;
  const s = Math.max(0, Math.round(n));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) {
    const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
    return m ? `${h}h ${m}m` : `${h}h`;
  }
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
  return h ? `${d}d ${h}h` : `${d}d`;
}
function fmtAgo(t, now) {
  const a = num(t), b = num(now);
  if (a === null || b === null) return DASH;
  if (a > b + 5) return `in ${fmtDur(a - b)}`;
  return `${fmtDur(b - a)} ago`;
}
function pad2(n) { return String(n).padStart(2, '0'); }
/** Epoch seconds -> local HH:MM[:SS]. */
function fmtClock(t, seconds) {
  const d = dateOf(t);
  if (!d) return DASH;
  return `${pad2(d.getHours())}:${pad2(d.getMinutes())}` + (seconds ? `:${pad2(d.getSeconds())}` : '');
}
/** Epoch seconds -> local YYYY-MM-DD HH:MM. */
function fmtDate(t) {
  const d = dateOf(t);
  return d ? `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${fmtClock(t)}` : DASH;
}
function fmtDay(t) { const d = dateOf(t); return d ? `${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}` : DASH; }
/** Epoch seconds -> Date, or null (not a number, or outside the range a Date can hold). */
function dateOf(t) {
  const n = num(t);
  const d = n === null ? null : new Date(n * 1000);
  return d && Number.isFinite(d.getTime()) ? d : null;
}
function isoToEpoch(s) { const ms = typeof s === 'string' ? Date.parse(s) : NaN; return Number.isFinite(ms) ? ms / 1000 : null; }

// ------------------------------------------------------------------------------------------- UI atoms

function toneOf(t) { return TONES.has(t) ? t : 'neutral'; }

/** Status badge: a colored dot plus the label (never color alone). `extra` is a code constant class. */
function badge(label, tone, title, extra) {
  return `<span class="badge tone-${toneOf(tone)}${extra ? ' ' + extra : ''}"` +
         `${title ? ` title="${esc(title)}"` : ''}><span class="dot" aria-hidden="true"></span>${esc(label)}</span>`;
}

/** Collapsible block whose open state survives re-renders: ui.openState[key] (set by the wiring) wins over `open`. */
function det(ui, key, summaryHtml, bodyHtml, open, cls) {
  const state = obj(obj(ui).openState);
  const isOpen = hasOwn(state, key) ? !!state[key] : !!open;
  return `<details class="${cls || 'fold'}" data-key="${esc(key)}"${isOpen ? ' open' : ''}>` +
         `<summary>${summaryHtml}</summary><div class="fold-body">${bodyHtml}</div></details>`;
}

/** data-scroll marks a scrollable element whose scroll offsets survive re-renders (keyed by a stable data id). */
function scrollAttr(key) { return key ? ` data-scroll="${esc(key)}"` : ''; }

/** Preformatted text; `key` keeps its scroll position across re-renders. */
function pre(value, cls, key) {
  const t = text(value);
  return t ? `<pre class="pre${cls ? ' ' + cls : ''}"${scrollAttr(key)}>${esc(t)}</pre>` : '<p class="muted small">(empty)</p>';
}

function section(title, body, headExtra, cls) {
  return `<section class="panel${cls ? ' ' + cls : ''}"><div class="panel-head"><h2>${esc(title)}</h2>` +
         `${headExtra || ''}</div>${body}</section>`;
}

function note(msg, cls) { return `<p class="note${cls ? ' ' + cls : ''}">${esc(msg)}</p>`; }

/** Key/value grid; values are HTML that the caller escaped. */
function kv(rows) {
  return `<dl class="kv">${rows.filter(Boolean).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join('')}</dl>`;
}

/** A section renderer that fails renders an inline error instead of blanking the page. */
function safe(label, fn) {
  try {
    const out = fn();
    return typeof out === 'string' ? out : '';
  } catch (error) {
    return `<div class="render-error">Could not render ${esc(label)}: ${esc(error && error.message)}</div>`;
  }
}

/** Thin progress bar (done of total). `extra` is a code constant class. */
function bar(done, total, tone, label, extra) {
  const d = num(done), t = num(total);
  const pct = d !== null && t ? Math.max(0, Math.min(100, (d / t) * 100)) : 0;
  return `<span class="bar tone-${toneOf(tone)}${extra ? ' ' + extra : ''}" role="progressbar" aria-valuemin="0" aria-valuemax="${t === null ? 0 : t}"` +
         ` aria-valuenow="${d === null ? 0 : d}"${label ? ` aria-label="${esc(label)}"` : ''}>` +
         `<span class="bar-fill" style="width:${pct.toFixed(1)}%"></span></span>`;
}

/** Stat tile: label, value, optional sub line; tone colors the top rule. */
function tile(label, valueHtml, subHtml, tone) {
  return `<div class="tile tone-${toneOf(tone)}"><div class="tile-label">${esc(label)}</div>` +
         `<div class="tile-value">${valueHtml}</div>${subHtml ? `<div class="tile-sub">${subHtml}</div>` : ''}</div>`;
}

// -------------------------------------------------------------------------------------- index lookups

function isOfficial(run) { return typeof run === 'string' && /^official/.test(run); }

function indexTasks(index) {
  return arr(obj(index).tasks).filter((t) => isObj(t) && typeof t.task_id === 'string' && t.task_id);
}

function findTask(index, id) {
  if (!id) return null;
  return indexTasks(index).find((t) => t.task_id === id) || null;
}

/** The official run converting `task` (task.run, else a run entry naming the task), or null. */
function runOf(index, task) {
  const t = obj(task);
  const runs = arr(obj(index).runs).filter((r) => isObj(r) && isOfficial(r.run));
  if (isOfficial(t.run)) {
    const named = runs.find((r) => r.run === t.run);
    if (named) return named;
    return {run: t.run, task_id: t.task_id};             // listed on the task but missing from runs[]
  }
  return (t.task_id && runs.find((r) => r.task_id === t.task_id)) || null;
}

function byRank(a, b) {
  const ra = num(a.queue_rank), rb = num(b.queue_rank);
  return (ra === null ? 1e9 : ra) - (rb === null ? 1e9 : rb) || cmpStr(a.task_id, b.task_id);
}

// ------------------------------------------------------------------- batches from other trees (v1.6)

/** A runs[] entry of a batch (it names its source; hengyuf's official runs carry none). */
function isBatchRun(run) { const r = obj(run); return typeof r.source === 'string' && r.source !== '' && r.source !== 'hengyuf'; }

/** A node, run or document that says it is not audited (NODE.audit / runs[].audit / doc.audit {status: not_audited}). */
function notAudited(x) { return obj(obj(x).audit).status === 'not_audited'; }

/** The batch runs converting `task`, in its batch_runs order (oldest launch first); missing entries are skipped. */
function batchRunsOf(index, task) {
  const runs = arr(obj(index).runs).filter(isBatchRun);
  return arr(obj(task).batch_runs).filter(isObj)
    .map((b) => runs.find((r) => r.source === b.source && r.run === b.run && r.task_id === obj(task).task_id))
    .filter(Boolean);
}

/** index.json batches[] entry of a source ({} when absent). */
function batchInfo(index, source) { return arr(obj(index).batches).find((b) => isObj(b) && b.source === source) || {}; }

/** A batch's short name ("Baihe"): batches[].name, the run's or document's source_name, else its source. */
function batchName(index, x) {
  const s = obj(x);
  return text(batchInfo(index, s.source).name) || text(s.source_name) || text(s.source) || 'batch';
}

/** The "<name> batch · not audited" badge of a batch run, task or document (`extra`: code constant classes). */
function batchBadge(name, extra) {
  return badge(`${text(name) || 'batch'} batch · not audited`, 'muted', `${text(name) || 'This'} batch: ${BATCH_HELP}`, extra);
}

/** v1.8: the official runs[] entries converting `task`: its latest (runOf) first, then its older official runs in
    tasks[].official_runs order (newest launch first). An older collector's index (no official_runs): the latest alone. */
function officialRunsOf(index, task) {
  const t = obj(task), latest = runOf(index, t);
  if (!latest) return [];
  const runs = arr(obj(index).runs).filter((r) => isObj(r) && isOfficial(r.run) && !isBatchRun(r));
  const older = arr(t.official_runs).filter((o) => isObj(o) && o.latest === false && text(o.run) && o.run !== latest.run)
    .map((o) => runs.find((r) => r.run === o.run && r.task_id === t.task_id)).filter(Boolean);
  return [latest].concat(older);
}

/** What the task view can show for `task`: its official run first (when it has one; v1.8: its latest, then its older
    official runs, marked `older`), then its batch runs. */
function viewsOf(index, task) {
  return officialRunsOf(index, task).map((run, i) => (i ? {kind: 'official', run, older: true} : {kind: 'official', run}))
    .concat(batchRunsOf(index, task).map((run) => ({kind: 'batch', run})));
}

/** The view a #task=..&run=<run> hash selects among `views` (a run's name: a batch run's, v1.8 an older or the latest
    official run's; none or unknown: the first view). */
function selectedView(views, runName) {
  const list = arr(views);
  return (runName && list.find((v) => obj(v.run).run === runName)) || list[0] || null;
}

/** Where a view's document is served (official: data/task/<task>.json; v1.8 an older official run and a batch run:
    data/<runs[].detail>), and its key in the page's cache of documents. Null for a run without a well-formed detail
    path. */
function viewDetail(view, taskId) {
  const v = obj(view);
  if (v.kind !== 'batch' && !v.older) return {key: text(taskId), path: `data/task/${encodeURIComponent(text(taskId))}.json`};
  const detail = text(obj(v.run).detail);
  if (v.older) return OFFICIAL_DETAIL.test(detail) ? {key: `official:${text(v.run.run)}`, path: `data/${detail}`} : null;
  return BATCH_DETAIL.test(detail) ? {key: `batch:${text(v.run.source)}/${text(v.run.run)}`, path: `data/${detail}`} : null;
}

/** v1.8: the synthesis queues (index.json queues) that list `taskId` as queued and not launched yet (no runs[] entry
    names the run): [{queue, run}]. */
function queuedIn(index, taskId) {
  const launched = new Set(arr(obj(index).runs).filter(isObj).map((r) => r.run));
  const out = [];
  for (const q of arr(obj(index).queues).filter(isObj)) {
    for (const r of arr(q.runs).filter(isObj)) {
      if (r.task === taskId && r.status === 'queued' && !launched.has(r.run)) out.push({queue: text(q.name), run: text(r.run)});
    }
  }
  return out;
}

/** The task shown without a #task= hash: the first converting task (running first, newest submission first), else
    the queue head, else the first task. */
function defaultTask(index) {
  const groups = taskGroups(index, '');
  const first = (key) => (groups.find((g) => g.key === key) || {items: []}).items[0];
  return (first('converting') || first('eligible') || first('all') || {}).task_id || null;
}

function findNode(detail, id) {
  if (!id) return null;
  return arr(obj(detail).nodes).find((n) => isObj(n) && n.id === id) || null;
}

function shortId(id) { return text(id).replace(/^cand-/, ''); }

/** A card's name inside its lineage block: c01 for the round-0 node, r1b2-r2b0 below it (the block names c01). */
function nodeLabel(node) {
  const n = obj(node), id = text(n.id), lin = text(n.lineage);
  return lin && id !== lin && id.startsWith(`${lin}-`) ? id.slice(lin.length + 1) : shortId(id);
}

/** Badge info for a node: BUILD FAILED for a discard whose build failed; unknown decisions show as-is. */
function decisionOf(node) {
  const n = obj(node);
  const d = typeof n.decision === 'string' ? n.decision : '';
  if (d === 'discard' && (obj(n.build).ok === false || n.status === 'build_failed')) return DECISIONS.build_failed;
  if (hasOwn(DECISIONS, d) && d !== 'build_failed') return DECISIONS[d];
  return {key: 'other', label: d ? d.replace(/_/g, ' ').toUpperCase() : 'UNKNOWN', tone: 'neutral',
          help: 'decision not recognized by this page'};
}

/** NODE.seed_invalid (v1.5) as a secondary note: why the scored task may not seed evolution, when its decision does not
    say it already (an accepted task stays accepted: seed validity only governs evolution). '' for a discard, whose
    reason it is, for a valid seed, and for collectors that predate the field. */
function seedNote(node) {
  const n = obj(node);
  const note = typeof n.seed_invalid === 'string' ? n.seed_invalid.trim() : '';
  return note && n.decision !== 'discard' ? note : '';
}

/** The small "seed-invalid" badge that follows a node's decision badge (tooltip: the note); '' without a note. */
function seedBadge(node, extra) {
  const note = seedNote(node);
  return note ? badge(SEED_NOTE.label, SEED_NOTE.tone, `seed-invalid: ${note}\n${SEED_NOTE.help}`, extra) : '';
}

/** Where a node's seed note comes from: its round's seeds.json, else the collector's recomputed seed_check(). */
function seedSource(node, detail) {
  const n = obj(node), k = int(n.round);
  const seeds = obj(obj(k === null ? null : roundAt(detail, k)).seeds);
  return hasOwn(obj(seeds.invalid), text(n.id)) ? `round-${k}/seeds.json`
    : 'seed_check() recomputed (no seeds.json verdict yet)';
}

// ------------------------------------------------------------------- failure-match audit (v1.3, v1.4)

function isAccepted(node) { const d = obj(node).decision; return d === 'accepted' || d === 'accepted_evolve'; }

/** Whether the collector marks the audit's scope on this node (NODE.audit_eligible, v1.4 collectors). */
function marksScope(node) { return hasOwn(obj(node), 'audit_eligible'); }

/** Is the node's task in the failure-match audit's scope? NODE.audit_eligible (v1.4: built and probed, a graded sc14
    rollout, not discarded or withdrawn); for a collector that predates it, an accepted task (v1.3). */
function inAuditScope(node) {
  const n = obj(node);
  return marksScope(n) ? n.audit_eligible === true : isAccepted(n);
}

/** NODE.withdrawn (v1.4) as text parts: "by main", when, why; [] for a task not withdrawn. */
function withdrawnParts(w) {
  if (!isObj(w)) return [];
  const at = isoToEpoch(w.at);
  const parts = [text(w.by) ? `by ${text(w.by)}` : '', at === null ? text(w.at) : fmtDate(at), text(w.why)].filter(Boolean);
  return parts.length ? parts : ['its task dir was moved to runs/<run>/withdrawn/'];
}

/** Why a node the collector leaves out of the audit's scope (audit_eligible false) is out, from its other fields. */
function outOfScopeWhy(node) {
  const n = obj(node);
  if (isObj(n.withdrawn)) return `withdrawn (${withdrawnParts(n.withdrawn).join(' · ')})`;
  if (n.decision === 'discard') return `discarded${text(n.decision_reason) ? `: ${text(n.decision_reason)}` : ''}`;
  if (!isObj(n.score)) return 'not scored yet';
  return 'no graded sc14 rollout, or an abandoned build';
}

/** The task file's nodes: objects with a string id (the first of duplicate ids). */
function nodesOf(detail) {
  const seen = new Set();
  return arr(obj(detail).nodes).filter((n) => isObj(n) && typeof n.id === 'string' && n.id && !seen.has(n.id) && seen.add(n.id));
}

/** NODE.failure_match: the audit object; null when the collector found no audit of the node's task; undefined from a
    collector that predates the audit (no such key: the page then shows nothing about it). */
function failureMatch(node) {
  const n = obj(node);
  if (!hasOwn(n, 'failure_match')) return undefined;
  return isObj(n.failure_match) ? n.failure_match : null;
}

/** {key, label, tone, help} of an audit's verdict; "not audited" without an audit. */
function faVerdict(fm) {
  if (!isObj(fm)) return FA_NOT_AUDITED;
  return lookup(FA_VERDICTS, text(fm.verdict).trim().toLowerCase()) || FA_UNKNOWN;
}

/** The "FA match: <verdict>" badge; its tooltip explains the verdict and carries the audit's summary. `extra`: code
    constant classes. */
function faBadge(fm, extra) {
  const v = faVerdict(fm), summary = isObj(fm) ? text(fm.summary) : '';
  return badge(`FA match: ${v.label}`, v.tone, summary ? `${v.help}.\n\n${summary}` : v.help, extra);
}

/** A run's audit coverage {accepted, audited, strong, partial, weak, none}: index runs[].audit, else counted from the
    task file's nodes; null when neither carries the audit (a collector that predates it). */
function auditCounts(run, detail) {
  const a = obj(obj(run).audit);
  const out = {accepted: 0, audited: 0, strong: 0, partial: 0, weak: 0, none: 0};
  if (int(a.accepted) !== null && int(a.audited) !== null) {
    for (const k of Object.keys(out)) out[k] = Math.max(0, int(a[k]) || 0);
    return out;
  }
  const nodes = nodesOf(detail);
  if (!nodes.some((n) => hasOwn(n, 'failure_match'))) return null;
  for (const n of nodes.filter(isAccepted)) {
    const fm = failureMatch(n), k = faVerdict(fm).key;
    out.accepted += 1;
    if (fm) out.audited += 1;
    if (FA_ORDER.includes(k)) out[k] += 1;
  }
  return out;
}

/** The audit's coverage of a run's eligible tasks (v1.4) {eligible, audited, strong, partial, weak, none}: index
    runs[].audit (eligible, audited_eligible, eligible_verdicts), else counted from the task file's nodes
    (audit_eligible); null when neither has it (a collector that predates v1.4). */
function eligibleCounts(run, detail) {
  const a = obj(obj(run).audit);
  const out = {eligible: 0, audited: 0, strong: 0, partial: 0, weak: 0, none: 0};
  if (int(a.eligible) !== null && int(a.audited_eligible) !== null) {
    const v = obj(a.eligible_verdicts);
    out.eligible = Math.max(0, a.eligible);
    out.audited = Math.max(0, a.audited_eligible);
    for (const k of FA_ORDER) out[k] = Math.max(0, int(v[k]) || 0);
    return out;
  }
  const nodes = nodesOf(detail);
  if (!nodes.some(marksScope)) return null;
  for (const n of nodes.filter((x) => x.audit_eligible === true)) {
    const fm = failureMatch(n), k = faVerdict(fm).key;
    out.eligible += 1;
    if (fm) out.audited += 1;
    if (FA_ORDER.includes(k)) out[k] += 1;
  }
  return out;
}

/** "2 strong · 1 partial · 3 not audited": the verdicts that occur, then the tasks without an audit (of `eligible`
    for eligibleCounts(), of `accepted` for auditCounts()). */
function auditBreakdown(c) {
  const x = obj(c);
  const parts = FA_ORDER.filter((k) => num(x[k])).map((k) => `${fmtInt(x[k])} ${k}`);
  const total = num(x.eligible) !== null ? x.eligible : num(x.accepted) || 0;
  const missing = total - (num(x.audited) || 0);
  if (missing > 0) parts.push(`${fmtInt(missing)} not audited`);
  return parts.join(' · ');
}

/** The run bar's "Failure-match audit" cell: "audit n/m" over the eligible tasks with their verdicts, then the same for
    the accepted tasks (v1.4); "audit n/m" over the accepted tasks for v1.3 collectors; '' without audit data. */
function auditCell(scope, accepted) {
  const acc = accepted ? `audit ${fmtInt(accepted.audited)}/${fmtInt(accepted.accepted)}` : '';
  const accBreak = accepted ? auditBreakdown(accepted) : '';
  if (scope) {
    const b = auditBreakdown(scope);
    return `<b title="${esc('eligible tasks with a failure-match audit / eligible tasks (built and probed, a graded sc14 ' +
                            'rollout, not discarded or withdrawn)')}">audit ${fmtInt(scope.audited)}/${fmtInt(scope.eligible)}</b>` +
           (b ? `<div class="muted small">${esc(b)}</div>` : '') +
           (accepted ? `<div class="muted small">${esc(`accepted: ${acc}${accBreak ? ` · ${accBreak}` : ''}`)}</div>` : '');
  }
  if (!accepted) return '';
  return `<b title="${esc('accepted tasks with a failure-match audit / accepted tasks')}">${acc}</b>` +
         (accBreak ? `<div class="muted small">${esc(accBreak)}</div>` : '');
}

/** A target pattern's name, numbered "P2 · <name>" like the failure-pattern cards when it is one of the target's. */
function patternLabel(name, ctx) {
  const t = text(name), map = obj(ctx).focus;
  const i = t && map instanceof Map ? map.get(t) : undefined;
  return i ? `P${i} · ${t}` : t;
}

function reproducedBadge(value) {
  const r = text(value).trim().toLowerCase();
  return hasOwn(REPRODUCED_TONES, r) ? badge(r, REPRODUCED_TONES[r], null, 'badge-sm') : DASH;
}

// ------------------------------------------------------------------------------- task picker (header)

/** Picker groups (CONTRACT "UI"): Converting (has an official run), then (v1.6) one group per batch, "<name> batch ·
    not audited" (the tasks its runs converted), Eligible queue (by rank), All tasks. Each group: {key, label, items}. */
function taskGroups(index, query) {
  const tasks = indexTasks(index);
  const q = text(query).trim().toLowerCase();
  const runFor = (t) => runOf(index, t);
  const batchFor = (t) => batchRunsOf(index, t);
  const match = (t) => !q || [t.task_id, t.run, obj(runFor(t)).repo, obj(runFor(t)).run,
                              ...batchFor(t).flatMap((r) => [r.run, r.repo]),
                              ...officialRunsOf(index, t).flatMap((r) => [r.run, r.repo]),   // v1.8: older official runs
                              ...queuedIn(index, t.task_id).map((x) => x.run)]              // v1.8: queued runs
    .some((s) => text(s).toLowerCase().includes(q));
  const runRank = (t) => {
    const s = text(obj(runFor(t)).status);
    return hasOwn(RUN_ORDER, s) ? RUN_ORDER[s] : 9;
  };
  const submitted = (t) => num(obj(runFor(t)).submitted_at) || 0;
  const converting = tasks.filter((t) => runFor(t))     // running first, then the newest submission
    .sort((a, b) => runRank(a) - runRank(b) || submitted(b) - submitted(a) || cmpStr(obj(runFor(a)).run, obj(runFor(b)).run));
  const eligible = tasks.filter((t) => t.eligible === true).sort(byRank);
  const byIndex = (a, b) => {
    const ia = num(a.task_index), ib = num(b.task_index);
    return (ia === null ? 1e9 : ia) - (ib === null ? 1e9 : ib) || cmpStr(a.task_id, b.task_id);
  };
  const all = tasks.slice().sort(byIndex);
  // v1.6: one group per batch source (index.json batches order, then any other source its runs name)
  const sources = [...new Set([...arr(obj(index).batches).filter(isObj).map((b) => text(b.source)),
                               ...arr(obj(index).runs).filter(isBatchRun).map((r) => text(r.source))])].filter(Boolean);
  const batches = sources.map((source) => {
    const items = tasks.filter((t) => batchFor(t).some((r) => r.source === source)).sort(byIndex);
    const info = batchInfo(index, source);
    return {key: `batch:${source}`, source, label: `${batchName(index, {source})} batch · not audited`,
            title: `${text(info.label) || source}: ${fmtInt(info.tasks)} tasks (outputs/tasks.jsonl) from ` +
                   `${fmtInt(info.runs)} runs on ${fmtInt(info.targets)} DeepSWE tasks; ${BATCH_HELP}`,
            items: items.filter(match)};
  }).filter((g) => tasks.some((t) => batchFor(t).some((r) => r.source === g.source)));
  return [{key: 'converting', label: 'Converting', items: converting.filter(match)}, ...batches,
          {key: 'eligible', label: 'Eligible queue', items: eligible.filter(match)},
          {key: 'all', label: 'All tasks', items: all.filter(match)}];
}

function accMini(task, model) {
  const a = obj(obj(obj(task).accuracy)[model]);
  if (num(a.passed) === null) return `${model} ${DASH}`;
  return `${model} ${fmtInt(a.passed)}/${fmtInt(a.total)}`;
}

/** The picker's dropdown: grouped options with data-task (click/Enter picks) and data-opt (keyboard index). */
function renderTaskMenu(index, query, selectedId, active) {
  const groups = taskGroups(index, query);
  let i = 0;
  const html = groups.map((g) => {
    if (!g.items.length) return '';
    const opts = g.items.map((t) => {
      const run = runOf(index, t);
      const status = text(obj(run).status);
      const tags = [];
      if (g.source) {                                     // v1.6, a batch group: its runs of this task, "<n> in batch"
        const mine = batchRunsOf(index, t).filter((r) => r.source === g.source);
        const one = mine.length === 1 ? text(mine[0].status) : '';
        tags.push(mine.length === 1 ? badge(one || 'run', lookup(RUN_TONES, one), text(mine[0].stage), 'badge-sm')
                                    : badge(`${mine.length} runs`, 'evolve', mine.map((r) => text(r.run)).join('\n'), 'badge-sm'));
        tags.push(`<span class="rank">${esc(`${fmtInt(mine.reduce((s, r) => s + (num(r.batch) || 0), 0))} in batch`)}</span>`);
      } else if (run) tags.push(badge(status || 'run', lookup(RUN_TONES, status), text(obj(run).stage), 'badge-sm'));
      if (t.eligible === true && num(t.queue_rank) !== null) tags.push(`<span class="rank">#${fmtInt(t.queue_rank)}</span>`);
      const queued = queuedIn(index, t.task_id);                // v1.8: queued in a synthesis queue, not launched yet
      if (queued.length) {
        tags.push(`<span class="opt-queued" title="${esc(queued.map((x) => `${x.run}: queued in ${x.queue}, not launched yet`).join('\n'))}">` +
                  `${esc(`queued in ${[...new Set(queued.map((x) => x.queue))].join(', ')}`)}</span>`);
      }
      const cls = `opt${t.task_id === selectedId ? ' selected' : ''}${i === active ? ' active' : ''}`;
      const out = `<div class="${cls}" role="option" aria-selected="${t.task_id === selectedId}" data-task="${esc(t.task_id)}"` +
                  ` data-opt="${i}"><span class="opt-id">${esc(t.task_id)}</span><span class="opt-meta">${tags.join('')}` +
                  `<span class="opt-acc">${esc(accMini(t, STUDENT))} · ${esc(accMini(t, TEACHER))}</span></span></div>`;
      i += 1;
      return out;
    }).join('');
    return `<div class="menu-group"><div class="menu-head"${g.title ? ` title="${esc(g.title)}"` : ''}>${esc(g.label)}` +
           `<span>${g.items.length}</span></div>${opts}</div>`;
  }).join('');
  return html || `<div class="menu-empty">No task matches “${esc(query)}”.</div>`;
}

function menuCount(index, query) { return taskGroups(index, query).reduce((s, g) => s + g.items.length, 0); }

/** Task id of the n-th option (keyboard navigation), or null. */
function menuOption(index, query, n) {
  let i = 0;
  for (const g of taskGroups(index, query)) {
    for (const t of g.items) {
      if (i === n) return t.task_id;
      i += 1;
    }
  }
  return null;
}

// ------------------------------------------------------------------------------------- header status

/** Header status: "updated hh:mm:ss", the collector cycle the page shows, and its age; a badge while a newer cycle
    waits because text is selected; the staleness badge once the newest index.json is more than 5 min old.
    info (optional): {shown: generated_at of the cycle on screen, held: a newer cycle waits}. */
function renderStatus(index, now, info) {
  const i = obj(info);
  const g = num(obj(index).generated_at);
  if (g === null) return '<span class="updated muted">no data yet</span>';
  const shown = num(i.shown) !== null && i.held ? i.shown : g;
  const age = num(now) === null ? 0 : now - g;
  const stale = age > STALE_S;
  const tip = `Data of the collector cycle of ${fmtDate(shown)}${shown === g && text(obj(index).generated_at_iso) ? ` (${text(index.generated_at_iso)})` : ''}. ` +
              'The page checks every 60 s and re-renders only when a new cycle has landed.';
  return `<span class="updated${stale ? ' is-stale' : ''}" title="${esc(tip)}">updated ${esc(fmtClock(shown, true))}` +
         `<span class="age"> · ${esc(fmtAgo(shown, now))}</span></span>` +
         (i.held ? badge('newer data held', 'pending', `Cycle ${fmtClock(g, true)} is loaded, but text is selected on the page: ` +
                         'the view re-renders once the selection is cleared (checked every 15 s).') : '') +
         (stale ? badge(`stale ${fmtDur(age)}`, 'bad', 'index.json is more than 5 min old: is the collector running?') : '');
}

/** Warnings under the header: staleness, failed fetches, and the collector's non-fatal errors. */
function renderBanner(index, now, info) {
  const i = obj(info), ix = obj(index), out = [];
  const g = num(ix.generated_at);
  if (g !== null && num(now) !== null && now - g > STALE_S) {
    out.push(`<div class="alert alert-bad"><b>Stale data.</b> The collector last wrote index.json ${esc(fmtAgo(g, now))}` +
             ` (${esc(fmtDate(g))}); everything below may be out of date.</div>`);
  }
  for (const [src, err] of Object.entries(obj(i.errors))) {
    if (err) out.push(`<div class="alert alert-warn"><b>Refresh failed (${esc(src)}):</b> ${esc(err)}. Showing the last data received.</div>`);
  }
  const errors = arr(ix.errors).map(text).filter(Boolean);
  if (errors.length) {
    const list = `<ul class="errlist">${errors.map((e) => `<li>${esc(e)}</li>`).join('')}</ul>`;
    out.push(`<div class="alert alert-note">${det(i.ui, 'banner:collector-errors',
      `${esc(plural(errors.length, 'collector warning'))} (non-fatal)`, list, false, 'fold fold-inline')}</div>`);
  }
  return out.join('');
}

/** Shown on HTTP 403: the session cookie is missing or the server restarted with a new token. */
function renderAuthError() {
  return section('Not authorized',
    '<p class="note note-bad"><b>Missing or expired token: open the URL with ?token=...</b></p>' +
    note('The server prints the full URL when it starts (it is also stored in dashboard/state/server.json, mode 600). ' +
         'Opening it once sets a session cookie; the token is then removed from the address bar.'));
}

// ------------------------------------------------------------------------------------- task tab

function eligibilityBadges(task) {
  const t = obj(task);
  if (t.eligible === true) {
    return badge('ELIGIBLE', 'good', 'official-queue rule: sc14 ≤ 1 valid pass and kimi 5/5') +
           (num(t.queue_rank) !== null ? badge(`queue #${fmtInt(t.queue_rank)}`, 'evolve',
             'rank among eligible tasks by sc14 mean F2P ratio, descending') : '');
  }
  if (t.eligible === false) return badge('not eligible', 'muted', 'official-queue rule: sc14 ≤ 1 valid pass and kimi 5/5');
  return '';
}

/** One rollout chip: green pass, red fail, grey unscored/infra; labelled with its F2P fraction. An unscored or infra
    rollout shows a dash: its fraction ("0/0" from a verifier that never ran) only goes to the tooltip. */
function rolloutChip(r, i) {
  const x = obj(r);
  const score = num(x.score);
  const kind = x.status === 'infra' ? 'infra' : score === null ? 'unscored' : score >= 1 ? 'pass' : 'fail';
  const graded = kind === 'pass' || kind === 'fail';
  const tip = `${text(x.slot) || `rollout ${i + 1}`}: ${kind}` + (text(x.f2p) ? ` · F2P ${text(x.f2p)}` : '') +
              (text(x.p2p) ? ` · P2P ${text(x.p2p)}` : '');
  return `<span class="chip chip-${graded ? kind : 'unscored'}" title="${esc(tip)}">${graded ? esc(text(x.f2p) || '?') : DASH}</span>`;
}

/** Accuracy cards for kimi / super_vl / sc14 (shown whether or not the task has a run). */
function renderAccuracy(index, task) {
  const ix = obj(index), t = obj(task), models = obj(ix.models), acc = obj(t.accuracy);
  const keys = [...new Set([...MODEL_ORDER, ...Object.keys(models), ...Object.keys(acc)])];
  const cards = keys.map((k) => {
    const m = obj(models[k]), fb = MODEL_FALLBACK[k] || {};
    const label = text(m.label) || fb.label || k;
    const role = text(m.role) || fb.role || '';
    const a = isObj(acc[k]) ? acc[k] : null;
    let body;
    if (!a) {
      body = '<div class="acc-big acc-none">no data</div>';
    } else {
      const passed = num(a.passed), total = num(a.total);
      const rate = passed !== null && total ? passed / total : null;
      const chips = arr(a.rollouts).map(rolloutChip).join('');
      const extra = [];
      if (num(a.unscored)) extra.push(`${fmtInt(a.unscored)} unscored`);
      if (num(a.infra)) extra.push(`${fmtInt(a.infra)} infra`);
      body = `<div class="acc-big">${passed === null ? DASH : fmtInt(passed)}<span class="acc-den">/${total === null ? DASH : fmtInt(total)}</span>` +
             `${rate === null ? '' : `<span class="acc-rate">${fmtPct(rate)}</span>`}</div>` +
             `<div class="chips">${chips || '<span class="muted small">no rollouts listed</span>'}</div>` +
             (num(a.mean_f2p_ratio) !== null ? `<div class="acc-ratio">mean F2P ratio <b>${fmtFixed(a.mean_f2p_ratio, 2)}</b></div>` : '') +
             (extra.length ? `<div class="acc-extra">${esc(extra.join(' · '))}</div>` : '');
    }
    const src = text(m.source);
    return `<div class="acc-card${k === STUDENT ? ' acc-student' : ''}"><div class="acc-head"><span class="acc-label">${esc(label)}</span>` +
           `${role ? `<span class="acc-role">${esc(role)}</span>` : ''}</div>${body}` +
           `${src ? `<div class="acc-src" title="${esc(src)}">${esc(src)}</div>` : ''}</div>`;
  }).join('');
  const sub = [];
  if (num(t.task_index) !== null) sub.push(`task #${fmtInt(t.task_index)}`);
  if (num(t.f2p_total) !== null) sub.push(`${fmtInt(t.f2p_total)} F2P tests`);
  const head = `<div class="task-head"><h1 class="task-id">${esc(t.task_id)}</h1><div class="task-badges">${eligibilityBadges(t)}</div></div>` +
               (sub.length ? `<div class="task-sub">${esc(sub.join(' · '))}</div>` : '');
  return `<section class="panel panel-task">${head}<div class="acc-grid">${cards}</div></section>`;
}

function renderNoRun(task) {
  const t = obj(task);
  const queue = t.eligible === true
    ? `It is eligible for the official queue${num(t.queue_rank) !== null ? ` at rank #${fmtInt(t.queue_rank)}` : ''}.`
    : 'It is not in the eligible queue (needs sc14 ≤ 1 valid pass and kimi 5/5).';
  return section('Conversion', `<p class="note"><b>Not converted yet:</b> no official run targets this task. ${esc(queue)}</p>`);
}

/** Run status bar: run, repo, job state, stage, accepted count, the failure-match audit's coverage ("audit n/m" of the
    eligible tasks and of the accepted ones, v1.4; of the accepted ones, v1.3 collectors), and a collapsible log tail. */
function renderRunBar(run, detail, now, ui) {
  const r = obj(run), d = obj(detail), job = obj(d.job);
  const audit = auditCell(eligibleCounts(run, detail), auditCounts(run, detail));
  const status = text(r.status);
  const state = text(job.slurm_state || r.slurm_state);
  const started = num(r.started_at), submitted = num(r.submitted_at), ended = num(r.ended_at);
  const maxRounds = int(d.max_rounds);
  const slurm = [esc(job.job_id || r.job_id) || DASH, state ? badge(state, slurmTone(state), null, 'badge-sm') : '',
                 esc(job.partition || r.partition), esc(job.node)].filter(Boolean).join(' ');
  const elapsed = text(job.elapsed) || (started !== null ? fmtDur((ended === null ? now : ended) - started) : '');
  const left = text(job.time_left);
  const timing = [started !== null ? `started ${fmtAgo(started, now)}` : submitted !== null ? `submitted ${fmtAgo(submitted, now)}, not started` : '',
                  left ? `${left} left` : ''].filter(Boolean).join(', ');
  const cells = [
    ['Run', `<b>${esc(r.run || d.run) || DASH}</b>`],
    ['Repo', `<code>${esc(r.repo || d.repo) || DASH}</code>`],
    ['Status', status ? badge(status, lookup(RUN_TONES, status)) : DASH],
    ['Slurm job', slurm],
    ['Elapsed', `${esc(elapsed) || DASH}${timing ? ` <span class="muted small">(${esc(timing)})</span>` : ''}`],
    ['Accepted', `<span class="big-num">${fmtInt(r.accepted)}</span>`],
    audit ? ['Failure-match audit', audit] : null,
    ['Built / scored', `${fmtInt(r.tasks_built)} / ${fmtInt(r.tasks_scored)}`],
    ['Rounds started', `${fmtInt(r.rounds_started)}${maxRounds !== null ? ` / ${maxRounds + 1}` : ''}`],
  ].filter(Boolean).map(([k, v]) => `<div class="runcell"><div class="runcell-k">${esc(k)}</div><div class="runcell-v">${v}</div></div>`).join('');
  const stage = text(r.stage || job.stage);
  const lines = arr(job.log_tail).map(text);
  const log = det(ui, `log:${text(r.run)}`, `Log tail <span class="muted small">${esc(plural(lines.length, 'line'))}` +
                  `${text(job.log_path) ? ` · ${esc(job.log_path)}` : ''}</span>`,
                  lines.length ? pre(lines.join('\n'), 'pre-log', `log:${text(r.run)}`) : note('No log lines yet.'), false);
  return `<section class="panel panel-run"><div class="runbar">${cells}</div>` +
         `<div class="stage"><span class="stage-k">Stage</span> ${esc(stage) || '<span class="muted">unknown</span>'}</div>${log}</section>`;
}

function slurmTone(state) {
  const s = text(state).toUpperCase();
  if (s === 'RUNNING') return 'good';
  if (s === 'PENDING' || s === 'REQUEUED' || s === 'CONFIGURING' || s === 'RESIZING') return 'pending';
  if (s === 'COMPLETED' || s === 'COMPLETING') return 'muted';
  if (/FAIL|TIMEOUT|CANCEL|OUT_OF_MEMORY|PREEMPT|DEADLINE|REVOKED|STOPPED/.test(s)) return 'bad';
  return 'neutral';
}

/** Evolve branches per target pattern (NODE.focus_pattern; abandoned nodes left out): Map name -> {targeted, accepted,
    open}, open = neither scored nor failed to build yet. Empty when the collector does not record focus patterns. */
function focusTally(nodes) {
  const out = new Map();
  for (const n of arr(nodes)) {
    if (!isObj(n) || isAbandoned(n) || !text(n.focus_pattern)) continue;
    const k = text(n.focus_pattern), t = out.get(k) || {targeted: 0, accepted: 0, open: 0};
    t.targeted += 1;
    if (n.decision === 'accepted' || n.decision === 'accepted_evolve') t.accepted += 1;
    if (!isObj(n.score) && obj(n.build).ok !== false) t.open += 1;
    out.set(k, t);
  }
  return out;
}

function tallyText(t) {
  return `${fmtInt(t.targeted)} ${t.targeted === 1 ? 'branch' : 'branches'} targeted it · ${fmtInt(t.accepted)} accepted` +
         (t.open ? ` · ${fmtInt(t.open)} in progress` : '');
}

/** Target failure patterns (with the evolve branches aimed at each, when `nodes` carry focus_pattern), the transferable
    signature, pass stats and the full analysis (collapsible). */
function renderFailureAnalysis(fa, ui, nodes) {
  if (!isObj(fa)) return section('Target failure patterns', note('The target failure analysis is not available yet.'));
  const patterns = arr(fa.patterns).filter(isObj);
  const tally = focusTally(nodes);
  const cards = patterns.map((p, i) => {
    const slots = arr(p.student_slots).map((s) => `<span class="slot">${esc(s)}</span>`).join('');
    const contrast = text(p.teacher_contrast), evidence = text(p.evidence);
    const t = tally.get(text(p.name));
    const aimed = !tally.size ? '' : t
      ? `<div class="pattern-targets"><span class="tag tag-focus">P${i + 1}</span>${esc(tallyText(t))}</div>`
      : `<div class="pattern-targets muted"><span class="tag tag-focus">P${i + 1}</span>no evolve branch targeted it yet</div>`;
    return `<article class="pattern"><div class="pattern-head"><span class="pattern-num">${i + 1}</span>` +
           `<h3>${esc(p.name) || 'unnamed pattern'}</h3>${text(p.category) ? `<span class="tag">${esc(p.category)}</span>` : ''}</div>` +
           `<p class="pattern-desc">${esc(p.description)}</p>` +
           (slots ? `<div class="slots"><span class="muted small">student slots</span>${slots}</div>` : '') + aimed +
           (contrast ? det(ui, `fa:contrast:${i}`, 'Teacher contrast', `<p>${esc(contrast)}</p>`, false, 'fold fold-inline') : '') +
           (evidence ? det(ui, `fa:evidence:${i}`, 'Evidence', pre(evidence, '', `fa:evidence:${i}`), false, 'fold fold-inline') : '') +
           '</article>';
  }).join('');
  // Targets that are no pattern of this analysis ("overall", the fallback of task_builder.evolve, or a renamed one).
  const known = new Set(patterns.map((p) => text(p.name)));
  const others = [...tally].filter(([name]) => !known.has(name)).sort((a, b) => cmpStr(a[0], b[0]))
    .map(([name, t]) => `<li><span class="tag tag-focus">${esc(name)}</span> ${esc(tallyText(t))}</li>`).join('');
  const sig = text(fa.transferable_signature);
  const excluded = arr(fa.excluded);
  const stats = obj(fa.pass_stats);
  const statRows = Object.keys(stats).map((m) => {
    const s = obj(stats[m]);
    return `<tr><td>${esc(m)}</td><td class="num">${fmtInt(s.pass)}</td><td class="num">${fmtInt(s.fail)}</td><td class="num">${fmtPct(s.rate)}</td></tr>`;
  }).join('');
  const md = text(fa.markdown);
  const body =
    (cards ? `<div class="patterns">${cards}</div>` : note('No failure patterns were recorded.')) +
    (others ? `<div class="excluded"><span class="muted small">other evolve targets</span><ul class="plain">${others}</ul></div>` : '') +
    (sig ? `<div class="callout"><div class="callout-k">Transferable signature</div><p>${esc(sig)}</p></div>` : '') +
    (statRows ? `<table class="tbl tbl-compact"><thead><tr><th>Target rollouts</th><th class="num">pass</th><th class="num">fail</th><th class="num">rate</th></tr></thead><tbody>${statRows}</tbody></table>` : '') +
    (excluded.length ? `<div class="excluded"><span class="muted small">excluded rollouts</span><ul>${excluded.map((e) => `<li>${esc(e)}</li>`).join('')}</ul></div>` : '') +
    (md ? det(ui, 'fa:markdown', `Full failure_analysis.md <span class="muted small">${esc(fmtInt(md.length))} chars</span>`, pre(md, 'pre-md', 'fa:markdown'), false) : '');
  return section('Target failure patterns', body, `<span class="muted small">${esc(plural(patterns.length, 'pattern'))}</span>`);
}

/** The planner's candidates (built ones marked) and the mask-plan excerpt, collapsed by default. */
function renderPlan(plan, ui) {
  if (!isObj(plan)) return '';
  const cands = arr(plan.candidates).filter(isObj);
  const built = cands.filter((c) => c.built === true).length;
  const rows = cands.map((c) => {
    const files = arr(c.files).map(text);
    const tests = Array.isArray(c.tests) ? c.tests.length : num(c.tests);
    const targets = arr(c.target_patterns).map(text).filter(Boolean);
    return `<tr><td><code>${esc(c.id)}</code></td><td><div class="cand-title">${esc(c.title)}</div>` +
           `<div class="muted small">${esc(truncate(c.summary, 400))}</div>` +
           (targets.length ? `<div class="muted small">targets: ${esc(targets.join(' · '))}</div>` : '') + '</td>' +
           `<td class="num" title="${esc(files.join('\n'))}">${fmtInt(files.length)}</td><td class="num">${fmtInt(tests)}</td>` +
           `<td>${c.built === true ? badge('BUILT', 'good', null, 'badge-sm') : '<span class="muted small">not built</span>'}</td></tr>`;
  }).join('');
  const excerpt = text(plan.mask_plan_excerpt);
  const body = (rows ? `<table class="tbl"><thead><tr><th>id</th><th>candidate</th><th class="num">files</th><th class="num">tests</th><th>built</th></tr></thead><tbody>${rows}</tbody></table>`
                     : note('No candidates in the plan.')) +
               (excerpt ? det(ui, 'plan:excerpt', 'mask_plan.md (excerpt)', pre(excerpt, 'pre-md', 'plan:excerpt'), false, 'fold fold-inline') : '');
  return `<section class="panel panel-plan">${det(ui, 'plan', `<h2 class="inline-h2">Plan</h2><span class="muted small">` +
    `${esc(plural(cands.length, 'candidate'))} · ${fmtInt(built)} built</span>`, body, false, 'fold fold-section')}</section>`;
}

// ------------------------------------------------------------------------------------ generation tree

function isAbandoned(n) { const x = obj(n); return x.decision === 'abandoned' || x.status === 'abandoned'; }

/** Sum of [done, total] over models, or null. */
function progressOf(node) {
  const prog = obj(obj(obj(node).probe).progress);
  let done = 0, total = 0, any = false;
  for (const v of Object.values(prog)) {
    const p = pair(v);
    if (p) { done += p[0]; total += p[1]; any = true; }
  }
  return any ? {done, total} : null;
}

/** {pass, fail, infra, clean} of the student or teacher: the probe summary, else counted from rollouts. */
function probeStats(probe, role) {
  const p = obj(probe);
  const s = obj(role === 'student' ? p.student : p.teacher);
  const model = role === 'student' ? STUDENT : TEACHER;
  let pass = num(s.pass), fail = num(s.fail), infra = num(s.infra), cheated = num(s.cheated);
  if (pass === null && fail === null) {
    const rs = arr(p.rollouts).filter((r) => isObj(r) && r.model === model);
    if (!rs.length) return null;
    pass = rs.filter((r) => r.status === 'pass').length;
    fail = rs.filter((r) => r.status === 'fail').length;
    infra = rs.filter((r) => r.status === 'infra').length;
  }
  pass = pass || 0; fail = fail || 0;
  let clean = pass;
  if (role === 'teacher') {
    if (cheated !== null) clean = Math.max(0, pass - cheated);
    else if (num(s.clean_rate) !== null) clean = Math.round(s.clean_rate * (pass + fail));
  }
  return {pass, fail, infra: infra || 0, cheated: cheated || 0, clean, valid: pass + fail};
}

/** A node's probe before the node is scored: per model {model, done, total, pass, fail, infra} over its graded slots
    (student, teacher, then any other model), their done/total sums, and ownDone once all of the node's own slots are
    done (its score comes when the round's whole probe ends). NODE.probe.partial / own_done when the collector sends
    them; else probe.progress with the so-far counts (older collectors), all slots done counting as ownDone. Null for a
    scored, abandoned or build-failed node, or one without probe progress. */
function liveProbe(node) {
  const n = obj(node), p = obj(n.probe);
  if (!isObj(n.probe) || isObj(n.score) || isAbandoned(n) || obj(n.build).ok === false) return null;
  const partial = obj(p.partial), progress = obj(p.progress);
  const models = [...new Set([STUDENT, TEACHER, ...Object.keys(partial), ...Object.keys(progress)])];
  const rows = [];
  for (const m of models) {
    const x = obj(lookup(partial, m)), pr = pair(lookup(progress, m));
    if (num(x.done) !== null && num(x.total) !== null) {
      rows.push({model: m, done: x.done, total: x.total, pass: num(x.pass) || 0, fail: num(x.fail) || 0, infra: num(x.infra) || 0});
    } else if (pr) {                                     // older collectors: the so-far counts of the probe
      let s;
      if (m === STUDENT || m === TEACHER) {
        s = probeStats(p, m === STUDENT ? 'student' : 'teacher');
      } else {
        const rs = arr(p.rollouts).filter((r) => isObj(r) && r.model === m);
        const k = (st) => rs.filter((r) => r.status === st).length;
        s = {pass: k('pass'), fail: k('fail'), infra: k('infra')};
      }
      rows.push({model: m, done: pr[0], total: pr[1], pass: s ? s.pass : 0, fail: s ? s.fail : 0, infra: s ? s.infra : 0});
    }
  }
  if (!rows.length) return null;
  const done = rows.reduce((a, r) => a + r.done, 0), total = rows.reduce((a, r) => a + r.total, 0);
  const ownDone = p.own_done === true || (!hasOwn(p, 'own_done') && total > 0 && done >= total);
  return {rows, done, total, ownDone};
}

/** "sc14 1 pass / 3 of 5 done" (", 1 infra" for slots the verifier could not grade). */
function partialText(row) {
  const r = obj(row);
  return `${text(r.model)} ${fmtInt(r.pass)} pass${num(r.infra) ? `, ${fmtInt(r.infra)} infra` : ''} / ${fmtInt(r.done)} of ${fmtInt(r.total)} done`;
}

/** "sc14 1/4 +1 infra": passes of the graded (pass + fail) slots. */
function passText(row) {
  const r = obj(row), pass = num(r.pass) || 0, fail = num(r.fail) || 0;
  return `${text(r.model)} ${fmtInt(pass)}/${fmtInt(pass + fail)}${num(r.infra) ? ` +${fmtInt(r.infra)} infra` : ''}`;
}

/** The target failure pattern of an evolve branch: {name, label}; label "P2 · <name>" when it is the 2nd pattern of
    the target's analysis (ctx.focus: Map name -> 1-based index). Null without NODE.focus_pattern (older collectors). */
function focusLabel(node, ctx) {
  const name = text(obj(node).focus_pattern);
  if (!name) return null;
  const map = obj(ctx).focus;
  const i = map instanceof Map ? map.get(name) : undefined;
  return {name, label: i ? `P${i} · ${name}` : name};
}

/** ctx for node cards and the drawer: the target's failure patterns by name (for "P<n>" focus tags). */
function treeContext(detail) {
  const focus = new Map();
  arr(obj(obj(detail).failure_analysis).patterns).forEach((p, i) => {
    if (isObj(p) && text(p.name) && !focus.has(text(p.name))) focus.set(text(p.name), i + 1);
  });
  return {focus};
}

/** Pure layout: one block per round-0 lineage; leaves stack in rows, parents center on their children; columns are
    rounds (a child sits one column right of its parent). Abandoned nodes are skipped unless ui.showAbandoned. */
function treeLayout(detail, ui) {
  const d = obj(detail), u = obj(ui);
  const byId = new Map();
  for (const n of arr(d.nodes)) {
    if (isObj(n) && typeof n.id === 'string' && n.id && !byId.has(n.id)) byId.set(n.id, n);
  }
  const kids = new Map();
  const link = (p, c) => {
    if (p === c || !byId.has(p) || !byId.has(c)) return;
    const list = kids.get(p) || [];
    if (!list.includes(c)) list.push(c);
    kids.set(p, list);
  };
  for (const n of byId.values()) for (const c of arr(n.children)) if (typeof c === 'string') link(n.id, c);
  for (const n of byId.values()) if (typeof n.parent === 'string') link(n.parent, n.id);
  const order = (a, b) => {
    const x = byId.get(a), y = byId.get(b);
    const bx = int(x.branch), by = int(y.branch);
    return (bx === null ? 99 : bx) - (by === null ? 99 : by) || cmpStr(a, b);
  };
  for (const list of kids.values()) list.sort(order);

  const hasParent = (n) => typeof n.parent === 'string' && n.parent !== n.id && byId.has(n.parent);
  const rootSort = (a, b) => {
    const ra = int(a.round), rb = int(b.round);
    return (ra === null ? 0 : ra) - (rb === null ? 0 : rb) || cmpStr(a.id, b.id);
  };
  const roots = [...byId.values()].filter((n) => !hasParent(n)).sort(rootSort);
  const placed = new Set();
  const pitch = TREE.cardH + TREE.rowGap;
  const lineages = [];
  let abandonedTotal = 0;
  for (const n of byId.values()) if (isAbandoned(n)) abandonedTotal += 1;

  const build = (root) => {
    const items = [], edges = [];
    const counts = {total: 0, hidden: 0};
    let rows = 0;
    const count = (n) => {
      const k = decisionOf(n).key;
      counts.total += 1;
      counts[k] = (counts[k] || 0) + 1;
      if (obj(n).status === 'probing') counts.probing = (counts.probing || 0) + 1;
    };
    const hide = (id) => {                       // an abandoned subtree: count it, never place it
      if (placed.has(id)) return;
      placed.add(id);
      count(byId.get(id));
      counts.hidden += 1;
      for (const c of kids.get(id) || []) hide(c);
    };
    const place = (id, col) => {
      placed.add(id);
      const node = byId.get(id);
      count(node);
      const ys = [], shown = [];
      for (const c of kids.get(id) || []) {
        if (placed.has(c)) continue;
        if (!u.showAbandoned && isAbandoned(byId.get(c))) { hide(c); continue; }
        ys.push(place(c, col + 1));
        shown.push(c);
      }
      let y;
      if (!ys.length) { y = rows * pitch; rows += 1; } else { y = (ys[0] + ys[ys.length - 1]) / 2; }
      items.push({id, node, col, x: 0, y});                 // x is set below, once the card width is known
      for (const c of shown) edges.push({from: id, to: c});
      return y;
    };
    const r = int(root.round);
    place(root.id, r !== null && r >= 0 && r <= 20 ? r : 0);
    lineages.push({root: root.id, node: root, items, edges, rows, counts,
                   height: Math.max(0, rows * pitch - TREE.rowGap)});
  };
  for (const root of roots) if (!placed.has(root.id)) build(root);
  for (const n of [...byId.values()].sort(rootSort)) if (!placed.has(n.id)) build(n);   // cycles: break anywhere

  let maxCol = 0;
  for (const lin of lineages) for (const it of lin.items) maxCol = Math.max(maxCol, it.col);
  const mr = int(d.max_rounds);
  const cols = Math.max(maxCol + 1, mr !== null && mr >= 0 && mr <= 20 ? mr + 1 : 1);
  const avail = num(u.treeWidth);
  const fit = avail === null ? TREE.cardW : Math.floor((avail - (cols - 1) * TREE.colGap) / cols);
  const cardW = Math.max(TREE.minW, Math.min(TREE.maxW, fit));
  const geom = {cardW, cardH: TREE.cardH, colGap: TREE.colGap, rowGap: TREE.rowGap,
                width: cols * cardW + (cols - 1) * TREE.colGap};
  for (const lin of lineages) for (const it of lin.items) it.x = it.col * (cardW + TREE.colGap);
  return {lineages, cols, geom, abandonedTotal, byId};
}

/** Ids on the path from the root to `id` (for highlighting the selected node's ancestry). */
function ancestry(byId, id) {
  const out = new Set();
  let cur = id;
  while (typeof cur === 'string' && byId.has(cur) && !out.has(cur)) {
    out.add(cur);
    cur = byId.get(cur).parent;
  }
  return out;
}

/** The rollout lines of a scored node: sc14 passes of its graded slots (+ infra, similarity), clean kimi passes
    (+ infra). [line4, line5] as text, or null without probe results. */
function scoredLines(node) {
  const n = obj(node), p = obj(n.probe), sc = obj(n.score);
  const s = probeStats(p, 'student'), t = probeStats(p, 'teacher');
  if (!s && !t) return null;
  return [s ? `${STUDENT} ${s.pass}/${s.valid}${s.infra ? ` +${s.infra} infra` : ''}${num(sc.similarity) !== null ? ` · sim ${sc.similarity}` : ''}` : '',
          t ? `${TEACHER} ${t.clean}/${t.valid} clean${t.infra ? ` +${t.infra} infra` : ''}` : ''];
}

/** Compact node card, five lines: name and decision badge (v1.5: after a "seed-invalid" badge when the node carries a
    seed note, seedNote()); title; F2P/P2P (or the build state) with the evolve branch's
    target pattern; then sc14 passes and clean kimi passes, each with its infra count; a task in the failure-match
    audit's scope (v1.4: every eligible task; v1.3 collectors: the accepted ones) ends the last line with its audit badge
    ("FA match: strong" / "not audited"). While the probe runs: the partial results of each model's graded slots ("sc14
    1 pass / 3 of 5 done") over a slim progress bar along the bottom edge; once the node's own slots are done, a note
    that its score comes when the round's probe ends. */
function renderNodeCard(item, ui, geom, ctx) {
  const it = obj(item), n = obj(it.node), u = obj(ui), g = Object.assign({cardW: TREE.cardW, cardH: TREE.cardH}, obj(geom));
  const id = text(n.id), dec = decisionOf(n), b = obj(n.build);
  const live = liveProbe(n), focus = focusLabel(n, ctx);
  const muted = (s) => (s ? `<span class="muted">${esc(s)}</span>` : '');
  let line3, line4 = '', line5 = '';
  if (dec.key === 'build_failed' || b.ok === false) {
    line3 = esc(`build failed${int(b.attempts) !== null ? ` after ${plural(b.attempts, 'attempt')}` : ''}`);
    line4 = muted(truncate(firstLine(b.problems) || n.decision_reason, 90));
  } else {
    const counted = num(b.f2p) !== null || num(b.p2p) !== null;
    const statusText = lookup(NODE_STATUS_TEXT, n.status) || text(n.status);
    line3 = counted ? `F2P ${fmtInt(b.f2p)} / P2P ${fmtInt(b.p2p)}` : esc(statusText || DASH);
    const scored = live ? null : scoredLines(n);
    if (live && live.ownDone) {
      line4 = esc(live.rows.slice(0, 2).map(passText).join(' · '));
      line5 = muted("probe done, scoring when the round's probe ends");
    } else if (live) {
      line4 = esc(partialText(live.rows[0]));
      line5 = live.rows[1] ? esc(partialText(live.rows[1])) : '';
    } else if (scored) {
      [line4, line5] = scored.map(esc);
    } else {
      const reason = text(n.decision_reason);
      let detail = '';                                       // line 4 of a card that is still being built
      if (counted) detail = statusText;
      else if (reason && reason !== text(n.status)) detail = reason;
      else if (int(b.attempts)) detail = `author attempt ${b.attempts}`;
      line4 = muted(detail);
    }
  }
  const tag = focus ? `<span class="tag tag-focus" title="${esc(`target failure pattern: ${focus.name}`)}">${esc(focus.label)}</span>` : '';
  // The failure-match audit of a task in its scope (v1.4: eligible; v1.3: accepted): a badge at the end of the last line.
  const fm = inAuditScope(n) ? failureMatch(n) : undefined;
  // v1.6: a task of a batch (NODE.in_batch) is never audited: "batch · not audited" in the same place
  const inBatch = n.in_batch === true && notAudited(n);
  const fa = fm !== undefined ? faBadge(fm, 'badge-xs badge-fa')
    : inBatch ? badge('batch · not audited', 'muted', `in the batch (outputs/tasks.jsonl); ${BATCH_HELP}`, 'badge-xs badge-fa') : '';
  // v1.5: an invalid seed that did not decide the node (an accepted task stays accepted): a badge before the decision's
  const seed = seedBadge(n, 'badge-xs badge-seed'), decBadge = badge(dec.label, dec.tone, null, 'badge-sm');
  const tip = [id, text(n.title), `${dec.label}${text(n.decision_reason) ? `: ${text(n.decision_reason)}` : ''}`,
               seedNote(n) ? `seed-invalid: ${seedNote(n)}` : '',
               isObj(n.withdrawn) ? `withdrawn ${withdrawnParts(n.withdrawn).join(' · ')}` : '',
               focus ? `target pattern: ${focus.name}` : '', ...(live ? live.rows.map(partialText) : []),
               fm === undefined ? '' : `FA match: ${faVerdict(fm).label}${fm && text(fm.summary) ? ` · ${text(fm.summary)}` : ''}`,
               inBatch ? 'in the batch (outputs/tasks.jsonl) · not audited' : '']
    .filter(Boolean).join('\n');
  const cls = `node tone-${dec.tone}${u.selected === id ? ' selected' : ''}${it.onPath ? ' on-path' : ''}${live ? ' is-probing' : ''}`;
  return `<div class="${cls}" role="button" tabindex="0" data-node="${esc(id)}" data-fkey="card:${esc(id)}" aria-label="${esc(`${id}: ${dec.label}`)}"` +
         ` style="left:${num(it.x) || 0}px;top:${num(it.y) || 0}px;width:${num(g.cardW) || TREE.cardW}px;height:${num(g.cardH) || TREE.cardH}px" title="${esc(tip)}">` +
         `<div class="node-top"><span class="node-id">${esc(nodeLabel(n))}</span>${seed ? `<span class="node-badges">${seed}${decBadge}</span>` : decBadge}</div>` +
         `<div class="node-title">${esc(n.title) || '<span class="muted">untitled</span>'}</div>` +
         `<div class="node-meta node-l3"><span class="node-counts">${line3}</span>${tag}</div>` +
         `<div class="node-meta">${line4}</div>` +
         (fa ? `<div class="node-meta node-l5"><span class="node-l5-text">${line5}</span>${fa}</div>` : `<div class="node-meta">${line5}</div>`) +
         (live ? bar(live.done, live.total, live.ownDone ? 'good' : 'pending', `probe progress ${live.done}/${live.total}`, 'node-pbar') : '') +
         '</div>';
}

function lineageSummary(lin) {
  const root = obj(lin.node), c = obj(lin.counts);
  const parts = [`<span class="lin-count">${esc(plural(c.total || 0, 'task'))}</span>`];
  const acc = (c.accepted || 0) + (c.accepted_evolve || 0);
  if (acc) parts.push(badge(`${acc} accepted`, 'good', null, 'badge-sm'));
  if (c.probing) parts.push(badge(`${c.probing} probing`, 'pending', null, 'badge-sm'));
  const pend = (c.pending || 0) - (c.probing || 0);
  if (pend > 0) parts.push(badge(`${pend} pending`, 'pending', null, 'badge-sm'));
  const bad = (c.discard || 0) + (c.build_failed || 0);
  if (bad) parts.push(badge(`${bad} discarded`, 'bad', null, 'badge-sm'));
  if (c.hidden) parts.push(`<span class="muted small">${esc(fmtInt(c.hidden))} abandoned hidden</span>`);
  const dec = decisionOf(root);
  return `<span class="lin-id">${esc(shortId(root.id))}</span>${badge(dec.label, dec.tone, null, 'badge-sm')}` +
         `<span class="lin-title">${esc(root.title)}</span><span class="lin-stats">${parts.join('')}</span>`;
}

function renderLineage(lin, layout, ui, ctx) {
  const u = obj(ui), g = layout.geom;
  const width = g.width;
  const pos = new Map(lin.items.map((it) => [it.id, it]));
  const path = u.selected ? ancestry(layout.byId, u.selected) : new Set();
  const links = lin.edges.map((e) => {
    const a = pos.get(e.from), b = pos.get(e.to);
    if (!a || !b) return '';
    const x1 = a.x + g.cardW, y1 = a.y + g.cardH / 2, x2 = b.x, y2 = b.y + g.cardH / 2;
    const mx = (x1 + x2) / 2;
    const cls = `link${isAbandoned(b.node) ? ' link-abandoned' : ''}${path.has(e.from) && path.has(e.to) ? ' link-hl' : ''}`;
    return `<path class="${cls}" d="M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}"/>`;
  }).join('');
  const cards = lin.items.map((it) => renderNodeCard(Object.assign({}, it, {onPath: path.has(it.id) && it.id !== u.selected}), u, g, ctx)).join('');
  const canvas = `<div class="lineage-canvas" style="width:${width}px;height:${lin.height}px">` +
                 `<svg class="links" width="${width}" height="${lin.height}" viewBox="0 0 ${width} ${lin.height}" aria-hidden="true">${links}</svg>` +
                 `${cards}</div>`;
  return det(u, `lin:${lin.root}`, lineageSummary(lin), canvas, true, 'lineage');
}

function roundAt(detail, k) { return arr(obj(detail).rounds).find((r) => isObj(r) && int(r.index) === k) || null; }

/** Column header: round number and kind, status, probe slots or (while its tasks are built) "built x / y", seeds,
    abandoned probes. `counts` ({nodes, built}, from the tree) stands in for rounds[] without those fields. */
function renderRoundHead(round, k, maxRounds, counts) {
  const r = obj(round), p = isObj(r.probe) ? r.probe : null, c = obj(counts);
  const status = text(r.status);
  const kind = k === 0 ? 'mask tasks' : 'evolved tasks';
  let html = `<div class="round-title">Round ${k} <span class="muted">· ${kind}${k === maxRounds ? ' · last' : ''}</span>` +
             `${round ? badge(status || 'unknown', lookup(ROUND_TONES, status), null, 'badge-sm') : '<span class="muted small">not started</span>'}</div>`;
  if (p) {
    const done = num(p.slots_done), total = num(p.slots_total);
    const byModel = Object.entries(obj(p.by_model)).map(([m, v]) => {
      const pr = pair(v);
      return pr ? `${esc(m)} ${fmtInt(pr[0])}/${fmtInt(pr[1])}` : '';
    }).filter(Boolean).join(' · ');
    html += `<div class="round-probe" title="${esc(p.dir)}">${bar(done, total, status === 'probing' ? 'pending' : 'good', 'probe slots')}` +
            `<span>${fmtInt(done)}/${fmtInt(total)} slots · ${esc(plural(num(p.tasks) || 0, 'task'))}</span></div>` +
            (byModel ? `<div class="round-sub">${byModel}</div>` : '');
  } else if (status === 'building' || status === 'pending') {
    // Its tasks are being planned / authored / gated: how many passed their gates so far.
    const nodes = int(r.nodes) !== null ? r.nodes : int(c.nodes), built = int(r.built) !== null ? r.built : int(c.built);
    if (nodes) {
      html += `<div class="round-probe">${bar(built, nodes, 'pending', 'tasks built')}` +
              `<span>built ${fmtInt(built)} / ${fmtInt(nodes)}</span></div>`;
    }
  }
  const seeds = isObj(r.seeds) ? r.seeds : null;
  if (seeds) {
    const invalid = Object.entries(obj(seeds.invalid));
    const tip = invalid.map(([cand, why]) => `${cand}: ${text(why)}`).join('\n');
    html += `<div class="round-sub">seeds: ${fmtInt(arr(seeds.valid).length)} valid · ` +
            `${invalid.length ? `<span class="txt-bad" title="${esc(tip)}">${fmtInt(invalid.length)} invalid</span>` : '0 invalid'}</div>`;
  }
  const ab = arr(r.abandoned_probes).map(text);
  if (ab.length) html += `<div class="round-sub" title="${esc(ab.join('\n'))}">${esc(plural(ab.length, 'abandoned probe'))}</div>`;
  const unfinished = arr(r.unfinished_probes).map(text);              // v1.6: a batch run ended before they finished
  if (unfinished.length) {
    html += `<div class="round-sub" title="${esc(['the run ended before these probes finished:', ...unfinished].join('\n'))}">` +
            `${esc(plural(unfinished.length, 'unfinished probe'))}</div>`;
  }
  return `<div class="round-head">${html}</div>`;
}

function renderLegend() {
  return `<div class="legend">${DECISION_ORDER.map((k) => badge(DECISIONS[k].label, DECISIONS[k].tone, DECISIONS[k].help, 'badge-sm')).join('')}</div>`;
}

/** Generation tree: columns are rounds 0..max_rounds, rows are lineages (one collapsible block per round-0 node). */
function renderTree(detail, ui) {
  const d = obj(detail), u = obj(ui);
  const L = treeLayout(d, u);
  const tools = '<div class="tree-tools">' +
    (L.abandonedTotal ? `<button type="button" class="btn" data-action="toggle-abandoned" aria-pressed="${u.showAbandoned ? 'true' : 'false'}">` +
                        `${u.showAbandoned ? 'Hide' : 'Show'} abandoned (${fmtInt(L.abandonedTotal)})</button>` : '') +
    '<button type="button" class="btn" data-action="expand-lineages">Expand all</button>' +
    '<button type="button" class="btn" data-action="collapse-lineages">Collapse all</button></div>';
  const sub = [];
  if (int(d.max_rounds) !== null) sub.push(`${d.max_rounds} evolve rounds`);
  if (int(d.branches) !== null) sub.push(`${d.branches} branches per seed`);
  const head = `<span class="muted small">${esc(sub.join(' · '))}</span>${tools}`;
  if (!L.lineages.length) {
    return section('Generation tree', renderLegend() + note('No candidate tasks yet: the planner has not produced candidates, or none has been authored.'), head, 'panel-tree');
  }
  const g = L.geom;
  const mr = int(d.max_rounds);
  const ctx = treeContext(d);
  const counts = (k) => {                                // per round, for rounds[] without nodes/built
    const members = [...L.byId.values()].filter((n) => int(n.round) === k && !isAbandoned(n));
    return {nodes: members.length, built: members.filter((n) => obj(n.build).ok === true).length};
  };
  const heads = Array.from({length: L.cols}, (_, k) => renderRoundHead(roundAt(d, k), k, mr, counts(k))).join('');
  const blocks = L.lineages.map((lin) => renderLineage(lin, L, u, ctx)).join('');
  return section('Generation tree', renderLegend() +
    `<div class="tree-scroll"${scrollAttr(`tree:${text(d.task_id)}`)}><div class="tree" style="width:${g.width}px">` +
    `<div class="round-heads" style="grid-template-columns:repeat(${L.cols},${g.cardW}px);column-gap:${g.colGap}px">${heads}</div>` +
    `${blocks}</div></div>`, head, 'panel-tree');
}

/** "Round 0 at a glance": one row per round-0 task (name, title, F2P/P2P, sc14 x/5, clean kimi x/5, infra counts of
    both models, decision, reason); a row opens the node's detail panel like its card in the tree. While a task is
    probed its cells show the partial results ("1 pass / 3 of 5"). */
function renderRoundZero(detail, ui) {
  const d = obj(detail), u = obj(ui);
  const seen = new Set();
  const nodes = arr(d.nodes).filter((n) => isObj(n) && typeof n.id === 'string' && n.id && int(n.round) === 0 &&
                                           !seen.has(n.id) && seen.add(n.id)).sort((a, b) => cmpStr(a.id, b.id));
  if (!nodes.length) return '';
  const tally = {};
  const rows = nodes.map((n) => {
    const id = text(n.id), dec = decisionOf(n), b = obj(n.build), p = obj(n.probe), live = liveProbe(n);
    tally[dec.key] = (tally[dec.key] || 0) + 1;
    const statusText = lookup(NODE_STATUS_TEXT, n.status) || text(n.status);
    const counts = dec.key === 'build_failed' || b.ok === false ? '<span class="txt-bad">build failed</span>'
      : num(b.f2p) !== null || num(b.p2p) !== null ? `${fmtInt(b.f2p)} / ${fmtInt(b.p2p)}`
      : `<span class="muted">${esc(statusText) || DASH}</span>`;
    let sc14 = DASH, kimi = DASH, infra = DASH;
    if (live) {
      const of = (m) => live.rows.find((r) => r.model === m);
      const cell = (r) => (r ? `${fmtInt(r.pass)} pass / ${fmtInt(r.done)} of ${fmtInt(r.total)}` : DASH);
      const s = of(STUDENT), t = of(TEACHER);
      sc14 = cell(s);
      kimi = t ? `${cell(t)} <span class="muted small">not audited</span>` : DASH;
      infra = `${s ? fmtInt(s.infra) : DASH} · ${t ? fmtInt(t.infra) : DASH}`;
    } else {
      const s = probeStats(p, 'student'), t = probeStats(p, 'teacher');
      if (s) sc14 = `${fmtInt(s.pass)}/${fmtInt(s.valid)}`;
      if (t) kimi = `${fmtInt(t.clean)}/${fmtInt(t.valid)}`;
      if (s || t) infra = `${s ? fmtInt(s.infra) : DASH} · ${t ? fmtInt(t.infra) : DASH}`;
    }
    return `<tr class="r0-row${u.selected === id ? ' selected' : ''}" data-node="${esc(id)}">` +
           `<td><button type="button" class="node-link" data-node="${esc(id)}" data-fkey="r0:${esc(id)}">${esc(shortId(id))}</button></td>` +
           `<td class="r0-title">${esc(n.title) || '<span class="muted">untitled</span>'}</td><td class="num">${counts}</td>` +
           `<td class="num">${sc14}</td><td class="num">${kimi}</td><td class="num">${infra}</td>` +
           `<td>${badge(dec.label, dec.tone, dec.help, 'badge-sm')}${seedBadge(n, 'badge-xs badge-seed')}` +
           `${n.in_batch === true && notAudited(n) ? badge('batch', 'muted', `in the batch (outputs/tasks.jsonl); ${BATCH_HELP}`, 'badge-xs badge-seed') : ''}</td>` +
           `<td class="r0-reason">${esc(n.decision_reason) || DASH}</td></tr>`;
  }).join('');
  const seeds = (tally.evolve || 0) + (tally.accepted_evolve || 0), bad = (tally.discard || 0) + (tally.build_failed || 0);
  const sub = [plural(nodes.length, 'task'), seeds ? plural(seeds, 'seed') : '', tally.accepted_evolve ? `${fmtInt(tally.accepted_evolve)} accepted` : '',
               bad ? `${fmtInt(bad)} discarded` : ''].filter(Boolean).join(' · ');
  return section('Round 0 at a glance',
    `<div class="tbl-wrap"${scrollAttr(`r0:${text(d.task_id)}`)}><table class="tbl tbl-r0"><thead><tr><th>task</th><th>title</th>` +
    '<th class="num">F2P / P2P</th><th class="num">sc14 pass</th><th class="num">kimi clean</th>' +
    '<th class="num" title="rollouts the verifier could not grade">infra sc14 · kimi</th><th>decision</th><th>reason</th>' +
    `</tr></thead><tbody>${rows}</tbody></table></div>`,
    `<span class="muted small">${esc(sub)} · click a row for its details</span>`, 'panel-r0');
}

/** "Failure-match audit" (above the tree): one row per task in the audit's scope, the eligible tasks (v1.4: built and
    probed, a graded sc14 rollout, not discarded or withdrawn; v1.3 collectors: the accepted ones), grouped: the
    accepted tasks first, then the other eligible tasks round by round, each group headed by its audit count. Columns:
    task and title, round, decision, sc14 passes, clean kimi passes, failure similarity, the audit's verdict and the
    first line of its summary (in full on hover and in the drawer). A row opens the node's drawer and keeps the focus.
    '' when no task is in the scope, and for collectors that predate the audit. */
function renderAuditTable(detail, ui) {
  const d = obj(detail), u = obj(ui), all = nodesOf(d);
  if (!all.some((n) => hasOwn(n, 'failure_match'))) return '';
  const scoped = all.filter(inAuditScope);
  if (!scoped.length) return '';
  const round = (n) => int(n.round);
  const order = (a, b) => (round(a) === null ? 1e9 : round(a)) - (round(b) === null ? 1e9 : round(b)) || cmpStr(a.id, b.id);
  const groups = [{label: 'Accepted', nodes: scoped.filter(isAccepted).sort(order)}];
  for (const n of scoped.filter((x) => !isAccepted(x)).sort(order)) {          // then the others, round by round
    const label = `${round(n) === null ? 'Round ?' : `Round ${round(n)}`} · not accepted`;
    const last = groups[groups.length - 1];
    if (last.label === label) last.nodes.push(n);
    else groups.push({label, nodes: [n]});
  }
  const tally = (list) => {                                  // {audited, strong, partial, weak, none} of some rows
    const c = {audited: 0, strong: 0, partial: 0, weak: 0, none: 0};
    for (const n of list) {
      const fm = failureMatch(n), k = faVerdict(fm).key;
      if (fm) c.audited += 1;
      if (FA_ORDER.includes(k)) c[k] += 1;
    }
    return c;
  };
  const row = (n) => {
    const id = text(n.id), fm = failureMatch(n), p = obj(n.probe), sc = obj(n.score), dec = decisionOf(n);
    const s = probeStats(p, 'student'), t = probeStats(p, 'teacher');
    const infra = (x) => (x.infra ? ` <span class="muted small">+${fmtInt(x.infra)} infra</span>` : '');
    const summary = fm ? text(fm.summary) : '';
    const line = summary ? `<span title="${esc(summary)}">${esc(truncate(firstLine(summary), 240))}</span>`
      : `<span class="muted">${fm ? 'no summary' : 'not audited yet'}</span>`;
    return `<tr class="fa-row${u.selected === id ? ' selected' : ''}" data-node="${esc(id)}">` +
           `<td class="fa-task"><button type="button" class="node-link" data-node="${esc(id)}" data-fkey="audit:${esc(id)}">${esc(shortId(id))}</button>` +
           `<div class="fa-title muted small" title="${esc(n.title)}">${esc(n.title) || 'untitled'}</div></td>` +
           `<td class="num">${round(n) === null ? DASH : fmtInt(n.round)}</td>` +
           `<td class="fa-decision">${badge(dec.label, dec.tone, text(n.decision_reason) || dec.help, 'badge-sm')}` +
           `${seedBadge(n, 'badge-xs badge-seed')}</td>` +
           `<td class="num">${s ? `${fmtInt(s.pass)}/${fmtInt(s.valid)}${infra(s)}` : DASH}</td>` +
           `<td class="num">${t ? `${fmtInt(t.clean)}/${fmtInt(t.valid)}${infra(t)}` : DASH}</td>` +
           `<td class="num">${num(sc.similarity) === null ? DASH : `${esc(fmtCompact(sc.similarity))}/10`}</td>` +
           `<td>${faBadge(fm, 'badge-sm badge-fa')}</td><td class="fa-summary">${line}</td></tr>`;
  };
  const rows = groups.filter((g) => g.nodes.length).map((g) => {
    const c = tally(g.nodes);
    return `<tr class="fa-group"><th colspan="8" scope="rowgroup">${esc(g.label)} <span class="muted small">` +
           `${esc(`${plural(g.nodes.length, 'task')} · audit ${fmtInt(c.audited)}/${fmtInt(g.nodes.length)}`)}</span></th></tr>` +
           g.nodes.map(row).join('');
  }).join('');
  const marked = all.some(marksScope);                       // v1.4: the eligible tasks; v1.3: the accepted ones
  const c = Object.assign(tally(scoped), marked ? {eligible: scoped.length} : {accepted: scoped.length});
  const sub = [plural(scoped.length, marked ? 'eligible task' : 'accepted task'),
               `audit ${fmtInt(c.audited)}/${fmtInt(scoped.length)}`, auditBreakdown(c)].filter(Boolean).join(' · ');
  return section('Failure-match audit',
    `<div class="tbl-wrap"${scrollAttr(`audit:${text(d.task_id)}`)}><table class="tbl tbl-fa"><thead><tr><th>task</th>` +
    '<th class="num">round</th><th>decision</th><th class="num">sc14 pass</th><th class="num">kimi clean</th>' +
    '<th class="num">similarity</th>' +
    `<th title="${esc('does sc14 repeat the target failure patterns in this task?')}">audit verdict</th><th>summary</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`,
    `<span class="muted small">${esc(sub)} · click a row for its details</span>`, 'panel-fa');
}

// --------------------------------------------------------------------------------------- node detail

function renderSimilarity(value, key, ui) {
  let d = value;
  if (typeof d === 'string') { try { d = JSON.parse(d); } catch (error) { d = null; } }
  if (!isObj(d)) return text(value) ? det(ui, key, 'Similarity detail', pre(value, '', key), false, 'fold fold-inline') : '';
  const matched = arr(d.matched_patterns).filter(isObj);
  const missing = arr(d.missing_patterns).map(text).filter(Boolean);
  const body =
    (matched.length ? `<table class="tbl tbl-compact tbl-wrapcells"><thead><tr><th>synthetic pattern</th><th>target pattern</th></tr></thead><tbody>` +
      matched.map((m) => `<tr><td>${esc(m.synthetic)}</td><td>${esc(m.target)}</td></tr>`).join('') + '</tbody></table>' : '') +
    (missing.length ? `<h4>Missing target patterns</h4><ul class="plain">${missing.map((m) => `<li>${esc(m)}</li>`).join('')}</ul>` : '') +
    (text(d.rationale) ? `<h4>Rationale</h4><p>${esc(d.rationale)}</p>` : '') +
    det(ui, `${key}:raw`, 'Raw JSON', pre(JSON.stringify(d, null, 2), '', `${key}:raw`), false, 'fold fold-inline');
  return det(ui, key, `Similarity detail${num(d.score) !== null ? ` · score ${esc(d.score)}/10` : ''}`, body, false, 'fold fold-inline');
}

function prettyJSON(v) {
  if (typeof v === 'string') {
    try { return JSON.stringify(JSON.parse(v), null, 2); } catch (error) { return v; }   // truncated JSON stays raw
  }
  if (v === null || v === undefined) return '';
  try { return JSON.stringify(v, null, 2); } catch (error) { return text(v); }
}

function statusBadge(status) {
  const s = text(status);
  return badge(s || 'unknown', s === 'pass' ? 'good' : s === 'fail' ? 'bad' : 'muted', null, 'badge-sm');
}

/** The rollouts of a probe. Cells wrap; an agent/verifier error longer than one short line sits behind a <details>
    holding its full text (`key` prefixes the open-state and scroll keys of those blocks). */
function renderRollouts(rollouts, ui, key) {
  const prefix = text(key) || 'rollouts';
  const rows = arr(rollouts).filter(isObj).map((r, i) => {
    const errs = [text(r.agent_error) && `agent: ${text(r.agent_error)}`, text(r.verifier_error) && `verifier: ${text(r.verifier_error)}`]
      .filter(Boolean).join('\n');
    const k = `${prefix}:err:${text(r.slot) || i}`;
    const err = !errs ? '' : errs.length <= 80 && !errs.includes('\n') ? `<span class="err">${esc(errs)}</span>`
      : det(ui, k, `<span class="err">${esc(truncate(firstLine(errs), 72))}</span>`, pre(errs, 'pre-err', k), false, 'fold fold-inline fold-err');
    return `<tr><td><code>${esc(r.slot)}</code></td><td>${esc(r.model)}</td><td>${statusBadge(r.status)}</td>` +
           `<td class="num">${num(r.reward) === null ? DASH : fmtFixed(r.reward, 1)}</td><td class="num">${esc(r.f2p) || DASH}</td>` +
           `<td class="num">${esc(r.p2p) || DASH}</td><td class="num">${fmtInt(r.tool_calls)}</td><td class="num">${fmtInt(r.patch_chars)}</td>` +
           `<td class="err-cell">${err}</td></tr>`;
  }).join('');
  if (!rows) return note('No rollouts recorded.');
  return `<div class="tbl-wrap"${scrollAttr(`${prefix}:table`)}><table class="tbl tbl-compact tbl-roll"><thead><tr><th>slot</th><th>model</th>` +
         '<th>status</th><th class="num">reward</th><th class="num">F2P</th><th class="num">P2P</th><th class="num">tool calls</th>' +
         `<th class="num">patch chars</th><th>errors</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

/** Teacher audit: the collector's list ({slot, cheated, categories, summary[, rubric_score]} per audited teacher pass) as
    a table, open when a slot cheated; the older JSON-dumped string as preformatted text. */
function renderAudit(value, key, ui) {
  const label = 'Teacher audit (teacher_audit.json)';
  if (Array.isArray(value)) {
    const rows = value.filter(isObj);
    const cheated = rows.filter((r) => r.cheated === true).length;
    const rubric = rows.some((r) => num(r.rubric_score) !== null);   // the judge's process score, when it gave one
    const cats = (c) => (Array.isArray(c) ? c.map(text).filter(Boolean).map((x) => `<span class="tag">${esc(x)}</span>`).join(' ') : esc(c));
    const body = rows.length
      ? `<div class="tbl-wrap"${scrollAttr(`${key}:table`)}><table class="tbl tbl-compact tbl-wrapcells tbl-audit"><thead><tr><th>slot</th>` +
        `<th>cheated</th><th>categories</th><th>summary</th>${rubric ? '<th class="num" title="the judge&#39;s process rubric score">rubric</th>' : ''}` +
        '</tr></thead><tbody>' +
        rows.map((r) => `<tr><td><code>${esc(r.slot) || DASH}</code></td>` +
                        `<td>${r.cheated === true ? badge('yes', 'bad', null, 'badge-sm') : r.cheated === false ? 'no' : DASH}</td>` +
                        `<td>${cats(r.categories) || DASH}</td><td>${esc(r.summary) || DASH}</td>` +
                        `${rubric ? `<td class="num">${num(r.rubric_score) === null ? DASH : fmtCompact(r.rubric_score)}</td>` : ''}</tr>`).join('') +
        '</tbody></table></div>'
      : note('No teacher pass was audited.');
    return det(ui, key, `${esc(label)} <span class="muted small">${esc(plural(rows.length, 'slot'))} · ${fmtInt(cheated)} cheated</span>`,
               body, cheated > 0, 'fold fold-inline');
  }
  const t = prettyJSON(value);
  return t ? det(ui, key, `${esc(label)} <span class="muted small">${fmtInt(t.length)} chars</span>`, pre(t, 'pre-md', key), false, 'fold fold-inline')
           : `<div class="missing">${esc(label)}: not available</div>`;
}

/** The drawer's "Failure-pattern match (audit)" section (NODE.failure_match, v1.3): an auditor's verdict on whether sc14
    repeats the target's failure patterns in this task: the verdict and its summary, a row per target pattern
    (reproduced, sc14 slots, evidence), a row per sc14 rollout (status, failing tests, cause, matched pattern), the
    other causes, and the similarity judge's score with whether the auditor agrees. A task in the audit's scope without
    an audit says so (v1.4: an eligible task; v1.3 collectors: an accepted one); the audit of a task outside the scope
    (v1.4, e.g. withdrawn since) is shown with why it is not counted; '' for any other node without one, and for
    collectors that predate the audit. */
function renderFailureMatch(node, detail, ui, key, now) {
  const n = obj(node), fm = failureMatch(n);
  if (notAudited(n)) {                         // v1.6: a batch's node: never audited (a scored or batch task says so)
    return isObj(n.score) || n.in_batch === true ? '<h3 class="drawer-h">Failure-pattern match (audit)</h3>' +
      note(`Not audited: ${batchName(null, detail)}'s batch is not audited; no failure-match audit runs on it.`) : '';
  }
  if (fm === undefined || (fm === null && !inAuditScope(n))) return '';
  const head = '<h3 class="drawer-h">Failure-pattern match (audit)</h3>';
  if (!fm) {
    return head + note(`Not audited yet: no failure-match audit of this ${marksScope(n) ? 'eligible' : 'accepted'} task ` +
                       '(runs/<run>/audit/failure_match/<task_id>.json).');
  }
  const outside = marksScope(n) && !inAuditScope(n)
    ? note(`Not counted: this task is outside the audit's scope, ${outOfScopeWhy(n)}.`, 'note-wait') : '';
  const v = faVerdict(fm), ctx = treeContext(detail), k = text(key) || 'failure_match';
  const summary = text(fm.summary), other = text(fm.other_causes);
  const pattern = (name) => (text(name) ? `<span class="tag tag-fa tag-wrap">${esc(patternLabel(name, ctx))}</span>` : DASH);
  // The similarity judge's score (failure similarity, /10) and whether the auditor agrees with it.
  const judge = isObj(fm.similarity_judge) ? fm.similarity_judge : null;
  const judgeHtml = !judge ? DASH
    : `${num(judge.score) === null ? DASH : `${esc(fmtCompact(judge.score))}/10`}` +
      (judge.agrees === true ? ` ${badge('agrees', 'good', 'the auditor agrees with the similarity judge', 'badge-sm')}`
        : judge.agrees === false ? ` ${badge('disagrees', 'bad', 'the auditor disagrees with the similarity judge', 'badge-sm')}` : '') +
      (text(judge.comment) ? `<div class="muted small">${esc(judge.comment)}</div>` : '');
  const at = isoToEpoch(fm.audited_at);
  const audited = at === null ? esc(fm.audited_at) || DASH
    : `${esc(fmtDate(at))}${num(now) !== null ? ` <span class="muted small">(${esc(fmtAgo(at, now))})</span>` : ''}`;
  const slots = (list) => arr(list).map(text).filter(Boolean).map((s) => `<span class="slot">${esc(s)}</span>`).join(' ');
  const patterns = arr(fm.patterns).filter(isObj);
  const patternRows = patterns.map((p) => `<tr><td>${pattern(p.name)}</td><td>${reproducedBadge(p.reproduced)}</td>` +
    `<td>${slots(p.slots) || DASH}</td><td>${esc(p.evidence) || DASH}</td></tr>`).join('');
  const rollouts = arr(fm.student_rollouts).filter(isObj);
  const rolloutRows = rollouts.map((r) => {
    const tests = arr(r.failing_tests).map(text).filter(Boolean);
    return `<tr><td><code>${esc(r.slot) || DASH}</code></td><td>${text(r.status) ? statusBadge(r.status) : DASH}</td>` +
           `<td>${tests.length ? `<ul class="fa-tests">${tests.map((t) => `<li><code>${esc(t)}</code></li>`).join('')}</ul>` : DASH}</td>` +
           `<td>${esc(r.cause) || DASH}</td><td>${pattern(r.pattern)}</td></tr>`;
  }).join('');
  return head + outside +
    `<div class="callout tone-${toneOf(v.tone)}"><div class="callout-k">Verdict</div>` +
    `<p>${badge(`FA match: ${v.label}`, v.tone, v.help)} <span class="muted small">${esc(v.help)}</span></p>` +
    (summary ? `<p>${esc(summary)}</p>` : '') + '</div>' +
    kv([['Similarity judge', judgeHtml], ['Other causes', other ? esc(other) : '<span class="muted">none recorded</span>'],
        ['Audited', audited]]) +
    '<h4>Target failure patterns</h4>' +
    (patternRows ? `<div class="tbl-wrap"${scrollAttr(`${k}:patterns`)}><table class="tbl tbl-compact tbl-wrapcells tbl-fa-patterns">` +
                   '<thead><tr><th>target pattern</th><th>reproduced</th><th>sc14 slots</th><th>evidence</th></tr></thead>' +
                   `<tbody>${patternRows}</tbody></table></div>` : note('No per-pattern findings recorded.')) +
    '<h4>sc14 rollouts</h4>' +
    (rolloutRows ? `<div class="tbl-wrap"${scrollAttr(`${k}:rollouts`)}><table class="tbl tbl-compact tbl-wrapcells tbl-fa-rollouts">` +
                   '<thead><tr><th>slot</th><th>status</th><th>failing tests</th><th>cause</th><th>matched pattern</th></tr></thead>' +
                   `<tbody>${rolloutRows}</tbody></table></div>` : note('No per-rollout findings recorded.'));
}

/** The side-drawer for one node: every NODE field of the contract; long texts are collapsible <pre> blocks. */
function renderNodeDetail(node, detail, ui, now) {
  if (!isObj(node)) {
    return '<div class="drawer-head"><div class="drawer-id">node not found</div>' +
           '<button type="button" class="icon-btn" data-action="close-drawer" aria-label="Close details">×</button></div>' +
           note('This node is not in the run detail (yet).');
  }
  const n = node, u = obj(ui), id = text(n.id), dec = decisionOf(n);
  const b = obj(n.build), p = isObj(n.probe) ? n.probe : null, sc = isObj(n.score) ? n.score : null, x = obj(n.details);
  const key = (s) => `nd:${id}:${s}`;
  const byId = new Map(arr(obj(detail).nodes).filter(isObj).map((m) => [m.id, m]));
  const nodeLink = (nid) => {
    const target = byId.get(nid), dd = decisionOf(target);
    return `<button type="button" class="node-link" data-node="${esc(nid)}" data-fkey="link:${esc(nid)}">${esc(shortId(nid))}` +
           `${target ? badge(dd.label, dd.tone, null, 'badge-xs') : ''}</button>`;
  };
  const meta = [text(n.kind) || 'task', int(n.round) !== null ? `round ${n.round}` : '', int(n.branch) !== null ? `branch ${n.branch}` : '',
                text(n.status) ? `status: ${text(n.status)}` : ''].filter(Boolean).join(' · ');
  const children = arr(n.children).map(text).filter(Boolean);
  const focus = focusLabel(n, treeContext(detail));
  let html = `<div class="drawer-head"><div class="drawer-id">${esc(id)}</div>${badge(dec.label, dec.tone, dec.help)}` +
             '<button type="button" class="icon-btn" data-action="close-drawer" aria-label="Close details">×</button></div>' +
             `<div class="drawer-body"><h2 class="drawer-title">${esc(n.title) || 'untitled'}</h2><div class="drawer-meta">${esc(meta)}</div>` +
             `<div class="callout tone-${toneOf(dec.tone)}"><div class="callout-k">Decision</div><p><b>${esc(dec.label)}</b>` +
             `${text(n.decision_reason) ? ` · ${esc(n.decision_reason)}` : ''}</p></div>` +
             // v1.4: an accepted task main withdrew later (NODE.withdrawn): discarded, outside the failure-match audit
             (isObj(n.withdrawn) ? '<div class="callout tone-bad"><div class="callout-k">Withdrawn</div>' +
                                   `<p>${esc(withdrawnParts(n.withdrawn).join(' · '))}</p></div>` : '') +
             // v1.5: an invalid seed that did not decide the node, e.g. an accepted task (acceptance beats an invalid seed)
             (seedNote(n) ? `<div class="callout tone-${SEED_NOTE.tone}"><div class="callout-k">Seed-invalid</div>` +
                            `<p>${esc(seedNote(n))}</p><p class="muted small">${esc(`${seedSource(n, detail)} · ${SEED_NOTE.help}`)}</p></div>` : '') +
             // v1.6: one of a batch's tasks (NODE.in_batch): where it is, and that it is not audited
             (n.in_batch === true && notAudited(n) ? `<div class="callout tone-muted"><div class="callout-k">${esc(batchName(null, detail))} batch</div>` +
                `<p>${batchBadge(batchName(null, detail), 'badge-sm')} <code>${esc(`${text(obj(detail).outputs) || 'outputs'}/${text(b.task_id)}/`)}</code></p>` +
                `<p class="muted small">${esc(`one of the tasks of ${text(obj(detail).source_label) || 'the batch'} (outputs/tasks.jsonl); ${BATCH_HELP}`)}</p></div>` : '') +
             (text(n.summary) ? `<p class="summary">${esc(n.summary)}</p>` : '') +
             (focus ? `<div class="rel"><span class="rel-k">targets</span><span class="tag tag-focus tag-wrap" ` +
                      `title="the target failure pattern this evolve branch aims at">${esc(focus.label)}</span></div>` : '') +
             `<div class="rel"><span class="rel-k">lineage</span><code>${esc(n.lineage) || DASH}</code></div>` +
             (text(n.parent) ? `<div class="rel"><span class="rel-k">parent</span>${nodeLink(text(n.parent))}</div>` : '') +
             (children.length ? `<div class="rel"><span class="rel-k">children</span>${children.map(nodeLink).join('')}</div>` : '');

  // Build and gates.
  const f2p = arr(b.f2p_tests).map(text).filter(Boolean), p2p = arr(b.p2p_tests_sample).map(text).filter(Boolean);
  const list = (names, k) => `<ul class="tests"${scrollAttr(k)}>${names.map((t) => `<li><code>${esc(t)}</code></li>`).join('')}</ul>`;
  html += '<h3 class="drawer-h">Build &amp; gates</h3>' + kv([
    ['Gates', b.ok === true ? badge('passed', 'good') : b.ok === false ? badge('failed', 'bad') : badge('not finished', 'pending')],
    ['Attempts', fmtInt(b.attempts)],
    ['F2P / P2P', `${fmtInt(b.f2p)} / ${fmtInt(b.p2p)}`],
    ['Task id', text(b.task_id) ? `<code>${esc(b.task_id)}</code>` : DASH],
  ]) +
  (f2p.length ? det(u, key('f2p'), `F2P tests <span class="muted small">${fmtInt(f2p.length)}${num(b.f2p) !== null && b.f2p > f2p.length ? ` of ${fmtInt(b.f2p)}` : ''}</span>`, list(f2p, key('f2p')), false, 'fold fold-inline') : '') +
  (p2p.length ? det(u, key('p2p'), `P2P tests (sample) <span class="muted small">${fmtInt(p2p.length)}${num(b.p2p) !== null ? ` of ${fmtInt(b.p2p)}` : ''}</span>`, list(p2p, key('p2p')), false, 'fold fold-inline') : '') +
  (text(b.problems) ? det(u, key('problems'), 'Gate / author problems', pre(b.problems, '', key('problems')), b.ok === false, 'fold fold-inline') : '');

  // Probe.
  html += '<h3 class="drawer-h">Probe</h3>';
  if (!p) {
    html += note(b.ok === false ? 'Never probed: the build failed.' : 'Not probed yet.');
  } else {
    // Before the score, the counts cover the graded slots only (NODE.probe.partial; no cheat audit yet): neutral tiles.
    const lp = liveProbe(n);
    const live = !!lp || (!sc && (p.live === true || n.status === 'probing'));
    const progRows = lp ? lp.rows.map((r) => [r.model, [r.done, r.total], r])
      : Object.entries(obj(p.progress)).map(([m, v]) => [m, pair(v), null]).filter(([, pr]) => pr);
    const prog = progRows.map(([m, pr, r]) => `<div class="prog-row"><span class="prog-k">${esc(m)}</span>` +
      `${bar(pr[0], pr[1], pr[0] >= pr[1] ? 'good' : 'pending', `${m} rollouts`)}<span class="prog-v">${fmtInt(pr[0])}/${fmtInt(pr[1])}` +
      `${r ? esc(` done · ${fmtInt(r.pass)} pass · ${fmtInt(r.fail)} fail${r.infra ? ` · ${fmtInt(r.infra)} infra` : ''}`) : ''}</span></div>`).join('');
    const asStats = (r) => (r ? {pass: r.pass, fail: r.fail, infra: r.infra, cheated: 0, clean: r.pass, valid: r.pass + r.fail} : null);
    const s = lp ? asStats(lp.rows.find((r) => r.model === STUDENT)) : probeStats(p, 'student');
    const t = lp ? asStats(lp.rows.find((r) => r.model === TEACHER)) : probeStats(p, 'teacher');
    const so = lp ? {} : obj(p.student), to = lp ? {} : obj(p.teacher);
    const soFar = live ? ' · so far' : '';
    const tiles = [];
    if (s) {
      const rate = num(so.rate) !== null ? so.rate : (s.valid ? s.pass / s.valid : null);
      tiles.push(tile(`${STUDENT} (student)${soFar}`, `${s.pass}/${s.valid}`,
                      esc(`pass rate ${fmtPct(rate)}${s.infra ? ` · ${s.infra} infra` : ''}`),
                      live || !s.valid ? 'neutral' : s.pass / s.valid <= 0.2 ? 'good' : 'bad'));
    }
    if (t) {
      const rate = num(to.clean_rate) !== null ? to.clean_rate : (t.valid ? t.clean / t.valid : null);
      tiles.push(tile(`${TEACHER} (teacher, clean)${soFar}`, `${t.clean}/${t.valid}`,
                      esc(`clean rate ${fmtPct(rate)}${live ? ' · not audited yet' : ` · ${t.cheated} cheated`}${t.infra ? ` · ${t.infra} infra` : ''}`),
                      live || !t.valid ? 'neutral' : t.clean / t.valid >= 0.6 ? 'good' : 'bad'));
    }
    html += (prog ? `<div class="prog">${prog}</div>` : '') +
            (lp && lp.ownDone ? note("Probe done for this task's own slots: it is scored when the round's probe ends.", 'note-wait') : '') +
            (tiles.length ? `<div class="tiles tiles-2">${tiles.join('')}</div>` : '') +
            (text(p.dir) ? `<div class="muted small">probe ${esc(p.dir)}${live ? ' (running)' : ''}</div>` : '') +
            renderRollouts(p.rollouts, u, key('rollouts'));
  }

  // Score.
  html += '<h3 class="drawer-h">Score</h3>';
  if (!sc) {
    html += note('Not scored yet.');
  } else {
    const valid = Array.isArray(sc.valid) ? `${STUDENT} ${fmtInt(sc.valid[0])} · ${TEACHER} ${fmtInt(sc.valid[1])}` : DASH;
    const gf = arr(sc.grading_failures).map(text).filter(Boolean);
    const rates = num(sc.student_rate) !== null || num(sc.teacher_clean_rate) !== null
      ? ['Pass rates', esc(`${STUDENT} ${fmtPct(sc.student_rate)} (accept ≤ 20%) · ${TEACHER} clean ${fmtPct(sc.teacher_clean_rate)} (accept ≥ 60%)`)]
      : null;
    html += kv([
      ['Accepted', sc.accept === true ? badge('yes', 'good') : sc.accept === false ? badge('no', 'muted') : DASH],
      ['Reason', esc(sc.reason) || DASH],
      rates,
      ['Similarity', num(sc.similarity) === null ? DASH : `${esc(sc.similarity)}/10 (accept ≥ 6)`],
      ['Valid rollouts', esc(valid)],
      ['Grading failures', gf.length ? esc(gf.join(', ')) : 'none'],
      ['Answer retrieval', sc.answer_retrieval === true ? badge('yes (leaky)', 'bad') : sc.answer_retrieval === false ? 'no' : DASH],
      ['Target rates', esc(`${STUDENT} ${fmtPct(sc.target_student_rate)} · ${TEACHER} ${fmtPct(sc.target_teacher_rate)}`)],
    ]) + renderSimilarity(sc.similarity_detail, key('similarity'), u);
  }

  // Failure-match audit of a task in its scope (v1.3: accepted; v1.4: eligible).
  html += safe('failure-match audit', () => renderFailureMatch(n, detail, u, key('failure_match'), now));

  // Long texts.
  const texts = [['instruction_md', 'Instruction (instruction.md)'], ['evolve_plan_md', 'Evolve plan (evolve_plan.md)'],
                 ['synthetic_failure_analysis_md', 'Synthetic failure analysis'], ['rubric_md', 'Rubric (rubric.md)']];
  html += '<h3 class="drawer-h">Documents</h3>' + texts.map(([f, label]) => {
    const t = text(x[f]);
    return t ? det(u, key(f), `${esc(label)} <span class="muted small">${fmtInt(t.length)} chars</span>`, pre(t, 'pre-md', key(f)), false, 'fold fold-inline')
             : `<div class="missing">${esc(label)}: not available</div>`;
  }).join('') + renderAudit(x.teacher_audit, key('teacher_audit'), u);
  if (num(now) !== null && num(obj(detail).generated_at) !== null) {
    html += `<p class="muted small">run detail generated ${esc(fmtAgo(detail.generated_at, now))}</p>`;
  }
  return html + '</div>';
}

/** The Task tab: accuracy always; for a converting task the run bar, failure patterns, plan, the round-0 table, the
    failure-match audit of the eligible tasks and the generation tree. v1.6: a task with batch runs shows the view of
    one run at a time (its official run first; opts.run, the hash's run=, picks a batch run), with a run selector when
    it has two or more runs; `detail` is the selected run's document. */
function renderTaskView(index, taskId, detail, opts) {
  const o = obj(opts), ui = obj(o.ui);
  const now = num(o.now) === null ? Date.now() / 1000 : o.now;
  if (!isObj(index)) return section('Loading', note('Waiting for data/index.json…'));
  if (!taskId) return section('No task selected', note('Pick a DeepSWE task in the selector above.'));
  const task = findTask(index, taskId);
  if (!task) return section('Unknown task', note(`"${text(taskId)}" is not in index.json. Pick a task in the selector above.`));
  let html = safe('accuracy', () => renderAccuracy(index, task));
  const views = viewsOf(index, task);
  if (!views.length) return html + safe('conversion', () => renderNoRun(task));
  const view = selectedView(views, o.run);
  if (views.length > 1) html += safe('runs', () => renderRunSelector(index, views, view));
  if (view.kind === 'batch') return html + renderBatchView(index, view.run, detail, o, now, ui);
  const run = view.run;
  // v1.8: the document of the run shown (a copy of another official run of the task is not this view's)
  const d = isObj(detail) && (!detail.task_id || detail.task_id === task.task_id) && !detail.source &&
            (view.older ? detail.run === run.run : !detail.run || !run.run || detail.run === run.run) ? detail : null;
  if (view.older) html += safe('older run', () => renderOlderRunNote(run, views[0].run));   // v1.8
  html += safe('run status', () => renderRunBar(run, d, now, ui));
  if (!d) {
    return html + section('Run detail', note(o.detailError ? `Run detail not available: ${text(o.detailError)}`
                                                          : `Loading data/${view.older ? text(run.detail) : `task/${text(taskId)}.json`}…`));
  }
  html += safe('failure patterns', () => renderFailureAnalysis(d.failure_analysis, ui, d.nodes));
  html += safe('plan', () => renderPlan(d.plan, ui));
  html += safe('round 0', () => renderRoundZero(d, ui));
  html += safe('failure-match audit', () => renderAuditTable(d, ui));
  html += safe('generation tree', () => renderTree(d, ui));
  return html;
}

/** v1.6, the view of one batch run: its status bar (with the batch's label and totals), then, from its document, the
    same failure patterns, plan, round-0 table and generation tree as an official run, with the batch's tasks in place of
    the failure-match audit table. */
function renderBatchView(index, run, detail, o, now, ui) {
  const r = obj(run);
  const d = isObj(detail) && detail.run === r.run && detail.source === r.source ? detail : null;
  const html = safe('run status', () => renderBatchRunBar(index, r, d, now, ui));
  if (!d) {
    return html + section('Run detail', note(o.detailError ? `Run detail not available: ${text(o.detailError)}`
                                                          : `Loading data/${text(r.detail)}…`));
  }
  return html + safe('failure patterns', () => renderFailureAnalysis(d.failure_analysis, ui, d.nodes)) +
         safe('plan', () => renderPlan(d.plan, ui)) + safe('round 0', () => renderRoundZero(d, ui)) +
         safe('batch tasks', () => renderBatchTable(index, d, ui)) + safe('generation tree', () => renderTree(d, ui));
}

/** v1.6: the runs of a task that has two or more (its official run first, then its batch runs, oldest launch first), one
    button each: data-run-select is the batch run's name ('' for the official run); the shown one is pressed. v1.8: the
    latest official run first ("latest" when the task has older official runs), then the older official runs (newest
    launch first; data-run-select is their name), then the batch runs. */
function renderRunSelector(index, views, current) {
  const officials = arr(views).filter((v) => v.kind !== 'batch').length;
  const tabs = arr(views).map((v) => {
    const r = obj(v.run), on = v === current, status = text(r.status), batch = v.kind === 'batch';
    const who = batch ? batchBadge(batchName(index, r), 'badge-xs') : badge('official', 'evolve', "hengyuf's official run", 'badge-xs') +
      (officials > 1 ? (v.older ? badge('older', 'muted', 'an older official run of this task: a later official run builds it again', 'badge-xs')
                                : badge('latest', 'neutral', 'the most recently launched official run of this task (its default view)', 'badge-xs')) : '');
    const facts = `${fmtInt(r.accepted)} accepted${batch ? ` · ${fmtInt(r.batch)} in batch` : ''}`;
    return `<button type="button" class="run-tab${on ? ' active' : ''}" data-run-select="${esc(batch || v.older ? r.run : '')}" ` +
           `aria-pressed="${on ? 'true' : 'false'}"><span class="run-tab-name">${esc(r.run) || DASH}</span>` +
           `<span class="run-tab-meta">${who}${status ? badge(status, lookup(RUN_TONES, status), text(r.stage), 'badge-xs') : ''}` +
           `<span class="muted small">${esc(facts)}</span></span></button>`;
  }).join('');
  return section('Runs', `<div class="run-tabs" role="group" aria-label="Runs that convert this task">${tabs}</div>`,
                 `<span class="muted small">${esc(plural(arr(views).length, 'run'))} convert this task · showing one</span>`, 'panel-runs');
}

/** v1.8, above an older official run's run bar: the task's latest official run is another (its default view), and this
    view shows the older run's own document (its tree, plan and failure-match audits). */
function renderOlderRunNote(run, latest) {
  const r = obj(run), l = obj(latest);
  return `<div class="older-run-note">${badge('older official run', 'muted', 'a later official run of this task builds it again', 'badge-sm')}` +
         `<span>${esc(`${text(r.run)}: an earlier official run of this task. ${text(l.run)} is its latest official run and its ` +
                      'default view; this view shows this run\'s own tree, plan and failure-match audits.')}</span></div>`;
}

/** v1.6, a batch run's status bar: the batch ("<name> batch · not audited", its label, the run's directory, the batch's
    own totals), then run, repo, status (finished: its build wrote summary.json; else stopped), the launches launch.jsonl
    lists (a batch's jobs are not polled), accepted, in batch, the audit ("not audited"), built / scored and rounds; the
    stage; no log (a batch's Slurm logs are not read). */
function renderBatchRunBar(index, run, detail, now, ui) {
  const r = obj(run), d = obj(detail), job = obj(d.job), info = batchInfo(index, r.source), name = batchName(index, r);
  const status = text(r.status), maxRounds = int(d.max_rounds), submitted = num(r.submitted_at);
  const jobs = (arr(job.jobs).length ? arr(job.jobs) : [r.job_id]).map(text).filter(Boolean);
  const totals = [num(info.tasks) !== null ? `${fmtInt(info.tasks)} tasks` : '', num(info.runs) !== null ? plural(info.runs, 'run') : '',
                  num(info.targets) !== null ? plural(info.targets, 'DeepSWE task') : ''].filter(Boolean).join(' · ');
  const strip = `<div class="batch-strip">${batchBadge(name)}<b class="batch-label">${esc(r.source_label || info.label || r.source)}</b>` +
                `<code class="batch-root" title="${esc(r.root)}">${esc(r.root)}</code>` +
                (totals ? `<span class="muted small">${esc(`the batch: ${totals}`)}</span>` : '') + '</div>';
  const launches = [submitted !== null ? `last submitted ${fmtDate(submitted)}` : '', text(r.partition)].filter(Boolean).join(' · ');
  const cells = [
    ['Run', `<b>${esc(r.run || d.run) || DASH}</b>`],
    ['Repo', `<code>${esc(r.repo || d.repo) || DASH}</code>`],
    ['Status', status ? badge(status, lookup(RUN_TONES, status), status === 'finished' ? 'its build wrote summary.json'
                                                                 : 'no summary.json: the run ended before its build finished') : DASH],
    ['Launches', `${esc(jobs.join(', ')) || DASH}${launches ? ` <span class="muted small">(${esc(launches)})</span>` : ''}` +
                 '<div class="muted small">from launch.jsonl · its jobs are not polled</div>'],
    ['Accepted', `<span class="big-num">${fmtInt(r.accepted)}</span>`],
    ['In batch', `<span class="big-num">${fmtInt(r.batch)}</span><div class="muted small">its tasks in outputs/tasks.jsonl</div>`],
    ['Failure-match audit', badge('not audited', 'muted', BATCH_HELP)],
    ['Built / scored', `${fmtInt(r.tasks_built)} / ${fmtInt(r.tasks_scored)}`],
    ['Rounds started', `${fmtInt(r.rounds_started)}${maxRounds !== null ? ` / ${maxRounds + 1}` : ''}`],
  ].map(([k, v]) => `<div class="runcell"><div class="runcell-k">${esc(k)}</div><div class="runcell-v">${v}</div></div>`).join('');
  const stage = text(r.stage || job.stage);
  return `<section class="panel panel-run panel-batch">${strip}<div class="runbar">${cells}</div>` +
         `<div class="stage"><span class="stage-k">Stage</span> ${esc(stage) || '<span class="muted">unknown</span>'}</div>` +
         note(`Log not read: ${name}'s Slurm logs are outside the dashboard's read roots.`, 'small') + '</section>';
}

/** v1.6, in place of the failure-match audit table for a batch run: its tasks in the batch (NODE.in_batch) and any
    accepted task that is not (marked), round by round: task and title, round, decision, sc14 passes, clean kimi passes,
    similarity, the audit ("not audited") and the task id. A row opens the node's drawer, as the audit table's rows do. */
function renderBatchTable(index, detail, ui) {
  const d = obj(detail), u = obj(ui), name = batchName(index, d);
  const shown = nodesOf(d).filter((n) => n.in_batch === true || (isAccepted(n) && !isAbandoned(n)));
  if (!shown.length) return '';
  const round = (n) => int(n.round);
  const order = (a, b) => (round(a) === null ? 1e9 : round(a)) - (round(b) === null ? 1e9 : round(b)) || cmpStr(a.id, b.id);
  const groups = [];
  for (const n of shown.slice().sort(order)) {
    const label = round(n) === null ? 'Round ?' : `Round ${round(n)}`;
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.nodes.push(n);
    else groups.push({label, nodes: [n]});
  }
  const row = (n) => {
    const id = text(n.id), p = obj(n.probe), sc = obj(n.score), dec = decisionOf(n);
    const s = probeStats(p, 'student'), t = probeStats(p, 'teacher');
    const infra = (x) => (x.infra ? ` <span class="muted small">+${fmtInt(x.infra)} infra</span>` : '');
    const audit = n.in_batch === true ? badge('not audited', 'muted', BATCH_HELP, 'badge-sm badge-fa')
      : badge('not in batch', 'serious', 'accepted by the run, but not in the batch (outputs/tasks.jsonl)', 'badge-sm');
    return `<tr class="fa-row${u.selected === id ? ' selected' : ''}" data-node="${esc(id)}">` +
           `<td class="fa-task"><button type="button" class="node-link" data-node="${esc(id)}" data-fkey="audit:${esc(id)}">${esc(shortId(id))}</button>` +
           `<div class="fa-title muted small" title="${esc(n.title)}">${esc(n.title) || 'untitled'}</div></td>` +
           `<td class="num">${round(n) === null ? DASH : fmtInt(n.round)}</td>` +
           `<td class="fa-decision">${badge(dec.label, dec.tone, text(n.decision_reason) || dec.help, 'badge-sm')}` +
           `${seedBadge(n, 'badge-xs badge-seed')}</td>` +
           `<td class="num">${s ? `${fmtInt(s.pass)}/${fmtInt(s.valid)}${infra(s)}` : DASH}</td>` +
           `<td class="num">${t ? `${fmtInt(t.clean)}/${fmtInt(t.valid)}${infra(t)}` : DASH}</td>` +
           `<td class="num">${num(sc.similarity) === null ? DASH : `${esc(fmtCompact(sc.similarity))}/10`}</td>` +
           `<td>${audit}</td><td class="fa-summary"><code>${esc(obj(n.build).task_id) || DASH}</code></td></tr>`;
  };
  const rows = groups.map((g) => `<tr class="fa-group"><th colspan="8" scope="rowgroup">${esc(g.label)} <span class="muted small">` +
                                 `${esc(plural(g.nodes.length, 'task'))}</span></th></tr>` + g.nodes.map(row).join('')).join('');
  const inBatch = shown.filter((n) => n.in_batch === true).length, outside = shown.length - inBatch;
  const sub = [`${plural(inBatch, 'task')} in ${name}'s batch`, 'not audited',
               outside ? `${fmtInt(outside)} accepted, not in the batch` : ''].filter(Boolean).join(' · ');
  return section(`${name} batch · not audited`,
    `<div class="tbl-wrap"${scrollAttr(`batch:${text(d.run)}`)}><table class="tbl tbl-fa tbl-batch"><thead><tr><th>task</th>` +
    '<th class="num">round</th><th>decision</th><th class="num">sc14 pass</th><th class="num">kimi clean</th>' +
    `<th class="num">similarity</th><th title="${esc(BATCH_HELP)}">audit</th><th>task id</th>` +
    `</tr></thead><tbody>${rows}</tbody></table></div>`,
    `<span class="muted small">${esc(sub)} · click a row for its details</span>`, 'panel-fa panel-batch-tasks');
}

// ------------------------------------------------------------------------------------------- charts

/** Clean ticks from 0: {top, ticks}, the smallest 1/2/2.5/5 x 10^k step giving at most count+1 intervals (integer
    series: whole steps, no 2.5). */
function niceTicks(max, count, integer) {
  const top = num(max) !== null && max > 0 ? max : 1;
  const want = Math.max(1, count || 4);
  const mag = Math.pow(10, Math.floor(Math.log10(top / want)));
  let step = 10 * mag;
  for (const m of integer ? [1, 2, 5, 10, 20] : [1, 2, 2.5, 5, 10, 20]) {
    const s = integer ? Math.max(1, Math.round(m * mag)) : m * mag;
    if (Math.ceil(top / s - 1e-9) <= want + 1) { step = s; break; }
  }
  const n = Math.max(1, Math.ceil(top / step - 1e-9));
  const ticks = [];
  for (let i = 0; i <= n; i += 1) ticks.push(Number((i * step).toPrecision(12)));
  return {top: ticks[ticks.length - 1], ticks};
}

/** Time ticks aligned to local clock boundaries; at most ~maxTicks of them. */
function timeTicks(t0, t1, maxTicks) {
  const steps = [600, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 172800, 432000, 604800];
  const span = Math.max(1, t1 - t0);
  const step = steps.find((s) => span / s <= (maxTicks || 8)) || 604800;
  const off = new Date(t0 * 1000).getTimezoneOffset() * 60;        // local = utc - off
  const ticks = [];
  for (let t = Math.ceil((t0 - off) / step) * step + off; t <= t1 && ticks.length < 50; t += step) ticks.push(t);
  return {step, ticks};
}

/** [[t, v], ...] (or [{t, v}]) -> finite points inside [t0, t1], time-sorted. */
function cleanPoints(points, t0, t1) {
  const out = [];
  for (const p of arr(points)) {
    const t = Array.isArray(p) ? num(p[0]) : num(obj(p).t);
    const v = Array.isArray(p) ? num(p[1]) : num(obj(p).v);
    if (t !== null && v !== null && t >= t0 && t <= t1) out.push([t, v]);
  }
  return out.sort((a, b) => a[0] - b[0]);
}

/** A line breaks where samples are missing for longer than 5 typical intervals (collector down). */
function gapSeconds(points) {
  const dts = [];
  for (let i = 1; i < points.length; i += 1) dts.push(points[i][0] - points[i - 1][0]);
  if (!dts.length) return Infinity;
  dts.sort((a, b) => a - b);
  return Math.max(300, 5 * dts[Math.floor(dts.length / 2)]);
}

function segments(points, gap) {
  const out = [];
  let cur = [];
  for (const p of points) {
    if (cur.length && p[0] - cur[cur.length - 1][0] > gap) { out.push(cur); cur = []; }
    cur.push(p);
  }
  if (cur.length) out.push(cur);
  return out;
}

/** Last point at or before each of `n` equal time buckets, per series: rows for tooltips and the table view. */
function bucketRows(seriesList, t0, t1, n) {
  const w = (t1 - t0) / n, rows = [];
  const idx = seriesList.map(() => 0);
  for (let i = 0; i < n; i += 1) {
    const end = t0 + (i + 1) * w;
    const vals = seriesList.map((s, k) => {
      let last = null;
      while (idx[k] < s.points.length && s.points[idx[k]][0] <= end) { last = s.points[idx[k]]; idx[k] += 1; }
      return last;
    });
    if (vals.some(Boolean)) {
      const t = Math.max(...vals.filter(Boolean).map((v) => v[0]));
      rows.push({i, start: t0 + i * w, end, t, vals: vals.map((v) => (v ? v[1] : null))});
    }
  }
  return rows;
}

/** Inline SVG line chart (no libraries): axes with ticks, hairline grid, 2px lines, area wash for one series,
    end-dot and direct label, an optional reference line (target/budget), hover bands with <title> tooltips, and a
    collapsible table twin. spec: {key, title, series: [{label, points}] (at most 2), extra: [{label, points, fmt}]
    (tooltip/table only), ref: {value, label}, refAlways, refScale, t0, t1, fmt, fmtEnd, integer, area, ui, empty,
    tools (trusted HTML from this file, shown in the caption)}. */
function renderLineChart(spec) {
  const s = obj(spec);
  const W = 640, H = 214, M = {l: 54, r: 82, t: 18, b: 26};
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const t1 = num(s.t1) === null ? Date.now() / 1000 : s.t1;
  const t0 = Math.min(num(s.t0) === null ? t1 - 86400 : s.t0, t1 - 60);
  const fmt = typeof s.fmt === 'function' ? s.fmt : fmtCompact;
  const fmtEnd = typeof s.fmtEnd === 'function' ? s.fmtEnd : fmt;     // the direct label at a line's end
  const series = arr(s.series).filter(isObj).slice(0, 2)
    .map((x, i) => ({label: text(x.label) || `series ${i + 1}`, cls: `s${i + 1}`, points: cleanPoints(x.points, t0, t1)}));
  const extra = arr(s.extra).filter(isObj).map((x) => ({label: text(x.label), points: cleanPoints(x.points, t0, t1), fmt: x.fmt}));
  let dataMax = 0, count = 0;
  for (const x of series) for (const p of x.points) { dataMax = Math.max(dataMax, p[1]); count += 1; }
  const ref = isObj(s.ref) && num(s.ref.value) !== null && s.ref.value > 0 ? s.ref : null;
  const refIn = !!ref && (s.refAlways === true || ref.value <= Math.max(dataMax, 1e-9) * (num(s.refScale) || 2.5));
  const {top, ticks} = niceTicks(Math.max(dataMax, refIn ? ref.value : 0), s.integer ? 5 : 4, !!s.integer);
  const X = (t) => M.l + ((t - t0) / (t1 - t0)) * pw;
  const Y = (v) => M.t + ph - (Math.max(0, v) / top) * ph;
  const f1 = (v) => v.toFixed(1);
  let svg = '';
  // Grid and y ticks.
  for (const v of ticks) {
    svg += `<line class="grid" x1="${M.l}" x2="${M.l + pw}" y1="${f1(Y(v))}" y2="${f1(Y(v))}"/>` +
           `<text class="tick" x="${M.l - 6}" y="${f1(Y(v) + 3.5)}" text-anchor="end">${esc(fmt(v))}</text>`;
  }
  // x ticks.
  const tt = timeTicks(t0, t1, 7);
  for (const t of tt.ticks) {
    const x = X(t);
    if (x < M.l + 14 || x > M.l + pw - 14) continue;
    svg += `<line class="xtick" x1="${f1(x)}" x2="${f1(x)}" y1="${M.t + ph}" y2="${M.t + ph + 4}"/>` +
           `<text class="tick" x="${f1(x)}" y="${M.t + ph + 16}" text-anchor="middle">${esc(fmtClock(t) === '00:00' || tt.step >= 86400 ? fmtDay(t) : fmtClock(t))}</text>`;
  }
  svg += `<line class="axis" x1="${M.l}" x2="${M.l + pw}" y1="${M.t + ph}" y2="${M.t + ph}"/>`;
  // Reference line (target / budget), or a note when it is far above the data.
  // The label sits at the left end, above the line (end labels own the right margin); a line at the top of the plot
  // gets its label in the top margin.
  if (ref && refIn) {
    const ly = Y(ref.value) - M.t >= 14 ? Y(ref.value) - 5 : M.t - 6;
    svg += `<line class="ref" x1="${M.l}" x2="${M.l + pw}" y1="${f1(Y(ref.value))}" y2="${f1(Y(ref.value))}"/>` +
           `<text class="ref-label halo" x="${M.l + 4}" y="${f1(ly)}">${esc(ref.label)}</text>`;
  } else if (ref) {
    svg += `<text class="ref-label" x="${M.l + pw}" y="${M.t - 6}" text-anchor="end">${esc(ref.label)} ↑ off scale</text>`;
  }
  // Lines (broken at data gaps), area wash for a single series, end dots with direct labels.
  const ends = [];
  series.forEach((x) => {
    const segs = segments(x.points, gapSeconds(x.points));
    for (const seg of segs) {
      if (series.length === 1 && seg.length > 1 && s.area !== false) {
        svg += `<path class="area ${x.cls}" d="M${f1(X(seg[0][0]))} ${f1(Y(0))}` +
               seg.map((p) => `L${f1(X(p[0]))} ${f1(Y(p[1]))}`).join('') + `L${f1(X(seg[seg.length - 1][0]))} ${f1(Y(0))}Z"/>`;
      }
      if (seg.length === 1) svg += `<circle class="dot-solo ${x.cls}" cx="${f1(X(seg[0][0]))}" cy="${f1(Y(seg[0][1]))}" r="2.5"/>`;
      else svg += `<path class="line ${x.cls}" d="${seg.map((p, i) => `${i ? 'L' : 'M'}${f1(X(p[0]))} ${f1(Y(p[1]))}`).join('')}"/>`;
    }
    const last = x.points[x.points.length - 1];
    if (last) ends.push({x, last});
  });
  ends.forEach((e, i) => {
    const cy = Y(e.last[1]);
    const clash = i > 0 && Math.abs(cy - Y(ends[0].last[1])) < 13;     // converging ends: legend + tooltip carry it
    svg += `<circle class="end ${e.x.cls}" cx="${f1(X(e.last[0]))}" cy="${f1(cy)}" r="4"/>`;
    if (!clash) svg += `<text class="end-label" x="${f1(X(e.last[0]) + 8)}" y="${f1(cy + 4)}">${esc(fmtEnd(e.last[1]))}</text>`;
  });
  // Hover bands: one per time bucket, <title> lists every series (and the extra fields) at that time.
  const all = series.concat(extra);
  const fmts = series.map(() => fmt).concat(extra.map((x) => (typeof x.fmt === 'function' ? x.fmt : fmtCompact)));
  if (count) {
    const nb = Math.min(120, Math.max(1, Math.round(pw / 4)));
    for (const row of bucketRows(all, t0, t1, nb)) {
      const x0 = X(row.start), x1 = X(row.end);
      const tip = [fmtDate(row.t)].concat(all.map((x, k) => (row.vals[k] === null ? '' : `${x.label}: ${fmts[k](row.vals[k])}`)).filter(Boolean)).join('\n');
      const dots = series.map((x, k) => (row.vals[k] === null ? '' : `<circle class="hov-dot ${x.cls}" cx="${f1(X(row.t))}" cy="${f1(Y(row.vals[k]))}" r="4"/>`)).join('');
      svg += `<g class="hov"><title>${esc(tip)}</title><rect class="hov-band" x="${f1(x0)}" y="${M.t}" width="${f1(Math.max(1, x1 - x0))}" height="${ph}"/>` +
             `<line class="hov-x" x1="${f1(X(row.t))}" x2="${f1(X(row.t))}" y1="${M.t}" y2="${M.t + ph}"/>${dots}</g>`;
    }
  } else {
    svg += `<text class="empty" x="${M.l + pw / 2}" y="${M.t + ph / 2}" text-anchor="middle">${esc(s.empty || 'no data in this window')}</text>`;
  }
  const legend = series.length > 1
    ? `<div class="chart-legend">${series.map((x) => `<span class="lg"><svg width="18" height="8" aria-hidden="true"><line class="line ${x.cls}" x1="1" y1="4" x2="17" y2="4"/></svg>${esc(x.label)}</span>`).join('')}</div>`
    : '';
  const latest = ends.length ? ends.map((e) => `${series.length > 1 ? `${e.x.label} ` : ''}${fmtEnd(e.last[1])}`).join(' · ') : DASH;
  const rows = count ? bucketRows(all, t0, t1, 24).reverse() : [];
  const table = rows.length
    ? `<table class="tbl tbl-compact"><thead><tr><th>time</th>${all.map((x) => `<th class="num">${esc(x.label)}</th>`).join('')}</tr></thead><tbody>` +
      rows.map((r) => `<tr><td>${esc(fmtDate(r.t))}</td>${r.vals.map((v, k) => `<td class="num">${v === null ? DASH : esc(fmts[k](v))}</td>`).join('')}</tr>`).join('') +
      '</tbody></table>'
    : note('No data.');
  const key = `chart:${text(s.key) || text(s.title)}`;
  return `<figure class="chart"><figcaption><span class="chart-title">${esc(s.title)}</span>${typeof s.tools === 'string' ? s.tools : ''}` +
         `<span class="chart-now">${esc(latest)}</span></figcaption>` +
         legend + `<svg class="chart-svg" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(s.title)}">${svg}</svg>` +
         det(s.ui, key, 'Table', `<div class="chart-table-scroll"${scrollAttr(`${key}:table`)}>${table}</div>`, false,
             'fold fold-inline chart-table') + '</figure>';
}

// ------------------------------------------------------------------------------------------- monitor

/** A worker's status as the fleet daemon buckets it (student/fleet.py): a ready/draining registry record counts as
    collect.py's `routing` says (ready / draining), or "stale" when it is not routable (heartbeat older than 120 s, or
    past its end); otherwise the registry status, else the Slurm state. */
function workerStatus(w) {
  const x = obj(w);
  const rs = text(x.registry_status).toLowerCase();
  if ((rs === 'ready' || rs === 'draining') && hasOwn(x, 'routing')) {
    const routing = text(x.routing).toLowerCase();
    return routing === 'ready' || routing === 'draining' ? routing : 'stale';
  }
  if (rs) return rs;
  const st = text(x.state).toUpperCase();
  return st === 'PENDING' ? 'pending' : st ? st.toLowerCase() : 'unknown';
}

/** What the router statistics cover (fleet.router.scope): the requests of the official runs' probes
    (student-router-requests.jsonl), not every request the workers serve. */
function routerScope(fleet) { return text(obj(obj(fleet).router).scope) || 'official-run probes'; }

/** Utilization: {busy, routable}, the routable workers with at least one running request. fleet.totals.busy_workers
    when the collector sends it, else counted from workers[].metrics (busy null when no routable worker was scraped). */
function busyWorkers(fleet) {
  const f = obj(fleet), tot = obj(f.totals);
  const serving = arr(f.workers).filter((w) => isObj(w) && ['ready', 'draining'].includes(workerStatus(w)));
  const routable = num(tot.routable) !== null ? tot.routable : serving.length;
  if (num(tot.busy_workers) !== null) return {busy: tot.busy_workers, routable};
  const scraped = serving.filter((w) => num(obj(w.metrics).running) !== null);
  return {busy: scraped.length ? scraped.filter((w) => w.metrics.running >= 1).length : null, routable};
}

function renderFleetSummary(fleet, now) {
  const f = obj(fleet), cfg = obj(f.config), cov = obj(f.coverage), c = obj(f.counts), tot = obj(f.totals);
  const r = obj(f.router), dmn = obj(f.daemon), lh = obj(r.last_hour);
  const scope = routerScope(f), util = busyWorkers(f);
  // vLLM's prompt-token counter includes prefix-cache hits, so prompt tok/s is no throughput figure (gen tok/s is).
  const prompt = num(tot.prompt_tps_incl_cache) !== null ? tot.prompt_tps_incl_cache : tot.prompt_tps;
  const target = num(cfg.target) !== null ? cfg.target : num(cov.target);
  const routable = num(tot.routable);
  const countText = ['ready', 'draining', 'starting', 'pending', 'held', 'stale', 'unhealthy', 'failed', 'stopped', 'vanished', 'other']
    .filter((k) => num(c[k]) || ['ready', 'draining', 'starting', 'pending'].includes(k))
    .map((k) => `${k} ${fmtInt(c[k])}`).join(' · ');
  const kvs = arr(f.workers).map((w) => num(obj(obj(w).metrics).kv_usage)).filter((v) => v !== null);
  const kvMax = kvs.length ? Math.max(...kvs) : null;
  const errRate = num(r.error_rate);
  const hb = num(dmn.heartbeat_age_s);
  const breaker = text(dmn.breaker);
  const daemonTone = breaker === 'open' || hb === null || hb > 600 ? 'bad' : hb > 180 || breaker === 'half-open' || breaker === 'half_open' ? 'pending' : 'good';
  const tiles = [
    tile('Routable workers', `${fmtInt(routable)}<span class="tile-den">/${fmtInt(target)}</span>`, esc(countText),
         routable === null ? 'neutral' : routable === 0 ? 'bad' : target !== null && routable < target ? 'pending' : 'good'),
    tile('Coverage', `${fmtInt(cov.value)}<span class="tile-den">/${fmtInt(cov.target)}</span>`,
         esc(`horizon ${fmtInt(cfg.horizon_effective_min)} min (base ${fmtInt(cfg.horizon_min)}) · max ${fmtInt(cfg.max_jobs)} jobs`),
         num(cov.value) !== null && num(cov.target) !== null && cov.value < cov.target ? 'pending' : 'neutral'),
    tile('Busy workers', `${fmtInt(util.busy)}<span class="tile-den">/${fmtInt(util.routable)}</span>`,
         esc(util.busy !== null && util.routable ? `utilization ${fmtPct(util.busy / util.routable)} · routable workers with ≥ 1 running request`
                                                 : 'routable workers with ≥ 1 running request'), 'neutral'),
    tile('Running requests', fmtInt(tot.running), esc(`waiting ${fmtInt(tot.waiting)}`), num(tot.waiting) ? 'pending' : 'neutral'),
    tile('KV cache (mean)', fmtPct(tot.kv_usage_mean, 1), kvMax === null ? '' : esc(`max ${fmtPct(kvMax, 1)} on one worker`),
         num(tot.kv_usage_mean) !== null && tot.kv_usage_mean > 0.9 ? 'pending' : 'neutral'),
    tile('Generation', `${fmtCompact(tot.gen_tps)}<span class="tile-den"> tok/s</span>`,
         esc(`prompt ${fmtCompact(prompt)} tok/s (incl. prefix-cache hits)`), 'neutral'),
    tile(`Router requests (${fmtDur(num(r.window_s) === null ? 600 : r.window_s)})`, fmtInt(r.requests),
         esc(`${scope} · ${fmtInt(r.errors)} errors (${fmtPct(errRate, 1)})${num(r.clamps) ? ` · ${fmtInt(r.clamps)} clamps` : ''}`),
         errRate !== null && errRate > 0.05 ? 'bad' : errRate ? 'pending' : 'neutral'),
    tile('Latency p50 / p90', `${fmtFixed(r.p50_s, 1)}<span class="tile-den"> / ${fmtFixed(r.p90_s, 1)} s</span>`,
         esc(`${scope} · last hour ${fmtInt(lh.requests)} requests, ${fmtInt(lh.errors)} errors`), 'neutral'),
    tile('Fleet daemon', hb === null ? DASH : `${fmtDur(hb)}<span class="tile-den"> ago</span>`,
         esc(`breaker ${breaker || DASH} · job ${text(dmn.job_id) || DASH}${text(dmn.host) ? ` on ${text(dmn.host)}` : ''}`), daemonTone),
  ];
  const age = num(f.generated_at);
  return section('Inference workers', `<div class="tiles">${tiles.join('')}</div>`,
    `<span class="muted small">student vLLM fleet${age !== null ? ` · fleet.json ${esc(fmtAgo(age, now))}` : ''}</span>`);
}

function renderWorkers(fleet, now) {
  const f = obj(fleet), by = obj(obj(f.router).by_worker), scope = routerScope(f);
  const workers = arr(f.workers).filter(isObj).slice().sort((a, b) => {
    const sa = num(Number(a.slot)), sb = num(Number(b.slot));
    return (sa === null ? 1e9 : sa) - (sb === null ? 1e9 : sb) || cmpStr(a.job_id, b.job_id);
  });
  if (!workers.length) return section('Workers', note('No fleet workers.'));
  const rows = workers.map((w) => {
    const st = workerStatus(w), m = isObj(w.metrics) ? w.metrics : null, rr = obj(by[text(w.job_id)]);
    const pendingNote = text(w.pending_reason) ? `<div class="muted small">${esc(w.pending_reason)}</div>` : '';
    const ready = num(w.ready_at) !== null ? `${esc(fmtClock(w.ready_at))} <span class="muted small">(${esc(fmtAgo(w.ready_at, now))})</span>`
      : num(w.est_start) !== null ? `<span class="muted">est. start ${esc(fmtClock(w.est_start))}</span>` : DASH;
    const hb = num(w.heartbeat_age_s);
    const merr = m && text(m.error) ? `<span class="err" title="${esc(m.error)}">scrape error</span>` : '';
    return `<tr><td class="num">${esc(w.slot) || DASH}</td><td><code title="${esc(w.base_url)}">${esc(w.job_id) || DASH}</code></td>` +
           `<td>${esc(w.partition) || DASH}</td><td>${esc(w.node) || DASH}</td>` +
           `<td>${badge(st, lookup(WORKER_TONES, st), text(w.state) ? `Slurm ${text(w.state)}` : null, 'badge-sm')}${pendingNote}</td>` +
           `<td>${ready}</td><td class="num">${num(w.left_s) === null || st === 'pending' ? DASH : esc(fmtDur(w.left_s))}</td>` +
           `<td class="num${hb !== null && hb > 120 ? ' txt-bad' : ''}">${hb === null ? DASH : esc(fmtDur(hb))}</td>` +
           `<td class="num">${m ? fmtInt(m.running) : DASH}</td><td class="num">${m ? fmtInt(m.waiting) : DASH}</td>` +
           `<td class="num">${m ? fmtPct(m.kv_usage, 1) : DASH}</td><td class="num">${m ? fmtCompact(m.gen_tps) : DASH}${merr}</td>` +
           `<td class="num">${fmtInt(rr.requests)}${num(rr.errors) ? ` <span class="txt-bad">(${fmtInt(rr.errors)} err)</span>` : ''}</td>` +
           `<td class="num">${num(rr.p90_s) === null ? DASH : `${fmtFixed(rr.p90_s, 1)} s`}</td></tr>`;
  }).join('');
  return section('Workers', `<div class="tbl-wrap"${scrollAttr('workers')}><table class="tbl"><thead><tr><th class="num">slot</th><th>job</th>` +
    '<th>partition</th><th>node</th><th>status</th><th>ready since</th><th class="num">time left</th><th class="num">heartbeat</th>' +
    '<th class="num">running</th><th class="num">waiting</th><th class="num">KV</th><th class="num">gen tok/s</th>' +
    `<th class="num" title="${esc(scope)}">router req (10 min)</th><th class="num" title="${esc(scope)}">p90 (10 min)</th></tr></thead>` +
    `<tbody>${rows}</tbody></table></div>`,
    `<span class="muted small">${esc(plural(workers.length, 'job'))} · router columns: ${esc(scope)}</span>`);
}

function renderFleetCharts(fleet, now, ui) {
  const f = obj(fleet), h = arr(f.history).filter(isObj);
  const pts = (k) => h.map((p) => [p.t, p[k]]);
  const target = num(obj(f.config).target) !== null ? f.config.target : num(obj(f.coverage).target);
  const base = {t0: now - 86400, t1: now, ui, empty: 'no samples in the last 24 h'};
  const charts = [
    renderLineChart(Object.assign({}, base, {key: 'routable', title: 'Routable workers', integer: true, area: false,
      series: [{label: 'routable', points: pts('routable')}], extra: [{label: 'ready', points: pts('ready')}],
      ref: target === null ? null : {value: target, label: `target ${fmtInt(target)}`}, refAlways: true, fmt: fmtInt})),
    renderLineChart(Object.assign({}, base, {key: 'requests-inflight', title: 'Running requests', integer: true,
      series: [{label: 'running', points: pts('running')}, {label: 'waiting', points: pts('waiting')}], fmt: fmtInt,
      extra: [{label: 'KV mean', points: pts('kv'), fmt: (v) => fmtPct(v, 1)}]})),
    renderLineChart(Object.assign({}, base, {key: 'gen-tps', title: 'Generation throughput (tok/s)',
      series: [{label: 'gen tok/s', points: pts('gen_tps')}]})),
    renderLineChart(Object.assign({}, base, {key: 'router-req', integer: true,
      title: `Router requests per cycle (${fmtDur(num(obj(f.router).window_s) === null ? 600 : f.router.window_s)} window, ${routerScope(f)})`,
      series: [{label: 'requests', points: pts('req')}], fmt: fmtInt,
      extra: [{label: 'errors', points: pts('err'), fmt: fmtInt}, {label: 'p90 s', points: pts('p90_s'), fmt: (v) => fmtFixed(v, 1)}]})),
  ];
  return section('Fleet, last 24 h', `<div class="charts">${charts.join('')}</div>`,
    `<span class="muted small">${esc(plural(h.length, 'sample'))} · hover a chart for values</span>`);
}

/** Hub API panel: spend/budget progress (red under $500 left), the hub sample's age, burn over its actual windows,
    ETA, spend chart with budget, per-run usage with the health of the incremental usage scan. */
function renderApi(api, now, ui) {
  const a = obj(api), hub = obj(a.hub), burn = obj(a.burn), scan = isObj(a.scan) ? a.scan : null;
  const spend = num(hub.spend), max = num(hub.max_budget);
  const remaining = num(hub.remaining) !== null ? hub.remaining : spend !== null && max !== null ? max - spend : null;
  const low = remaining !== null && remaining < 500;
  const used = spend !== null && max ? spend / max : null;
  const eta = num(burn.eta_exhaust) !== null ? burn.eta_exhaust : isoToEpoch(burn.eta_exhaust_iso);
  const burnKnown = num(burn.per_hour_1h) !== null || num(burn.per_hour_6h) !== null;
  const r1 = num(burn.per_hour_1h) || 0, r6 = num(burn.per_hour_6h) || 0;
  const etaRate = r1 >= r6 ? '1 h' : '6 h';   // collect.py _burn(): the higher of the two rates (conservative)
  // The hub is sampled at most every 10 min: its age, from sampled_at (else age_s as of the cycle).
  const sampled = num(hub.sampled_at);
  const sampleAge = sampled !== null ? now - sampled
    : num(hub.age_s) !== null ? hub.age_s + (num(a.generated_at) !== null ? Math.max(0, now - a.generated_at) : 0) : null;
  const oldSample = sampleAge !== null && sampleAge > 30 * 60;
  // Each burn rate is measured over the samples it found: window_min_* is that span (minutes).
  const burnTile = (label, rate, windowMin, need) => tile(label,
    num(rate) === null ? DASH : `${fmtMoney(rate)}<span class="tile-den">/h</span>`,
    esc(num(windowMin) !== null ? `over the last ${fmtDur(windowMin * 60)} of hub samples` : num(rate) === null ? need : ''), 'neutral');
  const tiles = [
    tile('Spend', fmtMoney(spend), esc([used === null ? '' : `${fmtPct(used, 1)} of the budget`,
                                        sampleAge === null ? '' : `sampled ${fmtDur(sampleAge)} ago`].filter(Boolean).join(' · ')),
         oldSample ? 'pending' : 'neutral'),
    tile('Budget', fmtMoney(max), esc(`key ${text(hub.key) || DASH}`), 'neutral'),
    tile('Remaining', fmtMoney(remaining), low ? '<b class="txt-bad">below $500</b>' : '', low ? 'bad' : 'good'),
    burnTile('Burn (1 h)', burn.per_hour_1h, burn.window_min_1h, 'not enough hub samples in the last hour'),
    burnTile('Burn (6 h)', burn.per_hour_6h, burn.window_min_6h, 'not enough hub samples in the last 6 h'),
    tile('Exhaustion ETA', eta === null ? 'not projected' : esc(fmtDate(eta)),
         esc(eta !== null ? `in ${fmtDur(eta - now)} at the ${etaRate} burn` : burnKnown ? 'no recent spend' : 'burn rate not known yet (needs more hub samples)'),
         eta !== null && eta - now < 86400 ? 'bad' : 'neutral'),
  ];
  const meter = `<div class="meter${low ? ' meter-bad' : ''}" role="meter" aria-label="budget used" aria-valuemin="0" aria-valuemax="100"` +
                ` aria-valuenow="${used === null ? 0 : (used * 100).toFixed(1)}"><span class="meter-fill" style="width:${used === null ? 0 : Math.min(100, Math.max(0, used * 100)).toFixed(1)}%"></span></div>` +
                `<div class="meter-legend"><span>${esc(fmtMoney(spend))} spent</span><span>${esc(fmtMoney(remaining))} left of ${esc(fmtMoney(max))}</span></div>`;
  const sample = `<div class="muted small${oldSample ? ' txt-bad' : ''}">hub sample ${esc(sampleAge === null ? DASH : `${fmtDur(sampleAge)} old`)}` +
                 `${sampled !== null ? ` (${esc(fmtClock(sampled, true))})` : ''}${num(hub.status) !== null ? ` · HTTP ${esc(hub.status)}` : ''}` +
                 ' · sampled at most every 10 min</div>' +
                 (text(hub.error) ? `<div class="alert alert-warn">hub sample error: ${esc(hub.error)}</div>` : '');
  // Per-run usage comes from an incremental scan of the probes' logs: complete, or still catching up.
  const scanNote = !scan ? '' : scan.complete === true
    ? `<span class="muted small">usage scan complete${num(scan.files) !== null ? ` · ${esc(plural(scan.files, 'log'))}` : ''}</span>`
    : badge(`usage scan catching up${num(scan.pending_bytes) !== null ? `: ${fmtBytes(scan.pending_bytes)} to read` : ''}`, 'pending',
            'The collector reads the probe logs incrementally; the per-run numbers below are partial until it catches up.');
  const hist = arr(a.history).filter(isObj);
  const ts = hist.map((p) => num(p.t)).filter((t) => t !== null);
  const t0 = ts.length ? Math.min(Math.min(...ts), now - 6 * 3600) : now - 86400;
  const budget = max !== null ? max : (hist.length ? num(hist[hist.length - 1].max_budget) : null);
  // The budget line is drawn to scale once spend is within a third of it; before that the chart fits the spend (the
  // trend stays readable) and a toggle rescales it to show the budget line.
  const peak = hist.reduce((m, p) => Math.max(m, num(p.spend) || 0), spend || 0);
  const farBudget = budget !== null && budget > peak * 3;
  const showBudget = obj(ui).spendScale === 'budget';
  const chart = renderLineChart({key: 'spend', title: 'Hub spend', t0, t1: now, ui, fmt: fmtMoneyShort, fmtEnd: fmtMoney,
    series: [{label: 'spend', points: hist.map((p) => [p.t, p.spend])}],
    extra: [{label: 'budget', points: hist.map((p) => [p.t, p.max_budget]), fmt: fmtMoneyShort}],
    ref: budget === null ? null : {value: budget, label: `budget ${fmtMoneyShort(budget)}`}, refScale: 3, refAlways: showBudget,
    tools: farBudget ? `<button type="button" class="btn btn-xs" data-action="spend-scale" aria-pressed="${showBudget}">` +
                       `${showBudget ? 'Fit to spend' : 'Show budget line'}</button>` : '',
    empty: 'no hub samples yet'});
  const runs = arr(a.by_run).filter(isObj);
  const sum = (path) => runs.reduce((acc, r) => {
    const v = num(obj(r[path[0]])[path[1]]);
    return v === null ? acc : (acc || 0) + v;
  }, null);
  const tok = (o, keys) => keys.map((k) => fmtCompact(obj(o)[k])).join(' / ');
  const row = (name, kimi, codex, glm, cls) => `<tr${cls ? ` class="${cls}"` : ''}><td>${name}</td><td class="num">${fmtMoney(kimi.cost)}</td><td class="num">${fmtInt(kimi.calls)}</td>` +
    `<td class="num">${tok(kimi, ['input_tokens', 'cached_tokens', 'output_tokens'])}</td><td class="num">${fmtInt(codex.sessions)}</td>` +
    `<td class="num">${tok(codex, ['input_tokens', 'cached_tokens', 'output_tokens', 'reasoning_tokens'])}</td><td class="num">${fmtInt(glm.calls)}</td></tr>`;
  const body = runs.map((r) => row(`<b>${esc(r.run) || DASH}</b>`, obj(r.kimi), obj(r.codex), obj(r.glm))).join('');
  const total = runs.length > 1 ? row('total',
    {cost: sum(['kimi', 'cost']), calls: sum(['kimi', 'calls']), input_tokens: sum(['kimi', 'input_tokens']), cached_tokens: sum(['kimi', 'cached_tokens']), output_tokens: sum(['kimi', 'output_tokens'])},
    {sessions: sum(['codex', 'sessions']), input_tokens: sum(['codex', 'input_tokens']), cached_tokens: sum(['codex', 'cached_tokens']), output_tokens: sum(['codex', 'output_tokens']), reasoning_tokens: sum(['codex', 'reasoning_tokens'])},
    {calls: sum(['glm', 'calls'])}, 'row-total') : '';
  const table = runs.length
    ? '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>run</th><th class="num">kimi $</th><th class="num">kimi calls</th>' +
      '<th class="num">kimi tokens in / cached / out</th><th class="num">codex sessions</th><th class="num">codex tokens in / cached / out / reasoning</th>' +
      `<th class="num">glm calls</th></tr></thead><tbody>${body}${total}</tbody></table></div>`
    : note('No per-run usage recorded yet.');
  const notes = arr(a.notes).map(text).filter(Boolean);
  const age = num(a.generated_at);
  return section('Hub API usage', `<div class="tiles">${tiles.join('')}</div>${meter}${sample}<div class="charts charts-1">${chart}</div>` +
    `<h3 class="sub-h">Usage by run ${scanNote}</h3>${table}${notes.length ? `<ul class="notes">${notes.map((x) => `<li>${esc(x)}</li>`).join('')}</ul>` : ''}`,
    `<span class="muted small">${age !== null ? `api.json ${esc(fmtAgo(age, now))}` : ''}</span>`);
}

function fmtMoneyShort(v) {
  const n = num(v);
  if (n === null) return DASH;
  return Math.abs(n) >= 1000 ? `$${fmtCompact(n)}` : `$${n.toFixed(Math.abs(n) >= 100 || Number.isInteger(n) ? 0 : 2)}`;
}

/** The Monitor tab: fleet summary, worker table, 24 h charts, hub API panel. */
function renderMonitor(fleet, api, opts) {
  const o = obj(opts), ui = obj(o.ui);
  const now = num(o.now) === null ? Date.now() / 1000 : o.now;
  let html = arr(obj(o.index).queues).filter(isObj)                 // v1.8: the synthesis queues (index.json)
    .map((q) => safe('synthesis queue', () => renderQueue(q, o.index, now, ui))).join('');
  if (!isObj(fleet)) {
    html += section('Inference workers', note(o.fleetError ? `fleet.json not available: ${text(o.fleetError)}` : 'Loading data/fleet.json…'));
  } else {
    html += safe('fleet summary', () => renderFleetSummary(fleet, now)) + safe('workers', () => renderWorkers(fleet, now)) +
            safe('fleet charts', () => renderFleetCharts(fleet, now, ui));
  }
  if (!isObj(api)) html += section('Hub API usage', note(o.apiError ? `api.json not available: ${text(o.apiError)}` : 'Loading data/api.json…'));
  else html += safe('API usage', () => renderApi(api, now, ui));
  return html;
}

/** v1.8, the Monitor's section of one synthesis queue (index.json queues[]): the runner (alive / stale / stopped / never
    ran, its host, job and heartbeat age; the age and the liveness on the page's own clock, as every other age here: a
    heartbeat at most 15 min old whose note is not "stopped", whatever the index's age_s / alive said when the collector
    read the files), the budget line (the active hub key's NAME and its remaining dollars, as the
    runner last checked), its flags (launching stopped for budget, student serving), the runs in queue order (run with
    a link to its view once launched, DeepSWE task, repo, the batch whose failure analysis it reuses, status, leg,
    winner and candidates) and the runner log's last lines. */
function renderQueue(queue, index, now, ui) {
  const q = obj(queue), runner = obj(q.runner), name = text(q.name) || 'queue';
  const runs = arr(q.runs).filter(isObj);
  const beat = num(runner.heartbeat_at);
  const alive = beat !== null && now - beat <= QUEUE_STALE_S && text(runner.note) !== 'stopped';
  const state = alive ? badge('runner alive', 'good', 'its heartbeat is at most 15 min old')
    : beat === null ? badge('runner never ran', 'muted', 'no heartbeat.json yet: the runner has not started')
    : text(runner.note) === 'stopped' ? badge('runner stopped', 'muted', 'the runner finished and stopped')
    : badge('runner stale', 'bad', 'no heartbeat for more than 15 min: the runner is gone (its jobs go on)');
  const who = [text(runner.note) && runner.note !== 'stopped' ? text(runner.note) : '', runner.host ? `on ${text(runner.host)}` : '',
               runner.slurm_job_id ? `job ${text(runner.slurm_job_id)}` : '',
               beat !== null ? `heartbeat ${fmtAgo(beat, now)}` : ''].filter(Boolean).join(' · ');
  const b = isObj(q.budget) ? q.budget : null;
  const budget = b ? `${text(b.key) || 'key ?'}: ${num(b.remaining) === null ? DASH : fmtMoney(b.remaining)} remaining` +
                     (num(b.at) !== null ? ` (checked ${fmtAgo(b.at, now)})` : '') : 'not checked yet';
  const flags = [q.stop_launching === true ? badge('launching stopped', 'bad', 'the last key is below its floor: no new builds (running ones go on)', 'badge-sm') : '',
                 badge(q.serving === true ? 'student serving on' : 'student serving off', q.serving === true ? 'good' : 'muted',
                       'the runner sets the fleet target once a running build probes', 'badge-sm')].join('');
  const counts = QUEUE_ORDER.map((s) => [s, runs.filter((r) => r.status === s).length]).filter(([, n]) => n)
    .map(([s, n]) => `${fmtInt(n)} ${s.replace(/_/g, ' ')}`).join(' · ');
  const official = new Map(arr(obj(index).runs).filter((r) => isObj(r) && isOfficial(r.run) && !isBatchRun(r)).map((r) => [r.run, r]));
  const rows = runs.map((r) => {
    const entry = official.get(r.run), status = text(r.status);
    const link = entry ? taskHash(r.task, null, entry.latest === false ? r.run : null) : '';
    const cands = arr(r.candidates).map(text).filter(Boolean);
    return `<tr><td>${link ? `<a href="${esc(link)}"><code>${esc(r.run)}</code></a>` : `<code>${esc(r.run)}</code>`}</td>` +
           `<td><code>${esc(r.task)}</code></td><td><code>${esc(r.repo) || DASH}</code></td><td>${esc(r.batch) || DASH}</td>` +
           `<td>${status ? badge(status.replace(/_/g, ' '), lookup(QUEUE_TONES, status), status === 'queued' ? 'not launched yet' : '', 'badge-sm') : DASH}</td>` +
           `<td class="num">${fmtInt(r.leg)}</td><td>${esc(text(r.winner)) || DASH}</td>` +
           `<td class="small">${esc(cands.join(', ')) || DASH}</td></tr>`;
  }).join('');
  const table = runs.length
    ? `<div class="tbl-wrap"${scrollAttr(`queue:${name}`)}><table class="tbl tbl-compact tbl-queue"><thead><tr><th scope="col">run</th>` +
      '<th scope="col">DeepSWE task</th><th scope="col">repo</th><th scope="col" title="the batch whose failure analysis the run reuses">analysis</th>' +
      '<th scope="col">status</th><th class="num" scope="col" title="launches so far (each a pair of candidate jobs)">leg</th>' +
      `<th scope="col">winner</th><th scope="col">candidates</th></tr></thead><tbody>${rows}</tbody></table></div>`
    : note('The queue lists no run.');
  const tail = arr(q.log_tail).map(text);
  const log = det(ui, `queue:${name}:log`, `<span>runner.log, last ${esc(plural(tail.length, 'line'))}</span>`,
                  tail.length ? pre(tail.join('\n'), 'pre-log', `queue:${name}:log`) : note('No log yet.'), false, 'fold');
  const head = `<div class="queue-head">${state}<span class="muted small">${esc(who)}</span></div>` +
               `<div class="queue-head"><span class="queue-budget">${esc(`Hub budget · ${budget}`)}</span>${flags}</div>` +
               `<p class="muted small">${esc([`${plural(runs.length, 'run')}${num(q.max_active) !== null ? `, at most ${fmtInt(q.max_active)} at a time` : ''}`,
                                               counts, num(q.updated_at) !== null ? `state ${fmtAgo(q.updated_at, now)}` : ''].filter(Boolean).join(' · '))}</p>`;
  return section(`Synthesis queue ${name}`, head + table + log, `<span class="muted small">${esc(text(q.root))}</span>`, 'panel-queue');
}

// ------------------------------------------------------------------------------------------- explorer (v1.7)

/* CONTRACT "v1.7" (hengyuf 2026-10-22: filter the synthetic tasks and their trajectories by evolve round, similarity,
   Nemotron (sc14) / Kimi pass rates, failure-match audit verdict and trajectory outcome; show the tasks and summary
   statistics: filtered counts, coverage per DeepSWE task, coverage of the failure patterns, ...). data/explorer.json
   holds column-headed rows, one per built synthetic task of the task and run files and one per rollout their probes
   list. Everything here is pure: explorerModel() turns the file into objects (cached per document), parseExplorerParams()
   / explorerHash() round-trip the filters through the URL hash (#explorer&...), filterExplorer() applies them,
   explorerStats() counts, and the render* functions return HTML strings (every data string escaped). Charts are HTML/CSS
   bars: categorical slots 1-2 (--s1 / --s2) for the two sources (by their order in the file, never by rank), slot 3
   (--s3, a hue no source wears) for the single-series histograms, an ordinal one-hue ramp (--ord-yes / --ord-partial)
   for reproduced yes / partial; every mark has a hover / focus tooltip (data-tip, shown by the wiring) on a hit target
   larger than the mark, every chart a table; each panel of charts is one tab stop (arrow keys move between its marks). */

const EX_PAGE = 50;                                                   // rows per page of both tables
const EX_MODELS = {sc14: 'Nemotron (sc14)', kimi: 'Kimi K3'};          // the trajectories' models: student, teacher
const EX_AUDITS = ['strong', 'partial', 'weak', 'none', 'not_audited', 'not_eligible'];
const EX_AUDIT_TEXT = {strong: 'strong', partial: 'partial', weak: 'weak', none: 'none', not_audited: 'not audited',
                       not_eligible: 'not eligible'};
const EX_AUDIT_HELP = {
  not_audited: 'no failure-match audit of this task: an eligible official task not audited yet, or a task of a batch ' +
               '(Baihe\'s batch is never audited)',
  not_eligible: 'outside the failure-match audit\'s scope (discarded, withdrawn, not scored yet, no graded sc14 rollout, ' +
                'or an abandoned build); an audit that names it is not counted',
};
const EX_STATUSES = ['pass', 'fail', 'infra'];
const EX_KINDS = ['mask', 'evolve'];
const EX_DECISIONS = [['all', 'all'], ['acc', 'accepted'], ['not', 'not accepted']];
const EX_RESCORED = [['all', 'all'], ['only', 'only re-scored'], ['none', 'none re-scored']];
const EX_SERIES = ['s1', 's2'];                                       // categorical slots 1-2 (style.css --s1 / --s2)
const EX_HIST = 's3';                                                 // slot 3: the single-series histograms
const EX_TIP_KEYS = new Set(['s1', 's2', 's3', 's-other', 'seg-yes', 'seg-partial']);   // tooltip line keys: classes
const EX_LIST_KEYS = ['src', 'fam', 'round', 'kind', 'audit', 'tm', 'ts'];   // checkbox chips: null = every value
// The range filters: similarity 0..10; the pass rates are fractions 0..1 in the state and the hash, percent in the inputs.
const EX_RANGES = {smin: [0, 10], smax: [0, 10], nmin: [0, 1], nmax: [0, 1], kmin: [0, 1], kmax: [0, 1]};
const EX_PERCENT = new Set(['nmin', 'nmax', 'kmin', 'kmax']);
const EX_RANGE_GROUPS = {s: ['smin', 'smax', 'similarity'], n: ['nmin', 'nmax', 'Nemotron pass rate'],
                         k: ['kmin', 'kmax', 'Kimi pass rate']};
const EX_SORTS = ['task', 'target', 'source', 'round', 'decision', 'sim', 'srate', 'krate', 'rescored', 'audit', 'f2p', 'traj'];
const EX_TOKEN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,200}$/;               // ids, sources, enum values in the hash
const EX_MODEL_CACHE = new WeakMap();                                  // explorer.json document -> explorerModel()
const EX_EXPORTS = ['tasks.csv', 'tasks.json', 'trajectories.csv', 'trajectories.json'];
const EX_SORT_DESC = new Set(['sim', 'srate', 'krate', 'rescored', 'f2p', 'traj']);   // numbers: largest first, first click
const EX_RESCORED_HELP = 'Re-scored: these rates come from metrics.json, which was rewritten after the probe (a regrade or ' +
  'recovery re-scored the task, or main withdrew it); the rollout rows (rollouts/index.md, the trajectories here) show the ' +
  'probe as it was recorded, and its own rates differ.';
const EX_ABANDONED_NOTE = 'Abandoned build roots (runs/<run>/abandoned-<repo>, set aside by main) are never read, so never ' +
  'rows; "abandoned" here is a node\'s decision (its probe was superseded, or its parent was an invalid seed), and such ' +
  'tasks are rows.';

/** "1 trajectory" / "12 trajectories". */
function exCount(n, one, many) { return `${fmtInt(n)} ${n === 1 ? one : many}`; }

/** The Explorer's default filters: everything shown; sorted as the collector wrote the rows, first pages, coverage of
    the synthetic tasks. A list is null for "every value" (the chips) or the values chosen; `t` (DeepSWE tasks) is null
    for every task. `nj`: with a similarity bound set, also keep the tasks the similarity judge never scored. */
function explorerDefaults() {
  return {src: null, fam: null, round: null, kind: null, dec: 'all', smin: null, smax: null, nj: false, nmin: null,
          nmax: null, kmin: null, kmax: null, kb: 'clean', rd: 'all', audit: null, t: null, run: null, tm: null, ts: null,
          tc: false, tw: false, sort: null, desc: false, p: 1, tp: 1, cov: 'tasks'};
}

/** v1.8: a run's family, the Explorer's quick run filter: an official run's name prefix ("official1", "official2"), a
    batch run's source ("baiheh", shown as its name, "Baihe"). */
function exFamily(runName, source) {
  const s = obj(source);
  if (!s.audited) return text(s.source) || 'unknown';
  const m = /^(official\d*)/.exec(text(runName));
  return m ? m[1] : 'official';
}

/** Column positions of a column-headed table ({name: index}). */
function exColumns(cols) {
  const out = Object.create(null);
  arr(cols).forEach((c, i) => { if (typeof c === 'string' && !(c in out)) out[c] = i; });
  return out;
}

/** A rate (0..1) as a whole percent, the unit of the pass-rate inputs, columns and tables. */
function exPct(v) { const n = num(v); return n === null ? null : Math.round(n * 100); }

/** explorer.json as objects (null for anything else), cached per document: sources (with their series slot), runs,
    targets (failure patterns), tasks and trajectories, every row normalized (never throws on missing or mistyped
    cells). A task's `link` is the deep link the task view opens it by (runs[].link + "&node=<id>", app.js taskHash), its
    `dir` the candidate dir and `rollouts` (scored tasks) the dir of its transcripts: <rollouts>/<slot>.md / .patch;
    `rowsRates` its rates as its rollout rows give them (task_builder score()'s formulas: shown next to a re-scored task's
    metrics.json rates). */
function explorerModel(doc) {
  if (!isObj(doc)) return null;
  if (EX_MODEL_CACHE.has(doc)) return EX_MODEL_CACHE.get(doc);
  const sources = [], bySource = new Map();
  const addSource = (s) => {
    const x = obj(s), key = text(x.source);
    if (!key || bySource.has(key)) return bySource.get(key) || null;
    const slot = sources.length;
    const out = {source: key, name: text(x.name) || key, label: text(x.label) || key, audited: x.audited === true,
                 cls: EX_SERIES[slot] || 's-other'};
    sources.push(out);
    bySource.set(key, out);
    return out;
  };
  arr(doc.sources).forEach(addSource);
  const runs = arr(doc.runs).map((r, i) => {
    const x = obj(r), source = addSource({source: text(x.source) || 'unknown'});
    const family = exFamily(x.run, source);
    // v1.8: latest (an official run: false for an older run of its target; a v1.7 file: every official run is latest)
    return {i, run: text(x.run), source, target: text(x.deepswe_task), root: text(x.build_root), detail: text(x.detail),
            link: text(x.link), maxRounds: int(x.max_rounds), family, familyLabel: source.audited ? family : source.name,
            latest: source.audited ? x.latest !== false : null};
  });
  // v1.8: the run families (official ones by name, then each batch source) and every run name, for the run filter
  const families = [];
  for (const r of runs.slice().sort((a, b) => Number(!a.source.audited) - Number(!b.source.audited) ||
                                              (a.source.audited ? text(a.family).localeCompare(text(b.family), 'en', {numeric: true}) : 0))) {
    if (!families.some((f) => f.key === r.family)) families.push({key: r.family, label: r.familyLabel, source: r.source.source});
  }
  const runNames = [...new Set(runs.map((r) => r.run).filter(Boolean))];
  const targets = new Map();
  for (const t of arr(doc.targets)) {
    const x = obj(t), id = text(x.deepswe_task);
    if (!id || targets.has(id)) continue;
    targets.set(id, {id, runs: arr(x.runs).map((k) => runs[int(k)]).filter(Boolean),
                     patterns: arr(x.patterns).map((p) => ({name: text(obj(p).name), category: text(obj(p).category)}))});
  }
  for (const r of runs) {
    if (r.target && !targets.has(r.target)) targets.set(r.target, {id: r.target, runs: [r], patterns: []});
  }
  const tc = exColumns(doc.task_columns), rc = exColumns(doc.trajectory_columns);
  const cell = (row, cols, name) => (name in cols ? row[cols[name]] : undefined);
  const flag = (v) => v === true;
  const idxList = (v) => (Array.isArray(v) ? v.map(int).filter((k) => k !== null && k >= 0) : null);
  const byIndex = [], tasks = [];
  arr(doc.tasks).forEach((row, i) => {
    if (!Array.isArray(row)) return;
    const run = runs[int(cell(row, tc, 'run'))];
    if (!run) return;
    const node = text(cell(row, tc, 'node')), round = int(cell(row, tc, 'round')), scored = flag(cell(row, tc, 'scored'));
    const audit = text(cell(row, tc, 'audit'));
    const valid = cell(row, tc, 'valid');
    const t = {
      i, run, source: run.source.source, target: run.target, node, taskId: text(cell(row, tc, 'task_id')), round,
      branch: int(cell(row, tc, 'branch')), kind: text(cell(row, tc, 'kind')), title: text(cell(row, tc, 'title')),
      decision: text(cell(row, tc, 'decision')), accepted: flag(cell(row, tc, 'accepted')),
      withdrawn: flag(cell(row, tc, 'withdrawn')), inBatch: flag(cell(row, tc, 'in_batch')),
      sim: num(cell(row, tc, 'similarity')), sRate: num(cell(row, tc, 'student_rate')),
      kClean: num(cell(row, tc, 'teacher_clean_rate')), kRaw: num(cell(row, tc, 'teacher_rate')),
      valid: Array.isArray(valid) && valid.length === 2 ? [num(valid[0]), num(valid[1])] : null,
      audit: EX_AUDITS.includes(audit) ? audit : 'not_audited',
      yes: idxList(cell(row, tc, 'reproduced_yes')), partial: idxList(cell(row, tc, 'reproduced_partial')),
      f2p: num(cell(row, tc, 'f2p')), p2p: num(cell(row, tc, 'p2p')), scored, rescored: flag(cell(row, tc, 'rates_differ')),
      traj: [], rowsRates: null,
      link: run.link ? `${run.link}&node=${encodeURIComponent(node)}`
        : taskHash(run.target, node, run.source.audited ? null : run.run),
      dir: run.root && node ? `${run.root}/${node}` : '',
      rollouts: scored && run.root && node && round !== null ? `${run.root}/round-${round}/${node}/rollouts` : '',
    };
    byIndex[i] = t;
    tasks.push(t);
  });
  const trajectories = [];
  arr(doc.trajectories).forEach((row, i) => {
    if (!Array.isArray(row)) return;
    const task = byIndex[int(cell(row, rc, 'task'))];
    if (!task) return;
    const slot = text(cell(row, rc, 'slot')), clean = cell(row, rc, 'clean');
    const r = {i, task, slot, model: text(cell(row, rc, 'model')), status: text(cell(row, rc, 'status')),
               reward: num(cell(row, rc, 'reward')), f2p: text(cell(row, rc, 'f2p')), p2p: text(cell(row, rc, 'p2p')),
               tools: num(cell(row, rc, 'tool_calls')), patchChars: num(cell(row, rc, 'patch_chars')),
               agentErr: flag(cell(row, rc, 'agent_error')), verifierErr: flag(cell(row, rc, 'verifier_error')),
               clean: clean === true ? true : clean === false ? false : null,
               transcript: task.rollouts && slot ? `${task.rollouts}/${slot}.md` : '',
               patch: task.rollouts && slot ? `${task.rollouts}/${slot}.patch` : ''};
    task.traj.push(r);
    trajectories.push(r);
  });
  for (const t of tasks) {                     // the rollout rows' own rates (task_builder score(): pass / graded)
    const graded = (m) => t.traj.filter((r) => r.model === m && (r.status === 'pass' || r.status === 'fail'));
    const s = graded('sc14'), k = graded('kimi');
    t.rowsRates = {s: s.length ? s.filter((r) => r.status === 'pass').length / s.length : null,
                   k: k.length ? k.filter((r) => r.status === 'pass' && r.clean !== false).length / k.length : null,
                   valid: [s.length, k.length]};
  }
  const rounds = [...new Set(tasks.map((t) => t.round).filter((k) => k !== null))].sort((a, b) => a - b);
  const kinds = [...new Set([...EX_KINDS.filter((k) => tasks.some((t) => t.kind === k)),
                             ...tasks.map((t) => t.kind).filter(Boolean)])];
  const counts = new Map();
  for (const t of tasks) {
    const c = counts.get(t.target) || {tasks: 0, trajectories: 0, sources: new Set()};
    c.tasks += 1;
    c.trajectories += t.traj.length;
    c.sources.add(t.source);
    counts.set(t.target, c);
  }
  // The pass-rate columns of the distributions: every whole percent the data holds (5 graded rollouts: 0, 20, .., 100;
  // fewer: 25, 33, 67, 75), the same columns whatever the filters; past 16 values, closed 10-point bins instead.
  const rateColumns = (pick) => {
    const values = [...new Set(tasks.map(pick).map(exPct).filter((v) => v !== null))].sort((a, b) => a - b);
    if (values.length <= 16) return values.map((v) => ({lo: v, hi: v, label: String(v), title: `${v}%`}));
    return Array.from({length: 11}, (_, k) => (k < 10 ? {lo: k * 10, hi: k * 10 + 9, label: `${k * 10}–${k * 10 + 9}`,
                                                          title: `${k * 10}% to ${k * 10 + 9}%`}
                                                        : {lo: 100, hi: 100, label: '100', title: '100%'}));
  };
  const model = {generatedAt: num(doc.generated_at), sources, bySource, runs, targets, tasks, trajectories, rounds, kinds,
                 counts, targetIds: [...targets.keys()].sort(cmpStr), totals: obj(doc.totals), families, runNames,
                 rateColumns: {s: rateColumns((t) => t.sRate), clean: rateColumns((t) => t.kClean), raw: rateColumns((t) => t.kRaw)}};
  EX_MODEL_CACHE.set(doc, model);
  return model;
}

/** Every value a chip filter can take (null when the model is not loaded): sources, rounds, kinds of the data. */
function exAllValues(model, key) {
  const m = obj(model);
  if (key === 'src') return arr(m.sources).map((s) => s.source);
  if (key === 'fam') return arr(m.families).map((f) => f.key);
  if (key === 'round') return arr(m.rounds).slice();
  if (key === 'kind') return arr(m.kinds).slice();
  if (key === 'audit') return EX_AUDITS.slice();
  if (key === 'tm') return Object.keys(EX_MODELS);
  if (key === 'ts') return EX_STATUSES.slice();
  return [];
}

/** A range input as typed: {value (the filter's bound: similarity 0..10, a rate as a fraction 0..1, or null for an empty
    input), bad (why the input is not used, '' when it is)}. The pass-rate inputs take percent (0..100). An out-of-range or
    non-numeric input is not applied: its bound stays empty and the bar says why. */
function exParseBound(key, raw) {
  const t = text(raw).trim();
  if (!hasOwn(EX_RANGES, key) || t === '') return {value: null, bad: ''};
  // decimal notation (an exponent allowed) only: Number() alone would also take "0x10", "0b101" or "Infinity"
  if (!/^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$/.test(t)) {
    return {value: null, bad: `“${truncate(t, 12)}” is not a number`};
  }
  const n = Number(t);
  const pct = EX_PERCENT.has(key), [lo, hi] = pct ? [0, 100] : EX_RANGES[key];
  if (!Number.isFinite(n)) return {value: null, bad: `“${truncate(t, 12)}” is not a number`};
  if (n < lo || n > hi) return {value: null, bad: `${fmtCompact(n)} is outside ${lo}–${hi}${pct ? ' %' : ''}`};
  return {value: pct ? Math.round(n * 1000) / 100000 : n, bad: ''};
}

/** The filters of an Explorer hash ("#explorer&src=hengyuf&round=0,1&smin=6..." or its part after "explorer&"); unknown
    keys and malformed values are ignored (their default stays). Links of the first v1.7 page still read: its `sun=0`
    ("leave out the never-judged tasks") without a similarity bound becomes smin=0 (every judged task), `sun=1` is nj=1. */
function parseExplorerParams(query) {
  const f = explorerDefaults();
  let p;
  try { p = new URLSearchParams(text(query).replace(/^#?explorer&?/, '')); } catch (error) { p = new URLSearchParams(''); }
  // A list: "k=" chooses nothing (an empty list); values that are no valid choice are dropped, and a list left without
  // any (a mistyped link) is "every value" again.
  const list = (k, parse) => {
    if (!p.has(k)) return null;
    if (p.get(k) === '') return [];
    const out = [];
    for (const item of text(p.get(k)).split(',')) {
      const v = parse(item);
      if (v !== null && !out.includes(v)) out.push(v);
    }
    return out.length ? out : null;
  };
  const token = (v) => (EX_TOKEN.test(v) ? v : null);
  const roundOf = (v) => (/^\d{1,2}$/.test(v) ? Number(v) : null);
  const allowed = {audit: (v) => (EX_AUDITS.includes(v) ? v : null), tm: (v) => (hasOwn(EX_MODELS, v) ? v : null),
                   ts: (v) => (EX_STATUSES.includes(v) ? v : null), round: roundOf};
  for (const k of EX_LIST_KEYS) f[k] = list(k, allowed[k] || token);
  const t = list('t', token);
  f.t = t && t.length ? t : null;
  const run = list('run', token);                                      // v1.8: the runs chosen (absent: every run)
  f.run = run && run.length ? run : null;
  for (const [k, [lo, hi]] of Object.entries(EX_RANGES)) {                // the hash keeps fractions for the rates
    const raw = p.get(k);
    const v = raw === null || raw.trim() === '' ? null : num(Number(raw));
    f[k] = v !== null && v >= lo && v <= hi ? v : null;
  }
  if (['acc', 'not'].includes(p.get('dec'))) f.dec = p.get('dec');
  if (p.get('kb') === 'raw') f.kb = 'raw';
  if (['only', 'none'].includes(p.get('rd'))) f.rd = p.get('rd');
  for (const k of ['tc', 'tw', 'nj']) f[k] = p.get(k) === '1';
  if (p.get('sun') === '1') f.nj = true;                                  // the first v1.7 page's "include unscored"
  if (p.get('sun') === '0' && f.smin === null && f.smax === null) f.smin = 0;
  const sort = text(p.get('sort')).split(':');
  if (EX_SORTS.includes(sort[0])) { f.sort = sort[0]; f.desc = sort[1] === 'desc'; }
  for (const k of ['p', 'tp']) {
    const v = int(Number(p.get(k)));
    f[k] = v !== null && v >= 1 && v <= 100000 ? v : 1;
  }
  if (p.get('cov') === 'traj') f.cov = 'traj';
  return f;
}

/** The URL hash of an Explorer view: "#explorer" plus every filter that is not its default, in a fixed order. */
function explorerHash(filters) {
  const f = Object.assign(explorerDefaults(), obj(filters)), d = explorerDefaults();
  const parts = [];
  const add = (k, v) => parts.push(`${k}=${encodeURIComponent(text(v))}`);
  for (const k of EX_LIST_KEYS) if (Array.isArray(f[k])) parts.push(`${k}=${f[k].map((v) => encodeURIComponent(text(v))).join(',')}`);
  if (f.dec !== d.dec) add('dec', f.dec);
  for (const k of ['smin', 'smax']) if (num(f[k]) !== null) add(k, f[k]);
  if (f.nj === true) add('nj', 1);
  for (const k of ['nmin', 'nmax', 'kmin', 'kmax']) if (num(f[k]) !== null) add(k, f[k]);
  if (f.kb === 'raw') add('kb', 'raw');
  if (f.rd === 'only' || f.rd === 'none') add('rd', f.rd);
  if (Array.isArray(f.t) && f.t.length) parts.push(`t=${f.t.map((v) => encodeURIComponent(text(v))).join(',')}`);
  if (Array.isArray(f.run) && f.run.length) parts.push(`run=${f.run.map((v) => encodeURIComponent(text(v))).join(',')}`);
  for (const k of ['tc', 'tw']) if (f[k] === true) add(k, 1);
  if (EX_SORTS.includes(f.sort)) parts.push(`sort=${f.sort}:${f.desc ? 'desc' : 'asc'}`);   // code constants: no escaping
  for (const k of ['p', 'tp']) if (int(f[k]) !== null && f[k] > 1) add(k, f[k]);
  if (f.cov === 'traj') add('cov', 'traj');
  return `#explorer${parts.length ? `&${parts.join('&')}` : ''}`;
}

/** A chip filter after one checkbox: the values chosen (every value checked again: null). */
function exToggle(current, all, value, on) {
  const set = new Set(current === null || current === undefined ? all : current);
  if (on) set.add(value); else set.delete(value);
  const out = all.filter((v) => set.has(v));
  return out.length === all.length ? null : out;
}

/** The filters after an edit of one Explorer control (input.dataset.exf = its key; value / checked as the browser has
    them). Pure: returns a new object. */
function updateExplorerFilter(filters, key, control, model) {
  const f = Object.assign(explorerDefaults(), obj(filters));
  const c = obj(control), value = text(c.value), on = c.checked === true;
  if (EX_LIST_KEYS.includes(key)) {
    const v = key === 'round' ? int(Number(value)) : value;
    if (v !== null && v !== '') f[key] = exToggle(f[key], exAllValues(model, key), v, on);
  } else if (key === 't' || key === 'run') {                          // v1.8: the run checklist as the task one
    const set = new Set(f[key] || []);
    if (EX_TOKEN.test(value)) { if (on) set.add(value); else set.delete(value); }
    f[key] = set.size ? [...set].sort(cmpStr) : null;
  } else if (hasOwn(EX_RANGES, key)) {
    f[key] = exParseBound(key, value).value;
  } else if (key === 'nj' || key === 'tc' || key === 'tw') {
    f[key] = on;
  } else if (key === 'dec') {
    f.dec = ['acc', 'not'].includes(value) ? value : 'all';
  } else if (key === 'kb') {
    f.kb = value === 'raw' ? 'raw' : 'clean';
  } else if (key === 'rd') {
    f.rd = ['only', 'none'].includes(value) ? value : 'all';
  }
  return f;
}

/** The tasks and trajectories that pass the filters: {tasks, trajectories, matched (task -> its trajectories that pass
    the trajectory filters), trajActive, neverJudged, simBounded}. A range filter is off while both its bounds are empty;
    with a bound set, a task without that value fails it, also for the similarity unless `nj` ("include never judged") is
    on. A pass-rate bound applies to the whole percent the page shows for the rate (1/3 is 33 %: "max 33" keeps it, as
    its column and its cell read). `neverJudged`: the tasks the similarity judge never scored among those that pass every
    other filter (what `nj` adds). The trajectory filters (model, status, Kimi clean only) pick trajectories of the
    filtered tasks; "only tasks with >= 1 matching trajectory" also drops the tasks that have none. */
function filterExplorer(model, filters) {
  const f = Object.assign(explorerDefaults(), obj(filters));
  if (!isObj(model)) return {tasks: [], trajectories: [], matched: new Map(), trajActive: false, neverJudged: 0, simBounded: false};
  const has = (list, v) => list === null || list === undefined || list.includes(v);
  const bounded = (lo, hi) => num(lo) !== null || num(hi) !== null;
  const range = (v, lo, hi) => !bounded(lo, hi) ||
    (v !== null && (num(lo) === null || v >= lo - 1e-9) && (num(hi) === null || v <= hi + 1e-9));
  const kimi = (t) => (f.kb === 'raw' ? t.kRaw : t.kClean);
  const hundred = (b) => (num(b) === null ? null : b * 100);
  const rate = (v, lo, hi) => range(exPct(v), hundred(lo), hundred(hi));   // on the shown whole percent
  const targets = Array.isArray(f.t) && f.t.length ? new Set(f.t) : null;
  const runs = Array.isArray(f.run) && f.run.length ? new Set(f.run) : null;   // v1.8
  const simBounded = bounded(f.smin, f.smax);
  const others = (t) => has(f.src, t.source) && has(f.fam, t.run.family) && (runs === null || runs.has(t.run.run)) &&
    has(f.round, t.round) && has(f.kind, t.kind) &&
    (f.dec === 'all' || (f.dec === 'acc') === t.accepted) && rate(t.sRate, f.nmin, f.nmax) &&
    rate(kimi(t), f.kmin, f.kmax) && has(f.audit, t.audit) && (targets === null || targets.has(t.target)) &&
    (f.rd === 'all' || (f.rd === 'only') === t.rescored);
  const simOk = (t) => !simBounded || (t.sim === null ? f.nj === true : range(t.sim, f.smin, f.smax));
  const trajOk = (r) => has(f.tm, r.model) && has(f.ts, r.status) &&
    (f.tc !== true || r.model !== 'kimi' || r.status !== 'pass' || r.clean === true);
  const matched = new Map();
  const count = (t) => {
    if (!matched.has(t)) matched.set(t, t.traj.reduce((n, r) => n + (trajOk(r) ? 1 : 0), 0));
    return matched.get(t);
  };
  const base = model.tasks.filter(others);
  let tasks = base.filter(simOk);
  if (f.tw === true) tasks = tasks.filter((t) => count(t) > 0);
  const trajectories = [];
  for (const t of tasks) {
    count(t);
    for (const r of t.traj) if (trajOk(r)) trajectories.push(r);
  }
  const neverJudged = base.filter((t) => t.sim === null && (f.tw !== true || count(t) > 0)).length;
  return {tasks, trajectories, matched, trajActive: f.tm !== null || f.ts !== null || f.tc === true, neverJudged, simBounded};
}

/** What the filter bar says about the filters and their inputs: the never-judged count next to "include never judged",
    the ranges whose minimum is above their maximum, the rejected inputs (`bad`: {key: why}, from the wiring), and the
    DeepSWE task choice. Each entry is the HTML of one status slot of the bar (#ex-st-<name>). */
function explorerFilterStatus(model, filters, result, bad) {
  const m = obj(model), f = Object.assign(explorerDefaults(), obj(filters)), r = obj(result), b = obj(bad);
  const n = num(r.neverJudged) || 0;
  const nj = !r.simBounded
    ? `<span class="muted small" title="${esc('no similarity bound is set: every task passes the similarity filter, judged or not')}">` +
      `${esc(exCount(n, 'never judged', 'never judged'))}, shown</span>`
    : f.nj === true ? `<span class="small">${esc(`incl. ${fmtInt(n)} never judged`)}</span>`
    : `<span class="small" title="${esc('tasks the similarity judge never scored, left out while a bound is set; tick the box to keep them')}">` +
      `${esc(`+${fmtInt(n)} never judged`)}</span>`;
  const out = {nj};
  for (const [g, [lo, hi, label]] of Object.entries(EX_RANGE_GROUPS)) {
    const notes = [];
    if (b[lo]) notes.push(`min: ${b[lo]}`);
    if (b[hi]) notes.push(`max: ${b[hi]}`);
    const a = num(f[lo]), z = num(f[hi]);
    if (a !== null && z !== null && a > z) {
      const show = (v) => (EX_PERCENT.has(lo) ? `${fmtCompact(v * 100)} %` : fmtCompact(v));
      notes.unshift(`min ${show(a)} > max ${show(z)}: no task can pass`);
    }
    out[g] = notes.length ? `<span class="ex-warn" title="${esc(`${label}: ${notes.join('; ')}`)}">${esc(notes.join(' · '))}</span>` : '';
  }
  const picked = arr(f.t), total = arr(m.targetIds).length;
  out.targets = esc(picked.length ? `${fmtInt(picked.length)} of ${fmtInt(total)}` : `all ${fmtInt(total)}`);
  const runs = arr(f.run), allRuns = arr(m.runNames).length;            // v1.8: the run checklist's label
  out.runs = esc(runs.length ? `${fmtInt(runs.length)} of ${fmtInt(allRuns)}` : `all ${fmtInt(allRuns)}`);
  return out;
}

/** Bin of a pass rate among a model's rate columns (exRateColumns: whole percents, or 10-point bins), or null. */
function exRateColumn(columns, v) {
  const p = exPct(v);
  if (p === null) return null;
  const k = arr(columns).findIndex((c) => p >= c.lo && p <= c.hi);
  return k < 0 ? null : k;
}

/** Everything the results show, from the filtered rows: the KPIs, coverage per DeepSWE task (synthetic tasks and
    trajectories, by source; the scope: the DeepSWE tasks the chosen sources converted, within the DeepSWE task filter),
    failure-pattern coverage per target over the audited tasks (reproduced yes / partial; with how many filtered tasks
    lack a counted audit, and why), and the distributions (similarity, Nemotron rate, Kimi rate in the chosen basis,
    audit verdicts, rounds). */
function explorerStats(model, result, filters) {
  const m = obj(model), r = obj(result), f = Object.assign(explorerDefaults(), obj(filters));
  const tasks = arr(r.tasks), trajs = arr(r.trajectories);
  const model14 = trajs.filter((x) => x.model === 'sc14'), kimi = trajs.filter((x) => x.model === 'kimi');
  const pass = (x) => x.status === 'pass';
  // Coverage: the DeepSWE tasks in scope (a task of a chosen source, within the DeepSWE task filter), zeros included.
  const sources = arr(m.sources), chosen = (s) => f.src === null || arr(f.src).includes(s);
  const picked = Array.isArray(f.t) && f.t.length ? new Set(f.t) : null;
  const scope = arr(m.targetIds).filter((id) => {
    const c = m.counts instanceof Map ? m.counts.get(id) : null;
    return c && [...c.sources].some(chosen) && (picked === null || picked.has(id));
  });
  const kpi = {tasks: tasks.length, allTasks: arr(m.tasks).length, trajectories: trajs.length,
               allTrajectories: arr(m.trajectories).length, targets: new Set(tasks.map((t) => t.target)).size,
               inScope: scope.length, converted: m.counts instanceof Map ? m.counts.size : 0,
               accepted: tasks.filter((t) => t.accepted).length, rescored: tasks.filter((t) => t.rescored).length,
               scPass: model14.filter(pass).length, scTotal: model14.length, kimiPass: kimi.filter(pass).length,
               kimiClean: kimi.filter((x) => pass(x) && x.clean === true).length, kimiTotal: kimi.length};
  const coverage = new Map(scope.map((id) => [id, {id, tasks: sources.map(() => 0), traj: sources.map(() => 0)}]));
  const slot = (source) => sources.findIndex((s) => s.source === source);
  for (const t of tasks) {
    const c = coverage.get(t.target), k = slot(t.source);
    if (c && k >= 0) c.tasks[k] += 1;
  }
  for (const x of trajs) {
    const c = coverage.get(x.task.target), k = slot(x.task.source);
    if (c && k >= 0) c.traj[k] += 1;
  }
  // Failure-pattern coverage: the filtered tasks with a counted audit (a verdict), per target.
  const audited = tasks.filter((t) => FA_ORDER.includes(t.audit));
  const patterns = new Map();
  for (const t of audited) {
    const target = m.targets instanceof Map ? m.targets.get(t.target) : null;
    const g = patterns.get(t.target) || {id: t.target, audited: 0, patterns: arr(target && target.patterns).map((p, i) =>
      ({i, name: p.name, category: p.category, yes: 0, partial: 0}))};
    g.audited += 1;
    for (const [list, key] of [[t.yes, 'yes'], [t.partial, 'partial']]) {
      for (const k of arr(list)) if (g.patterns[k]) g.patterns[k][key] += 1;
    }
    patterns.set(t.target, g);
  }
  const official = new Set(sources.filter((s) => s.audited).map((s) => s.source));
  const excluded = {notAudited: 0, notAuditedBatch: 0, notEligible: 0};
  for (const t of tasks) {
    if (t.audit === 'not_eligible') excluded.notEligible += 1;
    else if (t.audit === 'not_audited' && official.has(t.source)) excluded.notAudited += 1;
    else if (t.audit === 'not_audited') excluded.notAuditedBatch += 1;
  }
  // Distributions: similarity 0..10, the pass rates over the model's columns (whole percents), verdicts, rounds.
  const cols = obj(m.rateColumns), kCols = arr(f.kb === 'raw' ? cols.raw : cols.clean), sCols = arr(cols.s);
  const sim = Array(11).fill(0), sRate = sCols.map(() => 0), kRate = kCols.map(() => 0);
  let simNone = 0, sNone = 0, kNone = 0;
  for (const t of tasks) {
    if (t.sim === null) simNone += 1; else sim[Math.max(0, Math.min(10, Math.round(t.sim)))] += 1;
    const s = exRateColumn(sCols, t.sRate), k = exRateColumn(kCols, f.kb === 'raw' ? t.kRaw : t.kClean);
    if (s === null) sNone += 1; else sRate[s] += 1;
    if (k === null) kNone += 1; else kRate[k] += 1;
  }
  const audits = EX_AUDITS.map((a) => tasks.filter((t) => t.audit === a).length);
  const rounds = arr(m.rounds).map((k) => ({round: k, n: tasks.filter((t) => t.round === k).length}));
  return {kpi, coverage: [...coverage.values()], patterns: [...patterns.values()], audited: audited.length, excluded,
          dist: {sim, simNone, sRate, sNone, sCols, kRate, kNone, kCols, audits, rounds}};
}

/** Sort key of a task row for a table column (null sorts last). */
function exSortValue(t, key, f, matched) {
  if (key === 'task') return t.i;
  if (key === 'target') return t.target;
  if (key === 'source') return t.source;
  if (key === 'round') return t.round;
  if (key === 'decision') return decisionOf({decision: t.decision}).label;
  if (key === 'sim') return t.sim;
  if (key === 'srate') return t.sRate;
  if (key === 'krate') return obj(f).kb === 'raw' ? t.kRaw : t.kClean;
  if (key === 'rescored') return t.rescored ? 1 : 0;
  if (key === 'audit') return EX_AUDITS.indexOf(t.audit);
  if (key === 'f2p') return t.f2p;
  if (key === 'traj') return matched instanceof Map ? matched.get(t) || 0 : t.traj.length;
  return t.i;
}

/** The filtered tasks in the table's order (the collector's row order unless a column was chosen; ties keep it). */
function exSortedTasks(result, filters) {
  const f = Object.assign(explorerDefaults(), obj(filters)), r = obj(result);
  const list = arr(r.tasks).slice();
  if (!f.sort || f.sort === 'task') return f.desc ? list.reverse() : list;
  const dir = f.desc ? -1 : 1;
  return list.sort((a, b) => {
    const x = exSortValue(a, f.sort, f, r.matched), y = exSortValue(b, f.sort, f, r.matched);
    if (x === null || y === null) return x === y ? a.i - b.i : x === null ? 1 : -1;
    const c = typeof x === 'number' && typeof y === 'number' ? x - y : cmpStr(x, y);
    return c ? c * dir : a.i - b.i;
  });
}

/** A source's name with its series swatch (identity next to the text, never text in the series color). */
function exSource(model, source) {
  const s = obj(model).bySource instanceof Map ? model.bySource.get(source) : null;
  return `<span class="ex-src"><span class="swatch ${s ? s.cls : 's-other'}" aria-hidden="true"></span>${esc(s ? s.name : source)}</span>`;
}

/** The audit cell of a task row: the verdict badge (FA match colors) or "not audited" / "not eligible". */
function exAuditBadge(audit, extra) {
  if (hasOwn(FA_VERDICTS, audit)) return badge(FA_VERDICTS[audit].label, FA_VERDICTS[audit].tone, FA_VERDICTS[audit].help, extra);
  return badge(lookup(EX_AUDIT_TEXT, audit) || text(audit), audit === 'not_eligible' ? 'neutral' : 'muted',
               lookup(EX_AUDIT_HELP, audit), extra);
}

/** "P1 · <name>" of a target's pattern index. */
function exPatternLabel(model, target, k) {
  const t = obj(model).targets instanceof Map ? model.targets.get(target) : null;
  const p = t ? t.patterns[k] : null;
  return p ? `P${k + 1}${p.name ? ` · ${p.name}` : ''}` : `P${k + 1}`;
}

/** A re-scored task's rates, as metrics.json has them and as its rollout rows give them (the marker's tooltip). */
function exRescoredTitle(t) {
  const x = obj(t), r = obj(x.rowsRates), pct = (v) => (num(v) === null ? DASH : `${exPct(v)}%`);
  const valid = arr(x.valid), rows = arr(r.valid);
  return `${EX_RESCORED_HELP}\nmetrics.json: Nemotron ${pct(x.sRate)} · Kimi clean ${pct(x.kClean)}` +
         `${valid.length ? ` · valid ${fmtInt(valid[0])} / ${fmtInt(valid[1])}` : ''}` +
         `\nrollout rows: Nemotron ${pct(r.s)} · Kimi clean ${pct(r.k)} · graded ${fmtInt(rows[0])} / ${fmtInt(rows[1])}`;
}

/** A tooltip's content (data-tip: {t: title, r: [[key class, value, label], ...]}): values lead, labels follow; each row
    keyed by a short line in its series color. '' for anything else. */
function renderExplorerTip(raw) {
  let d = raw;
  if (typeof raw === 'string') { try { d = JSON.parse(raw); } catch (error) { d = null; } }
  if (!isObj(d)) return '';
  const rows = arr(d.r).filter(Array.isArray).map((x) =>
    `<div class="ex-tip-row"><span class="ex-tip-key${EX_TIP_KEYS.has(x[0]) ? ` ${x[0]}` : ''}" aria-hidden="true"></span>` +
    `<b>${esc(x[1])}</b> <span>${esc(x[2])}</span></div>`).join('');
  return `${text(d.t) ? `<div class="ex-tip-title">${esc(d.t)}</div>` : ''}${rows}`;
}

/** data-tip value of a mark: JSON {t, r} (escaped for the attribute by the caller). */
function exTip(title, rows) { return JSON.stringify({t: text(title), r: rows.map((x) => [text(x[0]), text(x[1]), text(x[2])])}); }

/** A chart mark's focus attributes: one roving tab stop per panel (data-roving; arrow keys move between its marks:
    the wiring), so the charts cost a few Tab presses, not one per mark. `first`: this mark takes the panel's stop. */
function exMarkAttrs(key, i, first) {
  return ` data-mark="${esc(key)}:${i}" data-fkey="mark:${esc(key)}:${i}" tabindex="${first && i === 0 ? '0' : '-1'}"`;
}

/** Horizontal bars: one row per category (the row is the hover / focus hit target), up to two stacked series with a
    2px surface gap, 4px rounded data end and square baseline, <= 24px thick; a hairline grid on clean ticks; the value
    at the bar's tip; a legend for two or more series (none for one: the title names it); a table twin.
    spec: {key, title, sub, category, unit: [one, many], series: [{label, cls}], rows: [{label, title, values: [n...]}],
    ui, empty, labelWidth, max (a scale shared by small multiples), legend (false: the panel shows the one legend),
    first (this chart's first mark is its panel's tab stop)}. */
function renderHBars(spec) {
  const s = obj(spec), series = arr(s.series).filter(isObj), rows = arr(s.rows).filter(isObj);
  const total = (row) => arr(row.values).reduce((a, v) => a + (num(v) || 0), 0);
  const {top, ticks} = niceTicks(Math.max(0, num(s.max) || 0, ...rows.map(total)), 4, true);   // max: a scale shared
  const pct = (v) => (top > 0 ? Math.max(0, Math.min(100, (v / top) * 100)) : 0);
  const unit = Array.isArray(s.unit) ? s.unit : ['', ''];
  const legend = series.length > 1 && s.legend !== false
    ? `<div class="chart-legend">${series.map((x) => `<span class="lg"><span class="swatch ${esc(x.cls)}" aria-hidden="true"></span>${esc(x.label)}</span>`).join('')}</div>` : '';
  const bars = rows.map((row, i) => {
    const values = series.map((_, k) => num(arr(row.values)[k]) || 0), sum = values.reduce((a, b) => a + b, 0);
    const segs = values.map((v, k) => ({v, cls: series[k].cls})).filter((x) => x.v > 0);
    const html = segs.map((x, k) => `<span class="hbar-seg ${esc(x.cls)}${k === segs.length - 1 ? ' end' : ''}" style="width:` +
      (k ? `calc(${pct(x.v).toFixed(3)}% - 2px);margin-left:2px` : `${pct(x.v).toFixed(3)}%`) + '"></span>').join('');
    const tipRows = series.map((x, k) => [x.cls, fmtInt(values[k]), x.label]);
    if (series.length > 1) tipRows.push(['', fmtInt(sum), 'total']);
    const what = (n) => (unit[0] ? ` ${n === 1 ? unit[0] : unit[1]}` : '');
    const aria = `${text(row.title || row.label)}: ${series.length > 1
      ? series.map((x, k) => `${fmtInt(values[k])} ${text(x.label)}`).join(', ') + `; ${fmtInt(sum)}${what(sum)} in all`
      : `${fmtInt(sum)}${what(sum)}`}`;
    return `<div class="hbar" role="listitem"${exMarkAttrs(s.key, i, s.first !== false)} data-tip="${esc(exTip(row.title || row.label, tipRows))}" aria-label="${esc(aria)}">` +
           `<span class="hbar-label" title="${esc(row.title || row.label)}">${esc(row.label)}</span>` +
           `<span class="hbar-plot" style="--n:${ticks.length - 1}">${html}` +
           `<span class="hbar-val">${esc(fmtInt(sum))}</span></span></div>`;
  }).join('');
  const axis = `<div class="hbar-axis" aria-hidden="true"><span></span><span class="hbar-ticks">` +
    ticks.map((v) => `<span style="left:${pct(v).toFixed(3)}%">${esc(fmtInt(v))}</span>`).join('') + '</span></div>';
  const table = rows.length
    ? `<table class="tbl tbl-compact"><thead><tr><th>${esc(s.category || 'category')}</th>${series.map((x) => `<th class="num">${esc(x.label)}</th>`).join('')}` +
      `${series.length > 1 ? '<th class="num">total</th>' : ''}</tr></thead><tbody>` +
      rows.map((row) => `<tr><td>${esc(row.title || row.label)}</td>${series.map((_, k) => `<td class="num">${fmtInt(num(arr(row.values)[k]) || 0)}</td>`).join('')}` +
                        `${series.length > 1 ? `<td class="num">${fmtInt(total(row))}</td>` : ''}</tr>`).join('') + '</tbody></table>'
    : note('No data.');
  const key = `ex:chart:${text(s.key) || text(s.title)}`;
  return `<figure class="chart ex-chart"><figcaption><span class="chart-title">${esc(s.title)}</span>` +
         `${text(s.sub) ? `<span class="chart-now">${esc(s.sub)}</span>` : ''}</figcaption>${legend}` +
         (rows.length ? `<div class="hbars" role="list" aria-label="${esc(s.title)}"${s.labelWidth ? ` style="--label-w:${esc(s.labelWidth)}"` : ''}>${bars}${axis}</div>`
                      : `<p class="note small">${esc(s.empty || 'No data.')}</p>`) +
         det(s.ui, key, 'Table', `<div class="chart-table-scroll"${scrollAttr(`${key}:table`)}>${table}</div>`, false, 'fold fold-inline chart-table') +
         '</figure>';
}

/** Columns (a histogram): one series in slot 3 (a hue no source wears), <= 24px wide with a 4px rounded cap and square
    baseline, the value on the cap, a hairline grid on clean ticks; each column's full-height slot is its hover / focus hit
    target; x labels wrap (the full label is in the tooltip and the table); a table twin.
    spec: {key, title, sub, bins: [{label, n, title}], ui, unit: [one, many], note, first}. */
function renderVBars(spec) {
  const s = obj(spec), bins = arr(s.bins).filter(isObj);
  const {top, ticks} = niceTicks(Math.max(0, ...bins.map((b) => num(b.n) || 0)), 4, true);
  const pct = (v) => (top > 0 ? Math.max(0, Math.min(100, (v / top) * 100)) : 0);
  const unit = Array.isArray(s.unit) ? s.unit : ['task', 'tasks'];
  const cols = bins.map((b, i) => {
    const n = num(b.n) || 0, h = pct(n).toFixed(3), what = n === 1 ? unit[0] : unit[1];
    return `<div class="vbar" role="listitem"${exMarkAttrs(s.key, i, s.first !== false)} data-tip="${esc(exTip(b.title || b.label, [[EX_HIST, fmtInt(n), what]]))}"` +
           ` aria-label="${esc(`${text(b.title || b.label)}: ${fmtInt(n)} ${what}`)}">` +
           (n ? `<span class="vbar-col ${EX_HIST}" style="height:${h}%"></span><span class="vbar-val" style="bottom:${h}%">${esc(fmtInt(n))}</span>` : '') +
           '</div>';                                         // an empty column: no mark, no label (tooltip and table keep 0)
  }).join('');
  const labels = bins.map((b) => `<span title="${esc(b.title || b.label)}">${esc(b.label)}</span>`).join('');
  const table = `<table class="tbl tbl-compact"><thead><tr><th>${esc(s.category || 'bin')}</th><th class="num">${esc(unit[1])}</th></tr></thead><tbody>` +
    bins.map((b) => `<tr><td>${esc(b.title || b.label)}</td><td class="num">${fmtInt(num(b.n) || 0)}</td></tr>`).join('') + '</tbody></table>';
  const key = `ex:chart:${text(s.key) || text(s.title)}`;
  return `<figure class="chart ex-chart ex-hist"><figcaption><span class="chart-title">${esc(s.title)}</span>` +
         `${text(s.sub) ? `<span class="chart-now">${esc(s.sub)}</span>` : ''}</figcaption>` +
         `<div class="vbars" role="list" aria-label="${esc(s.title)}" style="--n:${ticks.length - 1}">${cols}</div>` +
         `<div class="vbar-x" aria-hidden="true">${labels}</div>` +
         (text(s.note) ? `<div class="muted small">${esc(s.note)}</div>` : '') +
         det(s.ui, key, 'Table', `<div class="chart-table-scroll"${scrollAttr(`${key}:table`)}>${table}</div>`, false, 'fold fold-inline chart-table') +
         '</figure>';
}

/** The filter bar (one row above everything it scopes): source, round, kind, decision, similarity (with "include never
    judged" and how many that is), Nemotron and Kimi pass rates in percent (Kimi: clean or raw), re-scored tasks, audit
    verdict, DeepSWE task (a searchable checklist), the trajectory filters (model, status, Kimi clean only), "only tasks
    with >= 1 matching trajectory" and Reset. Every control carries data-exf (its filter) and data-fkey (focus kept across
    re-renders); the status slots (#ex-st-*) say what an input does not or cannot do (the wiring refreshes them after an
    edit). opts: {ui, query (the checklist's search), result (filterExplorer of these filters), bad (rejected inputs:
    {key: raw text}, shown as typed)}. */
function renderExplorerFilters(model, filters, opts) {
  const m = obj(model), f = Object.assign(explorerDefaults(), obj(filters)), o = obj(opts), bad = obj(o.bad);
  const status = explorerFilterStatus(m, f, o.result || filterExplorer(model, f),
                                      Object.fromEntries(Object.entries(bad).map(([k, raw]) => [k, exParseBound(k, raw).bad])));
  const on = (k, v) => f[k] === null || arr(f[k]).includes(v);
  const chip = (k, v, label, title) => `<label class="ex-chip"${title ? ` title="${esc(title)}"` : ''}>` +
    `<input type="checkbox" data-exf="${k}" value="${esc(v)}" data-fkey="exf:${k}:${esc(v)}"${on(k, v) ? ' checked' : ''}>` +
    `<span>${esc(label)}</span></label>`;
  const group = (legend, body, cls) => `<fieldset class="ex-f${cls ? ` ${cls}` : ''}"><legend class="ex-f-k">${esc(legend)}</legend>` +
    `<div class="ex-f-v">${body}</div></fieldset>`;
  const slot = (name) => `<span id="ex-st-${name}" class="ex-st" role="status">${status[name] || ''}</span>`;
  const numIn = (k, label) => {
    const pct = EX_PERCENT.has(k);
    const shown = hasOwn(bad, k) ? text(bad[k]) : num(f[k]) === null ? '' : pct ? String(Math.round(f[k] * 10000) / 100) : String(f[k]);
    // v1.8: a text input (inputmode decimal: a numeric keypad on touch devices), so that whatever is typed reaches the
    // bar's own check (exParseBound), which names the problem: a number input would hand the page "" for "abc"
    // (validity.badInput) and the bound would silently clear
    return `<input type="text" inputmode="decimal" class="ex-num" data-exf="${k}" data-fkey="exf:${k}"` +
           ` value="${esc(shown)}" placeholder="${k.endsWith('min') ? 'min' : 'max'}" aria-label="${esc(label)}"` +
           ` autocomplete="off" spellcheck="false"${hasOwn(bad, k) ? ' aria-invalid="true"' : ''}>`;
  };
  const range = (g, legend, unitText) => {
    const [lo, hi, label] = EX_RANGE_GROUPS[g];
    return `${numIn(lo, `${label} minimum${unitText ? `, ${unitText}` : ''}`)}<span class="ex-dash">–</span>` +
           `${numIn(hi, `${label} maximum${unitText ? `, ${unitText}` : ''}`)}${unitText ? `<span class="ex-unit">${esc(unitText)}</span>` : ''}`;
  };
  const select = (k, options, label) => `<select data-exf="${k}" data-fkey="exf:${k}" aria-label="${esc(label)}">` +
    options.map(([v, l]) => `<option value="${esc(v)}"${f[k] === v ? ' selected' : ''}>${esc(l)}</option>`).join('') + '</select>';
  const check = (k, label, checked, title) => `<label class="ex-check"${title ? ` title="${esc(title)}"` : ''}><input type="checkbox" data-exf="${k}" data-fkey="exf:${k}"` +
    `${checked ? ' checked' : ''}><span>${esc(label)}</span></label>`;
  const parts = [
    group('Source', arr(m.sources).map((s) => chip('src', s.source, s.name, s.label)).join('')),
    group('Round', arr(m.rounds).map((k) => chip('round', k, String(k), k === 0 ? 'round 0: mask tasks' : `round ${k}: evolved tasks`)).join('')),
    group('Kind', arr(m.kinds).map((k) => chip('kind', k, k)).join('')),
    group('Decision', select('dec', EX_DECISIONS, 'decision')),
    group('Similarity', range('s', 'Similarity', '') +
                        check('nj', 'include never judged', f.nj === true,
                              'with a similarity bound set, also keep the tasks the similarity judge never scored') +
                        slot('nj') + slot('s'), 'ex-f-sim'),
    group('Nemotron pass rate', range('n', 'Nemotron pass rate', '%') + slot('n')),
    group('Kimi pass rate', range('k', 'Kimi pass rate', '%') + select('kb', [['clean', 'clean'], ['raw', 'raw']], 'Kimi pass rate basis') +
                            slot('k')),
    group('Re-scored', select('rd', EX_RESCORED, 're-scored tasks (metrics.json rates differ from the rollout rows)')),
    group('Audit', EX_AUDITS.map((a) => chip('audit', a, EX_AUDIT_TEXT[a], hasOwn(FA_VERDICTS, a) ? FA_VERDICTS[a].help : EX_AUDIT_HELP[a])).join('')),
    group('DeepSWE task', det(o.ui, 'ex:targets', `<span id="ex-st-targets" class="ex-dd-label">${status.targets}</span>`,
      `<input type="search" class="ex-search" data-exf="tq" data-fkey="exf:tq" value="${esc(o.query)}" placeholder="search DeepSWE tasks" aria-label="search DeepSWE tasks" autocomplete="off" spellcheck="false">` +
      `<div id="ex-tlist" class="ex-tlist"${scrollAttr('ex:tlist')}>${renderExplorerTargetList(model, f, o.query)}</div>` +
      '<div class="ex-dd-tools"><button type="button" class="btn btn-xs" data-exaction="targets-shown" data-fkey="exa:targets-shown">Select shown</button>' +
      '<button type="button" class="btn btn-xs" data-exaction="targets-clear" data-fkey="exa:targets-clear">All tasks</button></div>', false, 'ex-dd'), 'ex-f-targets'),
    group('Run', `<div class="ex-runs-fam">${arr(m.families).map((x) => chip('fam', x.key, x.label,
      x.source === 'hengyuf' ? `official runs named ${x.key}-*` : `${x.label}'s batch runs`)).join('')}</div>` +
      det(o.ui, 'ex:runs', `<span id="ex-st-runs" class="ex-dd-label">${status.runs}</span>`,
        `<input type="search" class="ex-search" data-exf="rq" data-fkey="exf:rq" value="${esc(o.runQuery)}" placeholder="search runs" aria-label="search runs" autocomplete="off" spellcheck="false">` +
        `<div id="ex-rlist" class="ex-tlist"${scrollAttr('ex:rlist')}>${renderExplorerRunList(model, f, o.runQuery)}</div>` +
        '<div class="ex-dd-tools"><button type="button" class="btn btn-xs" data-exaction="runs-shown" data-fkey="exa:runs-shown">Select shown</button>' +
        '<button type="button" class="btn btn-xs" data-exaction="runs-clear" data-fkey="exa:runs-clear">All runs</button></div>', false, 'ex-dd'), 'ex-f-runs'),
    group('Trajectories', Object.keys(EX_MODELS).map((k) => chip('tm', k, k === 'sc14' ? 'Nemotron' : 'Kimi', EX_MODELS[k])).join('') +
                          '<span class="ex-sep" aria-hidden="true"></span>' + EX_STATUSES.map((v) => chip('ts', v, v)).join('') +
                          check('tc', 'Kimi clean only', f.tc === true, 'a Kimi pass counts only when the teacher audit (teacher_audit.json) found no cheating'), 'ex-f-traj'),
    `<div class="ex-f ex-f-end">${check('tw', 'only tasks with ≥ 1 matching trajectory', f.tw === true,
      'drop the tasks none of whose trajectories pass the trajectory filters')}` +
      '<button type="button" class="btn" data-exaction="reset" data-fkey="exa:reset">Reset</button></div>',
  ];
  return `<section class="panel ex-filterbar" aria-label="Explorer filters"><div class="ex-filters">${parts.join('')}</div></section>`;
}

/** The DeepSWE task checklist of the filter bar: every target (matching the search), checked when chosen, with its
    synthetic task count. */
function renderExplorerTargetList(model, filters, query) {
  const m = obj(model), f = obj(filters), q = text(query).trim().toLowerCase();
  const picked = new Set(arr(f.t));
  const ids = arr(m.targetIds).filter((id) => !q || id.toLowerCase().includes(q));
  if (!ids.length) return `<p class="muted small">No DeepSWE task matches “${esc(query)}”.</p>`;
  return ids.map((id) => {
    const c = m.counts instanceof Map ? m.counts.get(id) : null;
    return `<label class="ex-titem"><input type="checkbox" data-exf="t" value="${esc(id)}" data-fkey="exf:t:${esc(id)}"${picked.has(id) ? ' checked' : ''}>` +
           `<span class="ex-titem-id">${esc(id)}</span><span class="muted small">${esc(exCount(c ? c.tasks : 0, 'task', 'tasks'))}</span></label>`;
  }).join('');
}

/** v1.8, the run checklist of the filter bar: every run of the file (matching the search) with its DeepSWE task, its
    family and, for an older official run, "older"; checked when chosen, with its synthetic task count. */
function renderExplorerRunList(model, filters, query) {
  const m = obj(model), f = obj(filters), q = text(query).trim().toLowerCase();
  const picked = new Set(arr(f.run));
  const runs = arr(m.runs).filter((r) => r.run && (!q || [r.run, r.target, r.familyLabel].some((v) => text(v).toLowerCase().includes(q))));
  if (!runs.length) return `<p class="muted small">No run matches “${esc(query)}”.</p>`;
  const tasks = new Map();
  for (const t of arr(m.tasks)) tasks.set(t.run, (tasks.get(t.run) || 0) + 1);
  return runs.map((r) => `<label class="ex-titem" title="${esc(`${r.run}: ${r.target}${r.latest === false ? ' (an older official run of it)' : ''}`)}">` +
    `<input type="checkbox" data-exf="run" value="${esc(r.run)}" data-fkey="exf:run:${esc(r.run)}"${picked.has(r.run) ? ' checked' : ''}>` +
    `<span class="ex-titem-id">${esc(r.run)}</span><span class="muted small">${esc(`${r.latest === false ? 'older · ' : ''}` +
    exCount(tasks.get(r) || 0, 'task', 'tasks'))}</span></label>`).join('');
}

/** Page of a list: {n, page (1-based, clamped), pages, from, to, rows}. */
function exPage(list, page) {
  const n = arr(list).length, pages = Math.max(1, Math.ceil(n / EX_PAGE));
  const p = Math.max(1, Math.min(pages, int(page) || 1));
  return {n, page: p, pages, from: n ? (p - 1) * EX_PAGE + 1 : 0, to: Math.min(n, p * EX_PAGE),
          rows: arr(list).slice((p - 1) * EX_PAGE, p * EX_PAGE)};
}

/** The pager under a table: "page 2 of 7 · tasks 51–100", or the row count on one page ("1 task"). */
function exPager(pg, key, one, many) {
  if (pg.pages <= 1) return `<div class="ex-pager muted small">${esc(exCount(pg.n, one, many))}</div>`;
  const btn = (p, text_, dis) => `<button type="button" class="btn btn-xs" data-expage="${key}:${p}" data-fkey="exp:${key}:${text_}"${dis ? ' disabled' : ''}>${esc(text_)}</button>`;
  return `<div class="ex-pager">${btn(1, 'First', pg.page === 1)}${btn(pg.page - 1, 'Previous', pg.page === 1)}` +
         `<span class="muted small">${esc(`page ${fmtInt(pg.page)} of ${fmtInt(pg.pages)} · ${many} ${fmtInt(pg.from)}–${fmtInt(pg.to)} of ${fmtInt(pg.n)}`)}</span>` +
         `${btn(pg.page + 1, 'Next', pg.page === pg.pages)}${btn(pg.pages, 'Last', pg.page === pg.pages)}</div>`;
}

/** The synthetic-task table (#ex-tasks): sortable columns, 50 rows a page; a row opens the task view with the node's
    drawer (its deep link; the first cell's button for the keyboard). A re-scored task (its metrics.json rates differ from
    its rollout rows) carries a marker on its rates and in its own column. */
function renderExplorerTaskTable(model, result, filters) {
  const m = obj(model), r = obj(result), f = Object.assign(explorerDefaults(), obj(filters));
  const sorted = exSortedTasks(r, f), pg = exPage(sorted, f.p);
  const head = (key, label, cls, title) => {
    const on = (f.sort || 'task') === key;
    const arrow = on ? (f.desc ? ' ↓' : ' ↑') : '';
    return `<th${cls ? ` class="${cls}"` : ''} scope="col" aria-sort="${on ? (f.desc ? 'descending' : 'ascending') : 'none'}">` +
           `<button type="button" class="ex-sort" data-exsort="${key}" data-fkey="exs:${key}"${title ? ` title="${esc(title)}"` : ''}>${esc(label)}${arrow}</button></th>`;
  };
  const rows = pg.rows.map((t) => {
    const dec = decisionOf({decision: t.decision});
    const n = r.matched instanceof Map ? r.matched.get(t) || 0 : t.traj.length;
    const reproduced = [...arr(t.yes).map((k) => [k, 'yes']), ...arr(t.partial).map((k) => [k, 'partial'])]
      .map(([k, how]) => `<span class="tag tag-fa ex-rep ex-rep-${how}" title="${esc(`${exPatternLabel(m, t.target, k)}: reproduced ${how}`)}">P${k + 1}${how === 'partial' ? '~' : ''}</span>`).join(' ');
    const kimi = f.kb === 'raw' ? t.kRaw : t.kClean, other = f.kb === 'raw' ? t.kClean : t.kRaw;
    // a re-scored task: * on each metrics.json rate its rollout rows disagree with (valid counts alone: the badge)
    const rows = obj(t.rowsRates), same = (a, b) => (num(a) === null ? num(b) === null : num(b) !== null && Math.abs(a - b) < 1e-9);
    const mark = `<span class="ex-mark" title="${esc(exRescoredTitle(t))}" aria-label="re-scored">*</span>`;
    const rate = (v, rowsValue) => (num(v) === null ? DASH : `${fmtPct(v)}${t.rescored && rowsValue !== undefined && !same(v, rowsValue) ? mark : ''}`);
    return `<tr class="ex-row" data-exlink="${esc(t.link)}">` +
           `<td class="ex-task"><button type="button" class="node-link" data-exlink="${esc(t.link)}" data-fkey="ex:row:${t.i}" title="${esc(`open ${t.node} in its task view`)}">${esc(shortId(t.node))}</button>` +
           `<div class="ex-title muted small" title="${esc(t.title)}">${esc(t.title) || 'untitled'}</div></td>` +
           `<td class="ex-target"><code>${esc(t.target)}</code><div class="muted small">${esc(t.run.run)}` +
           `${t.run.latest === false ? ' · older' : ''}</div></td>` +
           `<td>${exSource(m, t.source)}</td><td class="num">${t.round === null ? DASH : fmtInt(t.round)}</td>` +
           `<td class="ex-dec">${badge(dec.label, dec.tone, dec.help, 'badge-sm')}${t.withdrawn ? badge('withdrawn', 'bad', 'main withdrew it after accepting it', 'badge-xs') : ''}</td>` +
           `<td class="num">${t.sim === null ? DASH : `${esc(fmtCompact(t.sim))}/10`}</td><td class="num">${rate(t.sRate, rows.s)}</td>` +
           `<td class="num" title="${esc(`${f.kb === 'raw' ? 'clean' : 'raw'} ${num(other) === null ? DASH : fmtPct(other)}`)}">${rate(kimi, f.kb === 'raw' ? undefined : rows.k)}</td>` +
           `<td>${t.rescored ? badge('re-scored', 'pending', exRescoredTitle(t), 'badge-xs') : DASH}</td>` +
           `<td>${exAuditBadge(t.audit, 'badge-sm')}</td><td class="ex-rep-cell">${reproduced || DASH}</td>` +
           `<td class="num">${fmtInt(t.f2p)} / ${fmtInt(t.p2p)}</td><td class="num">${fmtInt(n)}${n !== t.traj.length ? `<span class="muted small"> / ${fmtInt(t.traj.length)}</span>` : ''}</td></tr>`;
  }).join('');
  const table = rows
    ? `<div id="ex-tasks" class="tbl-wrap"${scrollAttr('ex:tasks')}><table class="tbl tbl-compact ex-tbl"><thead><tr>` +
      head('task', 'task') + head('target', 'DeepSWE task · run') + head('source', 'source') + head('round', 'round', 'num') +
      head('decision', 'decision') + head('sim', 'similarity', 'num') + head('srate', 'Nemotron', 'num', 'sc14 pass rate (metrics.json student_rate)') +
      head('krate', `Kimi ${f.kb}`, 'num', f.kb === 'raw' ? 'kimi passes / graded kimi rollouts' : 'kimi clean pass rate (metrics.json teacher_clean_rate)') +
      head('rescored', 're-scored', '', 'metrics.json rates or valid counts that differ from the task\'s own rollout rows (* on a rate that differs)') +
      head('audit', 'audit') + '<th scope="col" title="target failure patterns the audit marks reproduced (yes; ~ partial)">reproduced</th>' +
      head('f2p', 'F2P / P2P', 'num') + head('traj', 'trajectories', 'num', 'trajectories passing the trajectory filters / all of the task\'s') +
      `</tr></thead><tbody>${rows}</tbody></table></div>${exPager(pg, 'p', 'task', 'tasks')}`
    : note('No synthetic task matches the filters.');
  return table;
}

/** The trajectory table: 50 rows a page, each with its transcript (<slot>.md) and patch paths when the task was scored
    (a live probe's slots have none yet); the task button opens the task view with the node's drawer. */
function renderExplorerTrajTable(model, result, filters) {
  const m = obj(model), r = obj(result), f = Object.assign(explorerDefaults(), obj(filters));
  const pg = exPage(r.trajectories, f.tp);
  const rows = pg.rows.map((x) => {
    const t = x.task;
    const errs = [x.agentErr ? 'agent' : '', x.verifierErr ? 'verifier' : ''].filter(Boolean).join(' · ');
    const clean = x.model !== 'kimi' || x.status !== 'pass' ? DASH : x.clean === true ? 'clean' : x.clean === false
      ? badge('cheated', 'bad', 'the teacher audit flagged this pass', 'badge-xs') : '<span class="muted">not audited</span>';
    return `<tr><td class="ex-task"><button type="button" class="node-link" data-exlink="${esc(t.link)}" data-fkey="ex:traj:${x.i}">${esc(shortId(t.node))}</button>` +
           `<div class="muted small"><code>${esc(t.target)}</code></div></td><td>${exSource(m, t.source)}</td>` +
           `<td>${esc(lookup(EX_MODELS, x.model) || x.model)}</td><td><code>${esc(x.slot)}</code></td><td>${statusBadge(x.status)}</td>` +
           `<td class="num">${x.reward === null ? DASH : fmtFixed(x.reward, 1)}</td><td class="num">${esc(x.f2p) || DASH}</td>` +
           `<td class="num">${esc(x.p2p) || DASH}</td><td class="num">${fmtInt(x.tools)}</td><td class="num">${fmtInt(x.patchChars)}</td>` +
           `<td>${errs ? `<span class="err">${esc(errs)}</span>` : DASH}</td><td>${clean}</td>` +
           `<td class="ex-path">${x.transcript ? `<code>${esc(x.transcript)}</code><div class="muted small"><code>${esc(x.patch)}</code></div>`
                                               : '<span class="muted small">live probe: no transcript yet</span>'}</td></tr>`;
  }).join('');
  return rows
    ? `<div class="tbl-wrap"${scrollAttr('ex:traj')}><table class="tbl tbl-compact ex-tbl ex-tbl-traj"><thead><tr><th scope="col">task</th>` +
      '<th scope="col">source</th><th scope="col">model</th><th scope="col">slot</th><th scope="col">status</th><th scope="col" class="num">reward</th>' +
      '<th scope="col" class="num">F2P</th><th scope="col" class="num">P2P</th><th scope="col" class="num">tool calls</th>' +
      '<th scope="col" class="num">patch chars</th><th scope="col">errors</th><th scope="col">teacher audit</th>' +
      `<th scope="col">transcript · patch</th></tr></thead><tbody>${rows}</tbody></table></div>${exPager(pg, 'tp', 'trajectory', 'trajectories')}`
    : note('No trajectory matches the filters.');
}

/** The ranges whose minimum is above their maximum ("Similarity: min 8 > max 6"), for the results' warning. */
function exInverted(filters) {
  const f = Object.assign(explorerDefaults(), obj(filters));
  return Object.values(EX_RANGE_GROUPS).filter(([lo, hi]) => num(f[lo]) !== null && num(f[hi]) !== null && f[lo] > f[hi])
    .map(([lo, hi, label]) => {
      const show = (v) => (EX_PERCENT.has(lo) ? `${fmtCompact(v * 100)} %` : fmtCompact(v));
      return `${label}: min ${show(f[lo])} > max ${show(f[hi])}`;
    });
}

/** The results under the filter bar: KPI tiles, coverage per DeepSWE task (synthetic tasks or trajectories: a toggle, one
    scale at a time), failure-pattern coverage over the audited tasks, the distributions, the task table and the
    trajectory table, with client-side CSV / JSON exports of the filtered rows. opts.index (index.json) names how many
    DeepSWE tasks there are. */
function renderExplorerResults(doc, index, opts) {
  const o = obj(opts), ui = obj(o.ui), f = Object.assign(explorerDefaults(), obj(o.filters));
  const m = explorerModel(doc);
  if (!m) return '';
  const r = o.result || filterExplorer(m, f), st = explorerStats(m, r, f), k = st.kpi;
  const deepswe = arr(obj(index).tasks).length;                       // every DeepSWE task (index.json), converted or not
  const of = (a, b) => `${fmtInt(a)}<span class="tile-den"> / ${fmtInt(b)}</span>`;
  const tiles = [
    tile('Synthetic tasks', of(k.tasks, k.allTasks), esc(f.tw ? 'built tasks that pass the filters, with ≥ 1 matching trajectory'
                                                               : 'built tasks that pass the filters'), 'neutral'),
    tile('Trajectories', of(k.trajectories, k.allTrajectories), esc(r.trajActive ? 'of those tasks, passing the trajectory filters' : 'every rollout of those tasks'), 'neutral'),
    tile('DeepSWE tasks covered', of(k.targets, k.inScope),
         esc(`with ≥ 1 filtered task, of the ${fmtInt(k.inScope)} converted by the chosen sources` +
             `${deepswe ? ` (of ${fmtInt(deepswe)} DeepSWE tasks)` : ''}`), 'neutral'),
    tile('Accepted', fmtInt(k.accepted), esc(k.tasks ? `${fmtPct(k.accepted / k.tasks)} of the filtered tasks` : 'of the filtered tasks'), 'neutral'),
    tile('Nemotron passes', of(k.scPass, k.scTotal), esc('passing sc14 trajectories / sc14 trajectories shown'), 'neutral'),
    tile('Kimi passes', of(k.kimiPass, k.kimiTotal), esc(`${fmtInt(k.kimiClean)} clean (teacher audit) · passing / Kimi trajectories shown`), 'neutral'),
  ];
  const inverted = exInverted(f);
  const warn = inverted.length
    ? `<div class="alert alert-warn ex-alert" role="status"><b>${esc(inverted.length === 1 ? 'A range is inverted' : 'Ranges are inverted')}:</b> ` +
      `${esc(inverted.join(' · '))}; no task can pass it.</div>` : '';
  const sources = m.sources;
  const traj = f.cov === 'traj';
  const toggle = `<div class="ex-toggle" role="group" aria-label="coverage of">` +
    `<button type="button" class="btn btn-xs" data-exaction="cov:tasks" data-fkey="exa:cov:tasks" aria-pressed="${traj ? 'false' : 'true'}">Synthetic tasks</button>` +
    `<button type="button" class="btn btn-xs" data-exaction="cov:traj" data-fkey="exa:cov:traj" aria-pressed="${traj ? 'true' : 'false'}">Trajectories</button></div>`;
  const covRows = st.coverage.map((c) => ({label: c.id, title: c.id, values: traj ? c.traj : c.tasks}))
    .sort((a, b) => b.values.reduce((x, y) => x + y, 0) - a.values.reduce((x, y) => x + y, 0) || cmpStr(a.label, b.label));
  const covered = covRows.filter((x) => x.values.some(Boolean)).length;
  const coverage = renderHBars({key: `coverage-${traj ? 'traj' : 'tasks'}`, title: traj ? 'Trajectories per DeepSWE task' : 'Synthetic tasks per DeepSWE task',
    sub: `${fmtInt(covered)} of ${fmtInt(covRows.length)} converted DeepSWE tasks covered${deepswe ? ` (of ${fmtInt(deepswe)} in all)` : ''}`,
    category: 'DeepSWE task', ui, labelWidth: '300px', unit: traj ? ['trajectory', 'trajectories'] : ['task', 'tasks'],
    series: sources.map((s) => ({label: s.name, cls: s.cls})), rows: covRows, empty: 'No DeepSWE task in scope.'});
  // Failure-pattern coverage per target: audited tasks only.
  const ex = st.excluded, skipped = ex.notAudited + ex.notAuditedBatch + ex.notEligible;
  const why = [ex.notAuditedBatch ? `${fmtInt(ex.notAuditedBatch)} of a batch (never audited)` : '',
               ex.notAudited ? `${fmtInt(ex.notAudited)} eligible, not audited yet` : '',
               ex.notEligible ? `${fmtInt(ex.notEligible)} outside the audit's scope` : ''].filter(Boolean).join(' · ');
  const excludedNote = skipped
    ? `<p class="note small ex-excluded"><b>${esc(`${fmtInt(skipped)} of the ${exCount(k.tasks, 'filtered task', 'filtered tasks')} ` +
                                               `${skipped === 1 ? 'is' : 'are'} not counted here`)}</b>` +
      ` <span class="muted">(${esc(why)}): only a counted failure-match audit says which target patterns sc14 reproduces.</span></p>` : '';
  const shared = Math.max(0, ...st.patterns.flatMap((g) => g.patterns.map((p) => p.yes + p.partial)));
  const repSeries = [{label: 'reproduced: yes', cls: 'seg-yes'}, {label: 'partial', cls: 'seg-partial'}];
  const repLegend = `<div class="chart-legend ex-legend">${repSeries.map((x) => `<span class="lg"><span class="swatch ${x.cls}" aria-hidden="true"></span>${esc(x.label)}</span>`).join('')}` +
    '<span class="muted small">one scale for every target</span></div>';
  const groups = st.patterns.slice().sort((a, b) => b.audited - a.audited || cmpStr(a.id, b.id)).map((g, gi) => renderHBars({
    key: `patterns:${g.id}`, title: g.id, sub: `${plural(g.audited, 'audited task')}`, category: 'target failure pattern', ui, labelWidth: '320px',
    series: repSeries, max: shared, legend: false, unit: ['task', 'tasks'], first: gi === 0,
    rows: g.patterns.map((p) => ({label: `P${p.i + 1} · ${p.name || 'unnamed'}`, title: `P${p.i + 1} · ${p.name || 'unnamed'}${p.category ? ` (${p.category})` : ''}`,
                                 values: [p.yes, p.partial]})),
    empty: 'This target lists no failure patterns.'})).join('');
  const patterns = section('Failure-pattern coverage', excludedNote + (groups ? `${repLegend}<div class="ex-pattern-groups" data-roving="patterns">${groups}</div>`
    : note(k.tasks ? 'None of the filtered tasks has a counted failure-match audit.' : 'No synthetic task matches the filters.')),
    `<span class="muted small">${esc(`${plural(st.audited, 'audited task')} · per target pattern: how many of them reproduce it (yes / partial)`)}</span>`, 'ex-panel');
  // Distributions.
  const d = st.dist;
  const rateBins = (list, cols) => list.map((n, i) => ({label: cols[i].label, n, title: `pass rate ${cols[i].title}`}));
  const notScored = (n) => (n ? `${exCount(n, 'task', 'tasks')} not scored yet (no rate)` : '');
  const dists = [
    renderVBars({key: 'dist-sim', title: 'Similarity (failure similarity, /10)', category: 'similarity', ui, first: true,
                 bins: d.sim.map((n, i) => ({label: String(i), n, title: `similarity ${i}/10`})),
                 note: d.simNone ? `${exCount(d.simNone, 'task', 'tasks')} never judged` : ''}),
    renderVBars({key: 'dist-srate', title: 'Nemotron (sc14) pass rate, %', category: 'pass rate', ui, first: false,
                 bins: rateBins(d.sRate, d.sCols), note: notScored(d.sNone)}),
    renderVBars({key: 'dist-krate', title: `Kimi pass rate (${f.kb}), %`, category: 'pass rate', ui, first: false,
                 bins: rateBins(d.kRate, d.kCols), note: notScored(d.kNone)}),
    renderVBars({key: 'dist-audit', title: 'Audit verdict', category: 'audit', ui, first: false,
                 bins: EX_AUDITS.map((a, i) => ({label: EX_AUDIT_TEXT[a], n: d.audits[i], title: `audit: ${EX_AUDIT_TEXT[a]}`}))}),
    renderVBars({key: 'dist-round', title: 'Evolve round', category: 'round', ui, first: false,
                 bins: d.rounds.map((x) => ({label: String(x.round), n: x.n, title: `round ${x.round}${x.round === 0 ? ' (mask tasks)' : ''}`}))}),
  ];
  const exportBtn = (what, label) => `<button type="button" class="btn btn-xs" data-exexport="${what}" data-fkey="exe:${what}">${esc(label)}</button>`;
  const skip = '<button type="button" class="ex-skip" data-exaction="skip-tables" data-fkey="exa:skip-tables">Skip the charts: go to the task table</button>';
  return `<section class="panel ex-panel">${warn}<div class="tiles ex-kpis">${tiles.join('')}</div>${skip}</section>` +
    section('Coverage per DeepSWE task', `<div data-roving="coverage">${coverage}</div>`, toggle, 'ex-panel') + patterns +
    section('Distributions', `<div class="charts ex-dists" data-roving="dists">${dists.join('')}</div>`,
            `<span class="muted small">${esc(`over the ${exCount(k.tasks, 'filtered task', 'filtered tasks')}`)}</span>`, 'ex-panel') +
    section('Synthetic tasks', renderExplorerTaskTable(m, r, f),
            `<span class="muted small">${esc(`${exCount(k.tasks, 'task', 'tasks')}${k.rescored ? ` (${fmtInt(k.rescored)} re-scored: * on a rate that differs)` : ''}` +
                                             ' · click a row to open it in its task view')}</span>` +
            `<span class="ex-exports">${exportBtn('tasks.csv', 'CSV')}${exportBtn('tasks.json', 'JSON')}</span>`, 'ex-panel') +
    section('Trajectories', renderExplorerTrajTable(m, r, f),
            `<span class="muted small">${esc(`${exCount(k.trajectories, 'trajectory', 'trajectories')} of the filtered tasks`)}</span>` +
            `<span class="ex-exports">${exportBtn('trajectories.csv', 'CSV')}${exportBtn('trajectories.json', 'JSON')}</span>`, 'ex-panel');
}

/** The Explorer's head (#ex-head, refreshed with a new explorer.json): its title, how many rows from how many runs, the
    file's stamp (and a failed refresh), and what the rows cover (explorerScope, collapsible). */
function renderExplorerHead(model, index, opts) {
  const m = obj(model), o = obj(opts), stamp = num(m.generatedAt);
  const sources = arr(m.sources).map((s) => {
    const t = obj(obj(m.totals)[s.source]);
    return `${s.name}: ${exCount(num(t.runs) || 0, 'run', 'runs')}, ${exCount(num(t.tasks) || 0, 'task', 'tasks')}`;
  }).join(' · ');
  const scope = det(o.ui, 'ex:scope', `<span class="muted small">${esc(`What the rows are · ${sources}`)}</span>`,
                    `<ul class="plain small">${explorerScope(m, index).map((x) => `<li>${esc(x)}</li>`).join('')}</ul>`, false,
                    'fold fold-inline ex-scope');
  return `<div class="ex-head"><h1 class="ex-h1">Explorer</h1><span class="muted small">${esc(`synthetic tasks and their trajectories · ` +
    `${exCount(arr(m.tasks).length, 'task', 'tasks')}, ${exCount(arr(m.trajectories).length, 'trajectory', 'trajectories')} from ` +
    `${exCount(arr(m.runs).length, 'run', 'runs')}${stamp !== null ? ` · explorer.json of ${fmtDate(stamp)}` : ''}`)}</span>` +
    (o.error ? badge('refresh failed', 'pending', `data/explorer.json: ${text(o.error)}; showing the last copy received`) : '') + `</div>${scope}`;
}

/** The Explorer tab: its scope (what the rows are), the filter bar, then the results (re-rendered alone while filters
    change), or why there is nothing yet (explorer.json loading, or not available from this collector). opts: {ui, filters,
    query, runQuery (v1.8: the run checklist's search), bad, error, now}. */
function renderExplorer(doc, index, opts) {
  const o = obj(opts), f = Object.assign(explorerDefaults(), obj(o.filters));
  const m = explorerModel(doc);
  if (!m) {
    return section('Explorer', note(o.error ? `data/explorer.json not available: ${text(o.error)} (a v1.7 collector writes it)`
                                            : 'Loading data/explorer.json…'));
  }
  const result = filterExplorer(m, f);
  return `<div class="ex"><div id="ex-head">${renderExplorerHead(m, index, o)}</div>` +
         `${renderExplorerFilters(m, f, {ui: o.ui, query: o.query, runQuery: o.runQuery, bad: o.bad, result})}` +
         `<div id="ex-results" class="ex-results">${renderExplorerResults(doc, index, Object.assign({}, o, {result}))}</div>` +
         '<div id="ex-tip" class="ex-tip" role="tooltip" hidden></div></div>';
}

/** What the Explorer's rows cover, as sentences: each official DeepSWE target's latest run (how many official runs
    index.json lists, how many of them are older runs left out), each batch's runs (only those behind its outputs batch;
    the other runs of that tree are never read), and abandoned build roots (never rows) vs the abandoned decision. */
function explorerScope(model, index) {
  const m = obj(model), ix = obj(index), out = [];
  const runs = arr(ix.runs).filter(isObj), official = runs.filter((r) => !isBatchRun(r)).length;
  const shown = (source) => num(obj(obj(m.totals)[source]).runs) || 0;
  for (const s of arr(m.sources)) {
    if (s.audited && arr(m.runs).some((r) => r.source === s && r.latest === false)) {   // v1.8: every official run
      const older = arr(m.runs).filter((r) => r.source === s && r.latest === false).length;
      out.push(`${s.name}: every built task of every official run, ${exCount(shown(s.source), 'run', 'runs')}: each DeepSWE ` +
               "target's latest run (data/task/<task>.json) and " + exCount(older, 'older official run', 'older official runs') +
               ' of the same targets (data/run/hengyuf/<run>.json; their rows link with run=).');
    } else if (s.audited) {
      const left = Math.max(0, official - shown(s.source));
      out.push(`${s.name}: every built task of each official DeepSWE target's latest run (its data/task/<task>.json), ` +
               `${exCount(shown(s.source), 'run', 'runs')}` + (runs.length ? (left ? `; ${exCount(left, 'older official run', 'older official runs')} ` +
               'of the same targets left out' : ', no older run left out') : '') + '.');
    } else {
      const batch = arr(ix.batches).find((b) => isObj(b) && b.source === s.source);
      out.push(`${s.name}: only the ${exCount(shown(s.source), 'run', 'runs')} behind ${batch ? `the ${exCount(num(batch.tasks) || 0, 'task', 'tasks')} of ` : ''}` +
               `${s.name}'s batch (outputs/tasks.jsonl); the other runs of that tree are never read, so not shown. Never audited.`);
    }
  }
  out.push(EX_ABANDONED_NOTE);
  return out;
}

/** One CSV cell (RFC 4180): quoted when it holds a comma, quote or line break; text that a spreadsheet would run as a
    formula (its first character, after any leading whitespace, = + - @ or their full-width forms; or a tab, CR or LF
    first) is prefixed with an apostrophe. */
function csvCell(v) {
  if (v === null || v === undefined) return '';
  if (typeof v === 'boolean') return v ? 'true' : 'false';
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
  let s = text(v);
  if (/^[\t\r\n]/.test(s) || /^\s*[=+\-@＝＋－＠]/.test(s)) s = `'${s}`;
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

/** A CSV file: a UTF-8 byte-order mark (spreadsheets then read "·" and non-ASCII titles right), a header, CRLF rows. */
function csvText(columns, rows) {
  return '\uFEFF' + [columns.map(csvCell).join(','), ...rows.map((row) => row.map(csvCell).join(','))].join('\r\n') + '\r\n';
}

/** "7/8" as [7, 8] (passed, total); [null, null] for anything else. */
function exFraction(v) {
  const m = /^(\d+)\/(\d+)$/.exec(text(v));
  return m ? [Number(m[1]), Number(m[2])] : [null, null];
}

/** A client-side export of the filtered rows: {name, type, text}. what: tasks.csv | tasks.json | trajectories.csv |
    trajectories.json. The derived fields (DeepSWE task, source, links, dirs, transcript paths) are spelled out; a
    trajectory's F2P / P2P are passed and total numbers. */
function explorerExport(doc, filters, what) {
  const m = explorerModel(doc), f = Object.assign(explorerDefaults(), obj(filters));
  if (!m || !EX_EXPORTS.includes(what)) return null;
  const r = filterExplorer(m, f);
  const stamp = dateOf(m.generatedAt);
  const tag = stamp ? `${stamp.getFullYear()}${pad2(stamp.getMonth() + 1)}${pad2(stamp.getDate())}-${pad2(stamp.getHours())}${pad2(stamp.getMinutes())}` : 'data';
  const labels = (t, list) => arr(list).map((k) => exPatternLabel(m, t.target, k)).join('; ');
  const taskCols = ['source', 'run', 'deepswe_task', 'node', 'task_id', 'round', 'branch', 'kind', 'title', 'decision', 'accepted',
                    'withdrawn', 'in_batch', 'similarity', 'student_rate', 'teacher_clean_rate', 'teacher_rate', 'valid_sc14',
                    'valid_kimi', 'rates_differ', 'rollouts_student_rate', 'rollouts_teacher_clean_rate', 'audit',
                    'reproduced_yes', 'reproduced_partial', 'f2p_tests', 'p2p_tests', 'matching_trajectories', 'trajectories',
                    'link', 'cand_dir', 'rollouts_dir'];
  const taskRow = (t) => [t.source, t.run.run, t.target, t.node, t.taskId, t.round, t.branch, t.kind, t.title, t.decision, t.accepted,
                          t.withdrawn, t.inBatch, t.sim, t.sRate, t.kClean, t.kRaw, t.valid ? t.valid[0] : null,
                          t.valid ? t.valid[1] : null, t.rescored, obj(t.rowsRates).s, obj(t.rowsRates).k, t.audit,
                          labels(t, t.yes), labels(t, t.partial), t.f2p, t.p2p, r.matched.get(t) || 0, t.traj.length,
                          t.link, t.dir, t.rollouts];
  const trajCols = ['source', 'run', 'deepswe_task', 'node', 'task_id', 'slot', 'model', 'status', 'reward', 'f2p_passed',
                    'f2p_total', 'p2p_passed', 'p2p_total', 'tool_calls', 'patch_chars', 'agent_error', 'verifier_error', 'clean',
                    'transcript', 'patch', 'link'];
  const trajRow = (x) => [x.task.source, x.task.run.run, x.task.target, x.task.node, x.task.taskId, x.slot, x.model, x.status,
                          x.reward, ...exFraction(x.f2p), ...exFraction(x.p2p), x.tools, x.patchChars, x.agentErr, x.verifierErr,
                          x.clean, x.transcript || null, x.patch || null, x.task.link];
  const tasks = what.startsWith('tasks.');
  const cols = tasks ? taskCols : trajCols;
  const rows = tasks ? exSortedTasks(r, f).map(taskRow) : r.trajectories.map(trajRow);
  const base = `explorer-${tasks ? 'tasks' : 'trajectories'}-${tag}`;
  if (what.endsWith('.csv')) return {name: `${base}.csv`, type: 'text/csv;charset=utf-8', text: csvText(cols, rows)};
  const objects = rows.map((row) => Object.fromEntries(cols.map((c, i) => [c, row[i] === undefined ? null : row[i]])));
  return {name: `${base}.json`, type: 'application/json', text: JSON.stringify({
    explorer_generated_at: m.generatedAt, view: explorerHash(Object.assign({}, f, {p: 1, tp: 1})), count: objects.length,
    [tasks ? 'tasks' : 'trajectories']: objects}, null, 1) + '\n'};
}

/** The next mark of a roving tab stop: from position i of n, by an arrow (+1 / -1) or Home / End (-Infinity / Infinity),
    clamped to the panel's marks. */
function exRovingIndex(i, n, step) {
  if (!n) return -1;
  if (step === -Infinity) return 0;
  if (step === Infinity) return n - 1;
  return Math.max(0, Math.min(n - 1, (int(i) === null ? 0 : i) + step));
}

/** What a filter-bar or data refresh may leave in place: the bar's choices (sources, rounds, kinds, DeepSWE tasks). */
function exBarKey(model) {
  const m = obj(model);
  return JSON.stringify([arr(m.sources).map((s) => [s.source, s.name, s.label]), arr(m.rounds), arr(m.kinds), arr(m.targetIds),
                         arr(m.families).map((f) => f.key), arr(m.runNames)]);   // v1.8: the run families and runs
}

// ---------------------------------------------------------------------------------------- routing

/** '#task=<id>[&run=<batch run>]&node=<id>' | '#monitor' | '#explorer[&<filters>]' -> {view, task, node, run} (run, v1.6:
    a batch run's view; v1.7: the Explorer, its filters in `explorer`, the query after "explorer&"). */
function parseHash(hash) {
  const h = text(hash).replace(/^#/, '');
  if (h === 'monitor' || h.startsWith('monitor&') || h.startsWith('monitor/')) return {view: 'monitor', task: null, node: null, run: null};
  if (h === 'explorer' || h.startsWith('explorer&')) {
    return {view: 'explorer', task: null, node: null, run: null, explorer: h.slice('explorer'.length).replace(/^&/, '')};
  }
  let params;
  try { params = new URLSearchParams(h); } catch (error) { params = new URLSearchParams(''); }
  return {view: 'task', task: params.get('task') || null, node: params.get('node') || null, run: params.get('run') || null};
}

function taskHash(task, node, run) {
  return `#task=${encodeURIComponent(text(task))}${run ? `&run=${encodeURIComponent(text(run))}` : ''}` +
         `${node ? `&node=${encodeURIComponent(text(node))}` : ''}`;
}

// --------------------------------------------------------------------------------- refresh helpers (pure)

/** What a render call does with the main view and the drawer: 'skip' when nothing they show changed (no new collector
    cycle, same selection: the DOM, its scroll positions and any text selection stay untouched); 'hold' when an
    automatic refresh would rebuild them while text is selected (retried at the next tick); else 'render'.
    o = {changed, force, auto, selecting}. */
function refreshAction(o) {
  const x = obj(o);
  if (!x.changed && !x.force) return 'skip';
  return x.auto && x.selecting ? 'hold' : 'render';
}

/** True for a non-empty text selection (window.getSelection(): a Selection, whose isCollapsed is a boolean). */
function hasTextSelection(sel) {
  if (!sel || typeof sel !== 'object' || sel.isCollapsed !== false) return false;
  try { return typeof sel.toString === 'function' && String(sel.toString()).length > 0; } catch (error) { return false; }
}

/** Scroll offsets of the elements marked data-scroll, by key: {key: [left, top]}, scrolled ones only (first wins). */
function captureScroll(elements) {
  const out = Object.create(null);
  for (const e of arr(elements)) {
    const k = e && e.dataset ? text(e.dataset.scroll) : '';
    const left = num(e && e.scrollLeft) || 0, top = num(e && e.scrollTop) || 0;
    if (k && !hasOwn(out, k) && (left || top)) out[k] = [left, top];
  }
  return out;
}

/** Puts captured offsets back on the (re-rendered) elements with the same data-scroll key; returns how many. */
function restoreScroll(elements, saved) {
  const s = obj(saved), done = new Set();
  for (const e of arr(elements)) {
    const k = e && e.dataset ? text(e.dataset.scroll) : '';
    const v = k && !done.has(k) && hasOwn(s, k) ? pair(s[k]) : null;
    if (!v) continue;
    done.add(k);
    e.scrollLeft = v[0];
    e.scrollTop = v[1];
  }
  return done.size;
}

/** Open state of <details data-key> elements: {key: open} (first wins). */
function captureOpen(elements) {
  const out = Object.create(null);
  for (const e of arr(elements)) {
    const k = e && e.dataset ? text(e.dataset.key) : '';
    if (k && !hasOwn(out, k)) out[k] = !!e.open;
  }
  return out;
}

function cssEscape(s) {
  return typeof CSS !== 'undefined' && CSS && typeof CSS.escape === 'function' ? CSS.escape(text(s)) : text(s).replace(/["\\]/g, '\\$&');
}

// -------------------------------------------------------------------------------------- DOM wiring

/** The page controller. env = {window, document, fetch, now?}; only plain DOM calls, all null-guarded. */
function createApp(env) {
  const win = env.window, doc = env.document, fetchFn = env.fetch;
  const clock = typeof env.now === 'function' ? env.now : () => Date.now() / 1000;
  const el = (id) => doc.getElementById(id);
  const all = (root, sel) => (root && typeof root.querySelectorAll === 'function' ? Array.from(root.querySelectorAll(sel)) : []);
  const find = (root, sel) => (root && typeof root.querySelector === 'function' ? root.querySelector(sel) : null);
  const bodyClass = (cls, on) => { if (doc.body && doc.body.classList) doc.body.classList.toggle(cls, !!on); };
  const st = {
    index: null, details: {}, detailErr: {}, fleet: null, api: null, fetchedAt: {}, seen: {}, auth: false,
    errors: {index: null, fleet: null, api: null},
    view: 'task', taskId: null, nodeId: null, lastTask: null, runSel: null, resolveNode: null,
    ui: {openState: Object.create(null), showAbandoned: false, selected: null},
    menu: {open: false, query: '', active: 0},
    loading: null, again: null, timers: [], lastKey: null, lastPage: null, toTop: false, uiVersion: 0,
    held: false, shownGen: null, html: {}, drawerW: null,
    // v1.7: the Explorer: data/explorer.json, its filters (from the hash), the DeepSWE task checklist's search text, the
    // range inputs rejected as typed ({key: raw text}); exBar: the filter bar's choices on screen (exBarKey); exFull: the
    // next render must rebuild the bar too (Reset, or a hash that set other filters)
    explorer: null, ex: {f: explorerDefaults(), tq: '', rq: '', bad: {}}, exBar: null, exFull: false,
  };
  st.errors.explorer = null;
  const UNCHANGED = {};                                    // getJSON on HTTP 304: keep the copy we hold

  async function getJSON(path) {
    // no-cache: the browser revalidates its copy (ETag), so an unchanged file costs a 304 instead of a download.
    const res = await fetchFn(path, {cache: 'no-cache', credentials: 'same-origin', headers: {Accept: 'application/json'}});
    if (res.status === 403 || res.status === 401) {
      const error = new Error('missing or expired token');
      error.auth = true;
      throw error;
    }
    if (res.status === 304) return UNCHANGED;
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res.json();
  }

  const fresh = (k) => st.fetchedAt[k] && clock() - st.fetchedAt[k] < REFRESH_MS / 1000 - 1;
  // v1.6: the run the task view shows ({kind, run}: its official run, or the batch run of the hash's run=), and where its
  // document is ({key, path}); a batch run's document changes only when index.json's detail_generated_at does (v1.8: so
  // does an older official run's, data/run/hengyuf/<run>.json).
  const curView = () => { const task = findTask(st.index, st.taskId); return task ? selectedView(viewsOf(st.index, task), st.runSel) : null; };
  const curDetail = () => { const v = curView(); return v ? viewDetail(v, st.taskId) : null; };
  const detailKey = () => { const d = curDetail(); return d ? d.key : text(st.taskId); };
  // collect.py rewrites every data file once per cycle: refetch a task/fleet/api file only when index.json shows a
  // new cycle (the task files reach megabytes), or when the copy we hold predates the cycle (fetched mid-write).
  const generation = () => num(obj(st.index).generated_at);
  const behind = (doc, seen) => !isObj(doc) || seen !== generation() ||
    (num(doc.generated_at) !== null && generation() !== null && doc.generated_at < generation() - 5);

  /** v1.8: a #task=<t>&node=<n> link without run= (st.resolveNode, set by route) whose node is not in the default run's
      document, e.g. a link to an official1 node shared before official2 became the task's latest run: the first other
      run of the task in the selector's order (older official runs, then batch runs) whose document has the node is shown
      instead, and the hash names it (replaceState). A node in the default run, or in no run, leaves the view alone. */
  async function resolveDeepLink() {
    const task = findTask(st.index, st.taskId), node = st.resolveNode;
    const views = task ? viewsOf(st.index, task) : [];
    const first = views.length ? viewDetail(views[0], st.taskId) : null;
    if (!first || !isObj(st.details[first.key])) return;         // the default document is not here yet: next load
    st.resolveNode = null;
    if (findNode(st.details[first.key], node) || views.length < 2) return;
    for (const view of views.slice(1)) {
      const where = viewDetail(view, st.taskId);
      if (!where) continue;
      if (!isObj(st.details[where.key])) {
        try {
          const d = await getJSON(where.path);
          if (d === UNCHANGED) continue;
          st.details[where.key] = d;
          st.detailErr[where.key] = null;
          st.seen[`task:${where.key}`] = generation();
        } catch (error) {
          if (error.auth) throw error;
          continue;
        }
      }
      if (findNode(st.details[where.key], node)) {
        if (st.view !== 'task' || !task || st.taskId !== task.task_id || st.nodeId !== node || st.runSel) return;   // moved on
        st.runSel = view.run.run;
        win.history.replaceState(null, '', taskHash(st.taskId, node, st.runSel));
        return;
      }
    }
  }

  /** Fetch what the current view needs, then render. force: refetch index.json (and the view's files if a new
      cycle landed); 'all' (the Refresh button): refetch everything. auto: a timer asked (its render waits while text
      is selected). Calls during a load are coalesced into one more run, as forced as the strongest request and
      automatic only if every request was. */
  function load(force, auto) {
    if (st.loading) {
      const rank = force === 'all' ? 3 : force ? 2 : 1;
      st.again = {rank: Math.max(rank, st.again ? st.again.rank : 0), auto: (st.again ? st.again.auto : true) && !!auto};
      return st.loading;
    }
    const all = force === 'all';
    st.loading = (async () => {
      setBusy(true);
      try {
        if (force || !st.index || !fresh('index')) {
          const index = await getJSON('data/index.json');
          if (index !== UNCHANGED) st.index = index;
          st.fetchedAt.index = clock();
          st.errors.index = null;
          st.auth = false;
        }
        if (st.view === 'task' && !st.taskId && st.index) {
          st.taskId = defaultTask(st.index);
          st.lastTask = st.taskId;
          if (st.taskId) win.history.replaceState(null, '', taskHash(st.taskId, st.nodeId));
        }
        if (st.view === 'task' && st.taskId) {
          const view = curView(), where = view ? viewDetail(view, st.taskId) : null;
          const id = where ? where.key : st.taskId;
          const stamp = view && (view.kind === 'batch' || view.older) ? num(view.run.detail_generated_at) : null;
          const stale = stamp !== null ? !isObj(st.details[id]) || st.details[id].generated_at !== stamp
            : behind(st.details[id], st.seen[`task:${id}`]);
          if (where && (all || stale)) {
            try {
              const d = await getJSON(where.path);
              if (d !== UNCHANGED) st.details[id] = d;
              st.detailErr[id] = null;
              st.seen[`task:${id}`] = generation();
            } catch (error) {
              if (error.auth) throw error;
              st.detailErr[id] = error.message;
            }
          }
          if (st.resolveNode && st.resolveNode === st.nodeId && !st.runSel) await resolveDeepLink();   // v1.8
        } else if (st.view === 'monitor' && (all || behind(st.fleet, st.seen.fleet) || behind(st.api, st.seen.api))) {
          const [fleet, api] = await Promise.allSettled([getJSON('data/fleet.json'), getJSON('data/api.json')]);
          for (const [k, res] of [['fleet', fleet], ['api', api]]) {
            if (res.status === 'fulfilled') {
              if (res.value !== UNCHANGED) st[k] = res.value;
              st.errors[k] = null;
              st.seen[k] = generation();
            } else if (res.reason && res.reason.auth) throw res.reason;
            else st.errors[k] = text(res.reason && res.reason.message) || 'fetch failed';
          }
        } else if (st.view === 'explorer') {
          // v1.7: explorer.json changes only when index.json "explorer" names a newer stamp (else: a new cycle, a 304)
          const stamp = num(obj(obj(st.index).explorer).generated_at);
          const stale = !isObj(st.explorer) || (stamp !== null ? st.explorer.generated_at !== stamp : st.seen.explorer !== generation());
          if (all || stale) {
            try {
              const d = await getJSON('data/explorer.json');
              if (d !== UNCHANGED) st.explorer = d;
              st.errors.explorer = null;
              st.seen.explorer = generation();
            } catch (error) {
              if (error.auth) throw error;
              st.errors.explorer = error.message;
            }
          }
        }
      } catch (error) {
        if (error && error.auth) st.auth = true;
        else st.errors.index = text(error && error.message) || 'fetch failed';
      } finally {
        setBusy(false);
        render(false, !!auto);
      }
    })().finally(() => {
      st.loading = null;
      if (st.again) {
        const again = st.again;
        st.again = null;
        load(again.rank === 3 ? 'all' : again.rank === 2, again.auto);
      }
    });
    return st.loading;
  }

  function setBusy(on) { bodyClass('is-loading', on); }

  /** innerHTML of a header element, written only when it changed (a refresh leaves an unchanged banner alone). */
  function setHTML(id, html) {
    const e = el(id);
    if (!e || st.html[id] === html) return;
    st.html[id] = html;
    e.innerHTML = html;
  }

  /** Everything the main view and the drawer depend on; a refresh that changes none of it skips their re-render, so
      scroll positions, text selections and focus survive the 60 s cycle. */
  function viewKey() {
    const gen = (o) => (isObj(o) ? text(o.generated_at) : '-');
    if (st.view === 'explorer') {   // v1.7: its own file and filters, never index.json's stamp (a cycle that leaves
      const ix = obj(st.index);     // explorer.json as it was changes nothing here: inputs keep their focus and caret)
      return JSON.stringify(['explorer', st.auth, st.uiVersion, gen(st.explorer), text(st.errors.explorer),
                             explorerHash(st.ex.f), arr(ix.tasks).length, arr(ix.runs).length,
                             arr(ix.batches).map((b) => [obj(b).source, obj(b).tasks])]);
    }
    const key = detailKey();
    const data = st.view === 'task' ? [gen(st.details[key]), text(st.detailErr[key]), st.runSel]
      : [gen(st.fleet), gen(st.api), text(st.errors.fleet), text(st.errors.api)];
    return JSON.stringify([st.view, st.taskId, st.nodeId, st.auth, st.uiVersion, st.ui.showAbandoned, st.ui.treeWidth,
                           gen(st.index), data]);
  }

  function selecting() {
    try { return hasTextSelection(typeof win.getSelection === 'function' ? win.getSelection() : null); } catch (error) { return false; }
  }

  function windowScroll() {
    const x = num(win.scrollX) !== null ? win.scrollX : num(win.pageXOffset), y = num(win.scrollY) !== null ? win.scrollY : num(win.pageYOffset);
    return x === null || y === null ? null : [x, y];
  }

  /** Width of main's content box: clientWidth less its padding (which makes room for the drawer when it is open). */
  function contentWidth(main) {
    const w = main ? num(main.clientWidth) : null;
    if (!w) return null;
    let pad = 48;                                           // main padding 2x24 (style.css)
    try {
      const cs = typeof win.getComputedStyle === 'function' ? win.getComputedStyle(main) : null;
      const l = cs ? parseFloat(cs.paddingLeft) : NaN, r = cs ? parseFloat(cs.paddingRight) : NaN;
      if (Number.isFinite(l) && Number.isFinite(r)) pad = l + r;
    } catch (error) { /* keep the default */ }
    return w - pad;
  }

  /** Before a re-render of the same page: the scroll offsets in main and in the drawer (keyed by data-scroll), and the
      open state of every <details data-key> that no toggle event recorded yet. */
  function capture(winPos) {
    const main = el('main'), drawer = el('drawer');
    const open = captureOpen(all(main, 'details[data-key]').concat(all(drawer, 'details[data-key]')));
    for (const k of Object.keys(open)) if (!hasOwn(st.ui.openState, k)) st.ui.openState[k] = open[k];
    const inDrawer = drawer && !drawer.hidden && drawer.dataset && drawer.dataset.node;
    return {win: winPos, main: captureScroll(all(main, '[data-scroll]')),
            drawer: inDrawer ? {node: drawer.dataset.node, scroll: captureScroll(all(drawer, '[data-scroll]'))} : null};
  }

  function restore(saved) {
    restoreScroll(all(el('main'), '[data-scroll]'), saved.main);
    const drawer = el('drawer');
    if (saved.drawer && drawer && !drawer.hidden && drawer.dataset && drawer.dataset.node === saved.drawer.node) {
      restoreScroll(all(drawer, '[data-scroll]'), saved.drawer.scroll);
    }
    const now = windowScroll();
    if (saved.win && typeof win.scrollTo === 'function' && (!now || now[0] !== saved.win[0] || now[1] !== saved.win[1])) {
      win.scrollTo(saved.win[0], saved.win[1]);
    }
  }

  /** Re-render the header, and the main view and the drawer when what they show changed (refreshAction): a new cycle
      keeps the window, tree, drawer and <pre> scroll positions, the open collapsibles and the focus; an automatic
      refresh waits while text is selected. force: re-render even if nothing changed; auto: a timer asked. */
  function render(force, auto) {
    const now = clock();
    const winPos = windowScroll();                          // before the drawer state changes the layout
    // The drawer narrows the main column on wide screens: set its state before measuring the tree's room.
    bodyClass('drawer-open', !st.auth && st.view === 'task' && !!st.nodeId);
    const main = el('main');
    const width = contentWidth(main);
    if (width) st.ui.treeWidth = width - 42;                // panel padding 2x18 + borders
    const key = viewKey();
    const action = refreshAction({changed: key !== st.lastKey, force, auto, selecting: !!auto && selecting()});
    st.held = action === 'hold';
    if (action !== 'render') {
      renderHeader(now);
      if (st.menu.open) renderMenu();
      return;
    }
    st.lastKey = key;
    st.shownGen = generation();
    const page = JSON.stringify([st.auth, st.view, st.taskId]);
    const saved = page === st.lastPage && !st.toTop ? capture(winPos) : null;
    st.lastPage = page;
    renderHeader(now);
    const active = doc.activeElement && doc.activeElement.dataset ? doc.activeElement.dataset.fkey : null;
    if (main) {
      if (st.auth) main.innerHTML = renderAuthError();
      else if (st.view === 'monitor') {
        main.innerHTML = renderMonitor(st.fleet, st.api, {now, ui: st.ui, fleetError: st.errors.fleet, apiError: st.errors.api,
                                                          index: st.index});   // v1.8: the queues are index.json's
      } else if (st.view === 'explorer') {                    // v1.7
        const m = explorerModel(st.explorer), bar = m ? exBarKey(m) : null;
        if (saved && !st.exFull && bar !== null && bar === st.exBar && el('ex-results')) explorerUpdate('data');
        else {
          main.innerHTML = renderExplorer(st.explorer, st.index, {now, ui: st.ui, filters: st.ex.f, query: st.ex.tq, runQuery: st.ex.rq,
                                                                  bad: st.ex.bad, error: st.errors.explorer});
        }
        st.exBar = bar;
        st.exFull = false;
      } else {
        main.innerHTML = renderTaskView(st.index, st.taskId, st.details[detailKey()],
                                        {now, ui: st.ui, detailError: st.detailErr[detailKey()], run: st.runSel});
      }
    }
    renderDrawer(now);
    if (saved) restore(saved);
    else if (st.toTop && typeof win.scrollTo === 'function') win.scrollTo(0, 0);   // another task or tab: from the top
    st.toTop = false;
    if (active) refocus(active);
    if (st.menu.open) renderMenu();
  }

  function renderHeader(now) {
    setHTML('status', renderStatus(st.index, now, {shown: st.shownGen, held: st.held}));
    const kept = (k, doc) => (doc ? st.errors[k] : null);   // a failed refresh of a file still shown (older copy)
    const errors = {index: st.errors.index};
    const key = detailKey();
    if (st.view === 'task' && st.details[key] && st.detailErr[key]) errors[`task/${st.taskId}`] = st.detailErr[key];
    if (st.view === 'monitor') Object.assign(errors, {fleet: kept('fleet', st.fleet), api: kept('api', st.api)});
    if (st.view === 'explorer') errors.explorer = kept('explorer', st.explorer);
    setHTML('banner', renderBanner(st.index, now, {errors, ui: st.ui}));
    const tabTask = el('tab-task'), tabMon = el('tab-monitor'), tabEx = el('tab-explorer');
    if (tabEx) {                                              // v1.7: the Explorer tab keeps the last filters
      tabEx.classList.toggle('active', st.view === 'explorer');
      tabEx.setAttribute('href', explorerHash(st.ex.f));
      tabEx.setAttribute('aria-current', st.view === 'explorer' ? 'page' : 'false');
    }
    if (tabTask) {
      tabTask.classList.toggle('active', st.view === 'task');
      const last = st.lastTask || st.taskId;
      tabTask.setAttribute('href', last ? taskHash(last) : '#');
      tabTask.setAttribute('aria-current', st.view === 'task' ? 'page' : 'false');
    }
    if (tabMon) {
      tabMon.classList.toggle('active', st.view === 'monitor');
      tabMon.setAttribute('aria-current', st.view === 'monitor' ? 'page' : 'false');
    }
    const input = el('task-input');
    if (input && doc.activeElement !== input) input.value = st.view === 'task' ? text(st.taskId) : '';
    doc.title = st.view === 'monitor' ? 'Monitor · swe-task official synthesis'
      : st.view === 'explorer' ? 'Explorer · swe-task official synthesis'
      : `${st.taskId ? `${st.taskId} · ` : ''}swe-task official synthesis`;
  }

  function renderDrawer(now) {
    const drawer = el('drawer'), grip = el('drawer-grip');
    if (!drawer) return;
    const detail = st.details[detailKey()];
    if (st.auth || st.view !== 'task' || !st.nodeId) {
      drawer.hidden = true;
      drawer.innerHTML = '';
      if (grip) grip.hidden = true;
      bodyClass('drawer-open', false);
      return;
    }
    const top = drawer.dataset && drawer.dataset.node === st.nodeId ? drawer.scrollTop : 0;
    drawer.innerHTML = renderNodeDetail(findNode(detail, st.nodeId), detail, st.ui, now);
    if (drawer.dataset) drawer.dataset.node = st.nodeId;
    drawer.hidden = false;
    if (grip) grip.hidden = false;
    drawer.scrollTop = top;
    bodyClass('drawer-open', true);
  }

  /** Focus the element (re-rendered) that had it: data-fkey names tree cards, round-0 rows and drawer links. */
  function refocus(fkey) {
    const sel = `[data-fkey="${cssEscape(fkey)}"]`;
    const e = find(el('main'), sel) || find(el('drawer'), sel);
    if (e && typeof e.focus === 'function') e.focus({preventScroll: true});
    return e;
  }

  // Routing: the hash is the source of truth for the view, task and open node.
  function route(navigated) {
    const h = parseHash(win.location.hash);
    const before = JSON.stringify([st.view, st.taskId]);
    st.view = h.view;
    if (h.view === 'task') {
      if (h.task !== st.taskId) st.nodeId = null;
      st.taskId = h.task || st.taskId || (st.index ? defaultTask(st.index) : null);
      st.runSel = h.run || null;                           // v1.6: a batch run's view of the task
      st.nodeId = h.node || null;
      st.resolveNode = h.node && !h.run ? h.node : null;  // v1.8: a node link without run=: is the node in the default run?
      if (st.taskId) st.lastTask = st.taskId;
    } else {
      st.nodeId = null;            // v1.7: another view has no node drawer; the task view's hash names its node again
    }
    if (h.view === 'explorer') {   // v1.7: the hash holds every filter; other filters than the bar shows: rebuild it
      const f = parseExplorerParams(h.explorer);
      if (explorerHash(f) !== explorerHash(st.ex.f)) { st.exFull = true; st.ex.bad = {}; }
      st.ex.f = f;
    }
    st.ui.selected = st.view === 'task' ? st.nodeId : null;
    if (navigated && JSON.stringify([st.view, st.taskId]) !== before) st.toTop = true;
    render();
    return load(false);
  }

  /** Open (or, for the open one, close) a node's drawer. origin: 'card' / 'link' bring its tree card into view and
      focus it, 'row' / 'audit-row' focus the round-0 / failure-match audit table row it came from; CSS scroll margins
      keep the target clear of the sticky header and of a drawer that overlays the page. */
  function selectNode(id, origin) {
    if (st.view !== 'task') return;                          // v1.7: only the task view has node drawers (and its hash)
    st.nodeId = id && id !== st.nodeId ? id : null;
    st.ui.selected = st.nodeId;
    if (st.taskId) win.history.replaceState(null, '', taskHash(st.taskId, st.nodeId, st.runSel));
    render();
    if (!st.nodeId) return;
    const row = origin === 'row' ? 'r0' : origin === 'audit-row' ? 'audit' : null;   // the table row keeps the focus
    const target = row ? find(el('main'), `[data-fkey="${row}:${cssEscape(st.nodeId)}"]`)
                       : find(el('main'), `.node[data-node="${cssEscape(st.nodeId)}"]`);
    if (target && typeof target.focus === 'function') target.focus({preventScroll: true});
    if (target && typeof target.scrollIntoView === 'function') target.scrollIntoView({block: 'nearest', inline: 'nearest'});
  }

  // The drawer's width: drag the grip on its left edge (or use the arrow keys on it); --drawer-w, kept in localStorage.
  function setDrawerWidth(px, persist) {
    const vw = num(win.innerWidth) || 1600;
    const w = Math.round(Math.max(DRAWER.min, Math.min(num(px) || DRAWER.min, vw - DRAWER.keep)));
    const root = doc.documentElement;
    if (root && root.style && typeof root.style.setProperty === 'function') root.style.setProperty('--drawer-w', `${w}px`);
    st.drawerW = w;
    const grip = el('drawer-grip');
    if (grip && grip.setAttribute) {
      grip.setAttribute('aria-valuenow', String(w));
      grip.setAttribute('aria-valuemin', String(DRAWER.min));
      grip.setAttribute('aria-valuemax', String(Math.max(DRAWER.min, vw - DRAWER.keep)));
    }
    if (persist) {
      try { win.localStorage.setItem(DRAWER.storageKey, String(w)); } catch (error) { /* storage off: this page only */ }
    }
    return w;
  }

  function drawerWidthNow() {
    const drawer = el('drawer');
    return (drawer && num(drawer.offsetWidth)) || st.drawerW || 600;
  }

  function startResize(ev) {
    const grip = el('drawer-grip');
    if (!grip || (num(ev.button) || 0) > 0) return;
    if (ev.preventDefault) ev.preventDefault();
    const startX = num(ev.clientX) || 0, startW = drawerWidthNow();
    try { if (grip.setPointerCapture && ev.pointerId !== undefined) grip.setPointerCapture(ev.pointerId); } catch (error) { /* ignore */ }
    bodyClass('is-resizing', true);
    const move = (e) => setDrawerWidth(startW + (startX - (num(e.clientX) || 0)), false);
    const end = () => {
      grip.removeEventListener('pointermove', move);
      grip.removeEventListener('pointerup', end);
      grip.removeEventListener('pointercancel', end);
      bodyClass('is-resizing', false);
      setDrawerWidth(st.drawerW || startW, true);
      render();                                            // the tree's card width follows the room left
    };
    grip.addEventListener('pointermove', move);
    grip.addEventListener('pointerup', end);
    grip.addEventListener('pointercancel', end);
  }

  function onGripKey(ev) {
    const step = ev.key === 'ArrowLeft' ? DRAWER.step : ev.key === 'ArrowRight' ? -DRAWER.step : 0;
    if (!step) return;
    ev.preventDefault();
    setDrawerWidth(drawerWidthNow() + step, true);
    render();
  }

  // Task picker: a combobox over the grouped task list.
  function renderMenu() {
    const menu = el('task-menu'), input = el('task-input');
    if (!menu) return;
    menu.innerHTML = renderTaskMenu(st.index, st.menu.query, st.taskId, st.menu.active);
    menu.hidden = !st.menu.open;
    if (input) input.setAttribute('aria-expanded', st.menu.open ? 'true' : 'false');
  }
  function openMenu() { st.menu.open = true; renderMenu(); }
  function closeMenu() {
    st.menu.open = false;
    st.menu.query = '';
    st.menu.active = 0;
    const menu = el('task-menu'), input = el('task-input');
    if (menu) { menu.hidden = true; menu.innerHTML = ''; }
    if (input) {
      input.setAttribute('aria-expanded', 'false');
      if (doc.activeElement !== input) input.value = st.view === 'task' ? text(st.taskId) : '';
    }
  }
  function chooseTask(id) {
    if (!id) return;
    closeMenu();
    const input = el('task-input');
    if (input) { input.value = id; if (input.blur) input.blur(); }
    win.location.hash = taskHash(id);       // -> hashchange -> route()
  }
  function onPickerKey(ev) {
    const n = menuCount(st.index, st.menu.query);
    if (ev.key === 'ArrowDown' || ev.key === 'ArrowUp') {
      ev.preventDefault();
      if (!st.menu.open) openMenu();
      st.menu.active = n ? (st.menu.active + (ev.key === 'ArrowDown' ? 1 : -1) + n) % n : 0;
      renderMenu();
    } else if (ev.key === 'Enter') {
      ev.preventDefault();
      if (ev.stopPropagation) ev.stopPropagation();
      chooseTask(menuOption(st.index, st.menu.query, st.menu.active));
    } else if (ev.key === 'Escape') {
      ev.preventDefault();
      if (ev.stopPropagation) ev.stopPropagation();       // one Escape closes the menu, not also the drawer
      closeMenu();
      const input = el('task-input');
      if (input && input.blur) input.blur();
    }
  }

  /** v1.6: show another run of the task (a batch run's name; '' for its official run): the hash changes, route(). */
  function chooseRun(name) {
    if (st.taskId) win.location.hash = taskHash(st.taskId, null, name || null);
  }

  function lineageKeys() {
    const d = st.details[detailKey()];
    return d ? treeLayout(d, st.ui).lineages.map((l) => `lin:${l.root}`) : [];
  }

  function doAction(action) {
    if (action === 'close-drawer') selectNode(null);
    else if (action === 'toggle-abandoned') { st.ui.showAbandoned = !st.ui.showAbandoned; render(); }
    else if (action === 'expand-lineages' || action === 'collapse-lineages') {
      for (const k of lineageKeys()) st.ui.openState[k] = action === 'expand-lineages';
      st.uiVersion += 1;
      render();
    } else if (action === 'spend-scale') {
      st.ui.spendScale = st.ui.spendScale === 'budget' ? 'spend' : 'budget';
      st.uiVersion += 1;
      render();
    } else if (action === 'refresh') load('all');
  }

  // v1.7, the Explorer. A filter edit re-renders the results and the bar's status slots alone (the bar keeps its focus
  // and caret; Reset rebuilds it), and history.replaceState records every filter in the hash (no history entry per
  // edit: Back leaves the Explorer). A task row opens its task view through its deep link.
  function explorerSet(next, part) {
    st.ex.f = next;
    if (st.view === 'explorer') win.history.replaceState(null, '', explorerHash(next));
    explorerUpdate(part);
  }

  /** The range inputs rejected as typed: {key: why} (exParseBound). */
  function badReasons() {
    return Object.fromEntries(Object.entries(st.ex.bad).map(([k, raw]) => [k, exParseBound(k, raw).bad]).filter(([, why]) => why));
  }

  /** The filter bar's status slots (#ex-st-*): the never-judged count, inverted or rejected ranges, the task choice. */
  function explorerStatus(model, result) {
    const status = explorerFilterStatus(model, st.ex.f, result, badReasons());
    for (const name of ['nj', 's', 'n', 'k', 'targets', 'runs']) {
      const e = el(`ex-st-${name}`);
      if (e && e.innerHTML !== status[name]) e.innerHTML = status[name];
    }
  }

  /** part: 'results' (a filter edit: the results and the status slots), 'targets' (also the DeepSWE task checklist),
      'tlist' (the checklist alone: its search), v1.8 'runs' / 'rlist' (the same for the run checklist), 'data' (a new
      explorer.json, same bar choices: the head too), or 'all' (render(true): the bar as well, the focus kept by
      data-fkey). */
  function explorerUpdate(part) {
    if (st.view !== 'explorer') return;
    const results = el('ex-results'), list = el('ex-tlist'), rlist = el('ex-rlist'), m = explorerModel(st.explorer);
    if (part === 'all' || !results || !m) { st.exFull = true; render(true); return; }
    const active = doc.activeElement && doc.activeElement.dataset ? doc.activeElement.dataset.fkey : null;
    if (list && ['targets', 'tlist', 'data'].includes(part)) list.innerHTML = renderExplorerTargetList(m, st.ex.f, st.ex.tq);
    if (rlist && ['runs', 'rlist', 'data'].includes(part)) rlist.innerHTML = renderExplorerRunList(m, st.ex.f, st.ex.rq);
    if (part !== 'tlist' && part !== 'rlist') {
      hideTip();
      const result = filterExplorer(m, st.ex.f);
      const saved = captureScroll(all(results, '[data-scroll]'));
      results.innerHTML = renderExplorerResults(st.explorer, st.index, {now: clock(), ui: st.ui, filters: st.ex.f, result});
      restoreScroll(all(results, '[data-scroll]'), saved);
      explorerStatus(m, result);
      const head = part === 'data' ? el('ex-head') : null;
      if (head) head.innerHTML = renderExplorerHead(m, st.index, {ui: st.ui, error: st.errors.explorer});
      st.lastKey = viewKey();                                // what is on screen now: a refresh of the same data skips
      renderHeader(clock());
    }
    if (active && !(doc.activeElement && doc.activeElement.dataset && doc.activeElement.dataset.fkey === active)) refocus(active);
  }

  /** input / change of a filter control (data-exf): numbers apply as they are typed, the rest when they change; the
      pages go back to the first. A range input out of its range or not a number is not applied: it stays as typed, marked
      invalid, and the bar says why. Repeated events of one edit (input, then change) change nothing the second time. */
  function onExplorerInput(ev) {
    const t = ev.target, key = t && t.dataset ? t.dataset.exf : null;
    if (!key || st.view !== 'explorer') return;
    if (key === 'tq') {
      if (text(t.value) !== st.ex.tq) { st.ex.tq = text(t.value); explorerUpdate('tlist'); }
      return;
    }
    if (key === 'rq') {                                       // v1.8: the run checklist's search
      if (text(t.value) !== st.ex.rq) { st.ex.rq = text(t.value); explorerUpdate('rlist'); }
      return;
    }
    if (hasOwn(EX_RANGES, key)) {
      const rejected = exParseBound(key, t.value).bad;
      if (rejected) st.ex.bad[key] = text(t.value); else delete st.ex.bad[key];
      if (typeof t.setAttribute === 'function') t.setAttribute('aria-invalid', rejected ? 'true' : 'false');
    }
    const next = Object.assign(updateExplorerFilter(st.ex.f, key, {value: t.value, checked: t.checked}, explorerModel(st.explorer)),
                               {p: 1, tp: 1});
    if (explorerHash(Object.assign({}, next, {p: st.ex.f.p, tp: st.ex.f.tp})) === explorerHash(st.ex.f)) {
      explorerStatus(explorerModel(st.explorer), filterExplorer(explorerModel(st.explorer), st.ex.f));   // its validity
      return;
    }
    explorerSet(next, key === 'run' ? 'runs' : 'results');
  }

  function explorerAction(action) {
    const f = Object.assign({}, st.ex.f);
    if (action === 'reset') {
      st.ex.tq = '';
      st.ex.rq = '';
      st.ex.bad = {};
      explorerSet(Object.assign(explorerDefaults(), {cov: f.cov}), 'all');
    } else if (action === 'targets-shown' || action === 'targets-clear') {   // the checklist's search hits, or all tasks
      const q = st.ex.tq.trim().toLowerCase(), m = explorerModel(st.explorer);
      const shown = arr(m && m.targetIds).filter((id) => !q || id.toLowerCase().includes(q));
      const t = action === 'targets-clear' ? null : [...new Set([...arr(f.t), ...shown])].sort(cmpStr);
      explorerSet(Object.assign(f, {t: t && t.length ? t : null, p: 1, tp: 1}), 'targets');
    } else if (action === 'runs-shown' || action === 'runs-clear') {         // v1.8: the run checklist's hits, or all runs
      const q = st.ex.rq.trim().toLowerCase(), m = explorerModel(st.explorer);
      const shown = arr(m && m.runs).filter((r) => r.run && (!q || [r.run, r.target, r.familyLabel]
        .some((v) => text(v).toLowerCase().includes(q)))).map((r) => r.run);
      const run = action === 'runs-clear' ? null : [...new Set([...arr(f.run), ...shown])].sort(cmpStr);
      explorerSet(Object.assign(f, {run: run && run.length ? run : null, p: 1, tp: 1}), 'runs');
    } else if (action === 'cov:tasks' || action === 'cov:traj') {
      explorerSet(Object.assign(f, {cov: action === 'cov:traj' ? 'traj' : 'tasks'}), 'results');
    } else if (action === 'skip-tables') {                    // past the charts' tab stops, to the first task row
      const root = el('ex-results') || el('main');
      const target = find(root, 'button.node-link[data-exlink]') || find(root, 'button.ex-sort');
      if (target && typeof target.focus === 'function') target.focus({preventScroll: true});
      if (target && typeof target.scrollIntoView === 'function') target.scrollIntoView({block: 'start'});
    }
  }

  /** Arrow keys (and Home / End) on a chart mark move the panel's one tab stop to the next mark (roving tabindex). */
  const ROVING_KEYS = {ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, Home: -Infinity, End: Infinity};
  function onRovingKey(ev) {
    const t = ev.target;
    if (st.view !== 'explorer' || !t || !t.dataset || t.dataset.mark === undefined || !hasOwn(ROVING_KEYS, ev.key)) return false;
    const marks = all(closest(t, '[data-roving]'), '[data-mark]');
    const j = exRovingIndex(marks.indexOf(t), marks.length, ROVING_KEYS[ev.key]);
    if (ev.preventDefault) ev.preventDefault();
    if (j < 0) return true;
    for (const m of marks) if (typeof m.setAttribute === 'function') m.setAttribute('tabindex', m === marks[j] ? '0' : '-1');
    if (typeof marks[j].focus === 'function') marks[j].focus();
    return true;
  }

  /** A mark that takes the focus (by a click, or the focus kept across a re-render) becomes its panel's one tab stop. */
  function onRovingFocus(ev) {
    const t = ev.target;
    if (st.view !== 'explorer' || !t || !t.dataset || t.dataset.mark === undefined) return;
    const panel = closest(t, '[data-roving]');
    if (!panel) return;
    for (const m of all(panel, '[data-mark]')) if (typeof m.setAttribute === 'function') m.setAttribute('tabindex', m === t ? '0' : '-1');
  }

  /** A sort button: another column sorts by it (numbers largest first), the same one flips the direction. */
  function explorerSort(key) {
    if (!EX_SORTS.includes(key)) return;
    const f = st.ex.f, same = (f.sort || 'task') === key;
    explorerSet(Object.assign({}, f, {sort: key === 'task' && !same ? null : key, desc: same ? !f.desc : EX_SORT_DESC.has(key), p: 1}), 'results');
  }

  /** A pager button: data-expage="p:<n>" (tasks) or "tp:<n>" (trajectories). */
  function explorerPage(spec) {
    const [key, n] = text(spec).split(':');
    const page = int(Number(n));
    if ((key === 'p' || key === 'tp') && page !== null && page >= 1) explorerSet(Object.assign({}, st.ex.f, {[key]: page}), 'results');
  }

  /** CSV / JSON of the filtered rows, made here and saved by the browser (a Blob behind a download link). */
  function explorerDownload(what) {
    const file = explorerExport(st.explorer, st.ex.f, what);
    const B = win.Blob, U = win.URL;
    if (!file || typeof B !== 'function' || !U || typeof U.createObjectURL !== 'function' || typeof doc.createElement !== 'function') return false;
    const url = U.createObjectURL(new B([file.text], {type: file.type}));
    const a = doc.createElement('a');
    a.href = url;
    a.download = file.name;
    a.rel = 'noopener';
    if (doc.body && typeof doc.body.appendChild === 'function') doc.body.appendChild(a);
    if (typeof a.click === 'function') a.click();
    if (typeof a.remove === 'function') a.remove();
    win.setTimeout(() => { try { U.revokeObjectURL(url); } catch (error) { /* already gone */ } }, 60 * 1000);
    return true;
  }

  // The Explorer's tooltip: one element (#ex-tip), filled from the hovered or focused mark's data-tip, kept in the window.
  let tipTarget = null;
  function hideTip() {
    const tip = el('ex-tip');
    tipTarget = null;
    if (tip && !tip.hidden) tip.hidden = true;
  }
  function placeTip(ev) {
    const tip = el('ex-tip');
    if (!tip || !tipTarget || !tip.style) return;
    const pointer = text(ev && ev.type).startsWith('pointer') && num(ev.clientX) !== null;
    const r = typeof tipTarget.getBoundingClientRect === 'function' ? tipTarget.getBoundingClientRect() : null;
    const x = pointer ? ev.clientX : r ? r.left + Math.min(num(r.width) || 0, 360) : 0;
    const y = pointer ? ev.clientY : r ? r.top : 0;
    const w = num(tip.offsetWidth) || 240, h = num(tip.offsetHeight) || 64;
    const vw = num(win.innerWidth) || 1600, vh = num(win.innerHeight) || 900;
    tip.style.left = `${Math.round(Math.max(8, Math.min(vw - w - 8, x + 14)))}px`;
    tip.style.top = `${Math.round(Math.max(8, Math.min(vh - h - 8, y - h - 12 < 8 ? y + 20 : y - h - 12)))}px`;
  }
  function showTip(ev) {
    const target = st.view === 'explorer' ? closest(ev.target, '[data-tip]') : null;
    const tip = el('ex-tip');
    if (!tip) return;
    if (!target) { if (ev.type === 'focusin') hideTip(); return; }
    if (target !== tipTarget) {
      tipTarget = target;
      tip.innerHTML = renderExplorerTip(target.dataset.tip);
      tip.hidden = false;
    }
    placeTip(ev);
  }
  function leaveTip(ev) {
    if (!tipTarget) return;
    const to = ev.relatedTarget ? closest(ev.relatedTarget, '[data-tip]') : null;
    if (to !== tipTarget) hideTip();
  }

  function closest(target, sel) { return target && typeof target.closest === 'function' ? target.closest(sel) : null; }

  function onClick(ev) {
    const t = ev.target;
    const opt = closest(t, '[data-task]');
    if (opt) { ev.preventDefault(); chooseTask(opt.dataset.task); return; }
    const act = closest(t, '[data-action]');
    if (act) { ev.preventDefault(); doAction(act.dataset.action); return; }
    const pick = closest(t, '[data-run-select]');
    if (pick) { ev.preventDefault(); chooseRun(pick.dataset.runSelect); return; }
    // v1.7, the Explorer's controls (attributes no other view writes)
    const exAction = closest(t, '[data-exaction]');
    if (exAction) { ev.preventDefault(); explorerAction(exAction.dataset.exaction); return; }
    const exSort = closest(t, '[data-exsort]');
    if (exSort) { ev.preventDefault(); explorerSort(exSort.dataset.exsort); return; }
    const exPager = closest(t, '[data-expage]');
    if (exPager) { ev.preventDefault(); explorerPage(exPager.dataset.expage); return; }
    const exFile = closest(t, '[data-exexport]');
    if (exFile) { ev.preventDefault(); explorerDownload(exFile.dataset.exexport); return; }
    const exLink = closest(t, '[data-exlink]');
    if (exLink) {
      const link = text(exLink.dataset.exlink);           // a task row (or its button): open the node in its task view
      if (link.startsWith('#task=') && !(exLink.tagName !== 'BUTTON' && selecting())) {
        ev.preventDefault();
        win.location.hash = link;
      }
      return;
    }
    const node = closest(t, '[data-node]');
    if (node && !closest(t, 'summary')) {
      const origin = node.classList && node.classList.contains('node') ? 'card' : closest(node, '.r0-row') ? 'row'
        : closest(node, '.fa-row') ? 'audit-row' : 'link';
      selectNode(node.dataset.node, origin);
      return;
    }
    if (st.menu.open && !closest(t, '.picker')) closeMenu();
  }

  function onKey(ev) {
    const t = ev.target;
    if (onRovingKey(ev)) return;                             // v1.7: a chart mark of the Explorer
    if (ev.key === 'Escape') {
      hideTip();                                            // v1.7: the Explorer's tooltip is dismissible
      if (st.menu.open) closeMenu();
      else if (st.view === 'task' && st.nodeId) selectNode(null);   // v1.7: never another view's hash
      return;
    }
    if ((ev.key === 'Enter' || ev.key === ' ') && t && t.getAttribute && t.getAttribute('role') === 'button' && t.dataset && t.dataset.node) {
      ev.preventDefault();
      selectNode(t.dataset.node, 'card');
      return;
    }
    const typing = t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable);
    if (ev.key === '/' && !typing) {
      const input = el('task-input');
      if (input && input.focus) { ev.preventDefault(); input.focus(); }
    }
  }

  /** Drop ?token= from the address bar (the server has set the session cookie by now). */
  function stripToken() {
    const loc = win.location;
    if (!/[?&]token=/.test(text(loc.search))) return;
    const params = new URLSearchParams(loc.search);
    params.delete('token');
    const q = params.toString();
    win.history.replaceState(null, '', `${loc.pathname}${q ? `?${q}` : ''}${loc.hash}`);
  }

  function start() {
    stripToken();
    win.addEventListener('hashchange', () => route(true));
    doc.addEventListener('click', onClick);
    doc.addEventListener('keydown', onKey);
    // <details> toggles do not bubble: capture them to remember open/closed state across re-renders.
    doc.addEventListener('toggle', (ev) => {
      const t = ev.target;
      if (t && t.dataset && t.dataset.key) st.ui.openState[t.dataset.key] = !!t.open;
    }, true);
    doc.addEventListener('visibilitychange', () => { if (doc.visibilityState === 'visible' && !fresh('index')) load(true, true); });
    // v1.7, the Explorer: its filter controls (delegated), and the tooltip of its chart marks on hover and focus
    doc.addEventListener('input', onExplorerInput);
    doc.addEventListener('change', onExplorerInput);
    doc.addEventListener('pointerover', showTip);
    doc.addEventListener('pointermove', (ev) => { if (tipTarget) placeTip(ev); });
    doc.addEventListener('pointerout', leaveTip);
    doc.addEventListener('focusin', showTip);
    doc.addEventListener('focusin', onRovingFocus);
    doc.addEventListener('focusout', leaveTip);
    let resizing = null;                                    // card width follows the window: re-render, debounced
    win.addEventListener('resize', () => {
      if (resizing) win.clearTimeout(resizing);
      resizing = win.setTimeout(() => { resizing = null; if (st.view === 'task') render(); }, 200);
    });
    const input = el('task-input'), menu = el('task-menu'), refresh = el('refresh'), grip = el('drawer-grip');
    if (input) {
      input.addEventListener('focus', () => { st.menu.query = ''; st.menu.active = 0; openMenu(); if (input.select) input.select(); });
      input.addEventListener('input', () => { st.menu.query = input.value; st.menu.active = 0; openMenu(); });
      input.addEventListener('keydown', onPickerKey);
      input.addEventListener('blur', () => win.setTimeout(() => { if (doc.activeElement !== input) closeMenu(); }, 150));
    }
    if (menu) menu.addEventListener('mousedown', (ev) => ev.preventDefault());   // keep focus in the input
    if (refresh) refresh.addEventListener('click', () => load('all'));
    if (grip) {
      grip.addEventListener('pointerdown', startResize);
      grip.addEventListener('keydown', onGripKey);
    }
    let kept = null;
    try { kept = num(Number(win.localStorage.getItem(DRAWER.storageKey))); } catch (error) { kept = null; }
    if (kept) setDrawerWidth(kept, false);
    st.timers.push(win.setInterval(() => load(true, true), REFRESH_MS));
    // Every 15 s: the header's age, and a refresh held back by a text selection once the selection is gone.
    st.timers.push(win.setInterval(() => { if (st.held && !selecting()) render(false, true); else renderHeader(clock()); }, 15 * 1000));
    return route(false);
  }

  function stop() { for (const t of st.timers) win.clearInterval(t); st.timers = []; }

  return {start, stop, route, load, render, selectNode, chooseTask, chooseRun, setDrawerWidth, state: st,
          idle: () => (st.loading ? st.loading.then(() => (st.loading ? st.loading : null)) : Promise.resolve())};
}

function boot() {
  return createApp({window, document, fetch: window.fetch.bind(window)}).start();
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    // helpers
    esc, text, truncate, fmtInt, fmtCompact, fmtPct, fmtMoney, fmtDur, fmtAgo, fmtClock, fmtDate, fmtBytes, niceTicks, timeTicks,
    parseHash, taskHash, decisionOf, probeStats, progressOf, liveProbe, partialText, focusLabel, treeContext, focusTally,
    treeLayout, taskGroups, defaultTask, runOf, findTask, menuCount, menuOption, busyWorkers, routerScope, workerStatus,
    failureMatch, faVerdict, auditCounts, auditBreakdown, patternLabel, inAuditScope, eligibleCounts, withdrawnParts,
    outOfScopeWhy, seedNote, seedBadge, seedSource,
    isBatchRun, notAudited, batchRunsOf, batchInfo, batchName, batchBadge, viewsOf, selectedView, viewDetail,
    DECISIONS, FA_VERDICTS, SEED_NOTE, TREE, DRAWER, STALE_S, BATCH_HELP,
    // refresh helpers (pure)
    refreshAction, hasTextSelection, captureScroll, restoreScroll, captureOpen,
    // renderers
    renderStatus, renderBanner, renderAuthError, renderTaskMenu, renderAccuracy, renderNoRun, renderRunBar,
    renderFailureAnalysis, renderPlan, renderTree, renderRoundZero, renderAuditTable, renderNodeCard, renderNodeDetail,
    renderFailureMatch, renderRollouts, renderAudit, renderTaskView, renderLineChart, renderFleetSummary, renderWorkers,
    renderFleetCharts, renderApi, renderMonitor, renderRunSelector, renderBatchRunBar, renderBatchTable, renderBatchView,
    // v1.7, the Explorer: model, filters (hash round trip), statistics, renderers, exports
    explorerDefaults, explorerModel, parseExplorerParams, explorerHash, updateExplorerFilter, exToggle, filterExplorer,
    explorerStats, exSortedTasks, exPage, exParseBound, explorerFilterStatus, exInverted, exRateColumn, exPct, exRovingIndex,
    exBarKey, explorerScope, exRescoredTitle, renderExplorer, renderExplorerHead, renderExplorerFilters, renderExplorerResults,
    renderExplorerTargetList, renderExplorerTaskTable, renderExplorerTrajTable, renderExplorerTip, renderHBars, renderVBars,
    explorerExport, csvCell, csvText, EX_AUDITS, EX_PAGE, EX_SORTS, EX_MODELS, EX_RESCORED_HELP,
    // v1.8, every official run and the synthesis queue
    officialRunsOf, queuedIn, renderOlderRunNote, renderQueue, exFamily, renderExplorerRunList, QUEUE_TONES,
    // wiring (tests drive it with a fake DOM)
    createApp,
  };
} else if (typeof window !== 'undefined') {
  window.addEventListener('DOMContentLoaded', boot);
}
