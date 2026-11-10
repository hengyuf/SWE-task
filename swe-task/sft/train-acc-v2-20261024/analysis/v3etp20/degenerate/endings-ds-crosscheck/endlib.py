"""Shared detectors of the session-ending lens (used identically on rollouts and on the SFT corpora).

Call category of one tool call (cat_call): the tool decides for non-bash tools (edit / write / multiedit / patch ->
'edit'; read / grep / glob / list -> 'read'; todowrite / todoread -> 'todo'; task -> 'subagent'; webfetch -> 'web');
a bash command is matched, after removing heredoc bodies and quoted strings, against these patterns in priority order:
  kill    kill / pkill / killall
  commit  git ... commit
  test    a test runner (pytest, go test, cargo test/nextest, npm/yarn/pnpm/bun test, jest / vitest / mocha as a command,
          node --test, deno/bun test, make test/check, ctest, rspec, phpunit, tox / nox, unittest, test262, run-tests)
  run     an ad-hoc program run (python x.py / -c, node x.js / -e, go run, cargo run, tsx / ts-node, deno run, ./bin)
  build   build / lint / typecheck / format / install (go build/vet, cargo build/check/clippy/fmt, tsc, npm run build|lint,
          make, gofmt, ruff, black, mypy, eslint, prettier, pip install, ...)
  edit    a file write from the shell (sed -i, perl -pi, cat > / heredoc > file, tee file, git apply, patch -p)
  git     any other git command
  read    an inspection command (cat, head, tail, grep, rg, find, ls, wc, sed -n, awk, nl, tree, diff, ...)
  sleep   a bare sleep
  other   anything else
"""
import re

RE_HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1[^\n]*\n.*?\n\s*\2\s*(?:\n|$)", re.S)
RE_QUOTED = re.compile(r"'[^'\n]*'|\"(?:[^\"\\\n]|\\.)*\"")
RE_KILL = re.compile(r"(\bkill\b|\bpkill\b|\bkillall\b)")
RE_KILL_BROAD = re.compile(r"(\bpkill\s+(-\d+\s+|-[A-Z]+\s+)?-f\s+['\"]?(node|python|go|cargo|npm|bash|sh|opencode|bun|deno|ruby|java)\b"
                           r"|\bkillall\s+(-\d+\s+)?(node|python3?|go|bash|sh|opencode)\b|\bxargs\s+(-r\s+)?kill\b"
                           r"|\bkill\s+(-9\s+|-KILL\s+|-QUIT\s+|-TERM\s+)?(-1|0|\$\$|\$PPID)\b|\bkill\s+%)")
RE_COMMIT = re.compile(r"\bgit\b[^;&|\n]*\bcommit\b")
RE_TEST = re.compile(
    r"(\bpytest\b|\bpy\.test\b|-m\s+(pytest|unittest)\b|\btox\b|\bnox\b|\bgo\s+test\b|\bgotestsum\b|\bcargo\s+(test|nextest)\b"
    r"|\b(npm|pnpm|yarn|bun)\s+(run\s+)?(test|cover|coverage)(:\S+)?\b|(^|[\s;&|(/])(jest|vitest|mocha|ava)(\.js|\.mjs)?(?=\s|$)"
    r"|\bnode\s+(--\S+\s+)*--test\b|\bdeno\s+test\b|\bbun\s+test\b|\bmake\s+(test|check|tests)\b|\bctest\b|\brspec\b"
    r"|\bphpunit\b|\bmix\s+test\b|\bdotnet\s+test\b|\bmvn\b[^;&|\n]*\btest\b|\bgradlew?\s+test\b|\btest262\b|\brun[-_]tests?\b"
    r"|\bunittest\b|\bpnpm\s+-\S+\s+\S+\s+test\b|\bnpx\s+(jest|vitest|mocha|playwright\s+test)\b|\byarn\s+(jest|vitest)\b)")
RE_RUN = re.compile(
    r"(\bpython[0-9.]*\s+(-[a-zA-Z]+\s+)*(?!-m\b)[\w./-]+\.py\b|\bpython[0-9.]*\s+-[a-zA-Z]*c\b|\bpython[0-9.]*\s+-\s*<<|\bpython[0-9.]*\s*<<"
    r"|\bnode\s+(-[\w-]+(=\S+)?\s+)*[\w./@-]+\.(m?js|cjs|ts|mts)\b|\bnode\s+(-[\w-]+\s+)*-e\b|\bnode\s*<<|\bnode\s+-\s*<<|\bgo\s+run\b|\bcargo\s+run\b"
    r"|\bnpx\s+(tsx|ts-node)\b|(^|[\s;&|(])tsx\s+\S|\bts-node\b|\bdeno\s+run\b|\bbun\s+(run\s+)?[\w./-]+\.(ts|js|mjs)\b|(^|[;&|]\s*)\./[\w./-]+"
    r"|\bruby\s+[\w./-]+\.rb\b|\bphp\s+[\w./-]+\.php\b|\bcargo\s+script\b|\bpython[0-9.]*\s+(-\s+)?HEREDOC|\bnode\s+(-\s+)?HEREDOC"
    r"|\bpython[0-9.]*\s+-m\s+(?!pytest\b|unittest\b|pip\b)\w+|\bdeno\s+eval\b|(^|[;&|]\s*)(\w+=\S*\s+)*/tmp/[\w./-]+)")
RE_BUILD = re.compile(
    r"(\bgo\s+(build|vet|mod|generate|install|fmt)\b|\bcargo\s+(build|check|clippy|fmt|doc|update|fetch)\b|(^|[\s;&|(])tsc\b"
    r"|\b(npm|pnpm|yarn|bun)\s+(run\s+)?(build|lint|typecheck|type-check|tsc|format|fmt|compile|install|ci|check)\b|\bnpm\s+i\b"
    r"|(^|[\s;&|(])make\b|\bgofmt\b|\bgofumpt\b|\bgoimports\b|\bstaticcheck\b|\bgolangci-lint\b|\bruff\b|\bblack\b|\bflake8\b"
    r"|\bmypy\b|\bpyright\b|\beslint\b|\bprettier\b|\bbiome\b|\bpip3?\s+install\b|\buv\s+(pip|sync)\b|\bpoetry\s+(install|build)\b"
    r"|\brustfmt\b|\bcmake\b|\bninja\b|\bgcc\b|\bclang\b|\bwasm-pack\b|\bnpx\s+(tsc|eslint|prettier|biome|xo|oxlint)\b|/tsc\b"
    r"|\b(pnpm|npm|yarn)\s+(-\S+\s+)*(run\s+)?\w*(lint|check|typecheck|build|format|prettier|eslint|tsc)\w*\b|\brustc\b)")
RE_EDIT_SH = re.compile(
    r"(\bsed\s+(-[a-zA-Z]*i\b|--in-place)|\bperl\s+-[a-zA-Z]*i|\bcat\s*>|<<-?\s*['\"]?\w+['\"]?\s*>\s*\S|>\s*[\w./-]+\.(py|go|rs|ts|tsx|js|jsx|mjs|cjs"
    r"|json|md|toml|yaml|yml|c|h|cc|cpp|java|rb|sh|abs|txt|snap|lock|mod|sum|html|css)\b|\btee\s+(-a\s+)?(?!/dev/)\S|\bgit\s+apply\b"
    r"|\bpatch\s+-p|\bapplypatch\b|\bmv\s+\S+\s+\S+|\bcp\s+\S+\s+\S+|\brm\s+(-[rf]+\s+)?\S)")
RE_GIT = re.compile(r"\bgit\s+")
RE_READ_SH = re.compile(
    r"^\s*(cd\s+\S+\s*(&&|;)\s*)?(cat|head|tail|less|more|grep|egrep|fgrep|rg|ag|find|fd|ls|wc|sed\s+-n|awk|nl|tree|diff|stat|file"
    r"|which|echo|pwd|du|cmp|od|xxd|hexdump|strings|readlink|realpath|type|env|printenv|uname|date|whoami|id|ps|pgrep|lsof|df|free"
    r"|nproc|jq|sort|uniq|cut|tr|column|basename|dirname|true|test\s|\[)\b")
RE_SLEEP_ONLY = re.compile(r"^\s*(cd\s+\S+\s*&&\s*)?sleep\s+\d+(\.\d+)?\s*(;|&&)?\s*$")
RE_SLEEP = re.compile(r"\bsleep\s+\d")
RE_BG = re.compile(r"(?:^|[^&|>])&\s*(?:$|;|\n|disown|sleep|echo|\()|\bnohup\b|\bsetsid\b|\bdisown\b")

TOOL_CAT = {"edit": "edit", "write": "edit", "multiedit": "edit", "patch": "edit", "apply_patch": "edit",
            "read": "read", "grep": "read", "glob": "read", "list": "read",
            "todowrite": "todo", "todoread": "todo", "task": "subagent", "webfetch": "web"}
CATS = ("kill", "commit", "test", "run", "build", "edit", "git", "read", "sleep", "todo", "subagent", "web", "other")


def strip_cmd(cmd):
    s = RE_HEREDOC.sub(" HEREDOC\n", cmd or "")
    return RE_QUOTED.sub("''", s)


def cat_bash(cmd):
    raw = cmd or ""
    s = strip_cmd(raw)
    if RE_KILL.search(s):
        return "kill"
    if RE_COMMIT.search(raw):
        return "commit"
    if RE_TEST.search(s):
        return "test"
    if RE_RUN.search(s):
        return "run"
    if RE_BUILD.search(s):
        return "build"
    if RE_EDIT_SH.search(s):
        return "edit"
    if RE_GIT.search(s):
        return "git"
    if RE_SLEEP_ONLY.search(s):
        return "sleep"
    if RE_READ_SH.search(s):
        return "read"
    return "other"


def cat_call(tool, cmd=None):
    if tool == "bash":
        return cat_bash(cmd)
    return TOOL_CAT.get(tool, "other")


def norm_cmd(cmd):
    return re.sub(r"\s+", " ", (cmd or "")).strip()


# final-text detectors (visible text of the last turn)
RE_CLAIM_DONE = re.compile(r"(\b(done|complete[d]?|implemented|finished|all (?:\w+ )?(?:tests?|checks?|specs?) (?:pass|passed|passing|green)"
                           r"|summary of (?:the )?changes|everything (?:is|was) (?:done|committed|in place|implemented)|work is (?:done|complete)"
                           r"|task is (?:done|complete)|ready for review)\b|^#+\s*summary)", re.I | re.M)
RE_MENTIONS_COMMIT = re.compile(r"\bcommit(?:ted|s)?\b", re.I)
RE_CLAIM_COMMITTED = re.compile(r"(\bcommitted\b(?! yet)|\bcommit(?:ted)? (?:as|on|to) `?[0-9a-f]{6,}|\bcommit [0-9a-f]{7,}\b|`[0-9a-f]{7,12}`)", re.I)
RE_DEFERS_COMMIT = re.compile(  # analyze.py RE_DEFERS_COMMIT (unchanged)
    r"(want me to (go ahead and )?commit|shall i commit|should i commit|like me to commit|me to commit"
    r"|ha(ve|s)n['’]?t (been )?committed|have not committed|not (yet )?committed|nothing (is |was )?committed"
    r"|no commit(s)? (yet|made|was made)|uncommitted|won['’]?t commit|will not commit|wouldn['’]?t commit"
    r"|without (being asked|your|an explicit|explicit)|didn['’]?t commit|did not commit"
    r"|not commit(ting)? (anything|unless|until)|ready to commit|hold(ing)? (off )?on committing|say the word"
    r"|go ahead and commit|commit (whenever|when you|if you|once you)|only commit(ting)? when|commit only on"
    r"|no-commit|unrequested commits|before committing|explicitly ask)", re.I)
RE_ASKS_USER = re.compile(r"\?\s*$|\b(would you like|do you want|let me know|should i|shall i|please confirm|which (one|option))\b", re.I)
# an announced next action at the very end of the text (the turn then ends without a tool call)
RE_INTENT_END = re.compile(r"(\blet me\b|\bi'll\b|\bi will\b|\blet's\b|\bnow (?:i|we)\b|\bnext,? (?:i|we)\b|\bi need to\b|\bi should\b"
                           r"|\bi'm going to\b|\bgoing to\b)[^\n]{0,240}[:.…]?\s*$", re.I)
RE_STATUS_SUMMARY = re.compile(r"(what (?:did )?we (?:did|do|have done) so far|so far,? (?:we|i) (?:have|'ve)|## (?:progress|status|current state)"
                               r"|\bnext steps?\b|\bremaining (?:work|steps|tasks)\b|\bstill (?:need|to do)\b|\bTODO\b)", re.I)

# the task's commit instruction carried into a compaction summary
RE_COMMIT_INSTR = re.compile(
    r"(commit (?:everything|all(?: the)? (?:changes|work)|the (?:changes|work)|my (?:changes|work)|our (?:changes|work))"
    r"|commit\b[^.\n]{0,40}\b(?:when|once|after)\b[^.\n]{0,30}\b(?:done|finished|complete)"
    r"|(?:must|should|need to|needs to|required to|then|and|,)\s+commit\b"
    r"|commit(?:ting)? (?:it |them |everything |all )?at (?:the )?end"
    r"|commit (?:it |them |everything |all )?(?:on|to|in) (?:the |a |new |its own |feature )?(?:branch\b|`)"
    r"|new branch from main and commit|\bcommit\s*[.;]?\s*(?:\n|$)|^\s*[-*\d.)]+\s*commit\b)", re.I | re.M)
# the model cites the system prompt's "NEVER commit changes unless the user explicitly asks you to" (or AGENTS.md / policy)
RE_NOCOMMIT_RULE = re.compile(
    r"((?:never|not|n't|must not|mustn't|should not|shouldn't|won't|will not|do not|don't)\s+(?:\w+\s+){0,2}commit(?:ting)?\b[^.\n]{0,60}"
    r"\b(?:unless|until|without)\b[^.\n]{0,40}\b(?:ask|request|told|instruct)"
    r"|commit\w*\b[^.\n]{0,60}\bunless\b[^.\n]{0,40}\b(?:ask|request|told|instruct)"
    r"|(?:not|n't|never)\s+(?:been\s+|being\s+)?(?:explicitly\s+)?asked\s+(?:me\s+|us\s+)?to\s+commit"
    r"|(?:didn['’]?t|did not|hasn['’]?t|has not|haven['’]?t|have not)\s+(?:explicitly\s+)?ask(?:ed)?\s+(?:me\s+|us\s+)?(?:to\s+|for\s+(?:a\s+)?)commit"
    r"|(?:only|should only|must only)\s+commit\s+when\s+(?:explicitly\s+)?(?:asked|requested)"
    r"|no[- ]commit(?:-without-ask)?\s+(?:rule|policy)|commit\w*\b[^.\n]{0,40}\b(?:per|by|under)\s+(?:my |the |standing )?(?:instructions?|policy|rules?|AGENTS\.md|system prompt)"
    r"|(?:instructions?|policy|rules?|AGENTS\.md|system prompt)\b[^.\n]{0,60}\b(?:never|not|n't)\s+(?:\w+\s+){0,2}commit)", re.I)
# the model notes that the user / task asked for the commit (the permission the no-commit rule needs)
RE_COMMIT_OVERRIDE = re.compile(
    r"((?:user|task|prompt|instructions?|request)\b[^.\n]{0,40}\b(?:explicitly\s+)?(?:asked|asks|requested|requests|instructs?|instructed|says|said|wants|wanted)"
    r"\b[^.\n]{0,40}\bcommit|(?:asked|told|instructed|requested)\s+(?:me\s+|us\s+)?to\s+commit|commit everything when (?:you are|you're|I'm|I am|we are) done)", re.I)
RE_WDWDSF = re.compile(r"what did we do so far", re.I)
RE_CONTINUE_PROMPT = re.compile(r"continue if you have next steps|stop and ask for clarification", re.I)


def final_flags(visible, reasoning):
    v = visible or ""
    r = reasoning or ""
    both = v + "\n" + r
    vs = v.rstrip()
    return {
        "claims_done": bool(RE_CLAIM_DONE.search(v)), "mentions_commit": bool(RE_MENTIONS_COMMIT.search(v)),
        "claims_committed": bool(RE_CLAIM_COMMITTED.search(v)), "defers_commit": bool(RE_DEFERS_COMMIT.search(v)),
        "asks_user": bool(RE_ASKS_USER.search(v[-300:])),
        "intent_end": bool(vs) and (vs.endswith(":") or bool(RE_INTENT_END.search(vs[-300:]))),
        "intent_end_r": (not vs) and bool(RE_INTENT_END.search(r.rstrip()[-300:])),
        "status_summary": bool(RE_STATUS_SUMMARY.search(v)), "nocommit_rule": bool(RE_NOCOMMIT_RULE.search(both)),
        "commit_override": bool(RE_COMMIT_OVERRIDE.search(both)),
        "wdwdsf": bool(RE_WDWDSF.search(both)), "continue_prompt": bool(RE_CONTINUE_PROMPT.search(both)),
        "anchored_summary": "## Goal" in v or "# Anchored Summary" in v,
    }

# a final message that says it is breaking out of its own repetition loop
RE_LOOP_BREAK = re.compile(r"(filler loop|break (?:this|the|out of (?:this|the)) loop|stop (?:this|the) (?:loop|repetition)|STOP this"
                           r"|going in circles|I(?:'m| am) looping|stuck in a loop|stop theorizing|hmm(?:\s*[—–-]\s*hmm){3,})", re.I)
