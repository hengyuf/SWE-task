r"""Text-level degeneration detectors shared by the model scan (scan_models.py) and the SFT-corpus scan (scan_corpus.py).

Every detector runs on one piece of text at a time: the reasoning R of a call / trained assistant turn (text before
</think>) or its visible text V (text after </think>; for corpus turns the <tool_call> blocks and <|im_end|> are
removed first). Definitions (identical for models and corpus):

  zlib ratio       len(zlib.compress(utf8 bytes, 6)) / len(utf8 bytes)
  tail ratio       zlib ratio of the last 20,000 chars (thinkx.py / notool_kinds.py rule)
  window ratio     zlib ratio of 2,000-char windows, stride 1,000 (last window aligned to the end); texts of 500-2,000
                   chars get one window
  loop region      maximal run of consecutive windows with ratio < LOOP_T (0.10) -> [start, end) in chars; a call has a
                   "repetition loop" if it has >= 1 loop region. Region unit: the dominant token k-gram (k = 1..8,
                   tokens = \w+ | single non-space symbol) of the region, the smallest k covering >= 60% of the
                   region's tokens (else the best-covering k), canonicalised to its minimal rotation; class by
                   classify_unit(). terminal = region reaches the end of the text (within 1 stride); escaped = >= 2,000
                   chars of text follow the region.
  char run         maximal run of one non-space character repeated >= 20 times ('✓✓✓…'); spaced run = one symbol
                   repeated >= 20 times separated by 1-2 spaces/tabs ('— — —', '← ← ←')
  control token    special tokens / other chat formats' control tokens written as text: <|...|> (ASCII pipes, no
                   spaces), <｜...｜> (fullwidth, DeepSeek), [INST], [/INST], [AVAILABLE_TOOLS], [TOOL_CALLS],
                   [TOOL_RESULTS], <start_of_turn>, <end_of_turn>, <bos>/<eos>/<pad>/<unk>/<mask>, <SPECIAL_n>,
                   <extra_id_n>, byte-fallback <0xNN>, [gMASK], <sop>, <eop>, <s>, </s>
  think tag        '<think>' / '</think>' inside R (nested) or inside V (a second think block)
  tool markup      <tool_call>, </tool_call>, <function=, </function>, <parameter=, </parameter>, <tool_response> (the
                   classifier's RE_TOOLCALL plus closers; reported only for completeness -- the format lens owns it)
  foreign script   maximal runs of letters of a non-Latin script (Greek, Cyrillic, Armenian, Hebrew, Arabic, Indic,
                   Thai, CJK, Kana, Hangul, ...). Script = first word of the Unicode name of the run's first char.
                   'CJK punct' = U+3000-303F / fullwidth U+FF01-FF60 characters.
  garbage char     U+FFFD, private use, C0/C1 controls (not \t \n \r), zero-width / format chars (U+200B-200F,
                   U+2060-2064, U+FEFF), bidi controls, line/paragraph separators, unassigned code points (Cn),
                   surrogates; mojibake = 'Ã'/'Â'/'â€' followed by a Latin-1 continuation
  copied           a control token / foreign run / garbage char counts as copied if the same string occurs in the
                   context the model had seen before the call (user prompts + tool outputs; corpus: earlier
                   system/user/tool messages of the conversation); otherwise "spontaneous"
  interjection     (?i) hm+ | wait | hold on | actually | oh+ | ah+ | ugh+ | whoa | oops | argh+ | huh | whatever |
                   hmph as a word; per-text counts by marker, and the maximum number inside any 1,000-char window
                   (with that window's zlib ratio)
  shouting         case-sensitive WAIT, AH+, OH+, AHA+, UGH+, WHOA, OOPS, ARGH+, STOP, NOO+ as words; '!!' runs
  cycling          the largest number of times one sentence segment (>= 40 chars of [^\n.!?], normalised) recurs in a
                   text (not necessarily consecutively)
  floods           max number of em dashes / '?' / all-caps words ([A-Z][A-Z']{2,}) in any 1,000-char window, computed
                   outside loop regions (with the window's zlib ratio when >= 25)
  bang separator   longest run of >= 8 consecutive tokens each ending in '!' ('the! AST! positions!' or 'what!if!the!')
  stutter          immediate repeats of a word of >= 3 letters ('row row', 'delimiter delimiter'; interjections and
                   'the' excluded): count and longest repeat
  no-space run     longest run of >= 150 non-space characters
"""
import hashlib
import re
import unicodedata
import zlib
from collections import Counter

LOOP_T = 0.10
W, S = 2000, 1000
TAIL = 20000

TOK = re.compile(r"\w+|[^\w\s]")
RE_CHARRUN = re.compile(r"(\S)\1{19,}")
RE_SPACEDRUN = re.compile(r"([^\w\s])(?:[ \t]{1,2}\1){19,}")
RE_CTRL = re.compile(
    r"<\|[^\s|<>]{1,60}\|>|<｜[^｜\n]{1,60}｜>|\[/?INST\]|\[/?(?:AVAILABLE_TOOLS|TOOL_RESULTS|TOOL_CALLS)\]"
    r"|<(?:start|end)_of_turn>|<(?:bos|eos|pad|unk|mask)>|<SPECIAL_\d+>|<extra_id_\d+>|<0x[0-9A-Fa-f]{2}>"
    r"|\[gMASK\]|<sop>|<eop>|</s>|<s>")
RE_THINKTAG = re.compile(r"</?think>")
RE_TOOLMARK = re.compile(r"</?tool_call>|<function=|</function>|<parameter=|</parameter>|</?tool_response>")
RE_FOREIGN = re.compile(
    "[Ͱ-ϿЀ-ԯ԰-֏֐-׿؀-ۿ܀-ࣿऀ-෿"
    "฀-๿຀-࿿က-႟Ⴀ-ჿᄀ-ᇿሀ-᎟぀-ヿ"
    "㄀-ㄯ㄰-㆏ㆠ-ㇿ㐀-䶿一-鿿ꀀ-꓏가-힯"
    "豈-﫿ｦ-ﾟ]+")
RE_CJKPUNCT = re.compile("[　-〿！-｠]+")
RE_GARB = re.compile(
    "[�-\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏  ‪-‮"
    "⁠-⁤⁦-⁩﻿\U000f0000-\U0010ffff]")
RE_MOJIBAKE = re.compile("(?:Ã[\u0080-¿]|â€[\u0080-¿‘-„™œžš]|Â[ -¿])")
RE_INTERJ = re.compile(r"(?i)(?<![\w'])(hm+|wait|hold on|actually|oh+|ah+|ugh+|whoa|oops|argh+|huh|whatever|hmph)(?![\w'])")
RE_SHOUT = re.compile(r"(?<![\w'])(WAIT|AH+|OH+|AHA+|UGH+|WHOA|OOPS|ARGH+|STOP|NOO+)(?![\w'])")
RE_EXCL = re.compile(r"!{2,}")
RE_SENT = re.compile(r"[^\n.!?]{40,}")
RE_BANG = re.compile(r"(?:[^\s!]{1,40}!\s?){8,}")
RE_STUTTER = re.compile(r"(?i)(?<![\w'])([a-z]{3,})(?:\s+\1(?![\w']))+")
RE_NOSPACE = re.compile(r"\S{150,}")
RE_CAPSWORD = re.compile(r"(?<![\w'])[A-Z][A-Z']{2,}(?![\w'])")
STUTTER_SKIP = {"hmm", "wait", "hold", "actually", "ahh", "ohh", "whatever", "ugh", "argh", "hmmm", "huh", "the"}
ZW = set("​‌‍‎‏⁠⁡⁢⁣⁤﻿")
CHECKS = set("✓✗✔✘☑☒✅❌")
ARROWS = set("←→↑↓↔↩↪⇐⇒⇔⟵⟶")
INTERJ_KEYS = ("hmm", "hm/hmmm", "wait", "hold on", "actually", "oh", "ah", "ugh", "whoa/oops/argh/huh/hmph", "whatever")

_script_cache = {}
_cat_cache = {}


def zr(b):
    return len(zlib.compress(b, 6)) / max(1, len(b))


def tail_ratio(text):
    return zr(text[-TAIL:].encode()) if text else None


def window_ratios(text):
    n = len(text)
    if n < 500:
        return []
    if n <= W:
        return [(0, n, zr(text.encode()))]
    starts = list(range(0, n - W + 1, S))
    if starts[-1] != n - W:
        starts.append(n - W)
    return [(s, s + W, zr(text[s:s + W].encode())) for s in starts]


def loop_regions(wins, thr=LOOP_T):
    regs = []
    cur = None
    for s, e, r in wins:
        if r < thr:
            if cur and s <= cur[1]:
                cur[1] = e
                cur[2] = min(cur[2], r)
            else:
                if cur:
                    regs.append(cur)
                cur = [s, e, r]
        else:
            if cur:
                regs.append(cur)
                cur = None
    if cur:
        regs.append(cur)
    return [tuple(r) for r in regs]


def canon_rotation(t):
    rots = [t[i:] + t[:i] for i in range(len(t))]
    return min(rots)


def dominant_unit(seg):
    """(canonical unit tuple, coverage, k, most frequent rotation as text) of a repetitive segment."""
    toks = TOK.findall(seg[:30000])
    n = len(toks)
    if n < 4:
        return None
    best = None
    for k in range(1, 9):
        if n < 2 * k:
            break
        c = Counter(tuple(toks[i:i + k]) for i in range(n - k + 1))
        # merge rotations
        merged = Counter()
        rep = {}
        for g, cnt in c.items():
            cg = canon_rotation(g)
            merged[cg] += cnt
            if cg not in rep or cnt > c.get(rep[cg], 0):
                rep[cg] = g
        g, cnt = merged.most_common(1)[0]
        cov = min(1.0, cnt * k / n)
        if best is None or cov > best[1] + 1e-9:
            best = (g, cov, k, rep[g])
        if cov >= 0.6:
            return (g, cov, k, rep[g])
    return best


def classify_unit(unit):
    if not unit:
        return "unknown"
    toks = list(unit)
    s = " ".join(toks)
    low = s.lower()
    if re.search(r"(?<![a-z])h+m+(?![a-z])", low):
        return "hmm"
    if any(re.fullmatch(r"(?:AH+|OH+|AHA+|WAIT|OHH+|UGH+)", t) for t in toks):
        return "shouting (AH/OH/WAIT)"
    if "hold" in toks and "on" in toks:
        return "hold on"
    if "wait" in low:
        return "wait"
    if all(t in "—–-" for t in toks):
        return "bare dash"
    if any(t in CHECKS for t in toks):
        return "check marks"
    if any(t in ZW for t in toks):
        return "zero-width"
    if any(t in ARROWS for t in toks):
        return "arrows"
    if all(not t[0].isalnum() and t[0] != "_" for t in toks):
        return "other symbols"
    if any(ch in s for ch in "(){}[];=<>/\\\"'`$") or any(t.isdigit() for t in toks):
        return "code/data"
    return "words"


def script_of(ch):
    sc = _script_cache.get(ch)
    if sc is None:
        try:
            sc = unicodedata.name(ch).split()[0]
        except ValueError:
            sc = "UNNAMED"
        _script_cache[ch] = sc
    return sc


def garbage_kind(ch):
    o = ord(ch)
    if ch == "�":
        return "U+FFFD"
    if 0xE000 <= o <= 0xF8FF or o >= 0xF0000:
        return "private use"
    if o < 0x20 or 0x7F <= o <= 0x9F:
        return "control"
    if ch in ZW:
        return "zero-width"
    if 0x202A <= o <= 0x202E or 0x2066 <= o <= 0x2069:
        return "bidi"
    if ch in "  ":
        return "line/para sep"
    return "format"


def unassigned_chars(text):
    out = []
    for ch in set(text):
        if ord(ch) < 128:
            continue
        cat = _cat_cache.get(ch)
        if cat is None:
            cat = unicodedata.category(ch)
            _cat_cache[ch] = cat
        if cat in ("Cn", "Cs"):
            out.append(ch)
    return out


def interj_key(w):
    lw = w.lower()
    if lw == "hmm":
        return "hmm"
    if lw.startswith("hm"):
        return "hm/hmmm"
    if lw in ("wait", "hold on", "actually", "whatever"):
        return lw
    if lw.startswith("oh"):
        return "oh"
    if lw.startswith("ah"):
        return "ah"
    if lw.startswith("ugh"):
        return "ugh"
    return "whoa/oops/argh/huh/hmph"


def _find_all(text, sub):
    i = text.find(sub)
    while i >= 0:
        yield i
        i = text.find(sub, i + 1)


def max_window_count(positions, width=1000):
    best, bi, i = 0, 0, 0
    for j in range(len(positions)):
        while positions[j] - positions[i] > width:
            i += 1
        if j - i + 1 > best:
            best, bi = j - i + 1, positions[i]
    return best, bi


def text_features(text, ctx_has=None, want_units=True, ntok=None):
    """Feature dict of one text (reasoning or visible). ctx_has(s) -> bool tells whether s occurs in the context."""
    f = {"chars": len(text)}
    if not text:
        return f
    ascii_only = text.isascii()
    # repetition: windows / regions / units
    wins = window_ratios(text)
    if wins:
        f["min_win"] = round(min(r for _, _, r in wins), 4)
        f["wins_lt10"] = sum(r < 0.10 for _, _, r in wins)
        f["wins_lt15"] = sum(r < 0.15 for _, _, r in wins)
        f["wins_lt20"] = sum(r < 0.20 for _, _, r in wins)
        f["n_wins"] = len(wins)
    if len(text) >= 1000:
        f["tail"] = round(tail_ratio(text), 4)
    regs = loop_regions(wins) if wins else []
    if regs:
        out = []
        for s, e, r in regs:
            seg = text[s:e]
            u = dominant_unit(seg) if want_units else None
            unit, cov, k, rot = (u if u else (None, 0.0, 0, None))
            rec = {"start": s, "end": e, "min": round(r, 4), "unit": " ".join(rot) if rot else None,
                   "canon": " ".join(unit) if unit else None, "k": k, "cov": round(cov, 3),
                   "cls": classify_unit(unit), "terminal": e >= len(text) - S, "after": len(text) - e,
                   "sample": seg[len(seg) // 2:len(seg) // 2 + 90]}
            if ntok is not None and not out:
                rec["tok_before"] = ntok(text[:s])
            out.append(rec)
        f["regions"] = out
        f["loop_chars"] = sum(e - s for s, e, _ in regs)
    # char runs
    best = None
    for m in RE_CHARRUN.finditer(text):
        L = m.end() - m.start()
        if best is None or L > best[0]:
            best = (L, m.group(1), m.start())
    if best:
        f["charrun"] = {"len": best[0], "ch": best[1], "at": best[2]}
    best = None
    for m in RE_SPACEDRUN.finditer(text):
        L = len(TOK.findall(m.group(0)))
        if best is None or L > best[0]:
            best = (L, m.group(1), m.start())
    if best:
        f["spacedrun"] = {"n": best[0], "ch": best[1], "at": best[2]}
    # control tokens, think tags, tool markup
    ctrl = RE_CTRL.findall(text)
    if ctrl:
        c = Counter(ctrl)
        f["ctrl"] = dict(c)
        if ctx_has is not None:
            f["ctrl_spont"] = {t: n for t, n in c.items() if not ctx_has(t)}
        m = RE_CTRL.search(text)
        f["ctrl_ex"] = text[max(0, m.start() - 120):m.end() + 160]
    tt = RE_THINKTAG.findall(text)
    if tt:
        f["thinktag"] = dict(Counter(tt))
    tm = RE_TOOLMARK.findall(text)
    if tm:
        f["toolmark"] = len(tm)
    if not ascii_only:
        # foreign scripts
        runs = Counter()
        spont = Counter()
        ex = None
        for m in RE_FOREIGN.finditer(text):
            run = m.group(0)
            sc = script_of(run[0])
            if sc == "GREEK" and len(run) < 3:
                continue
            runs[sc] += len(run)
            if ctx_has is not None and not ctx_has(run):
                spont[sc] += len(run)
                if ex is None:
                    ex = text[max(0, m.start() - 100):m.end() + 100]
        if runs:
            f["foreign"] = dict(runs)
            if spont:
                f["foreign_spont"] = dict(spont)
                f["foreign_ex"] = ex
        cp = RE_CJKPUNCT.findall(text)
        if cp:
            f["cjkpunct"] = sum(len(x) for x in cp)
            if ctx_has is not None:
                sp = [x for x in cp if not ctx_has(x)]
                if sp:
                    f["cjkpunct_spont"] = sum(len(x) for x in sp)
        # garbage
        g = Counter()
        gs = Counter()
        gex = None
        for m in RE_GARB.finditer(text):
            k = garbage_kind(m.group(0))
            g[k] += 1
            if ctx_has is not None and not ctx_has(m.group(0)):
                gs[k] += 1
                if gex is None:
                    gex = text[max(0, m.start() - 80):m.end() + 80]
        for ch in unassigned_chars(text):
            n = text.count(ch)
            g["unassigned"] += n
            if ctx_has is not None and not ctx_has(ch):
                gs["unassigned"] += n
        mj = RE_MOJIBAKE.findall(text)
        if mj:
            g["mojibake"] += len(mj)
            if ctx_has is not None:
                n = sum(1 for x in mj if not ctx_has(x))
                if n:
                    gs["mojibake"] += n
        if g:
            f["garbage"] = dict(g)
        if gs:
            f["garbage_spont"] = dict(gs)
            if gex:
                f["garbage_ex"] = gex
        f["emdash"] = text.count("—")
    # interjections / shouting
    pos = []
    ic = Counter()
    for m in RE_INTERJ.finditer(text):
        pos.append(m.start())
        ic[interj_key(m.group(1))] += 1
    if ic:
        f["interj"] = dict(ic)
        mw, at = max_window_count(pos)
        f["interj_maxwin"] = mw
        if mw >= 8:
            f["interj_maxwin_ratio"] = round(zr(text[at:at + 1000].encode()), 4)
            f["interj_maxwin_at"] = at
    # the same outside loop regions (regions cut out; 1,001-space gaps keep windows from spanning a cut)
    if regs:
        pieces, last = [], 0
        for s, e, _ in regs:
            pieces.append(text[last:s])
            last = e
        pieces.append(text[last:])
        outside = (" " * 1001).join(p for p in pieces if p)
        f["chars_out"] = sum(len(p) for p in pieces)
        pos_o, ic_o = [], Counter()
        for m in RE_INTERJ.finditer(outside):
            pos_o.append(m.start())
            ic_o[interj_key(m.group(1))] += 1
        if ic_o:
            f["interj_out"] = dict(ic_o)
            mw, at = max_window_count(pos_o)
            f["interj_out_maxwin"] = mw
            if mw >= 8:
                f["interj_out_maxwin_ratio"] = round(zr(outside[at:at + 1000].encode()), 4)
                f["interj_out_maxwin_ex"] = outside[at:at + 300]
        f["emdash_out"] = outside.count("—")
        f["shout_out"] = len(RE_SHOUT.findall(outside))
    elif ic and f.get("interj_maxwin", 0) >= 8:
        f["interj_maxwin_ex"] = text[f["interj_maxwin_at"]:f["interj_maxwin_at"] + 300]
    # floods of em dashes / question marks / all-caps words (max per 1,000-char window, outside loop regions)
    base = outside if regs else text
    for key, pat in (("emdash", "—"), ("qmark", "?")):
        pos = [i for i in _find_all(base, pat)]
        if len(pos) >= 8:
            mw, at = max_window_count(pos)
            f[f"{key}_maxwin"] = mw
            if mw >= 25:
                f[f"{key}_maxwin_ratio"] = round(zr(base[at:at + 1000].encode()), 4)
    cw = [m.start() for m in RE_CAPSWORD.finditer(base)]
    if cw:
        f["caps_words"] = len(cw)
        if len(cw) >= 8:
            f["caps_maxwin"] = max_window_count(cw)[0]
    # 'word! word! word!' / 'word!word!' separator runs, word stutters, long no-space runs
    best = 0
    bex = None
    for m in RE_BANG.finditer(base):
        n = m.group(0).count("!")
        if n > best:
            best, bex = n, m.group(0)[:200]
    if best:
        f["bangsep"] = best
        f["bangsep_ex"] = bex
    st, stmax, stex = 0, 0, None
    for m in RE_STUTTER.finditer(base):
        w = m.group(1).lower()
        if w in STUTTER_SKIP:
            continue
        k = len(m.group(0).split())
        st += 1
        if k > stmax:
            stmax, stex = k, base[max(0, m.start() - 60):m.end() + 60]
    if st:
        f["stutter"] = st
        f["stutter_max"] = stmax
        f["stutter_ex"] = stex
    ns = 0
    for m in RE_NOSPACE.finditer(base):
        if m.end() - m.start() > ns:
            ns = m.end() - m.start()
            f["nospace_ex"] = m.group(0)[:160]
    if ns:
        f["nospace_max"] = ns
    sh = RE_SHOUT.findall(text)
    if sh:
        f["shout"] = len(sh)
    ex = RE_EXCL.findall(text)
    if ex:
        f["excl_runs"] = len(ex)
    # sentence cycling
    if len(text) >= 400:
        c = Counter(" ".join(x.split()).lower() for x in RE_SENT.findall(text))
        if c:
            seg, n = c.most_common(1)[0]
            if n >= 3:
                f["cycle"] = n
                f["cycle_seg"] = seg[:120]
    return f


def md5(text):
    return hashlib.md5(text.encode()).hexdigest()


def split_think(text):
    """analyze.py's rule: (reasoning or None, visible text or None if the think block is unclosed)."""
    s = text.strip()
    if not s.startswith("<think>"):
        return None, s
    if "</think>" not in s:
        return s[len("<think>"):], None
    inner, after = s[len("<think>"):].split("</think>", 1)
    return inner, after.strip()
