#!/usr/bin/env python3
"""REPORT.md for the DeepSWE failure-pattern taxonomy (stdlib; reads this directory's files only)."""
import json, re
from collections import Counter, defaultdict
from pathlib import Path
OUT = Path(__file__).resolve().parent
TASKS = Path("/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-task/synth/cache/deep-swe/tasks")
r = json.loads((OUT / "workflow_result.json").read_text())
tax, fin = r["taxonomy"], r["final"]
pats = {p["id"]: p for p in (json.loads(l) for l in open(OUT / "primary_patterns.jsonl"))}
lab = {x["id"]: x for x in fin["final"]}
blind = json.loads((OUT / "blind_check_agreement.json").read_text())
EXT = {"py": "python", "go": "go", "ts": "typescript", "tsx": "typescript", "js": "javascript", "mjs": "javascript", "rs": "rust"}
def lang(t):
    exts = Counter(EXT[m.group(1)] for m in re.finditer(r"^\+\+\+ b/\S+\.([A-Za-z0-9]+)$", (TASKS / t / "tests/test.patch").read_text(), re.M) if m.group(1) in EXT)
    return exts.most_common(1)[0][0] if exts else "?"
targets = sorted({p["target"] for p in pats.values()})
L = {t: lang(t) for t in targets}
order = [c["key"] for c in tax["categories"] if c["key"] != "other"]
name = {c["key"]: c["name_zh"] for c in tax["categories"]}
short = {c["key"]: re.split(r"[：（(]", c["name_zh"])[0] for c in tax["categories"]}
en = {c["key"]: c["name_en"] for c in tax["categories"]}
narr = {c["key"]: c for c in fin["categories"]}
n_pat, tg, sl = Counter(), defaultdict(set), Counter()
per_t = defaultdict(Counter)
for pid, p in pats.items():
    c = lab[pid]["primary"]; w = p["n_student_slots"] or 0
    n_pat[c] += 1; tg[c].add(p["target"]); sl[c] += w; per_t[p["target"]][c] += w
P, T, S = len(pats), len(targets), sum(sl.values())
dom = Counter(max(cnt, key=lambda c: (cnt[c], -order.index(c))) for cnt in per_t.values())
lang_sl = defaultdict(Counter); lang_T = Counter(L.values())
for pid, p in pats.items():
    lang_sl[L[p["target"]]][lab[pid]["primary"]] += p["n_student_slots"] or 0
order.sort(key=lambda c: -sl[c])
layer = {"no_delivery": "过程", "evidence_override": "过程", "self_confirming_verification": "过程"}
md = []
md.append("# DeepSWE target failure analysis 汇总：sc14 的失败模式可以分成哪几类\n")
md.append("2026-11-03。汇总范围：hengyuf synth runs（official1–4、val1）和 Baihe fork 中所有 DeepSWE target 的 failure analysis，"
          f"每个 target 取一份主分析（优先级 official4 > official3 > official2 > official1 > val1 > Baihe fork），共 {T} 个 target、{P} 条 failure pattern。"
          "这些 target 都是 Kimi K3 能做、sc14 失败的任务，所以下面描述的是 sc14 相对 Kimi 的差距，不是 sc14 全部失败的分布。\n")
md.append("## 结论\n")
md.append("分析者原本给 246 条 pattern 写了 227 种不同的 category，归并后可以稳定地分成 **9 类**，分属两层：\n")
md.append(f"- **过程层**（{sum(n_pat[c] for c in layer)} 条，按失败 slot 加权 {100*sum(sl[c] for c in layer)/S:.0f}%）：交付、验证、对待反证的方式出了问题。")
md.append(f"- **产品缺陷层**（{P - sum(n_pat[c] for c in layer)} 条，加权 {100*(S - sum(sl[c] for c in layer))/S:.0f}%）：提交的代码本身的缺陷。\n")
md.append("| # | 类别 | 层 | pattern 数 | 涉及 target | 失败 slot 加权占比 | 是主导类别的 target |")
md.append("|---|---|---|---|---|---|---|")
for i, c in enumerate(order, 1):
    md.append(f"| {i} | **{short[c]}**（{en[c]}） | {layer.get(c, '产品')} | {n_pat[c]} ({100*n_pat[c]/P:.0f}%) | {len(tg[c])}/{T} | {100*sl[c]/S:.1f}% | {dom[c]} |")
md.append(f"\n'失败 slot 加权' = 每条 pattern 按它覆盖的失败 sc14 rollout 数计权（合计 {S} 次引用），更接近'这种机制让多少次 rollout 失败'。"
          "一条 pattern 只算一个主类；同一 target 通常同时有 2–3 类（平均 2.8 类）。\n")
md.append("## 各类说明\n")
for c in order:
    n = narr.get(c, {})
    md.append(f"### {short[c]}（{c}）\n")
    md.append(f"{name[c]}。{n_pat[c]} 条 / {len(tg[c])} 个 target，加权 {100*sl[c]/S:.1f}%。\n")
    md.append(n.get("summary_zh", "") + "\n")
    md.append("**Kimi 的做法：**" + n.get("teacher_contrast_zh", "") + "\n")
    ex = n.get("examples") or []
    if ex:
        md.append("例子：")
        for e in ex[:4]:
            md.append(f"- `{e['target']}`：{e['one_line_zh']}")
        md.append("")
md.append("## 跨类别的观察\n")
for x in fin["cross_cutting_zh"]:
    if x.startswith("语言与批次"):
        continue  # replaced by the corrected language table below
    md.append(f"- {x}")
md.append("\n### 按语言（语言按测试文件判定，失败 slot 加权占比；样本小的语言仅供参考）\n")
langs = [l for l, _ in lang_T.most_common()]
md.append("| 类别 | " + " | ".join(f"{l} ({lang_T[l]} 个 target)" for l in langs) + " |")
md.append("|---|" + "---|" * len(langs))
for c in order:
    md.append(f"| {short[c]} | " + " | ".join(f"{100*lang_sl[l][c]/max(1, sum(lang_sl[l].values())):.0f}%" for l in langs) + " |")
md.append("\n## 方法与可靠性\n")
md.append(f"- 数据：`patterns.jsonl`（全部 315 条，含非主分析）、`primary_patterns.jsonl`（{P} 条主 pattern）、`analyses.jsonl`（89 份分析）；`extract_patterns.py` 生成。")
md.append("- 分类（workflow wf_7b43bb70-398）：两个 agent 分别从'根因/行为'和'缺失的工程能力'两个视角独立提出分类，第三个 agent 合并成 9 类并写出判定规则；"
          "3 个 agent 分批标注全部 pattern，2 个 agent 再独立标注一遍，最后一个 agent 裁决分歧并写各类说明（`workflow_result.json`）。")
md.append(f"- 两轮标注主类 244/246 一致，但合并后的分类说明里按编号点名了不少 pattern，这个一致率偏高。因此另做了一次真正盲测（wf_4e621b04-9ac）："
          f"随机抽 {blind['n']} 条，只给去掉所有 pattern 编号和例子的定义与规则（`taxonomy_blind.json`），"
          f"主类与最终标签一致 {blind['exact']}/{blind['n']}（{100*blind['exact']/blind['n']:.0f}%，Cohen κ = {blind['kappa']:.2f}），"
          f"考虑第二标签后 {blind['lenient']}/{blind['n']} 重合。分歧集中在'状态/组合'与'覆盖/集成缺口'之间。")
md.append("- 语言：DeepSWE 的 task.toml 有 4 处误标（httpx-deterministic-cookie-store 实为 Python，prometheus-transactional-reload-status 实为 Go，"
          "katex-multicolumn-array-spans 与 koota-entity-snapshot-rollback 的测试是 TypeScript），本报告按测试文件判定语言。")
md.append("\n## 注意事项\n")
for x in fin["caveats_zh"]:
    if "/tmp" in x or "语言标签来自 task.toml" in x:
        continue  # the /tmp drafts were deleted on 2026-11-03; language handled above
    md.append(f"- {x}")
md.append("- 上一步生成 taxonomy 时，有 agent 违反只读要求，在共享的 /tmp 留下了其他用户可读的草稿文件（由 pattern 文本派生，不含密钥）；已于 2026-11-03 删除。")
md.append("\n## 文件\n")
md.append("`REPORT.md`（本文）、`category_stats.json`、`workflow_result.json`（分类、两轮标注的裁决结果、各类说明）、`taxonomy_blind.json`、`blind_check_labels.json`、`blind_check_agreement.json`、"
          "`primary_patterns.jsonl` / `patterns.jsonl` / `analyses.jsonl`、`extract_patterns.py`、`make_report.py`。\n")
(OUT / "REPORT.md").write_text("\n".join(md) + "\n")
labels_out = [{"id": pid, "target": pats[pid]["target"], "language": L[pats[pid]["target"]], "name": pats[pid]["name"],
               "primary": lab[pid]["primary"], "secondary": lab[pid].get("secondary") or "", "n_student_slots": pats[pid]["n_student_slots"]}
              for pid in sorted(pats)]
(OUT / "final_labels.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in labels_out))
print("REPORT.md written:", len("\n".join(md)), "chars; languages:", dict(lang_T))
