"""Emit markdown tables from mem_math.py's model. usage: python3 -I mem_table_md.py <model_meta_dir>"""
import importlib.util, sys, io, contextlib, os
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mm", os.path.join(here, "mem_math.py"))
meta = sys.argv[1]
sys.argv = ["mem_math.py", meta]
with contextlib.redirect_stdout(io.StringIO()):
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
GiB = 2**30
R = 8.0
rows = [
    # label, nodes, tp, pp, dp, ep, sp, partition, pools, note
    ("TP16 (MoE TP16 or EP16)", 2, 16, 1, 1, 16, False, None),
    ("TP8xPP2, partition 39/39", 2, 8, 2, 1, 1, False, [39, 39]),
    ("TP8xPP2, VLLM_PP_LAYER_PARTITION=41,37", 2, 8, 2, 1, 1, False, [41, 37]),
    ("DP2xTP8 + EP16 (SP-MoE)", 2, 8, 1, 2, 16, True, None),
    ("DP16xEP16 (TP1 attention)", 2, 1, 1, 16, 16, False, None),
    ("DP4xTP4 + EP16 (SP-MoE)", 2, 4, 1, 4, 16, True, None),
    ("P/D tier = TP8xPP2 (41,37), per tier", 2, 8, 2, 1, 1, False, [41, 37]),
    ("P/D tier = DP2xTP8 + EP16, per tier", 2, 8, 1, 2, 16, True, None),
    ("DP32xEP32 (TP1 attention)", 4, 1, 1, 32, 32, False, None),
    ("TP8xPP4 (21,20,19,18)", 4, 8, 4, 1, 1, False, [21, 20, 19, 18]),
    ("DP4xTP8 + EP32 (SP-MoE)", 4, 8, 1, 4, 32, True, None),
    ("DP8xTP4 + EP32 (SP-MoE)", 4, 4, 1, 8, 32, True, None),
    ("DP16xTP2 + EP32 (SP-MoE)", 4, 2, 1, 16, 32, True, None),
]
for kvdt in ("fp8_e4m3", "fp8_ds_mla", "bf16"):
    print(f"\n#### KV dtype `{kvdt}` (MTP layer included, reserve R = {R:.0f} GiB/GPU)\n")
    print("| layout | nodes | weights GiB/GPU | KV B/token/GPU | u | free GiB (before R) | KV tok/GPU | KV tok/engine | 100k seqs |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, nodes, tp, pp, dp, ep, sp, part in rows:
        sts = mm.stages(pp, part)
        for u in (0.85, 0.90, 0.93):
            toks = None; wmax = 0; kvmax = 0; freemin = None
            for si, st in enumerate(sts):
                first, last = si == 0, si == len(sts) - 1
                w = mm.gpu_bytes(tp, ep, st, first, last, True, sp, mtp_own_embed=(pp > 1))
                kvb, _, _ = mm.kv_per_token(st, last, True, kvdt)
                free = u * mm.TOTAL - w
                t = max(0.0, free - R * GiB) / kvb
                toks = t if toks is None else min(toks, t)
                wmax = max(wmax, w); kvmax = max(kvmax, kvb)
                freemin = free if freemin is None else min(freemin, free)
            eng = toks * dp
            seqs = int(toks // 100_000) * dp
            print(f"| {label if u == 0.85 else ''} | {nodes if u == 0.85 else ''} | {wmax/GiB:.1f} | {kvmax:,} | {u} | {freemin/GiB:.1f} | {toks/1e3:,.0f}k | {eng/1e6:.2f}M | {seqs} |")
