"""TP8xPP2 + DCP8 capacity (KV sharded 8x across each TP group). usage: python3 -I dcp_calc.py <model_meta_dir>"""
import importlib.util, sys, io, contextlib, os
here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("mm", os.path.join(here, "mem_math.py"))
sys.argv = ["mem_math.py", sys.argv[1]]
with contextlib.redirect_stdout(io.StringIO()):
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
GiB = 2**30; R = 8.0; DCP = 8
QB_FULL = mm.fp8(mm.NH * (mm.NOPE + mm.ROPE), mm.QL)
for kvdt in ("fp8_ds_mla",):
    for qrep in (False, True):
        out = []
        for u in (0.90, 0.93):
            toks = []
            for si, st in enumerate(mm.stages(2, [41, 37])):
                last = si == 1
                w = mm.gpu_bytes(8, 1, st, si == 0, last, True, False, mtp_own_embed=True)
                nl = len(st) + (1 if last else 0)
                if qrep:
                    w += nl * QB_FULL * (1 - 1 / 8)
                kvb, _, _ = mm.kv_per_token(st, last, True, kvdt)
                toks.append((u * mm.TOTAL - w - R * GiB) / (kvb / DCP))
            out.append(f"u={u}: {min(toks)/1e6:.2f}M tokens/engine")
        print(f"TP8xPP2(41,37)+DCP8 kv={kvdt} q_replicate={qrep}: " + "; ".join(out))
