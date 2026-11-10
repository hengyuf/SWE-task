import importlib.util, sys, io, contextlib
spec = importlib.util.spec_from_file_location("mm", "mem_math.py")
sys.argv = ["mem_math.py", sys.argv[1]]
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
GiB = 2**30
for part in ([39, 39], [40, 38], [41, 37], [42, 36]):
    sts = mm.stages(2, part)
    res = []
    for u in (0.90, 0.93):
        toks = []
        for si, st in enumerate(sts):
            last = si == 1
            w = mm.gpu_bytes(8, 1, st, si == 0, last, True, False, mtp_own_embed=True)
            kvb, _, _ = mm.kv_per_token(st, last, True, "fp8_e4m3")
            toks.append((u * mm.TOTAL - w - 8 * GiB) / kvb)
        res.append(f"u={u}: stage tokens {toks[0]/1e3:.0f}k/{toks[1]/1e3:.0f}k -> engine {min(toks)/1e3:.0f}k")
    print(part, " | ".join(res))
