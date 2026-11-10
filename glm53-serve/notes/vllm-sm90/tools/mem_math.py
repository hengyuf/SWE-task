#!/usr/bin/env python3
"""GLM-5.3 on H100-80GB: per-GPU weight bytes, free memory and KV capacity per parallel layout.

stdlib only; usage: python3 -I mem_math.py <model_meta_dir>
Shapes come from config.json; byte totals reproduce index metadata.total_size exactly
(see weight_bytes.py). Sharding rules follow vLLM's DSA model code:
  replicated under TP : fused_qkv_a_proj (q_a + kv_a), indexer wq_b/wk/weights_proj, norms,
                        router gate (+bias), MTP eh_proj (nn.Linear)
  TP-sharded          : q_b_proj, kv_b_proj, o_proj, embed_tokens (vocab), lm_head (vocab),
                        dense MLP + shared expert (replicated instead under SP-MoE: TP>1 & DP>1 & EP)
  routed experts      : / EP size (EP) or / TP size (TP-MoE)
  MLA absorbed copies : W_UK_T + W_UV in bf16 = heads_per_rank * (192*512 + 512*256) * 2 B per layer
KV per token per GPU  : sum over that GPU's attention layers of the MLA latent row
                        (bf16 1152 B, fp8_e4m3 576 B, fp8_ds_mla 656 B) + 132 B per indexer layer.
                        MLA latent is replicated on every TP rank (no DCP).
"""
import json, math, sys

d = sys.argv[1]
cfg = json.load(open(f"{d}/config.json"))
H = cfg["hidden_size"]; V = cfg["vocab_size"]; NH = cfg["num_attention_heads"]
QL = cfg["q_lora_rank"]; KVL = cfg["kv_lora_rank"]; NOPE = cfg["qk_nope_head_dim"]
ROPE = cfg["qk_rope_head_dim"]; VH = cfg["v_head_dim"]; I = cfg["intermediate_size"]
MI = cfg["moe_intermediate_size"]; E = cfg["n_routed_experts"]; L = cfg["num_hidden_layers"]
IH = cfg["index_n_heads"]; ID = cfg["index_head_dim"]
idx_layers = [i for i, t in enumerate(cfg["indexer_types"]) if t == "full"]
dense_layers = [i for i, t in enumerate(cfg["mlp_layer_types"]) if t == "dense"]
GiB = 2**30
B = 128

def fp8(o, i):  # fp8 weight + fp32 128x128 block scales
    return o * i + math.ceil(o / B) * math.ceil(i / B) * 4

ATT_REPL = fp8(QL + KVL + ROPE, H) + (QL + KVL + 2 * H) * 2           # fused qkv_a + q_a/kv_a/in/post norms
ATT_TP = fp8(NH * (NOPE + ROPE), QL) + fp8(NH * (NOPE + VH), KVL) + fp8(H, NH * VH)
ABSORB_PER_HEAD = (NOPE * KVL + KVL * VH) * 2                            # W_UK_T + W_UV bf16
IDX_REPL = fp8(IH * ID, QL) + (ID * H + IH * H) * 2 + 2 * ID * 2          # wq_b fp8, wk(bf16 after load)+weights_proj, k_norm
EXPERT = 3 * fp8(MI, H)                                                   # one routed expert (gate, up, down)
SHARED = 3 * fp8(MI, H)
ROUTER = E * H * 2 + E * 4
DENSE = 3 * fp8(I, H)
EMB = V * H * 2
MTP_EXTRA = H * 2 * H * 2 + 3 * H * 2                                     # eh_proj bf16 + enorm/hnorm/shared_head.norm
FINAL_NORM = H * 2
# sanity: total checkpoint bytes
tot = (L + 1) * (ATT_REPL + ATT_TP) + (len(idx_layers) + 1) * (fp8(IH * ID, QL) + fp8(ID, H) + IH * H * 2 + 2 * ID * 2) \
    + (L - len(dense_layers) + 1) * (E * EXPERT + SHARED + ROUTER) + len(dense_layers) * DENSE + 2 * EMB + MTP_EXTRA + FINAL_NORM
print(f"checkpoint bytes (sanity vs index 755,617,140,416): {tot:,}")

KV_MLA = {"bf16": 1152, "fp8_e4m3": 576, "fp8_ds_mla": 656}
KV_IDX = 132

def gpu_bytes(tp, ep, pp_stage_layers, first, last, mtp, sp_moe, mtp_own_embed):
    """bytes of weights on one GPU of a PP stage holding layers pp_stage_layers."""
    b = 0
    moe_div = ep if ep > 1 else tp
    for li in pp_stage_layers:
        b += ATT_REPL + ATT_TP / tp + NH // tp * ABSORB_PER_HEAD
        if li in idx_layers:
            b += IDX_REPL
        if li in dense_layers:
            b += DENSE if sp_moe else DENSE / tp
        else:
            b += E * EXPERT / moe_div + (SHARED if sp_moe else SHARED / tp) + ROUTER
    if first:
        b += EMB / tp
    if last:
        b += EMB / tp + FINAL_NORM       # lm_head
        if mtp:
            b += ATT_REPL + ATT_TP / tp + NH // tp * ABSORB_PER_HEAD + IDX_REPL
            b += E * EXPERT / moe_div + (SHARED if sp_moe else SHARED / tp) + ROUTER + MTP_EXTRA
            if mtp_own_embed:
                b += EMB / tp
    return b

def kv_per_token(stage_layers, last, mtp, kvdt):
    n_mla = len(stage_layers) + (1 if (last and mtp) else 0)
    n_idx = sum(1 for li in stage_layers if li in idx_layers) + (1 if (last and mtp) else 0)
    return n_mla * KV_MLA[kvdt] + n_idx * KV_IDX, n_mla, n_idx

TOTAL = 81559 * 2**20          # H100 80GB HBM3 as reported by nvidia-smi (81,559 MiB)

def stages(pp, partition=None):
    if pp == 1:
        return [list(range(L))]
    part = partition or [L // pp] * pp
    out, s = [], 0
    for n in part:
        out.append(list(range(s, s + n))); s += n
    return out

configs = [
    # name, nodes, tp, pp, dp, ep, sp_moe, reserve_GiB, partition, note
    ("TP16 (MoE TP16)",          2, 16, 1, 1, 1,  False, 8.0, None, "one engine"),
    ("TP16 + EP16",              2, 16, 1, 1, 16, False, 8.0, None, "one engine"),
    ("TP8xPP2 (even 39/39)",     2, 8, 2, 1, 1,  False, 8.0, None, "one engine"),
    ("TP8xPP2 (VLLM_PP_LAYER_PARTITION=40,38)", 2, 8, 2, 1, 1, False, 8.0, [40, 38], "one engine"),
    ("DP2xTP8 + EP16 (SP-MoE)",  2, 8, 1, 2, 16, True,  8.0, None, "2 DP ranks"),
    ("DP16xEP16 (TP1 attn)",     2, 1, 1, 16, 16, False, 8.0, None, "16 DP ranks"),
    ("DP4xTP4 + EP16 (SP-MoE)",  2, 4, 1, 4, 16, True,  8.0, None, "4 DP ranks"),
    ("DP32xEP32 (TP1 attn)",     4, 1, 1, 32, 32, False, 8.0, None, "32 DP ranks"),
    ("2 x [DP16xEP16] replicas", 4, 1, 1, 16, 16, False, 8.0, None, "per replica"),
    ("TP8xPP4 (VLLM_PP_LAYER_PARTITION=21,20,19,18)", 4, 8, 4, 1, 1, False, 8.0, [21, 20, 19, 18], "one engine"),
    ("DP4xTP8 + EP32 (SP-MoE)",  4, 8, 1, 4, 32, True,  8.0, None, "4 DP ranks"),
    ("DP8xTP4 + EP32 (SP-MoE)",  4, 4, 1, 8, 32, True,  8.0, None, "8 DP ranks"),
    ("DP16xTP2 + EP32 (SP-MoE)", 4, 2, 1, 16, 32, True, 8.0, None, "16 DP ranks"),
]

for kvdt in ("fp8_e4m3", "fp8_ds_mla", "bf16"):
    print(f"\n===== KV dtype {kvdt}  (MTP layer included; reserve = activations+workspace+graphs, see notes) =====")
    print(f"{'config':44s} {'GiB wts/GPU':>11s} {'B/tok/GPU':>9s} | " + " | ".join(
        f"u={u}: free  KVtok/GPU  KVtok/engine  100k-seqs" for u in (0.85, 0.90, 0.93)))
    for name, nodes, tp, pp, dp, ep, sp, R, part, note in configs:
        sts = stages(pp, part)
        worst_tokens = None; worst_w = 0; worst_kvb = 0
        for si, st in enumerate(sts):
            first, last = si == 0, si == len(sts) - 1
            w = gpu_bytes(tp, ep, st, first, last, True, sp, mtp_own_embed=(pp > 1))
            kvb, _, _ = kv_per_token(st, last, True, kvdt)
            worst_w = max(worst_w, w)
            worst_kvb = max(worst_kvb, kvb)
            per_u = []
            for u in (0.85, 0.90, 0.93):
                free = u * TOTAL - w - R * GiB
                per_u.append(max(0, free) / kvb)
            worst_tokens = per_u if worst_tokens is None else [min(a, b) for a, b in zip(worst_tokens, per_u)]
        cells = []
        for u, toks in zip((0.85, 0.90, 0.93), worst_tokens):
            free = u * TOTAL / GiB - worst_w / GiB
            ranks = dp  # independent KV pools per engine
            eng = toks * ranks
            seqs = int(toks // 100_000) * ranks
            cells.append(f"{free:6.1f}  {toks/1e3:8.0f}k  {eng/1e6:8.2f}M  {seqs:5d}")
        print(f"{name:44s} {worst_w/GiB:11.2f} {worst_kvb:9d} | " + " | ".join(cells))

print("\nfree = u*79.65 GiB - weights (before the reserve); KVtok/GPU uses free - reserve(8 GiB);")
print("each 1 GiB of reserve = 1 GiB / (B/tok) tokens; engine = DP ranks x per-GPU (MLA KV replicated across TP).")
# per-stage detail for PP2
for part in (None, [40, 38]):
    sts = stages(2, part)
    for si, st in enumerate(sts):
        first, last = si == 0, si == 1
        w = gpu_bytes(8, 1, st, first, last, True, False, mtp_own_embed=True)
        kvb, nm, ni = kv_per_token(st, last, True, "fp8_e4m3")
        print(f"PP2 partition={part or '39,39'} stage{si}: layers {st[0]}-{st[-1]} weights/GPU {w/GiB:.2f} GiB, "
              f"MLA layers {nm}, indexer layers {ni}, fp8_e4m3 B/token {kvb}")
# components
print("\ncomponent bytes: ATT_REPL %.1f MB, ATT_TP %.1f MB, absorb/layer@TP1 %.1f MB, IDX_REPL %.1f MB, expert %.2f MB, "
      "routed/MoE-layer %.2f GB, shared %.1f MB, router %.1f MB, dense %.1f MB, emb %.2f GB, MTP extra %.1f MB" % (
      ATT_REPL/1e6, ATT_TP/1e6, NH*ABSORB_PER_HEAD/1e6, IDX_REPL/1e6, EXPERT/1e6, E*EXPERT/1e9, SHARED/1e6,
      ROUTER/1e6, DENSE/1e6, EMB/1e9, MTP_EXTRA/1e6))
# GB200 calibration: HSG E13 DP8xEP8 TP1, GB200 total ~ 189,471 MiB (UNVERIFIED)
w = gpu_bytes(1, 8, list(range(L)), True, True, True, False, False)
print(f"\nHSG calibration DP8xEP8 TP1 weights/GPU (model) = {w/GiB:.1f} GiB (HSG NOTES: 106.6 GiB)")
for u, kv in ((0.90, 39.3), (0.93, 55.7)):
    tot = 189471 * 2**20
    print(f"  GB200 u={u}: budget {u*tot/GiB:.1f} GiB - weights {w/GiB:.1f} - KV {kv} = other {u*tot/GiB - w/GiB - kv:.1f} GiB")
