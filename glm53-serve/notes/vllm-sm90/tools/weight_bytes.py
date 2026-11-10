#!/usr/bin/env python3
"""Derive GLM-5.3 per-tensor byte sizes from config.json + safetensors index names.

The index has no per-tensor sizes, so shapes come from config.json. Unknown dtypes
(router gate weight, indexer weights_proj, k_norm) are resolved by matching the
index's metadata.total_size. Usage: python3 -I weight_bytes.py <model_meta_dir>
"""
import json, math, re, sys, itertools, collections

d = sys.argv[1]
cfg = json.load(open(f"{d}/config.json"))
idx = json.load(open(f"{d}/model.safetensors.index.json"))
TOTAL = idx["metadata"]["total_size"]
names = list(idx["weight_map"])

H = cfg["hidden_size"]; V = cfg["vocab_size"]; NH = cfg["num_attention_heads"]
QL = cfg["q_lora_rank"]; KVL = cfg["kv_lora_rank"]; NOPE = cfg["qk_nope_head_dim"]
ROPE = cfg["qk_rope_head_dim"]; VH = cfg["v_head_dim"]; I = cfg["intermediate_size"]
MI = cfg["moe_intermediate_size"]; E = cfg["n_routed_experts"]
IH = cfg["index_n_heads"]; ID = cfg["index_head_dim"]
B = cfg["quantization_config"]["weight_block_size"][0]

def fp8(o, i):
    """fp8 weight [o,i] + fp32 block scales [ceil(o/B), ceil(i/B)]"""
    return o * i, math.ceil(o / B) * math.ceil(i / B) * 4

def shape_bytes(name, dt):
    """returns (bytes, category). dt: dict of unresolved dtype sizes"""
    n = re.sub(r"layers\.\d+\.", "", name)
    lay = re.search(r"layers\.(\d+)\.", name)
    lay = int(lay.group(1)) if lay else None
    is_scale = n.endswith("weight_scale_inv")
    base = n.replace(".weight_scale_inv", ".weight")
    def pick(o, i):
        w, s = fp8(o, i)
        return s if is_scale else w
    cat = "nonexpert"
    if "mlp.experts." in n:
        cat = "routed_expert"
        if base.endswith("down_proj.weight"):
            return pick(H, MI), cat
        return pick(MI, H), cat
    if lay is not None and lay >= cfg["num_hidden_layers"]:
        cat = "mtp"
    if base == "model.embed_tokens.weight" or base == "lm_head.weight":
        return V * H * 2, ("embed" if "embed" in base else "lm_head")
    if "shared_experts." in n:
        if base.endswith("down_proj.weight"):
            return pick(H, MI), (cat if cat == "mtp" else "shared_expert")
        return pick(MI, H), (cat if cat == "mtp" else "shared_expert")
    if re.search(r"mlp\.(gate|up)_proj", n):
        return pick(I, H), "dense_mlp"
    if re.search(r"mlp\.down_proj", n):
        return pick(H, I), "dense_mlp"
    if n.endswith("mlp.gate.weight"):
        return E * H * dt["router"], (cat if cat == "mtp" else "router")
    if n.endswith("e_score_correction_bias"):
        return E * 4, (cat if cat == "mtp" else "router")
    c = cat if cat == "mtp" else "attn"
    if "indexer." in n:
        c = cat if cat == "mtp" else "indexer"
        if "wq_b" in n: return pick(IH * ID, QL), c
        if "indexer.wk" in n: return pick(ID, H), c
        if "k_norm" in n: return ID * dt["knorm"], c
        if "weights_proj" in n: return IH * H * dt["wproj"], c
    if "q_a_proj" in n: return pick(QL, H), c
    if "q_b_proj" in n: return pick(NH * (NOPE + ROPE), QL), c
    if "kv_a_proj_with_mqa" in n: return pick(KVL + ROPE, H), c
    if "kv_b_proj" in n: return pick(NH * (NOPE + VH), KVL), c
    if "o_proj" in n: return pick(H, NH * VH), c
    if "q_a_layernorm" in n: return QL * 2, c
    if "kv_a_layernorm" in n: return KVL * 2, c
    if n.endswith("layernorm.weight") or n.endswith("norm.weight"):
        return H * 2, c
    if "eh_proj" in n: return H * 2 * H * 2, "mtp"
    raise KeyError(name)

best = None
for r, k, w in itertools.product([2, 4], [2, 4], [2, 4]):
    dt = {"router": r, "knorm": k, "wproj": w}
    tot = sum(shape_bytes(nm, dt)[0] for nm in names)
    print(f"router={r} knorm={k} wproj={w}: total={tot:,} diff={tot-TOTAL:,}")
    if tot == TOTAL:
        best = dt
print("MATCH:", best)
if best:
    cats = collections.Counter()
    per_layer = collections.Counter()
    for nm in names:
        b, c = shape_bytes(nm, best)
        cats[c] += b
        m = re.search(r"layers\.(\d+)\.", nm)
        if m and "mlp.experts." in nm:
            per_layer[int(m.group(1))] += b
    for c, b in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"{c:15s} {b:>18,} B = {b/2**30:9.3f} GiB")
    print("total", f"{sum(cats.values()):,}")
    one = per_layer[3]
    print("routed expert bytes per MoE layer:", f"{one:,}", "per expert:", f"{one//E:,}")
