"""Run the REAL SFTDataset.__getitem__ (sft_dataset.py sha cadbab60, = HSG runtime pin) on one packed row.

Megatron imports are stubbed; the method bodies are executed unchanged from the source file. The tokenizer is the
real SFTTokenizer (sft_tokenizer.py) with prompt format identity and the sc14 HF tokenizer. Read-only on inputs;
writes only a JSON summary to --out.

Checks: cu_seqlens == cumulative ceil32(conversation token lengths) (last = seq_len), position ids reset at every
segment start, every loss-carrying position predicts a label inside its own segment, loss_mask sum ==
sum(assistant_token_counts * assistant_weights), Mamba seq_idx (MambaMixer._create_packed_seq_idx logic) changes
exactly at cu_seqlens, MTP packed roll (multi_token_prediction._roll_tensor_packed_seq, cp_size 1 path) never pairs a
loss-carrying position with a label from another segment.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import types

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

RT = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_agents_dev/users/zhenghaox/swe-rubric-pipeline/"
      "weighted-sft-dfw-v1/runtime/megatron-lm-ultra-step-v10")
SFT_DATASET = RT + "/megatron/training/datasets/sft_dataset.py"
SFT_TOKENIZER = RT + "/megatron/core/tokenizers/text/libraries/sft_tokenizer.py"
SFT_TOKENIZER_LSTRIP = ("/lustre/fsw/portfolios/nemotron/projects/nemotron_rl_algo/users/aayushk/megatron-lm-ultra-adi/"
                        "megatron/core/tokenizers/text/libraries/sft_tokenizer.py")
HF = "/lustre/fsw/portfolios/nemotron/users/hengyuf/swe-rl-v2-sc-migration/models/swonlysc-step0014/hf"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_module(name, path):
    src = open(path, encoding="utf-8").read()
    mod = types.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod


def stub_megatron():
    for name in ["megatron", "megatron.core", "megatron.core.datasets", "megatron.core.datasets.gpt_dataset",
                 "megatron.core.datasets.megatron_dataset", "megatron.core.datasets.utils"]:
        sys.modules[name] = types.ModuleType(name)
    sys.modules["megatron.core.datasets.gpt_dataset"].GPTDatasetConfig = object

    class MegatronDataset:  # only the attributes __getitem__ reads are set by the caller
        def __init__(self, *a, **k):
            pass
    sys.modules["megatron.core.datasets.megatron_dataset"].MegatronDataset = MegatronDataset
    sys.modules["megatron.core.datasets.megatron_dataset"].LowLevelDataset = object
    sys.modules["megatron.core.datasets.utils"].Split = object


class TokWrap:
    """megatron.core.tokenizers.text.text_tokenizer: pad -> SFTTokenizer.pad_id, eod -> eod."""

    def __init__(self, sft):
        self._t = sft
        self.pad = sft.pad_id
        self.eod = sft.eod

    def tokenize_conversation(self, conversation, return_target, add_generation_prompt):
        return self._t.tokenize_conversation(conversation=conversation, return_target=return_target,
                                             add_generation_prompt=add_generation_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--max-convs", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=524288)
    ap.add_argument("--cp", type=int, default=16)
    ap.add_argument("--tokenizer-variant", choices=("commit", "lstrip"), default="commit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()

    import numpy as np
    import torch

    stub_megatron()
    ds_mod = load_module("sft_dataset_real", SFT_DATASET)
    tok_path = SFT_TOKENIZER if args.tokenizer_variant == "commit" else SFT_TOKENIZER_LSTRIP
    tok_mod = load_module("sft_tokenizer_real", tok_path)
    sft = tok_mod.SFTTokenizer(HF, "identity")
    # transformers 5.x defaults apply_chat_template(tokenize=True) to return_dict=True; the training container's
    # transformers 4.x returned plain ids. Restore the 4.x return type so the unchanged SFTTokenizer code runs.
    _act = sft._tokenizer.apply_chat_template

    def _act_v4(*a, **k):
        k.setdefault("return_dict", False)
        return _act(*a, **k)
    sft._tokenizer.apply_chat_template = _act_v4
    tok = TokWrap(sft)

    with open(args.shard, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == args.row:
                msgs = json.loads(line)["messages"]
                break
    # datasets' json loader turns the union of message keys into nulls on the other messages; mimic it.
    keys = set()
    for m in msgs:
        keys |= set(m)
    msgs = [{k: m.get(k) for k in sorted(keys)} for m in msgs]
    convs_meta = []
    cut = len(msgs)
    n_sys = 0
    for j, m in enumerate(msgs):
        if m["role"] == "system":
            n_sys += 1
            if args.max_convs and n_sys > args.max_convs:
                cut = j
                break
            convs_meta.append({"id": m["step_weight_agent_run_id"], "counts": m["assistant_token_counts"],
                               "weights": m["assistant_weights"]})
    msgs = msgs[:cut]

    ds = object.__new__(ds_mod.SFTDataset)
    ds.dataset = [msgs]
    ds.indices = np.array([0])
    ds.config = types.SimpleNamespace(tokenizer=tok, sequence_length=args.seq_len, context_parallel_size=args.cp,
                                      reset_position_ids=False, create_attention_mask=False,
                                      reset_attention_mask=False)
    t1 = time.time()
    item = ds.__getitem__(0)
    t2 = time.time()

    tokens, labels, loss_mask = item["tokens"], item["labels"], item["loss_mask"]
    pos, cu, max_seqlen = item["position_ids"], item["cu_seqlens"], item["max_seqlen"]
    cu_l = cu.tolist()
    # per-conversation raw token lengths, from the real tokenizer
    conv_lens = []
    for k, c in enumerate(ds._split_conversations(msgs)):
        t, _ = tok.tokenize_conversation(c, return_target=True, add_generation_prompt=False)
        conv_lens.append(int(len(t)))
    g = 2 * args.cp
    expect_cu = [0]
    for L in conv_lens:
        expect_cu.append(expect_cu[-1] + (L + g - 1) // g * g)
    expect_cu_final = expect_cu[:-1] + [args.seq_len]

    res = {"shard": args.shard, "row": args.row, "max_convs": args.max_convs, "tokenizer_variant": args.tokenizer_variant,
           "sft_dataset_sha256": sha(SFT_DATASET), "sft_tokenizer_sha256": sha(tok_path),
           "pad_id": tok.pad, "eod_id": tok.eod, "n_conversations": len(conv_lens), "conv_token_lengths": conv_lens,
           "conv_len_mod32": [L % 32 for L in conv_lens], "cu_seqlens": cu_l, "expected_cu_seqlens": expect_cu_final,
           "cu_seqlens_ok": cu_l == expect_cu_final, "max_seqlen": int(max_seqlen), "tokens_len": int(tokens.numel())}

    # position ids reset to 0 at each segment start and count up inside the segment (pads continue the count)
    pos_ok = True
    for a, b in zip(cu_l[:-1], cu_l[1:]):
        seg = pos[a:b]
        if not torch.equal(seg, torch.arange(b - a, dtype=seg.dtype)):
            pos_ok = False
    res["position_ids_reset_per_segment"] = pos_ok

    # loss-carrying positions must predict a label inside their own segment (label of p = target at p+1)
    seg_of = torch.zeros(args.seq_len, dtype=torch.long)
    for k, (a, b) in enumerate(zip(cu_l[:-1], cu_l[1:])):
        seg_of[a:b] = k
    nz = torch.nonzero(loss_mask > 0).flatten()
    cross = [int(p) for p in nz if int(p) + 1 >= args.seq_len or int(seg_of[min(int(p) + 1, args.seq_len - 1)]) != int(seg_of[int(p)])]
    res["loss_positions"] = int(nz.numel())
    res["loss_positions_with_label_from_other_segment"] = len(cross)
    # boundary labels: last input position of each segment
    bl = []
    for b in cu_l[1:]:
        p = b - 1
        bl.append({"pos": p, "label": int(labels[p]), "loss_mask": float(loss_mask[p]),
                   "label_is_pad": int(labels[p]) == tok.pad, "label_is_ignore": int(labels[p]) == -100})
    res["segment_last_position_labels"] = bl
    # first real token of each conversation: input token and its own target is never trained (it is a system token)
    res["loss_mask_sum"] = float(loss_mask.sum())
    exp = sum(c * w for m in convs_meta for c, w in zip(m["counts"], m["weights"]))
    res["expected_loss_mask_sum"] = float(exp)
    res["loss_mask_sum_ok"] = abs(float(loss_mask.sum()) - exp) < 0.5
    # per-segment loss mass == that conversation's counts x weights
    per_seg = []
    for k, (a, b) in enumerate(zip(cu_l[:-1], cu_l[1:])):
        per_seg.append({"seg": k, "id": convs_meta[k]["id"], "loss_mass": float(loss_mask[a:b].sum()),
                        "expected": float(sum(c * w for c, w in zip(convs_meta[k]["counts"], convs_meta[k]["weights"])))})
    res["per_segment_loss_mass"] = per_seg

    # Mamba: MambaMixer._create_packed_seq_idx (cu_seqlens_q, total_tokens = full sequence after CP undo)
    cu_with_max = torch.cat([cu, torch.tensor([args.seq_len], dtype=cu.dtype)])
    seq_lengths = cu_with_max[1:] - cu_with_max[:-1]
    seq_idx = torch.repeat_interleave(torch.arange(seq_lengths.numel()), seq_lengths)
    changes = (torch.nonzero(seq_idx[1:] != seq_idx[:-1]).flatten() + 1).tolist()
    res["mamba_seq_idx_change_points"] = changes
    res["mamba_seq_idx_changes_match_cu_seqlens"] = changes == cu_l[1:-1]

    # MTP: _roll_tensor_packed_seq, cp_size == 1 path (labels and loss_mask rolled left by 1 inside each segment)
    def roll_packed(t):
        out = t.clone()
        for a, b in zip(cu_l[:-1], cu_l[1:]):
            s = t[..., a:b]
            r = torch.roll(s, shifts=-1, dims=-1)
            r[..., -1:] = 0
            out[..., a:b] = r
        return out
    mtp_labels = roll_packed(labels)
    mtp_mask = roll_packed(loss_mask)
    nz2 = torch.nonzero(mtp_mask > 0).flatten()
    cross2 = [int(p) for p in nz2 if int(p) + 2 >= args.seq_len or int(seg_of[min(int(p) + 2, args.seq_len - 1)]) != int(seg_of[int(p)])]
    res["mtp1_loss_positions"] = int(nz2.numel())
    res["mtp1_loss_positions_with_label_from_other_segment"] = len(cross2)
    res["seconds"] = {"setup": round(t1 - t0, 1), "getitem": round(t2 - t1, 1), "total": round(time.time() - t0, 1)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("per_segment_loss_mass",)}, indent=1))


if __name__ == "__main__":
    main()
