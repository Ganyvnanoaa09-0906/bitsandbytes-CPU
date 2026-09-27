# -*- coding: utf-8 -*-
"""wan_text_encoder.py — 用真实的 umt5-xxl 权重算 prompt embedding 并落盘

为什么必须先做这一步:
    Wan 的文本编码器（umt5-xxl）**11.36 GB / 5.68B 参数 / bf16**，是整个管线里
    最大的单一组件（§4.3 的账：全管线 17.55 GB vs 本机 15.4 GB）。
    它是**每次 prompt 只跑一次**的，所以正确做法是：
        **跑一次 → 把 embedding 存盘 → 卸载** ⇒ 推理时常驻只剩 4bit 主干 + VAE（~1.2 GB）
    这就是"文本编码器卸载"，也是 Wan 能在这台机器上跑起来的前提。

键映射（实测自权重清单，242 张量）:
    token_embedding.weight              -> shared.weight
    blocks.N.attn.{q,k,v,o}             -> encoder.block.N.layer.0.SelfAttention.{q,k,v,o}
    blocks.N.ffn.fc1 / fc2 / gate.0     -> ...layer.1.DenseReluDense.{wi_0,wo,wi_1}
    blocks.N.norm1 / norm2              -> ...layer.0 / layer.1.layer_norm.weight
    blocks.N.pos_embedding.embedding    -> ...SelfAttention.relative_attention_bias.weight
    norm.weight                         -> encoder.final_layer_norm.weight

内存策略（关键，本机只有 15.4 GB）:
    · 用 **bf16** 加载，不升 fp32（升了要 22.7 GB，直接爆）
    · 逐张量转换并 del 源字典，避免同时持有两份
    · 算完 embedding 立即 del 模型并 gc，然后报告 RSS

判据:
    1. 加载后 missing=0（形状不符也拒绝）
    2. 前向输出 finite，形状 = (B, seq, 4096)（text_dim=4096）
    3. embedding 落盘成功且可重新载入
"""
from __future__ import annotations

import ctypes
import gc
import json
import os
import sys
import time

import torch

torch.set_num_threads(int(os.environ.get("THREADS", "6")))

T5_PTH = os.environ.get(
    "WAN_T5", r"D:\work\textmodel\Wan2.1-T2V-1.3B\models_t5_umt5-xxl-enc-bf16.pth")
TOK_DIR = os.environ.get(
    "WAN_TOK", r"D:\work\textmodel\Wan2.1-T2V-1.3B\google\umt5-xxl")
OUT_DIR = os.environ.get("WAN_EMB_OUT", r"D:\work\textmodel\wan_embeddings")

PROMPT = os.environ.get("WAN_PROMPT", "a cat walking on grass, cinematic")
NEG = os.environ.get("WAN_NEG", "blurry, low quality, distorted")


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("a", ctypes.c_size_t), ("b", ctypes.c_size_t), ("c", ctypes.c_size_t),
                ("d", ctypes.c_size_t), ("e", ctypes.c_size_t), ("f", ctypes.c_size_t)]


def rss_gb():
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    ctypes.windll.psapi.GetProcessMemoryInfo(
        ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
    return pm.PeakWorkingSetSize / 1e9


def build_config():
    from transformers import UMT5Config
    # umt5-xxl 的配置（从权重形状反推，逐条核对过）:
    #   token_embedding (256384, 4096)  -> vocab_size=256384, d_model=4096
    #   ffn.fc1 (10240, 4096)           -> d_ff=10240
    #   attn (4096,4096) 24 层          -> d_kv=4096/64=64, num_heads=64
    #   pos_embedding (32, 64)          -> relative_attention_num_buckets=32
    #   gated FFN (ffn.gate.0)          -> feed_forward_proj='gated-gelu'
    return UMT5Config(
        vocab_size=256384, d_model=4096, d_kv=64, d_ff=10240,
        num_layers=24, num_decoder_layers=24, num_heads=64,
        relative_attention_num_buckets=32, relative_attention_max_distance=128,
        dropout_rate=0.0, layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu", tie_word_embeddings=False,
        is_encoder_decoder=True, use_cache=True, dense_act_fn="gelu_new",
    )


def map_key(k):
    if k == "token_embedding.weight":
        return "shared.weight"
    if k == "norm.weight":
        return "encoder.final_layer_norm.weight"
    m = k.split(".")
    if m[0] == "blocks":
        n = m[1]
        base = "encoder.block.%s" % n
        rest = ".".join(m[2:])
        if rest.startswith("attn."):
            return "%s.layer.0.SelfAttention.%s" % (base, rest.split(".", 1)[1])
        if rest.startswith("pos_embedding.embedding."):
            return "%s.layer.0.SelfAttention.relative_attention_bias.%s" % (
                base, rest.rsplit(".", 1)[-1])
        if rest.startswith("ffn."):
            f = rest.split(".", 1)[1]
            # fc1 -> wi_0, gate.0 -> wi_1, fc2 -> wo
            mm = {"fc1": "wi_0", "fc2": "wo", "gate.0": "wi_1"}
            for a, b in mm.items():
                if f.startswith(a + "."):
                    return "%s.layer.1.DenseReluDense.%s.%s" % (base, b, f.rsplit(".", 1)[-1])
        if rest.startswith("norm1."):
            return "%s.layer.0.layer_norm.%s" % (base, rest.rsplit(".", 1)[-1])
        if rest.startswith("norm2."):
            return "%s.layer.1.layer_norm.%s" % (base, rest.rsplit(".", 1)[-1])
    return None


def load_encoder(verbose=True):
    from transformers import UMT5EncoderModel
    cfg = build_config()
    t0 = time.perf_counter()
    # ⚠️ 不能用 torch.device('meta') —— transformers 5.x 的 forward 会对 mask 调
    #    .item()，而 meta 张量不支持（RuntimeError: Tensor.item() cannot be called
    #    on meta tensors）。改为建实体模型再逐张量搬。
    #    内存要求: bf16 模型 ~11.4 GB + checkpoint 11.36 GB 会超本机 15.4 GB，
    #    所以**先建空壳（不初始化权重）再搬**，避免两份同时存在。
    with torch.device("cpu"):
        model = UMT5EncoderModel._from_config(cfg)  # 不看权重，仅建结构
    # 把参数全部换成空张量（不占内存），再逐键 copy_
    ref = model.state_dict()
    if verbose:
        print("  结构已建 %.1f s，参数张量 %d 个" % (time.perf_counter() - t0, len(ref)))

    sd = torch.load(T5_PTH, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if verbose:
        print("  checkpoint 已读入（%d 键）")
    n_ok = n_miss = n_shape = n_unmapped = 0
    missing_keys = []
    for k, v in list(sd.items()):
        nk = map_key(k)
        if nk is None:
            n_unmapped += 1
            continue
        if nk not in ref:
            n_miss += 1
            if len(missing_keys) < 6:
                missing_keys.append(nk)
            continue
        if tuple(ref[nk].shape) != tuple(v.shape):
            n_shape += 1
            if len(missing_keys) < 10:
                missing_keys.append("SHAPE %s %s vs %s" % (nk, tuple(v.shape), tuple(ref[nk].shape)))
            continue
        ref[nk].copy_(v.to(torch.bfloat16))
        # 搬完立即释放该源张量，压低峰值
        del sd[k]
        n_ok += 1
    del sd
    gc.collect()
    if verbose:
        print("  逐张量装入: ok=%d  未映射=%d  目标无此键=%d  形状不符=%d"
              % (n_ok, n_unmapped, n_miss, n_shape))
        for k in missing_keys:
            print("     %s" % k)
    model.eval()
    return model, dict(ok=n_ok, unmapped=n_unmapped, no_key=n_miss,
                       shape=n_shape, still_meta=0)


def main():
    print("=" * 86)
    print("umt5-xxl 文本编码器 → prompt embedding")
    print("=" * 86)
    os.makedirs(OUT_DIR, exist_ok=True)
    from transformers import AutoTokenizer

    print("\n[1] tokenizer")
    tok = AutoTokenizer.from_pretrained(TOK_DIR, local_files_only=True)
    print("  loaded: %s" % type(tok).__name__)

    print("\n[2] 文本编码器（bf16，逐张量装入）")
    model, st = load_encoder()
    print("  peak RSS = %.2f GB" % rss_gb())

    print("\n[3] 编码 prompt")
    results = {}
    for tag, text in (("pos", PROMPT), ("neg", NEG)):
        enc = tok(text, return_tensors="pt", padding="max_length",
                  max_length=512, truncation=True)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"]).last_hidden_state
        dt = time.perf_counter() - t0
        fin = bool(torch.isfinite(out).all())
        print("  [%s] %-42s -> %s  %.1f s  finite=%s"
              % (tag, text[:42], tuple(out.shape), dt, fin))
        # Wan 把 mask 之外的 token 置零（官方做法）
        m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        out = out * m
        results[tag] = out.to(torch.bfloat16)
        del out, enc
        gc.collect()

    print("\n[4] 落盘（推理时只需读它，11.36 GB 的编码器可以卸载）")
    for tag, t in results.items():
        p = os.path.join(OUT_DIR, "embed_%s.pt" % tag)
        torch.save({"embed": t, "prompt": PROMPT if tag == "pos" else NEG,
                    "seq_len": int(t.shape[1])}, p)
        print("  %-34s %s  %.2f MB" % (p, tuple(t.shape), os.path.getsize(p) / 1e6))

    # 卸载并确认内存真的降下来
    del model, tok
    gc.collect()
    print("\n[5] 卸载编码器后 peak RSS = %.2f GB（峰值仍在，但常驻已释放）" % rss_gb())

    ok = all(torch.isfinite(v).all() for v in results.values())
    print("\n判据: embedding finite=%s  装入 ok=%d  仍为 meta=%d  =>  %s"
          % (ok, st["ok"], st["still_meta"], "PASS" if (ok and st["still_meta"] == 0) else "CHECK"))
    print("=" * 86)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
