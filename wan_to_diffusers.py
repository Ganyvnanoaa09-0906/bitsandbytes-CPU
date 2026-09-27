# -*- coding: utf-8 -*-
"""wan_to_diffusers.py — 把 Wan2.1-T2V-1.3B 的原始权重装进 diffusers 的 WanTransformer3DModel

为什么需要:
    diffusers 0.40.0 **原生支持 Wan**（`transformer_wan.py` / `autoencoder_kl_wan.py`），
    所以不必自己写架构 —— 但本机这份权重是 **Wan GitHub 的原始格式**
    （`blocks.N.self_attn.q.*`），diffusers 用的是另一套命名
    （`blocks.N.attn1.to_q.*`）。两边一一对应，可以精确映射。

映射（已核对每一条）:
    self_attn.{q,k,v,o}   ->  attn1.to_{q,k,v,out.0}
    self_attn.norm_{q,k}  ->  attn1.norm_{q,k}
    cross_attn.{q,k,v,o}  ->  attn2.to_{q,k,v,out.0}
    cross_attn.norm_{q,k} ->  attn2.norm_{q,k}
    ffn.0 / ffn.2         ->  ffn.net.0.proj / ffn.net.2
    norm3                 ->  norm2      （每块 2 个 norm：norm2 给 cross-attn、norm3 给 FFN）
    modulation            ->  scale_shift_table
    time_projection.1     ->  scale_shift_table 派生（9216 = 6 × 1536，**丢弃**）
    patch_embedding / text_embedding / time_embedding / head / head.modulation -> 同名

1.3B 的配置对上了:
    num_attention_heads=12, attention_head_dim=128  ⇒ 12×128 = 1536 = dim ✓
    （若照默认 40×128 = 5120，加载必然失败 —— 所以这几个数必须显式给）
    ffn_dim=8960, text_dim=4096, freq_dim=256, num_layers=30,
    patch_size=(1,2,2), in/out_channels=16, qk_norm='rms_norm_across_heads'

判据:
    加载后 missing=0 且 unexpected=0（除刻意丢弃的 time_projection.1），
    并**跑一次真实前向**确认输出 finite、形状正确。
    只 load 不 forward 不算通过 —— 形状对但语义错是很可能的。
"""
from __future__ import annotations

import json
import os
import struct
import sys

import torch

SRC = os.environ.get(
    "WAN_WEIGHTS",
    r"D:\work\textmodel\Wan2.1-T2V-1.3B\diffusion_pytorch_model.safetensors")

CONFIG_1_3B = dict(
    patch_size=(1, 2, 2),
    num_attention_heads=12,
    attention_head_dim=128,
    in_channels=16,
    out_channels=16,
    text_dim=4096,
    freq_dim=256,
    ffn_dim=8960,
    num_layers=30,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=1024,
)

# 顶层参数的映射（实测自 diffusers 0.40.0 的 state_dict 对比，不是猜的）
TOP_MAP = {
    # 时间嵌入：两段 MLP -> condition_embedder.time_embedder.linear_{1,2}
    "time_embedding.0.weight": "condition_embedder.time_embedder.linear_1.weight",
    "time_embedding.0.bias": "condition_embedder.time_embedder.linear_1.bias",
    "time_embedding.2.weight": "condition_embedder.time_embedder.linear_2.weight",
    "time_embedding.2.bias": "condition_embedder.time_embedder.linear_2.bias",
    # 文本嵌入：两段 MLP -> condition_embedder.text_embedder.linear_{1,2}
    "text_embedding.0.weight": "condition_embedder.text_embedder.linear_1.weight",
    "text_embedding.0.bias": "condition_embedder.text_embedder.linear_1.bias",
    "text_embedding.2.weight": "condition_embedder.text_embedder.linear_2.weight",
    "text_embedding.2.bias": "condition_embedder.text_embedder.linear_2.bias",
    # ⚠️ 这一对**不能丢**：diffusers 里叫 condition_embedder.time_proj，
    #    形状 (9216,1536) 与原始 time_projection.1 完全一致。
    #    （我第一版把它当"多余"丢弃，导致 missing 里出现 time_proj。）
    "time_projection.1.weight": "condition_embedder.time_proj.weight",
    "time_projection.1.bias": "condition_embedder.time_proj.bias",
    # 输出头
    "head.head.weight": "proj_out.weight",
    "head.head.bias": "proj_out.bias",
    "head.modulation": "scale_shift_table",
    # patch_embedding 同名，无需映射
}


def read_safetensors(path):
    """返回 {key: tensor}（只读头部 + 按需读数据）。"""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
        base = 8 + n
        meta = {k: v for k, v in hdr.items() if k != "__metadata__"}
        out = {}
        for k, v in meta.items():
            f.seek(base + v["data_offsets"][0])
            raw = f.read(v["data_offsets"][1] - v["data_offsets"][0])
            dt = {"F32": torch.float32, "F16": torch.float16,
                  "BF16": torch.bfloat16}[v["dtype"]]
            out[k] = torch.frombuffer(bytearray(raw), dtype=dt).reshape(v["shape"]).clone()
    return out


def map_key(k):
    """原始 key -> diffusers key。返回 None 表示丢弃（本版本不丢任何东西）。"""
    if k in TOP_MAP:
        return TOP_MAP[k]
    if not k.startswith("blocks."):
        return k                       # patch_embedding 等同名
    parts = k.split(".")
    idx = parts[1]
    rest = ".".join(parts[2:])
    # ⚠️ 这里必须拼上 "blocks.<idx>." 前缀。第一版漏了，于是 30 个 modulation
    #    原样返回、变成 unexpected，而 scale_shift_table 全部 missing。
    prefix = "blocks.%s." % idx
    m = {
        "self_attn.q": "attn1.to_q", "self_attn.k": "attn1.to_k",
        "self_attn.v": "attn1.to_v", "self_attn.o": "attn1.to_out.0",
        "self_attn.norm_q": "attn1.norm_q", "self_attn.norm_k": "attn1.norm_k",
        "cross_attn.q": "attn2.to_q", "cross_attn.k": "attn2.to_k",
        "cross_attn.v": "attn2.to_v", "cross_attn.o": "attn2.to_out.0",
        "cross_attn.norm_q": "attn2.norm_q", "cross_attn.norm_k": "attn2.norm_k",
        "ffn.0": "ffn.net.0.proj", "ffn.2": "ffn.net.2",
        "norm3": "norm2",
        "modulation": "scale_shift_table",
    }
    for src, dst in m.items():
        if rest == src or rest.startswith(src + "."):
            return prefix + dst + rest[len(src):]
    return k                            # 未知的交给 strict 检查暴露


def build_and_load(verbose=True):
    from diffusers import WanTransformer3DModel
    raw = read_safetensors(SRC)
    if verbose:
        print("[1] 读入 %d 个张量" % len(raw))
    sd = {}
    dropped = []
    unmapped = []
    for k, v in raw.items():
        nk = map_key(k)
        if nk is None:
            dropped.append(k)
            continue
        if nk == k and k.startswith("blocks."):
            unmapped.append(k)
        sd[nk] = v
    if verbose:
        print("[2] 映射完成: %d 个 -> state_dict；丢弃 %d 个（%s）；未识别 %d 个"
              % (len(sd), len(dropped), ",".join(dropped) or "-", len(unmapped)))
        if unmapped:
            for k in unmapped[:6]:
                print("     未识别: %s" % k)

    m = WanTransformer3DModel(**CONFIG_1_3B)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if verbose:
        print("[3] 加载: missing=%d unexpected=%d" % (len(missing), len(unexpected)))
        for k in list(missing)[:6]:
            print("     missing   : %s" % k)
        for k in list(unexpected)[:6]:
            print("     unexpected: %s" % k)
    m.eval()
    return m, missing, unexpected


def forward_check(m, verbose=True):
    """真实前向：形状 + finite。只 load 不 forward 不算通过。"""
    B, F, H, W = 1, 5, 16, 16          # latent (F 会被 patch_size[0]=1 切成 F' 段)
    x = torch.randn(B, 16, F, H, W)
    ctx = torch.randn(B, 32, 4096)     # text embedding: (B, seq, text_dim)
    t = torch.tensor([500.0])
    with torch.no_grad():
        y = m(hidden_states=x, encoder_hidden_states=ctx, timestep=t,
              return_dict=False)[0]
    ok = bool(torch.isfinite(y).all())
    if verbose:
        print("[4] 前向: 输出 %s  finite=%s  %s"
              % (tuple(y.shape), ok, "OK" if ok else "FAIL"))
    return y, ok


if __name__ == "__main__":
    print("=" * 84)
    print("Wan2.1-T2V-1.3B 原始权重 -> diffusers WanTransformer3DModel")
    print("=" * 84)
    print("源: %s" % SRC)
    m, missing, unexpected = build_and_load()
    npar = sum(p.numel() for p in m.parameters())
    print("     模型参数 = %.1fM" % (npar / 1e6))
    try:
        y, ok = forward_check(m)
    except Exception as e:
        print("[4] 前向失败: %s: %s" % (type(e).__name__, e))
        sys.exit(1)
    good = (len(missing) == 0 and len(unexpected) == 0 and ok)
    print("\n判据: missing=0 且 unexpected=0 且前向 finite  =>  %s"
          % ("PASS" if good else "FAIL"))
    print("=" * 84)
    sys.exit(0 if good else 1)
