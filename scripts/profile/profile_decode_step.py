# -*- coding: utf-8 -*-
"""profile_decode_step.py — batch=1 逐 token 解码的成本拆解（AVX2 视角）

为什么要测这个：
  · 训练时算术强度 ≈ 4 FLOP/byte（拐点 12.2 已更正为 7.7，见报告 §10.80）
    ⚠️ 2026-09-25 标注：原文接的是「⇒ 算力受限」。但按 §10.80 的拐点 7.7，
       强度 4 < 7.7 落在【带宽受限】那一侧，与该结论方向相反。
       ⇒ 这一行**不能照抄旧结论**，需要重新推一遍（本轮只标数、不改结论）。
  · 逐 token 解码时强度 = 61 MFLOP / 122 MB = 【0.5 FLOP/byte】⇒ 极端带宽受限
  ⇒ 【训练和推理需要不同的架构优化】

本脚本回答：20.11 ms/步 里，多少是「读权重」（带宽），多少是别的？
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import sys, time, math
import torch
import torch.nn

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

torch.set_num_threads(6)
print('=' * 80)
print('batch=1 逐 token 解码的成本拆解（AVX2 视角）')
print('=' * 80)

cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
m = SmallARImageModelV2(cfg)
m.eval()

nparam = sum(p.numel() for p in m.parameters())
nbytes = nparam * 4
print()
print('[0] 模型规模')
print('    参数量      %.2f M' % (nparam / 1e6))
print('    权重字节    %.1f MB (fp32)' % (nbytes / 1e6))
print('    每 token FLOPs (前向)  %.1f MFLOP' % (2 * nparam / 1e6))
print('    ⇒ 算术强度  %.2f FLOP/byte' % (2 * nparam / nbytes))
print('    拐点(实测)  7.7 FLOP/byte   ← 原写 12.2；口径见报告 §10.80')
print('    ⇒ 【%.1fx 低于拐点 ⇒ 极端带宽受限】' % (7.7 / (2 * nparam / nbytes)))

# ---- 纯读权重的理论时间 ----
# ★★ 2026-09-25 修正：BW 原写 18.8e9，那是**已被 §10.79 撤回**的口径。
#   这一处【不是显示字符串，是计算的一部分】：L50 用它打印「理论下限」，
#   而本脚本的全部结论就是"实测 20.11 ms/步 里，多少是读权重、多少是别的" ——
#   下游全部拿这个下限做参照。用 18.8 会把下限算成 6.49 ms（偏高 59%），
#   于是"读权重占多少"被系统性高估。改用 §10.80 的权威值 29.8 GB/s ⇒ 下限 4.09 ms。
BW = 29.8e9
print()
print('[1] 理论下限')
print('    读一遍权重 @ %.1f GB/s = %.2f ms' % (BW / 1e9, nbytes / BW * 1e3))

# ---- 实测单个 decode step ----
print()
print('[2] 实测单步（cache 里已有 t 个 token）')


def decode_step(t):
    cache = {}
    for i in range(cfg.n_layer):
        k = torch.randn(1, cfg.n_head, t, cfg.d_model // cfg.n_head)
        v = torch.randn(1, cfg.n_head, t, cfg.d_model // cfg.n_head)
        cache[i] = dict(k=k, v=v, n=t, cap=t + 2)
    x = torch.randn(1, 1, cfg.d_model)
    with torch.no_grad():
        for _ in range(3):
            _ = m.blocks[0](x, pooled=None, ctx=None, cache=dict(cache[0]), append=False)
        t0 = time.perf_counter()
        N = 10
        for _ in range(N):
            for i in range(cfg.n_layer):
                x2 = m.blocks[i](x, pooled=None, ctx=None, cache=dict(cache[i]), append=False)
        return (time.perf_counter() - t0) / N


print('    %-14s %14s %14s' % ('已生成长度 t', 'ms/步', '相对 t=1'))
print('    ' + '-' * 46)
base = None
for t in (1, 16, 64, 256, 1024, 4096):
    d = decode_step(t)
    if base is None:
        base = d
    print('    %-14d %14.2f %13.2fx' % (t, d * 1e3, d / base))
print('    ⇒ 若随 t 增长，说明 attention 部分在长上下文下占主导')

# ---- 拆解：权重读取 vs attention ----
print()
print('[3] 拆解单步成本（t=256）')
t = 256
# a) 只做权重读取（空转一次所有 Linear 的 matvec）
ms = [mod for mod in m.modules() if isinstance(mod, nn.Linear)]
print('    Linear 层数 %d，权重合计 %.1f MB' % (len(ms), sum(x.weight.numel() for x in ms) * 4 / 1e6))
x1 = torch.randn(1, cfg.d_model)
with torch.no_grad():
    for _ in range(3):
        for L in ms:
            if L.in_features == cfg.d_model:
                _ = L(x1)
    t0 = time.perf_counter()
    for _ in range(10):
        for L in ms:
            if L.in_features == cfg.d_model:
                _ = L(x1)
    t_lin = (time.perf_counter() - t0) / 10
print('    单次遍历全部 d->* Linear: %.2f ms' % (t_lin * 1e3))

# b) 只做 attention（t=256）
from scripts.lib.small_image_model_v2 import SelfAttention
attn = SelfAttention(cfg)
attn.eval()
cache = dict(k=torch.randn(1, cfg.n_head, t, cfg.d_model // cfg.n_head),
             v=torch.randn(1, cfg.n_head, t, cfg.d_model // cfg.n_head), n=t, cap=t + 2)
xa = torch.randn(1, 1, cfg.d_model)
with torch.no_grad():
    for _ in range(3):
        _ = attn(xa, cache=dict(cache), append=False)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = attn(xa, cache=dict(cache), append=False)
    t_att = (time.perf_counter() - t0) / 20
print('    单层 attention (t=256): %.3f ms  ×%d 层 = %.2f ms' % (t_att * 1e3, cfg.n_layer, t_att * cfg.n_layer * 1e3))

print()
print('    ⇒ 权重读取 %.2f ms + attention %.2f ms = %.2f ms' % (
    t_lin * 1e3, t_att * cfg.n_layer * 1e3, t_lin * 1e3 + t_att * cfg.n_layer * 1e3))

# ---- 关键结论 ----
print()
print('[4] AVX2 视角的结论')
nbytes_mb = nbytes / 1e6
print('    每 token 必须读 %.0f MB 权重（fp32）' % nbytes_mb)
print('    ⇒ 读权重时间 %.2f ms（理论下限）' % (nbytes / BW * 1e3))
print('    ⇒ 实测 %.2f ms/步（t=256）' % (decode_step(256) * 1e3))
print()
print('    若权重降到 8-bit: %.0f MB ⇒ 读权重 %.2f ms（-%.0f%%）'
      % (nbytes_mb / 4, nbytes / 4 / BW * 1e3, 75))
print('    若用循环 L=2（权重读一次用 2 次）: 强度 ×2')
print('    若用 MoE top-1/4: 读 %.0f MB ⇒ 强度 ×4' % (nbytes_mb / 4))
print()
print('=' * 80)
print('DONE')
