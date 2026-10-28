# -*- coding: utf-8 -*-
"""measure_video_cost.py — 实测 AR 视频路线的成本

拆解：
  GEMM 部分:      O(T)  每步读全部权重（与已生成长度无关）
  attention 部分: O(T²) 逐 token 累积，第 t 步 attend t 个历史 token

用真实模型结构测 attention 的 T 缩放，再外推到视频长度。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys, time, math
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SelfAttention

torch.set_num_threads(6)
print('=' * 78)
print('AR 视频路线的成本实测')
print('=' * 78)
print()

cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
D, NH, NKV, NL, HD = cfg.d_model, cfg.n_head, cfg.n_kv_head, cfg.n_layer, cfg.d_model // cfg.n_head
print('模型: d_model=%d n_head=%d n_kv_head=%d n_layer=%d head_dim=%d' % (D, NH, NKV, NL, HD))
print()

print('[1] SelfAttention 全序列前向（单层，batch=1）')
print('    %-10s %14s %14s %16s' % ('T', 'ms', '相对 T=256', 'T^2 归一'))
print('    ' + '-' * 56)
attn = SelfAttention(cfg)
attn.eval()
base = None
for T in (256, 512, 1024, 2048):
    x = torch.randn(1, T, D)
    with torch.no_grad():
        for _ in range(1):
            _ = attn(x, cache=None, append=True)
        t0 = time.perf_counter()
        for _ in range(2):
            _ = attn(x, cache=None, append=True)
        t = (time.perf_counter() - t0) / 2
    if base is None:
        base = t
    print('    %-10d %14.2f %13.2fx %15.2fx' % (T, t * 1e3, t / base, (T / 256) ** 2))
print('    ⇒ 「相对」≈「T^2 归一」则 attention 是 O(T²)')

print()
print('[2] 逐 token 解码：单步 attention 成本随「已生成长度 t」的增长')
print('    %-12s %18s' % ('已生成长度 t', '单步 ms'))
print('    ' + '-' * 32)
attn2 = SelfAttention(cfg)
attn2.eval()


def step_cost(t):
    """模拟：cache 里已有 t 个 token，来一个新的"""
    cache = {'k': torch.randn(1, NKV, t, HD), 'v': torch.randn(1, NKV, t, HD),
             'n': t, 'cap': t + 2}
    x = torch.randn(1, 1, D)
    with torch.no_grad():
        for _ in range(2):
            _ = attn2(x, cache=dict(cache), append=False)
        t0 = time.perf_counter()
        for _ in range(5):
            _ = attn2(x, cache=dict(cache), append=False)
        return (time.perf_counter() - t0) / 5


per = {}
for t in (64, 256, 1024, 4096, 16384, 32768):
    per[t] = step_cost(t)
    print('    %-12d %18.4f' % (t, per[t] * 1e3))

ts = sorted(per)
cs = [per[t] / t for t in ts]
c_avg = sum(cs) / len(cs)
print()
print('    线性拟合 c = 单步成本/t: %s' % ['%.3e' % c for c in cs])
print('    ⇒ 平均 c = %.4e 秒/token（每层）' % c_avg)

print()
print('[3] 外推到视频（%d 层合计）' % NL)
# 基准：报告实测 batch=1, T=256 时 20.11 ms/步
B1_STEP_MS = 20.11
GEMM_SHARE = 0.612
ATTN_SHARE = 0.137
gemm_step = B1_STEP_MS * GEMM_SHARE * NL      # 每步 GEMM（与 T 无关）
attn_step_avg = B1_STEP_MS * ATTN_SHARE * NL  # 每步 attention（T=256 时平均长度 128）
# 用实测的 c_avg 交叉验证
c_meas = c_avg * NL
print('    基准（报告实测）: batch=1, T=256 ⇒ %.2f ms/步' % B1_STEP_MS)
print('      GEMM 部分    %.2f%% ⇒ %.3f ms/步（%d 层）' % (GEMM_SHARE * 100, gemm_step, NL))
print('      attention    %.2f%% ⇒ %.3f ms/步（平均长度 128，%d 层）' % (ATTN_SHARE * 100, attn_step_avg, NL))
print('      实测 c 交叉验证: %.3f ms/步（平均长度 128 时）⇒ 与上面 %.3f 同量级 %s'
      % (c_meas * 128 * 1e3, attn_step_avg,
         'OK' if 0.2 < (c_meas * 128 * 1e3) / attn_step_avg < 5 else '差异大'))
print()

c_attn = (attn_step_avg / 1e3) / 128.0      # 秒/token/层组

print('    %-32s %12s %14s %14s %12s' % ('场景', 'token', 'attention', 'GEMM', '合计'))
print('    ' + '-' * 90)
scenarios = [
    ('图像 256×256 (T=256)', 256),
    ('视频 256×144 2s@30fps', 2 * 30 * 144),
    ('视频 256×144 5s@30fps', 5 * 30 * 144),
    ('视频 256×144 15s@30fps', 15 * 30 * 144),
    ('视频 256×256 15s@30fps', 15 * 30 * 256),
]
for name, T in scenarios:
    gemm_s = gemm_step * T / 1e3
    attn_s = c_attn * (T * (T + 1) / 2)
    tot = gemm_s + attn_s
    def fmt(s):
        if s < 90:
            return '%.1f 秒' % s
        if s < 5400:
            return '%.1f 分' % (s / 60)
        return '%.1f 小时' % (s / 3600)
    print('    %-32s %12d %14s %14s %12s' % (name, T, fmt(attn_s), fmt(gemm_s), fmt(tot)))

print()
print('[4] 换线性注意力（GDN，O(1)/token）')
print('    GDN 报告实测: 单层 fwd+bwd T=1024 B=2 H=4 d=64 ⇒ 51.6 ms')
print('    ⇒ 推理 fwd 约 1/3 = 17.2 ms / (T=1024, B=2)')
print('    ⇒ 单样本 T=1024: 约 8.6 ms/层  ⇒  %.4f ms/token/层' % (8.6 / 1024))
c_gdn = (8.6 / 1024) * 1e-3
print()
print('    %-32s %12s %14s %14s %12s' % ('场景', 'token', 'attention', 'GEMM', '合计'))
print('    ' + '-' * 90)
for name, T in scenarios:
    gemm_s = gemm_step * T / 1e3
    attn_s = c_gdn * T * NL
    tot = gemm_s + attn_s
    def fmt(s):
        if s < 90:
            return '%.1f 秒' % s
        if s < 5400:
            return '%.1f 分' % (s / 60)
        return '%.1f 小时' % (s / 3600)
    print('    %-32s %12d %14s %14s %12s' % (name, T, fmt(attn_s), fmt(gemm_s), fmt(tot)))

print()
print('[5] 对比：速度倍数（video 15s@30fps 256×144 = 64800 token）')
T = 64800
gemm_s = gemm_step * T / 1e3
attn_naive = c_attn * (T * (T + 1) / 2)
attn_gdn = c_gdn * T * NL
print('    朴素 attention: %.1f 小时' % (attn_naive / 3600))
print('    GDN 线性注意力: %.1f 秒' % attn_gdn)
print('    ⇒ 加速 %.0fx' % (attn_naive / attn_gdn))
print()
print('=' * 78)
print('DONE')
