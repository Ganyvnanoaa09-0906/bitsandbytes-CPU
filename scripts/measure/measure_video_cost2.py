# -*- coding: utf-8 -*-
"""measure_video_cost2.py — 测大 T 下的 attention 与内存，外推到视频

关键问题：T=64,800 时 attention 到底多久、要多少内存？
  · 若 PyTorch SDPA 走 flash 路径 ⇒ 内存 O(T)，时间仍 O(T²)
  · 若 materialize T×T 矩阵 ⇒ 内存爆炸（64800²×8head×4B = 134 GB）

同时测：单层完整 Block（attention + FFN）的 T 缩放
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

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SelfAttention, BlockV2

torch.set_num_threads(6)
print('=' * 80)
print('大 T 下的 attention 成本与内存')
print('=' * 80)

cfg = SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=256, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
D, NL, HD, NKV = cfg.d_model, cfg.n_layer, cfg.d_model // cfg.n_head, cfg.n_kv_head
attn = SelfAttention(cfg)
attn.eval()

print()
print('[1] SelfAttention 全序列前向（单层, batch=1）')
print('    %-10s %12s %12s %14s %14s %14s' % ('T', 'ms', '相对', 'ms/T²(nexp)', 'RSS 增量', 'T² 预期'))
print('    ' + '-' * 78)
res = {}
base_t = base_m = None
for T in (256, 512, 1024, 2048, 4096, 8192, 16384):
    x = torch.randn(1, T, D)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        import gc, ctypes

        class PMC(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t),
                        ('d', ctypes.c_size_t), ('e', ctypes.c_size_t), ('f', ctypes.c_size_t)]
        pm = PMC(); pm.cb = ctypes.sizeof(pm)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
        rss0 = pm.WorkingSetSize
    except Exception:
        rss0 = 0
    with torch.no_grad():
        for _ in range(1):
            _ = attn(x, cache=None, append=True)
        t0 = time.perf_counter()
        reps = 1 if T >= 8192 else 2
        for _ in range(reps):
            _ = attn(x, cache=None, append=True)
        t = (time.perf_counter() - t0) / reps
    try:
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pm), pm.cb)
        rss1 = pm.WorkingSetSize
    except Exception:
        rss1 = 0
    if base_t is None:
        base_t, base_m = t, (rss1 - rss0)
    res[T] = (t, rss1 - rss0)
    print('    %-10d %12.2f %11.2fx %13.3e %12.1f MB %13.2fx'
          % (T, t * 1e3, t / base_t, t / (T * T), (rss1 - rss0) / 1e6, (T / 256) ** 2))

print()
print('[2] 单位成本随 T 的变化（判断是否趋于线性于 T²）')
print('    %-10s %18s %14s' % ('T', 'ns per T² unit', '相对 T=256'))
print('    ' + '-' * 44)
u0 = None
for T in sorted(res):
    t, _ = res[T]
    u = t / (T * T)
    if u0 is None:
        u0 = u
    print('    %-10d %18.4f %13.2fx' % (T, u * 1e9, u / u0))

print()
print('[3] 外推到视频（用最大 T 实测的单位成本，最保守）')
Tmax = max(res)
u_max = res[Tmax][0] / (Tmax * Tmax)
print('    取自 T=%d 的单位成本: %.4f ns per T² unit' % (Tmax, u_max * 1e9))
print()
print('    %-30s %10s %14s %14s %12s' % ('场景', 'token', 'attention(单层)', f'×{NL}层', '内存峰值'))
print('    ' + '-' * 84)
for name, T in [('图像 256×256', 256), ('视频 144p 2s@30fps', 2 * 30 * 144),
                ('视频 144p 5s@30fps', 5 * 30 * 144), ('视频 144p 15s@30fps', 15 * 30 * 144),
                ('视频 256p 15s@30fps', 15 * 30 * 256)]:
    a1 = u_max * T * T
    tot = a1 * NL
    # flash 路径内存 O(T·chunk)，否则 O(T²)
    mem_t2 = T * T * 8 * 4 / 1e9          # materialize (B,NH,T,T) fp32
    mem_flash = 8 * 1024 * T * 4 / 1e9    # chunk=1024 的 flash 峰值
    def fmt(s):
        if s < 90:
            return '%.2f 秒' % s
        if s < 5400:
            return '%.1f 分' % (s / 60)
        return '%.1f 小时' % (s / 3600)
    print('    %-30s %10d %14s %14s %12s' % (name, T, fmt(a1), fmt(tot),
                                             '%.1f GB' % min(mem_t2, mem_flash) if T > 2048 else '-'))

print()
print('[4] 若换线性注意力（O(T)，GDN 已有内核）')
# GDN 报告实测: 单层 fwd+bwd T=1024 B=2 H=4 d=64 = 51.6 ms ⇒ 推理 fwd ~17.2ms/(T=1024,B=2)
# ⇒ 单样本 T=1024 单层 ~8.6 ms
gdn_per_tok = (8.6 / 1024) / 1e3    # 秒/token/层
for name, T in [('视频 144p 15s@30fps', 15 * 30 * 144), ('视频 256p 15s@30fps', 15 * 30 * 256)]:
    t_gdn = gdn_per_tok * T * NL
    t_naive = u_max * T * T * NL
    print('    %-26s GDN %8.2f 秒    vs   朴素 %10s   ⇒ %.0fx'
          % (name, t_gdn, ('%.1f 小时' % (t_naive / 3600)) if t_naive > 5400 else ('%.1f 分' % (t_naive / 60)), t_naive / t_gdn))

print()
print('=' * 80)
print('DONE')
