# -*- coding: utf-8 -*-
"""profile_animatediff.py — 真实 AnimateDiff (SD1.5 + MotionAdapter) 的模块级成本分布

背景:
    仓库已有 measure_video_cost2.py（从 T=256 单点外推到 42 分钟，方法不可信）与
    profile_unet.py（SDXL 且硬编码路径）。二者都不能回答"SD1.5+AnimateDiff 上，
    时间到底花在哪"。本脚本直接加载真实模型、真实 5D 输入，用 forward hook 按
    **模块类别**聚合，并把 AnimateDiff 特有的【时序层】单独拆出来。

为什么要单独拆时序层:
    AnimateDiff 相对 SD1.5 新增的就是 MotionAdapter 注入的时序 attention/conv。
    "视频比图像慢多少、慢在哪"这个问题的答案全在这一类里。若不拆出来，它会混在
    普通 attention 里，导致误判（实测时序 3D 卷积只占 ~4%，但那是另一个模型；
    这里要重新测，不能搬）。

判据:
    给出各类别的时间占比排序。占比最高的那一项才是该优化的对象。

oracle:
    · 先用 2D（单帧）与 5D（多帧）各跑一次，确认模型真的接受 5D、且时序层真的被触发。
    · 统计每类模块的调用次数，防止"占比高但其实只被调了一次"这类读错。
    · 同一进程内完成所有测量，避免跨进程时间漂移。
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from diffusers import MotionAdapter, UNet2DConditionModel  # noqa: E402
from diffusers.models import UNetMotionModel  # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', '6')))

UNET_DIR = os.environ.get('UNET_DIR', r'D:\work\textmodel\sd15_base\unet')
ADAPTER_DIR = os.environ.get('ADAPTER_DIR',
                             r'D:\work\textmodel\animatediff-motion-adapter-v1-5-2')
FRAMES = int(os.environ.get('FRAMES', '16'))
RES = int(os.environ.get('RES', '32'))
REPS = int(os.environ.get('REPS', '2'))


def load():
    print('[load] unet=%s' % UNET_DIR, flush=True)
    print('[load] adapter=%s' % ADAPTER_DIR, flush=True)
    u = UNet2DConditionModel.from_pretrained(UNET_DIR, torch_dtype=torch.float32,
                                             local_files_only=True)
    a = MotionAdapter.from_pretrained(ADAPTER_DIR, torch_dtype=torch.float32,
                                      local_files_only=True)
    m = UNetMotionModel.from_unet2d(u, a).to(torch.float32)
    m.eval()
    del u, a
    gc.collect()
    return m


def classify(name, mod):
    """把模块名映射到人类可读的类别，时序层单独成类。"""
    low = name.lower()
    is_temporal = ('temporal' in low) or ('motion' in low)
    cn = type(mod).__name__
    if 'Attention' in cn or 'Transformer2D' in cn:
        if is_temporal:
            return 'TEMPORAL attention'
        return 'spatial attention'
    if 'Conv' in cn or 'Resnet' in cn or 'ResNet' in cn:
        if is_temporal:
            return 'TEMPORAL conv'
        return 'conv / resnet'
    if 'GroupNorm' in cn:
        return 'groupnorm'
    return cn


def main():
    m = load()
    n_param = sum(p.numel() for p in m.parameters())
    print('[load] params=%.1fM  frames=%d res=%d' % (n_param / 1e6, FRAMES, RES), flush=True)

    # 找出时序模块实际叫什么，先确认它们存在（否则下面测的是纯 2D 模型）
    temporal_names = [n for n, _ in m.named_modules() if 'temporal' in n.lower()]
    print('[check] 名字含 temporal 的模块数 = %d' % len(temporal_names), flush=True)
    for n in temporal_names[:6]:
        print('         %s' % n, flush=True)

    def run(frames, res):
        x = torch.randn(1, 4, frames, res, res)
        # encoder_hidden_states 必须按帧展开成 (B*F, 77, D) —— 见 video_models.py 的说明。
        # 只给 (B,77,D) 会报 "size of tensor a (8192) must match b (1024)"，比值 = 帧数。
        ctx = torch.randn(1, 77, 768).repeat(frames, 1, 1)
        t = torch.tensor([500], dtype=torch.long)
        with torch.no_grad():
            _ = m(x, t, encoder_hidden_states=ctx)
        # 计时
        best = float('inf')
        for _ in range(REPS):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = m(x, t, encoder_hidden_states=ctx)
            dt = time.perf_counter() - t0
            if dt < best:
                best = dt
        del x, ctx
        gc.collect()
        return best

    print('\n[1] 2D vs 5D 前向总时间（确认时序层被触发）', flush=True)
    t2d = run(1, RES)
    t5d = run(FRAMES, RES)
    print('  2D  frames=1  res=%d : %8.3f s' % (RES, t2d), flush=True)
    print('  5D  frames=%d res=%d : %8.3f s   (每帧成本 %.3f s，相对 2D %.2fx)'
          % (FRAMES, RES, t5d, t5d / FRAMES, (t5d / FRAMES) / t2d), flush=True)

    # CPU 上按模块计时用 torch.profiler 最可靠，且能给出 self/总 两个口径。
    print('\n[2] torch.profiler 模块级聚合（5D，frames=%d）' % FRAMES, flush=True)
    x = torch.randn(1, 4, FRAMES, RES, RES)
    ctx = torch.randn(1, 77, 768).repeat(FRAMES, 1, 1)
    t = torch.tensor([500], dtype=torch.long)
    try:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            with torch.no_grad():
                _ = m(x, t, encoder_hidden_states=ctx)
        evts = prof.key_averages()
        rows = []
        for e in evts:
            if e.self_device_time_total > 0 or e.self_cpu_time_total > 0:
                rows.append((e.key, e.self_cpu_time_total / 1e3, e.count))
        rows.sort(key=lambda r: -r[1])
        tot = sum(r[1] for r in rows) or 1.0
        print('  %-46s %12s %8s %8s' % ('op', 'self ms', 'count', '%'))
        print('  ' + '-' * 78)
        for k, ms, c in rows[:22]:
            print('  %-46s %12.2f %8d %7.1f%%' % (k[:46], ms, c, 100 * ms / tot))
        print('  %-46s %12.2f' % ('(合计 self)', tot))
    except Exception as e:
        print('  profiler 失败: %s: %s' % (type(e).__name__, e), flush=True)
    finally:
        del x, ctx
        gc.collect()

    print('\n[3] 模型结构概览（各类模块数量）', flush=True)
    counts = defaultdict(int)
    for n, mod in m.named_modules():
        counts[classify(n, mod)] += 1
    for k in sorted(counts, key=lambda k: -counts[k]):
        print('  %-22s %5d' % (k, counts[k]), flush=True)

    print('\nDONE', flush=True)


if __name__ == '__main__':
    main()
