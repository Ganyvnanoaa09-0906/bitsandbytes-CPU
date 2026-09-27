# -*- coding: utf-8 -*-
"""adiff_firstcall_ab.py — conv 的 22940 ms 里，有多少是 oneDNN 一次性开销？

线索来源（report §10.153 待写）:
    模型内 98 次 conv 共 22940 ms，而**单独**跑同形状 conv 只需 17-27 ms。
    且 conv_ceiling_ab.py 里同一形状在对照块读 24.518 ms、在 best-of-5 读 18.968 ms
    —— 首次明显更慢。若这个大差值是 oneDNN 原语创建的**固定开销**，那它是可回收的
    （无精度风险），这也是 97 秒里唯一还没被解释的部分。

判据:
    同一输入连跑两次，比较：
      · 整体时间  t2/t1
      · conv 总时间 conv2/conv1
    若 conv2 << conv1 ⇒ 大头是**一次性**开销，可用"预热/缓存"消掉。
    若 conv2 ≈ conv1 ⇒ conv 时间是真实算术，别再往这方向投入。

oracle:
    · 用同一份输入张量（不要重新 randn，避免把分配算进去）；
    · conv 时间用 torch.profiler 的 self 时间单独取，不看整体；
    · 只跑 2 次（第二次就是"热"），并对两次都用 profiler 以保持口径一致。
"""
from __future__ import annotations

import gc
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from video_models import load_animatediff, video_input, forward_once  # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', '6')))
FRAMES = int(os.environ.get('FRAMES', '4'))
RES = int(os.environ.get('RES', '64'))
WANT = ('aten::mkldnn_convolution', 'aten::addmm', 'aten::mm', 'aten::copy_',
        'aten::_scaled_dot_product_flash_attention_for_cpu')


def prof_once(m, latent, t, ctx):
    from torch.profiler import ProfilerActivity, profile
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            _ = forward_once(m, latent, t, ctx)
    wall = time.perf_counter() - t0
    d = {}
    for e in prof.key_averages():
        if e.key in WANT:
            d[e.key] = (e.self_cpu_time_total / 1e3, e.count)
    return wall, d


def main():
    print('=' * 84)
    print('conv 成本：一次性开销 vs 真实算术')
    print('=' * 84)
    m = load_animatediff()
    latent, t, ctx = video_input(1, FRAMES, RES)
    print('frames=%d res=%d' % (FRAMES, RES))

    w1, d1 = prof_once(m, latent, t, ctx)
    w2, d2 = prof_once(m, latent, t, ctx)

    print('\n%-46s %12s %12s %10s' % ('op', 'run1 ms', 'run2 ms', 'run2/run1'))
    print('-' * 84)
    for k in WANT:
        a = d1.get(k, (float('nan'), 0))[0]
        b = d2.get(k, (float('nan'), 0))[0]
        r = (b / a) if a and a == a and a > 0 else float('nan')
        print('%-46s %12.1f %12.1f %9.2fx' % (k.replace('aten::', ''), a, b, r))
    print('-' * 84)
    print('%-46s %12.2f %12.2f %9.2fx' % ('WALL (含 profiler 开销)', w1, w2, w2 / w1))

    # 关键净指标：conv 之外的时间
    def nonconv(d):
        tot = sum(v[0] for v in d.values())
        return tot - d.get('aten::mkldnn_convolution', (0, 0))[0]
    print('\nconv 合计    : run1 %.1f ms -> run2 %.1f ms' %
          (d1.get('aten::mkldnn_convolution', (0,))[0], d2.get('aten::mkldnn_convolution', (0,))[0]))
    print('非-conv 合计 : run1 %.1f ms -> run2 %.1f ms' % (nonconv(d1), nonconv(d2)))

    c1 = d1.get('aten::mkldnn_convolution', (0,))[0]
    c2 = d2.get('aten::mkldnn_convolution', (0,))[0]
    if c1 > 0:
        saved = c1 - c2
        print('\n⇒ 一次性部分 ≈ %.1f ms（占总前向 %.1f%%）'
              % (saved, 100 * saved / (w1 * 1e3) if w1 else 0))
        if c2 > 0 and c1 / c2 > 1.15:
            print('   判据: conv2/conv1 = %.2f < 0.87 ⇒ 大头是**一次性**开销，可回收'
                  % (c2 / c1))
        else:
            print('   判据: conv2/conv1 = %.2f ⇒ 基本是真实算术，此方向收益有限'
                  % (c2 / c1))
    del latent, ctx
    gc.collect()
    print('=' * 84)


if __name__ == '__main__':
    main()
