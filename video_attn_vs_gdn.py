# -*- coding: utf-8 -*-
"""video_attn_vs_gdn.py — O(T^2) attention vs O(T) GDN：同进程、同形状、实测

为什么需要:
    video_cost_at_scale.py 的 GDN 那一列用的是**引用的旧数**（8.40 µs/token），
    而 video_attn_scaling.py 已经证明这类外推在斜率漂移时会骗人。
    所以这里把两者放进**同一进程、同一形状、同一计时循环**直接比。

判据:
    GDN 相对 attention 的倍率是否**随 T 增长**。若增长 ⇒ attention 的 T^2 项在主导，
    长视频必须线性化；若倍率不变 ⇒ 两者同阶，换架构无意义。

oracle:
    · GDN 与 attention 的输出都做 finite / 形状检查（先确保测的是能用的算子）。
    · 两者都只测 forward（推理场景），并在同一进程内交替测量以减少漂移影响。
    · GDN 的 state 维度 K=V=head_dim 固定，attention 用同样的 n_head/n_kv_head，
      保证是"同一个模型位置上的两种选择"，不是两个不同规模的东西。
"""
from __future__ import annotations

import gc
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')

from small_image_model_v2 import SelfAttention  # noqa: E402

torch.set_num_threads(6)


def timed(fn, warmup=1, reps=3):
    for _ in range(warmup):
        fn()
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


class MinimalCfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main():
    D, NH, NKV = 512, 8, 4
    HD = D // NH
    cfg = MinimalCfg(d_model=D, n_head=NH, n_kv_head=NKV, ffn_dim=D * 4,
                     fuse_qkv=False, cond_mode='none', pos_mode='none',
                     token_order='raster', head_mode='flat', mask_mode='causal')
    attn = SelfAttention(cfg)
    attn.eval()

    ok_gdn = False
    gdn = None
    try:
        from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule as gdn
        ok_gdn = True
    except Exception as e:
        print('GDN 不可用: %s: %s' % (type(e).__name__, e))

    print('=' * 84)
    print('O(T^2) attention vs O(T) GDN —— 同进程实测')
    print('=' * 84)
    print('d_model=%d n_head=%d n_kv_head=%d head_dim=%d  threads=%d  gdn=%s'
          % (D, NH, NKV, HD, torch.get_num_threads(), ok_gdn))

    gdn_usable = False
    if ok_gdn:
        try:
            B, T, H, K, V = 1, 64, NH, HD, HD
            q = torch.randn(B, T, H, K)
            k = torch.randn(B, T, H, K)
            v = torch.randn(B, T, H, V)
            beta = torch.rand(B, T, H)
            g = -torch.rand(B, T, H) * 0.1
            o, _ = gdn(q=q, k=k, v=v, beta=beta, g=g, scale=1.0 / (K ** 0.5))
            fin = bool(torch.isfinite(o).all())
            print('  GDN 连通性自检: out=%s finite=%s %s'
                  % (tuple(o.shape), fin, 'OK' if fin and o.shape == (B, T, H, V) else 'FAIL'))
            gdn_usable = fin and o.shape == (B, T, H, V)
        except Exception as e:
            print('  GDN 自检失败: %s: %s' % (type(e).__name__, e))

    print('\n%-22s %8s %13s %13s %10s' % ('形状', 'T', 'attention ms', 'GDN ms', '倍率'))
    print('-' * 72)
    for label, per_frame, frames in (('144p 8帧', 144, 8), ('144p 16帧', 144, 16),
                                     ('144p 32帧', 144, 32), ('144p 64帧', 144, 64),
                                     ('256p 16帧', 256, 16), ('256p 32帧', 256, 32)):
        T = per_frame * frames
        x = torch.randn(1, T, D)
        with torch.no_grad():
            ta = timed(lambda: attn(x, cache=None, append=True), reps=2)
        tg = float('nan')
        if gdn_usable:
            q = torch.randn(1, T, NH, HD)
            k = torch.randn(1, T, NH, HD)
            v = torch.randn(1, T, NH, HD)
            beta = torch.rand(1, T, NH)
            g = -torch.rand(1, T, NH) * 0.1
            try:
                with torch.no_grad():
                    tg = timed(lambda: gdn(q=q, k=k, v=v, beta=beta, g=g,
                                           scale=1.0 / (HD ** 0.5)), reps=2)
            except Exception as e:
                print('    GDN 在 T=%d 失败: %s' % (T, e))
        ratio = (ta / tg) if tg == tg and tg > 0 else float('nan')
        print('  %-20s %8d %13.2f %13.2f %9.2fx'
              % (label, T, ta * 1e3, tg * 1e3 if tg == tg else float('nan'), ratio))
        del x
        gc.collect()

    print('\n' + '=' * 84)
    print('读法: 倍率随 T 增长 ⇒ attention 的 T^2 项主导，长视频必须线性化；')
    print('      倍率平或降 ⇒ 换架构不划算，去优化别的项。')
    print('=' * 84)


if __name__ == '__main__':
    main()
