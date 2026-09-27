# -*- coding: utf-8 -*-
"""adiff_contig_ab.py — 最小验证：去掉 AnimateDiffTransformer3D 里的那次 .contiguous()

背景（report §10.151）:
    copy_ 占真实分辨率前向的 14.2%（399 次 / 1889 ms）。按形状归因后，前三个形状
    （16384×320 / 4096×640 / 1024×1280）各出现 **35 次**，恰等于时序 attention 块数
    ⇒ 每个时序块每次前向都在搬。来源定位到 diffusers 的
    AnimateDiffTransformer3D.forward 第 186 行（permute+reshape 隐式物化）与
    第 206 行（显式 .contiguous()）。

本脚本做的事:
    用 monkey-patch 把 forward 的**出口**改成不调用 .contiguous()（改成 .reshape，
    等价于"允许非连续"），然后在**同一进程内、同一输入**下对比：
      · 数值：两次输出是否一致（max|Δ|）；这是硬门槛，不通过就不看时间
      · 时间：min-of-N

criterion:
    数值必须一致（max|Δ| 在 fp32 累加误差量级内），且时间有可复现的下降。
    若数值不一致 ⇒ 说明那次 contiguous 是**语义必需**的，优化思路作废（这也是结论）。

⚠️ 只做诊断，不改 diffusers 安装目录的文件——patch 只在本次进程内生效。
"""
from __future__ import annotations

import gc
import os
import sys
import time
import types

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from video_models import load_animatediff, structure, video_input  # noqa: E402

torch.set_num_threads(int(os.environ.get('THREADS', '6')))
FRAMES = int(os.environ.get('FRAMES', '4'))
RES = int(os.environ.get('RES', '64'))


def timed(fn, reps):
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def main():
    from diffusers.models.unets import unet_motion_model as U
    cls = U.AnimateDiffTransformer3D
    orig_fwd = cls.forward
    print('=' * 84)
    print('去掉时序块出口的 .contiguous()：数值 + 时间 A/B')
    print('=' * 84)

    m = load_animatediff()
    st = structure(m)
    print('params=%.1fM motion_modules=%d frames=%d res=%d'
          % (st['params_M'], st['motion_named_modules'], FRAMES, RES))

    latent, t, ctx = video_input(1, FRAMES, RES)

    # ---- A: 原实现 ----
    with torch.no_grad():
        y_a = m(latent, t, encoder_hidden_states=ctx).sample
    ta = timed(lambda: m(latent, t, encoder_hidden_states=ctx), reps=1)
    print('[A] 原实现          : %.2f s  out=%s' % (ta, tuple(y_a.shape)))

    # ---- B: 出口不 contiguous（只改 reshape 那一支） ----
    def patched(self, hidden_states, encoder_hidden_states=None, timestep=None,
                return_dict=True, attention_mask=None, num_frames=1, **kw):
        batch_frames, channel, height, width = hidden_states.shape
        batch_size = batch_frames // num_frames
        residual = hidden_states
        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, channel, height, width)
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4)
        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.permute(0, 3, 4, 2, 1).reshape(
            batch_size * height * width, num_frames, channel)
        hidden_states = self.proj_in(input=hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                cross_attention_kwargs=kw.get('cross_attention_kwargs'),
                class_labels=kw.get('class_labels'),
            )
        hidden_states = self.proj_out(input=hidden_states)
        # >>> 唯一改动：去掉 .contiguous()，直接 permute 后 reshape <<<
        hidden_states = (
            hidden_states[None, None, :]
            .reshape(batch_size, height, width, num_frames, channel)
            .permute(0, 3, 4, 1, 2)
        )
        hidden_states = hidden_states.reshape(batch_frames, channel, height, width)
        return hidden_states + residual

    cls.forward = patched
    try:
        with torch.no_grad():
            y_b = m(latent, t, encoder_hidden_states=ctx).sample
        tb = timed(lambda: m(latent, t, encoder_hidden_states=ctx), reps=1)
        d = (y_a - y_b).abs().max().item()
        rel = d / (y_a.abs().max().item() + 1e-12)
        print('[B] 去 contiguous   : %.2f s  max|Δ|=%.3e  rel=%.3e  %s'
              % (tb, d, rel, 'NUMERICALLY OK' if rel < 1e-5 else 'NUMERICALLY DIFFERENT'))
        if ta > 0:
            print('    speedup = %.3fx   (时间差 %.2f s)' % (ta / tb, ta - tb))
    finally:
        cls.forward = orig_fwd

    print('\n读法: 若 rel < 1e-5 且 speedup > 1.05 ⇒ 这条优化真实可用；')
    print('      若 rel 明显非零 ⇒ 那次 contiguous 是语义必需的，此路作废。')
    del latent, ctx, y_a, y_b
    gc.collect()


if __name__ == '__main__':
    main()
