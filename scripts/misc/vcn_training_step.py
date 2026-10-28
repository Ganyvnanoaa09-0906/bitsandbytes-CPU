"""Does hardware decode actually shorten the training step? The only number that matters.

7.65 showed d3d11va decodes at the same wall clock with 2.33x less CPU. But decoding was the only thing
running, so the saved CPU had nothing to compete with. In a training loop the loader and the model are
both on the CPU, and the question is whether the step gets shorter.

Measured: a real loader pulling clips, feeding a real forward and backward pass, run three ways -- no
decode at all as the floor, software decode, and d3d11va decode. Per-step timings are taken for the
whole loop, so the loader's cost and the model's are both included and their interference is visible.

Also measured: the loader alone, for both decoders, to see the raw difference in CPU time.

Decision rule, fixed in advance: hardware decode must shorten the full step by at least 5% for this to
matter in practice. Below that the saving exists on paper and not in the training loop, which is what
happened with the L2 blocking result in 7.54.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import av  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

CLIPS = r'D:\work\bitsandbytes-CPU\_vclip'
W, H, NFRAMES, NCLIPS, CLIPLEN = 640, 360, 60, 12, 16


def make_clips():
    os.makedirs(CLIPS, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(NCLIPS):
        p = os.path.join(CLIPS, 'c%02d.mp4' % i)
        if os.path.exists(p):
            continue
        c = av.open(p, mode='w')
        st = c.add_stream('libx264', rate=30)
        st.width, st.height, st.pix_fmt = W, H, 'yuv420p'
        base = (rng.random((H, W, 3)) * 255).astype(np.uint8)
        for k in range(NFRAMES):
            arr = np.roll(base, (i * 7 + k * 3) % W, axis=1).copy()
            for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format='rgb24')):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
        c.close()


def load_batch(hwaccel, n_clips=2):
    """Decode CLIPLEN frames from n_clips files, return a tensor."""
    out = []
    files = sorted(os.listdir(CLIPS))[:n_clips]
    opts = {'hwaccel': hwaccel} if hwaccel else {}
    for f in files:
        with av.open(os.path.join(CLIPS, f), options=opts) as c:
            got = 0
            for fr in c.decode(video=0):
                arr = fr.to_ndarray(format='rgb24')
                out.append(arr[::8, ::8].astype(np.float32) / 255.0)
                got += 1
                if got >= CLIPLEN:
                    break
    a = np.stack(out)
    return torch.from_numpy(a).permute(0, 3, 1, 2)


assert_idle()
stamp('硬解能不能缩短训练步')
torch.set_num_threads(6)
if not os.path.isdir(CLIPS) or len(os.listdir(CLIPS)) < NCLIPS:
    make_clips()
print()
print('  %d 个片段 · %dx%d · 每批取 2 个片段各 %d 帧' % (NCLIPS, W, H, CLIPLEN))
print()

cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                         ffn_dim=2048, total_tokens=1024, grid=32, head_mode='flat',
                         mask_mode='causal', n_cluster=32, use_bos=False)
model = SmallARImageModelV2(cfg)
model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                          loop_L_max=3, loop_random=False, loop_cache='shared'))
model.train()
tok = torch.load(r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt', map_location='cpu')
opt = torch.optim.SGD(model.parameters(), lr=0.0)


def model_step():
    x = tok[torch.randint(0, len(tok), (2,), generator=torch.Generator().manual_seed(0))].long()
    out = model.forward_logits(x, None)
    lg = out[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size),
                                      x[:, 1:].reshape(-1)).backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


print('  --- 1. 只有模型（不解码,下限）---')
for _ in range(2):
    model_step()
ts = []
for _ in range(4):
    t0 = time.perf_counter()
    model_step()
    ts.append(time.perf_counter() - t0)
only_model = statistics.median(ts)
print('     步时 %.3f s' % only_model)
print()

print('  --- 2. 加载器单独 ---')
for name, hw in (('软解', None), ('d3d11va', 'd3d11va')):
    load_batch(hw)
    c0 = os.times()
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        load_batch(hw)
        ts.append(time.perf_counter() - t0)
    ct = os.times()
    print('     %-10s 墙钟 %.3f s · CPU %.3f s' %
          (name, statistics.median(ts), (ct.user + ct.system) - (c0.user + c0.system)))
print()

print('  --- 3. 加载 + 训练步（真实循环）---')
print('  %-12s %12s %14s %12s' % ('解码方式', '步时 s', '相对无解码', '相对软解'))
print('  ' + '-' * 56)
res = {}
for name, hw in (('无解码', 'NONE'), ('软解', None), ('d3d11va', 'd3d11va')):
    def one():
        if hw == 'NONE':
            model_step()
        else:
            load_batch(hw)
            model_step()
    one()
    ts = []
    for _ in range(4):
        t0 = time.perf_counter()
        one()
        ts.append(time.perf_counter() - t0)
    med = statistics.median(ts)
    res[name] = med
    print('  %-12s %12.3f %13.3f× %11s' %
          (name, med, med / only_model,
           ('%.3f×' % (med / res['软解'])) if '软解' in res else '—'))
print()
sw = res['软解']
hv = res['d3d11va']
print('  判据（事先固定）: 硬解让完整步时缩短 ≥ 5% ⇒ 值得用')
gain = (sw - hv) / sw * 100
print('  ⇒ 硬解相对软解省 %.1f%% ⇒ %s' % (gain, '【值得】✓' if gain >= 5 else '【不值得】✗'))
