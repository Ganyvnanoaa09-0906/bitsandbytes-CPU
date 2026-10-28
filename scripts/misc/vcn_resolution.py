"""Does the VCN advantage actually scale with resolution? The projection needs checking.

At 640x360 hardware decode saves 2.9% of a training step, below the 5% bar. The reasoning for caring
anyway was that decode cost scales with pixels while the model step does not, so 720p would put decode
at 48% of the step and 1080p at 108%, where it would cost more than the model. That is arithmetic, not
measurement, and this project has been burned by extrapolation before.

So generate clips at three resolutions and measure the decode cost of both paths on each, plus the
frames per second each sustains. The model's step time is fixed at 3.807 s for 32 frames from 7.66, so
the share of the step that decode takes follows directly.

Decision rule, fixed in advance: if hardware decode holds its CPU advantage at 720p and above, the
projection stands and VCN is worth wiring into the video loader. If the advantage shrinks with
resolution, the projection was wrong and 2.9% is all there is.

Only decode is measured, not a full training step, because decode cost is what scales and the model
side does not change. Using the measured step time from 7.66 keeps it comparable.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import os
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import av  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

OUT = r'D:\work\bitsandbytes-CPU\_vres'
RES = [(640, 360), (1280, 720), (1920, 1080)]
NFRAMES = 40
STEP_S = 3.807          # measured in 7.66: model forward+backward for 32 frames
FRAMES_PER_STEP = 32


def cpu_time():
    t = os.times()
    return t.user + t.system


def make(w, h):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, 'v_%dx%d.mp4' % (w, h))
    if os.path.exists(p):
        return p
    c = av.open(p, mode='w')
    st = c.add_stream('libx264', rate=30)
    st.width, st.height, st.pix_fmt = w, h, 'yuv420p'
    rng = np.random.default_rng(0)
    base = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    for i in range(NFRAMES):
        arr = np.roll(base, i * 5, axis=1).copy()
        for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format='rgb24')):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    return p


def decode(path, hw):
    opts = {'hwaccel': hw} if hw else {}
    n = 0
    with av.open(path, options=opts) as c:
        for _ in c.decode(video=0):
            n += 1
    return n


assert_idle()
stamp('VCN：三个分辨率下的解码成本')
print()
print('  %-12s %-10s %10s %12s %12s %14s' %
      ('分辨率', '解码', 'CPU ms', 'CPU/帧 ms', '墙钟 ms', '解码占步时'))
print('  ' + '-' * 76)
rows = []
for (w, h) in RES:
    p = make(w, h)
    dec = {}
    for name, hw in (('软解', None), ('d3d11va', 'd3d11va')):
        try:
            decode(p, hw)
        except Exception as e:
            print('  %-12s %-10s 打不开: %s' % ('%dx%d' % (w, h), name, str(e)[:40]))
            continue
        c0 = cpu_time()
        t0 = time.perf_counter()
        n = decode(p, hw)
        wall = time.perf_counter() - t0
        cpu = cpu_time() - c0
        dec[name] = (n, wall, cpu)
        per_frame = cpu / max(n, 1)
        step_cpu = per_frame * FRAMES_PER_STEP
        share = step_cpu / STEP_S * 100
        if name == '软解':
            sw_share = share
        print('  %-12s %-10s %10.1f %12.3f %12.1f %13.1f%%' %
              ('%dx%d' % (w, h), name, cpu * 1e3, per_frame * 1e3, wall * 1e3, share))
    if '软解' in dec and 'd3d11va' in dec:
        ratio = dec['软解'][2] / max(dec['d3d11va'][2], 1e-9)
        saved = (dec['软解'][2] - dec['d3d11va'][2]) / STEP_S * 100
        print('  %-12s ⇒ CPU 省 【%.2f×】· 步时省 【%.1f%%】' % ('', ratio, saved))
        rows.append((w * h, ratio, saved))
    print()

print('  === 判读 ===')
print('  %-12s %12s %14s' % ('分辨率', 'CPU 省', '步时省'))
for px, r, s in rows:
    print('  %-12s %11.2f× %13.1f%%' % ('%d 万像素' % (px // 10000), r, s))
print()
print('  7.66 的外推: 640×360 ⇒ 2.9%% · 720p ⇒ 48%% · 1080p ⇒ 108%%')
print('  ⇒ 上表若与它吻合 ⇒ 外推成立 ⇒ VCN 值得接进视频加载器')
