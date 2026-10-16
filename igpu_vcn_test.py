"""Can the iGPU's video block decode without spending CPU? That is a different resource.

Everything measured so far concerned the shader cores, and the answer there is no: any CPU activity
puts the iGPU into a low power state, so the two cannot both run at full rate. But the video decode
block, VCN, is fixed-function hardware. It does not use the FP units, it does not run shaders, and it
draws a small fraction of the power budget. If it works, video data loading stops competing with
training for CPU cycles.

amfrt64.dll is present, which is AMD's media framework runtime, and PyAV 16.0.0 is installed and
exposes hardware acceleration. So the test is whether a hardware decoder actually opens and whether
the CPU cost of decoding drops.

Method: generate a short H.264 clip with the software encoder, then decode it repeatedly three ways --
software, d3d11va, and dxva2 -- measuring frames per second and this process's own CPU time. The CPU
figure is the one that matters: a decoder that is faster but uses the same CPU time has not freed
anything for training.

Decision rule, fixed in advance: a hardware path that decodes at least as fast as software while using
under half the CPU time is worth using for video data loading. A hardware path that fails to open, or
that is slower, is not.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

import av  # noqa: E402
from measure_guard import assert_idle, stamp  # noqa: E402

CLIP = r'D:\work\bitsandbytes-CPU\_vcn_test.mp4'
W, H, NFRAMES = 640, 360, 120


def cpu_time():
    t = os.times()
    return t.user + t.system


def make_clip():
    """Encode with the software encoder; only the decode side is under test."""
    container = av.open(CLIP, mode='w')
    stream = container.add_stream('libx264', rate=30)
    stream.width, stream.height, stream.pix_fmt = W, H, 'yuv420p'
    rng = np.random.default_rng(0)
    base = (rng.random((H, W, 3)) * 255).astype(np.uint8)
    for i in range(NFRAMES):
        arr = np.roll(base, i * 3, axis=1).copy()
        frame = av.VideoFrame.from_ndarray(arr, format='rgb24')
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def decode_all(hwaccel=None):
    n = 0
    opts = {}
    if hwaccel:
        opts['hwaccel'] = hwaccel
    try:
        with av.open(CLIP, options=opts) as c:
            for frame in c.decode(video=0):
                n += 1
    except Exception as e:
        return None, str(e)[:70]
    return n, None


assert_idle()
stamp('核显的视频解码块：能不能不花 CPU')
print()
if not os.path.exists(CLIP):
    make_clip()
    print('  生成测试片段: %s (%d 帧 %dx%d)' % (CLIP, NFRAMES, W, H))
else:
    print('  复用片段: %s' % CLIP)
print()

print('  %-12s %10s %12s %12s %14s' % ('解码方式', '帧数', '墙钟 ms', 'CPU ms', 'CPU/帧 ms'))
print('  ' + '-' * 66)
results = {}
for name, hw in (('软解', None), ('d3d11va', 'd3d11va'), ('dxva2', 'dxva2'), ('auto', 'auto')):
    n, err = decode_all(hw)
    if err:
        print('  %-12s 打不开: %s' % (name, err))
        results[name] = None
        continue
    c0 = cpu_time()
    t0 = time.perf_counter()
    n, err = decode_all(hw)
    wall = time.perf_counter() - t0
    cpu = cpu_time() - c0
    results[name] = (n, wall, cpu)
    print('  %-12s %10d %12.1f %12.1f %14.4f' %
          (name, n, wall * 1e3, cpu * 1e3, cpu / max(n, 1) * 1e3))
print()
sw = results.get('软解')
if sw:
    for name in ('d3d11va', 'dxva2', 'auto'):
        r = results.get(name)
        if r:
            print('  %s 相对软解: 墙钟 %.2f× · 【CPU %.2f×】' %
                  (name, sw[1] / r[1], sw[2] / max(r[2], 1e-9)))
print()
print('  判据（事先固定）: 不慢于软解,且 CPU 时间 < 软解的一半 ⇒ 值得用于视频数据加载')
ok = any(results.get(k) and results[k][2] < sw[2] * 0.5 and results[k][1] <= sw[1] * 1.2
         for k in ('d3d11va', 'dxva2', 'auto')) if sw else False
print('  ⇒ %s' % ('【可用】✓' if ok else '【不可用】✗'))
