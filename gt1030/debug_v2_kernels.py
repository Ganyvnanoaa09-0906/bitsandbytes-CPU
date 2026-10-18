"""Debug the v2 fused RMSNorm kernel and the transpose kernel in isolation."""
from __future__ import annotations

import sys

import numpy as np
import pyopencl as cl
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_step_v2 as v2
from gt1030.full_gpu_step_v2 import KERNELS, ctx, mf, queue, WG

prog = cl.Program(ctx, KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
K = {n: cl.Kernel(prog, n) for n in
     ('rms_apply', 'silu_gate', 'silu_bwd', 'resid_add', 'rms_bwd', 'transpose', 'adamw4')}

rng = np.random.default_rng(1)
rows, d = 7, 512
x = (rng.standard_normal((rows, d)).astype(np.float32) * 0.1).astype(np.float16)
w = np.ones(d, dtype=np.float16)
xb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
wb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
yb = cl.Buffer(ctx, mf.READ_WRITE, rows * d * 2)
K['rms_apply'](queue, (rows,), (WG,), xb, wb, yb, np.int32(rows), np.int32(d))
queue.finish()
y = np.empty((rows, d), dtype=np.float16)
cl.enqueue_copy(queue, y, yb)
queue.finish()
ref = torch.from_numpy(x.astype(np.float32))
ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + 1e-6)
err = np.abs(y.astype(np.float32) - ref.numpy()).max()
print('rms_apply: max abs err %.3e  (row0 first3 gpu=%s ref=%s)'
      % (err, y[0, :3], ref.numpy()[0, :3]))

# transpose 2D
R, C = 33, 17
src = (rng.standard_normal((R, C)).astype(np.float32) * 0.1).astype(np.float16)
sb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
db = cl.Buffer(ctx, mf.READ_WRITE, R * C * 2)
K['transpose'](queue, (C, R), None, sb, db, np.int32(R), np.int32(C))
queue.finish()
dst = np.empty((C, R), dtype=np.float16)
cl.enqueue_copy(queue, dst, db)
queue.finish()
print('transpose: max abs err %.3e  (expected dst == src.T)' % np.abs(dst.astype(np.float32) - src.T.astype(np.float32)).max())

# adamw4 coherence check on a small vector
n = 64
din, dout = 16, 4
master = (rng.standard_normal((din * dout,)).astype(np.float32))
mom = np.zeros_like(master)
vel = np.zeros_like(master)
grad = (rng.standard_normal((din * dout,)).astype(np.float32) * 0.1).astype(np.float16)
mb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=master)
mob = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=mom)
vb = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
gb = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=grad)
wf = cl.Buffer(ctx, mf.WRITE_ONLY, din * dout * 2)
wb2 = cl.Buffer(ctx, mf.WRITE_ONLY, din * dout * 2)
K['adamw4'](queue, ((din * dout) // 4,), (256,), mb, mob, vb, gb, wf, wb2,
            np.int32(din), np.int32(dout), np.float32(1e-4), np.float32(0.9),
            np.float32(0.999), np.float32(1e-8), np.float32(0.0),
            np.float32(1.0), np.float32(1.0))
queue.finish()
got_master = np.empty_like(master)
cl.enqueue_copy(queue, got_master, mb)
queue.finish()
print('adamw4 first 4 master: got', got_master[:4], 'moved by', np.abs(got_master - master).max())
