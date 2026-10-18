"""Debug the new workgroup softmax: compare P (and S) against torch for a few (bh,row) pairs."""
from __future__ import annotations

import sys

import numpy as np
import pyopencl as cl
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.attention_gpu as A

prog = cl.Program(A.ctx, A.KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
K = {n: cl.Kernel(prog, n) for n in ('softmax_causal',)}

rng = np.random.default_rng(3)
T, BH = A.T, A.BH
S = (rng.standard_normal((BH, T, T)).astype(np.float32) * 0.5).astype(np.float16)
Sb = cl.Buffer(A.ctx, A.mf.READ_ONLY | A.mf.COPY_HOST_PTR, hostbuf=S)
Pb = cl.Buffer(A.ctx, A.mf.WRITE_ONLY, BH * T * T * 2)
K['softmax_causal'](A.queue, (T, BH), (256, 1), Sb, Pb, np.int32(T), np.float32(1.0 / (A.D ** 0.5)))
A.queue.finish()
P = np.empty((BH, T, T), dtype=np.float16)
cl.enqueue_copy(A.queue, P, Pb)
A.queue.finish()

s = torch.from_numpy(S.astype(np.float32)) * (1.0 / (A.D ** 0.5))
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
s = s.masked_fill(~mask, float('-inf'))
ref = torch.softmax(s, -1).numpy()

for bh in (0, 5):
    for row in (0, 1, 17, 255):
        got = P[bh, row].astype(np.float32)
        r = ref[bh, row]
        print('bh=%2d row=%3d  got[:3]=%s  ref[:3]=%s  got[row]=%.5f ref=%0.5f sum=%.4f'
              % (bh, row, np.round(got[:3], 4), np.round(r[:3], 4),
                 got[row], r[row], got.sum()))
err = np.abs(P.astype(np.float32) - ref).max()
print('max abs err vs torch: %.4f' % err)
print('P rows sum range: %.4f .. %.4f' % (P.astype(np.float32).sum(-1).min(),
                                          P.astype(np.float32).sum(-1).max()))
