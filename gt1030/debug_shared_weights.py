"""N=10 forward with ONE device weight set: per-layer host weights are copied into the
same device weight buffers before each layer. Tests whether the corruption is caused by
allocating many device weight buffers (COPY_HOST_PTR / many cl_mem) rather than by layer math.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM  # noqa: E402
import gt1030.full_gpu_layer as L  # noqa: E402
from gt1030.debug_full_gpu_model import cpu_layer  # noqa: E402

M, MD, FF, WROW = FM.M, FM.MD, FM.FF, FM.WROW
A = L.A
rng = np.random.default_rng(0)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

scratch = FM.Layer(rng, 0)
input_buf = FM.buf(M * MD * 2)
hosts = []
for _ in range(N):
    hosts.append(dict(
        WQKV=(rng.standard_normal((MD, WROW)).astype(np.float32) * 0.02).astype(np.float16),
        WO=(rng.standard_normal((MD, MD)).astype(np.float32) * 0.02).astype(np.float16),
        W1=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W3=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W2=(rng.standard_normal((FF, MD)).astype(np.float32) * 0.02).astype(np.float16),
    ))

x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
cur = x
prev = x
for i in range(N):
    h = hosts[i]
    # one shared device weight set; copy this layer's weights in
    for key in ('WQKV', 'WO', 'W1', 'W3', 'W2'):
        A.cl.enqueue_copy(A.queue, scratch.__dict__[key]['dev'], h[key])
    scratch.b['wqkv'] = scratch.WQKV['dev']
    scratch.b['wo'] = scratch.WO['dev']
    scratch.b['w1'] = scratch.W1['dev']
    scratch.b['w3'] = scratch.W3['dev']
    scratch.b['w2'] = scratch.W2['dev']
    A.cl.enqueue_copy(A.queue, input_buf, cur)
    out = scratch.forward(input_buf)
    A.queue.finish()
    cur = np.empty((M, MD), dtype=np.float16)
    A.cl.enqueue_copy(A.queue, cur, out)
    A.queue.finish()
    ref = cpu_layer(torch.from_numpy(prev.astype(np.float32)),
                    torch.from_numpy(h['WQKV'].astype(np.float32)),
                    torch.from_numpy(h['WO'].astype(np.float32)),
                    torch.from_numpy(h['W1'].astype(np.float32)),
                    torch.from_numpy(h['W3'].astype(np.float32)),
                    torch.from_numpy(h['W2'].astype(np.float32)))
    err = float(np.abs(cur.astype(np.float32) - ref.numpy()).max())
    print('layer %d maxerr %.3e' % (i, err), flush=True)
    prev = cur
