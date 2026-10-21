"""N-layer full-GPU step with device checkpoints and ONE shared weight/scratch set.

- checkpoint arena: one cl_mem, slices via enqueue_copy offsets; no per-layer host roundtrip
- weights: host fp16 per layer -> one shared device weight buffer before each layer
- forward: ckpt[i] -> input_buf -> layer -> ckpt[i+1]
- backward: ckpt[i] -> input_buf -> recompute forward -> backward(grad_buf) -> grad_buf=dx
No new cl_mem is created during the step.
"""
from __future__ import annotations

import sys
import time

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
grad_buf = FM.buf(M * MD * 2)
ckpt = FM.buf((N + 1) * M * MD * 2)   # one arena: x_entry for each layer + final output
ROW = M * MD * 2

hosts = []
for _ in range(N):
    hosts.append(dict(
        WQKV=(rng.standard_normal((MD, WROW)).astype(np.float32) * 0.02).astype(np.float16),
        WO=(rng.standard_normal((MD, MD)).astype(np.float32) * 0.02).astype(np.float16),
        W1=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W3=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W2=(rng.standard_normal((FF, MD)).astype(np.float32) * 0.02).astype(np.float16),
    ))


def bind(h):
    for key in ('WQKV', 'WO', 'W1', 'W3', 'W2'):
        A.cl.enqueue_copy(A.queue, scratch.__dict__[key]['dev'], h[key])
    scratch.b['wqkv'] = scratch.WQKV['dev']
    scratch.b['wo'] = scratch.WO['dev']
    scratch.b['w1'] = scratch.W1['dev']
    scratch.b['w3'] = scratch.W3['dev']
    scratch.b['w2'] = scratch.W2['dev']


x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
dout = (rng.standard_normal((M, MD)).astype(np.float32) * 0.01).astype(np.float16)
A.cl.enqueue_copy(A.queue, ckpt, x)
A.cl.enqueue_copy(A.queue, grad_buf, dout)
A.queue.finish()

t0 = time.perf_counter()
for i in range(N):
    bind(hosts[i])
    A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROW, src_offset=i * ROW, dst_offset=0)
    out = scratch.forward(input_buf)
    A.cl.enqueue_copy(A.queue, ckpt, out, byte_count=ROW, src_offset=0, dst_offset=(i + 1) * ROW)
A.queue.finish()
t_fwd = time.perf_counter() - t0

t1 = time.perf_counter()
for i in reversed(range(N)):
    bind(hosts[i])
    A.cl.enqueue_copy(A.queue, input_buf, ckpt, byte_count=ROW, src_offset=i * ROW, dst_offset=0)
    scratch.forward(input_buf)
    dx = scratch.backward(grad_buf)
    A.cl.enqueue_copy(A.queue, grad_buf, dx, byte_count=ROW)
A.queue.finish()
t_bwd = time.perf_counter() - t1

# final output vs CPU chain
out = np.empty((M, MD), dtype=np.float16)
A.cl.enqueue_copy(A.queue, out, ckpt, src_offset=N * ROW)
A.queue.finish()
ref = torch.from_numpy(x.astype(np.float32))
with torch.no_grad():
    for i in range(N):
        h = hosts[i]
        ref = cpu_layer(ref, torch.from_numpy(h['WQKV'].astype(np.float32)),
                        torch.from_numpy(h['WO'].astype(np.float32)),
                        torch.from_numpy(h['W1'].astype(np.float32)),
                        torch.from_numpy(h['W3'].astype(np.float32)),
                        torch.from_numpy(h['W2'].astype(np.float32)))
err = float(np.abs(out.astype(np.float32) - ref.numpy()).max())
print('N=%d fwd %.2fs bwd %.2fs final maxerr %.3e' % (N, t_fwd, t_bwd, err), flush=True)
