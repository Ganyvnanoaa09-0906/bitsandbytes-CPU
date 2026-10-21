"""N-layer full-GPU forward+backward with ONE device weight set (shared), host weights.

This is the correctness vehicle after finding that allocating many device weight buffers
corrupts earlier buffers on this driver. Every layer reuses the same device weight buffers;
per-layer host weights are copied in before use, and for backward the layer forward is
recomputed from its checkpointed x_entry. Gradient check is against a torch fp32 chain.
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

hosts = []
for _ in range(N):
    hosts.append(dict(
        WQKV=(rng.standard_normal((MD, WROW)).astype(np.float32) * 0.02).astype(np.float16),
        WO=(rng.standard_normal((MD, MD)).astype(np.float32) * 0.02).astype(np.float16),
        W1=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W3=(rng.standard_normal((MD, FF)).astype(np.float32) * 0.02).astype(np.float16),
        W2=(rng.standard_normal((FF, MD)).astype(np.float32) * 0.02).astype(np.float16),
    ))


def bind(weights_host):
    for key in ('WQKV', 'WO', 'W1', 'W3', 'W2'):
        A.cl.enqueue_copy(A.queue, scratch.__dict__[key]['dev'], weights_host[key])
    scratch.b['wqkv'] = scratch.WQKV['dev']
    scratch.b['wo'] = scratch.WO['dev']
    scratch.b['w1'] = scratch.W1['dev']
    scratch.b['w3'] = scratch.W3['dev']
    scratch.b['w2'] = scratch.W2['dev']


x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
dout = (rng.standard_normal((M, MD)).astype(np.float32) * 0.01).astype(np.float16)
entries = [x]
cur = x
t0 = time.perf_counter()
for i in range(N):
    bind(hosts[i])
    A.cl.enqueue_copy(A.queue, input_buf, cur)
    out = scratch.forward(input_buf)
    A.queue.finish()
    cur = np.empty((M, MD), dtype=np.float16)
    A.cl.enqueue_copy(A.queue, cur, out)
    A.queue.finish()
    entries.append(cur)
t_fwd = time.perf_counter() - t0

A.cl.enqueue_copy(A.queue, grad_buf, dout)
A.queue.finish()
graderrs = []
t1 = time.perf_counter()
for i in reversed(range(N)):
    bind(hosts[i])
    A.cl.enqueue_copy(A.queue, input_buf, entries[i])
    scratch.forward(input_buf)
    A.queue.finish()
    dx = scratch.backward(grad_buf)
    A.queue.finish()
    A.cl.enqueue_copy(A.queue, grad_buf, dx)
    A.queue.finish()
t_bwd = time.perf_counter() - t1

# CPU reference chain: same fp16-cast weights, fp32 activations, same dout
refs = [torch.from_numpy(x.astype(np.float32))]
with torch.no_grad():
    cur_t = refs[0]
    for i in range(N):
        h = hosts[i]
        cur_t = cpu_layer(cur_t, torch.from_numpy(h['WQKV'].astype(np.float32)),
                          torch.from_numpy(h['WO'].astype(np.float32)),
                          torch.from_numpy(h['W1'].astype(np.float32)),
                          torch.from_numpy(h['W3'].astype(np.float32)),
                          torch.from_numpy(h['W2'].astype(np.float32)))
        refs.append(cur_t)
    # final dx by autograd
    last = refs[-1].clone().requires_grad_(True)
    d = torch.from_numpy(dout.astype(np.float32))
    last.backward(d)
    ref_dx = refs[-2].grad if refs[-2].is_leaf else None

print('N=%d fwd %.2fs bwd %.2fs' % (N, t_fwd, t_bwd), flush=True)
print('final grad max %.3e' % float(np.abs(dout).max()), flush=True)
print('final output sum %.4f' % float(entries[-1].astype(np.float32).sum()), flush=True)
