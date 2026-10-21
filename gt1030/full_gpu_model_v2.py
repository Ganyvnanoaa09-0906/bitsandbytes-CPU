"""Full-GPU stack with ONE scratch set and per-layer checkpoint recomputation.

The 10-layer preallocation experiment showed that creating 2+ complete buffer sets corrupts
layer0 (N=2 works vs CPU, N=3+ fails). This variant keeps only one Layer's activation/scratch
buffers; each layer has its own weights. Forward stores x_entry per layer on the host; backward
recomputes the layer forward from x_entry, then runs the existing backward.

This is a correctness/debug vehicle, not the final zero-host-roundtrip trainer: the downloads are
used only for checkpointing (and can later be replaced by one saved device buffer per layer).
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

M, MD = FM.M, FM.MD
A = L.A


def main() -> int:
    n_layers = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rng = np.random.default_rng(0)
    # scratch layer replaces its own weights for each layer; only ONE buffer set exists.
    scratch = FM.Layer(rng, 0)
    input_buf = FM.buf(M * MD * 2)
    grad_buf = FM.buf(M * MD * 2)

    weights = []
    for _ in range(n_layers):
        weights.append(dict(
            WQKV=scratch._w(rng, (MD, FM.WROW)),
            WO=scratch._w(rng, (MD, MD)),
            W1=scratch._w(rng, (MD, FM.FF)),
            W3=scratch._w(rng, (MD, FM.FF)),
            W2=scratch._w(rng, (FM.FF, MD)),
        ))
        A.queue.finish()

    def bind(wi):
        w = weights[wi]
        scratch.WQKV, scratch.WO = w['WQKV'], w['WO']
        scratch.W1, scratch.W3, scratch.W2 = w['W1'], w['W3'], w['W2']
        scratch.b['wqkv'] = w['WQKV']['dev']
        scratch.b['wo'] = w['WO']['dev']
        scratch.b['w1'] = w['W1']['dev']
        scratch.b['w3'] = w['W3']['dev']
        scratch.b['w2'] = w['W2']['dev']

    x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
    dout_host = (rng.standard_normal((M, MD)).astype(np.float32) * 0.01).astype(np.float16)
    A.queue.finish()
    c = A.mf.COPY_HOST_PTR

    entries = []
    cur = x
    prev = x
    t0 = time.perf_counter()
    for i in range(n_layers):
        bind(i)
        A.cl.enqueue_copy(A.queue, input_buf, cur)
        A.queue.finish()
        out = scratch.forward(input_buf)
        A.queue.finish()
        cur = np.empty((M, MD), dtype=np.float16)
        A.cl.enqueue_copy(A.queue, cur, out)
        A.queue.finish()
        # per-layer CPU check on the actual entry
        w = weights[i]
        ref = cpu_layer(torch.from_numpy(prev.astype(np.float32)),
                        torch.from_numpy(w['WQKV']['host'].astype(np.float32)),
                        torch.from_numpy(w['WO']['host'].astype(np.float32)),
                        torch.from_numpy(w['W1']['host'].astype(np.float32)),
                        torch.from_numpy(w['W3']['host'].astype(np.float32)),
                        torch.from_numpy(w['W2']['host'].astype(np.float32)))
        err = float(np.abs(cur.astype(np.float32) - ref.numpy()).max())
        print('layer %d maxerr %.3e' % (i, err), flush=True)
        entries.append(cur.copy())
        prev = cur
    t_fwd = time.perf_counter() - t0

    # backward: recompute each layer from its entry, then run backward
    A.cl.enqueue_copy(A.queue, grad_buf, dout_host)
    A.queue.finish()
    for i in reversed(range(n_layers)):
        bind(i)
        A.cl.enqueue_copy(A.queue, input_buf, entries[i])
        scratch.forward(input_buf)
        A.queue.finish()
        dx = scratch.backward(grad_buf)
        A.queue.finish()
        A.cl.enqueue_copy(A.queue, grad_buf, dx)
        scratch.opt()
        A.queue.finish()
    t_all = time.perf_counter() - t0

    print('n_layers=%d fwd %.2fs total fwd+bwd+opt %.2fs' % (n_layers, t_fwd, t_all), flush=True)
    print('final grad max %.3e' % float(np.abs(dout_host).max()), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
