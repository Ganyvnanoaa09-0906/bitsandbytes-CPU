"""Which allocation pattern corrupts an already-computed buffer on gfx902/WDDM?

Builds one correct full-GPU layer, reads its output back, then applies an allocation pattern
and reads the same buffer again. A nonzero diff means the driver/allocator corrupted an
existing live buffer.

Patterns:
  A: 100 FF-sized COPY_HOST_PTR buffers allocated after the layer (known to corrupt)
  B: 100 FF-sized plain READ_WRITE buffers allocated at startup, reused after the layer
  C: 100 FF-sized COPY_HOST_PTR buffers allocated at startup, reused after the layer
  D: 100 FF-sized USE_HOST_PTR buffers allocated after the layer
  E: 100 FF-sized buffers already allocated before the layer, then allocate 100 more after
  F: one large arena + subbuffers
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.full_gpu_model as FM  # noqa: E402
import gt1030.full_gpu_layer as L  # noqa: E402

M, MD, FF = FM.M, FM.MD, FM.FF
A = L.A


def make_layer_and_out():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((M, MD)).astype(np.float32) * 0.1).astype(np.float16)
    xdev = FM.upload(x)
    lay = FM.Layer(rng, 0)
    out = lay.forward(xdev)
    L.A.queue.finish()
    return lay, out


def read(out):
    a = np.empty((M, MD), dtype=np.float16)
    L.cl.enqueue_copy(A.queue, a, out)
    A.queue.finish()
    return a


def diff(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).max())


def alloc_copy(n):
    rng = np.random.default_rng(1)
    return [L.cl.Buffer(A.ctx, L.A.mf.READ_WRITE | L.A.mf.COPY_HOST_PTR,
                        hostbuf=rng.standard_normal((MD, FF)).astype(np.float16)) for _ in range(n)]


def alloc_rw(n):
    return [L.cl.Buffer(A.ctx, A.mf.READ_WRITE, MD * FF * 2) for _ in range(n)]


def alloc_use(n):
    rng = np.random.default_rng(1)
    arrs = [rng.standard_normal((MD, FF)).astype(np.float16) for _ in range(n)]
    bufs = [L.cl.Buffer(A.ctx, A.mf.READ_WRITE | A.mf.USE_HOST_PTR, hostbuf=a)
            for a in arrs]
    return bufs + arrs  # keep host arrays alive


def alloc_subbufs(n):
    parent = L.cl.Buffer(A.ctx, A.mf.READ_WRITE, n * MD * FF * 2)
    subs = []
    for i in range(n):
        subs.append(L.cl.Buffer(A.ctx, A.mf.READ_WRITE, MD * FF * 2,
                                buffer=parent, offset=i * MD * FF * 2))
    return [parent] + subs


def run_pattern(name, fn):
    lay, out = make_layer_and_out()
    base = read(out)
    keep = fn()
    L.A.queue.finish()
    after = read(out)
    print('%-8s diff %.4e' % (name, diff(base, after)), flush=True)
    del keep




def run_pre(name, pre_n, post_n):
    pre = alloc_rw(pre_n)
    lay, out = make_layer_and_out()
    base = read(out)
    post = alloc_rw(post_n)
    L.A.queue.finish()
    after = read(out)
    print('%-12s pre=%d post=%d diff %.4e' % (name, pre_n, post_n, diff(base, after)), flush=True)
    del pre, post


def main():
    run_pattern('A_copy', lambda: alloc_copy(100))
    run_pattern('B_rw', lambda: alloc_rw(100))
    run_pattern('C_copy_pre', lambda: alloc_copy(100))  # preallocated inside fn, but after layer
    run_pattern('D_use', lambda: alloc_use(100))
    run_pattern('E_extra', lambda: alloc_rw(100))
    run_pre('P100_no_post', 100, 0)
    run_pre('P100_post100', 100, 100)
    run_pre('P200_no_post', 200, 0)
    for n in (10,20,40,60,80):
        run_pre('P%d_no_post' % n, n, 0)


if __name__ == '__main__':
    main()
