"""Patch the iGPU LoRA trainer: fp16 storage + fp32-accumulate GEMM kernels for the frozen base.

The 10-step test showed the iGPU path learns but far slower than the CPU (step-9 loss 7.32 vs 0.083).
Suspected cause: CLBlast Hgemm's half-precision accumulation over K=2048/5632 corrupts the dX signal
that trains the LoRA adapters. Fix: keep fp16 storage (weights fp16, resident on the device once) but
accumulate in float in our own tiled kernels:

  gemm_nt_acc : C[M,N] = A[M,K] @ B[N,K]^T   (frozen base forward:  y = x @ W^T)
  gemm_nn_acc : C[M,N] = A[M,K] @ B[K,N]     (frozen base backward: dX = gy @ W)

Output is fp32, so the autograd chain sees fp32 numbers like the CPU reference.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)

import io

P = r'D:\work\bitsandbytes-CPU\gt1030\igpu_train_lora.py'
t = io.open(P, encoding='utf-8').read()

KERNELS_CODE = '''

GEMM_KERNELS = r"""
#define TS 16
__kernel void gemm_nt_acc(__global const half* A, __global const half* B, __global float* C,
                          int M, int N, int K) {
    int tx = get_local_id(0), ty = get_local_id(1);
    int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    __local half As[TS][TS];
    __local half Bs[TS][TS];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int m = by + ty, gk = k0 + tx;
        As[ty][tx] = (m < M && gk < K) ? A[m * K + gk] : (half)0;
        int n = bx + tx, kk = k0 + ty;
        Bs[tx][ty] = (n < N && kk < K) ? B[n * K + kk] : (half)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TS; ++q) acc += (float)As[ty][q] * (float)Bs[tx][q];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int m2 = by + ty, n2 = bx + tx;
    if (m2 < M && n2 < N) C[m2 * N + n2] = acc;
}
__kernel void gemm_nn_acc(__global const half* A, __global const half* B, __global float* C,
                          int M, int N, int K) {
    int tx = get_local_id(0), ty = get_local_id(1);
    int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    __local half As[TS][TS];
    __local half Bs[TS][TS];
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += TS) {
        int m = by + ty, gk = k0 + tx;
        As[ty][tx] = (m < M && gk < K) ? A[m * K + gk] : (half)0;
        int gk2 = k0 + ty, n = bx + tx;
        Bs[tx][ty] = (gk2 < K && n < N) ? B[gk2 * N + n] : (half)0;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int q = 0; q < TS; ++q) acc += (float)As[ty][q] * (float)Bs[tx][q];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    int m2 = by + ty, n2 = bx + tx;
    if (m2 < M && n2 < N) C[m2 * N + n2] = acc;
}
"""

_GEMM_PROG = cl.Program(ctx, GEMM_KERNELS).build(options=['-cl-mad-enable', '-cl-fast-relaxed-math'])
_GEMM_NT = cl.Kernel(_GEMM_PROG, 'gemm_nt_acc')
_GEMM_NN = cl.Kernel(_GEMM_PROG, 'gemm_nn_acc')

'''

ANCHOR = 'STATS = {\'gemm\': 0.0, \'h2d\': 0.0, \'d2h\': 0.0, \'calls\': 0}\n'
assert t.count(ANCHOR) == 1
t = t.replace(ANCHOR, KERNELS_CODE + '\n' + ANCHOR, 1)

HELPERS = '''

def gemm_nt_acc(a16, b_dev, m, n, k):
    """C[m,n] = A[m,k] @ B[n,k]^T, fp16 in, fp32 accumulate/out. b_dev is a cached device buffer."""
    ad = _buf('a16', m * k * 2)
    cl.enqueue_copy(queue, ad, a16)
    cd = _buf('c32', m * n * 4)
    gsz = (((n + 15) // 16) * 16, ((m + 15) // 16) * 16)
    _GEMM_NT(queue, gsz, (16, 16), ad, b_dev, cd, np.int32(m), np.int32(n), np.int32(k))
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, cd)
    queue.finish()
    STATS['calls'] += 1
    return out


def gemm_nn_acc(a16, b_dev, m, n, k):
    """C[m,n] = A[m,k] @ B[k,n], fp16 in, fp32 accumulate/out."""
    ad = _buf('a16', m * k * 2)
    cl.enqueue_copy(queue, ad, a16)
    cd = _buf('c32', m * n * 4)
    gsz = (((n + 15) // 16) * 16, ((m + 15) // 16) * 16)
    _GEMM_NN(queue, gsz, (16, 16), ad, b_dev, cd, np.int32(m), np.int32(n), np.int32(k))
    out = np.empty((m, n), dtype=np.float32)
    cl.enqueue_copy(queue, out, cd)
    queue.finish()
    STATS['calls'] += 1
    return out
'''
H_ANCHOR = 'class FrozenBaseFn(torch.autograd.Function):\n'
assert t.count(H_ANCHOR) == 1
t = t.replace(H_ANCHOR, HELPERS + '\n\n' + H_ANCHOR, 1)

# function signature keeps the device weight buffer
t = t.replace("""class FrozenBaseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp32, w_dev):
""", """class FrozenBaseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_fp32, w_dev):
""")

OLD_FWD = """        x2 = np.ascontiguousarray(x.detach().reshape(-1, k).float().numpy().astype(np.float16))
        m = x2.shape[0]
        t0 = time.perf_counter()
        ad = up(x2)
        cd = _buf('c', m * n * 2)
        hgemm_nt(ad, w_dev, cd, m, n, k)
        out = down(cd, (m, n))
        STATS['gemm'] += time.perf_counter() - t0
        STATS['calls'] += 1
        ctx.x16 = x2
        ctx.w_dev = w_dev
        ctx.orig = orig
        return torch.from_numpy(out.astype(np.float32)).reshape(*orig[:-1], n)
"""
NEW_FWD = """        x2 = np.ascontiguousarray(x.detach().reshape(-1, k).float().numpy().astype(np.float16))
        m = x2.shape[0]
        t0 = time.perf_counter()
        out = gemm_nt_acc(x2, w_dev, m, n, k)
        STATS['gemm'] += time.perf_counter() - t0
        ctx.x16 = x2
        ctx.orig = orig
        ctx.w_dev = w_dev
        return torch.from_numpy(out).reshape(*orig[:-1], n)
"""
assert t.count(OLD_FWD) == 1
t = t.replace(OLD_FWD, NEW_FWD, 1)

OLD_BWD = """        g = np.ascontiguousarray(gy.detach().reshape(-1, n).float().numpy().astype(np.float16))
        m = g.shape[0]
        ad = up(g)
        cd = _buf('c', m * k * 2)
        hgemm_nn(ad, ctx.w_dev, cd, m, k, n)   # dX = g @ W, W stored (out,in)
        out = down(cd, (m, k))
        return torch.from_numpy(out.astype(np.float32)).reshape(*ctx.orig[:-1], k), None, None
"""
NEW_BWD = """        g = np.ascontiguousarray(gy.detach().reshape(-1, n).float().numpy().astype(np.float16))
        m = g.shape[0]
        out = gemm_nn_acc(g, ctx.w_dev, m, k, n)   # dX = g @ W, W stored (out,in)
        return torch.from_numpy(out).reshape(*ctx.orig[:-1], k), None, None
"""
assert t.count(OLD_BWD) == 1
t = t.replace(OLD_BWD, NEW_BWD, 1)

# module: Lazy device weight buffer, host copy freed after upload
t = t.replace("""        self.weight16 = None
        self.w_host16 = None
""", """        self.weight16 = None
        self.w_dev = None
""")
t = t.replace("""    def ensure_host16(self):
        if self.w_host16 is None:
            self.w_host16 = np.ascontiguousarray(
                self.base_weight.detach().float().numpy().astype(np.float16))

    def forward(self, x):
        self.ensure_host16()
        y = FrozenBaseFn.apply(x, self.base_weight, self.w_host16)
""", """    def ensure_dev(self):
        if self.w_dev is None and not CPU_MODE:
            host16 = np.ascontiguousarray(
                self.base_weight.detach().float().numpy().astype(np.float16))
            self.w_dev = up(host16)          # upload once; host copy is dropped
            del host16

    def forward(self, x):
        self.ensure_dev()
        y = FrozenBaseFn.apply(x, self.base_weight, self.w_dev)
""")
io.open(P, 'w', encoding='utf-8', newline='').write(t)
print('patched: fp16 storage + fp32-accumulate tiled GEMM (device-cached weights)')
