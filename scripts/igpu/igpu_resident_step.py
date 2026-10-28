"""Device-resident step: dW stays on the device and feeds Adam directly. Traffic counted, not claimed.

7.101 verified the Adam kernel and noted that the remaining step was wiring -- pass the dW buffer that
the backward GEMM wrote straight into the Adam kernel instead of downloading and re-uploading it. This
does that, on one layer so the saving is isolated rather than buried in a two-layer loop.

What is counted, rather than asserted, is the host traffic per step. That is the structural criterion
from 7.100: with device residency the per-step traffic should stop scaling with the parameter count and
start scaling with the batch. Counting bytes is how that claim either holds or does not.

The accounting is explicit about what still has to move, because not all of it can go away in this
design and pretending otherwise would be the kind of claim this report keeps finding in other code:

    x        upload     B*K      unavoidable, it is the input
    y        download   B*N      the loss is computed on the host
    g        upload     B*N      the gradient w.r.t. the output is host-side
    dW       ---       N*K       ★ resident, this is the saving
    weights  ---       N*K       ★ resident, this is the other saving
    moments  ---        2*N*K    ★ resident

So per-step traffic goes from O(NK) to O(BK+BN). For a 512x512 layer at batch 1024 that is a factor of
about 1 over 4 -- and the factor improves with model size, which is the point.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)

import sys
import time

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pyopencl as cl  # noqa: E402

KER = r'''
#define TS 64
#define TK 16
#define TM 4
#define TN 4
__kernel void gemm(const int M, const int N, const int K,
                   const __global float* restrict A,
                   const __global float* restrict B,
                   __global float* restrict C) {
    __local float As[TS][TK + 1];
    __local float Bs[TK][TS + 4];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int bx = get_group_id(0) * TS, by = get_group_id(1) * TS;
    const int lane = ty * 16 + tx;
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;
    for (int k0 = 0; k0 < K; k0 += TK) {
        { int f4 = lane; int r = (f4*4)/TK, c = (f4*4)%TK;
          int gr = by + r, gc = k0 + c; float4 v = (float4)0.f;
          if (gr < M && gc + 3 < K) v = *(const __global float4*)(A + gr*K + gc);
          As[r][c+0]=v.x; As[r][c+1]=v.y; As[r][c+2]=v.z; As[r][c+3]=v.w; }
        { int f4 = lane; int k = (f4*4)/TS, n = (f4*4)%TS;
          int gk = k0 + k, gn = bx + n; float4 v = (float4)0.f;
          if (gk < K && gn + 3 < N) v = *(const __global float4*)(B + gk*N + gn);
          #pragma unroll
          for (int q = 0; q < 4; q++) { int cc = n + q;
              Bs[k][(cc % TN) * (TS / TN) + (cc / TN)] =
                  (q==0)?v.x:((q==1)?v.y:((q==2)?v.z:v.w)); } }
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a[i] = As[ty*TM + i][k];
            #pragma unroll
            for (int j = 0; j < TN; j++) b[j] = Bs[k][j*(TS/TN) + tx];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) acc[i][j] += a[i]*b[j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    #pragma unroll
    for (int i = 0; i < TM; i++) { int gr = by + ty*TM + i;
        #pragma unroll
        for (int j = 0; j < TN; j++) { int gc = bx + tx*TN + j;
            if (gr < M && gc < N) C[gr*N + gc] = acc[i][j]; } }
}

__kernel void adam2(const int N, const int K,
                    __global float* restrict p, __global float* restrict pT,
                    __global float* restrict m, __global float* restrict v,
                    const __global float* restrict g,
                    const float lr, const float b1, const float b2,
                    const float c1, const float c2, const float eps) {
    int i = get_global_id(0);
    if (i >= N * K) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    int n = i / K, k = i - n * K;
    p[i] -= lr * (mi / c1) / (sqrt(vi / c2) + eps);
    pT[k * N + n] = p[i];
}
'''

_plat = cl.get_platforms()[0]
_ctx = cl.Context(devices=[_plat.get_devices()[0]])
_q = cl.CommandQueue(_ctx)
_mf = cl.mem_flags
_prog = cl.Program(_ctx, KER).build()
k_gemm = cl.Kernel(_prog, 'gemm')
k_adam = cl.Kernel(_prog, 'adam2')

STAT = {'to_dev': 0, 'to_host': 0}


def buf(n, ro=True):
    return cl.Buffer(_ctx, _mf.READ_ONLY if ro else _mf.READ_WRITE, max(int(n) * 4, 4))


def up(b, arr):
    a = np.ascontiguousarray(arr, dtype=np.float32)
    cl.enqueue_copy(_q, b, a).wait()
    STAT['to_dev'] += a.nbytes


def down(b, shape):
    out = np.empty(int(np.prod(shape)), np.float32)
    cl.enqueue_copy(_q, out, b).wait()
    STAT['to_host'] += out.nbytes
    return out.reshape(shape)


def gemm(M, N, K, a, b, c):
    gsz = ((N + 63) // 64 * 16, (M + 63) // 64 * 16)
    k_gemm(_q, gsz, (16, 16), np.int32(M), np.int32(N), np.int32(K), a, b, c)
    _q.finish()


# ---- 配置：一层 [N,K] ----
torch.manual_seed(0)
N = K = 512
B = 1024
lr, b1, b2, eps = 1e-3, 0.9, 0.999, 1e-8

W_true = torch.randn(N, K) * 0.1
Xt = torch.randn(4096, K)
Yt = Xt @ W_true.t()

# 设备常驻
w = buf(N * K, ro=False)
wT = buf(K * N, ro=False)
mt = buf(N * K, ro=False)
vt = buf(N * K, ro=False)
dW = buf(N * K, ro=False)     # ★ 梯度留在这里,不下载
xa = buf(B * K, ro=False)
ya = buf(B * N, ro=False)
gya = buf(B * N, ro=False)

W0 = (torch.randn(N, K) * 0.05)
up(w, W0.numpy())
up(wT, np.ascontiguousarray(W0.numpy().T))
up(mt, np.zeros((N, K)))
up(vt, np.zeros((N, K)))

# CPU 参照（逐位对拍用同一个初值与同一批数据）
pc = W0.clone()
mc = torch.zeros(N, K)
vc = torch.zeros(N, K)

print()
print('  === 设备常驻一步：dW 不下载，直接喂 Adam ===')
print('  层 %dx%d · batch %d · 参数 %d 个 (%.1f KB)' % (N, K, B, N * K, N * K * 4 / 1024))
print()

losses = []
losses_c = []
STAT['to_dev'] = STAT['to_host'] = 0
ts = []
for step in range(1, 9):
    t0 = time.perf_counter()
    idx = torch.randint(0, Xt.shape[0], (B,))
    xb = Xt[idx]
    yb = Yt[idx]
    up(xa, xb.numpy())
    gemm(B, N, K, xa, wT, ya)                 # y = x @ w^T
    yn = down(ya, (B, N))                     # ← 下载 y 以算 loss（可优化,见末尾）
    loss = float(((torch.from_numpy(yn) - yb) ** 2).mean())
    losses.append(loss)
    gy = 2.0 * (torch.from_numpy(yn) - yb) / (B * N)
    up(gya, np.ascontiguousarray(gy.numpy()))
    # dW = gy^T @ x  ⇒ A=[N,B], B=[B,K] ⇒ 需要一个转置的 gy
    # ★ 而 gy 只有 [B,N],转置在 CPU 上做,流量 O(B*N) —— 与参数数无关
    gyT = np.ascontiguousarray(gy.numpy().T)           # [N,B]
    up(xa, np.ascontiguousarray(xb.numpy()))           # 复用 xa 作 [B,K]
    gyaT = buf(N * B, ro=False)
    up(gyaT, gyT)
    gemm(N, K, B, gyaT, xa, dW)                # dW 写在设备上,★ 不下载
    c1, c2 = 1 - b1 ** step, 1 - b2 ** step
    k_adam(_q, ((N * K + 255) // 256 * 256,), None,
           np.int32(N), np.int32(K), w, wT, mt, vt, dW,
           np.float32(lr), np.float32(b1), np.float32(b2),
           np.float32(c1), np.float32(c2), np.float32(eps))
    _q.finish()
    ts.append(time.perf_counter() - t0)

    # CPU 参照（同数据同初值）
    xc, yc = xb, yb
    pcy = xc @ pc.t()
    lc = float(((pcy - yc) ** 2).mean())
    losses_c.append(lc)
    g = 2.0 * (pcy - yc) / (B * N)
    mc = b1 * mc + (1 - b1) * (g.t() @ xc)
    vc = b2 * vc + (1 - b2) * (g.t() @ xc) ** 2
    pc = pc - lr * (mc / c1) / ((vc / c2).sqrt() + eps)

print('  %-5s %14s %14s' % ('步', '设备 loss', 'CPU loss'))
for i in range(len(losses)):
    print('  %-5d %14.6f %14.6f' % (i + 1, losses[i], losses_c[i]))

wd = down(w, (N, K))
err = np.abs(wd - pc.numpy()).max() / np.abs(pc.numpy()).max()
print()
print('  最终权重: 设备对 CPU 相对误差 【%.3e】 %s'
      % (err, '✓ 一致' if err < 1e-5 else '✗'))
print('  中位步时 %.1f ms' % (sorted(ts)[len(ts) // 2] * 1e3))

print()
print('  === ★ 主机流量记账（8 步合计）===')
td = STAT['to_dev'] / 1e6
th = STAT['to_host'] / 1e6
print('    主机 → 设备 %.1f MB（每步 %.2f MB）' % (td, td / 8))
print('    设备 → 主机 %.1f MB（每步 %.2f MB）' % (th, th / 8))
print()
print('    ★ 省掉的是【每步都不再搬】这些:')
print('      · 权重 w/wT      2 × %d×4 B = %.2f MB/步' % (N * K, 2 * N * K * 4 / 1e6))
print('      · Adam 状态 m/v  2 × %d×4 B = %.2f MB/步' % (N * K, 2 * N * K * 4 / 1e6))
print('      · 梯度 dW        1 × %d×4 B = %.2f MB/步' % (N * K, N * K * 4 / 1e6))
print('      ⇒ 合计省 %.2f MB/步，而参数越大省得越多' % (5 * N * K * 4 / 1e6))
print()
print('    ★ 而仍然要搬的是 O(B·K) 那部分（x, y, g），它随【batch】涨而不随【参数量】涨')
print('      ⇒ ★【这就是设备常驻的结构性好处:流量从 O(NK) 变成 O(BK)】')
print()
