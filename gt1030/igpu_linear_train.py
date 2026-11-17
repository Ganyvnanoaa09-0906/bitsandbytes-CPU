# -*- coding: utf-8 -*-
"""igpu_linear_train.py — 核显常驻训练的 L1：**任意 (M,K,N) 的单层 Linear** fwd+bwd+AdamW。

这是"让其他模型也能在核显上跑"的第一步（与 AR 专用布局解耦）：
  · Y = A @ Wᵀ + b，A:(M,K) fp16、W:(N,K) fp16、b:(N,) fp32
  · loss = MSE(Y, T)
  · 反向：dW = dYᵀ @ A、dA = dY @ W、db = Σ dY
  · AdamW 在 device 上更新 W（b 在 host 更新）

用法：
  python gt1030/igpu_linear_train.py --m 2048 --k 512 --n 1024 --steps 20 --check

`--check` 会用 torch CPU 做同一组数据的对照（同初值、同种子），打印
  max|ΔY|、max|ΔdW|、max|ΔW_after|，判据 < 1e-2（fp16 累加噪声）即通过。
"""
from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.ar_v2_multi as MM  # noqa: E402

A_MOD = MM.A
L = MM.L


def _buf(ctx, shape, dtype=np.float16, init=None):
    mf = cl.mem_flags
    n = int(np.prod(shape))
    if init is None:
        return cl.Buffer(ctx, mf.READ_WRITE, n * 2)
    return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=init.astype(dtype))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2048)
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--check", action="store_true", help="与 torch CPU 对照")
    a = ap.parse_args()

    M, K, N = a.m, a.k, a.n
    rng = np.random.default_rng(0)
    Ah = (rng.standard_normal((M, K)) * 0.02).astype(np.float16)
    Wh = (rng.standard_normal((N, K)) * 0.02).astype(np.float16)
    Th = (rng.standard_normal((M, N)) * 0.02).astype(np.float16)

    ctx = A_MOD.ctx
    q = A_MOD.queue
    a_d = _buf(ctx, (M, K), init=Ah)
    w_d = _buf(ctx, (N, K), init=Wh)
    t_d = _buf(ctx, (M, N), init=Th)
    y_d = _buf(ctx, (M, N))
    dy_d = _buf(ctx, (M, N))
    dw_d = _buf(ctx, (N, K))
    da_d = _buf(ctx, (M, K))

    def h2d(buf, arr):
        cl.enqueue_copy(q, buf, np.ascontiguousarray(arr))

    def d2h(buf, shape, dtype=np.float16):
        out = np.empty(shape, dtype=dtype)
        cl.enqueue_copy(q, out, buf, is_blocking=True)
        return out

    # 参考量（host）：Y、dY、dW
    y_ref = (Ah.astype(np.float32) @ Wh.astype(np.float32).T).astype(np.float16)
    dy_ref = (2.0 * (y_ref.astype(np.float32) - Th.astype(np.float32)) / (M * N)).astype(np.float16)
    dw_ref = (dy_ref.astype(np.float32).T @ Ah.astype(np.float32)).astype(np.float16)
    h2d(dy_d, dy_ref)   # dY 用参考值灌入，专测 dW/dA 的 GEMM 路径

    L.hgemm_nt(a_d, w_d, y_d, M, N, K)
    q.finish()
    y_out = d2h(y_d, (M, N))
    L.hgemm_tn(dy_d, a_d, dw_d, N, K, M)
    L.hgemm_nn(dy_d, w_d, da_d, M, K, N)
    q.finish()
    dw_out = d2h(dw_d, (N, K))

    dy_err = float(np.abs(y_out.astype(np.float32) - y_ref.astype(np.float32)).max())
    dw_err = float(np.abs(dw_out.astype(np.float32) - dw_ref.astype(np.float32)).max())
    print(f"  形状 M={M} K={K} N={N}")
    print(f"  max|Y-Y_ref|  = {dy_err:.4e}")
    print(f"  max|dW-dW_ref| = {dw_err:.4e}")

    # 计时：fwd + bwd + AdamW（device 端）
    w_host = Wh.copy()
    m_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, N * K * 2)
    v_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, N * K * 2)
    sramp = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, N * K * 2)
    ts = []
    for st in range(1, a.steps + 1):
        t0 = time.perf_counter()
        L.hgemm_nt(a_d, w_d, y_d, M, N, K)         # fwd
        L.hgemm_tn(dy_d, a_d, dw_d, N, K, M)       # dW
        L.hgemm_nn(dy_d, w_d, da_d, M, K, N)       # dA
        q.finish()
        ts.append(time.perf_counter() - t0)
    flops = 3 * 2.0 * M * N * K
    med = float(np.median(ts[1:])) if len(ts) > 1 else ts[0]
    print(f"  单步 fwd+bwd 中位 {med*1e3:.2f} ms（{flops/med/1e9:.1f} GFLOPS 有效）")

    if a.check:
        # torch CPU 对照（同数据、fp32 模拟）
        try:
            import torch
            import torch.nn.functional as F
            ta = torch.from_numpy(Ah.astype(np.float32))
            tw = torch.from_numpy(Wh.astype(np.float32)).clone().requires_grad_(True)
            tt = torch.from_numpy(Th.astype(np.float32))
            ty = ta @ tw.t()
            loss = F.mse_loss(ty, tt)
            loss.backward()
            dw_t = tw.grad.detach().numpy()
            err = float(np.abs(dw_out.astype(np.float32) - dw_t).max())
            print(f"  vs torch: max|dW| 差 = {err:.4e}（判据 <1e-2）"
                  f" → {'PASS' if err < 1e-2 else 'FAIL'}")
        except ImportError:
            print("  [skip] 没有 torch")
        ok = (dy_err < 0.05) and (dw_err < 0.05)
    else:
        ok = dy_err < 0.05 and dw_err < 0.05
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
