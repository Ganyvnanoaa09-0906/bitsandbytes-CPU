# -*- coding: utf-8 -*-
"""igpu_attn_chunk.py — self-attention 的**按查询分块**实现（T=4096 的内存解法）。

为什么不用 online softmax：
  整流法要 `S = Tq×Tk` 整块（T=4096 ⇒ 16.8M 元素/33 MB，四块 ~537 MB）；
  online（流式）虽然省内存，但要么慢（一行一 work-item 的标量循环），
  要么容易写错（我试过，T=1 对、T=2 就偏）。

本实现：**沿 query 维切块**，每块只算 `Bq×Tk` 的 S：
  - GEMM 用已验证的 CLBlast；
  - softmax 用已验证的 `softmax_window`（按样本窗口归一化）；
  - 输出 GEMM 后再拼回。
内存 = O(Bq·Tk)，Bq=512、Tk=4096 时 **4 MB/块**（half），数值与整流法**逐位等价**（同一个 kernel）。

    python gt1030/igpu_attn_chunk.py --check
"""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, r"D:\work\bitsandbytes-CPU")
import numpy as np  # noqa: E402
import pyopencl as cl  # noqa: E402

import gt1030.igpu_ctx as CTX  # noqa: E402
from gt1030.ar_v2_multi import L as GEMM  # noqa: E402
from gt1030.igpu_attn_full import K as FULLK  # noqa: E402


def _buf(dev, n):
    return dev.buf(max(4096, int(n) * 2))


def attn_chunked(dev, q_buf, k_buf, v_buf, o_buf, B, Tq, Tk, C, bq=512):
    """q/v/o: (B,Tq,C)；k: (B,Tk,C)。按 query 维分块，逐块做 GEMM+softmax+GEMM。"""
    scale = 1.0 / np.sqrt(float(C))
    sb = _buf(dev, bq * Tk)
    pb = _buf(dev, bq * Tk)
    rowmax = max(4096, bq * 4)
    clamp = dev.buf(rowmax)
    for b in range(B):
        qbase = (b * Tq) * C
        kbase = (b * Tk) * C
        obase = (b * Tq) * C
        # K/V 每个样本只拷一次（每个 query 块复用）
        tmpk = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpk, k_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        tmpv = _buf(dev, Tk * C)
        cl.enqueue_copy(dev.q, tmpv, v_buf, src_offset=kbase * 2, byte_count=Tk * C * 2)
        done = 0
        while done < Tq:
            rows = min(bq, Tq - done)
            # pyopencl 不支持子缓冲切片，query 块靠 offset 拷到临时缓冲
            tmpq = _buf(dev, rows * C)
            cl.enqueue_copy(dev.q, tmpq, q_buf, src_offset=(qbase + done * C) * 2,
                            byte_count=rows * C * 2)
            GEMM.hgemm_nt(tmpq, tmpk, sb, rows, Tk, C)          # S = Q Kᵀ
            FULLK(dev)["softmax_window"](
                dev.q, (rows,), None, sb, pb,
                np.int32(1), np.int32(Tk), np.float32(scale))
            GEMM.hgemm_nn(pb, tmpv, tmpq, rows, C, Tk)          # O = P V
            cl.enqueue_copy(dev.q, o_buf, tmpq,
                            dst_offset=(obase + done * C) * 2, byte_count=rows * C * 2)
            done += rows
    return o_buf


def check(t=512, c=64, b=1, bq=128, seed=0):
    import torch

    rng = np.random.default_rng(seed)
    q = (rng.standard_normal((b, t, c)) * 0.5).astype(np.float32)
    k = (rng.standard_normal((b, t, c)) * 0.5).astype(np.float32)
    v = (rng.standard_normal((b, t, c)) * 0.5).astype(np.float32)
    dev = CTX.DevCtx(arena_mb=800, name="chunk")
    dev.activate()
    qb = dev.up(q.astype(np.float16))
    kb = dev.up(k.astype(np.float16))
    vb = dev.up(v.astype(np.float16))
    ob = _buf(dev, b * t * c)
    import time

    t0 = time.time()
    attn_chunked(dev, qb, kb, vb, ob, b, t, t, c, bq=bq)
    dev.q.finish()
    ms = 1000 * (time.time() - t0)
    ov = dev.down(ob, (b, t, c)).astype(np.float32)
    ref = torch.nn.functional.scaled_dot_product_attention(
        torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v)).numpy()
    rel = float(np.linalg.norm(ov - ref) / np.linalg.norm(ref))
    print("  chunk B%d T%d C%d bq=%d: rel = %.2e · %.0f ms" % (b, t, c, bq, rel, ms))
    return rel < 2e-2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--big", action="store_true", help="额外跑 T=1024")
    a = ap.parse_args()
    if not a.check:
        print(__doc__)
        return 0
    ok = check(t=256, c=64, bq=64)
    ok &= check(t=512, c=320, bq=128)
    if a.big:
        ok &= check(t=1024, c=320, bq=256)
    print("RESULT " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
