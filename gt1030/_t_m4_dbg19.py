# -*- coding: utf-8 -*-
"""探针 19：构建成功后，分别验证 (a) 权重可读 (b) raw buffer 可写可读 (c) kernel 能算。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.ar_v2_model as MM

CKPT = r'F:\weibo_cos\ar_run_full3\ckpt_12000.pt'


def main():
    m = MM.V2ModelDev(CKPT)
    d, K, q = m.dev, m.dev.K, m.dev.q
    M, MD = MM.M, MM.MD
    a = d.down(m.layers[0].W['Wqkv'].w, (MD, MM.WROW)).astype(np.float32)
    print('(a) L0.Wqkv absmax=%.4f' % np.abs(a).max())

    raw = d.buf(M * MD * 2)
    src = np.full(M * MD, 0.5, np.float16)
    cl.enqueue_copy(q, raw, src, is_blocking=True)
    q.finish()
    back = d.down(raw, (M, MD)).astype(np.float32)
    print('(b) raw write->read absmax=%.4f (期望 0.5)' % np.abs(back).max())

    # (c) kernel: add_vec(raw, raw) -> out
    out = d.buf(M * MD * 2)
    K['add_vec'](q, (M * MD,), (256,), raw, raw, out, np.int32(M * MD))
    q.finish()
    o = d.down(out, (M, MD)).astype(np.float32)
    print('(c) add_vec(absmax) = %.4f (期望 1.0)' % np.abs(o).max())

    # (d) 直接跑到 embed + 第一层 rms
    tok = np.zeros((MM.B, MM.T), np.int64)
    x = m.embed(tok)
    xa = d.down(m.xa, (M, MD), np.float16).astype(np.float32)
    print('(d) embed host absmax=%.4f  device absmax=%.4f' % (np.abs(x).max(), np.abs(xa).max()))


main()
