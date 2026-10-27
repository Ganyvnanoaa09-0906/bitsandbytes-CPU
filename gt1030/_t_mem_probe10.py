# -*- coding: utf-8 -*-
"""探针 10：200 个小 buffer，比较 4 种写入/读取路径是否都会在 ~128 处错位。"""
import sys
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np
import pyopencl as cl
import gt1030.attention_gpu as A

ctx, q, mf = A.ctx, A.queue, A.mf
N = 1024
n = 200
vals = [float(i + 1) for i in range(n)]


def run(tag, mk, write, read):
    bufs = []
    firstbad = None
    for i in range(n):
        b = mk(i)
        bufs.append(b)
        write(b, vals[i])
        got = read(b)
        if got != vals[i] and firstbad is None:
            firstbad = (i, got)
    # 再全量回读一遍
    bad2 = [i for i in range(n) if read(bufs[i]) != vals[i]]
    print('%-22s first_bad=%s  全量不一致 %d/%d  首例 %s'
          % (tag, firstbad, len(bad2), n, bad2[:3]))


def mk_host(i):
    return cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,
                     hostbuf=np.full(N, vals[i], np.float16))


def wr_copy(b, v):
    cl.enqueue_copy(q, b, np.full(N, v, np.float16))
    q.finish()


def wr_buf(b, v):
    cl.enqueue_write_buffer(q, b, np.full(N, v, np.float16))
    q.finish()


def rd_copy(b):
    out = np.empty(N, np.float16)
    cl.enqueue_copy(q, out, b)
    q.finish()
    return float(out[0])


def rd_buf(b):
    out = np.empty(N, np.float16)
    cl.enqueue_read_buffer(q, b, out)
    q.finish()
    return float(out[0])


def mk_empty(i):
    return cl.Buffer(ctx, mf.READ_WRITE, N * 2)


run('COPY_HOST_PTR/copy', mk_host, lambda b, v: None, rd_copy)
run('empty+write_buffer', mk_empty, wr_buf, rd_buf)
run('empty+copy', mk_empty, wr_copy, rd_copy)
run('empty+write_buffer/copy', mk_empty, wr_buf, rd_copy)
