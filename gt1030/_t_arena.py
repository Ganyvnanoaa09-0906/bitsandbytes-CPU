# -*- coding: utf-8 -*-
"""探究 22：单块 1.4 GB arena + 偏移读写是否稳定（为 arena 方案做可行性判断）。

背景：如果一张大 arena 上按任意偏移读写都稳定，就可以把 LoRA/激活/优化器状态放进同一块
device 内存，省掉反复分配。这个脚本只验证"偏移读写正确性"，不测速度。

历史：该文件曾经因为编码损坏（GBK 文本被当作 UTF-8 写入）导致第 2 行 docstring 的终止符
失效，整个文件无法解析；这里按原意重写，并修正 cl.enqueue_copy 的偏移参数名
（旧代码写的 src_dst_offset 不是合法关键字）。
"""
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

import numpy as np
import pyopencl as cl

import gt1030.ar_v2_model as MM

d = MM.Dev()
q, ctx, mf = d.q, d.ctx, d.mf
ARENA = 1400 << 20
arena = cl.Buffer(ctx, mf.READ_WRITE, ARENA)
print('arena %.2f GB allocated' % (ARENA / 1e9))
rng = np.random.default_rng(0)
# 模拟 400 个子区间（不同偏移/大小），逐个写不同值再回读
slots = []
off = 0
for i in range(400):
    n = int(rng.choice([512, 1024, 2048, 8192, 65536, 1 << 20, 4 << 20]))
    if off + n * 2 > ARENA:
        break
    slots.append((off, n))
    off += n * 2
print('%d 个子区间, 共 %.1f MB' % (len(slots), off / 1e6))
refs = []
for i, (o, n) in enumerate(slots):
    v = (rng.standard_normal(n) * 0.03).astype(np.float16)
    cl.enqueue_copy(q, arena, v, dst_offset=o)
    q.finish()
    refs.append((o, n, v))
bad = 0
for i, (o, n, v) in enumerate(refs):
    out = np.empty(n, np.float16)
    cl.enqueue_copy(q, out, arena, src_offset=o)
    q.finish()
    if not np.array_equal(out, v):
        bad += 1
        if bad <= 3:
            print('  第 %d 个区间 (off=%d, n=%d) 不一致: out[:3]=%s v[:3]=%s'
                  % (i, o, n, out[:3], v[:3]))
print('回读不一致: %d/%d' % (bad, len(refs)))
