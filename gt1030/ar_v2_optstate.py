# -*- coding: utf-8 -*-
"""优化器状态存取（AdamW 的 master/mom/vel）。

为什么需要：ckpt 只存权重，resume 时 AdamW 的 m/v 从 0 开始，第一步的更新量
≈ lr·3.16·sign(g)，远大于热状态 ⇒ 从"已经比较尖锐"的点续跑会立刻发散
（10-26 run B 从 ckpt_30000 续跑、val 5.1963 → 5.47 → 7.18 → 8.90 一路垮掉就是这个原因）。

格式：`<ckpt>.opt.npz`（和权重分开存，权重 ckpt 保持与 CPU trainer 兼容，不动格式）
  W.<param>.master / mom / vel   (fp32, 展平)
  E.tok_m / tok_v / row_m / row_v / col_m / col_v
  体积：29.5M 参数 × 3 × 4B ≈ 354 MB

    from ar_v2_optstate import save_opt_state, load_opt_state
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import numpy as np


def _param_keys():
    return ('Wqkv', 'Wo', 'W12', 'W3')


def save_opt_state(model, ckpt_path):
    """把 device 上的 master/mom/vel 和 embedding 的 m/v 存成 .opt.npz"""
    import gt1030.ar_v2_multi as MM
    d = model.dev
    d.activate()
    store = {}
    for i, blk in enumerate(model.blocks):
        for k in _param_keys():
            p = blk.W[k]
            for nm, buf in (('master', p.master), ('mom', p.mom), ('vel', p.vel)):
                store['W.%d.%s.%s' % (i, k, nm)] = d.down(buf, (p.n,), np.float32).copy()
    p = model.head.W
    for nm, buf in (('master', p.master), ('mom', p.mom), ('vel', p.vel)):
        store['H.%s' % nm] = d.down(buf, (p.n,), np.float32).copy()
    for nm, arr in (('tok_m', model.tok_m), ('tok_v', model.tok_v),
                    ('row_m', model.row_m), ('row_v', model.row_v),
                    ('col_m', model.col_m), ('col_v', model.col_v)):
        store['E.' + nm] = np.asarray(arr, dtype=np.float32)
    out = ckpt_path + '.opt.npz'
    np.savez(out, **store)
    mb = os.path.getsize(out) / 1e6
    print('[optstate] 存 %s (%.0f MB, %d 个数组)' % (out, mb, len(store)))
    return out


def load_opt_state(model, ckpt_path):
    """从 .opt.npz 恢复；缺文件返回 False（调用方决定要不要继续）"""
    import pyopencl as cl
    import gt1030.ar_v2_multi as MM
    path = ckpt_path + '.opt.npz'
    if not os.path.exists(path):
        return False
    z = np.load(path)
    d = model.dev
    d.activate()

    def up(buf, arr):
        cl.enqueue_copy(d.q, buf, np.ascontiguousarray(arr.astype(np.float32)))

    n = 0
    for i, blk in enumerate(model.blocks):
        for k in _param_keys():
            p = blk.W[k]
            for nm, buf in (('master', p.master), ('mom', p.mom), ('vel', p.vel)):
                key = 'W.%d.%s.%s' % (i, k, nm)
                if key in z:
                    up(buf, z[key]); n += 1
    p = model.head.W
    for nm, buf in (('master', p.master), ('mom', p.mom), ('vel', p.vel)):
        key = 'H.' + nm
        if key in z:
            up(buf, z[key]); n += 1
    d.finish()
    for nm, target in (('tok_m', 'tok_m'), ('tok_v', 'tok_v'), ('row_m', 'row_m'),
                       ('row_v', 'row_v'), ('col_m', 'col_m'), ('col_v', 'col_v')):
        key = 'E.' + nm
        if key in z:
            setattr(model, target, z[key].astype(np.float32).copy())
            n += 1
    print('[optstate] 恢复 %s（%d 个数组）' % (path, n))
    return n > 0
