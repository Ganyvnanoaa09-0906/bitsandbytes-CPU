# -*- coding: utf-8 -*-
"""EventProf sub-timing for one L3-shape SelfAttention (no per-section finish)."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import gt1030.igpu_block as B  # noqa: E402
from gt1030.igpu_profile import EventProf  # noqa: E402

rows, d, nh, hd, t = 512, 512, 8, 64, 256
lc = B.LayerCtx(arena_mb=300, name="attn", profiling=True)
lc.activate()
sa = B.SelfAttention(rows, d, nh, hd, t, bias=False, rng=np.random.default_rng(0))
rng = np.random.default_rng(0)
xh = (rng.standard_normal((rows, d)) * 0.02).astype(np.float16)
dyh = (rng.standard_normal((rows, d)) * 1e-6).astype(np.float16)
xd = B._upload(xh)
dyd = B._upload(dyh)
q = (rng.standard_normal((sa.b * nh, t, hd)) * 0.05).astype(np.float16)
k = (rng.standard_normal((sa.b * nh, t, hd)) * 0.05).astype(np.float16)
v = (rng.standard_normal((sa.b * nh, t, hd)) * 0.05).astype(np.float16)
sa.att.upload(q, k, v)
B.A.queue.finish()
prof = EventProf(lc.queue, warmup=2)


def fwd():
    with prof.sec("qkv"):
        qkv = sa.qkv.forward(xd)
    with prof.sec("split"):
        B._K["split_qkv"](
            B.A.queue,
            (sa.b * nh * t * hd,),
            None,
            qkv,
            sa.att.q,
            sa.att.k,
            sa.att.v,
            np.int32(sa.b),
            np.int32(t),
            np.int32(nh),
            np.int32(hd),
        )
    with prof.sec("attn_core_fwd"):
        sa.att.forward()
    with prof.sec("merge"):
        B._K["heads_to_tok"](
            B.A.queue,
            (rows * d,),
            None,
            sa.att.o,
            sa.o_tokens,
            np.int32(sa.b),
            np.int32(t),
            np.int32(nh),
            np.int32(hd),
        )
    with prof.sec("out"):
        sa.wo.forward(sa.o_tokens)


def bwd(dy):
    with prof.sec("out_b"):
        do_tokens = sa.wo.backward(dy)
    with prof.sec("to_heads"):
        B._K["tok_to_heads"](
            B.A.queue,
            (rows * d,),
            None,
            do_tokens,
            sa.att.do,
            np.int32(sa.b),
            np.int32(t),
            np.int32(nh),
            np.int32(hd),
        )
    with prof.sec("attn_core_bwd"):
        dq, dk, dv = sa.att.backward()
    with prof.sec("merge_qkv"):
        B._K["merge_qkv"](
            B.A.queue,
            (rows * d,),
            None,
            dq,
            dk,
            dv,
            sa.dqkv,
            np.int32(sa.b),
            np.int32(t),
            np.int32(nh),
            np.int32(hd),
        )
    with prof.sec("qkv_b"):
        sa.qkv.backward(sa.dqkv)


for _ in range(7):
    prof.new_step()
    fwd()
    bwd(dyd)
lc.queue.finish()
prof.report()
