# -*- coding: utf-8 -*-
"""bench_diffusion_shapes.py — 用【扩散模型的真实形状】比较 fp32 / 8bit / 4bit。

为什么之前测错了：我用的是 (M=256, K=N=4096) —— 那偏 GEMM 但还不够大。
SDXL 在 768² 时 latent = 96×96 = **9216 个 token** ⇒ M=9216，是彻底的 GEMM 场景。
项目已知结论（§10.2）：**"M≳8 时手写 GEMV 比 oneDNN GEMM 慢 2~3×"**
  ⇒ 我们的 4-bit 路径对大 M 用 GEMV 内核 = 用错工具。

要回答：
  ① 在 M=9216 这种形状下，4bit / 8bit / fp32 谁快？
  ② Linear8bitLt 是否走 `gemm_8bit`（融合反量化）？它是不是更快？
  ③ 线程扩展如何（6 核能吃满吗）？
"""
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
import sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Linear8bitLt

# SDXL UNet 的实际层形状（attention 的 to_q/to_k/to_v/to_out、ff 的 linear1/2）
SHAPES = [
    ('attn qkv   (M=9216,K=640,N=640)', 9216, 640, 640),
    ('attn proj  (M=9216,K=640,N=640)', 9216, 640, 640),
    ('ff         (M=9216,K=640,N=2560)', 9216, 640, 2560),
    ('ff down    (M=9216,K=2560,N=640)', 9216, 2560, 640),
    ('txt attn   (M=9216,K=640,N=640)', 9216, 640, 640),
]


def make_4bit(lin):
    q = Linear4bit(lin.in_features, lin.out_features, bias=True,
                   compute_dtype=torch.float32, quant_type='nf4', compress_statistics=True)
    with torch.no_grad():
        q.weight = bnb.nn.Params4bit(lin.weight.data.clone(), requires_grad=False,
                                     quant_type='nf4', compress_statistics=True)
        q.weight = q.weight.to('cpu')
        q.bias.data.copy_(lin.bias.data)
    return q


def make_8bit(lin):
    q = bnb.nn.Linear8bitLt(lin.in_features, lin.out_features, bias=True,
                            has_fp16_weights=False, threshold=6.0)
    with torch.no_grad():
        q.weight = bnb.nn.Int8Params(lin.weight.data.clone(), requires_grad=False,
                                     has_fp16_weights=False)
        q.bias.data.copy_(lin.bias.data)
    return q


def bench(fn, iters=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


torch.manual_seed(0)
for name, M, K, N in SHAPES:
    x = torch.randn(M, K)
    lin32 = torch.nn.Linear(K, N)
    q4 = make_4bit(lin32)
    try:
        q8 = make_8bit(lin32)
    except Exception as e:
        q8 = None
        print('  8bit 构造失败: %s' % str(e)[:80])
    print('\n  === %s ===' % name)
    print('  %-8s %10s %10s %10s' % ('类型', '1线程', '4线程', '6线程'))
    for tag, mod in [('fp32', lin32), ('8bit', q8), ('4bit', q4)]:
        if mod is None:
            continue
        row = []
        for th in (1, 4, 6):
            torch.set_num_threads(th)
            try:
                with torch.no_grad():
                    row.append(bench(lambda: mod(x)))
            except Exception as e:
                row.append(float('nan'))
        print('  %-8s %10.3f %10.3f %10.3f' % (tag, row[0], row[1], row[2]))
    del x, lin32, q4, q8
