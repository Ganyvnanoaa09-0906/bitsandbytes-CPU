"""Can one quantized layer be rebuilt and dequantized correctly? Check before rebuilding 197.

The checkpoint's naming differs from the format qwen21_load.py was written against: the packed tensor is
`<key>.wq`, not `<key>`, and the metadata carries only `absmax` -- no quant_map, no shape -- so
QuantState.from_dict has nothing to work from. QuantState's constructor takes what is needed directly,
so it is built by hand instead.

Before rebuilding all 197 modules, one is worth verifying, because a wrong dequantization produces a
model that loads, runs, and is quietly wrong. The layout arithmetic is the first check:

    down_proj [2048, 6144] = 12,582,912 elements
    packed 4-bit           12,582,912 / 2  = 6,291,456 bytes   matches .wq shape [6291456, 1]
    absmax                 12,582,912 / 64 =   196,608         matches .absmax shape [196608]

and the second is whether the numbers that come out look like weights: NF4 stores 16 levels scaled by a
per-64-element absmax, so a dequantized row should be roughly symmetric about zero with a spread set by
its absmax, not all zeros and not saturated.

The definitive check comes after: a forward pass that produces coherent text. But a layer-level check
fails faster and says which of the two it is.
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

import json
import os
import sys

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


CKPT = r'D:\work\models\models\qwen3-1.7B-nf4'
KEY = 'model.layers.0.mlp.down_proj.weight'

import bitsandbytes as bnb  # noqa: E402
from bitsandbytes.functional import QuantState, dequantize_4bit  # noqa: E402
import safetensors.torch as st  # noqa: E402

with open(os.path.join(CKPT, 'quant_meta.json'), encoding='utf-8') as fh:
    meta = json.load(fh)
info = meta['layers'][KEY]
print()
print('  层: %s' % KEY)
print('  元数据: shape=%s dtype=%s blocksize=%s bias=%s'
      % (info['shape'], info['dtype'], info['blocksize'], info.get('bias')))

t = st.load_file(os.path.join(CKPT, 'model.safetensors'), device='cpu')
packed = t[KEY + '.wq']
absmax = t[KEY + '.absmax']
out_f, in_f = info['shape']
print('  磁盘: .wq %s %s · .absmax %s %s'
      % (tuple(packed.shape), packed.dtype, tuple(absmax.shape), absmax.dtype))
print('  核对: %d 元素 ÷2 = %d（.wq 是 %d）· ÷%d = %d（.absmax 是 %d）'
      % (out_f * in_f, out_f * in_f // 2, packed.numel(),
         info['blocksize'], out_f * in_f // info['blocksize'], absmax.numel()))
assert packed.numel() == out_f * in_f // 2, '打包尺寸不符'
assert absmax.numel() == out_f * in_f // info['blocksize'], 'absmax 尺寸不符'
print('  ✓ 尺寸完全吻合')

print()
print('  === 造 QuantState ===')
for flat in (False, True):
    try:
        p = packed.reshape(-1) if flat else packed
        qs = QuantState(absmax=absmax, shape=torch.Size([out_f, in_f]),
                        blocksize=info['blocksize'], quant_type=info['dtype'],
                        dtype=torch.bfloat16)   # ★ 原始未量化张量的 dtype,不是打包存储的
        print('  flat=%s ⇒ QuantState 建成功' % flat)
        print('    quant_type=%s blocksize=%s shape=%s' % (qs.quant_type, qs.blocksize, tuple(qs.shape)))
        print('    code: %s' % (None if qs.code is None else tuple(qs.code.shape)))
        dq = dequantize_4bit(p, qs, quant_type=info['dtype'])
        print('    反量化成功: %s %s' % (tuple(dq.shape), dq.dtype))
        print('    统计: mean %.6f · std %.6f · min %.4f · max %.4f · 零占比 %.1f%%'
              % (dq.float().mean(), dq.float().std(), dq.min(), dq.max(),
                 (dq == 0).float().mean() * 100))
        row = dq[0].float()
        print('    第一行: mean %.6f · std %.6f · absmax %.4f' % (row.mean(), row.std(), absmax[0]))
        print('    ⇒ 权重看起来%s' % ('正常 ✓' if dq.std() > 1e-4 and (dq == 0).float().mean() < 0.9
                                      else '★ 可疑 ✗'))
        break
    except Exception as e:
        print('  flat=%s ⇒ 失败: %s: %s' % (flat, type(e).__name__, str(e)[:150]))
