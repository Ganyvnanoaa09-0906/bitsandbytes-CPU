# -*- coding: utf-8 -*-
"""test_cpu_4bit.py — 最小决定性测试：我们 CPU fork 的 Linear4bit 能不能真的在 CPU 上前向。

背景：diffusers 的 bnb 集成硬性检查 GPU
     （`bnb_quantizer.validate_environment` → "No GPU found. A GPU is needed for quantization."）
     但本项目就是对 bitsandbytes 做的**纯 CPU 爆改**，含 4-bit GEMV 推理内核
     （报告 §3.5：带 CUDA 参考的逐位对齐验证）。
     ⇒ 要绕开 diffusers 的检查，手动替换 nn.Linear。

本测试要回答三个问题（缺一不可）：
  ① Linear4bit 能否在 CPU 上构造（量化）？
  ② 前向能否跑通？
  ③ 结果与 fp32 的误差有多大？—— 若是垃圾值，说明内核在这台机器上不可用。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
torch.set_num_threads(4)

import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit

print('  bnb', bnb.__version__)
print('  torch', torch.__version__)

torch.manual_seed(0)
IN, OUT = 768, 768
x = torch.randn(4, IN, dtype=torch.float32)
ref = torch.nn.Linear(IN, OUT, bias=True)
with torch.no_grad():
    y_ref = ref(x)

print('\n=== ① 构造 Linear4bit（CPU）===')
try:
    q = Linear4bit(IN, OUT, bias=True, compute_dtype=torch.float32,
                   quant_type='nf4', compress_statistics=True)
    print('  构造 OK')
    with torch.no_grad():
        q.weight = bnb.nn.Params4bit(ref.weight.data, requires_grad=False,
                                     quant_type='nf4', compress_statistics=True)
        q.weight = q.weight.to('cpu')
        if q.bias is not None:
            q.bias.data.copy_(ref.bias.data)
    print('  权重量化 OK   dtype=%s  shape=%s' % (q.weight.dtype, tuple(q.weight.shape)))
except Exception as e:
    import traceback; traceback.print_exc()
    print('  构造/量化失败')
    sys.exit(1)

print('\n=== ② CPU 前向 ===')
try:
    t0 = time.perf_counter()
    with torch.no_grad():
        y_q = q(x)
    dt = time.perf_counter() - t0
    print('  前向 OK  %.1f ms  输出 shape=%s dtype=%s' % (dt * 1000, tuple(y_q.shape), y_q.dtype))
except Exception as e:
    import traceback; traceback.print_exc()
    print('  前向失败 —— 说明该内核在本机不可用')
    sys.exit(2)

print('\n=== ③ 与 fp32 的误差 ===')
with torch.no_grad():
    d = (y_q.float() - y_ref.float())
    rel = d.norm() / y_ref.norm()
    print('  相对误差 ‖Δ‖/‖y‖ = %.4f' % rel.item())
    print('  逐元素最大绝对误差 = %.4f   参考量级 |y|≈%.2f'
          % (d.abs().max().item(), y_ref.abs().mean().item()))
    bad = (~torch.isfinite(y_q)).sum().item()
    print('  非有限值个数 = %d' % bad)
print('\n  判据：相对误差应在 0.05~0.20（nf4 的正常范围）；')
print('       若 >1.0 或出现 NaN/Inf ⇒ 内核在本机不可用，SDXL 那条路要换方案。')
