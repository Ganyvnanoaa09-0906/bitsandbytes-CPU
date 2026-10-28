"""Diagnose whether the resident engine's AdamW updates actually change the LoRA weights."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.llama_resident as E

model = r'D:\deepsleep\_test\models\deepseek-coder-1.3b-base'
eng = E.Engine(model, 64)
g = np.random.default_rng(1234)
ids = g.integers(0, 2000, size=(eng.M,)).astype(np.int64)
targets = np.roll(ids, -1)

# --- one manual step with grad inspection ---
eng.forward(ids)
logits = E.down(eng.logits, (eng.M, eng.VOCAB)).astype(np.float64)
mx = logits.max(axis=1, keepdims=True)
p = np.exp(logits - mx); p /= p.sum(axis=1, keepdims=True)
loss0 = float(-np.log(np.clip(p[np.arange(eng.M), targets], 1e-12, None)).mean())
dlog = p.copy(); dlog[np.arange(eng.M), targets] -= 1.0; dlog /= eng.M
E.cl.enqueue_copy(E.queue, eng.dlogits, dlog.astype(np.float32))
eng.backward()
E.queue.finish()

# inspect a few grads (before the update)
def norm(dev, n):
    return float(np.linalg.norm(E.down(dev, (n,))))
lo0 = eng.lora[0]['qkv']
print('loss before update: %.6f' % loss0)
print('grad norms: lm.B %.3e | lm.A %.3e | layer0 qkv.B %.3e | layer0 qkv.A %.3e'
      % (norm(eng.lora_lm.gB, eng.lora_lm.gB.size // 4), norm(eng.lora_lm.gA, eng.lora_lm.gA.size // 4),
         norm(lo0.gB, lo0.gB.size // 4), norm(lo0.gA, lo0.gA.size // 4)))

B_before = E.down(eng.lora_lm.B, eng.lora_lm.Bh.shape)
# update manually with a large lr
eng._step = 1
for li in range(eng.NL):
    for name in eng.lora[li]:
        lo = eng.lora[li][name]
        eng._adamw(lo.A, lo.gA, lo.mA, lo.vA, lo.A.size, 1e-2, 0.1, 1e-3)
        eng._adamw(lo.B, lo.gB, lo.mB, lo.vB, lo.B.size, 1e-2, 0.1, 1e-3)
E.queue.finish()
B_after = E.down(eng.lora_lm.B, eng.lora_lm.Bh.shape)
print('lm.B changed by: %.3e (relative %.3e)'
      % (np.abs(B_after - B_before).max(), np.abs(B_after - B_before).max() / (np.abs(B_before).max() + 1e-9)))
