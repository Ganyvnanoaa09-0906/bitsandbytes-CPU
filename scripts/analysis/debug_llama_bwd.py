"""Debug/validate one llama layer backward: dX and LoRA dA/dB vs torch autograd."""
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
eng = E.Engine(model, 64, limit_layers=1, keep_host=True)
g = np.random.default_rng(0)

# nonzero LoRA B so the adapters have gradients
for name, lo in eng.lora[0].items():
    lo.Bh = (g.standard_normal(lo.Bh.shape) * 0.01).astype(np.float32)
    E.cl.enqueue_copy(E.queue, lo.B, lo.Bh)
E.queue.finish()

x = (g.standard_normal((eng.M, eng.DH)) * 0.1).astype(np.float32)
dOut = (g.standard_normal((eng.M, eng.DH)) * 0.01).astype(np.float32)

E.cl.enqueue_copy(E.queue, eng.xb[0], x)
eng.forward_layer(0, eng.xb[0], eng.xb[1])
dx_out = E.up(dOut)
dx_in = E.buf(eng.M * eng.DH * 4)
eng.backward_layer(0, dx_out, dx_in)
E.queue.finish()
dx = E.down(dx_in, (eng.M, eng.DH))
grads = {}
for name, lo in eng.lora[0].items():
    grads[name] = (E.down(lo.gA, lo.Ah.shape), E.down(lo.gB, lo.Bh.shape))

# ---------------- torch reference ----------------
hw = eng.host[0]
T, NH, HD, D, FF = eng.T, eng.NH, eng.HD, eng.DH, eng.FF
eps, theta = eng.EPS, eng.THETA

xt = torch.from_numpy(x).requires_grad_(True)
w = {k: torch.from_numpy(v) for k, v in hw.items()}
loras = {}
for name, lo in eng.lora[0].items():
    a = torch.from_numpy(lo.Ah.copy()).requires_grad_(True)
    b = torch.from_numpy(lo.Bh.copy()).requires_grad_(True)
    loras[name] = (a, b)

def rms(t, wt):
    return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps) * wt

h = rms(xt, w['n1'])
Wqkv = torch.from_numpy(np.concatenate([hw['q'].T, hw['k'].T, hw['v'].T], axis=1))
qkv = h @ Wqkv + (h @ loras['qkv'][0].T) @ loras['qkv'][1].T
q = qkv[:, :D].reshape(T, NH, HD).transpose(0, 1)
k = qkv[:, D:2 * D].reshape(T, NH, HD).transpose(0, 1)
v = qkv[:, 2 * D:].reshape(T, NH, HD).transpose(0, 1)
hd2 = HD // 2
inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
ang = torch.from_numpy(np.arange(T, dtype=np.float64)[:, None] * inv[None, :])
c, s = torch.cos(ang).float(), torch.sin(ang).float()
def rope(tens):
    a, b = tens[..., :hd2], tens[..., hd2:]
    return torch.cat([a * c[None] - b * s[None], a * s[None] + b * c[None]], dim=-1)
qr, kr = rope(q), rope(k)
sm = (qr @ kr.transpose(-1, -2)) * (1.0 / np.sqrt(HD))
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
sm = sm.masked_fill(~mask, float('-inf'))
pr = torch.softmax(sm, -1)
o = pr @ v
o_tok = o.transpose(0, 1).reshape(T, D)
attn = o_tok @ w['o'] + (o_tok @ loras['o'][0].T) @ loras['o'][1].T
x1 = xt + attn
h2 = rms(x1, w['n2'])
gate = h2 @ w['g'] + (h2 @ loras['gate'][0].T) @ loras['gate'][1].T
up = h2 @ w['u'] + (h2 @ loras['up'][0].T) @ loras['up'][1].T
gz = torch.nn.functional.silu(gate) * up
y = gz @ w['d'] + (gz @ loras['down'][0].T) @ loras['down'][1].T
out = x1 + y
loss = (out * torch.from_numpy(dOut)).sum()
loss.backward()

err = float(np.abs(dx - xt.grad.numpy()).max())
print('dX: max abs err %.3e (relative %.3e)'
      % (err, err / (float(np.abs(xt.grad.numpy()).max()) + 1e-9)))
for name in loras:
    a, b = loras[name]
    ga, gb = grads[name]
    ea = float(np.abs(ga - a.grad.numpy()).max())
    eb = float(np.abs(gb - b.grad.numpy()).max())
    print('%-5s dA err %.3e | dB err %.3e' % (name, ea, eb))
