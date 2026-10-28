"""Debug the llama engine one-layer forward stage by stage against torch."""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.llama_resident as E

eng = E.Engine(r'D:\deepsleep\_test\models\deepseek-coder-1.3b-base', 64, limit_layers=1, keep_host=True)
g = np.random.default_rng(0)
x = (g.standard_normal((eng.M, eng.DH)) * 0.1).astype(np.float32)
E.cl.enqueue_copy(E.queue, eng.xb[0], x)
eng.forward_layer(0, eng.xb[0], eng.xb[1])
E.queue.finish()
out = E.down(eng.xb[1], (eng.M, eng.DH))
a = eng.acts[0]

def dl(name, shape):
    return E.down(a[name], shape)

h = dl('h', (eng.M, eng.DH))
qkv = dl('qkv', (eng.M, 3 * eng.DH))
Q = dl('Q', (eng.BH, eng.T, eng.HD))
Kk = dl('K', (eng.BH, eng.T, eng.HD))
V = dl('V', (eng.BH, eng.T, eng.HD))
S = dl('S', (eng.BH, eng.T, eng.T))
P = dl('P', (eng.BH, eng.T, eng.T))
o_tok = dl('o_tok', (eng.M, eng.DH))
x1 = dl('x1', (eng.M, eng.DH))
h2 = dl('h2', (eng.M, eng.DH))
ga = dl('a', (eng.M, eng.FF))
gb = dl('b', (eng.M, eng.FF))
gz = dl('g', (eng.M, eng.FF))

hw = eng.host[0]
xt = torch.from_numpy(x)
wr = lambda k: torch.from_numpy(hw[k])
n1 = wr('n1')
href = xt * torch.rsqrt(xt.pow(2).mean(-1, keepdim=True) + eng.EPS) * n1
print('rms h err:', np.abs(h - href.numpy()).max())

qkvr = href @ torch.from_numpy(np.concatenate([hw['q'].T, hw['k'].T, hw['v'].T], axis=1))
print('qkv err:', np.abs(qkv - qkvr.numpy()).max())

T, HD, NH, D = eng.T, eng.HD, eng.NH, eng.DH
q = qkvr[:, :D].reshape(T, NH, HD).transpose(0, 1)
k = qkvr[:, D:2 * D].reshape(T, NH, HD).transpose(0, 1)
v = qkvr[:, 2 * D:].reshape(T, NH, HD).transpose(0, 1)
hd2 = HD // 2
inv = 1.0 / (eng.THETA ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
ang = torch.from_numpy(np.arange(T, dtype=np.float64)[:, None] * inv[None, :])
c, s = torch.cos(ang).float(), torch.sin(ang).float()
def rope(tens):
    x0, x1_ = tens[..., :hd2], tens[..., hd2:]
    return torch.cat([x0 * c[None] - x1_ * s[None], x0 * s[None] + x1_ * c[None]], dim=-1)
qr, kr = rope(q), rope(k)
print('Q err:', np.abs(Q - qr.numpy()).max(), 'K err:', np.abs(Kk - kr.numpy()).max(),
      'V err:', np.abs(V - v.numpy()).max())

sm = (qr @ kr.transpose(-1, -2)) * (1.0 / np.sqrt(HD))
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
sm = sm.masked_fill(~mask, float('-inf'))
pr = torch.softmax(sm, -1)
print('S err:', np.abs(S - sm.masked_fill(~mask, 0).numpy()).max())
print('P err:', np.abs(P - pr.numpy()).max())
orv = pr @ v
print('O ref vs engine: o_tok err:', np.abs(o_tok - orv.transpose(0, 1).reshape(T, D).numpy()).max())
og = orv.transpose(0, 1).reshape(T, D) @ torch.from_numpy(hw['o'].T)
print('x1 err:', np.abs(x1 - (xt + og).numpy()).max())
h2r = (xt + og) * torch.rsqrt((xt + og).pow(2).mean(-1, keepdim=True) + eng.EPS) * torch.from_numpy(hw['n2'])
print('h2 err:', np.abs(h2 - h2r.numpy()).max())
gar = h2r @ torch.from_numpy(hw['g'].T)
gbr = h2r @ torch.from_numpy(hw['u'].T)
gref = torch.nn.functional.silu(gar) * gbr
print('gate err:', np.abs(ga - gar.numpy()).max(), 'up err:', np.abs(gb - gbr.numpy()).max())
print('g err:', np.abs(gz - gref.numpy()).max())
yr = gref @ torch.from_numpy(hw['d'].T)
print('out err:', np.abs(out - (xt + og + yr).numpy()).max())
