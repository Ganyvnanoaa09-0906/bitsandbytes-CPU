"""debug_igpu_layerwise.py — hook every linear and find the first diverging layer."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
from qwen4bit_igpu_linear_fp32 import GPU4BitLinearF32, to_igpu_f32

torch.set_num_threads(6)
torch.manual_seed(0)
m = load_transformer()
S, T = 64, 8
hs = torch.randn(1, S, 64)
enc = torch.randn(1, T, 4096)
ts = torch.tensor([0.5])
shapes = [[(1, 8, 8)]]
mask = torch.cat([torch.zeros(1, T, dtype=torch.bool),
                  torch.ones(1, S // 4, dtype=torch.bool)], dim=1)

refs = {}
hooks = []


def mk_hook(name):
    def h(module, inp, out):
        if torch.is_tensor(out):
            refs[name] = out.detach()
    return h


for name, mod in m.named_modules():
    if isinstance(mod, torch.nn.Linear):
        hooks.append(mod.register_forward_hook(mk_hook(name)))
with torch.no_grad():
    m(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
      img_shapes=shapes, img_mask=mask, return_dict=False)
for h in hooks:
    h.remove()
print('cpu refs', len(refs), flush=True)

to_igpu_f32(m)
outs = {}


def mk_hook2(name):
    def h(module, inp, out):
        if torch.is_tensor(out):
            outs[name] = out.detach()
    return h


hooks = []
for name, mod in m.named_modules():
    if isinstance(mod, GPU4BitLinearF32):
        hooks.append(mod.register_forward_hook(mk_hook2(name)))
with torch.no_grad():
    m(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
      img_shapes=shapes, img_mask=mask, return_dict=False)
for h in hooks:
    h.remove()

bad = []
for name, r in refs.items():
    g = outs.get(name)
    if g is None:
        continue
    rel = float((g - r).norm() / (r.norm() + 1e-9))
    if rel > 0.01:
        bad.append((name, rel, float(r.std()), float(g.std())))
print('bad layers', len(bad), '/', len(refs))
for b in bad[:15]:
    print('%-60s rel %.4f ref_std %.3f got_std %.3f' % b)
