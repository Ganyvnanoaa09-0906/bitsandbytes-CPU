"""debug_igpu_blocks.py — compare per-transformer-block outputs CPU vs iGPU hybrid."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU\gt1030')
from qwen4bit_loader import load_transformer
from qwen4bit_igpu_linear_fp32 import to_igpu_f32

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

blocks = [mod for n, mod in m.named_modules() if n.startswith('transformer_blocks.') and
          n.count('.') == 1]
print('blocks', len(blocks), flush=True)
ref = {}
hooks = []


def mk(i):
    def h(module, inp, out):
        ref[i] = out.detach() if torch.is_tensor(out) else out[0].detach()
    return h


for i, b in enumerate(blocks):
    hooks.append(b.register_forward_hook(mk(i)))
with torch.no_grad():
    m(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
      img_shapes=shapes, img_mask=mask, return_dict=False)
for h in hooks:
    h.remove()
print('cpu blocks captured', len(ref), flush=True)

to_igpu_f32(m)
got = {}
hooks = []


def mk2(i):
    def h(module, inp, out):
        got[i] = out.detach() if torch.is_tensor(out) else out[0].detach()
    return h


for i, b in enumerate(blocks):
    hooks.append(b.register_forward_hook(mk2(i)))
with torch.no_grad():
    m(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
      img_shapes=shapes, img_mask=mask, return_dict=False)
for h in hooks:
    h.remove()

for i in sorted(ref):
    r = ref[i]
    g = got.get(i)
    if g is None:
        print(i, 'missing')
        continue
    rel = float((g - r).norm() / (r.norm() + 1e-9))
    print('block %2d rel %.5f | ref std %.3f got std %.3f' % (i, rel, float(r.std()), float(g.std())))
