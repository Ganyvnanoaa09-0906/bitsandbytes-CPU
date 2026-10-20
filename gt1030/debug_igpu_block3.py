"""debug_igpu_block3.py — compare all submodules of transformer_blocks.3 CPU vs hybrid."""
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

block = m.transformer_blocks[3]


def run_and_capture():
    out = {}

    def rel_name(name):
        return name

    hooks = []
    for name, mod in block.named_modules():
        if name == '':
            continue

        def mk(n):
            def h(module, inp, o):
                t = o[0] if isinstance(o, tuple) else o
                if torch.is_tensor(t):
                    out[n] = t.detach()
            return h
        hooks.append(mod.register_forward_hook(mk(name)))
    with torch.no_grad():
        m(hidden_states=hs, encoder_hidden_states=enc, timestep=ts,
          img_shapes=shapes, img_mask=mask, return_dict=False)
    for h in hooks:
        h.remove()
    return out


ref = run_and_capture()
print('cpu captured', len(ref), flush=True)
to_igpu_f32(m)
got = run_and_capture()
print('gpu captured', len(got), flush=True)

rows = []
for name in ref:
    if name not in got:
        continue
    r, g = ref[name], got[name]
    rel = float((g - r).norm() / (r.norm() + 1e-9))
    rows.append((rel, name, float(r.std()), float(g.std())))
rows.sort(reverse=True)
for rel, name, rs, gs in rows[:20]:
    print('%-45s rel %.4f | ref_std %.3f got_std %.3f' % (name, rel, rs, gs))
