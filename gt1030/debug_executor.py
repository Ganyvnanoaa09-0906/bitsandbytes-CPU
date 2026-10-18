"""Debug the executor's forward parity step by step (small config, identical weights)."""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
import gt1030.ar_step_executor as ex

torch.set_grad_enabled(False)
D, FF, NL, NH, NKV, VOC, B, T = 64, 128, 2, 2, 1, 64, 2, 32
torch.manual_seed(0)
ref = ex.cpu_ref_model(D, FF, NL, NH, NKV, VOC)
torch.manual_seed(0)
ref2 = ex.cpu_ref_model(D, FF, NL, NH, NKV, VOC)
print('ref == ref2:', all(torch.equal(a, b) for a, b in zip(ref.parameters(), ref2.parameters())))

ig = ex.MiniAR(D, FF, NL, NH, NKV, VOC)
ex.sync_ref_weights(ref, ig)
inp = torch.randint(0, VOC, (B, T))

# embedding parity
a_emb = ref.emb(inp)
b_emb = ig.emb(inp)
print('emb max diff:', float((a_emb - b_emb).abs().max()))

# first block parity
a1 = ref.blocks[0](a_emb)
b1 = ig.blocks[0](b_emb)
print('block0 max diff:', float((a1 - b1).abs().max()))

# inside block: after ln1
ha = ref.blocks[0].ln1(a_emb)
hb = ig.blocks[0].ln1(b_emb)
print('ln1 max diff:', float((ha - hb).abs().max()))

# qkv projections
qa_ref = ref.blocks[0].qi(ha)
ka_ref = ref.blocks[0].ki(ha)
va_ref = ref.blocks[0].vi(ha)
qkv_ig = ig.blocks[0].qkv(hb)
print('qkv shapes:', qkv_ig.shape, 'q ref:', qa_ref.shape)
print('q max diff:', float((qa_ref - qkv_ig[..., :D]).abs().max()))
print('k max diff:', float((ka_ref - qkv_ig[..., D:D + NKV * (D // NH)]).abs().max()))
print('v max diff:', float((va_ref - qkv_ig[..., D + NKV * (D // NH):]).abs().max()))

# full forward logits
la = ref(inp)
lb = ig(inp)
print('full logits max diff:', float((la - lb).abs().max()))
