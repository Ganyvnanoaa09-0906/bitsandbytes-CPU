"""
Where do the 746 aten::mul calls per training step come from?

The profiler would not report input shapes for key_averages() in this torch
build, so intercept the dispatch instead and bucket every elementwise op by its
exact argument shapes.  One warm-up step, then one step under the spy.

Output is (op, shapes) -> count plus the total self time the profiler attributes
to that op, so the biggest buckets are obvious.
"""
import os, sys
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys
from collections import Counter

import torch
from torch.utils._python_dispatch import TorchDispatchMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2, build_cluster_map
from scripts.lib.small_vqvae import VQVAE
import bitsandbytes

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".autopilot")
torch.set_num_threads(5)
torch.backends.mkldnn.enabled = True

tokens = torch.load(os.path.join(BASE, "tokens_final2.pt"), map_location="cpu")
N = tokens.shape[0]
vk = torch.load(os.path.join(BASE, "vqvae_ema_full2", "ckpt_last.pt"),
                map_location="cpu", weights_only=False)
vq = VQVAE(embed_dim=64, n_codes=1024)
vq.load_state_dict(vk["model"])
cluster_of, offset_of, per = build_cluster_map(vq.vq.codebook.weight.data, 32)
n_cluster = int(cluster_of.max().item()) + 1
del vq

cfg = SmallImageConfigV2(d_model=512, n_layer=10, n_head=8, n_kv_head=4, ffn_dim=2048,
                         n_cluster=n_cluster, cond_mode="none", head_mode="flat")
model = SmallARImageModelV2(cfg)
opt = bnb.optim.AdamW8bit(model.parameters(), lr=3e-4)
B, T = 8, 256
print("params=%.2fM  vocab=%d  n_cluster=%d  per=%d" %
      (sum(p.numel() for p in model.parameters()) / 1e6, cfg.vocab_size, n_cluster, per))


def one_step():
    idx = torch.randint(0, N, (B,))
    x = tokens[idx, :T].long()
    th, ge = model.forward_logits(x, None)
    if isinstance(ge, tuple) and ge[1] is not None:
        gec, ged = ge
        loss = (torch.nn.functional.cross_entropy(gec[:, :-1].reshape(-1, n_cluster),
                                                  cluster_of[x][:, 1:].reshape(-1))
                + torch.nn.functional.cross_entropy(ged[:, :-1].reshape(-1, per),
                                                    offset_of[x].clamp(max=per - 1)[:, 1:].reshape(-1)))
    else:
        g0 = ge[0] if isinstance(ge, tuple) else ge
        loss = torch.nn.functional.cross_entropy(g0[:, :-1].reshape(-1, cfg.vocab_size),
                                                 x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


WATCH = {"mul", "sum", "pow", "div", "mean", "rsqrt", "add", "silu", "copy_", "cat"}


class Spy(TorchDispatchMode):
    def __init__(self):
        self.buckets = Counter()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.__name__
        base = name.split(".")[0]
        if base in WATCH:
            shapes = []
            for a in args:
                if isinstance(a, torch.Tensor):
                    shapes.append(tuple(a.shape))
                elif isinstance(a, (int, float)):
                    shapes.append(("scalar",))
            shapes = tuple(shapes[:3])
            self.buckets[(f"aten::{name.split('.')[-1]}", shapes)] += 1
        return func(*args, **kwargs)


s = Spy()
for _ in range(1):
    one_step()
with s:
    one_step()

print()
print("=== 每个元素级算子按【参数形状】分桶（单步）===")
for (op, shapes), c in s.buckets.most_common(28):
    sh = "  ".join(str(x) for x in shapes)
    print(f"  {c:6d}  {op:14s} {sh[:130]}")

print()
tot = sum(s.buckets.values())
print(f"被监视算子的总调用数 = {tot}（单步）")
by_op = Counter()
for (op, _), c in s.buckets.items():
    by_op[op] += c
for op, c in by_op.most_common():
    print(f"  {c:6d}  {op}")
