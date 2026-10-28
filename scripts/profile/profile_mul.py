"""
Pin down the 560 aten::mul calls per training step (10.9% of self CPU).

Same model/config as profile_train_step.py, but the profiler runs with
record_shapes=True and the table is filtered to the elementwise ops, so the
shapes identify which part of the model is doing them.
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

import torch

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
print("params=%.2fM vocab=%d n_cluster=%d per=%d" %
      (sum(p.numel() for p in model.parameters()) / 1e6, cfg.vocab_size, n_cluster, per))


def one_step():
    idx = torch.randint(0, N, (B,))
    x = tokens[idx, :T].long()
    th, ge = model.forward_logits(x, None)
    if isinstance(ge, tuple) and ge[1] is not None:
        gec, ged = ge
        tgt_c = cluster_of[x]
        tgt_o = offset_of[x].clamp(max=per - 1)
        loss = (torch.nn.functional.cross_entropy(gec[:, :-1].reshape(-1, n_cluster),
                                                  tgt_c[:, 1:].reshape(-1))
                + torch.nn.functional.cross_entropy(ged[:, :-1].reshape(-1, per),
                                                    tgt_o[:, 1:].reshape(-1)))
    else:
        g0 = ge[0] if isinstance(ge, tuple) else ge
        loss = torch.nn.functional.cross_entropy(g0[:, :-1].reshape(-1, cfg.vocab_size),
                                                 x[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


from torch.profiler import profile, ProfilerActivity  # noqa: E402

for _ in range(2):
    one_step()

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(2):
        one_step()

print()
print("=== 逐算子（带形状），按 self CPU 排序，只看非 mm 的 ===")
evs = sorted(prof.key_averages(), key=lambda e: -e.self_cpu_time_total)
for e in evs:
    if e.self_cpu_time_total <= 0:
        continue
    name = e.key
    if "mm" in name and "mul" not in name:
        continue
    shapes = ""
    try:
        if e.input_shapes:
            shapes = str(e.input_shapes[:2])
    except Exception:
        pass
    print(f"  {e.self_cpu_time_total/1e6:9.3f}s  x{e.count:6d}  {name:52s} {shapes[:110]}")

print()
print("=== 只看 aten::mul / aten::add / aten::sum / aten::pow 的输入形状分布 ===")
for e in prof.key_averages():
    if e.key in ("aten::mul", "aten::add", "aten::sum", "aten::pow", "aten::div",
                 "aten::rsqrt", "aten::mean"):
        try:
            shp = getattr(e, "input_shapes", None)
        except Exception:
            shp = None
        print(f"  {e.key:14s} x{e.count:5d}  self={e.self_cpu_time_total/1e6:7.3f}s  "
              f"shapes={str(shp)[:160]}")
