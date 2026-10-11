"""What does the loop transformer actually cost per training step?

Prediction from the config: the looped range is layers [4..7] repeated L times,
so the block-pass count is 6 + 4L:
    L=1 -> 10 passes (1.00x)   L=2 -> 14 (1.40x)   L=3 -> 18 (1.80x)
The training config uses --loop_random with L in 1..3, i.e. an average around
L=2 => ~1.4x the FLOPs of L=1. If that holds, pinning L=1 is a ~1.4x training
speedup at zero code cost.

Why a direct microbenchmark instead of the training loop: an earlier A/B through
train_ar_v2.py was ruined by validation overhead and process warm-up (it reported
4.88 s/step for L=1, worse than the real training's 2.854 s/step for random L,
which is impossible). Here only the model's fwd+bwd is timed -- no val, no
checkpointing, no data loading, no warm-up ambiguity beyond explicit warm-up
iterations.

Leak check: this measures TIME only, never quality, so the random-L regulariser's
effect on generalisation is not assessed here.
"""
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from loop_transformer import LoopConfig                    # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402

CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt\ckpt_last.pt'
T, B = 256, 8

torch.set_num_threads(6)
print('threads =', torch.get_num_threads())

ck = torch.load(CK, map_location='cpu', weights_only=False)
cfgd = ck['cfg']
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
model = SmallARImageModelV2(cfg)
sd = ck['model']
if any(k.startswith('loop_mods.') for k in sd):
    lmax = 3
    for k, v in sd.items():
        if k.endswith('step_enc.table.weight') or k.endswith('cross_res.gate'):
            lmax = max(1, v.shape[0] - 1)
            break
    print('ckpt 有循环模块，L_max =', lmax)
    # MUST mount loop_mods BEFORE load_state_dict -- the modules do not exist
    # until set_loop builds them, otherwise load reports them as unexpected keys.
    # (This exact ordering was already fixed once today in sample_ar_cond.py and
    # then forgotten here.)
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=lmax, loop_L_max=lmax, loop_random=False,
                              loop_cache='perloop'))
model.load_state_dict(sd)
model.train()

x = torch.randint(0, cfg.vocab_size, (B, T))
params = [p for p in model.parameters() if p.requires_grad]
print('可训练参数 %.2f M' % (sum(p.numel() for p in params) / 1e6))

def _t(o):
    """Unwrap until a tensor shows up -- forward_logits returns a nested tuple
    (first attempt assumed one level; it is deeper, which cost two runs)."""
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o


print('\n%-8s %10s %10s %10s' % ('loop_L', '秒/步', '相对 L=1', '理论 pass 比'))
print('-' * 42)
base = None
res = {}
for L in (1, 2, 3):
    model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=L, loop_L_max=L, loop_random=False,
                              loop_cache='perloop'))
    # warm-up (allocator + oneDNN primitives)
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        _t(model.forward_logits(x, None)).float().pow(2).mean().backward()
    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        model.zero_grad(set_to_none=True)
        _t(model.forward_logits(x, None)).float().pow(2).mean().backward()
    dt = (time.perf_counter() - t0) / n
    if base is None:
        base = dt
    pred = (6 + 4 * L) / 10.0
    res[L] = dt
    print('%-8d %10.3f %9.2fx %9.2fx' % (L, dt, dt / base, pred))

print('\n判读：')
print('  实测比值若与"理论 pass 比"接近 ⇒ pass 数模型成立 ⇒ 固定 L=1 可省 %.0f%%'
      % ((1 - res[1] / res[2]) * 100 if 2 in res else 0))
print('  若接近 1.0x ⇒ 循环几乎不花钱（说明瓶颈不在这些 pass 的 FLOPs）')
