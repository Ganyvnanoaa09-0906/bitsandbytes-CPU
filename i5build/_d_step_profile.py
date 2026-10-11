import os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')
from loop_transformer import LoopConfig
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2
torch.set_num_threads(6)

CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt\ckpt_last.pt'
T, B = 256, 8
ck = torch.load(CK, map_location='cpu', weights_only=False)
cfgd = ck['cfg']
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in cfgd.items() if k in keep})
model = SmallARImageModelV2(cfg)
sd = ck['model']
_lmax = 3
for _k, _v in sd.items():
    if _k.endswith('step_enc.table.weight') or _k.endswith('cross_res.gate'):
        _lmax = max(1, _v.shape[0] - 1)
        break
model.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=_lmax, loop_L_max=_lmax, loop_random=False, loop_cache='perloop'))
model.load_state_dict(sd)
model.train()
x = torch.randint(0, cfg.vocab_size, (B, T))


def _t(o):
    while isinstance(o, (tuple, list)):
        o = o[0]
    return o


def fwd():
    y = _t(model.forward_logits(x, None)).float().pow(2).mean()
    return y


def step():
    model.zero_grad(set_to_none=True)
    y = fwd()
    y.backward()


def timeit(fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


tf = timeit(fwd)
tt = timeit(step)
print('forward  %.3f s' % tf)
print('完整步   %.3f s   => backward ≈ %.3f s (%.0f%%)' % (tt, tt - tf, (tt - tf) / tt * 100))

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
    step()
print()
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=24))
