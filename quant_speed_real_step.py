"""quant_speed_real_step.py -- does narrow weight storage actually make a training step faster?

The reasoning that says it might:

    the CPU's measured arithmetic peak is 269 GFLOPS at n=768, whose working set is 6.9 MB,
    just inside the 8 MB L3. Larger matrices fall to 225-230, and training sustains 198.5.
    So the limit is data supply, not the FMA units -- and Zen2 issues one 256-bit FMA per
    cycle per core, not two, so there is no arithmetic headroom to chase either.

    Arithmetic intensity is FLOPs per byte, so reducing the bytes is the lever. A 3-bit
    weight is 0.4375 bytes against 4, a factor of 9. If the dequantisation is done per layer
    and only one layer's worth of fp32 exists at a time, the resident working set shrinks
    even though the arithmetic is unchanged.

The counter-argument is equally real: dequantising costs work, and 2.56x was measured for
the 4-bit optimiser kernel doing similar unpacking. Whether the traffic saving exceeds that
cost is an empirical question with no shortcut.

So: one real training step, forward and backward, on the actual model, with weights stored
at 32, 4, 3 and 2 bits. Dequantisation is not cached, because caching would restore the
full fp32 working set and defeat the point. Reported as seconds per step and as effective
GFLOPS, plus forward and backward separately since the backward pass has the larger live
set.

Correctness of the quantised paths is already established (7.23): 3-bit and 2-bit reproduce
independently measured losses. This is about speed only.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402
# quant_lora lives in this checkout and now carries the nf2 branch; quant2bit
# is alongside it, so a plain import resolves both.
from quant_lora import quantize_model_8bit_lora as _quantize

CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
torch.set_num_threads(6)


def build():
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items()
                                if k in names})
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(ck['model'], strict=False)
    return m, cfg


def step_time(m, cfg, x, reps=3, train=True):
    """Time forward (+backward) on a fixed batch, forwarding and backward separately."""
    tf, tb = [], []
    params = [p for p in m.parameters() if p.requires_grad]
    for _ in range(reps):
        m.train(train)
        t0 = time.perf_counter()
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        t1 = time.perf_counter()
        if train:
            if params:
                for p in params:
                    p.grad = None
            loss.backward()
        t2 = time.perf_counter()
        tf.append(t1 - t0)
        tb.append(t2 - t1)
    return statistics.median(tf), statistics.median(tb)


tok = torch.load(TOKENS, map_location='cpu')
x = tok[:8].long()          # batch 8, the training batch size
BT = x.shape[0] * (x.shape[1] - 1)

ARMS = [
    ('fp32 基线', 'none'),
    ('3 bit 权重（每步解包）', 'uni3'),
    ('2 bit 权重（每步解包）', 'nf2'),
]
print('batch 8, T=1024，%d 个预测位置' % BT)
print()
print('%-24s %9s %9s %9s %11s' % ('配置', '前向 s', '反向 s', '合计 s', '等效 GFLOPS'))
print('-' * 70)
base = None
for label, mode in ARMS:
    m, cfg = build()
    if mode != 'none':
        # lora_r must be >= 1: the scaling factor divides by it
        n, saved = _quantize(m, lora_r=1, lora_alpha=1,
                             cache_dequant=False, quant_dtype=mode)
        print('  （%s: 替换 %d 层，省 %.1f MB）' % (label, n, saved / 2**20))
    tf, tb = step_time(m, cfg, x)
    tot = tf + tb
    # forward+backward is about 3x the forward FLOPs; report the forward-equivalent rate
    flops = 2.0 * 30.5e6 * BT * 3
    gf = flops / tot / 1e9
    if base is None:
        base = tot
    print('%-24s %9.3f %9.3f %9.3f %11.1f   %s'
          % (label, tf, tb, tot, gf, '' if base == tot else '%.2fx' % (base / tot)))
    del m
print()
print('判据: 若窄存储的合计时间【低于】fp32，说明省下的流量大于解包的开销；')
print('      否则说明解包代价更高，量化提速的假设不成立。')
