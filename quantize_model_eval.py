"""quantize_model_eval.py -- does 3-bit or 2-bit weight quantisation cost anything real?

The codebook study only measured a proxy: per-block L2 relative error on weights.
That put uniform ahead of NF4 at 4 bits, which contradicts the published result, and
the reason is that NF4's levels are quantiles -- an information statement about where
probability mass sits -- not a least-squares optimum. A proxy that disagrees with the
literature is not evidence about the thing that matters.

What matters is the model's own loss. This takes the 32x32 AR checkpoint that was
just trained (3000 steps, validation loss 5.5038 measured earlier, which is also the
sanity check for this script) and re-evaluates it with every 2-D weight quantised
blockwise at 8, 4, 3 and 2 bits, using:

  uniform   equal spacing over the per-block range
  nf        normal mid-quantiles, which won at 2 bits in the proxy study

The block size and absmax scaling match what the kernels do, so the numbers describe
the format rather than an idealised version of it. No training is involved: this is
plain evaluation of an existing model, so it is cheap and it isolates the effect of
the format.

Reported: mean validation loss per configuration, plus the delta from fp32 and the
bits per parameter including the scale.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)
CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
N_VAL = 512
BLOCK = 64


def nf_table(levels):
    from statistics import NormalDist
    nd = NormalDist()
    n = 1 << levels
    q = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])
    q = q / np.abs(q).max()
    q[0], q[-1] = -1.0, 1.0
    return np.sort(q).astype(np.float32)


def uniform_table(levels):
    return np.linspace(-1.0, 1.0, 1 << levels).astype(np.float32)


def quantize(w: torch.Tensor, levels, table) -> torch.Tensor:
    """Blockwise absmax quantisation, mirroring the kernel's scheme."""
    flat = w.detach().reshape(-1).numpy().astype(np.float32)
    pad = (-len(flat)) % BLOCK
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    wb = flat.reshape(-1, BLOCK)
    s = np.abs(wb).max(axis=1, keepdims=True)
    s[s == 0] = 1.0
    x = wb / s
    idx = np.abs(x[:, :, None] - table[None, None, :]).argmin(axis=2)
    dq = (table[idx] * s).reshape(-1)[:w.numel()]
    return torch.from_numpy(dq.copy()).reshape(w.shape)


ck = torch.load(CKPT, map_location='cpu', weights_only=False)
sd = ck['model']
cfg_dict = ck.get('cfg') or {}
import dataclasses  # noqa: E402
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in cfg_dict.items() if k in names})

tok = torch.load(TOKENS, map_location='cpu')
val = tok[-N_VAL:].long()
print('模型 %s' % CKPT.split('\\')[-1])
print('验证集 %d 条, 块大小 %d, 参数 %.1fM'
      % (N_VAL, BLOCK, sum(v.numel() for v in sd.values() if torch.is_tensor(v)) / 1e6))
print()
print('%-24s %12s %10s %10s %s' % ('配置', 'val loss', '相对 fp32', '每参位数', '说明'))
print('-' * 78)


def evaluate(state):
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(state, strict=False)
    m.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(val), 8):
            x = val[i:i + 8]
            o = m.forward_logits(x, None)
            lg = o[1]
            lg = lg[0] if isinstance(lg, tuple) else lg
            tot += float(torch.nn.functional.cross_entropy(
                lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1),
                reduction='sum'))
            cnt += x[:, 1:].numel()
    return tot / max(cnt, 1)


base = evaluate({k: v.clone() for k, v in sd.items() if torch.is_tensor(v)})
print('%-24s %12.4f %10s %10s %s' % ('fp32（基线）', base, '-', '32.0', 'sanity: 应约 5.5038'))
print()

for levels in (4, 3, 2):
    for tname, table in (('uniform', uniform_table(levels)), ('nf', nf_table(levels))):
        st = {}
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue
            st[k] = quantize(v, levels, table) if (v.dtype == torch.float32 and v.ndim >= 2) else v
        vl = evaluate(st)
        bpp = levels + 32.0 / BLOCK          # levels per weight + fp32 scale per block
        print('%-24s %12.4f %+10.4f %10.2f %s'
              % ('%d bit %s' % (levels, tname), vl, vl - base, bpp,
                 '✓ 可接受' if vl - base < 0.05 else ('⚠ 有损失' if vl - base < 0.3 else '✗ 损失明显')))
