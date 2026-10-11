"""linear2bit.py -- a Linear layer whose weights live at 2.5 bits per parameter.

The format is already characterised, so this only has to be faithful to it:

    quantize_model_eval.py, on the trained 32x32 AR checkpoint
        fp32          5.5038
        2 bit nf      5.6206      <- the number this module must reproduce
        2 bit uniform 5.6354

That pre-measured figure is the acceptance test. A quantiser that "looks right" but
lands somewhere else has a bug -- wrong bit order, wrong block scale, wrong levels --
and comparing against a number produced by an independent implementation is the only
way to catch it. Today's earlier failures were all of this shape: a result believed
without a check that could have contradicted it.

Design: the weight is stored packed (4 indices per byte) plus one fp32 absmax per
block of 64, and dequantised on the fly in forward. No backward support yet -- this
is the forward-only validation step; training through it needs the dequantisation to
be differentiable, which is a separate question from whether the format works.

Memory at 2.5 bits per weight against 32 for fp32 is 12.8x; combined with the 4-bit
optimiser state (1.03 bytes/param, measured previously) the whole training footprint
becomes about 1.34 bytes per parameter, which is what makes a 3B model fit on this
machine at all.
"""
from __future__ import annotations

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bnb-quant')
sys.stdout.reconfigure(encoding='utf-8')

from quant2bit import BLOCK, NF2, UNIFORM2, bytes_per_weight, dequantize, quantize  # noqa: E402


class Linear2bit(nn.Module):
    """y = x @ W^T + b, with W stored at 2 bits per weight."""

    def __init__(self, weight: torch.Tensor, bias=None, table=None):
        super().__init__()
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        packed, absmax, n, pad, shape = quantize(weight.detach().float(), table)
        self.register_buffer('packed', packed)
        self.register_buffer('absmax', absmax)
        self.register_buffer('bias', None if bias is None else bias.detach().float().clone())
        self._n, self._pad, self._shape = n, pad, shape
        self._table = NF2 if table is None else table

    def weight(self) -> torch.Tensor:
        return dequantize(self.packed, self.absmax, self._n, self._pad, self._shape,
                          self._table)

    def forward(self, x):
        return F.linear(x, self.weight(), self.bias)

    def extra_repr(self):
        return ('in=%d, out=%d, %d bytes packed (%.2f bit/weight, %.1fx vs fp32)'
                % (self.in_features, self.out_features, self.packed.numel(),
                   bytes_per_weight() * 8, 4.0 / bytes_per_weight()))


def convert(model: nn.Module, table=None, include_emb=False):
    skip_names = () if include_emb else ('token_emb', 'row_emb', 'col_emb',
                                         'pos_emb', 'step_emb')
    """Replace every nn.Linear whose weight is 2-D with a Linear2bit, in place."""
    replaced, kept = 0, 0
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear) and child.weight.ndim == 2:
            if any(s in name for s in skip_names):
                kept += 1
                continue
            setattr(model, name, Linear2bit(child.weight.data, child.bias, table))
            replaced += 1
        else:
            r, k = convert(child, table, skip_names)
            replaced += r
            kept += k
    return replaced, kept


def footprint(model: nn.Module) -> dict:
    packed = sum(m.packed.numel() for m in model.modules() if isinstance(m, Linear2bit))
    scales = sum(m.absmax.numel() * 4 for m in model.modules() if isinstance(m, Linear2bit))
    other = 0
    for m in model.modules():
        if isinstance(m, Linear2bit):
            if m.bias is not None:
                other += m.bias.numel() * 4
            continue
        for p in m.parameters(recurse=False):
            other += p.numel() * p.element_size()
    return dict(packed=packed, scales=scales, other=other, total=packed + scales + other)


if __name__ == '__main__':
    import dataclasses
    sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
    from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2
    from loop_transformer import LoopConfig

    torch.set_num_threads(6)
    CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
    TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
    EXPECT_NF = 5.6206          # measured independently in quantize_model_eval.py
    EXPECT_UNI = 5.6354

    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
    cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
    tok = torch.load(TOKENS, map_location='cpu')
    val = tok[-512:].long()

    def build(table=None, include_emb=False):
        m = SmallARImageModelV2(cfg)
        m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                              loop_L=1, loop_L_max=3, loop_random=False,
                              loop_cache='shared'))
        m.load_state_dict(ck['model'], strict=False)
        r, k = convert(m, table, include_emb)
        m.eval()
        return m, r, k

    def evaluate(m):
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

    print('判据: 独立实现测得的 2 bit nf = %.4f，uniform = %.4f' % (EXPECT_NF, EXPECT_UNI))
    print()
    # scope matters: the study quantised every ndim>=2 tensor including embeddings
    for label, table, expect, inc in (
            ('fp32 对照', None, 5.5038, False),
            ('2 bit nf（含 emb）', NF2, EXPECT_NF, True),
            ('2 bit nf（不含 emb）', NF2, None, False),
            ('2 bit uniform（含 emb）', UNIFORM2, EXPECT_UNI, True)):
        if table is None:
            m = SmallARImageModelV2(cfg)
            m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                                  loop_L=1, loop_L_max=3, loop_random=False,
                                  loop_cache='shared'))
            m.load_state_dict(ck['model'], strict=False)
            m.eval()
            r, k = 0, 0
        else:
            m, r, k = build(table, inc)
        fp = footprint(m)
        vl = evaluate(m)
        if expect is None:
            print('%-24s val %.4f   （无对应期望值，供对比）' % (label, vl))
        else:
            print('%-24s val %.4f   期望 %.4f   差 %+.4f %s'
                  % (label, vl, expect, vl - expect,
                     '✓ 与独立实现一致' if abs(vl - expect) < 0.01 else '✗ 不一致'))
        if r:
            n_w = sum(mm.packed.numel() for mm in m.modules() if isinstance(mm, Linear2bit))
            print('%-24s 替换 %d 个 Linear，保留 %d 个；量化后 %.1f MB'
                  % ('', r, k, fp['total'] / 2**20))
        print()
