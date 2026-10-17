"""test_quantnbit_accept.py -- 3-bit and 2-bit checked against numbers measured earlier.

Both expectations come from quantize_model_eval.py, which quantised every ndim>=2
tensor blockwise with absmax 64 and evaluated on the last 512 rows of the token file:

    3 bit uniform    5.5234
    2 bit nf         5.6206

The 2-bit arm is a regression test: quantnbit.py replaces the packing code that
produced 5.6198 through linear2bit.py, so if it does not land back near 5.6206 the
general packer has changed behaviour. The 3-bit arm is the new case, and its
interesting property is that it exercises a byte-spanning bit layout -- 3 does not
divide 8 -- which the self-test covers synthetically and this covers on real data.

Anything else, including "it ran and the number looks plausible", is not a pass.
"""
from __future__ import annotations

import dataclasses
import sys

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


import quantnbit as qn  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)
CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
val = torch.load(TOKENS, map_location='cpu')[-512:].long()


def build():
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(ck['model'], strict=False)
    return m


def quantize_all(m, bits, table):
    packed_bytes = 0
    fp_bytes = 0
    for name, child in list(m.named_children()):
        if isinstance(child, torch.nn.Linear):
            w = child.weight.data
            pk, am, meta = qn.quantize(w, bits, table=table)
            setattr(m, name, _QLayer(pk, am, meta, child.bias))
            packed_bytes += pk.numel() + am.numel() * 4
            fp_bytes += w.numel() * 4
        elif isinstance(child, torch.nn.Embedding):
            pk, am, meta = qn.quantize(child.weight.data, bits, table=table)
            setattr(m, name, _ELayer(pk, am, meta))
            packed_bytes += pk.numel() + am.numel() * 4
            fp_bytes += child.weight.numel() * 4
        else:
            pb, fb = quantize_all(child, bits, table)
            packed_bytes += pb
            fp_bytes += fb
    return packed_bytes, fp_bytes


class _QLayer(torch.nn.Module):
    def __init__(self, pk, am, meta, bias):
        super().__init__()
        self.register_buffer('pk', pk)
        self.register_buffer('am', am)
        self.register_buffer('bias', None if bias is None else bias.detach().float().clone())
        self.meta = meta

    def forward(self, x):
        return torch.nn.functional.linear(x, qn.dequantize(self.pk, self.am, self.meta),
                                          self.bias)


class _ELayer(torch.nn.Module):
    def __init__(self, pk, am, meta):
        super().__init__()
        self.register_buffer('pk', pk)
        self.register_buffer('am', am)
        self.meta = meta

    def forward(self, idx):
        return torch.nn.functional.embedding(idx, qn.dequantize(self.pk, self.am, self.meta))


def evaluate(m):
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


ARMS = [
    ('fp32 对照', None, None, 5.5038),
    ('2 bit nf（回归）', 2, qn.nf_table(2), 5.6206),
    ('3 bit uniform（新）', 3, qn.uniform_table(3), 5.5234),
    ('3 bit nf', 3, qn.nf_table(3), 5.5231),
]
print('%-20s %10s %10s %9s %s' % ('配置', 'val loss', '期望', '差', '判定'))
print('-' * 70)
for label, bits, table, expect in ARMS:
    m = build()
    if bits is None:
        pb, fb = 0, 0
    else:
        pb, fb = quantize_all(m, bits, table)
    vl = evaluate(m)
    ok = abs(vl - expect) < 0.01
    print('%-20s %10.4f %10.4f %+9.4f %s'
          % (label, vl, expect, vl - expect, '✓ 一致' if ok else '✗ 不一致'))
    if bits:
        print('%-20s 打包 %.2f MB vs fp32 %.2f MB ⇒ %.1fx（每权重 %.4f 字节）'
              % ('', pb / 2**20, fb / 2**20, fb / max(pb, 1), pb / (fb / 4)))
    del m
