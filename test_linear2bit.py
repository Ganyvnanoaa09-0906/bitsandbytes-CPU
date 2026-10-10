"""test_linear2bit.py -- the acceptance test, with a reference that actually matches.

Previous attempt compared against 5.6206 while quantising a strictly smaller set of
tensors (convert() only rewrites nn.Linear, and the study quantised every ndim>=2
tensor including embeddings). An "include_emb" flag existed but changed nothing,
because embeddings are not Linears -- so both arms were bit-identical and the
comparison was empty.

Now there are three arms, each with a matching expectation:

    fp32                     5.5038   (known)
    Linears only, 2 bit nf   5.5538   (measured in the previous run)
    2 bit nf, all ndim>=2    5.6206   (the study's number -- the real test)

If the third lands on 5.6206, the quantiser agrees with an independent
implementation and the format is validated end to end. The gap between arms two and
three is the cost of compressing the embedding tables.
"""
from __future__ import annotations

import dataclasses
import sys

import torch
import torch.nn as nn

sys.path.insert(0, r'D:\work\bnb-quant')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from quant2bit import NF2, UNIFORM2  # noqa: E402
from linear2bit import Linear2bit, convert, footprint  # noqa: E402
from embed2bit import Embedding2bit  # noqa: E402
from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)
CKPT = r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt'
TOKENS = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'

ck = torch.load(CKPT, map_location='cpu', weights_only=False)
names = {f.name for f in dataclasses.fields(SmallImageConfigV2)}
cfg = SmallImageConfigV2(**{k: v for k, v in (ck.get('cfg') or {}).items() if k in names})
val = torch.load(TOKENS, map_location='cpu')[-512:].long()


def base_model():
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=False,
                          loop_cache='shared'))
    m.load_state_dict(ck['model'], strict=False)
    m.eval()
    return m


def quantize_embeddings(m, table):
    n = 0
    for name, child in list(m.named_children()):
        if isinstance(child, nn.Embedding):
            setattr(m, name, Embedding2bit.from_embedding(child, table))
            n += 1
        else:
            n += quantize_embeddings(child, table)
    return n


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


ARMS = [
    ('fp32 基线', None, False, 5.5038),
    ('2 bit nf，只 Linear', NF2, False, 5.5538),
    ('2 bit nf，含 embedding', NF2, True, 5.6206),
    ('2 bit uniform，含 embedding', UNIFORM2, True, 5.6354),
]
print('%-26s %10s %10s %9s %s' % ('配置', 'val loss', '期望', '差', '判定'))
print('-' * 78)
for label, table, with_emb, expect in ARMS:
    m = base_model()
    nlin = nemb = 0
    if table is not None:
        nlin, _ = convert(m, table)
        if with_emb:
            nemb = quantize_embeddings(m, table)
    fp = footprint(m)
    vl = evaluate(m)
    ok = abs(vl - expect) < 0.01
    print('%-26s %10.4f %10.4f %+9.4f %s'
          % (label, vl, expect, vl - expect, '✓ 一致' if ok else '✗ 不一致'))
    if table is not None:
        print('%-26s 替换 %d 个 Linear + %d 个 Embedding，量化后 %.1f MB'
              % ('', nlin, nemb, fp['total'] / 2**20))
    del m
print()
