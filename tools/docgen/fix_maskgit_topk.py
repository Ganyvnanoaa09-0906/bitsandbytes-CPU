"""Fix the MaskGIT sampler: it samples from the full 1024-way distribution with no top-k.

The symptom is the samples at step 4000: high-frequency fragmented noise, a chaotic patchwork of
small colour fragments. That is not the same failure as the AR run's smooth colour fields and it is
the classic signature of sampling from an under-trained distribution without filtering.

The cause is visible in the comparison. sample_ar.py's AR path passes top_k to model.generate.
Its MaskGIT path calls generate_masked, which has no top_k parameter at all, so sampling runs over
all 1024 codewords. With the validation loss at 6.19 against 6.93 for uniform, the distribution is
close to flat, and sampling a flat distribution per position gives noise.

MaskGIT's own papers sample from a filtered or temperature-adjusted distribution for exactly this
reason. Filtering to the top 100 codewords, renormalising, then sampling and using the renormalised
probability as the confidence is the standard construction and matches what the AR path already
does here.

This needs no retraining: the checkpoint at step 4000 is fine, only the sampler was wrong. So the
fix can be validated in minutes rather than three hours.

Default stays 0 (no filtering) so existing behaviour is unchanged unless asked for.
"""
from __future__ import annotations
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import io
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
s = io.open(P, encoding='utf-8').read()

if '_MG_TOPK' not in s:
    # 1. signature
    old_sig = """    def generate_masked(self, total=None, n_iter=16, mask_id=None, temperature=1.0,
                        batch=1, seed=None, device=None, confidence='max'):"""
    new_sig = """    def generate_masked(self, total=None, n_iter=16, mask_id=None, temperature=1.0,
                        batch=1, seed=None, device=None, confidence='max', top_k=0):
        # _MG_TOPK: top_k was missing entirely, so sampling ran over all 1024 codewords while the
        # AR path filtered to 100. With val 6.19 against 6.93 for uniform the distribution is nearly
        # flat, and sampling a flat distribution per position produces high-frequency noise --
        # exactly what the step-4000 samples show. Filter, renormalise, then sample."""
    assert old_sig in s, 'signature anchor'
    s = s.replace(old_sig, new_sig, 1)

    # 2. filtering before multinomial
    old_p = """            p = torch.softmax(lg, dim=-1)
            _B = x.shape[0]
            pred = torch.multinomial(p.view(-1, V), 1).view(_B, T)"""
    new_p = """            p = torch.softmax(lg, dim=-1)
            if top_k and 0 < int(top_k) < V:
                _v, _ = torch.topk(p, int(top_k), dim=-1)
                _thr = _v[..., -1, None]
                p = torch.where(p < _thr, torch.zeros_like(p), p)
                p = p / p.sum(-1, keepdim=True).clamp_min(1e-12)
            _B = x.shape[0]
            pred = torch.multinomial(p.view(-1, V), 1).view(_B, T)"""
    assert old_p in s, 'sampling anchor'
    s = s.replace(old_p, new_p, 1)

    io.open(P, 'w', encoding='utf-8', newline='').write(s)
    import py_compile
    py_compile.compile(P, doraise=True)
    print('  + generate_masked 已加 top_k 与过滤')
    print('  ✓ 语法通过')
else:
    print('  已加过')

# 3. pass it from sample_ar.py
Q = r'D:\work\bitsandbytes-CPU\sample_ar.py'
q = io.open(Q, encoding='utf-8').read()
old_call = """        toks = model.generate_masked(total=cfg.total_tokens, n_iter=a.maskgit_iter,
                                     mask_id=cfg.vocab_size - 1,
                                     temperature=a.temperature,
                                     batch=a.n, seed=a.seed)"""
new_call = """        toks = model.generate_masked(total=cfg.total_tokens, n_iter=a.maskgit_iter,
                                     mask_id=cfg.vocab_size - 1,
                                     temperature=a.temperature,
                                     batch=a.n, seed=a.seed,
                                     top_k=a.top_k)"""
if old_call in q:
    q = q.replace(old_call, new_call, 1)
    io.open(Q, 'w', encoding='utf-8', newline='').write(q)
    import py_compile
    py_compile.compile(Q, doraise=True)
    print('  + sample_ar.py 已传 top_k（沿用 --top-k，默认 100）')
    print('  ✓ 语法通过')
else:
    print('  ? sample_ar 调用锚点没找到')
