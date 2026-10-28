"""Is torch's CPU flash attention already good, or is there room to beat it?

The operator profile shows `aten::_scaled_dot_product_flash_attention_` at 13.43% of a step
(1514.5 ms), plus a backward pass at 4.87%. That name means PyTorch already ships a fused CPU
implementation, so the plan to "write a tiled attention like FlashAttention" may be solving a
problem that is already solved. Before spending a day on it, measure the headroom.

Three implementations of the same computation, same shapes as the real model (T=1024, 8 heads,
4 KV heads, d=64, batch 2):

  * F.scaled_dot_product_attention -- what the training actually uses
  * a hand-written path: q@k^T, softmax, @v, with GQA expansion
  * the hand-written path with the score matrix kept in one allocation

If SDPA is much faster, the direction is closed and the 18.3% is simply what attention costs.
If the hand-written version is faster, there is something to chase. Equal means no room either way.

Two things to be careful about, both learned the hard way in this project:

  * measure the forward alone and the forward+backward separately, since they are 13% and 5% of
    the step;
  * warm up properly and take medians. Our step is seconds long while individual attention calls
    are milliseconds, so a single run is noise.

Decision rule, fixed in advance: the hand-written version must beat SDPA by more than 15% to be
worth pursuing, since anything less would not survive integration into the trainer.
"""
from __future__ import annotations
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from scripts.lib.measure_guard import assert_idle, stamp

assert_idle()
stamp('CPU attention: SDPA 对 手写')
torch.set_num_threads(6)

B, T, H, KVH, D = 2, 1024, 8, 4, 64
G = H // KVH

torch.manual_seed(0)
q = torch.randn(B, H, T, D, requires_grad=True)
k = torch.randn(B, KVH, T, D, requires_grad=True)
v = torch.randn(B, KVH, T, D, requires_grad=True)
print()
print('形状: batch %d · T %d · %d 头 · %d KV 头 · d %d （与真实配置一致）' % (B, T, H, KVH, D))
print()


def sdpa(q, k, v):
    # enable_gqa lets SDPA handle 8 query heads against 4 KV heads without materialising the
    # expansion. Without it the call fails on the head-count mismatch, which is how the first
    # run of this comparison produced no SDPA number at all.
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                            enable_gqa=True)


def manual(q, k, v, expand=True):
    kk = k.repeat_interleave(G, dim=1) if expand else k
    vv = v.repeat_interleave(G, dim=1) if expand else v
    s = q @ kk.transpose(-2, -1) / (D ** 0.5)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    s = s.masked_fill(mask, float('-inf'))
    p = torch.softmax(s, dim=-1)
    return p @ vv


def manual_gqa(q, k, v):
    """Same maths without materialising the expanded K/V: loop the KV heads."""
    outs = []
    for h in range(KVH):
        qh = q[:, h * G:(h + 1) * G]
        s = qh @ k[:, h].unsqueeze(1).transpose(-2, -1) / (D ** 0.5)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        s = s.masked_fill(mask, float('-inf'))
        outs.append(torch.softmax(s, dim=-1) @ v[:, h].unsqueeze(1))
    return torch.cat(outs, dim=1)


def timeit(fn, fwd_only, reps=5):
    for _ in range(2):
        out = fn(q, k, v)
        if not fwd_only:
            out.sum().backward()
    ts = []
    for _ in range(reps):
        for p in (q, k, v):
            p.grad = None
        t0 = time.perf_counter()
        out = fn(q, k, v)
        if not fwd_only:
            out.sum().backward()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


print('%-26s %12s %16s' % ('实现', '前向 ms', '前向+反向 ms'))
print('-' * 58)
res = {}
for name, fn in (('SDPA（训练在用的）', sdpa),
                 ('手写 matmul+softmax', manual),
                 ('手写（GQA 不展开）', manual_gqa)):
    try:
        f = timeit(fn, True)
        fb = timeit(fn, False)
        res[name] = (f, fb)
        print('%-26s %12.2f %16.2f' % (name, f, fb))
    except Exception as e:
        print('%-26s 失败: %s' % (name, str(e)[:40]))

print()
if 'SDPA（训练在用的）' in res:
    base_f, base_fb = res['SDPA（训练在用的）']
    for name, (f, fb) in res.items():
        if name == 'SDPA（训练在用的）':
            continue
        print('  %s 相对 SDPA: 前向 %.3fx · 前向+反向 %.3fx'
              % (name, base_f / f, base_fb / fb))
print()
print('判据（事先固定）: 手写版比 SDPA 快 > 15% 才值得做；否则这条关闭。')
print('参考资料: 一步里 attention 占前向 13.43% + 反向 4.87%')
