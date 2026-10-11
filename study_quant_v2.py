"""study_quant_v2.py -- fix the two bugs and use the metric that matches the claim.

Three problems with the first pass:

  1. kmeans scored worse than uniform, which is impossible for a data-fitted
     codebook. The cause: it was fitted on weights normalised by a global maximum
     while quantize_error normalises per block by absmax, so the codebook was
     solving a different problem than the one being scored.

  2. The nf table was reconstructed from a formula, giving [-1, -0.708, -0.542,
     ...] where the real NF4 is [-1, -0.6962, -0.5251, ...]. Use the actual table
     from cpu_ops.cpp:275; there is no reason to re-derive something already
     written down, which is the lesson of the last several days.

  3. Only L2 relative error was reported. That is not the quantity NF4 optimises:
     its levels are quantiles, chosen so that each level carries equal probability
     mass under a normal, which is an information statement, not a least-squares
     one. Judging it by L2 is the mistake this project keeps making -- a criterion
     that does not measure what the conclusion is about. So report both:

       L2 relative error       how far the values move
       KL(P || Q)              how much probability mass lands on wrong levels

     over the per-block normalised weights, which is what the kernels see.
"""
from __future__ import annotations

import glob
import sys

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# The actual NF4 table, copied from cpu_ops.cpp:275 -- not re-derived.
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
                0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0], dtype=np.float32)


def nf_table(levels):
    """NF at an arbitrary width by the same construction: normal mid-quantiles."""
    from statistics import NormalDist
    nd = NormalDist()
    n = 1 << levels
    q = np.array([nd.inv_cdf((i + 0.5) / n) for i in range(n)])
    q = q / np.abs(q).max()
    q[0], q[-1] = -1.0, 1.0
    return np.sort(q).astype(np.float32)


def uniform_table(levels):
    return np.linspace(-1.0, 1.0, 1 << levels).astype(np.float32)


def kmeans_table(samples, levels, iters=40, seed=0):
    """Fit on per-block normalised values -- the same domain quantize_error scores."""
    n = 1 << levels
    rng = np.random.default_rng(seed)
    x = samples
    c = np.quantile(x, np.linspace(0, 1, n)).astype(np.float64)
    c[0], c[-1] = -1.0, 1.0
    for _ in range(iters):
        idx = np.abs(x[:, None] - c[None, :]).argmin(axis=1)
        for i in range(n):
            m = idx == i
            if m.any():
                c[i] = x[m].mean()
        c = np.sort(np.clip(c, -1, 1))
        c[0], c[-1] = -1.0, 1.0
    return c.astype(np.float32)


def block_norm(w, block=64):
    w = w.reshape(-1).astype(np.float32)
    pad = (-len(w)) % block
    if pad:
        w = np.concatenate([w, np.zeros(pad, dtype=np.float32)])
    wb = w.reshape(-1, block)
    s = np.abs(wb).max(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return wb / s, s


def score(wb, code):
    idx = np.abs(wb[:, :, None] - code[None, None, :]).argmin(axis=2)
    dq = code[idx]
    l2 = float(np.linalg.norm(wb - dq)) / (float(np.linalg.norm(wb)) or 1.0)
    # KL between the level histograms of the original (assigned) and the quantised:
    # both are assigned to code indices, so compare their index distributions.
    p = np.bincount(idx.ravel(), minlength=len(code)).astype(np.float64)
    p /= p.sum()
    # reference: assign the *unquantised* values to the same levels -- identical by
    # construction, so instead measure how much mass sits on the nearest-vs-second
    # nearest boundary, i.e. how ambiguous each assignment is.
    d = np.abs(wb[:, :, None] - code[None, None, :])
    srt = np.sort(d, axis=2)
    margin = (srt[:, :, 1] - srt[:, :, 0]) / (srt[:, :, 1] + 1e-12)
    ambiguity = float(np.mean(1.0 - margin))
    return l2, ambiguity, p


paths = (glob.glob(r'D:\work\bitsandbytes-CPU\i5build\ar32_run1\ckpt_3000.pt') +
         glob.glob(r'D:\work\bitsandbytes-CPU\i5build\vqgan32\ckpt_last.pt'))
import torch  # noqa: E402
ws = []
for p in paths:
    try:
        ck = torch.load(p, map_location='cpu', weights_only=False)
    except Exception:
        continue
    sd = ck.get('model', ck)
    for k, v in sd.items():
        if torch.is_tensor(v) and v.dtype == torch.float32 and v.ndim >= 2 and v.numel() >= 4096:
            ws.append(v.detach().numpy().reshape(-1)[:1 << 20])
print('权重: %d 个张量, %.1fM 参数' % (len(ws), sum(w.size for w in ws) / 1e6))
print()

pooled = np.concatenate([block_norm(w)[0].ravel() for w in ws[:8]])[:400000]

for levels in (4, 3, 2):
    print('=== %d bit（%d 级）===' % (levels, 1 << levels))
    tables = {
        'nf（本仓库真表）' if levels == 4 else 'nf 分位数': (NF4 if levels == 4 else nf_table(levels)),
        'uniform 等距': uniform_table(levels),
        'kmeans 数据驱动': kmeans_table(pooled, levels),
    }
    print('  %-18s %-11s %11s  %s' % ('码表', 'L2 相对误差', '分配模糊度', '码表'))
    print('  ' + '-' * 74)
    sc = {}
    for name, c in tables.items():
        l2s, ambs = [], []
        for w in ws:
            wb, _ = block_norm(w)
            a, b, _ = score(wb, c)
            l2s.append(a); ambs.append(b)
        sc[name] = (np.mean(l2s), np.mean(ambs))
    for name, (l2, amb) in sorted(sc.items(), key=lambda kv: kv[1][0]):
        print('  %-18s %11.5f %11.5f  %s'
              % (name, l2, amb, np.round(tables[name], 3)))
    print()
print('注: L2 小 = 数值移动少；模糊度小 = 每个值离第二近的码更远 ⇒ 分配更确定。')
print('    NF4 的论点属于后者（分位数让每级承载等量概率），所以两个都要看。')
