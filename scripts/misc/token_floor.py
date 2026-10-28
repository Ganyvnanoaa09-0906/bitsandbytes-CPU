# -*- coding: utf-8 -*-
"""token_floor.py — 每个 token 化方案的【理论下界】，判断瓶颈在分词器还是在 AR

## 为什么需要它

实测（云端）：
    ar_swiglu  T=64  (8x8)   loss = 1.0024   ← 模型很自信
    ar_flat256 T=256 (16x16) val  = 5.2603   ← 几乎不informative（均匀分布是 6.931）
同一个数据、同一个 30M 架构、同样的步数，差 5 倍。这不正常，必须先区分：
    ① 是 token 流本身近乎随机（分词器的锅）？→ 看【一元熵】和【二元条件熵】
    ② 还是 AR 没学好（优化的锅）？→ 看 AR 的 loss 相对下界的位置

## 指标

    H1   = 一元熵（bits/token）        = 任何模型的下界（不看上下文）
    H2   = 二元条件熵 H(x_t|x_{t-1})   = 一阶马尔可夫下界
    H2d  = H(x_t | 左邻, 上邻)          = 一个"简单因果模型"能到的下界
    AR 相对 H1 的解释比例 = 1 - AR_loss/H1   ⇒ 这才是真实的学习进度
       （直接比 loss 数值是错的：不同 token 化的熵本来就不同）
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _here
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
for _p in (_root, _here):
    if _p and _p not in _sys.path:
        _sys.path.insert(0, _p)
import sys, math
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')


def entropy_bits(counts):
    tot = sum(counts.values())
    if tot <= 0:
        return 0.0
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0)


def cond_entropy_bits(pairs, min_group=1):
    """H(X|Y)，单位 bits"""
    g = defaultdict(Counter)
    for y, x in pairs:
        g[y][x] += 1
    num, den = 0.0, 0
    for _y, c in g.items():
        n = sum(c.values())
        if n < min_group:
            continue
        num += n * entropy_bits(c)
        den += n
    return (num / den) if den > 0 else float('nan')


def report(path, name, grid=None):
    print()
    print('=' * 96)
    print('%s   %s' % (name, path))
    print('=' * 96)
    try:
        t = torch.load(path, map_location='cpu')
    except Exception as e:
        print('  载入失败: %s' % str(e)[:70]); return
    if not torch.is_tensor(t):
        print('  不是 tensor（可能是 dict）:', type(t)); return
    if t.ndim != 2:
        t = t.reshape(t.shape[0], -1)
    t = t.long()
    N, T = t.shape
    print('  形状 (N=%d, T=%d)  dtype=%s' % (N, T, t.dtype))
    if grid is None:
        g = int(round(T ** 0.5))
        grid = g if g * g == T else None

    flat = t.reshape(-1).numpy()
    cnt = Counter(flat.tolist())
    H1 = entropy_bits(cnt)
    used = len(cnt)
    V = max(int(flat.max()) + 1, used)
    print('  唯一码 %d/%d (%.1f%%)   一元熵 H1 = %.3f bits (上限 %.3f = log2(%d)) ⇒ %.1f%%'
          % (used, V, 100 * used / V, H1, math.log2(V), V, 100 * H1 / math.log2(V)))
    print('  ⇒ 一元下界 CE = %.4f nats/token' % (H1 * math.log(2)))

    # 二元（只看水平相邻，同图内）
    a = t[:, :-1].reshape(-1).tolist()
    b = t[:, 1:].reshape(-1).tolist()
    H2 = cond_entropy_bits(list(zip(a, b)))
    print('  二元条件熵 H(x_t|x_{t-1}) = %.3f bits  ⇒ 下界 CE %.4f nats'
          % (H2, H2 * math.log(2)))

    if grid:
        g = t.view(N, grid, grid)
        left = g[:, :, :-1].reshape(-1).tolist()
        cur = g[:, :, 1:].reshape(-1).tolist()
        Hl = cond_entropy_bits(list(zip(left, cur)))
        up = g[:, :-1, :].reshape(-1).tolist()
        dn = g[:, 1:, :].reshape(-1).tolist()
        Hu = cond_entropy_bits(list(zip(up, dn)))
        # 左+上联合
        l2 = g[:, 1:, :-1].reshape(-1).tolist()
        u2 = g[:, :-1, 1:].reshape(-1).tolist()
        c2 = g[:, 1:, 1:].reshape(-1).tolist()
        Hlu = cond_entropy_bits(list(zip([l * 100000 + u for l, u in zip(l2, u2)], c2)))
        print('  H(x_t | 左邻)      = %.3f bits  ⇒ CE %.4f nats' % (Hl, Hl * math.log(2)))
        print('  H(x_t | 上邻)      = %.3f bits  ⇒ CE %.4f nats' % (Hu, Hu * math.log(2)))
        print('  H(x_t | 左邻+上邻) = %.3f bits  ⇒ CE %.4f nats  ← 简单因果模型的下界'
              % (Hlu, Hlu * math.log(2)))
    # 相邻同码率
    same_h = float((g[:, :, :-1] == g[:, :, 1:]).float().mean()) if grid else \
        float((t[:, :-1] == t[:, 1:]).float().mean())
    perm = torch.randperm(flat.size)
    rnd = float((torch.from_numpy(flat) == torch.from_numpy(flat)[perm]).float().mean())
    print('  相邻同码率 %.3f%%   随机基线 %.4f%%  ⇒ %.1fx'
          % (same_h * 100, rnd * 100, same_h / max(rnd, 1e-9)))
    return dict(N=N, T=T, H1=H1, H2=H2, used=used, V=V)


print('=' * 96)
print('token 化方案的【理论下界】对比')
print('=' * 96)
res = {}
for p, n, g in [
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_8x8.pt', '8x8  (T=64)', 8),
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt', '16x16 (T=256)', 16),
    (r'D:\work\bitsandbytes-CPU\.autopilot\ms_tokens_full.pt', '多尺度 (T=341, 14433图)', None),
]:
    try:
        res[n] = report(p, n, g)
    except Exception as e:
        print('  %s 处理失败: %s' % (n, str(e)[:80]))


# ---------------- 多尺度专属：逐尺度熵 + 跨尺度条件熵 ----------------
# 这是判断"多尺度到底帮不帮忙"的关键：
#   若最细尺度(256 token)在给定粗尺度后熵大幅下降 ⇒ 多尺度真的降低了 AR 的负担
#   若各尺度熵差不多 ⇒ 只是把 256 个难 token 拆成 85 个难 token + 256 个一样难的 token
PATCH = (1, 2, 4, 8, 16)
p_ms = r'D:\work\bitsandbytes-CPU\.autopilot\ms_tokens_full.pt'
try:
    t = torch.load(p_ms, map_location='cpu').long()
    print()
    print('=' * 96)
    print('多尺度专属：逐尺度熵 + 粗尺度能解释多少（patch_nums=%s）' % (PATCH,))
    print('=' * 96)
    N = t.shape[0]
    begins, cur = [], 0
    for pn in PATCH:
        begins.append((cur, cur + pn * pn)); cur += pn * pn
    print('  %-6s %6s %8s %10s %12s %14s' %
          ('尺度', 'token数', '唯一码', '边缘熵H1', 'H(x|上一尺度)', '粗尺度解释率'))
    scale_views = {}
    for i, pn in enumerate(PATCH):
        a, b = begins[i]
        v = t[:, a:b].view(N, pn, pn)
        scale_views[i] = v
        flat = v.reshape(-1)
        H1 = entropy_bits(Counter(flat.tolist()))
        xu = float(v.to(torch.float64).mean())
        Hprev = float('nan'); gain = float('nan')
        if i > 0:
            # 粗尺度对应位置：scale i 的 (r,c) ↔ scale i-1 的 (r//2, c//2)
            # ⚠️ 这是近似对应（真实残差链是 f_k − down(f_hat)，不是简单的 2×2 分块），
            #    但足以给出"粗尺度能解释多少"的量级
            pv = scale_views[i - 1]
            up = pv.repeat_interleave(pn // PATCH[i - 1], dim=1).repeat_interleave(
                pn // PATCH[i - 1], dim=2)
            pairs = list(zip(up.reshape(-1).tolist(), flat.tolist()))
            Hprev = cond_entropy_bits(pairs)
            gain = 1 - Hprev / H1 if H1 > 0 else float('nan')
        print('  %-6d %6d %8d %10.3f %12s %14s'
              % (pn, pn * pn, int(flat.unique().numel()), H1,
                 ('%.3f' % Hprev) if Hprev == Hprev else '-',
                 ('%.1f%%' % (100 * gain)) if gain == gain else '-'))
    # 各尺度的"总信息量"（bits/图）
    tot = 0.0
    for i, pn in enumerate(PATCH):
        a, b = begins[i]
        flat = scale_views[i].reshape(-1)
        H1 = entropy_bits(Counter(flat.tolist()))
        tot += H1 * pn * pn
    print('  ⇒ 各尺度边缘熵之和（上界，忽略尺度间相关） = %.1f bits/图' % tot)
    t16 = torch.load(r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt',
                     map_location='cpu').long()
    flat16 = t16.reshape(-1)
    H16 = entropy_bits(Counter(flat16.tolist()))
    g16 = int(round(t16.shape[1] ** 0.5))
    v16 = t16.view(t16.shape[0], g16, g16)
    lu = cond_entropy_bits(list(zip(
        v16[:, 1:, :-1].reshape(-1).tolist(), v16[:, 1:, 1:].reshape(-1).tolist())))
    print('  ⇒ 单尺度 16x16 的边缘熵之和 = %.1f bits/图（256 token × %.3f）'
          % (H16 * 256, H16))
except Exception as e:
    print('多尺度专属分析失败: %s' % str(e)[:120])

print()
print('=' * 96)
print('判读：')
print('  ① 若 16x16 的 H1 ≈ log2(1024)=10 bits（即接近均匀），说明分词器把码本当哈希用，')
print('     token 流近乎随机 ⇒ 【瓶颈在分词器】，加大 AR 没用')
print('  ② 若 16x16 的 H1 明显低于 10 bits 但 AR 的 val(5.26 nats=7.6 bits) 还远高于 H1，')
print('     说明【瓶颈在 AR】，有大量可压缩的信息没被学到')
print('  ③ 比较 8x8 与 16x16 的 AR 解释比例 = 1 - AR_loss/H1*nats，这才是可比的进度指标')
print('  ④ 多尺度：若"粗尺度解释率"在第 4 尺度上很高 ⇒ 多尺度真的降低了 AR 负担；')
print('     若各尺度 H1 都接近 10 bits 且解释率低 ⇒ 只是把难 token 拆成更多难 token')
print('DONE')

