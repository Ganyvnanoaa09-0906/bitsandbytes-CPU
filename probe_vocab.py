# -*- coding: utf-8 -*-
"""probe_vocab.py — 码本大小 K 对"每图可榨出结构"的影响（5 分钟，可能揭示一个 flag 的杠杆）

## 动机

已知（都是 bias-free 的稠密表结果）：
    8x8  (64 tok) : 一元熵 9.545 bits, 结构 8.1 bits/token ⇒ 518 bits/图
    16x16(256 tok): 一元熵 9.776 bits, 结构 2.2 bits/token ⇒ 560 bits/图
⇒ **粗表示每个 token 携带的结构量是细表示的 3.7 倍**。

那么把【码本变小】（1024 → 256/128/64）呢？更少的码 ⇒ token 流更集中 ⇒
相邻更容易同码 ⇒ 条件熵更低。若"每图结构"随之上升，那就是**一个 flag 的改动**
（训练 VQ-VAE 时 --n_codes 256），比改架构简单得多。

## 方法

把码按【出现频率排名】合并成 K 个桶（最频繁的码各自成桶），对每个 K 测：
    H1(K)           = 该表示的一元熵（bits/token）
    H(x|左邻)        = 稠密表，可信
    结构/token       = H1 - H(x|左邻)
    结构/图          = token数 × 结构/token

## ⚠️ caveat

降 K 会同时降低分词器的信息容量（重建变差），所以"结构/图"上升不自动等于更好
—— 还需要在**同等或更好的重建质量**下比较。本探针只给方向，不给结论。
"""
import sys, math
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
V = 1024


def entropy_bits(counts):
    tot = sum(counts.values())
    return -sum((c / tot) * math.log2(c / tot) for c in counts.values() if c > 0) if tot else 0.0


def cond_entropy_bits(pairs):
    g = defaultdict(Counter)
    for y, x in pairs:
        g[y][x] += 1
    num, den = 0.0, 0
    for _k, c in g.items():
        n = sum(c.values())
        num += n * entropy_bits(c); den += n
    return num / max(den, 1)


def freq_map(all_codes, K):
    """按频率排名映射到 K 个桶：最频繁的码各自成桶，其余归入最后一个桶"""
    cnt = Counter(all_codes.tolist())
    order = [c for c, _ in cnt.most_common()]
    m = np.full(V, K - 1, dtype=np.int64)
    for i, c in enumerate(order[:K - 1]):
        m[c] = i
    return m


def analyse(tok_path, name, grid):
    t = torch.load(tok_path, map_location='cpu').long()
    N = t.shape[0]
    g = t.view(N, grid, grid)
    flat = g.reshape(-1).numpy()
    print()
    print('=' * 90)
    print('%s   N=%d  grid=%d  token/图=%d' % (name, N, grid, grid * grid))
    print('=' * 90)
    print('  %6s %10s %12s %12s %14s %12s' %
          ('K', '一元熵', 'H(x|左邻)', '结构/token', '结构/图(bits)', '唯一桶/图'))
    print('  ' + '-' * 82)
    base = None
    for K in (1024, 512, 256, 128, 64, 32, 16):
        if K == 1024:
            m = np.arange(V)
        else:
            m = freq_map(flat, K)
        b = m[flat]
        H1 = entropy_bits(Counter(b.tolist()))
        gb = b.reshape(N, grid, grid)
        Hl = cond_entropy_bits(list(zip(gb[:, :, :-1].reshape(-1).tolist(),
                                        gb[:, :, 1:].reshape(-1).tolist())))
        struct = H1 - Hl
        per_img = struct * grid * grid
        if base is None:
            base = per_img
        print('  %6d %10.3f %12.3f %12.3f %14.1f %11.1f%%'
              % (K, H1, Hl, struct, per_img, 100 * per_img / base))
    print('  ⇒ "结构/图" = token数 × (一元熵 − H(x|左邻))，即这个表示每图有多少可榨出的结构')
    print('  ⚠️ 降 K 会同时降低分词器容量（重建变差），上升不自动等于更好')


print('=' * 90)
print('码本大小 K → 每图可榨出结构')
print('=' * 90)
for p, n, g in [
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt', '16x16 (C_mse token)', 16),
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_8x8.pt', '8x8', 8),
]:
    try:
        analyse(p, n, g)
    except Exception as e:
        print('  %s 失败: %s' % (n, str(e)[:100]))

print()
print('=' * 90)
print('判读：')
print('  * 若"结构/图"在某个 K<1024 处【显著上升】⇒ 小码本是一个 flag 的杠杆，值得试')
print('  * 若单调下降 ⇒ 大码本更好，方向不在这里')
print('  * 8x8 那一行的绝对量级可作为参照（它每图 518 bits）')
print('DONE')
