# -*- coding: utf-8 -*-
"""floor_bracket.py — 用【夹逼】修正 §10.13⑦ 的下界估计（子代理指出的关键漏洞）

## 问题

我原来的 H(x|左+上) 用的是【插值（plug-in）估计量】：
    按 (左,上) 分组，组内算经验熵再加权平均。
vocab=1024 ⇒ (左,上) 有 1024² ≈ 105 万个格子，而 16x16 只有 740 万个样本
⇒ 绝大多数格子计数是 0 或 1。**计数为 1 的格子经验熵 = 0**，所以这个估计量
【系统性偏低】，2.345 nats 是乐观下界，不是真值。

同样的偏差也存在于 8x8（1024² 格子 / 91 万个样本）。

## 夹逼

    插值估计（同批数据，偏低）  ≤  真实 H  ≤  留出 CE（训练/评估分图片，偏高）
真值落在两者之间。而且：
  * 若 AR 实测 loss 仍【高于留出的悲观下界】⇒ 差距是真的，与估计偏差无关
  * 若 AR 实测 loss 落在区间内 ⇒ 原来的"2.24 倍差距"可能主要是估计偏差

## 另外做一次【降 vocab】估计

把码按【出现频率排名】合并成 K 个桶（最频繁的码各自成桶），
格子数从 1024² 降到 K²，表密得多 ⇒ 偏差小得多。
虽然分辨率变粗，但如果 K=64/256 下"上邻贡献巨大"的结论仍成立，就说明不是偏差造的。

## 附带：加性平滑

留出 CE 用 Laplace 平滑（α=0.1）避免 log(0)；插值估计另给一个 α 平滑版本。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import sys, math, json
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')


def plugin_cond_entropy_bits(y, x, V, alpha=0.0):
    """插值（或平滑）条件熵 H(x|y)，单位 bits。alpha=0 即原插值估计量。"""
    tab = defaultdict(Counter)
    for a, b in zip(y, x):
        tab[a][b] += 1
    num, den = 0.0, 0
    for _k, c in tab.items():
        s = sum(c.values())
        if alpha <= 0:
            e = -sum((v / s) * math.log2(v / s) for v in c.values())
        else:
            Z = s + alpha * V
            e = -sum(((v + alpha) / Z) * math.log2((v + alpha) / Z) for v in c.values())
            e -= (V - len(c)) * (alpha / Z) * math.log2(alpha / Z)
        num += s * e
        den += s
    return num / max(den, 1)


def heldout_ce_nats(ktr, ytr, kte, yte, V, alpha=0.1):
    tab = defaultdict(Counter)
    for a, b in zip(ktr, ytr):
        tab[a][b] += 1
    tot, n = 0.0, 0
    for a, b in zip(kte, yte):
        c = tab.get(a)
        if c is None:
            tot += math.log(V); n += 1; continue
        s = sum(c.values())
        tot += -math.log((c.get(b, 0) + alpha) / (s + alpha * V)); n += 1
    return tot / max(n, 1)


def freq_bucket_map(all_codes, V, K):
    """按频率排名映射到 K 个桶：最频繁的码各自成桶（0..K-2），其余归入 K-1"""
    cnt = Counter(all_codes.tolist())
    order = [c for c, _ in cnt.most_common()]
    m = np.full(V, K - 1, dtype=np.int64)
    for i, c in enumerate(order[:K - 1]):
        m[c] = i
    return m


def analyse(tok_path, name, grid, V=1024):
    t = torch.load(tok_path, map_location='cpu').long().numpy()
    N = t.shape[0]
    g = t.reshape(N, grid, grid)
    print()
    print('=' * 92)
    print('%s   N=%d  grid=%d  V=%d' % (name, N, grid, V))
    print('=' * 92)
    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    ntr = int(N * 0.8)
    tr, te = perm[:ntr], perm[ntr:]

    def pairs(imgs):
        sub = g[imgs]
        # 内部像素：上、左都存在
        cur = sub[:, 1:, 1:].reshape(-1)
        L = sub[:, 1:, :-1].reshape(-1)
        U = sub[:, :-1, 1:].reshape(-1)
        return L, U, cur

    Ltr, Utr, ytr = pairs(tr)
    Lte, Ute, yte = pairs(te)
    print('  训练对 %d   评估对 %d   (left,up) 可能格子数 %d'
          % (ytr.size, yte.size, V * V))

    print()
    print('  %-30s %12s %12s' % ('估计方式（H(x|左+上)）', 'bits', 'nats'))
    print('  ' + '-' * 58)
    # 插值（偏低）
    key_tr = (Ltr.astype(np.int64) * V + Utr).tolist()
    key_te = (Lte.astype(np.int64) * V + Ute).tolist()
    he_plug = plugin_cond_entropy_bits(key_tr, ytr.tolist(), V, alpha=0.0)
    print('  %-30s %12.3f %12.4f' % ('插值（同批数据，偏低）',
                                     he_plug, he_plug * math.log(2)))
    he_plug_a = plugin_cond_entropy_bits(key_tr, ytr.tolist(), V, alpha=0.1)
    print('  %-30s %12.3f %12.4f' % ('插值 + Laplace α=0.1',
                                     he_plug_a, he_plug_a * math.log(2)))
    ce = heldout_ce_nats(key_tr, ytr.tolist(), key_te, yte.tolist(), V, alpha=0.1)
    print('  %-30s %12s %12.4f' % ('留出 CE（偏高）', '-', ce))
    print('  ⇒ 真实的 H(x|左+上) 落在 [%.3f, %.3f] nats' % (he_plug * math.log(2), ce))

    # 降 vocab
    print()
    print('  %-14s %14s %14s %14s' % ('降 vocab 到 K', '插值(bits)', '留出 CE(nats)', '格子数 K²'))
    print('  ' + '-' * 60)
    for K in (1024, 256, 64):
        if K == 1024:
            m = np.arange(V)
        else:
            m = freq_bucket_map(np.concatenate([ytr, yte]), V, K)
        ktr = (m[Ltr].astype(np.int64) * K + m[Utr]).tolist()
        kte = (m[Lte].astype(np.int64) * K + m[Ute]).tolist()
        ytr_k = m[ytr].tolist()
        yte_k = m[yte].tolist()
        hb = plugin_cond_entropy_bits(ktr, ytr_k, K, alpha=0.0)
        hc = heldout_ce_nats(ktr, ytr_k, kte, yte_k, K, alpha=0.1)
        print('  %-14d %14.3f %14.4f %14d  (比例 %.2f)'
              % (K, hb, hc, K * K, (hc * math.log(2)) / max(hb, 1e-9)))
    print('  ⇒ 降 vocab 后格子变密、偏差变小：若"上邻贡献巨大"依然成立，')
    print('    就说明结论不是稀疏偏差造的')
    return he_plug * math.log(2), ce


print('=' * 92)
print('条件熵下界的【夹逼】修正  —— 检验 §10.13⑦"2.24 倍差距"是否被稀疏偏差高估')
print('=' * 92)
r = {}
for p, n, gid in [
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_8x8.pt', '8x8 (T=64)', 8),
    (r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt', '16x16 (T=256)', 16),
]:
    try:
        r[n] = analyse(p, n, gid)
    except Exception as e:
        import traceback; traceback.print_exc()

print()
print('=' * 92)
print('判读：')
print('  8x8 的 AR 实测 loss = 1.0024 nats；16x16 的 AR 实测 val = 5.2603 nats')
for n, (lo, hi) in r.items():
    print('  %-16s 下界区间 [%.3f, %.3f]' % (n, lo, hi))
print('  * 若 AR 的 loss 仍【高于 hi(悲观上界)】⇒ 差距是真的，与估计偏差无关')
print('  * 若 AR 的 loss 落在区间内 ⇒ 原来的"倍数差距"主要可能是估计偏差')
print('DONE')
