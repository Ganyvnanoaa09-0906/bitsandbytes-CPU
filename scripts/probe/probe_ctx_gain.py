# -*- coding: utf-8 -*-
"""probe_ctx_gain.py — 不同的"局部上下文集合"能解释多少？（决定邻居融合的窗口宽度）

## 为什么需要它

严格结论（已记入报告）：光栅序位置 p=r*G+c 的前缀【包含】上邻 (r-1,c) 和左邻 (r,c-1)，
所以
    H(x_t | 前缀) <= H(x_t | 左+上) = 3.384 bits = 2.345 nats
⇒ 一个完美的因果光栅模型本来就能到 2.345 nats，而我们实测 5.26 nats。
⇒ 【不是信息不够，是模型提取不出来】。

于是问题变成：把多宽的局部窗口【直接】喂给模型才够？窗口越宽成本越高，
所以要量清楚每加一个邻居能赚多少。

## 方法：留出集交叉熵（不是稀疏计数熵）

直接估 H(x|4 个邻居) 需要 1024^4 的条件表，样本根本不够（7.4M 样本 / 1e12 格）。
用【留出集上计数预测器的交叉熵】代替：
  - 80% 图片建计数表，20% 图片算 CE（拉普拉斯平滑）
  - 表稀疏会让 CE 偏高，但【所有上下文集合受到同样的偏差】
    ⇒ 它们之间的比较是公平的
"""
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
import sys, math, itertools
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')

TOK = r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt'
G = 16
SMOOTH = 0.5
MAX_KEYS = 40_000_000      # 安全阀：表太大就跳过该配置


def held_out_ce(ctx_train, y_train, ctx_test, y_test, V, smooth=SMOOTH):
    """计数预测器的留出集交叉熵（nats）。ctx_* 是 (N,) 的整数键。"""
    tab = defaultdict(Counter)
    for k, v in zip(ctx_train, y_train):
        tab[k][v] += 1
    tot, n = 0.0, 0
    for k, v in zip(ctx_test, y_test):
        c = tab.get(k)
        if c is None:
            # 没见过的上下文 ⇒ 退回均匀分布
            tot += math.log(V); n += 1; continue
        s = sum(c.values())
        p = (c.get(v, 0) + smooth) / (s + smooth * V)
        tot += -math.log(p); n += 1
    return tot / max(n, 1), len(tab)


print('=' * 96)
print('局部上下文增益：留出集交叉熵（nats/token，越低越好）')
print('=' * 96)
t = torch.load(TOK, map_location='cpu').long()
N = t.shape[0]
g = t.view(N, G, G).numpy()
print('  tokens %s  grid=%d' % (tuple(t.shape), G))

# 切分：80% 建表 / 20% 评估（按图片切，避免同图泄漏）
rng = np.random.RandomState(0)
perm = rng.permutation(N)
ntr = int(N * 0.8)
tr_img, te_img = perm[:ntr], perm[ntr:]
V = int(t.max()) + 1

# 只取内部像素（保证四个邻居都存在）
r0, r1, c0, c1 = 2, G - 2, 2, G - 2


def gather(imgs):
    sub = g[imgs][:, r0:r1, c0:c1]           # (n, H, W)
    d = {}
    for name, (dr, dc) in [('L', (0, -1)), ('U', (-1, 0)), ('UL', (-1, -1)),
                           ('UR', (-1, 1)), ('LL', (0, -2)), ('UU', (-2, 0))]:
        # 从完整图里取偏移后的窗口，与 sub 对齐
        d[name] = g[imgs][:, r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return sub, d


y_tr, n_tr = gather(tr_img)
y_te, n_te = gather(te_img)
ytr = y_tr.reshape(-1)
yte = y_te.reshape(-1)
print('  训练 token %d  评估 token %d  V=%d' % (ytr.size, yte.size, V))

CONFIGS = [
    ('(无上下文·一元)', []),
    ('左', ['L']),
    ('上', ['U']),
    ('左+上  ← 已知 3.384 bits', ['L', 'U']),
    ('左+上+左上', ['L', 'U', 'UL']),
    ('左+上+右上', ['L', 'U', 'UR']),
    ('左+上+左上+右上  (3宽扇形)', ['L', 'U', 'UL', 'UR']),
    ('加上 左左+上上  (5点)', ['L', 'U', 'UL', 'UR', 'LL', 'UU']),
]

print()
print('  %-34s %10s %12s %10s' % ('上下文集合', '留出CE', '相对一元', '条件表键数'))
print('  ' + '-' * 76)
base = None
rows = []
for name, keys in CONFIGS:
    if not keys:
        ktr = np.zeros(ytr.size, dtype=np.int64)
        kte = np.zeros(yte.size, dtype=np.int64)
    else:
        def keyify(d, imgs_index):
            k = np.zeros(ytr.size if imgs_index == 0 else yte.size, dtype=np.int64)
            for kk in keys:
                arr = (n_tr if imgs_index == 0 else n_te)[kk].reshape(-1)
                k = k * V + arr
            return k
        ktr = keyify(n_tr, 0)
        kte = keyify(n_te, 1)
    nkeys = len(np.unique(ktr))
    if nkeys > MAX_KEYS:
        print('  %-34s %10s %12s %10d  ← 键太多，跳过' % (name, '-', '-', nkeys))
        continue
    ce, nt = held_out_ce(ktr.tolist(), ytr.tolist(), kte.tolist(), yte.tolist(), V)
    if base is None:
        base = ce
    rows.append((name, ce, nkeys))
    print('  %-34s %10.4f %11.1f%% %10d' % (name, ce, 100 * (1 - ce / base), nt))

print()
print('=' * 96)
print('判读：')
print('  * 留出 CE 会系统性高于真实条件熵（计数表稀疏），但各配置的偏差同向 ⇒ 比较公平')
print('  * 看"3宽扇形(左+上+左上+右上)"相对"左+上"赚了多少：')
print('      赚得多 ⇒ 窗口要加宽（nbr_fuse 应该融 4 个 tap）')
print('      赚得少 ⇒ 2 个 tap 就够，模型提取不出的是【同一个交互】而非信息缺失')
print('DONE')
