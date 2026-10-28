# -*- coding: utf-8 -*-
"""probe_ceiling.py — 用【可学习探针】测各上下文集合的真实可达 loss（替代失效的计数估计）

## 为什么要换掉计数估计

`floor_bracket.py` 证明：vocab=1024 时 (左,上) 有 105 万个格子，而 16x16 只有
740 万个样本 ⇒ 插值估计量给 3.384 bits（严重偏低），留出 CE 给 9.38 bits（偏高），
区间宽到没用。**我原来那句"加上上邻熵掉 5.29 bits"主要是稀疏偏差造的。**

唯一 bias-free 的参照是稠密表：H(x|左) = 8.674 bits、H(x|上) = 8.486 bits
（1024 个格子，每格 ~7200 样本）。而我们的 AR 达到 7.6 bits
⇒ 【255 个 token 的上下文只比"只看左邻"多榨出 1.07 bits】。

## 本脚本

训一组小探针，每个只看一个【固定的上下文集合】，测留出 CE：

    {左}  {上}  {左,上}  {左,上,左上,右上}  {再+上上行}  ...

探针是可学习的（MLP），所以不会像计数表那样被稀疏性毁掉；只要容量够、训够，
留出 CE 就逼近真实的 H(x | 该上下文集合)。

## 判读

  * 若 {左,上} 的留出 CE 已经【接近 AR 的 5.26 nats】⇒ 局部上下文就是全部可达信息，
    "2D 邻居够不着"不是瓶颈，我们的 AR 已接近该表示的天花板
  * 若 {左,上} 明显【低于】5.26 nats（比如 3 nats）⇒ 局部上下文里还有大量信息
    没被 AR 榨出来 ⇒ 归纳偏置确实是瓶颈，邻居融合/局部注意力值得做
  * 看 CE 随上下文集合增大的【饱和点】—— 那就是"够用"的窗口宽度
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
import sys, os, math, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)

TOK = r'D:\work\bitsandbytes-CPU\.autopilot\tokens_16x16_full.pt'
G = 16
V = 1024
EMB = 64
HID = 256
STEPS = 6000
BATCH = 4096
LR = 2e-3


def build_contexts(g):
    """返回 {名字: (dr,dc) 列表}，全部只含【因果前缀内】的邻居。"""
    return {
        '左': [(0, -1)],
        '上': [(-1, 0)],
        '左+上': [(0, -1), (-1, 0)],
        '左+上+左上+右上': [(0, -1), (-1, 0), (-1, -1), (-1, 1)],
        '上一整行+左': [(0, -1), (-1, -1), (-1, 0), (-1, 1)],
        '2行+左 (7点)': [(0, -1), (-1, -1), (-1, 0), (-1, 1),
                        (-2, -1), (-2, 0), (-2, 1)],
        '2行+左+左左 (8点)': [(0, -1), (0, -2), (-1, -1), (-1, 0), (-1, 1),
                            (-2, -1), (-2, 0), (-2, 1)],
    }


class Probe(nn.Module):
    def __init__(self, n_tap, emb=EMB, hid=HID, V=V):
        super().__init__()
        self.emb = nn.Embedding(V, emb)
        self.net = nn.Sequential(nn.Linear(n_tap * emb, hid), nn.GELU(),
                                 nn.Linear(hid, hid), nn.GELU(),
                                 nn.Linear(hid, V))
    def forward(self, ctx):          # ctx: (B, n_tap)
        e = self.emb(ctx).flatten(1)
        return self.net(e)


def main():
    t = torch.load(TOK, map_location='cpu').long()
    N = t.shape[0]
    g = t.view(N, G, G)
    print('=' * 92)
    print('可学习探针：各上下文集合的真实可达 loss   tokens=%s' % (tuple(t.shape),))
    print('=' * 92)

    rng = np.random.RandomState(0)
    perm = rng.permutation(N)
    ntr = int(N * 0.8)
    tr_img, te_img = perm[:ntr], perm[ntr:]
    # 取内部像素，保证所有偏移存在
    r0, r1, c0, c1 = 2, G - 2, 2, G - 2

    def make(imgs, offs):
        cur = g[imgs][:, r0:r1, c0:c1].reshape(-1)
        ctx = torch.stack([g[imgs][:, r0 + dr:r1 + dr, c0 + dc:c1 + dc].reshape(-1)
                           for (dr, dc) in offs], dim=1)
        return ctx, cur

    print('参考（稠密计数表，可信）：H(x|左)=8.674 bits=6.012 nats，H(x|上)=8.486 bits')
    print('AR 实测 val = 5.2603 nats（256 个 token 的完整前缀）')
    print()
    print('  %-22s %6s %14s %14s %10s' %
          ('上下文集合', 'tap', '训练CE(nats)', '留出CE(nats)', '相对左'))
    print('  ' + '-' * 74)

    base_ce = None
    for name, offs in build_contexts(g).items():
        ctr, ytr = make(tr_img, offs)
        cte, yte = make(te_img, offs)
        torch.manual_seed(0)
        m = Probe(len(offs))
        opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.0)
        n = ytr.shape[0]
        t0 = time.time()
        last_tr = None
        for s in range(STEPS):
            idx = torch.randint(0, n, (BATCH,))
            lo = m(ctr[idx])
            loss = F.cross_entropy(lo, ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            last_tr = float(loss.detach())
            if (s + 1) % 2000 == 0:
                print('    [%s] step %d  trainCE=%.4f  %.0fs'
                      % (name, s + 1, last_tr, time.time() - t0), flush=True)
        # 留出
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for i in range(0, yte.shape[0], BATCH):
                lo = m(cte[i:i + BATCH])
                tot += float(F.cross_entropy(lo, yte[i:i + BATCH], reduction='sum'))
                cnt += lo.shape[0]
        ce = tot / cnt
        if base_ce is None:
            base_ce = ce
        print('  %-22s %6d %14.4f %14.4f %10.4f'
              % (name, len(offs), last_tr, ce, ce - base_ce), flush=True)

    print()
    print('=' * 92)
    print('判读：')
    print('  * {左+上} 的留出 CE 若已接近 5.26 ⇒ 局部上下文就是全部可达信息，')
    print('    我们的 AR 已接近该表示的天花板，"2D 邻居够不着"不是瓶颈')
    print('  * 若 {左+上} 明显低于 5.26 ⇒ 局部上下文里还有大量信息没被榨出，')
    print('    归纳偏置确实是瓶颈')
    print('  * CE 随上下文集合增大的【饱和点】= 够用的窗口宽度')
    print('DONE')


if __name__ == '__main__':
    main()
