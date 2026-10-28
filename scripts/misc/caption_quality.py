# -*- coding: utf-8 -*-
"""caption_quality.py — 定量确认：我们的自动打标 caption 到底有多少信息量。

为什么要查：caption 决定了条件化 AR 的【条件信号】上界。
  若 caption 在图像间近似常数，那么无论采样器怎么修，
  "条件控制"都没有可学的信号 —— 这会解释历史上"条件注入太弱"的观测
  （我们之前把那条归因于 4 个静默 bug，但即使全修好，条件本身可能也没信息）。

指标：
  1. 每个 tag 的出现频率分布 → top-k 覆盖了多大比例的 tag 总量
  2. 任意两条 caption 的 Jaccard 相似度均值 → 若接近 1，说明几乎一样
  3. 条件熵 H(tag | 图) 的粗略估计：用"平均每条 caption 相对全局高频集的偏离"
  4. 与随机基线对比：随机抽 tag 的话，应该多接近？
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
import io, json, collections, itertools, random, math

P = r'D:\work\bitsandbytes-CPU\.autopilot\captions\captions_merged.jsonl'
rows = []
for l in io.open(P, encoding='utf-8'):
    l = l.strip()
    if l:
        try:
            rows.append(json.loads(l))
        except Exception:
            pass
N = len(rows)
print('caption 数: %d' % N)

sets = [set(t.strip() for t in r['caption'].split(',') if t.strip()) for r in rows]
cnt = collections.Counter()
for s in sets:
    for t in s:
        cnt[t] += 1
tags = list(cnt)
M = len(tags)
print('不同 tag: %d' % M)

# 1) top-k 覆盖
tot = sum(cnt.values())
print()
print('=== tag 出现次数的集中度（总量 %d）===' % tot)
run = 0
for k in (1, 3, 5, 10, 20, 40, M):
    run = sum(c for _, c in cnt.most_common(k))
    print('  top-%-3d 占总量 %5.1f%%' % (k, 100.0 * run / tot))

# 2) 平均 Jaccard
random.seed(0)
pairs = [(random.randrange(N), random.randrange(N)) for _ in range(20000)]
js = []
for i, j in pairs:
    if i == j:
        continue
    a, b = sets[i], sets[j]
    u = len(a | b)
    if u:
        js.append(len(a & b) / u)
print()
print('=== 两条 caption 的相似度（20000 随机对）===')
print('  平均 Jaccard = %.3f   （1.0 = 完全相同；随机独立抽样会很低）' % (sum(js) / len(js)))
print('  >0.8 的比例  = %.1f%%' % (100.0 * sum(1 for x in js if x > 0.8) / len(js)))

# 3) 与"按频率随机抽 tag"的对照：
#    若真实 caption 只是"高频 tag 的随机子集"，则它的信息量 ≈ 0
freq = [cnt[t] / N for t in tags]
rnd_js = []
for _ in range(4000):
    k_a = len(random.choice(sets))
    a = set(random.choices(tags, weights=freq, k=k_a))
    b = set(random.choices(tags, weights=freq, k=k_a))
    u = len(a | b)
    if u:
        rnd_js.append(len(a & b) / u)
print('  ---- 对照 ----')
print('  按 tag 频率独立随机抽(同大小)的平均 Jaccard = %.3f' % (sum(rnd_js) / len(rnd_js)))
print('  ⇒ 若两者接近 ⇒ caption 的信息量 ≈ 【只知道 tag 频率】', )

# 4) 条件化 AR 真正能学到的：每个 tag 的"存在/不存在"有多好预测
print()
print('=== 最没有信息量的 8 个 tag（出现率过高/过低）===')
for t, c in cnt.most_common(4):
    print('  %.1f%%  %s' % (100.0 * c / N, t))
for t, c in cnt.most_common()[-4:]:
    print('  %.1f%%  %s' % (100.0 * c / N, t))

# 5) 二值熵：把所有 tag 当独立二值变量，算总熵（信息量上界，单位 bit/图）
H = 0.0
for t in tags:
    p = cnt[t] / N
    if 0 < p < 1:
        H += -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
print()
print('=== 信息量上界 ===')
print('  66 个二值 tag 的总熵 = %.1f bit/图（若互相独立）' % H)
print('  图像本身（256x256x3 的 8bit）约 = %d bit' % (256 * 256 * 3 * 8))
print('  ⇒ 条件是图像的 %.2e 比例' % (H / (256 * 256 * 3 * 8)))
