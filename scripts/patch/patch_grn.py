# -*- coding: utf-8 -*-
"""patch_grn.py — 加 mask_mode='full' + --grn-refine 训练路径

mask_mode='full':  注意力全双向（无因果）—— GRN refinement 的前提
--grn-refine:      训练时把输入 token 的一部分替换成随机 token，目标仍是真值
                   F_t = S·Y_true ⊕ (1−S)·Y_rand，  S ~ Bernoulli(l), l ~ U[0,1]

论文依据：GRN (arXiv 2604.13030) Eq.4-7
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
import io

# ---------- ① 模型：加 'full' mask ----------
P = r'D:\work\bitsandbytes-CPU\small_image_model_v2.py'
s = io.open(P, encoding='utf-8-sig').read()

OLD = "        if causal and mask_mode == 'block' and T > 1:"
NEW = ("        if mask_mode == 'full' and T > 1:\n"
       "            # GRN 风格：全双向（无因果）。每个位置都能看到所有其他位置。\n"
       "            # 论文依据：GRN (arXiv 2604.13030) 用 global refinement，\n"
       "            # 所有位置平等、可反复修改，从根本上绕开因果序对 2D 结构的破坏。\n"
       "            y = F.scaled_dot_product_attention(q, kk, vv)\n"
       "            return self.o(y.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd))\n"
       "        if causal and mask_mode == 'block' and T > 1:")
if 'GRN 风格：全双向' in s:
    print('① full mask 已存在')
else:
    assert OLD in s, '找不到 block mask 锚点'
    s = s.replace(OLD, NEW, 1)
    print('① 已加 full mask')

# 配置项注释更新
OLD2 = "    mask_mode: str = 'causal'"
NEW2 = ("    # 'full'   = 全双向（GRN global refinement 的前提）\n"
        "    mask_mode: str = 'causal'")
if "'full'   = 全双向" not in s:
    s = s.replace(OLD2, NEW2, 1)
    print('   配置注释已更新')

io.open(P, 'w', encoding='utf-8', newline='').write(s)

# ---------- ② 训练脚本：加 --grn-refine ----------
Q = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
t = io.open(Q, encoding='utf-8-sig').read()
changed = []

if "'--grn-refine'" not in t:
    anchor = "    ap.add_argument('--mask-mode'"
    assert anchor in t
    # 插到 mask-mode 之前
    ins = ("    ap.add_argument('--grn-refine', action='store_true',\n"
           "                    help='GRN(arXiv 2604.13030) 风格 global refinement 训练：'\n"
           "                         '输入 token 按 l~U[0,1] 随机替换成随机 token，'\n"
           "                         '目标仍是真值，注意力必须全双向（配合 --mask-mode full）。'\n"
           "                         '论文硬数字：refine vs 冻结式 mask 是 FID 3.63 vs 185.62；'\n"
           "                         'random vs confidence 是 3.63 vs 10.64。')\n")
    t = t[:t.find(anchor)] + ins + t[t.find(anchor):]
    changed.append('argparse')

io.open(Q, 'w', encoding='utf-8', newline='').write(t)
print('②', changed if changed else '（无需修改）')
print('行数: model=%d  train=%d' % (s.count('\n') + 1, t.count('\n') + 1))
