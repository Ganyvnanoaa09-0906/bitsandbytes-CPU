# -*- coding: utf-8 -*-
"""patch_grn_loop.py — 把 GRN refinement 的输入构造接进训练循环

改动点：训练时把 forward_logits 的输入 x 换成 F_t = S·x ⊕ (1−S)·x_rand，
        【目标仍是 x】（真值），这就是 GRN 的 self-refinement 训练。
        l ~ U[0,1] 每步重采（论文 Eq.6）。
"""
import os, sys
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
import io

P = r'D:\work\bitsandbytes-CPU\train_ar_v2.py'
s = io.open(P, encoding='utf-8-sig').read()
n = 0

# ---- flat 路径：L478 `(thc, thd), (gec, ged) = model.forward_logits(x, c)` ----
OLD1 = "            (thc, thd), (gec, ged) = model.forward_logits(x, c)"
NEW1 = ("            # ---- GRN global refinement（arXiv 2604.13030）----\n"
        "            # 输入按 l~U[0,1] 随机替换成随机 token，【目标仍是真值 x】\n"
        "            # 论文硬数字：refine vs 冻结式 mask = FID 3.63 vs 185.62\n"
        "            x_in = x\n"
        "            if getattr(a, 'grn_refine', False):\n"
        "                from grn_refine import grn_make_input\n"
        "                x_in, _grn_l, _grn_S = grn_make_input(x, cfg.vocab_size)\n"
        "            (thc, thd), (gec, ged) = model.forward_logits(x_in, c)")
if OLD1 in s:
    s = s.replace(OLD1, NEW1, 1)
    n += 1
    print('① flat 路径已接入')
else:
    print('① 未找到 flat 路径锚点')

# ---- hier 路径：L478 附近的 forward_logits(x, c)（hier 分支里）----
# hier 路径同样用 `model.forward_logits(x, c)`，若上一步只替换了第一处，
# 这里再找剩下的
if s.count("model.forward_logits(x, c)") > 0:
    cnt = s.count("model.forward_logits(x, c)")
    # 只替换 hier 分支里的那一处（用缩进区分：hier 分支 16 空格，flat 分支 12 空格）
    OLD2 = "                (thc, thd), (gec, ged) = model.forward_logits(x, c)"
    if OLD2 in s:
        NEW2 = ("                x_in = x\n"
                "                if getattr(a, 'grn_refine', False):\n"
                "                    from grn_refine import grn_make_input\n"
                "                    x_in, _grn_l, _grn_S = grn_make_input(x, cfg.vocab_size)\n"
                "                (thc, thd), (gec, ged) = model.forward_logits(x_in, c)")
        s = s.replace(OLD2, NEW2, 1)
        n += 1
        print('② hier 路径已接入（剩余 %d 处未改）' % (s.count("model.forward_logits(x, c)")))
    else:
        print('② hier 路径锚点未找到（可能缩进不同），剩余 %d 处未改'
              % s.count("model.forward_logits(x, c)"))

# ---- 启动时打印提示 ----
if '[grn]' not in s:
    anchor = "    print('[v2] params="
    i = s.find(anchor)
    if i >= 0:
        ins = ("    if getattr(a, 'grn_refine', False):\n"
               "        print('[grn] global refinement 训练：输入按 l~U[0,1] 随机替换，'\n"
               "              '目标为真值；mask_mode 应为 full（全双向）', flush=True)\n"
               "        if getattr(a, 'mask_mode', 'causal') != 'full':\n"
               "            print('[grn] ⚠️ grn_refine 要求 mask_mode=full', flush=True)\n")
        s = s[:i] + ins + s[i:]
        n += 1
        print('③ 启动提示已加')

io.open(P, 'w', encoding='utf-8', newline='').write(s)
print('共改 %d 处，行数 %d' % (n, s.count('\n') + 1))
