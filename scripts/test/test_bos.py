# -*- coding: utf-8 -*-
"""test_bos.py — 验证 BOS 改动正确：形状、损失对齐、生成从 BOS 起步。

关键验证点：
  1. use_bos=True/False 的参数量差恰好 2*d（bos_emb + bos_pos）
  2. use_bos=True 时 forward_logits 返回 (B,T,...)，可【不移位】直接对 x 算 CE
  3. use_bos=False 的老路径仍然要移位（向后兼容）
  4. generate() 两条路径都能跑通，输出形状 (B,T)
  5. BOS 确实被用上了（对比生成结果与旧路径不同）
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
import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

T, D, NL = 256, 512, 10


def build(use_bos):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=16,
        d_model=D, n_layer=NL, n_head=8, n_kv_head=4, ffn_dim=2048,
        head_mode='flat', cond_mode='none', pos_mode='2d',
        token_order='raster', loop_times=1, use_bos=use_bos,
    )
    return SmallARImageModelV2(cfg), cfg


print('=' * 70)
print('BOS 改动验证')
print('=' * 70)

torch.manual_seed(0)
m0, cfg0 = build(False)
m1, cfg1 = build(True)
n0 = sum(p.numel() for p in m0.parameters())
n1 = sum(p.numel() for p in m1.parameters())
print()
print('1) 参数量')
print('   use_bos=False  %.6fM' % (n0 / 1e6))
print('   use_bos=True   %.6fM' % (n1 / 1e6))
print('   差值 %d  期望 2*d = %d   %s'
      % (n1 - n0, 2 * D, 'OK' if n1 - n0 == 2 * D else '❌'))

print()
print('2) 前向形状与损失对齐')
B = 4
x = torch.randint(0, 1024, (B, T))
for tag, m, ub in (('use_bos=False', m0, False), ('use_bos=True', m1, True)):
    out = m.forward_logits(x)
    gen = out[1]
    logits = gen[0] if isinstance(gen, tuple) else gen
    if ub:
        src, tgt = logits, x                     # 不移位
    else:
        src, tgt = logits[:, :-1], x[:, 1:]      # 移位
    loss = F.cross_entropy(src.reshape(-1, 1024), tgt.reshape(-1))
    print('   %-14s logits=%s  src=%s  tgt=%s  CE=%.4f'
          % (tag, tuple(logits.shape), tuple(src.shape), tuple(tgt.shape), float(loss)))

print()
print('3) BOS 是否真的参与了计算')
# 把 bos_emb 归零，输出应该改变
x2 = torch.randint(0, 1024, (2, T))
with torch.no_grad():
    a = m1.forward_logits(x2)[1][0].clone()
    saved = m1.bos_emb.data.clone()
    m1.bos_emb.data.zero_()
    b = m1.forward_logits(x2)[1][0].clone()
    m1.bos_emb.data.copy_(saved)
diff = float((a - b).abs().max())
print('   把 bos_emb 置零后 logits 最大变化 %.6f  %s'
      % (diff, 'OK（BOS 有作用）' if diff > 1e-6 else '❌ BOS 没被用上'))

print()
print('4) generate() 两条路径')
for tag, m, ub in (('use_bos=False', m0, False), ('use_bos=True', m1, True)):
    try:
        m.eval()
        with torch.no_grad():
            toks = m.generate(None, steps=32, top_k=64, temperature=1.0, batch=2, seed=1)
        u = int(toks.unique().numel())
        print('   %-14s 输出 %s  用了 %d 个码  %s'
              % (tag, tuple(toks.shape), u, 'OK' if toks.shape == (2, 32) else '❌'))
    except Exception as e:
        import traceback
        print('   %-14s 失败: %s' % (tag, type(e).__name__))
        traceback.print_exc()

print()
print('5) 向后兼容：旧 checkpoint（无 use_bos 字段）能否加载')
import glob
olds = glob.glob(r'D:\work\bitsandbytes-CPU\.autopilot\v2_run2\ckpt_*.pt')
if olds:
    ck = torch.load(olds[-1], map_location='cpu', weights_only=False)
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    c2 = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
    mm = SmallARImageModelV2(c2)
    r = mm.load_state_dict(ck['model'], strict=False)
    print('   旧 ckpt cfg 里有 use_bos 吗: %s' % ('use_bos' in ck['cfg']))
    print('   load_state_dict: missing=%s unexpected=%s'
          % (len(r.missing_keys), len(r.unexpected_keys)))
    print('   %s' % ('OK' if not r.unexpected_keys and not r.missing_keys else '❌ 不兼容'))
else:
    print('   (未找到旧 checkpoint，跳过)')
print()
print('DONE')
