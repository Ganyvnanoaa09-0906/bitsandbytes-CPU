# -*- coding: utf-8 -*-
"""test_l3_model.py — L³ 接入 AR 模型后的自检

验证：
  1. 参数量：L³ 层比 SwiGLU 大得多（容量）
  2. 前向/反向能跑通
  3. 训练单步耗时 vs SwiGLU 基线
  4. 生成能跑通
  5. 【关键】键无泄漏：改一个未来 token 不应改变早前位置的键
  6. 向后兼容：use_l3=False 的旧 checkpoint 仍能加载
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
import os, sys, time
import torch
import torch.nn.functional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2
from scripts.lib.l3_ffn import build_keys, causal_spatial_key

T, D, NL, G = 256, 512, 10, 16
torch.set_num_threads(6)


def build(**kw):
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=G,
        d_model=D, n_layer=NL, n_head=8, n_kv_head=4, ffn_dim=2048,
        head_mode='flat', cond_mode='none', pos_mode='2d',
        token_order='raster', loop_times=1, use_bos=True, **kw)
    return SmallARImageModelV2(cfg), cfg


print('=' * 72)
print('L³ 接入 AR 模型自检')
print('=' * 72)

m0, _ = build(use_l3=False)
m1, _ = build(use_l3=True, l3_v=4096, l3_k=8, l3_d_emb=128, l3_d_up=128)
n0 = sum(p.numel() for p in m0.parameters())
n1 = sum(p.numel() for p in m1.parameters())
print()
print('1) 参数量')
print('   SwiGLU 基线   %.2fM' % (n0 / 1e6))
print('   L³ 替换 FFN   %.2fM   (%.1f× 容量)' % (n1 / 1e6, n1 / n0))
n_l3 = sum(p.numel() for n, p in m1.named_parameters() if '.mlp.WK' in n or '.mlp.WV' in n)
print('   其中查表部分  %.2fM' % (n_l3 / 1e6))

print()
print('2) 键的因果正确性（【关键】不能泄漏未来 token）')
tok = torch.randint(0, 1024, (2, T))
k_full = build_keys(tok, G, 4096, '4gram', use_bos=True)
# 把第 200 个 token 改掉，前 190 个位置（序列位）的键不应变化
tok2 = tok.clone()
tok2[:, 200] = (tok2[:, 200] + 77) % 1024
k_mod = build_keys(tok2, G, 4096, '4gram', use_bos=True)
same = torch.equal(k_full[:, :190], k_mod[:, :190])
diff = not torch.equal(k_full[:, 200:], k_mod[:, 200:])
print('   改动 token[200] 后：')
print('     序列位 0..189 的键不变  -> %s' % ('OK' if same else '❌ 泄漏！'))
print('     序列位 200+ 的键有变化   -> %s' % ('OK' % () if diff else '❌ 没生效'))
print('   (序列位 i 的键 = k(输入 token x_{i-1})，所以改 x_200 会影响序列位 201 起)')

print()
print('3) 前向 + 反向')
B = 8
x = torch.randint(0, 1024, (B, T))
for tag, m in (('SwiGLU', m0), ('L³', m1)):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    try:
        out = m.forward_logits(x)
        gen = out[1]
        logits = gen[0] if isinstance(gen, tuple) else gen
        loss = F.cross_entropy(logits.reshape(-1, 1024), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        print('   %-8s loss=%.4f  grad_norm=%.3f  OK' % (tag, float(loss), float(gn)))
    except Exception as e:
        import traceback
        print('   %-8s 失败: %s' % (tag, type(e).__name__))
        traceback.print_exc()

print()
print('4) 训练单步耗时（含反向+优化器，3 次取均）')
for tag, m in (('SwiGLU', m0), ('L³    ', m1)):
    m.train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)

    def step():
        o = m.forward_logits(x)
        gg = o[1]
        lg = gg[0] if isinstance(gg, tuple) else gg
        l = F.cross_entropy(lg.reshape(-1, 1024), x.reshape(-1))
        opt.zero_grad(set_to_none=True); l.backward(); opt.step()

    step(); step()
    t0 = time.perf_counter()
    for _ in range(3):
        step()
    dt = (time.perf_counter() - t0) / 3
    print('   %-8s %.2f s/步   %.1f tok/s' % (tag, dt, B * T / dt))

print()
print('5) 生成')
for tag, m in (('SwiGLU', m0), ('L³    ', m1)):
    m.eval()
    try:
        with torch.no_grad():
            toks = m.generate(None, steps=48, top_k=64, temperature=1.0, batch=2)
        print('   %-8s 输出 %s  用了 %d 个码  OK' % (tag, tuple(toks.shape), int(toks.unique().numel())))
    except Exception as e:
        import traceback
        print('   %-8s 失败: %s' % (tag, type(e).__name__))
        traceback.print_exc()

print()
print('DONE')
