# -*- coding: utf-8 -*-
"""test_looped.py — 验证循环 Transformer 的三个组件

检查：
  ① 步数编码：初始为 0（不干扰），训练范围内用可学习表，超范围外推
  ② 跨轮门控残差：初始接近恒等（不干扰），梯度可达
  ③ 循环前向：形状正确、L 变化时 KV cache 正确
  ④ 成本：不同 L 下的 pass 数与实测时间
"""
import os, sys
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
import sys, time
import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from scripts.lib.loop_transformer import LoopStepEncoding, GatedCrossIteration, LoopConfig, make_loop_modules, run_blocks_looped

torch.set_num_threads(6)
torch.manual_seed(0)
print('=' * 78)
print('循环 Transformer 组件验证')
print('=' * 78)

# ---- ① 步数编码 ----
print()
print('【①】循环步数编码')
enc = LoopStepEncoding(d_model=512, max_steps=8)
print('  可学习表初始值全 0:', bool((enc.table.weight == 0).all()))
v0 = enc(0, torch.device('cpu'), torch.float32)
print('  step=0 编码范数: %.6f  (应为 0 不干扰)' % float(v0.norm()))
with torch.no_grad():
    enc.table.weight.normal_(0, 1.0)
v1 = enc(1, torch.device('cpu'), torch.float32)
v2 = enc(2, torch.device('cpu'), torch.float32)
v9 = enc(9, torch.device('cpu'), torch.float32)
print('  step=1 范数 %.3f   step=2 范数 %.3f  (可学习, 不同轮不同)' % (float(v1.norm()), float(v2.norm())))
print('  step=9 (超出训练范围) 范数 %.3f  -> 正弦外推' % float(v9.norm()))
print('  step=1 与 step=9 不同:', not torch.allclose(v1, v9))

# ---- ② 跨轮门控残差 ----
print()
print('【②】跨轮门控残差')
cr = GatedCrossIteration(d_model=512, max_steps=8, init_bias=-4.0)
x = torch.randn(2, 16, 512)
xe = torch.randn(2, 16, 512)
out = cr(x, xe, step=1)
delta = float((out - x).abs().mean() / x.abs().mean())
print('  初始相对改变量: %.5f  (应极小 -> 不干扰原有行为)' % delta)
out.sum().backward()
ng = sum(1 for p in cr.parameters() if p.grad is not None)
print('  梯度可达: %d/%d' % (ng, sum(1 for _ in cr.parameters())))
print('  参数量: %d' % sum(p.numel() for p in cr.parameters()))

# ---- ③ 循环前向 ----
print()
print('【③】循环前向 + KV cache 一致性')


class DummyBlock(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = torch.nn.Linear(d, d)

    def forward(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
        y = self.lin(x)
        if cache is not None:
            if cache.get('n') is None:
                cache['n'] = 0
            if append:
                cache['n'] += x.shape[1]
        return x + y


d, T = 512, 256
blocks = torch.nn.ModuleList([DummyBlock(d) for _ in range(10)])
loop_idx = [4, 5, 6, 7]
lcfg = LoopConfig(loop_start=4, loop_end=7, loop_L=3, loop_L_max=3, loop_random=False)
mods = make_loop_modules(type('C', (), {'d_model': d})(), lcfg)

x = torch.randn(2, T, d)
caches = [dict(k=None, v=None) for _ in range(10)]
out3 = run_blocks_looped(blocks, x, loop_idx, lcfg, mods, None, None, caches)
print('  L=3 输出形状 %s  (输入 %s)' % (tuple(out3.shape), tuple(x.shape)))

print('  KV cache 一致性（L 变化时 cache 长度应只增长一次序列长度）:')
for L in (1, 2, 4):
    lc = LoopConfig(loop_L=L, loop_L_max=L, loop_random=False)
    cs = [dict(k=None, v=None) for _ in range(10)]
    run_blocks_looped(blocks, x, loop_idx, lc, mods, None, None, cs)
    ns = sorted(set(c['n'] for c in cs))
    print('    L=%d -> cache 长度集合 %s  (应只有 %d)' % (L, ns, T))

lc3 = LoopConfig(loop_L_max=3, loop_random=True)
Ls = sorted(set(lc3.sample_L() for _ in range(30)))
print('  随机采样 L 的取值: %s' % Ls)

# ---- ④ 成本 ----
print()
print('【④】成本')
print('  公式: pass 数 = n_layer + |loop_idx| x (L - 1)')
for L in (1, 2, 3, 4):
    p = 10 + 4 * (L - 1)
    print('    L=%d -> %2d pass  (相对 L=1 的 %.2fx)' % (L, p, p / 10.0))
print('  随机 L~U{1,2,3} 平均 L=2 -> %d pass (%.2fx)' % (10 + 4, 1.4))
print()
print('  实测（dummy block, 本地 6 线程）:')
for L in (1, 2, 3):
    lc = LoopConfig(loop_L=L, loop_L_max=L, loop_random=False)
    for _ in range(2):
        cs = [dict(k=None, v=None) for _ in range(10)]
        run_blocks_looped(blocks, x, loop_idx, lc, mods, None, None, cs)
    t0 = time.perf_counter()
    for _ in range(5):
        cs = [dict(k=None, v=None) for _ in range(10)]
        run_blocks_looped(blocks, x, loop_idx, lc, mods, None, None, cs)
    t = (time.perf_counter() - t0) / 5
    print('    L=%d  %.2f ms  相对 L=1: %.2fx' % (L, t * 1e3, t * 1e3 / 1.0))

print()
print('=' * 78)
print('DONE')
