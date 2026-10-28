# -*- coding: utf-8 -*-
"""bench_fused_e2e3.py — v2 融合核的端到端收益（含消融）

v1 的教训：
  RMSNorm +3.5% ✅ / SwiGLU -6.5% ❌ / 残差 -7.4% ❌
v2 的改动：
  SwiGLU 支持 stride（零拷贝吃 chunk 视图）+ 向量化反向
  新增「残差+LayerScale+RMSNorm」8 算子融合链
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
import torch.nn
import torch.nn.functional

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 82)
print('v2 融合核端到端收益')
print('=' * 82)

from scripts.lib import fused_cpu
from scripts.lib import small_image_model_v2

B, T = 8, 256
cfg = M2.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
torch.manual_seed(0)
model = M2.SmallARImageModelV2(cfg)
tok = torch.randint(0, 1024, (B, T))
nparam = sum(p.numel() for p in model.parameters())
flops = 2 * nparam * B * T * 3

ORIG = {
    'rms': M2.RMSNorm.forward,
    'swiglu': M2.SwiGLU.forward,
    'block': M2.BlockV2.forward,
}


def reset():
    M2.RMSNorm.forward = ORIG['rms']
    M2.SwiGLU.forward = ORIG['swiglu']
    M2.BlockV2.forward = ORIG['block']


# ---------------- patches ----------------
def p_rms(self, x):
    return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)


def p_swiglu(self, x):
    a, b = self.w12(x).chunk(2, dim=-1)
    return self.w3(fused_cpu.fused_swiglu(a, b))   # 零拷贝（stride 版）


def p_block_v1style(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    """v1 风格：残差用 fused_add_scale"""
    if self.cm == 'adaln' and pooled is not None:
        return ORIG['block'](self, x, pooled, ctx, cache, append, keys)
    h = self.ln1(x)
    x = fused_cpu.fused_add_scale(x, self.attn(h, cache=cache, append=append), self.ls1)
    x = fused_cpu.fused_add_scale(x, self.mlp(self.ln2(x)), self.ls2)
    return x


def p_block_chain(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    """v2 风格：把「残差 + LayerScale + 下一个 RMSNorm 的输入准备」合成一条链

    ln2 的输入是 x + ls1*attn(ln1(x))，而这正好又是一次 6 算子的 RMSNorm
    的输入 —— 两个链合并成 8 个算子。
    ⚠️ 不能用 in-place 版（x 之后还要用），所以这里先算 h 再 norm。
    """
    if self.cm == 'adaln' and pooled is not None:
        return ORIG['block'](self, x, pooled, ctx, cache, append, keys)
    h = fused_cpu.fused_rmsnorm(x, self.ln1.w, None, self.ln1.eps)
    a = self.attn(h, cache=cache, append=append)
    # x + ls1*a 就是 ln2 的输入；用 fused_add_scale 一步得到（不再单独做残差加）
    h2 = fused_cpu.fused_add_scale(x, a, self.ls1)      # h2 = x + ls1*a
    x = fused_cpu.fused_add_scale(h2, self.mlp(fused_cpu.fused_rmsnorm(h2, self.ln2.w, None, self.ln2.eps)), self.ls2)
    return x


print()
print('[1] 数值等价性（同一权重，只做前向）')
model.eval()
with torch.no_grad():
    _, g_ref = model.forward_logits(tok)
g_ref = g_ref[0] if isinstance(g_ref, (tuple, list)) else g_ref
sc = g_ref.abs().max().item()

cases = [
    ('只 patch RMSNorm', lambda: setattr(M2.RMSNorm, 'forward', p_rms)),
    ('只 patch SwiGLU(v2 strided)', lambda: setattr(M2.SwiGLU, 'forward', p_swiglu)),
    ('只 patch Block(v1 残差)', lambda: setattr(M2.BlockV2, 'forward', p_block_v1style)),
    ('RMSNorm + SwiGLU', lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                                  setattr(M2.SwiGLU, 'forward', p_swiglu))),
    ('RMSNorm + SwiGLU + Block链', lambda: (setattr(M2.RMSNorm, 'forward', p_rms),
                                            setattr(M2.SwiGLU, 'forward', p_swiglu),
                                            setattr(M2.BlockV2, 'forward', p_block_chain))),
]
for label, fn in cases:
    reset(); fn()
    model.eval()
    with torch.no_grad():
        _, gx = model.forward_logits(tok)
    gx = gx[0] if isinstance(gx, (tuple, list)) else gx
    d = (gx - g_ref).abs().max().item()
    print('    %-30s 相对 %.3e  %s' % (label, d / sc, 'OK' if d / sc < 1e-4 else 'FAIL'))
reset()

# ---------------- 训练步时 ----------------
def run_steps(m, opt, n=5, warm=2):
    m.train()
    for _ in range(warm):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.perf_counter()
    for _ in range(n):
        t_, g_ = m.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return (time.perf_counter() - t0) / n


print()
print('[2] 训练步时 + 消融')


def measure(label, fn):
    reset(); fn()
    torch.manual_seed(0)
    m = M2.SmallARImageModelV2(cfg)
    o = torch.optim.AdamW(m.parameters(), lr=1e-4)
    t = run_steps(m, o)
    return t


t_base = measure('baseline', lambda: None)
print('    %-30s %.3f s/步   %6.1f GFLOPS   【基线】' % ('eager', t_base, flops / t_base / 1e9))
results = {}
for label, fn in cases:
    t = measure(label, fn)
    results[label] = t
    print('    %-30s %.3f s/步   %6.1f GFLOPS   【%+.1f%%】'
          % (label, t, flops / t / 1e9, (t_base / t - 1) * 100))

print()
print('[3] 结论')
best = min(results, key=results.get)
print('    最好组合: %s' % best)
print('    基线 %.3f s  →  %.3f s  ⇒  【提速 %.1f%%】(%.3fx)'
      % (t_base, results[best], (t_base / results[best] - 1) * 100, t_base / results[best]))

reset()
print()
print('=' * 82)
print('DONE')
