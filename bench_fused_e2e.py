# -*- coding: utf-8 -*-
"""bench_fused_e2e.py — 融合核接进真实 AR 模型的端到端收益

做法：monkey-patch 模型的 RMSNorm / SwiGLU / LayerScale 残差，
      对比 patch 前後的训练步时与 loss（loss 必须几乎一致）。
"""
import sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
torch.set_num_threads(6)
print('=' * 82)
print('融合核接入真实 AR 模型的端到端收益')
print('=' * 82)

import fused_cpu
import small_image_model_v2 as M2

B, T = 8, 256
cfg = M2.SmallImageConfigV2(
    vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
    ffn_dim=2048, total_tokens=T, grid=16, think_tokens=16,
    text_dim=512, ada_dim=128, cond_mode='none', cond_dropout=0.1,
    pos_mode='2d', token_order='raster', head_mode='flat', n_cluster=32,
    hier=False, loop_start=4, loop_end=7, loop_times=1, use_bos=True)
torch.manual_seed(0)
model = M2.SmallARImageModelV2(cfg)
model.train()
tok = torch.randint(0, 1024, (B, T))
nparam = sum(p.numel() for p in model.parameters())
flops = 2 * nparam * B * T * 3


def run_steps(model, opt, n=5, warm=2):
    """返回 (平均步时, 最后一个 loss)"""
    for _ in range(warm):
        t_, g_ = model.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.perf_counter()
    last = None
    for _ in range(n):
        t_, g_ = model.forward_logits(tok)
        g_ = g_[0] if isinstance(g_, (tuple, list)) else g_
        loss = F.cross_entropy(g_.reshape(-1, 1024), tok.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    return (time.perf_counter() - t0) / n, last


print()
print('[A] 基线（eager RMSNorm / SwiGLU / 残差）')
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
t_base, l_base = run_steps(model, opt)
print('    步时 %.3f s   loss %.4f   有效算力 %.1f GFLOPS'
      % (t_base, l_base, flops / t_base / 1e9))

# ---------------- patch ----------------
print()
print('[B] 打补丁：把 3 个算子链换成融合核')

orig_rms_forward = M2.RMSNorm.forward
orig_swiglu_forward = M2.SwiGLU.forward


def patched_rms_forward(self, x):
    return fused_cpu.fused_rmsnorm(x, self.w, None, self.eps)


def patched_swiglu_forward(self, x):
    a, b = self.w12(x).chunk(2, dim=-1)
    # 原来的 F.silu(a) * b（2 个算子）→ 融合核（1 个）
    return self.w3(fused_cpu.fused_swiglu(a, b))


M2.RMSNorm.forward = patched_rms_forward
M2.SwiGLU.forward = patched_swiglu_forward

# 残差 + LayerScale：patch BlockV2.forward 的残差部分
orig_block_forward = M2.BlockV2.forward


def patched_block_forward(self, x, pooled=None, ctx=None, cache=None, append=True, keys=None):
    if self.cm == 'adaln' and pooled is not None:
        return orig_block_forward(self, x, pooled, ctx, cache, append, keys)
    h = self.ln1(x)
    x = fused_cpu.fused_add_scale(x, self.attn(h, cache=cache, append=append), self.ls1)
    x = fused_cpu.fused_add_scale(x, self.mlp(self.ln2(x)), self.ls2)
    return x


M2.BlockV2.forward = patched_block_forward

torch.manual_seed(0)
model2 = M2.SmallARImageModelV2(cfg)
model2.load_state_dict(model.state_dict())
model2.train()
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
t_fus, l_fus = run_steps(model2, opt2)
print('    步时 %.3f s   loss %.4f   有效算力 %.1f GFLOPS'
      % (t_fus, l_fus, flops / t_fus / 1e9))

print()
print('[C] 结论')
print('    融合前 %.3f s  融合后 %.3f s' % (t_base, t_fus))
sp = (t_base / t_fus - 1) * 100
print('    ⇒ 【提速 %.1f%%】  (%.2fx)' % (sp, t_base / t_fus))
print('    loss 差异 %.6f  %s' % (abs(l_base - l_fus),
                                   'OK' if abs(l_base - l_fus) < 1e-3 else 'WARN'))

# ---------------- 混合精度无损对比 ----------------
print()
print('[D] 数值等价性（同权重，前向 logits 对比）')
model.eval(); model2.eval()
with torch.no_grad():
    _, g1 = model.forward_logits(tok)
    _, g2 = model2.forward_logits(tok)
g1 = g1[0] if isinstance(g1, (tuple, list)) else g1
g2 = g2[0] if isinstance(g2, (tuple, list)) else g2
d = (g1 - g2).abs().max().item()
print('    logits 最大绝对差 %.3e  %s' % (d, 'OK' if d < 1e-4 else 'WARN'))

print()
print('=' * 82)
print('DONE')
