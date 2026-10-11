"""One training step, accounted analytically: FLOP-bound vs byte-bound.

Why analytical: the GEMM kernel is at its hardware wall (§7.11), so the
remaining headroom must be in the other 38% of the step -- norms, activations,
elementwise, Adam. Before touching any of them, work out where the time SHOULD
go and which resource each op class is limited by.

Model (from the checkpoint config):
  d_model 512, n_layer 10, n_head 8, n_kv_head 4, ffn_dim 2048, vocab 1024,
  T = 256 tokens, batch 8  ->  2048 tokens/step, 30.49 M params

Measured anchors used below (all taken today, same machine):
  training throughput                105 GFLOPS   (3.585 s/step at 375 GFLOP)
  fp32 GEMM ceiling on this box      185-215 GFLOPS
  L1 read bandwidth                  470 GB/s (aggregate, 6 threads)
  DRAM read                          22-44 GB/s
  DRAM triad (2r+1w)                 18-21 GB/s
"""
import sys

sys.stdout.reconfigure(encoding='utf-8')

D, L, H, KV, F, V = 512, 10, 8, 4, 2048, 1024
T, B = 256, 8
TOK = T * B                      # 2048 tokens per step
P = 30.49e6
MEAS_STEP_S = 2.854              # best observed for this stage
GEMM_CEIL = 200e9                # FP32 GEMM ceiling measured on this box
DRAM_R = 33e9                    # read bandwidth, middle of 22-44
TRIAD = 20e9                     # 2 reads + 1 write bound

print('每步：%d token，%d 层，d_model=%d，ffn=%d，vocab=%d，参数 %.2f M'
      % (TOK, L, D, F, V, P / 1e6))
print('实测步时 %.3f s ⇒ 实际吞吐 %.1f GFLOPS\n' % (MEAS_STEP_S, 375e9 / MEAS_STEP_S / 1e9))

# ---- FLOPs per op class (fwd + bwd = 3x fwd for matmuls) ----------------
rows = []
def add(name, flops, rd, wr, kind):
    rows.append(dict(name=name, flops=flops, rd=rd, wr=wr, kind=kind))

# matmul FLOPs (fwd) = 2 * tokens * in * out ; bwd ~ 2x fwd
qkv = 2 * TOK * D * 3 * D
attn_out = 2 * TOK * D * D
mlp_up = 2 * TOK * D * 2 * F          # gate + up
mlp_down = 2 * TOK * F * D
head = 2 * TOK * D * V
scores = 2 * B * H * T * T * (D // H) * 2   # qk^T and attn@v
per_layer_f = qkv + attn_out + mlp_up + mlp_down + scores
mm_flops = (per_layer_f * L + head) * 3      # x3 for fwd+bwd
# bytes: activations for the matmuls (fp32, in+out per layer, no cache reuse assumed)
layer_act = (TOK * D * 4) * 6                # rough: several activation tensors
mm_rd = layer_act * L * 3
add('矩阵乘（含反向）', mm_flops, mm_rd, mm_rd * 0.5, 'compute')

# ---- norms: elementwise, byte-bound -----------------------------------
# 3 norms per layer (pre-attn, qk-norm, pre-mlp) + final
norm_elems = TOK * D * 3 * L
add('LayerNorm/RMSNorm', norm_elems * 6,        # ~6 flop/element
    norm_elems * 4 * 2, norm_elems * 4, 'bytes')

# ---- activation + elementwise (SwiGLU, residual, scales) --------------
swiglu = TOK * 2 * F
add('SwiGLU 激活', swiglu * 8, swiglu * 4 * 2, swiglu * 4, 'bytes')
resid = TOK * D * L
add('残差/缩放', resid * 2, resid * 4 * 2, resid * 4, 'bytes')

# ---- softmax over attention -------------------------------------------
sm = B * H * T * T * L
add('attention softmax', sm * 5, sm * 4 * 2, sm * 4, 'bytes')

# ---- loss -------------------------------------------------------------
lg = TOK * V
add('loss（交叉熵）', lg * 5, lg * 4, lg * 4, 'compute')

# ---- Adam: reads g,m,v,p writes m,v,p -> ~6 passes over params ---------
add('Adam 更新', P * 5,
    P * 4 * 3,      # read g, m, v
    P * 4 * 3,      # write p, m, v
    'bytes')

print('%-22s %12s %12s %12s %10s' % ('算子类', 'GFLOP', '读 MB', '写 MB', '受限类型'))
print('-' * 74)
tot_f = tot_rd = tot_wr = 0
for r in rows:
    tot_f += r['flops']; tot_rd += r['rd']; tot_wr += r['wr']
    print('%-22s %12.1f %12.1f %12.1f %10s'
          % (r['name'], r['flops'] / 1e9, r['rd'] / 1e6, r['wr'] / 1e6, r['kind']))
print('-' * 74)
print('%-22s %12.1f %12.1f %12.1f' % ('合计', tot_f / 1e9, tot_rd / 1e6, tot_wr / 1e6))

print('\n=== 两个界各是多少时间 ===')
t_compute = tot_f / GEMM_CEIL
t_dram = (tot_rd + tot_wr) / DRAM_R
t_triad = (2 * tot_rd + tot_wr) / TRIAD
print('  FLOPs 界（按 GEMM 天花板 %.0f GFLOPS）: %.3f s' % (GEMM_CEIL / 1e9, t_compute))
print('  字节界（按 DRAM 读 %.0f GB/s）        : %.3f s' % (DRAM_R / 1e9, t_dram))
print('  字节界（按 triad 2读1写 %.0f GB/s）   : %.3f s' % (TRIAD / 1e9, t_triad))
print('  实测                                   : %.3f s' % MEAS_STEP_S)
print('\n  纯矩阵乘那一项的 FLOPs 界 = %.3f s (%.0f%% of 实测)'
      % (mm_flops / GEMM_CEIL, mm_flops / GEMM_CEIL / MEAS_STEP_S * 100))

print('\n=== Adam 单独看（最可能的隐藏大头）===')
adam_bytes = P * 4 * 6
print('  Adam 每步搬 %.0f MB（读 g/m/v，写 m/v/p）' % (adam_bytes / 1e6))
print('  按 triad 上限 ⇒ %.3f s' % (adam_bytes / TRIAD))
print('  占实测步时的 %.0f%%' % (adam_bytes / TRIAD / MEAS_STEP_S * 100))
print('  ⇒ 若实测里它明显高于这个数 ⇒ 有优化空间；若已经贴着 ⇒ 只能减少搬运量')

print('\n判读：')
print('  · 逐元素/norm/激活这些算子的 FLOPs 很小，它们全部是【字节界】的')
print('  · 所以"再快一点"的着力点有两个方向：')
print('      (a) 减少搬运量（融合、原地、低精度存储、少一次读写）')
print('      (b) 让搬运更靠近 L1/L2（分块、tiling），把 DRAM 换成缓存带宽')
print('  · 而它们的时间下界可以直接算出来 ⇒ 谁离下界远，谁就有余量 ✓')
