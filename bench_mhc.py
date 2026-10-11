# -*- coding: utf-8 -*-
"""bench_mhc.py — mHC 在中国 CPU 上的真实成本实测

要验证三个判断：
  ① Sinkhorn-Knopp 在 CPU 上便宜（论文说 GPU 上要专门做内核融合）
  ② n 条流的访存代价可接受
  ③ 双随机约束 vs 自由矩阵的差异

论文（arXiv 2512.24880, DeepSeek）：
  H̃ = α · mat(x̃' φ) + b        ← 由输入动态生成
  H  = Sinkhorn-Knopp(exp(H̃))   ← 投影到 Birkhoff 多面体（双随机矩阵）
  实测 3B~27B: Amax Gain 3000 → 1.6，额外开销 6.7%
"""
import os, time, sys
import torch
import torch.nn.functional as F

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
print('=' * 84)
print('mHC 成本实测  (线程=%d)' % TH)
print('=' * 84)
print()


def sinkhorn(logits, iters=20):
    """logits: (N, n, n) -> 双随机矩阵 (N, n, n)"""
    H = logits.exp()
    for _ in range(iters):
        H = H / H.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        H = H / H.sum(dim=-2, keepdim=True).clamp_min(1e-9)
    return H


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


# ============ ① Sinkhorn 的开销 ============
print('【①】Sinkhorn-Knopp 开销')
print('  场景A 训练: batch=8, T=256, n=4  => N = 8*256 = 2048 个 4x4 矩阵')
print('  场景B 推理: batch=96, T=1,  n=4  => N = 96 个 4x4 矩阵')
print()
print('  %-12s %-10s %10s %12s %12s' % ('场景', '迭代数', 'N', 'us/次', '相对训练步'))
print('  ' + '-' * 62)

TRAIN_STEP_MS = 2706.5     # 实测: batch=8, T=256, 30.49M 模型
INFER_STEP_MS = 95.8       # 实测: batch=96 生成每步

for name, N, base_ms in (('训练', 2048, TRAIN_STEP_MS), ('推理', 96, INFER_STEP_MS)):
    for it in (0, 5, 10, 20, 40):
        x = torch.randn(N, 4, 4)
        if it == 0:
            t = bench(lambda: x.exp())
        else:
            t = bench(lambda: sinkhorn(x, it))
        print('  %-12s %-10d %10d %12.1f %11.2f%%' % (name, it, N, t * 1e6, 100 * t * 1e3 / base_ms))

print()
print('  ⇒ 论文用 20 次迭代')

# ============ ② 两种实现方式的对比 ============
print()
print('【②】三种实现路径的对比（训练场景 N=2048, iters=20）')
N = 2048
x = torch.randn(N, 4, 4)
t1 = bench(lambda: sinkhorn(x, 20))
print('  纯 PyTorch 逐次归一化:      %8.1f us' % (t1 * 1e6))

# 手动展开 + 原地操作
def sinkhorn_fast(logits, iters=20):
    H = logits.exp()
    for _ in range(iters):
        H = H.div_(H.sum(-1, keepdim=True).clamp_min_(1e-9))
        H = H.div_(H.sum(-2, keepdim=True).clamp_min_(1e-9))
    return H
t2 = bench(lambda: sinkhorn_fast(x, 20))
print('  原地操作版:                 %8.1f us  (%.2fx)' % (t2 * 1e6, t1 / t2))

# log-space 版本（数值更稳，但更慢）
def sinkhorn_log(logits, iters=20):
    H = logits
    for _ in range(iters):
        H = H - H.logsumexp(-1, keepdim=True)
        H = H - H.logsumexp(-2, keepdim=True)
    return H.exp()
t3 = bench(lambda: sinkhorn_log(x, 20))
print('  log-space 版（更稳）:        %8.1f us  (%.2fx)' % (t3 * 1e6, t1 / t3))

# ============ ③ n 条流的访存代价 ============
print()
print('【③】n 条残差流的访存代价')
B, T, d = 8, 256, 512
for n in (1, 2, 4, 8):
    x = torch.randn(B, T, n * d)
    # 模拟混合: (n,n) @ (n,d) 每位置
    A = torch.randn(B * T, n, n)
    X = x.reshape(B * T, n, d)
    def mix():
        return torch.bmm(A, X)
    t = bench(mix, iters=10, warm=3)
    bytes_moved = B * T * n * d * 4 * 2   # 读 X 写 out
    gbps = bytes_moved / t / 1e9
    print('  n=%d  流宽=%4d  %8.1f us  访存 %6.1f MB  达成 %.1f GB/s'
          % (n, n * d, t * 1e6, bytes_moved / 1e6, gbps))
print('  （峰值带宽 29.8 GB/s —— 口径见报告 §10.80；原写 18.8，已被 §10.79 撤回）')

# ============ ④ 对比: 一次标准 matmul 的时间 ============
print()
print('【④】参照：模型里一次 FFN matmul 的时间')
A = torch.randn(8 * 256, 512)
W = torch.randn(512, 2048)
t = bench(lambda: A @ W, iters=20, warm=5)
print('  (2048 x 512) @ (512 x 2048):  %8.1f us' % (t * 1e6))
print('  ⇒ mHC 的 Sinkhorn(%.1f us) 相对它 = %.1f%%' % (t1 * 1e6, 100 * t1 / t))

print()
print('=' * 84)
print('结论')
print('=' * 84)
print('  Sinkhorn 20 迭代: %.1f us' % (t1 * 1e6))
print('  训练步时 2706.5 ms  => 占比 %.3f%%' % (100 * t1 * 1e3 / TRAIN_STEP_MS))
print('  推理步时   95.8 ms  => 占比 %.3f%%' % (100 * bench(lambda: sinkhorn(torch.randn(96,4,4),20)) * 1e3 / INFER_STEP_MS))
print('  论文报告 6.7%（含访存和梯度检查点）')
print('DONE')
