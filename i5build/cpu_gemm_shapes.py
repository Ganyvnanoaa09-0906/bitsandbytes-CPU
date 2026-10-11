"""训练步的算力去哪了：把模型里每个 GEMM 的形状单独测一遍。

判据：
  · 若各形状的效率之和 ≈ 104.6 GFLOPS（当前整步效率）⇒ 整步已经在形状天花板上，
    差距全在"形状太小"，非 GEMM 算子不是瓶颈 ⇒ 该动的是 batch/形状
  · 若各形状效率明显更高（比如 150+）⇒ 差距在非 GEMM 算子与调度 ⇒ 该动的是融合与去开销

形状取自 checkpoint 的配置（不加载权重，只读 config，内存开销可忽略）。
"""
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')
import torch  # noqa: E402

CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ar3\ckpt.pt'
import glob  # noqa: E402
import os  # noqa: E402

cands = sorted(glob.glob(os.path.join(CK, 'ckpt_*.pt'))) or sorted(
    glob.glob(r'D:\work\bitsandbytes-CPU\i5build\loop_ar2\ckpt.pt\ckpt_*.pt'))
path = cands[-1]
print('读配置:', path)
d = torch.load(path, map_location='cpu', weights_only=False)
cfg = d.get('cfg', {})
keys = ['d_model', 'n_layer', 'n_head', 'n_kv', 'hd', 'd_up', 'total_tokens',
        'vocab', 'batch', 'loop_start', 'loop_end']
for k in keys:
    if k in cfg:
        print('   %-14s %s' % (k, cfg[k]))
print('   （全部键：%s）' % ', '.join(sorted(cfg.keys())[:18]))

dm = cfg.get('d_model') or cfg.get('d_emb') or 512
nl = cfg.get('n_layer', 10)
dup = cfg.get('d_up') or dm * 4
V = cfg.get('vocab') or cfg.get('codebook') or 1024
T = cfg.get('total_tokens', 256)
B = 8
print('\n假设：d_model=%d  n_layer=%d  d_up=%d  vocab=%d  batch=%d  T=%d'
      % (dm, nl, dup, V, B, T))

torch.set_num_threads(2)          # 训练占着 6 核，用 2 核测形状（口径一致即可）
M = B * T
SHAPES = [
    ('attn qkv  (M,dm)x(dm,3dm)', (M, dm, 3 * dm)),
    ('attn out  (M,dm)x(dm,dm)', (M, dm, dm)),
    ('mlp up    (M,dm)x(dm,2dup)', (M, dm, 2 * dup)),
    ('mlp down  (M,dup)x(dup,dm)', (M, dup, dm)),
    ('head      (M,dm)x(dm,V)', (M, dm, V)),
    ('attn score(M,hd)x(hd,T)', (B * cfg.get('n_head', 8) * T, cfg.get('hd', 64), T)),
]

print('\n%-30s %12s %10s %12s' % ('形状', 'M×N×K', 'GFLOPS', '占比'))
tot_flop = 0.0
res = []
for name, (m, n, k) in SHAPES:
    try:
        a = torch.randn(m, k)
        b = torch.randn(k, n)
        c = a @ b
        t0 = time.perf_counter()
        for _ in range(3):
            c = a @ b
        dt = (time.perf_counter() - t0) / 3
        g = 2.0 * m * n * k / dt / 1e9
        res.append((name, m, n, k, g, dt))
        print('%-30s %5d×%-5d×%-5d %10.1f' % (name, m, n, k, g))
    except Exception as e:
        print('%-30s 失败 %s' % (name, str(e)[:40]))

print('\n【每层】重复数：qkv/out/mlp up/down 各 %d 次（n_layer）' % nl)
per_layer_s = sum(dt for (nm, m, n, k, g, dt) in res
                  if not nm.startswith('head') and not nm.startswith('attn score'))
per_layer_s *= nl
head_s = sum(dt for (nm, m, n, k, g, dt) in res if nm.startswith('head'))
tot_matmul_s = per_layer_s + head_s
print('  按各形状实测效率推算的纯矩阵乘耗时：%.2f s（每层 %.3f s × %d + head %.3f s）'
      % (tot_matmul_s, per_layer_s / nl, nl, head_s))
print('  实测整步：3.585 s')
print('  ⇒ 矩阵乘占整步的 %.0f%%，其余（norm/softmax/激活/Adam/调度）占 %.0f%%'
      % (tot_matmul_s / 3.585 * 100, 100 - tot_matmul_s / 3.585 * 100))
print('\n  对照：4096³ 的 torch.mm 天花板 = 185 GFLOPS')
