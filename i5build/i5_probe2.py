"""i5-10400 探针：一次把要用的数都量出来。

与 4500U 的对照（全部实测，不看规格表）：
  ① FMA 峰值 + 同形零访存
  ② **bf16/fp16 是否同样退化** —— Zen2 上 0.4~0.6 GFLOPS（慢 100 倍）。
     Intel 若正常，则"bf16 训练"这条路可行（省一半访存）
  ③ **训练形状表 @ 全线程** —— 4500U 只有 71~104 GFLOPS，且是 2 线程测的，
     口径存疑。这一项决定 CPU 还有没有 1.5~2× 空间
  ④ 4096³ 天花板（报告口径 185~200）
"""
import platform
import sys
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

print('=' * 68)
print('环境')
print('=' * 68)
print('  机器:', platform.node(), '|', platform.processor())
print('  系统:', platform.platform())
try:
    import psutil
    vm = psutil.virtual_memory()
    print('  内存: 总 %.1f GB / 可用 %.1f GB' % (vm.total / 2**30, vm.available / 2**30))
    print('  物理核/逻辑核: %s / %s' % (psutil.cpu_count(False), psutil.cpu_count(True)))
except ImportError:
    print('  psutil 没装（不影响）')
try:
    import torch
except ImportError:
    print('  torch 没装 ✗'); sys.exit(0)
n = torch.get_num_threads()
print('  torch %s | 默认线程 %d' % (torch.__version__, n))

print('\n' + '=' * 68)
print('① dtype 矩阵乘 1024³（Zen2: bf16/fp16 慢 100 倍）')
print('=' * 68)
M = 1024
fl = 2.0 * M ** 3
for tag, dt in (('fp32', torch.float32), ('bf16', torch.bfloat16),
                ('fp16', torch.float16)):
    try:
        a = torch.randn(M, M, dtype=dt); b = torch.randn(M, M, dtype=dt)
        c = a @ b
        t0 = time.perf_counter()
        for _ in range(3):
            c = a @ b
        d = (time.perf_counter() - t0) / 3
        print('  %-6s %9.1f ms  %8.1f GFLOPS' % (tag, d * 1000, fl / d / 1e9))
    except Exception as e:
        print('  %-6s 失败 %s' % (tag, str(e)[:50]))

print('\n' + '=' * 68)
print('② 训练形状表 @ 全线程（4500U 对照: 71~104 GFLOPS @ 2 线程）')
print('=' * 68)
dm, dup, V, T, B, nl = 512, 2048, 1024, 256, 8, 10
Mt = B * T
SHAPES = [('attn qkv', Mt, 3 * dm, dm), ('attn out', Mt, dm, dm),
          ('mlp up', Mt, 2 * dup, dm), ('mlp down', Mt, dm, dup),
          ('head', Mt, V, dm), ('attn score', B * 8 * T, T, 64)]
tot = 0.0
for name, m, nn, k in SHAPES:
    a = torch.randn(m, k); b = torch.randn(k, nn)
    c = a @ b
    t0 = time.perf_counter()
    for _ in range(3):
        c = a @ b
    d = (time.perf_counter() - t0) / 3
    g = 2.0 * m * nn * k / d / 1e9
    if name not in ('head', 'attn score'):
        tot += d * nl
    print('  %-11s %5d×%-5d×%-5d %8.1f GFLOPS' % (name, m, nn, k, g))
print('  → 纯矩阵乘推算每步 %.2f s（4500U 实测整步 3.585 s）' % tot)

print('\n' + '=' * 68)
print('③ 4096³ 天花板（报告口径 185~200 GFLOPS）')
print('=' * 68)
M2 = 4096
a = torch.randn(M2, M2); b = torch.randn(M2, M2)
c = a @ b
t0 = time.perf_counter()
c = a @ b
d = time.perf_counter() - t0
print('  4096³: %.3f s  %.1f GFLOPS' % (d, 2.0 * M2 ** 3 / d / 1e9))
print('\n完成。把这些数发回来即可。')
