import os, sys, time, statistics as st
import torch, numpy as np
sys.stdout.reconfigure(encoding='utf-8')


def bench(fn, target_s=0.5, max_reps=50):
    fn(); fn()
    t0 = time.perf_counter(); fn(); t1 = time.perf_counter() - t0
    reps = max(3, min(max_reps, int(target_s / max(t1, 1e-4))))
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def gf(M, K, N, t):
    return 2.0 * M * K * N / t / 1e9


print('=== 后端对比（6 线程，2 轮交替）===')
for (M, K, N) in [(2048, 512, 512), (2048, 512, 2048), (2048, 2048, 512), (4096, 4096, 4096)]:
    a = torch.randn(M, K); b = torch.randn(K, N)
    an, bn = a.numpy(), b.numpy()
    res = {'torch': [], 'nomkldnn': [], 'numpy': []}
    for rnd in range(2):
        torch.set_num_threads(6); torch.backends.mkldnn.enabled = True
        res['torch'].append(bench(lambda: a @ b))
        torch.backends.mkldnn.enabled = False
        res['nomkldnn'].append(bench(lambda: a @ b))
        torch.backends.mkldnn.enabled = True
        res['numpy'].append(bench(lambda: an @ bn))
    print(' %-14s torch %6.1f | no-mkldnn %6.1f | numpy %6.1f' % (
        '%dx%dx%d' % (M, K, N),
        gf(M, K, N, st.median(res['torch'])),
        gf(M, K, N, st.median(res['nomkldnn'])),
        gf(M, K, N, st.median(res['numpy']))))

print('\n=== 线程缩放（2048x512x2048）===')
a = torch.randn(2048, 512); b = torch.randn(512, 2048)
for th in (1, 2, 3, 4, 6):
    torch.set_num_threads(th)
    t = bench(lambda: a @ b)
    g = gf(2048, 512, 2048, t)
    print(' threads=%d: %6.1f GFLOPS (每核 %5.1f)' % (th, g, g / th))

print('\n=== 融合 A/B（交替 2 轮）===')
torch.set_num_threads(6)
a = torch.randn(2048, 512)
ws = [torch.randn(512, 512) for _ in range(3)]
wcat = torch.randn(512, 1536)


def f_split():
    for w in ws:
        a @ w


def f_cat():
    a @ wcat


r1 = {'s': [], 'c': []}
for rnd in range(2):
    r1['s'].append(bench(f_split, 0.4)); r1['c'].append(bench(f_cat, 0.4))
print(' qkv  3x(2048x512x512) %6.2f ms | 1x(2048x512x1536) %6.2f ms | 融合省 %.1f%%' % (
    st.median(r1['s']) * 1e3, st.median(r1['c']) * 1e3,
    (1 - st.median(r1['c']) / st.median(r1['s'])) * 100))

ws2 = [torch.randn(512, 2048) for _ in range(2)]
wcat2 = torch.randn(512, 4096)


def g_split():
    for w in ws2:
        a @ w


def g_cat():
    a @ wcat2


r2 = {'s': [], 'c': []}
for rnd in range(2):
    r2['s'].append(bench(g_split, 0.4)); r2['c'].append(bench(g_cat, 0.4))
print(' mlp  2x(2048x512x2048) %6.2f ms | 1x(2048x512x4096) %6.2f ms | 融合省 %.1f%%' % (
    st.median(r2['s']) * 1e3, st.median(r2['c']) * 1e3,
    (1 - st.median(r2['c']) / st.median(r2['s'])) * 100))

print('\n参考：6 线程零访存 GEMM 形内循环 = 494.5 GFLOPS')
