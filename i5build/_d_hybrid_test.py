import os, sys, time, statistics as st
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\i5build')
import torch, torch.nn.functional as F
torch.set_num_threads(6)
from hybrid_lin import OCLGEMM, HybridLinear

ocl = OCLGEMM()
HybridLinear.ocl = ocl
print('device:', ocl.dev.name)


def cpu_step(x0, w0):
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    y = F.linear(x, w)
    (y * y).mean().backward()
    return y.detach(), x.grad, w.grad


def hyb_step(x0, w0):
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    y = HybridLinear.apply(x, w)
    (y * y).mean().backward()
    return y.detach(), x.grad, w.grad


print('\n=== 正确性 ===')
x = torch.randn(2048, 512); w = torch.randn(2048, 512)
y1, gx1, gw1 = cpu_step(x, w)
y2, gx2, gw2 = hyb_step(x, w)
for name, a, b in (('y', y1, y2), ('dx', gx1, gx2), ('dw', gw1, gw2)):
    rel = (a - b).abs().max().item() / (a.abs().max().item() + 1e-12)
    print('  %-3s 相对误差 %.2e  %s' % (name, rel, 'OK' if rel < 1e-4 else 'FAIL'))

print('\n=== 速度：单个 Linear 的 fwd+bwd（交替 5 轮取中位）===')
SHAPES = [(2048, 512, 2048), (2048, 2048, 512), (2048, 512, 512), (2048, 512, 4096), (512, 2048, 2048)]
print('%-20s %10s %10s %8s %9s' % ('M x K x N', 'CPU ms', '核显 ms', '加速', '梯度误差'))
for (M, K, N) in SHAPES:
    x = torch.randn(M, K); w = torch.randn(N, K)
    cpu_step(x, w); hyb_step(x, w)
    _, ca, cb = cpu_step(x, w)
    _, ha, hb = hyb_step(x, w)
    err = max((ca - ha).abs().max().item() / (ca.abs().max().item() + 1e-12),
              (cb - hb).abs().max().item() / (cb.abs().max().item() + 1e-12))
    tc, th = [], []
    for r in range(5):
        t0 = time.perf_counter(); cpu_step(x, w); tc.append((time.perf_counter() - t0) * 1e3)
        t0 = time.perf_counter(); hyb_step(x, w); th.append((time.perf_counter() - t0) * 1e3)
    mc, mh = st.median(tc), st.median(th)
    print('%-20s %10.2f %10.2f %7.2fx %8.1e' % ('%dx%dx%d' % (M, K, N), mc, mh, mc / mh, err))

print('\n内核调用统计:', ocl.stats)
