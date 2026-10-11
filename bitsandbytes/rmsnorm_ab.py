"""真内核 A/B：同一个符号、两份 DLL（改前 / 改后），交错测量。

用法：python rmsnorm_ab.py <dll路径> [重复次数]
绕开包导入，直接 ctypes.CDLL 指到文件，确保加载的就是我指定的那一份。
正确性用 numpy 独立复算（不是拿另一份 DLL 的结果比对）。
"""
import ctypes
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

DLL = sys.argv[1]
REP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
M, D = 16384, 2048

lib = ctypes.CDLL(DLL)
f = lib.cfused_rmsnorm_fwd_cpu
f.restype = ctypes.c_longlong
P = ctypes.POINTER(ctypes.c_float)
f.argtypes = [P, P, P, P, P, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_float]

rng = np.random.default_rng(1234)
x = rng.random(M * D, dtype=np.float32)
res = rng.random(M * D, dtype=np.float32)
w = rng.random(D, dtype=np.float32)
out = np.zeros(M * D, np.float32)
xs = np.zeros(M, np.float32)

ali = out.ctypes.data % 32 == 0
if not ali:                       # NT 要求 32 字节对齐，不满足就测不到该路径
    out = np.zeros(M * D + 8, np.float32)[8:]
    out = out[:M * D]
    ali = out.ctypes.data % 32 == 0

p = lambda a: a.ctypes.data_as(P)  # noqa: E731
args = (p(x), p(res), p(w), p(out), p(xs), M, D, ctypes.c_float(1e-6))
f(*args)

# ---- 正确性：numpy 独立复算 ----
h = x.reshape(M, D) + res.reshape(M, D)
r = 1.0 / np.sqrt((h * h).mean(axis=1) + 1e-6)
ref = (h * r[:, None]) * w[None, :]
err = float(np.abs(out.reshape(M, D) - ref).max())
xerr = float(np.abs(xs - r).max())

# ---- 计时 ----
best = 1e9
for _ in range(REP):
    t0 = time.perf_counter()
    f(*args)
    best = min(best, time.perf_counter() - t0)

need = 16.0 * M * D          # 必要流量：读x+读res+读w+写out（w 每行重复但有缓存）
print(f'{DLL.split(chr(92))[-1]:<34} {best*1e3:8.3f} ms  '
      f'有效 {need/best/1e9:6.2f} GB/s  对齐={ali}  最大误差={err:.2e} '
      f'xs误差={xerr:.2e}')
