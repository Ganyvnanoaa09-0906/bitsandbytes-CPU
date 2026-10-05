"""rmsnorm_fwd 的回归测试（2026-10-05 起）。

为什么需要：该内核此前【没有任何测试覆盖】——`pytest -k rmsnorm` 会把全部用例
deselect。而它在 2026-10-05 被改过一次 1.5× 的性能改动（去掉"h=x+res 暂存进 out"
的老写法，改成两趟 + NT 写出），这种改动没有回归网是不行的。

测试直接对 DLL 的导出符号，不经过 Python 包装层 —— 因为 `cfused_rmsnorm_fwd_cpu`
在 Python 侧没有调用方（真正被用的是 `cfused_add_scale_rmsnorm_cpu`，它内部调它）。
两者都测。

参考实现用 numpy 独立复算，**不是**拿另一份 DLL 的结果比对。

覆盖的边界：
  · 有/无 res（残差）
  · 有/无 weight
  · D 不是 8 的倍数（标量尾段路径）
  · 小张量（低于 NT 阈值 → 走普通 store）
  · 大张量（高于 NT 阈值 → 走 NT store + sfence），并断言 out 32 字节对齐
  · xs 输出（保存的 1/rms）
"""
import ctypes
import os
import sys

import numpy as np
import pytest

sys.stdout.reconfigure(encoding='utf-8')

HERE = os.path.dirname(os.path.abspath(__file__))
DLL_CANDIDATES = [
    os.path.join(HERE, '..', 'bitsandbytes', 'libbitsandbytes_cpu.dll'),
    os.path.join(HERE, '..', 'bitsandbytes', 'libbitsandbytes_cpu.so'),
]


def _load_lib():
    for p in DLL_CANDIDATES:
        if os.path.isfile(p):
            return ctypes.CDLL(os.path.abspath(p))
    pytest.skip('本机没有 libbitsandbytes_cpu 扩展，跳过')


_P = ctypes.POINTER(ctypes.c_float)
_LL = ctypes.c_longlong


def _bind(lib):
    if not hasattr(lib, 'cfused_rmsnorm_fwd_cpu'):
        pytest.skip('DLL 里没有 cfused_rmsnorm_fwd_cpu')
    lib.cfused_rmsnorm_fwd_cpu.argtypes = [_P, _P, _P, _P, _P, _LL, _LL, ctypes.c_float]
    lib.cfused_rmsnorm_fwd_cpu.restype = _LL
    if hasattr(lib, 'cfused_add_scale_rmsnorm_cpu'):
        lib.cfused_add_scale_rmsnorm_cpu.argtypes = [_P, _P, _P, _P, _P, _P,
                                                     _LL, _LL, ctypes.c_float]
        lib.cfused_add_scale_rmsnorm_cpu.restype = _LL
    return lib


def _p(a):
    return a.ctypes.data_as(_P)


def _ref(x, res, w, eps):
    """numpy 独立复算。用 float64 累加，与内核的 double acc 对应。"""
    h = x.astype(np.float64)
    if res is not None:
        h = h + res.astype(np.float64)
    r = 1.0 / np.sqrt((h * h).mean(axis=1) + eps)
    y = h * r[:, None]
    if w is not None:
        y = y * w[None, :].astype(np.float64)
    return y.astype(np.float32), r.astype(np.float32)


CASES = [
    # (M, D, 有res, 有weight, 说明)
    (4, 64, False, False, '最小、无 res/weight、D 是 8 的倍数'),
    (4, 64, True, True, '最小、有 res/weight'),
    (7, 61, False, False, 'D 不是 8 的倍数（标量尾段）'),
    (7, 61, True, True, 'D 不是 8 的倍数 + res/weight'),
    (33, 2048, True, True, '常规训练形状（小张量，低于 NT 阈值）'),
    (16384, 2048, True, True, '大张量（高于 NT 阈值 ⇒ NT store 路径）'),
]


@pytest.mark.parametrize('M,D,with_res,with_w,desc', CASES)
def test_rmsnorm_fwd_matches_numpy(M, D, with_res, with_w, desc):
    lib = _bind(_load_lib())
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((M, D), dtype=np.float32)
    res = rng.standard_normal((M, D), dtype=np.float32) if with_res else None
    w = rng.standard_normal(D, dtype=np.float32) if with_w else None
    out = np.zeros((M, D), dtype=np.float32)
    xs = np.zeros(M, dtype=np.float32)
    eps = 1e-6

    n = lib.cfused_rmsnorm_fwd_cpu(
        _p(x), _p(res) if res is not None else None,
        _p(w) if w is not None else None, _p(out), _p(xs), M, D, ctypes.c_float(eps))
    assert n == M, f'{desc}: 返回值应为 M'

    ref, ref_r = _ref(x, res, w, eps)
    np.testing.assert_allclose(out, ref, rtol=2e-5, atol=2e-6,
                               err_msg=f'{desc}: 输出与 numpy 不符')
    np.testing.assert_allclose(xs, ref_r, rtol=2e-6, atol=1e-7,
                               err_msg=f'{desc}: xs (1/rms) 与 numpy 不符')


def test_rmsnorm_fwd_handles_zero_row():
    """全零行：rms 只由 eps 决定，不能让 NaN/Inf 漏出去。"""
    lib = _bind(_load_lib())
    M, D = 3, 128
    x = np.zeros((M, D), dtype=np.float32)
    w = np.ones(D, dtype=np.float32)
    out = np.zeros((M, D), dtype=np.float32)
    xs = np.zeros(M, dtype=np.float32)
    lib.cfused_rmsnorm_fwd_cpu(_p(x), None, _p(w), _p(out), _p(xs), M, D,
                              ctypes.c_float(1e-6))
    assert np.isfinite(out).all(), '全零行产生了 NaN/Inf'
    np.testing.assert_allclose(out, np.zeros((M, D), np.float32), atol=1e-6)


def test_rmsnorm_fwd_nt_alignment_requirement():
    """NT store 要求 out 32 字节对齐；对齐时必须与不对齐时给出相同结果。

    这是那次 1.5× 改动的关键路径：不对齐时会静默回落到普通 store，
    两条路径必须数值一致。
    """
    lib = _bind(_load_lib())
    M, D = 16384, 2048          # 128 MB > NT 阈值（L3/2 ≈ 2 MB）
    rng = np.random.default_rng(7)
    x = rng.standard_normal((M, D), dtype=np.float32)
    w = rng.standard_normal(D, dtype=np.float32)

    buf = np.zeros(M * D + 16, dtype=np.float32)
    base = buf.ctypes.data
    off = (32 - (base % 32)) // 4 % 8          # 使切片 32 字节对齐
    # ⚠️ 两份输出必须用【各自独立的缓冲区】：第一版把它们放在同一缓冲区的相邻
    # 偏移上（buf[off:] 与 buf[off+1:]），两段互相重叠，第二次写入覆盖了第一次的
    # 结果，比对的其实是"错位一格"的混合数据 —— 测试失败，内核无辜。
    aligned = buf[off:off + M * D].reshape(M, D)
    buf2 = np.zeros(M * D + 16, dtype=np.float32)
    off2 = (32 - (buf2.ctypes.data % 32)) // 4 % 8
    misaligned = buf2[off2 + 1:off2 + 1 + M * D].reshape(M, D)
    assert aligned.ctypes.data % 32 == 0
    assert misaligned.ctypes.data % 32 != 0

    xs = np.zeros(M, dtype=np.float32)
    lib.cfused_rmsnorm_fwd_cpu(_p(x), None, _p(w), _p(aligned), _p(xs), M, D,
                              ctypes.c_float(1e-6))
    lib.cfused_rmsnorm_fwd_cpu(_p(x), None, _p(w), _p(misaligned), _p(xs), M, D,
                              ctypes.c_float(1e-6))
    np.testing.assert_array_equal(aligned, misaligned,
                                  err_msg='NT 路径与普通 store 路径结果不一致')


def test_add_scale_rmsnorm_fused_matches_unfused():
    """融合链 cfused_add_scale_rmsnorm_cpu 必须与"先 add_scale 再 rmsnorm"等价。

    这是 Python 侧真正使用的入口（fused_cpu.py），也是这次改动的影响面。
    """
    lib = _load_lib()
    if not hasattr(lib, 'cfused_add_scale_rmsnorm_cpu'):
        pytest.skip('DLL 里没有 cfused_add_scale_rmsnorm_cpu')
    _bind(lib)
    lib.cfused_add_scale_cpu.argtypes = [_P, _P, _P, _P, _LL, _LL]
    lib.cfused_add_scale_cpu.restype = _LL

    M, D = 257, 512
    rng = np.random.default_rng(99)
    x = rng.standard_normal((M, D), dtype=np.float32)
    y = rng.standard_normal((M, D), dtype=np.float32)
    scale = rng.standard_normal(D, dtype=np.float32)
    w = rng.standard_normal(D, dtype=np.float32)
    eps = 1e-6

    fused = np.zeros((M, D), dtype=np.float32)
    xs = np.zeros(M, dtype=np.float32)
    # ⚠️ 融合版【原地改写 x】（内部走 fused_add_scale_inplace_cpu）⇒ 必须给它一份
    # 副本，否则后面那条对照路径拿到的是被改过的输入 —— 第一版就是这么挂的。
    x_fused = x.copy()
    lib.cfused_add_scale_rmsnorm_cpu(_p(x_fused), _p(y), _p(scale), _p(w), _p(fused),
                                     _p(xs), M, D, ctypes.c_float(eps))

    h = np.zeros((M, D), dtype=np.float32)
    lib.cfused_add_scale_cpu(_p(x.copy()), _p(y), _p(scale), _p(h), M, D)
    separate = np.zeros((M, D), dtype=np.float32)
    xs2 = np.zeros(M, dtype=np.float32)
    lib.cfused_rmsnorm_fwd_cpu(_p(h), None, _p(w), _p(separate), _p(xs2), M, D,
                              ctypes.c_float(eps))

    np.testing.assert_allclose(fused, separate, rtol=1e-6, atol=1e-7,
                               err_msg='融合链与分步结果不一致')
    np.testing.assert_allclose(xs, xs2, rtol=1e-6, atol=1e-7,
                               err_msg='融合链的 xs 与分步结果不一致')
    # 顺带固定住"融合版会原地改 x"这一行为，避免调用方误以为 x 未被改动
    assert not np.array_equal(x_fused, x), '融合版应当原地改写 x（本测试依赖此行为）'
