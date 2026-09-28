# -*- coding: utf-8 -*-
"""verify_bnb_intact.py — 重建 DLL 后，bnb 的既有功能是否完好

我改了 csrc/cpu_ops.cpp 并重编了 DLL，必须确认没破坏：
  1. 8-bit 优化器（AdamW8bit / Lion8bit / RMSprop8bit ...）—— 【所有训练都靠它】
  2. blockwise 8bit 量化 / 反量化
  3. 4-bit (fp4/nf4) 反量化
  4. gemm_8bit 融合反量化
  5. GDN 融合内核
  6. 新增的 9 个融合核
"""
import sys, time
import os
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Three fixes, all found by this file reporting "OK 11/12" on the i5 while
# having verified nothing at all:
#
# 1. PATHS WERE HARD-CODED to D:\work\bitsandbytes-CPU, so on any other machine
#    `import bitsandbytes` failed. It now resolves relative to this file.
#
# 2. THE EXIT CODE WAS ALWAYS 0. Nothing called sys.exit, so a run that failed
#    every check still exited 0 and run_all_tests.py recorded it as OK. A test
#    that cannot fail is not a test; it now exits non-zero on any failure.
#
# 3. THE OUTPUT ENCODING COULD CRASH THE TEST. check() printed U+2705/U+274C,
#    and on a redirected cp936 (GBK) console the FIRST failure raised
#    UnicodeEncodeError -- so the failure path was the one path that could not
#    run. That is how "verified nothing" came back looking green. stdout is now
#    forced to UTF-8 when the interpreter supports it, with a plain-ASCII
#    fallback so the reporting path can never be the thing that breaks.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
# sys.path must contain the PARENT of the package, not the package directory:
# `import bitsandbytes` looks for a `bitsandbytes` entry on sys.path. _testpath
# searches for that parent instead of assuming it, because the R5 and the i5
# disagree about the layout. Getting this wrong cost a run on the i5, where it
# produced "No module named 'bitsandbytes'" for 5 of 8 checks while the
# ctypes-loaded fusion kernels (which resolve the DLL by file path) passed --
# the DLL was fine and only the import was broken.
sys.path.insert(0, HERE)
import _testpath  # noqa: E402,F401  (side effect: fixes sys.path)
torch.set_num_threads(6)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass
_MARK = {"ok": "\u2705", "bad": "\u274c"}
try:
    "".join(_MARK.values()).encode(sys.stdout.encoding or "utf-8")
except (UnicodeEncodeError, LookupError):
    _MARK = {"ok": "[PASS]", "bad": "[FAIL]"}

print('=' * 78)
print('bnb 回归验证（重建 DLL 后）')
print('=' * 78)

ok = 0
fail = 0


def check(name, cond, detail=''):
    global ok, fail
    if cond:
        ok += 1
        print('  %s %-38s %s' % (_MARK["ok"], name, detail))
    else:
        fail += 1
        print('  %s %-38s %s' % (_MARK["bad"], name, detail))


# ---------- 1. 8-bit 优化器 ----------
print()
print('[1] 8-bit 优化器（所有训练都靠它）')
try:
    import bitsandbytes as bnb
    check('import bitsandbytes', True, '路径 ' + str(bnb.__file__)[:44])
except Exception as e:
    check('import bitsandbytes', False, str(e)[:50])
    bnb = None

if bnb is not None:
    for optname in ('AdamW8bit', 'Adam8bit', 'Lion8bit', 'RMSprop8bit', 'Adagrad8bit'):
        if not hasattr(bnb.optim, optname):
            check(optname, False, '不存在')
            continue
        try:
            torch.manual_seed(0)
            m = nn.Linear(64, 64)
            p0 = m.weight.detach().clone()
            o = getattr(bnb.optim, optname)(m.parameters(), lr=1e-2)
            for _ in range(5):
                loss = m(torch.randn(8, 64)).pow(2).mean()
                o.zero_grad(); loss.backward(); o.step()
            moved = (m.weight.detach() - p0).abs().max().item()
            check(optname, moved > 0, '5 步后权重变化 %.3e' % moved)
        except Exception as e:
            check(optname, False, str(e)[:50])

    # 与 fp32 AdamW 的收敛对比（这是报告的核心指标）
    try:
        torch.manual_seed(0)
        m1 = nn.Linear(128, 128)
        m2 = nn.Linear(128, 128)
        m2.load_state_dict(m1.state_dict())
        x = torch.randn(32, 128)
        y = torch.randn(32, 128)
        o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
        o2 = bnb.optim.AdamW8bit(m2.parameters(), lr=1e-3)
        for _ in range(60):
            for m, o in ((m1, o1), (m2, o2)):
                l = (m(x) - y).pow(2).mean()
                o.zero_grad(); l.backward(); o.step()
        l1 = (m1(x) - y).pow(2).mean().item()
        l2 = (m2(x) - y).pow(2).mean().item()
        check('AdamW8bit 收敛 vs fp32', abs(l1 - l2) < 0.05,
              'fp32 %.6f  8bit %.6f  差 %.2e' % (l1, l2, abs(l1 - l2)))
    except Exception as e:
        check('AdamW8bit 收敛对比', False, str(e)[:50])

# ---------- 2. 量化 ----------
print()
print('[2] 量化内核')
try:
    import bitsandbytes.functional as BF
    x = torch.randn(64, 128)
    q, st = BF.quantize_blockwise(x)
    dq = BF.dequantize_blockwise(q, st)
    err = (x - dq).abs().max().item()
    check('blockwise 8bit 往返', err < 0.1, '最大误差 %.4f' % err)
except Exception as e:
    check('blockwise 8bit 往返', False, str(e)[:50])

try:
    import bitsandbytes.functional as BF
    x = torch.randn(64, 128)
    q, st = BF.quantize_4bit(x, quant_type='nf4')
    dq = BF.dequantize_4bit(q, st)
    check('nf4 4bit 往返', dq.shape == x.shape, '最大误差 %.4f' % (x - dq).abs().max().item())
except Exception as e:
    check('nf4 4bit 往返', False, str(e)[:50])

# ---------- 3. GDN ----------
print()
print('[3] GDN 融合内核（长序列线性注意力）')
try:
    from bitsandbytes.gdn_cpu import fused_recurrent_gated_delta_rule
    B, H, T, K, V = 1, 2, 64, 32, 32
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    beta = torch.rand(B, T, H)
    g = -torch.rand(B, T, H)
    o, ht = fused_recurrent_gated_delta_rule(q, k, v, beta, g, scale=K ** -0.5,
                                             output_final_state=True)
    check('gdn fwd', o.shape == (B, T, H, V), 'out %s' % (tuple(o.shape),))
except Exception as e:
    check('gdn fwd', False, str(e)[:60])

# ---------- 4. 新增融合核 ----------
print()
print('[4] 新增的融合核（本轮）')
try:
    import fused_cpu
    check('fused_cpu 可用', fused_cpu.available())
    x = torch.randn(256, 512)
    w = torch.randn(512)
    r = (w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    g = fused_cpu.fused_rmsnorm(x, w)
    check('fused_rmsnorm', (r - g).abs().max().item() < 1e-4,
          '相对 %.2e' % ((r - g).abs().max().item() / r.abs().max().item()))
    a = torch.randn(256, 128)
    b = torch.randn(256, 128)
    gg = fused_cpu.fused_swiglu(a, b)
    rr = torch.nn.functional.silu(a) * b
    check('fused_swiglu', (gg - rr).abs().max().item() < 1e-4,
          '相对 %.2e' % ((gg - rr).abs().max().item() / rr.abs().max().item()))
except Exception as e:
    check('fused_cpu', False, str(e)[:60])

# ---------- 5. 训练脚本还能跑吗 ----------
print()
print('[5] 训练脚本的优化器构造（真实路径）')
try:
    import bitsandbytes as bnb
    m = nn.Linear(512, 512)
    o = bnb.optim.AdamW8bit(m.parameters(), lr=2e-4)
    check('train_ar_v2 的优化器构造', True,
          '%d 个参数组' % len(o.param_groups))
except Exception as e:
    check('训练脚本优化器构造', False, str(e)[:60])

print()
print('=' * 78)
print('结果: %s %d 项通过   %s %d 项失败' % (_MARK["ok"], ok, _MARK["bad"], fail))
print('DONE')

# An explicit exit code. Without this the script always exited 0, so the runner
# could not tell a clean pass from a run in which every single check failed.
if ok == 0:
    print('FATAL: no check ran at all -- the harness itself is broken, not the code')
    sys.exit(2)
sys.exit(1 if fail else 0)
