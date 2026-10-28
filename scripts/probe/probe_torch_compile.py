# -*- coding: utf-8 -*-
"""probe_torch_compile.py —— 试 torch.compile（CPU inductor）能不能融合那些小算子。

为什么盯这个：干净测量已经把别的路堵死了 ——
  · GEMM 242 GFLOPS，机器持续峰值 233 GFLOPS，已经到顶
  · batch 扫描每图成本几乎持平（349/339/346/336 ms），没有规模效应
  · 非 GEMM 那 1239 ms 里，按纯带宽算本该只要几十 ms，差额只能是
    每步约 1500 个 elementwise 算子的【调度/建队开销】
  ⇒ 唯一可能有数量级收益的就是「把多个算子融成一个」。

判据（都只看墙钟，min-of-N）：
  ① 编译能不能成功、有没有 graph break
  ② 编译后的整步时间 vs 未编译
  ③ loss 数值是否一致（编译不该改变数学）

风险提示：CPU inductor 用 C++ 代码生成 + 现场编译，首次编译可能几分钟；
模型里有 loop_idx / loop_mods / keys 这些数据相关的控制流，很可能 graph break。
所以先跑一次"能不能编过"，再谈快慢。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

CK = os.environ.get('CKPT', r'D:\work\bitsandbytes-CPU\.autopilot\v4_bos\ckpt_13000.pt')
B = int(os.environ.get('B', '8'))
TH = int(os.environ.get('TH', '6'))
MODE = os.environ.get('MODE', 'default')
torch.set_num_threads(TH)

print('=' * 96)
print('torch.compile 试验   torch=%s  线程=%d  B=%d  mode=%s' % (torch.__version__, TH, B, MODE))
print('=' * 96)

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
sd = ck.get('model') or ck.get('state_dict') or ck
T = cfg.total_tokens
torch.manual_seed(1234)
tok = torch.randint(0, cfg.vocab_size, (B, T))


def build():
    m = SmallARImageModelV2(cfg)
    m.load_state_dict(sd)
    m.train()
    return m


def flat(o):
    r = []
    if o is None:
        return r
    if isinstance(o, (tuple, list)):
        for i in o:
            r += flat(i)
        return r
    if torch.is_tensor(o):
        r.append(o)
    return r


def step(m, opt):
    out = m.forward_logits(tok)
    loss = sum(f.float().sum() for f in flat(out))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss


try:
    import bitsandbytes as bnb
    Opt = bnb.optim.AdamW8bit
except Exception:
    Opt = torch.optim.AdamW


def timed(m, opt, n=3, warm=2):
    for _ in range(warm):
        step(m, opt)
    best = 1e30
    for _ in range(n):
        t0 = time.perf_counter()
        step(m, opt)
        best = min(best, time.perf_counter() - t0)
    return best


# ---- 基线 ----
m = build()
opt = Opt(m.parameters(), lr=1e-4)
base = timed(m, opt)
L0 = float(step(m, opt).detach())
print()
print('  未编译           : %8.1f ms/步   loss=%.6f' % (base * 1e3, L0))

# ---- 编译 ----
print()
print('  正在编译 ...（CPU inductor 首次编译可能要几分钟）')
t_c0 = time.perf_counter()
try:
    mc = torch.compile(build(), mode=MODE)
    ok = True
except Exception as ex:
    print('  !! torch.compile 包装失败: %s' % str(ex)[:300])
    ok = False

if ok:
    optc = Opt(mc.parameters(), lr=1e-4)
    try:
        t0 = time.perf_counter()
        Lc = float(step(mc, optc).detach())
        compile_s = time.perf_counter() - t0
        print('  首次执行(含编译): %8.1f ms   loss=%.6f' % (compile_s * 1e3, Lc))
        t = timed(mc, optc)
        print('  已编译           : %8.1f ms/步   加速比 %.3fx' % (t * 1e3, base / t))
        print()
        print('  loss 相对差 %.3e   %s'
              % (abs(Lc - L0) / max(abs(L0), 1e-30),
                 'OK' if abs(Lc - L0) / max(abs(L0), 1e-30) < 1e-4 else '**不一致**'))
        print()
        print('  编译总耗时 %.1f 秒' % (time.perf_counter() - t_c0))
        if base / t > 1.05:
            print('  ⇒ 值得用：1000 步从 %.1f 分钟降到 %.1f 分钟'
                  % (base * 1000 / 60, t * 1000 / 60))
        else:
            print('  ⇒ 收益不足 5%%，不值得引入编译复杂度')
    except Exception as ex:
        print('  !! 编译后执行失败: %s' % str(ex)[:500])
        import traceback
        traceback.print_exc()
print()
print('DONE')
