# -*- coding: utf-8 -*-
"""profile_ar_gemm.py — AR 模型单步的算子分解 + 每个算子的 GEMM 效率

回答：
  1) 单步时间到底花在哪些算子上？
  2) 这些算子的 GEMM 效率是多少（对比 R5 峰值 672 GFLOPS）？
  3) batch 从 8 到 128，单步时间和吞吐怎么变？
  4) mkldnn 开关对真实模型的影响？
"""
import os, sys
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.abspath(__file__))
while _root and not _os.path.isdir(_os.path.join(_root, "bitsandbytes")):
    _parent = _os.path.dirname(_root)
    if _parent == _root:
        break
    _root = _parent
if _root not in _sys.path:
    _sys.path.insert(0, _root)
_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.lib.small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

PEAK = 672.0
T = 256   # 每图 token 数


def build():
    cfg = SmallImageConfigV2(
        vocab_size=1024, total_tokens=T, grid=16,
        d_model=512, n_layer=10, n_head=8, n_kv_head=4,
        ffn_dim=2048, head_mode='flat', cond_mode='none',
        pos_mode='2d', token_order='raster',
    )
    return SmallARImageModelV2(cfg)


def make_batch(b):
    x = torch.randint(0, 1024, (b, T))
    y = torch.randint(0, 1024, (b, T))
    return x, y


def step(m, opt, x, y):
    import torch.nn.functional as F
    # forward_logits 返回 (think_head.logits(...), gen_head.logits(x))
    # 而每个 head 的 logits() 自身又返回 (tensor, aux) —— 所以是两层元组
    out = m.forward_logits(x)
    gen = out[1]
    logits = gen[0] if isinstance(gen, tuple) else gen
    if logits.dim() == 3 and logits.shape[1] != y.shape[1]:
        logits = logits[:, :y.shape[1]]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return float(loss)


def timed(m, opt, x, y, iters=5, warm=2):
    for _ in range(warm):
        step(m, opt, x, y)
    t0 = time.perf_counter()
    for _ in range(iters):
        step(m, opt, x, y)
    return (time.perf_counter() - t0) / iters


print('=' * 68, flush=True)
print('torch', torch.__version__, flush=True)

print()
print('--- batch 扫描（threads=6, mkldnn 开）---', flush=True)
torch.set_num_threads(6)
results = {}
for b in (4, 8, 16, 32, 64, 128):
    try:
        m = build(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        x, y = make_batch(b)
        dt = timed(m, opt, x, y, iters=4, warm=2)
        tok_s = b * T / dt
        results[b] = dt
        print('  b=%-4d  %7.3f s/步   %8.1f tok/s' % (b, dt, tok_s), flush=True)
        del m, opt, x, y
    except Exception as e:
        print('  b=%-4d 失败 %s' % (b, str(e)[:70]), flush=True)

print()
print('--- mkldnn 开关（b=32）---', flush=True)
for flag in (True, False):
    torch.backends.mkldnn.enabled = flag
    m = build(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x, y = make_batch(32)
    dt = timed(m, opt, x, y, iters=4, warm=2)
    print('  mkldnn=%-5s %7.3f s/步   %8.1f tok/s' % (flag, dt, 32 * T / dt), flush=True)
    del m, opt, x, y
torch.backends.mkldnn.enabled = True

print()
print('--- 线程数扫描（b=32）---', flush=True)
for nth in (4, 5, 6):
    torch.set_num_threads(nth)
    m = build(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    x, y = make_batch(32)
    dt = timed(m, opt, x, y, iters=4, warm=2)
    print('  threads=%d  %7.3f s/步   %8.1f tok/s' % (nth, dt, 32 * T / dt), flush=True)
    del m, opt, x, y

print()
print('--- 算子分解（b=32, threads=6）---', flush=True)
if os.environ.get('WITH_PROFILER') != '1':
    print('  (跳过 profiler；torch 2.4.1 的 profiler 在本机会段错误)')
    print('  设 WITH_PROFILER=1 可单独尝试')
else:
    try:
        from torch.profiler import profile, ProfilerActivity
        torch.set_num_threads(6)
        m = build(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        x, y = make_batch(32)
        step(m, opt, x, y); step(m, opt, x, y)
        N = 3
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(N):
                step(m, opt, x, y)
        rows = [(e.key, e.self_device_time_total) for e in prof.key_averages()
                if e.self_device_time_total > 0]
        rows.sort(key=lambda r: -r[1])
        total = sum(r[1] for r in rows) or 1
        print('  算子累计 %.1f ms/步' % (total / N / 1e3), flush=True)
        for k, v in rows[:22]:
            print('  %-44s %9.2f ms %6.1f%%' % (k[:44], v / N / 1e3, 100 * v / total), flush=True)
        mm = sum(v for k, v in rows
                 if any(t in k for t in ('mm', 'addmm', 'bmm', 'matmul', 'linear')))
        print('  矩阵乘类合计 %.1f%% (%.1f ms/步)' % (100 * mm / total, mm / N / 1e3), flush=True)
    except Exception:
        import traceback

print()
print('DONE', flush=True)
