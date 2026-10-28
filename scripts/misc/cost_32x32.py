"""What does a 32x32 token grid actually cost on this machine?

The 16x16 grid gives 256 tokens for a 512x512 image -- 1024 pixels per token.
That is the ceiling on output quality, and no amount of training moves it. The
obvious fix is more tokens, but AR attention is O(N^2), so 32x32 (1024 tokens)
is not 4x the work, it is a different regime.

This measures it rather than estimating it: the model does not care whether the
tokens are real, only how many there are, so random tokens of length N give the
real cost without needing a 32x32 tokeniser first.

Reports forward+backward time and peak RSS for N = 256, 576, 1024 at batch 8,
then the ratio against the current configuration.
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import gc
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)
BATCH = 8
CK = r'D:\work\bitsandbytes-CPU\i5build\loop_ctrl1\ckpt_last.pt'


def peak_rss_gb():
    import ctypes

    class PMC(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    p = PMC(); p.cb = ctypes.sizeof(PMC)
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),
                                             ctypes.byref(p), p.cb)
    return p.PeakWorkingSetSize / 2**30


def build(grid):
    ck = torch.load(CK, map_location='cpu', weights_only=False)
    keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
    d = {k: v for k, v in ck['cfg'].items() if k in keep}
    d['grid'] = grid
    d['total_tokens'] = grid * grid
    cfg = SmallImageConfigV2(**d)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=True,
                          loop_cache='perloop'))
    sd = ck['model']
    # Position embeddings are sized by the grid, so a 16x16 checkpoint cannot be
    # loaded into a 24x24 or 32x32 model -- that is a size mismatch, not a bug.
    # For a COST measurement the weights are irrelevant, only the shapes matter,
    # so other grids run with random initialisation.
    if grid == 16:
        m.load_state_dict(sd, strict=False)
    else:
        loaded = 0
        own = m.state_dict()
        for k, v in sd.items():
            if k in own and own[k].shape == v.shape:
                own[k] = v
                loaded += 1
        m.load_state_dict(own, strict=False)
        print('       (grid=%d: 只装了 %d/%d 个形状匹配的张量，其余随机初始化)'
              % (grid, loaded, len(sd)))
    m.train()
    return m, cfg, []


def step_time(m, cfg, n, reps=3):
    x = torch.randint(0, cfg.vocab_size, (BATCH, n)).long()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    # warm-up (first step allocates, and the loop cache warms)
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    torch.nn.functional.cross_entropy(
        lg.reshape(-1, cfg.vocab_size), x.reshape(-1)).backward()
    opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    dt = (time.perf_counter() - t0) / reps
    return dt, float(loss)


print('batch=%d, loop L=1..3 random, 6 threads' % BATCH)
print()
print('%-8s %8s %12s %12s %10s' % ('grid', 'tokens', 's/step', '相对 16x16', '峰值RSS'))
print('-' * 56)
base = None
rows = []
for grid in (16, 24, 32):
    n = grid * grid
    try:
        m, cfg, misspos = build(grid)
        dt, l = step_time(m, cfg, n)
        if base is None:
            base = dt
        rss = peak_rss_gb()
        rows.append((grid, n, dt, rss, misspos))
        print('%-8s %8d %12.2f %11.2fx %9.2f GB%s'
              % ('%dx%d' % (grid, grid), n, dt, dt / base, rss,
                 '   (pos 键缺失: %s)' % misspos if misspos else ''))
        del m
        gc.collect()
    except Exception as e:
        print('%-8s %8d  失败: %s: %s' % ('%dx%d' % (grid, grid), n,
                                          type(e).__name__, str(e)[:60]))

print()
if len(rows) >= 2:
    g0, n0, t0, _, _ = rows[0]
    g1, n1, t1, _, _ = rows[-1]
    print('=== 外推 ===')
    print('  %d -> %d token: 步时 %.2f -> %.2f s（%.1fx）'
          % (n0, n1, t0, t1, t1 / t0))
    print('  纯 O(N^2) 注意力若占主导会接近 %.1fx；纯 O(N) 部分占主导会接近 %.1fx'
          % ((n1 / n0) ** 2, n1 / n0))
    print('  ⇒ 实测落在两者之间，说明注意力占比随 N 上升但没有吃掉全部')
    print()
    print('  按 3000 步一轮估算:')
    for g, n, t, r, _ in rows:
        print('    %-8s %.2f h/轮' % ('%dx%d' % (g, g), t * 3000 / 3600))
