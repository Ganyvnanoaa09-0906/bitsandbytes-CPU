"""Does --mask-mode msa actually save time at grid=32? Measure, do not reason.

Reading the implementation suggests it does not: the mask is passed explicitly to
scaled_dot_product_attention, which rules out the fused/memory-efficient path, and
the key/value set is T plus the compressed summaries -- i.e. slightly LARGER than
plain causal. If that is right, msa is a modelling feature (local window plus
two compressed scales in one softmax) and not a speed lever, and the honest answer
to "how do we make AR training faster" is "fewer tokens or real sparse attention",
not "turn on msa".

Also measures --head-mode factorized, which the model's own header claims cuts
output computation ~16x. At grid=32 that claim is about the head only; the
question is what fraction of a step the head is.

Reuses cost_32x32's harness: random tokens, forward+backward, batch 8, loop on.
"""
import argparse
import gc
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.stdout.reconfigure(encoding='utf-8')

from small_image_model_v2 import SmallARImageModelV2, SmallImageConfigV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

torch.set_num_threads(6)


def build(grid, mask_mode, head_mode, n_cluster=32):
    cfg = SmallImageConfigV2(grid=grid, total_tokens=grid * grid,
                             mask_mode=mask_mode, head_mode=head_mode,
                             n_cluster=n_cluster, use_bos=True)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end,
                          loop_L=1, loop_L_max=3, loop_random=True, loop_cache='perloop'))
    m.train()
    return m, cfg


def timeit(m, cfg, n, reps=3):
    x = torch.randint(0, 1024, (8, n)).long()
    o = m.forward_logits(x, None)
    lg = o[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1)).backward()
    m.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        o = m.forward_logits(x, None)
        lg = o[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(lg.reshape(-1, cfg.vocab_size), x.reshape(-1))
        loss.backward()
        m.zero_grad(set_to_none=True)
    return (time.perf_counter() - t0) / reps


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


ap = argparse.ArgumentParser()
ap.add_argument('--grid', type=int, default=32)
a = ap.parse_args()
N = a.grid * a.grid

cases = [
    ('causal', 'flat', '基线（现有训练就是这个）'),
    ('msa', 'flat', '多尺度压缩注意力'),
    ('causal', 'factorized', '因子化输出头（号称省 16x）'),
    ('msa', 'factorized', '两个一起'),
]
print('grid=%dx%d = %d token, batch 8, loop L=1..3' % (a.grid, a.grid, N))
print()
print('%-10s %-12s %10s %12s %s' % ('mask', 'head', 's/step', '相对基线', '说明'))
print('-' * 78)
base = None
for mm, hm, note in cases:
    try:
        m, cfg = build(a.grid, mm, hm)
        dt = timeit(m, cfg, N)
        if base is None:
            base = dt
        print('%-10s %-12s %10.2f %11.3fx %s' % (mm, hm, dt, dt / base, note))
        del m
        gc.collect()
    except Exception as e:
        print('%-10s %-12s %10s %12s %s: %s' % (mm, hm, 'FAIL', '-',
                                                type(e).__name__, str(e)[:40]))
print()
print('峰值 RSS %.2f GB' % peak_rss_gb())
