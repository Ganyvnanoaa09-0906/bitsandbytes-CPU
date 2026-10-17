"""Is the 4-bit optimizer a tool? A real A/B on the real model, not a microbenchmark.

Its own test suite already shows the kernel matches a reference bit for bit, that state memory is
1.031 bytes per parameter against 8.000 for fp32 Adam, and that a small model trains to a comparable
loss on both. What that does not show is the thing that decides whether it is usable: does it train
THIS model, on THIS data, to the same place, and what does it cost in wall clock.

So: the 30.5M SAR model, real tokens, identical initialisation and identical data order, one run per
optimizer, loss recorded every 25 steps. A drop-in replacement has to track the reference closely --
not exactly, since 4-bit state is quantised, but closely enough that the curves are the same shape and
end near the same value.

Three numbers decide it:

    loss trajectory   do the two curves track, and do they end within noise of each other
    state bytes       the actual torch storage held by each optimizer
    step time         the cost, which the module's own docstring says is about 4x slower for the
                      optimizer because the kernel has no AVX2 path

And the honest framing, since this is the "tool not a toy" question: the 4-bit optimizer trades speed
for memory. The docstring states it in the first section, which is what makes it a tool -- a toy would
advertise the 7.8x memory saving and leave the reader to discover the 4x kernel cost.

Decision rule, fixed in advance: if the final losses differ by less than 1% and the step-time cost is
under 10%, it is a tool for memory-constrained runs. If the loss diverges or the cost exceeds 10%, it
is not yet one and the AVX2 kernel has to come first.
"""
from __future__ import annotations

import importlib.util
import statistics
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


# ★ ORDER MATTERS, and the reason is not obvious. Load AdamW4bit -- and therefore its DLL --
#   before ANYTHING that might import bitsandbytes. Windows' LoadLibrary dedupes by module
#   basename: once `libbitsandbytes_cpu.dll` is in the process under that name, every later
#   CDLL call with the same basename returns the already-loaded module even from a different
#   path. The installed bitsandbytes (0.50.2.dev0) is an older build that does not export
#   coptimizer_update_4bit_blockwise_cpu, so if it gets loaded first, the symbol is simply not
#   there and adamw4bit.py raises ImportError. adamw4bit.py's own docstring documents this; my
#   first version of this script imported measure_guard first and hit exactly it.
#   Also: there are TWO copies of adamw4bit.py -- one at the repo root and one inside the package
#   at bitsandbytes/bitsandbytes/optim/. _find_dll's relative candidates (here/bitsandbytes/
#   bitsandbytes/...) only resolve for the ROOT copy, because here is that file's directory.
#   Loading the in-package copy puts here one level too deep and every candidate misses. The
#   passing test suite imports the root copy, which is why this was not obvious.
_spec = importlib.util.spec_from_file_location(
    'adamw4bit', r'D:\work\bnb-4bitopt\adamw4bit.py')
adamw4bit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adamw4bit)
AdamW4bit = adamw4bit.AdamW4bit
print('  AdamW4bit 从 worktree 加载: %s' % adamw4bit._dll_path)

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
from measure_guard import assert_idle, stamp  # noqa: E402

import bitsandbytes as bnb  # noqa: E402
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
STEPS = 300
VAL_EVERY = 25
B = 2
T = 1024


def build():
    torch.manual_seed(1234)
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                             ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                             mask_mode='causal', n_cluster=32, use_bos=False)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                          loop_L_max=3, loop_random=False, loop_cache='shared'))
    m.train()
    return m, cfg


def state_bytes(opt, label):
    tot = 0
    if label == '4bit':
        return opt.state_bytes()
    for st in opt.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                tot += v.numel() * v.element_size()
    return tot


def run(kind, tok):
    model, cfg = build()
    if kind == '8bit':
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=1e-4)
    else:
        opt = AdamW4bit(model.parameters(), lr=1e-4)
    gen = torch.Generator().manual_seed(0)
    losses, times = [], []
    t_start = time.perf_counter()
    for i in range(STEPS):
        x = tok[torch.randint(0, len(tok), (B,), generator=gen)].long()
        t0 = time.perf_counter()
        out = model.forward_logits(x, None)
        lg = out[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, cfg.vocab_size),
                                                 x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        times.append(time.perf_counter() - t0)
        if (i + 1) % VAL_EVERY == 0:
            losses.append((i + 1, float(loss.detach())))
    total = time.perf_counter() - t_start
    return losses, times, state_bytes(opt, kind), total


assert_idle()
stamp('4-bit 优化器：真模型真数据的 A/B')
torch.set_num_threads(6)
tok = torch.load(TOK, map_location='cpu')
print()
print('  模型 %.1fM 参数 · %d 步 · batch %d · T %d' %
      (sum(p.numel() for p in build()[0].parameters()) / 1e6, STEPS, B, T))
print()

res = {}
for kind in ('8bit', '4bit'):
    print('  --- %s ---' % kind)
    losses, times, sb, total = run(kind, tok)
    res[kind] = (losses, times, sb, total)
    print('     总时长 %.1f s · 每步中位 %.4f s' % (total, statistics.median(times)))
    print('     优化器状态 %.1f MB (%.3f 字节/参数)'
          % (sb / 1024 ** 2, sb / sum(p.numel() for p in build()[0].parameters())))
    print('     loss: ' + ' '.join('%.4f' % v for _, v in losses[:6]))
print()
print('  === 对比 ===')
l8 = res['8bit'][0]
l4 = res['4bit'][0]
print('  %-8s %12s %12s %10s' % ('步', '8bit loss', '4bit loss', '相对差'))
for (s1, v1), (s2, v2) in zip(l8, l4):
    print('  %-8d %12.4f %12.4f %9.2f%%' % (s1, v1, v2, (v2 - v1) / v1 * 100))
print()
final8, final4 = l8[-1][1], l4[-1][1]
t8, t4 = statistics.median(res['8bit'][1]), statistics.median(res['4bit'][1])
m8, m4 = res['8bit'][2], res['4bit'][2]
print('  最终 loss:  8bit %.4f · 4bit %.4f ⇒ 相对差 %.2f%%' %
      (final8, final4, (final4 - final8) / final8 * 100))
print('  步时:       8bit %.4f s · 4bit %.4f s ⇒ 【%.3f×】' % (t8, t4, t4 / t8))
print('  状态内存:   8bit %.1f MB · 4bit %.1f MB ⇒ 【省 %.1f×】' %
      (m8 / 1024 ** 2, m4 / 1024 ** 2, m8 / max(m4, 1)))
print()
print('  判据（事先固定）: 最终 loss 差 <1% 且步时代价 <10% ⇒ 是工具')
ok = abs((final4 - final8) / final8) < 0.01 and (t4 / t8 - 1) < 0.10
print('  ⇒ %s' % ('【是工具】✓' if ok else '【还不是】✗'))
