"""Try torch.compile on the model: the cheapest thing to borrow from the CUDA world.

CUDA's biggest structural win is fusion -- not launching a kernel per elementwise op. On CPU the
equivalent is Inductor, which fuses elementwise chains and cuts dispatch overhead. We have a
concrete target: 490 aten::mul calls per step, 11.6% of the operator profile, all from RMSNorm
expansion, plus the surrounding dispatch.

Why this is worth doing before the more elaborate ideas: it is one line, and the profile says the
non-GEMM cost is about 19% elementwise. If fusion recovers even a third of that it beats most of
what is left on the queue.

Two traps, both of which the earlier failures in this project taught:

  * compile time on CPU is minutes, while our step is about 4.5 s. So the comparison must be
    steady-state, with warmup excluded, or the compiler cost swamps the result. This measures
    uncompiled, compiles, then measures compiled after warmup, and reports all three.
  * Inductor may silently fall back on unsupported ops. It reports how many graphs it compiled and
    how many it skipped, so both are printed.

Decision rule, fixed in advance per the 7.39.4 discipline: steady-state step time must drop by
more than 5% to keep it. Below that it is not worth the compile cost on every run.
"""
from __future__ import annotations

import statistics
import sys
import time

import torch

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass   # ★ IDLE 的 stdout 是 StdOutputFile,没有 reconfigure
           #   （正常终端里是 io.TextIOWrapper,有这个方法）


from measure_guard import assert_idle, stamp  # noqa: E402
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2  # noqa: E402
from loop_transformer import LoopConfig  # noqa: E402

assert_idle()
stamp('torch.compile 对照')
torch.set_num_threads(6)

TOK = r'D:\work\cloud_salvage\ScPeP7\tokens_32x32_full.pt'
T, B = 1024, 2
tok = torch.load(TOK, map_location='cpu')


def build():
    cfg = SmallImageConfigV2(vocab_size=1024, d_model=512, n_layer=10, n_head=8, n_kv_head=4,
                             ffn_dim=2048, total_tokens=T, grid=32, head_mode='flat',
                             mask_mode='causal', n_cluster=32, use_bos=False)
    m = SmallARImageModelV2(cfg)
    m.set_loop(LoopConfig(loop_start=cfg.loop_start, loop_end=cfg.loop_end, loop_L=1,
                          loop_L_max=3, loop_random=False, loop_cache='shared'))
    m.train()
    return m, cfg


def step_time(m, cfg, n, warm=3):
    g = torch.Generator().manual_seed(0)
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    ts = []
    for i in range(n + warm):
        x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
        t0 = time.perf_counter()
        out = m.forward_logits(x, None)
        lg = out[1]
        lg = lg[0] if isinstance(lg, tuple) else lg
        loss = torch.nn.functional.cross_entropy(
            lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        if i >= warm:
            ts.append(dt)
    return statistics.median(ts)


print()
print('  building model...')
m, cfg = build()

t_plain = step_time(m, cfg, 10)
print('  未编译（10 步，含 3 步预热）: 【%.4f s/步】' % t_plain)

print('  编译中（CPU 上可能要几分钟）...')
t0 = time.perf_counter()
try:
    mc = torch.compile(m)
    # force compilation with one forward+backward
    g = torch.Generator().manual_seed(0)
    x = tok[torch.randint(0, len(tok), (B,), generator=g)].long()
    out = mc.forward_logits(x, None)
    lg = out[1]
    lg = lg[0] if isinstance(lg, tuple) else lg
    torch.nn.functional.cross_entropy(
        lg[:, :-1].reshape(-1, cfg.vocab_size), x[:, 1:].reshape(-1)).backward()
    t_compile = time.perf_counter() - t0
    print('  编译耗时: %.1f s' % t_compile)
    t_comp = step_time(mc, cfg, 10)
    print('  已编译（10 步，含 3 步预热）: 【%.4f s/步】' % t_comp)
    print()
    print('  加速比: 【%.4f×】(%.1f%%)' % (t_plain / t_comp, (1 - t_comp / t_plain) * 100))
    print('  判据: 稳态步时下降 > 5% ⇒ 留；否则关')
    print('  ⇒ %s' % ('通过' if (1 - t_comp / t_plain) > 0.05 else '不通过，关闭该方向'))
except Exception as e:
    print('  编译或运行失败: %s' % str(e)[:300])
    print('  ⇒ 记录为不可用')
