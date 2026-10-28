# -*- coding: utf-8 -*-
"""profile_ar_ops.py —— AR 预训练「一步」的算子级拆解。

为什么要这个：用户说预训练「超级慢、步数多」，想加速。但"加速 GEMM"只有在
GEMM 真的占大头时才值钱。profile_ar_train.py 只能分出 前向/反向/优化器 三段，
分不出「反向里面是 GEMM 还是 norm/激活」。

这里用 torch.profiler 拿到【算子清单 + 形状】，然后：
  ① 按 self time 排序看热点（注意：现在 SDXL 编码在抢 4.5 个核，绝对时间只能横向比）
  ② 把所有 mm/addmm/bmm 的形状列出来，算出它们的 FLOPs
  ③ 用「GEMM 的 self time 之和 / 整步时间」估 GEMM 占比
  ④ 用「总 GEMM FLOPs / GEMM 时间」算实测 GFLOPS —— 这个才是跟 AVX2 内核
     对照的【有效算力】。如果它远低于 200 GFLOPS，说明 GEMM 没跑满，有肉。

★ 关键判读：如果 GEMM 占比 < 50%，那么就算 GEMM 快 2 倍，整步也只快 ~1.3 倍，
  优先级要往后放；如果 > 70%，那 AVX2 内核就是主线。
"""
import os as _os, sys as _sys  # 归组 bootstrap：直接运行本文件时把仓库根加回 sys.path
_root = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import os
import sys
import time
import glob
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

sys.path.insert(0, r'D:\work\bitsandbytes-CPU')
sys.path.insert(0, r'D:\work\bitsandbytes-CPU\bitsandbytes')
from small_image_model_v2 import SmallImageConfigV2, SmallARImageModelV2

TH = int(os.environ.get('TH', '6'))
torch.set_num_threads(TH)
print('=' * 88)
print('AR 训练步 算子级拆解   线程=%d   torch=%s' % (TH, torch.__version__))
print('=' * 88)

# ---- checkpoint ----
# 只认带 'cfg' 的（AR 模型），并且默认用最近那次的 v4_bos。
# ★ 之前用 glob+最新时间戳抓，抓到了扩散模型的 ckpt（没有 cfg 键）直接 KeyError。
REL = os.environ.get('CKPT', r'.autopilot\v4_bos\ckpt_13000.pt')
CK = os.path.join(r'D:\work\bitsandbytes-CPU', REL)
if not os.path.exists(CK):
    cands = []
    for pat in (r'.autopilot\v4_bos\ckpt_*.pt', r'.autopilot\v3_noloop\ckpt_*.pt',
                r'.autopilot\v2_run*\ckpt_*.pt'):
        cands += glob.glob(os.path.join(r'D:\work\bitsandbytes-CPU', pat))
    cands = [c for c in cands if os.path.getsize(c) > 1e6]
    cands.sort(key=os.path.getmtime, reverse=True)
    if not cands:
        print('!! 找不到带 cfg 的 AR checkpoint，退出'); sys.exit(2)
    CK = cands[0]
print('  checkpoint: %s' % os.path.relpath(CK, r'D:\work\bitsandbytes-CPU'))

ck = torch.load(CK, map_location='cpu', weights_only=False)
keep = set(SmallImageConfigV2.__dataclass_fields__.keys())
cfg = SmallImageConfigV2(**{k: v for k, v in ck['cfg'].items() if k in keep})
m = SmallARImageModelV2(cfg)
sd = ck.get('model') or ck.get('state_dict') or ck
m.load_state_dict(sd)
m.train()
NP = sum(p.numel() for p in m.parameters())
T = cfg.total_tokens
print('  模型 %.2fM   T=%d   d=%d  L=%d  vocab=%d  use_bos=%s'
      % (NP / 1e6, T, cfg.d_model, cfg.n_layer, cfg.vocab_size, getattr(m, 'use_bos', '?')))
print()

try:
    import bitsandbytes as bnb
    Opt = bnb.optim.AdamW8bit
    oname = 'bnb AdamW8bit'
except Exception:
    Opt = torch.optim.AdamW
    oname = 'torch AdamW'
print('  优化器: %s' % oname)
# 配置里的关键开关，跑之前先看清楚架构，免得剖析的是个别的模型
_interesting = ('n_layer', 'd_model', 'total_tokens', 'grid', 'use_l3', 'l3_v', 'think_tokens',
                'loop_times', 'loop_start', 'loop_end', 'n_unique', 'vocab_size', 'use_bos',
                'n_cluster', 'per_token', 'head_mode', 'factored', 'hier', 'cond_dim')
print('  cfg: ' + '  '.join('%s=%s' % (k, getattr(cfg, k, None))
                          for k in _interesting if getattr(cfg, k, None) is not None))
opt = Opt(m.parameters(), lr=1e-4)

B = int(os.environ.get('B', '8'))


def _unwrap(x):
    """forward_logits 返回的可能是嵌套 tuple/list，一层层剥到 Tensor。"""
    while isinstance(x, (tuple, list)):
        if not x:
            raise RuntimeError('forward_logits 返回了空 tuple')
        x = x[0]
    return x


def _leaf_loss(o):
    """对返回结构里每一个 logits 张量都算一次 CE。

    ★ 目标 token 是随机的（不是真数据），但【形状和算量完全一致】—— 这一步只
      为了拿算子清单和 FLOPs，loss 数值无意义。真实训练里 hier 模式是
      (thc,thd)/(gec,ged) 这种因子化头，挨个 CE，这里保持一致。
    """
    if o is None:
        return 0.0
    if isinstance(o, (tuple, list)):
        return sum(_leaf_loss(i) for i in o)
    if not torch.is_tensor(o):
        return 0.0
    V = o.shape[-1]
    flat = o.reshape(-1, V)
    return F.cross_entropy(flat, torch.randint(0, V, (flat.shape[0],)))


def one_step():
    tok = torch.randint(0, cfg.vocab_size, (B, T))
    out = m.forward_logits(tok)
    loss = sum(_leaf_loss(o) for o in (out if isinstance(out, (tuple, list)) else (out,)))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss)


for _ in range(2):
    one_step()

# ---------------- 计时（无 profiler，避免开销污染） ----------------
n = 3
t0 = time.perf_counter()
for _ in range(n):
    L = one_step()
dt = (time.perf_counter() - t0) / n
print('  实测: %.1f ms/步  (batch=%d, %.2f ms/图, 1000 步 = %.1f s)  loss=%.3f'
      % (dt * 1e3, B, dt / B * 1e3, dt * 1000, L))
print()

# ---------------- 算子级 profile ----------------
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    one_step()

evs = prof.key_averages()
# ★ key_averages() 返回的是 FunctionEventAvg，【没有 .inputs】—— 想看形状必须
#   回到原始事件 prof.events()。之前在这儿 AttributeError 崩过一次。
raw_events = list(prof.events())
rows = []
for e in evs:
    if e.self_device_time_total > 0 or e.self_cpu_time_total > 0:
        rows.append((e.key, e.count,
                     e.self_cpu_time_total / 1000.0,        # ms
                     e.device_time_total / 1000.0 if e.device_time_total else 0.0))
rows.sort(key=lambda r: -r[2])

print('  ---- self CPU time Top 25 (ms) ----')
print('  %-46s %8s %10s %9s' % ('算子', '次数', 'self ms', '%步'))
tot_prof = sum(r[2] for r in rows)
for k, c, s, _ in rows[:25]:
    print('  %-46s %8d %10.2f %8.1f%%' % (k[:46], c, s, s / max(dt * 1e3, 1e-9) * 100))

# ---------------- GEMM 形状与 FLOPs ----------------
GEMM_OPS = ('mm', 'addmm', 'bmm', 'baddbmm', 'matmul', 'linear', 'einsum',
            '_scaled_dot_product', 'scaled_dot_product_attention', 'sdpa')
gemm_rows = []
tot_gflop = 0.0
shape_count = defaultdict(int)
for e in raw_events:
    key = e.name if hasattr(e, 'name') else str(e.key)
    if not any(g in key for g in GEMM_OPS):
        continue
    sh = getattr(e, 'input_shapes', None)
    if not sh:
        continue
    dims = [list(s) for s in sh if s]
    fl = None
    M = N2 = K = None
    if len(dims) >= 2 and all(len(d) == 2 for d in dims[:2]):
        (M, K), (N2, K2) = dims[0], dims[1]
        if K == K2:
            fl = 2.0 * M * N2 * K
    elif len(dims) >= 2 and len(dims[0]) == 3 and len(dims[1]) == 3:
        b, M, K = dims[0]
        _, N2, K2 = dims[1]
        if K == K2:
            fl = 2.0 * b * M * N2 * K
    if fl is None:
        continue
    sig = (key, M, N2, K)
    shape_count[sig] += 1
    gemm_rows.append((key, 1, M, N2, K, fl))
    tot_gflop += fl

# 按 (算子, 形状) 合并次数
merged = defaultdict(lambda: [0, 0.0])
for key, _, M, N2, K, fl in gemm_rows:
    e = merged[(key, M, N2, K)]
    e[0] += 1
    e[1] += fl
gemm_rows = [(k[0], v[0], k[1], k[2], k[3], v[1]) for k, v in merged.items()]

seen = set()
uniq = []
for key, c, M, N2, K, fl in sorted(gemm_rows, key=lambda r: -r[5]):
    sig = (key, M, N2, K)
    if sig in seen:
        continue
    seen.add(sig)
    uniq.append((key, c, M, N2, K, fl))

print()
print('  ---- 矩阵乘形状（按 FLOPs 排序）----')
print('  %-34s %6s %8s %8s %8s %10s' % ('算子', '次数', 'M', 'N', 'K', 'GFLOP'))
for key, c, M, N2, K, fl in uniq[:20]:
    print('  %-34s %6d %8d %8d %8d %10.2f' % (key[:34], c, M, N2, K, fl / 1e9))

# ---- GEMM self time ----
gemm_ms = sum(r[2] for r in rows if any(g in r[0] for g in GEMM_OPS))
step_ms = dt * 1e3
print()
print('  ---- 汇总 ----')
print('    整步            : %8.1f ms' % step_ms)
print('    GEMM(矩阵乘系)  : %8.1f ms   占 %.1f%%' % (gemm_ms, gemm_ms / step_ms * 100))
print('    GEMM 总 FLOPs   : %8.2f GFLOP/步' % (tot_gflop / 1e9))
if gemm_ms > 0:
    print('    GEMM 实测算力   : %8.1f GFLOPS   <-- 跟 AVX2 内核(峰值~250)比' % (tot_gflop / (gemm_ms / 1e3) / 1e9))
print('    整步实测算力    : %8.1f GFLOPS   (= 总FLOPs / 整步时间)' % (tot_gflop / dt / 1e9))
print()
print('    判读: GEMM 占比 %.0f%% ⇒ %s' % (
    gemm_ms / step_ms * 100,
    'GEMM 是主线，换 AVX2 内核值得' if gemm_ms / step_ms > 0.65
    else ('有一半肉在 GEMM，但也要治别的' if gemm_ms / step_ms > 0.4
          else 'GEMM 不是瓶颈，换内核收益有限，得先查别的')))

out = {
    'ckpt': CK, 'batch': B, 'threads': TH, 'ms_per_step': step_ms,
    'ms_per_image': dt / B * 1e3, 'gemm_ms': gemm_ms,
    'gemm_share': gemm_ms / step_ms, 'total_gflop': tot_gflop / 1e9,
    'gemm_gflops': tot_gflop / (gemm_ms / 1e3) / 1e9 if gemm_ms > 0 else None,
    'step_gflops': tot_gflop / dt / 1e9,
    'top_ops': [{'op': k, 'count': c, 'self_ms': s} for k, c, s, _ in rows[:40]],
    'gemm_shapes': [{'op': k, 'count': c, 'M': M, 'N': N2, 'K': K, 'gflop': fl / 1e9}
                    for k, c, M, N2, K, fl in uniq],
}
with open(r'D:\work\bitsandbytes-CPU\.autopilot\ar_ops_profile.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print()
print('  已写入 .autopilot\\ar_ops_profile.json')
print('DONE')
